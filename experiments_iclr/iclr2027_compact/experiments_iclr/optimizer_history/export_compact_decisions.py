"""Post-score, compact decision evidence for the mechanism and NORM-SYNC studies.

Export requires the independent checkpoint replay report and the frozen validation
locks. The public ``verify`` command needs only this directory and NumPy; it
checks class ranges, ordering, and accuracy. Replaying logits, pooling, and
official split construction requires the omitted public graph data and checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
FAMILIES = {
    "mechanism": {
        "score_dir": "scores",
        "audit": "MECHANISM_FINAL_SCORE_AUDIT.json",
        "lock": "MECHANISM_VALIDATION_LOCK.json",
        "audit_lock_field": "mechanism_validation_lock_sha256",
        "min_count": 36,
        "max_count": 48,
    },
    "norm_sync": {
        "score_dir": "scores_norm_sync",
        "audit": "NORM_SYNC_FINAL_SCORE_AUDIT.json",
        "lock": "NORM_SYNC_VALIDATION_LOCK.json",
        "audit_lock_field": "norm_validation_lock_sha256",
        "min_count": 12,
        "max_count": 12,
    },
    "norm_sync_v2": {
        "score_dir": "scores_norm_sync_v2",
        "audit": "NORM_SYNC_V2_FINAL_SCORE_AUDIT.json",
        "lock": "NORM_SYNC_V2_VALIDATION_LOCK.json",
        "audit_lock_field": "norm_validation_lock_sha256",
        "min_count": 12,
        "max_count": 12,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def cell_dataset(key: str) -> str:
    return key.split("/", 1)[0]


def expected_keys(family: str, lock: dict) -> set[str]:
    if family in ("norm_sync", "norm_sync_v2"):
        return {f"{dataset}/seed{seed}" for dataset in DATASETS for seed in range(3)}
    expected = set()
    for dataset in DATASETS:
        selected = tuple(lock["sync_selections"][dataset]["selected_candidate"])
        for arm in ("tied", "untied", "sync"):
            candidates = {(0.001, 0.0)} if arm != "sync" else {(0.001, 0.0), selected}
            for lr, wd in candidates:
                for seed in range(3):
                    expected.add(f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}")
    return expected


def export(family: str) -> None:
    # Imports live only in export: compact verification works without PyTorch/PyG.
    import torch
    import tuning

    cfg = FAMILIES[family]
    audit_path = ROOT / cfg["audit"]
    lock_path = ROOT / cfg["lock"]
    global_lock_path = ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"
    require(audit_path.is_file() and lock_path.is_file() and global_lock_path.is_file(),
            "Complete independent score audit and validation locks required")
    audit = read_json(audit_path)
    scores = audit["scores"]
    family_lock = read_json(lock_path)
    global_lock = read_json(global_lock_path)
    require(tuning.check_freeze() == global_lock["freeze_sha256"] and
            len(global_lock["cells"]) == 432 and
            audit["original_432_validation_lock_sha256"] == sha(global_lock_path) and
            audit[cfg["audit_lock_field"]] == sha(lock_path) and
            audit["freeze_sha256"] == family_lock["freeze_sha256"],
            "Score audit, frozen graph data, or validation locks disagree")
    require(set(scores) == expected_keys(family, family_lock) and
            cfg["min_count"] <= len(scores) <= cfg["max_count"],
            "Independent score audit differs from selected/default allowlist")
    scored_keys = {p.parent.relative_to(ROOT / cfg["score_dir"]).as_posix()
                   for p in (ROOT / cfg["score_dir"]).rglob("score.json")}
    require(scored_keys == set(scores), "Score directory differs from independent audit")
    output = ROOT / "compact_decisions" / family
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(lock_path, output / cfg["lock"])
    shutil.copyfile(audit_path, output / cfg["audit"])
    shutil.copyfile(global_lock_path, output / global_lock_path.name)
    manifest = {
        "protocol": "compact_hard_decisions_v1",
        "family": family,
        "cell_count": len(scores),
        "family_validation_lock_file": cfg["lock"],
        "independent_score_audit_file": cfg["audit"],
        "original_432_validation_lock_file": global_lock_path.name,
        "provenance": {
            "independent_score_audit_sha256": sha(audit_path),
            "family_validation_lock_sha256": sha(lock_path),
            "original_432_validation_lock_sha256": sha(global_lock_path),
            "scope": "all independently audited scored cells",
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        bundle, descriptor = tuning.load_graph(dataset, torch.device("cpu"), include_test=True)
        loader = {"cora": tuning.load_cora, "wikics": tuning.load_wikics,
                  "actor": tuning.load_actor,
                  "chameleon_filtered": tuning.load_filtered}[dataset]
        _, raw_labels, _, _, _, _, _ = loader()
        full_labels = raw_labels.numpy().astype(np.int64, copy=False)
        ids = bundle.test_idx.numpy().astype(np.int64, copy=False)
        labels = full_labels[ids].astype(np.int16)
        require(ids.ndim == labels.ndim == 1 and len(ids) == len(labels) > 0 and
                np.all(np.diff(ids) > 0) and
                np.min(labels) >= 0 and np.max(labels) < descriptor["classes"],
                f"Invalid official split anchor: {dataset}")
        require(tensor_sha(full_labels) == descriptor["tensor_sha256"]["labels"] and
                tensor_sha(ids) == descriptor["tensor_sha256"]["test_indices"] and
                np.array_equal(labels, bundle.test_y.numpy()),
                f"Frozen full labels or official split differ: {dataset}")
        arrays = {"full_labels": full_labels, "test_node_ids": ids,
                  "test_labels": labels}
        records = []
        keys = sorted(key for key in scores if cell_dataset(key) == dataset)
        require(len(keys) > 0, f"No audited scores: {dataset}")
        for number, key in enumerate(keys):
            score_dir = ROOT / cfg["score_dir"] / key
            score_path = score_dir / "score.json"
            prediction_path = score_dir / "predictions.npz"
            score = read_json(score_path)
            require(sha(score_path) == scores[key]["score_sha256"] and
                    sha(prediction_path) == scores[key]["predictions_sha256"] and
                    score["predictions_sha256"] == scores[key]["predictions_sha256"],
                    f"Audited score provenance mismatch: {key}")
            with np.load(prediction_path, allow_pickle=False) as raw:
                pooled = raw["test_pooled_logits"]
                members = raw["test_member_logits"]
            require(pooled.ndim == 2 and members.shape ==
                    (4, len(labels), descriptor["classes"]) and
                    pooled.shape == (len(labels), descriptor["classes"]) and
                    np.isfinite(pooled).all() and np.isfinite(members).all() and
                    np.allclose(pooled, members.mean(axis=0), atol=1e-5, rtol=1e-5),
                    f"Malformed or inconsistent raw logits: {key}")
            pooled_class = pooled.argmax(axis=-1).astype(np.int16)
            member_class = members.argmax(axis=-1).astype(np.int16)
            accuracy = float(np.mean(pooled_class == labels))
            require(abs(accuracy - score["test_accuracy"]) <= 1e-7 and
                    abs(accuracy - scores[key]["test_accuracy"]) <= 1e-7,
                    f"Audited accuracy mismatch: {key}")
            pool_name, member_name = f"pooled_{number:02d}", f"members_{number:02d}"
            arrays[pool_name] = pooled_class
            arrays[member_name] = member_class
            records.append({
                "key": key,
                "pooled_array": pool_name,
                "member_array": member_name,
                "test_accuracy": accuracy,
                "member_test_accuracies": [float(np.mean(row == labels)) for row in member_class],
                "score_sha256": sha(score_path),
                "raw_predictions_sha256": sha(prediction_path),
            })
        filename = f"{dataset}.npz"
        path = output / filename
        np.savez_compressed(path, **arrays)
        manifest["datasets"][dataset] = {
            "file": filename,
            "sha256": sha(path),
            "official_split": descriptor["split"],
            "full_label_tensor_sha256": descriptor["tensor_sha256"]["labels"],
            "test_indices_tensor_sha256": descriptor["tensor_sha256"]["test_indices"],
            "class_count": descriptor["classes"],
            "test_count": len(labels),
            "cells": records,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify(output)
    print(json.dumps({"family": family,
                      "cells": sum(len(d["cells"]) for d in manifest["datasets"].values()),
                      "bytes": sum(p.stat().st_size for p in output.iterdir()),
                      "manifest_sha256": sha(output / "manifest.json")}), flush=True)


def verify(folder: Path) -> None:
    manifest = read_json(folder / "manifest.json")
    require(manifest["protocol"] == "compact_hard_decisions_v1" and
            manifest["family"] in FAMILIES and
            set(manifest["datasets"]) == set(DATASETS), "Compact manifest schema mismatch")
    family = manifest["family"]
    cfg = FAMILIES[family]
    require(manifest["family_validation_lock_file"] == cfg["lock"] and
            manifest["independent_score_audit_file"] == cfg["audit"] and
            manifest["original_432_validation_lock_file"] ==
            "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json" and
            sha(folder / cfg["lock"]) == manifest["provenance"]["family_validation_lock_sha256"] and
            sha(folder / cfg["audit"]) == manifest["provenance"]["independent_score_audit_sha256"],
            "Copied validation lock or independent score audit changed")
    original_lock_path = folder / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"
    require(sha(original_lock_path) == manifest["provenance"]["original_432_validation_lock_sha256"],
            "Copied original 432-cell validation lock changed")
    expected = expected_keys(family, read_json(folder / cfg["lock"]))
    family_lock = read_json(folder / cfg["lock"])
    original_lock = read_json(original_lock_path)
    audit = read_json(folder / cfg["audit"])
    audit_scores = audit["scores"]
    require(len(original_lock["cells"]) == 432 and
            audit["original_432_validation_lock_sha256"] == sha(original_lock_path) and
            audit[cfg["audit_lock_field"]] == sha(folder / cfg["lock"]) and
            audit["freeze_sha256"] == family_lock["freeze_sha256"],
            "Score audit is not bound to the copied validation locks")
    require(set(audit_scores) == expected and manifest["cell_count"] == len(expected),
            "Compact cell set differs from validation-selected score audit")
    total = 0
    all_keys = set()
    for dataset in DATASETS:
        spec = manifest["datasets"][dataset]
        path = folder / spec["file"]
        require(sha(path) == spec["sha256"], f"Compact NPZ changed: {dataset}")
        with np.load(path, allow_pickle=False) as saved:
            ids, labels = saved["test_node_ids"], saved["test_labels"]
            full_labels = saved["full_labels"]
            require(ids.dtype == full_labels.dtype == np.int64 and
                    labels.dtype == np.int16 and
                    ids.shape == labels.shape == (spec["test_count"],) and
                    np.all(np.diff(ids) > 0) and np.min(ids) >= 0 and
                    np.max(ids) < len(full_labels) and
                    np.min(labels) >= 0 and np.max(labels) < spec["class_count"] and
                    tensor_sha(full_labels) == spec["full_label_tensor_sha256"] and
                    tensor_sha(ids) == spec["test_indices_tensor_sha256"] and
                    np.array_equal(full_labels[ids], labels),
                    f"Split anchors malformed: {dataset}")
            expected_arrays = {"full_labels", "test_node_ids", "test_labels"}
            keys = []
            for cell in spec["cells"]:
                pooled, members = saved[cell["pooled_array"]], saved[cell["member_array"]]
                expected_arrays.update((cell["pooled_array"], cell["member_array"]))
                require(pooled.dtype == members.dtype == np.int16 and
                        pooled.shape == (len(labels),) and members.shape == (4, len(labels)) and
                        np.min(pooled) >= 0 and np.max(pooled) < spec["class_count"] and
                        np.min(members) >= 0 and np.max(members) < spec["class_count"],
                        f"Decision array malformed: {cell['key']}")
                require(abs(float(np.mean(pooled == labels)) - cell["test_accuracy"]) <= 1e-12 and
                        all(abs(float(np.mean(row == labels)) - claimed) <= 1e-12
                            for row, claimed in zip(members, cell["member_test_accuracies"], strict=True)),
                        f"Compact decision accuracy mismatch: {cell['key']}")
                require(cell["key"].split("/", 1)[0] == dataset,
                        f"Wrong dataset in cell: {cell['key']}")
                require(cell["key"] in audit_scores and
                        cell["score_sha256"] == audit_scores[cell["key"]]["score_sha256"] and
                        cell["raw_predictions_sha256"] == audit_scores[cell["key"]]["predictions_sha256"] and
                        abs(cell["test_accuracy"] - audit_scores[cell["key"]]["test_accuracy"]) <= 1e-7,
                        f"Compact decision differs from independent score audit: {cell['key']}")
                keys.append(cell["key"])
            require(set(saved.files) == expected_arrays and keys == sorted(set(keys)),
                    f"Unexpected arrays or duplicate cells: {dataset}")
            total += len(keys)
            all_keys.update(keys)
    require(total == manifest["cell_count"] and
            all_keys == expected and
            FAMILIES[manifest["family"]]["min_count"] <= total <=
            FAMILIES[manifest["family"]]["max_count"],
            "Compact score count mismatch")
    print(json.dumps({"verified_compact_cells": total,
                      "family": manifest["family"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("family", choices=FAMILIES)
    ver = sub.add_parser("verify")
    ver.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        export(args.family)
    else:
        verify(args.folder)


if __name__ == "__main__":
    main()
