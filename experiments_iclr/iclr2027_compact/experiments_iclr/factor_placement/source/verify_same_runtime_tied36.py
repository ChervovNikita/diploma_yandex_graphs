"""Independent validation lock and held-out replay for the TIED36 replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import same_runtime_tied36 as study
import tuning as primary
import verify_tuning as original_auditor


OUT = Path(__file__).resolve().parent / "same_runtime_tied36_results"
DATASETS = ("cora", "wikics")
SEEDS = (0, 1, 2)
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01),
              (0.001, 0.0), (0.001, 0.01),
              (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)
EPOCHS = 1000
PROTOCOL = "same_runtime_tied36_v1"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def key(dataset: str, lr: float, wd: float, seed: int) -> str:
    return f"{dataset}/tied/lr{lr:g}_wd{wd:g}/seed{seed}"


def audit_cell(dataset, lr, wd, seed, freeze_sha, bundle, device, anchor):
    identity = key(dataset, lr, wd, seed)
    # Original independent verifier replays the full trace/checkpoint schema.
    saved_root = original_auditor.ROOT
    try:
        original_auditor.ROOT = OUT
        record = original_auditor.audit_one(dataset, "tied", lr, wd, seed, freeze_sha)
    finally:
        original_auditor.ROOT = saved_root
    folder = study.cell_dir(dataset, lr, wd, seed)
    result = json.loads((folder / "result.json").read_text())
    require(result["epochs_required"] == EPOCHS and
            result["initialization"]["parameter_count"] > 0,
            f"TIED36 matrix or initialization differs: {identity}")
    if (lr, wd) == DEFAULT:
        expected = anchor["rows"][f"{dataset}/seed{seed}"]
        require(all(result["initialization"][field] == value
                    for field, value in expected.items()
                    if field != "initial_member_logits_sha256"),
                f"Default initialization differs from same-runtime anchor: {identity}")
    if result["failure"] is not None:
        require(not (folder / "validation_companion.npz").exists() and
                not (folder / "validation_companion.json").exists(),
                f"Failed cell has a validation companion: {identity}")
        return record
    companion_path = folder / "validation_companion.npz"
    manifest_path = folder / "validation_companion.json"
    require(companion_path.is_file() and manifest_path.is_file(),
            f"Selected checkpoint lacks prelock validation companion: {identity}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["protocol"] == PROTOCOL and manifest["dataset"] == dataset and
            manifest["lr"] == lr and manifest["weight_decay"] == wd and
            manifest["seed"] == seed and
            manifest["checkpoint_sha256"] == record["checkpoint_sha256"] and
            manifest["result_sha256"] == record["result_sha256"] and
            manifest["predictions_sha256"] == study.sha(companion_path) and
            manifest["original_selected_pooled_logits_sha256"] ==
            result["selected_valid_pooled_logits_sha256"],
            f"Validation companion provenance differs: {identity}")
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    primary.seed_all(seed)
    model, _ = primary.make_model("tied", bundle, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    va, vc, pooled, members = original_auditor.independent_metrics(
        model, "tied", bundle, bundle.valid_idx, bundle.valid_y)
    require(abs(va - result["selected_valid_accuracy"]) <= 1e-7 and
            abs(vc - result["selected_valid_ce"]) <= 1e-5 and
            abs(va - manifest["valid_accuracy"]) <= 1e-7 and
            abs(vc - manifest["valid_ce"]) <= 1e-5,
            f"Independent selected validation replay differs: {identity}")
    with np.load(companion_path, allow_pickle=False) as saved:
        require(set(saved.files) == {"valid_indices", "valid_labels",
                                     "valid_pooled_logits", "valid_member_logits"} and
                np.array_equal(saved["valid_indices"], bundle.valid_idx.cpu().numpy()) and
                np.array_equal(saved["valid_labels"], bundle.valid_y.cpu().numpy()) and
                saved["valid_member_logits"].shape == members.shape and
                saved["valid_pooled_logits"].shape == pooled.shape and
                np.allclose(saved["valid_member_logits"], members.numpy(),
                            rtol=1e-5, atol=1e-5) and
                np.allclose(saved["valid_pooled_logits"], pooled.numpy(),
                            rtol=1e-5, atol=1e-5) and
                np.array_equal(saved["valid_member_logits"].argmax(-1),
                               members.argmax(-1).numpy()) and
                np.array_equal(saved["valid_pooled_logits"].argmax(-1),
                               pooled.argmax(-1).numpy()) and
                original_auditor.tensor_sha(torch.from_numpy(saved["valid_pooled_logits"])) ==
                manifest["companion_pooled_logits_sha256"],
                f"Bounded validation logits or exact class decisions differ: {identity}")
    record.update({"validation_companion_manifest_sha256": study.sha(manifest_path),
                   "validation_companion_predictions_sha256": study.sha(companion_path)})
    return record


def audit_and_lock(device: torch.device) -> None:
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    require(not lock_path.exists(), "Refusing to overwrite TIED36 validation lock")
    freeze_sha = study.check_freeze()
    study.exact_primary_lock()
    frozen = json.loads(study.FREEZE.read_text())
    matrix = frozen["matrix"]
    require(frozen["protocol"] == PROTOCOL and
            frozen["original_validation_lock_sha256"] == study.PRIMARY_LOCK_SHA and
            tuple(matrix["datasets"]) == DATASETS and matrix["arm"] == "tied" and
            tuple(matrix["seeds"]) == SEEDS and matrix["epochs"] == EPOCHS and
            tuple((row["lr"], row["weight_decay"]) for row in matrix["candidates"]) ==
            CANDIDATES and tuple(matrix["default_candidate"]) == DEFAULT,
            "Prospective TIED36 matrix or exact original lock differs")
    anchor = json.loads(study.ANCHOR.read_text())
    cells, selections = {}, {}
    for dataset in DATASETS:
        bundle, _ = primary.load_graph(dataset, device, include_test=False)
        candidates = []
        for lr, wd in CANDIDATES:
            records = []
            for seed in SEEDS:
                identity = key(dataset, lr, wd, seed)
                record = audit_cell(dataset, lr, wd, seed, freeze_sha,
                                    bundle, device, anchor)
                cells[identity] = record
                records.append(record)
            valid = all(row["failure"] is None for row in records)
            candidates.append({
                "lr": lr, "weight_decay": wd, "valid": valid,
                "mean_valid_accuracy": (sum(row["selected_valid_accuracy"]
                                            for row in records) / 3 if valid else None),
                "mean_valid_ce": (sum(row["selected_valid_ce"]
                                      for row in records) / 3 if valid else None),
                "seed_valid_accuracy": ([row["selected_valid_accuracy"]
                                         for row in records] if valid else None),
                "seed_selected_epoch": ([row["selected_epoch"]
                                        for row in records] if valid else None),
            })
        selected = original_auditor.select_candidate(candidates)
        require(selected is not None, f"All TIED36 candidates failed: {dataset}")
        selections[dataset] = {
            "candidate_table": candidates,
            "selected_candidate": [selected["lr"], selected["weight_decay"]],
            "selected_mean_valid_accuracy": selected["mean_valid_accuracy"],
            "selected_mean_valid_ce": selected["mean_valid_ce"],
            "selected_seed_valid_accuracy": selected["seed_valid_accuracy"],
            "selected_seed_epoch": selected["seed_selected_epoch"],
        }
    actual = {str(path.parent.relative_to(OUT / "results"))
              for path in (OUT / "results").rglob("result.json")}
    require(len(cells) == 36 and set(cells) == actual and
            set(selections) == set(DATASETS),
            "TIED36 validation result set is incomplete or has extra cells")
    lock = {
        "protocol": PROTOCOL, "freeze_sha256": freeze_sha,
        "original_validation_lock_sha256": study.PRIMARY_LOCK_SHA,
        "initialization_anchor_sha256": study.sha(study.ANCHOR),
        "cells": cells, "selections": selections,
        "selection_uses_test_labels": False,
        "test_scoring_allowlist": "selected candidate and declared (0.001,0) default only",
    }
    primary.write_json(lock_path, lock)
    print(json.dumps({"validation_lock_sha256": study.sha(lock_path),
                      "cells": len(cells), "selections": selections}), flush=True)


def audit_scores(device: torch.device) -> None:
    freeze_sha = study.check_freeze()
    study.exact_primary_lock()
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    require(lock["protocol"] == PROTOCOL and lock["freeze_sha256"] == freeze_sha and
            lock["original_validation_lock_sha256"] == study.PRIMARY_LOCK_SHA and
            lock["initialization_anchor_sha256"] == study.sha(study.ANCHOR) and
            len(lock["cells"]) == 36 and set(lock["selections"]) == set(DATASETS),
            "TIED36 validation lock differs")
    all_layer_lock_path = study.ROOT / "all_layer_factor_results" / "VALIDATION_SELECTION_LOCK.json"
    all_layer_freeze_path = study.ROOT / "all_layer_factor_results" / "FROZEN_ALL_LAYER_STUDY.json"
    require(all_layer_lock_path.is_file() and all_layer_freeze_path.is_file(),
            "Both TIED36 and all-layer validation locks are required for score audit")
    all_layer_lock = json.loads(all_layer_lock_path.read_text())
    require(all_layer_lock["protocol"] == "all_layer_factor_placement_posthoc_v1" and
            len(all_layer_lock["cells"]) == 72 and
            all_layer_lock["freeze_sha256"] == study.sha(all_layer_freeze_path) and
            all_layer_lock["primary_selection_lock_sha256"] == study.PRIMARY_LOCK_SHA,
            "All-layer validation lock differs from frozen source/primary lock")
    expected, report = set(), {
        "protocol": PROTOCOL, "freeze_sha256": freeze_sha,
        "validation_selection_lock_sha256": study.sha(lock_path),
        "all_layer_validation_lock_sha256": study.sha(all_layer_lock_path),
        "original_validation_lock_sha256": study.PRIMARY_LOCK_SHA,
        "scores": {},
    }
    for dataset in DATASETS:
        bundle, _ = primary.load_graph(dataset, device, include_test=True)
        selected = tuple(lock["selections"][dataset]["selected_candidate"])
        for lr, wd in dict.fromkeys((selected, DEFAULT)):
            for seed in SEEDS:
                identity = key(dataset, lr, wd, seed)
                expected.add(identity)
                folder = study.cell_dir(dataset, lr, wd, seed)
                score_dir = OUT / "scores" / identity
                score_path = score_dir / "score.json"
                predictions_path = score_dir / "predictions.npz"
                require(score_path.is_file() and predictions_path.is_file(),
                        f"TIED36 score missing: {identity}")
                row = json.loads(score_path.read_text())
                locked = lock["cells"][identity]
                require(study.sha(folder / "result.json") == locked["result_sha256"] and
                        study.sha(folder / "checkpoint.pt") == locked["checkpoint_sha256"] and
                        row["protocol"] == PROTOCOL and row["freeze_sha256"] == freeze_sha and
                        row["validation_selection_lock_sha256"] == study.sha(lock_path) and
                        row["all_layer_validation_lock_sha256"] == study.sha(all_layer_lock_path) and
                        row["original_validation_lock_sha256"] == study.PRIMARY_LOCK_SHA and
                        row["dataset"] == dataset and row["arm"] == "tied" and
                        row["lr"] == lr and row["weight_decay"] == wd and
                        row["seed"] == seed and
                        row["selected_candidate"] == ((lr, wd) == selected) and
                        row["predeclared_default"] == ((lr, wd) == DEFAULT) and
                        row["checkpoint_sha256"] == locked["checkpoint_sha256"] and
                        row["test_predictions_sha256"] == study.sha(predictions_path),
                        f"TIED36 score identity/hash differs: {identity}")
                primary.seed_all(seed)
                model, _ = primary.make_model("tied", bundle, device)
                checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                va, vc, vp, _ = original_auditor.independent_metrics(
                    model, "tied", bundle, bundle.valid_idx, bundle.valid_y)
                ta, tc, tp, tm = original_auditor.independent_metrics(
                    model, "tied", bundle, bundle.test_idx, bundle.test_y)
                with np.load(predictions_path, allow_pickle=False) as saved:
                    require(set(saved.files) == {"valid_pooled_logits", "test_pooled_logits",
                                                 "test_member_logits"} and
                            np.allclose(saved["valid_pooled_logits"], vp.numpy(),
                                        rtol=1e-5, atol=1e-5) and
                            np.allclose(saved["test_pooled_logits"], tp.numpy(),
                                        rtol=1e-5, atol=1e-5) and
                            np.allclose(saved["test_member_logits"], tm.numpy(),
                                        rtol=1e-5, atol=1e-5) and
                            np.array_equal(saved["test_pooled_logits"].argmax(-1),
                                           tp.argmax(-1).numpy()) and
                            np.array_equal(saved["test_member_logits"].argmax(-1),
                                           tm.argmax(-1).numpy()),
                            f"TIED36 exact class replay or bounded logits differ: {identity}")
                require(abs(va - row["valid_accuracy"]) <= 1e-7 and
                        abs(vc - row["valid_ce"]) <= 1e-5 and
                        abs(ta - row["test_accuracy"]) <= 1e-7 and
                        abs(tc - row["test_ce"]) <= 1e-5,
                        f"TIED36 independent metric replay differs: {identity}")
                report["scores"][identity] = {
                    "score_sha256": study.sha(score_path),
                    "predictions_sha256": study.sha(predictions_path),
                    "test_accuracy": ta, "test_ce": tc,
                }
    actual = {str(path.parent.relative_to(OUT / "scores"))
              for path in (OUT / "scores").rglob("score.json")}
    require(expected == actual and 6 <= len(expected) <= 12,
            "TIED36 score set has missing or unauthorized cells")
    path = OUT / "FINAL_SCORE_AUDIT.json"
    require(not path.exists(), "Refusing to overwrite TIED36 final audit")
    primary.write_json(path, report)
    print(json.dumps({"score_audit_sha256": study.sha(path),
                      "fresh_replays": len(expected)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-and-lock", "audit-scores"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "audit-and-lock":
        audit_and_lock(torch.device(args.device))
    else:
        audit_scores(torch.device(args.device))


if __name__ == "__main__":
    main()
