"""Derive a small, anonymous decision-level supplement after 48 CUDA replays."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

MASKS = (1, 2, 3, 4)
DEPTHS = (2, 5)
SEEDS = (0, 1, 2)
ARMS = ("tied", "untied_propagation")
SOURCES = ("roman_multimask.py", "verify_roman_multimask.py",
           "roman_multimask_orchestrator.py", "roman_multimask_protocol.md",
           "models.py", "ROMAN_MULTIMASK_FREEZE.json")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    z = logits - logits.max(axis=-1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=-1)) -
                         z[np.arange(len(labels)), labels]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src = args.source.resolve()
    dst = args.output.resolve()
    gate = read(src / "results/roman/complete_grid_audit.json")
    assert gate["status"] == "COMPLETE_48_CELL_CUDA_REPLAY_PASS"
    assert len(gate["records"]) == 48
    preflight = read(src / "results/roman/preflight_and_smoke_audit.json")
    assert preflight["status"] == "FROZEN_INPUTS_AND_GPU_SMOKE_PASS"
    freeze = read(src / "ROMAN_MULTIMASK_FREEZE.json")
    assert sha(src / "ROMAN_MULTIMASK_FREEZE.json") == gate["freeze_manifest_sha256"]
    assert preflight["freeze_manifest_sha256"] == gate["freeze_manifest_sha256"]
    assert freeze["expected_cells"] == 48
    assert set(freeze["input_sha256"]) == set(SOURCES) - {"ROMAN_MULTIMASK_FREEZE.json"} | {"data/roman_empire.npz"}
    for rel, digest in freeze["input_sha256"].items():
        assert sha(src / rel) == digest, rel
    for rel in SOURCES:
        copy(src / rel, dst / rel)
    for rel in ("results/roman/preflight_and_smoke_audit.json",
                "results/roman/complete_grid_audit.json"):
        copy(src / rel, dst / rel)
    copy(Path(__file__), dst / "prepare_roman_multimask_compact.py")
    copy(Path(__file__).with_name("verify_roman_multimask_compact.py"),
         dst / "verify_roman_multimask_compact.py")
    copy(Path(__file__).with_name("roman_multimask_compact_README.md"), dst / "README.md")
    with np.load(src / "data/roman_empire.npz", allow_pickle=False) as data:
        labels = data["node_labels"].copy()
        masks = {part: data[f"{part}_masks"][list(MASKS)].copy()
                 for part in ("train", "val", "test")}
    assert labels.shape == (22662,)
    anchor = dst / "official_labels_masks_1_4.npz"
    np.savez_compressed(anchor, node_labels=labels,
                        train_masks=masks["train"], val_masks=masks["val"],
                        test_masks=masks["test"])
    derivations = []
    for mask in MASKS:
        for depth in DEPTHS:
            prefix = Path("results/roman") / f"mask{mask}" / f"depth{depth}"
            spec_path = src / prefix / "source_manifest.json"
            spec = read(spec_path)
            spec_sha = sha(spec_path)
            assert spec["split"] == mask and spec["configuration"]["layers"] == depth
            assert spec["configuration"]["epochs"] == 1000
            assert spec["data_descriptor"]["undirected_edges"] == 65854
            assert spec["data_descriptor"]["explicit_self_loops_added"] is False
            assert gate["source_manifest_sha256"][f"mask{mask}/depth{depth}"] == spec_sha
            assert preflight["source_manifest_sha256"][f"mask{mask}/depth{depth}"] == spec_sha
            copy(spec_path, dst / prefix / "source_manifest.json")
            copy(src / prefix / "completion_audit.json", dst / prefix / "completion_audit.json")
            audit = read(src / prefix / "completion_audit.json")
            assert audit["status"] == "COMPLETE_CUDA_REPLAY_PASS"
            assert audit["mask"] == mask and audit["depth"] == depth
            assert len(audit["records"]) == 6
            for seed in SEEDS:
                for arm in ARMS:
                    run = prefix / f"seed{seed}" / arm
                    row = read(src / run / "result.json")
                    assert row["source_manifest_sha256"] == spec_sha
                    assert row["published_split"] == mask and row["depth"] == depth
                    assert row["optimization_seed"] == seed and row["arm"] == arm
                    for filename in ("result.json", "validation_trace.csv", "initialization.json"):
                        copy(src / run / filename, dst / run / filename)
                    raw = src / run / "selected_predictions.npz"
                    assert sha(raw) == row["artifact_sha256"]["selected_predictions.npz"]
                    with np.load(raw, allow_pickle=False) as p:
                        payload = {name: p[name].copy() for name in p.files}
                    output = {}
                    for part in ("valid", "test"):
                        idx = payload[f"{part}_indices"]
                        y = payload[f"{part}_labels"]
                        logits = payload[f"{part}_member_logits"]
                        official = np.flatnonzero(masks["val" if part == "valid" else "test"][mask - 1])
                        assert np.array_equal(idx, official)
                        assert np.array_equal(y, labels[idx])
                        assert logits.shape == (4, len(idx), 18) and np.isfinite(logits).all()
                        pool = logits.mean(axis=0)
                        output[f"{part}_indices"] = idx
                        output[f"{part}_labels"] = y
                        output[f"{part}_member_pred"] = logits.argmax(axis=-1)
                        output[f"{part}_pool_pred"] = pool.argmax(axis=-1)
                        assert abs(float(np.mean(output[f"{part}_pool_pred"] == y)) -
                                   row[f"{part}_accuracy"]) < 1e-6
                        assert abs(ce(pool, y) - row[f"{part}_ce"]) < 1e-4
                    derived = dst / run / "selected_decisions.npz"
                    np.savez_compressed(derived, **output)
                    derivations.append({"mask": mask, "depth": depth,
                                        "seed": seed, "arm": arm,
                                        "result_sha256": sha(src / run / "result.json"),
                                        "original_predictions_sha256": sha(raw),
                                        "derived_decisions_sha256": sha(derived)})
    assert len(derivations) == 48
    manifest = {"status": "COMPLETE_48_CELL_DECISION_DERIVATION_PASS",
                "freeze_manifest_sha256": sha(src / "ROMAN_MULTIMASK_FREEZE.json"),
                "complete_grid_audit_sha256": sha(src / "results/roman/complete_grid_audit.json"),
                "original_public_npz_sha256": sha(src / "data/roman_empire.npz"),
                "official_label_anchor_sha256": sha(anchor),
                "derivation_script_sha256": sha(Path(__file__)),
                "records": derivations}
    (dst / "decision_derivation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in dst.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".md", ".json", ".csv"):
            data = path.read_bytes().lower()
            for word in (b"/" + b"users/", b"/home/" + b"jovyan",
                         b"/disk/" + b"10tb/home", b"gen" + b"link",
                         b"ssh-" + b"sr003", b"ai000" + b"1053",
                         b"cher" + b"vov", b"niki" + b"ta",
                         b"shm" + b"elev"):
                assert word not in data, (path, word)
            assert not re.search(rb"gpu-[0-9a-f]{8}-[0-9a-f-]{20,}", data), path
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print("COMPACT_STAGE_PASS", dst, "files",
          sum(p.is_file() for p in dst.rglob("*")), "bytes", size)


if __name__ == "__main__":
    main()
