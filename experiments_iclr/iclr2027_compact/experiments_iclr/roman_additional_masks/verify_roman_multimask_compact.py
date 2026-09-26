"""Independent decision-level score audit of the 48-cell Roman supplement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MASKS = (1, 2, 3, 4)
DEPTHS = (2, 5)
SEEDS = (0, 1, 2)
ARMS = ("tied", "untied_propagation")
PROTOCOL = "roman_multimask_no_added_loops_depth2_5_1000_v1"
PINNED = {"roman_multimask.py", "verify_roman_multimask.py",
          "roman_multimask_orchestrator.py", "roman_multimask_protocol.md",
          "models.py", "data/roman_empire.npz"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def check_trace(path: Path, row: dict) -> None:
    with path.open(newline="") as f:
        trace = list(csv.DictReader(f))
    assert len(trace) == 1000, path
    best_acc, best_ce, best_epoch = -1.0, math.inf, 0
    for epoch, item in enumerate(trace, 1):
        assert int(item["epoch"]) == epoch
        acc, ce = float(item["valid_accuracy"]), float(item["valid_ce"])
        assert math.isfinite(acc) and math.isfinite(ce) and 0 <= acc <= 1
        improved = (acc > best_acc + 1e-12 or
                    (abs(acc - best_acc) <= 1e-12 and ce < best_ce - 1e-12))
        assert int(item["improved"]) == int(improved)
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, epoch
    assert best_epoch == row["selected_epoch"]
    assert abs(best_acc - row["valid_accuracy"]) < 1e-7
    assert abs(best_ce - row["valid_ce"]) < 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-npz", type=Path)
    cli = parser.parse_args()
    freeze_path = ROOT / "ROMAN_MULTIMASK_FREEZE.json"
    freeze = read(freeze_path)
    assert freeze["protocol"] == PROTOCOL
    assert freeze["masks"] == list(MASKS) and freeze["depths"] == list(DEPTHS)
    assert freeze["seeds"] == list(SEEDS) and freeze["arms"] == list(ARMS)
    assert freeze["expected_cells"] == 48
    assert set(freeze["input_sha256"]) == PINNED
    assert not (ROOT / "data/roman_empire.npz").exists()
    for rel, digest in freeze["input_sha256"].items():
        if rel != "data/roman_empire.npz":
            assert sha(ROOT / rel) == digest, rel
    preflight = read(ROOT / "results/roman/preflight_and_smoke_audit.json")
    complete_path = ROOT / "results/roman/complete_grid_audit.json"
    complete = read(complete_path)
    assert preflight["status"] == "FROZEN_INPUTS_AND_GPU_SMOKE_PASS"
    assert complete["status"] == "COMPLETE_48_CELL_CUDA_REPLAY_PASS"
    assert preflight["expected_cells"] == 48 and len(complete["records"]) == 48
    assert preflight["freeze_manifest_sha256"] == complete["freeze_manifest_sha256"] == sha(freeze_path)
    derivation = read(ROOT / "decision_derivation_manifest.json")
    assert derivation["status"] == "COMPLETE_48_CELL_DECISION_DERIVATION_PASS"
    assert derivation["freeze_manifest_sha256"] == sha(freeze_path)
    assert derivation["complete_grid_audit_sha256"] == sha(complete_path)
    assert derivation["original_public_npz_sha256"] == freeze["input_sha256"]["data/roman_empire.npz"]
    assert derivation["derivation_script_sha256"] == sha(ROOT / "prepare_roman_multimask_compact.py")
    anchor = ROOT / "official_labels_masks_1_4.npz"
    assert sha(anchor) == derivation["official_label_anchor_sha256"]
    with np.load(anchor, allow_pickle=False) as a:
        labels = a["node_labels"].copy()
        masks = {part: a[f"{part}_masks"].copy() for part in ("train", "val", "test")}
    assert labels.shape == (22662,)
    assert all(v.shape == (4, 22662) for v in masks.values())
    for i in range(4):
        tr, va, te = (masks[k][i].astype(bool) for k in ("train", "val", "test"))
        assert (int(tr.sum()), int(va.sum()), int(te.sum())) == (11331, 5665, 5666)
        assert not np.any(tr & va) and not np.any(tr & te) and not np.any(va & te)
    if cli.public_npz:
        assert sha(cli.public_npz) == freeze["input_sha256"]["data/roman_empire.npz"]
        with np.load(cli.public_npz, allow_pickle=False) as p:
            assert np.array_equal(labels, p["node_labels"])
            for part in ("train", "val", "test"):
                assert np.array_equal(masks[part], p[f"{part}_masks"][list(MASKS)])
    assert len(derivation["records"]) == 48
    derived = {(r["mask"], r["depth"], r["seed"], r["arm"]): r
               for r in derivation["records"]}
    assert len(derived) == 48
    complete_records = {(r["mask"], r["depth"], r["seed"], r["arm"]): r
                        for r in complete["records"]}
    assert len(complete_records) == 48
    scores = {}
    checks = 0
    for mask in MASKS:
        for depth in DEPTHS:
            prefix = Path("results/roman") / f"mask{mask}" / f"depth{depth}"
            spec_path = ROOT / prefix / "source_manifest.json"
            spec = read(spec_path)
            spec_sha = sha(spec_path)
            assert spec["protocol"] == PROTOCOL
            assert spec["split"] == mask and spec["configuration"]["layers"] == depth
            assert spec["configuration"]["epochs"] == 1000
            assert spec["data_descriptor"]["undirected_edges"] == 65854
            assert spec["data_descriptor"]["explicit_self_loops_added"] is False
            assert spec["optimization_seeds"] == list(SEEDS) and spec["arms"] == list(ARMS)
            assert set(spec["source_sha256"]) == PINNED
            for rel, digest in spec["source_sha256"].items():
                if rel != "data/roman_empire.npz":
                    assert sha(ROOT / rel) == digest, rel
            assert spec["source_sha256"]["data/roman_empire.npz"] == freeze["input_sha256"]["data/roman_empire.npz"]
            assert preflight["source_manifest_sha256"][f"mask{mask}/depth{depth}"] == spec_sha
            assert complete["source_manifest_sha256"][f"mask{mask}/depth{depth}"] == spec_sha
            audit = read(ROOT / prefix / "completion_audit.json")
            assert audit["status"] == "COMPLETE_CUDA_REPLAY_PASS"
            assert audit["mask"] == mask and audit["depth"] == depth
            assert audit["source_manifest_sha256"] == spec_sha
            assert len(audit["records"]) == 6
            audit_rows = {(r["seed"], r["arm"]): r for r in audit["records"]}
            assert len(audit_rows) == 6
            for seed in SEEDS:
                pair_init = {}
                for arm in ARMS:
                    key = mask, depth, seed, arm
                    run = prefix / f"seed{seed}" / arm
                    row_path = ROOT / run / "result.json"
                    row = read(row_path)
                    assert row["protocol"] == PROTOCOL and row["dataset"] == "roman"
                    assert row["published_split"] == mask and row["depth"] == depth
                    assert row["optimization_seed"] == seed and row["arm"] == arm
                    assert row["source_manifest_sha256"] == spec_sha
                    assert row["epochs_run"] == 1000
                    record = derived[key]
                    assert sha(row_path) == record["result_sha256"]
                    assert row["artifact_sha256"]["selected_predictions.npz"] == record["original_predictions_sha256"]
                    for filename in ("validation_trace.csv", "initialization.json"):
                        assert sha(ROOT / run / filename) == row["artifact_sha256"][filename]
                    check_trace(ROOT / run / "validation_trace.csv", row)
                    initial = read(ROOT / run / "initialization.json")
                    pair_init[arm] = initial
                    assert initial["paired_initial_logits_max_abs_diff"] <= 1e-5
                    selected_path = ROOT / run / "selected_decisions.npz"
                    assert sha(selected_path) == record["derived_decisions_sha256"]
                    with np.load(selected_path, allow_pickle=False) as p:
                        for part in ("valid", "test"):
                            expected = np.flatnonzero(masks["val" if part == "valid" else "test"][mask - 1])
                            idx, y = p[f"{part}_indices"], p[f"{part}_labels"]
                            member, pred = p[f"{part}_member_pred"], p[f"{part}_pool_pred"]
                            assert np.array_equal(idx, expected)
                            assert np.array_equal(y, labels[idx])
                            assert member.shape == (4, len(idx)) and pred.shape == (len(idx),)
                            assert np.issubdtype(member.dtype, np.integer) and np.issubdtype(pred.dtype, np.integer)
                            assert np.all((member >= 0) & (member < 18))
                            assert np.all((pred >= 0) & (pred < 18))
                            acc = float(np.mean(pred == y))
                            assert abs(acc - row[f"{part}_accuracy"]) < 1e-6
                            assert abs(acc - audit_rows[seed, arm][f"{part}_accuracy"]) < 1e-6
                            assert abs(acc - complete_records[key][f"{part}_accuracy"]) < 1e-6
                            if part == "test":
                                scores[key] = acc
                    checks += 1
                for field in ("canonical_tied_state_sha256", "python_rng_sha256",
                              "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
                    assert pair_init["tied"][field] == pair_init["untied_propagation"][field]
            stored = audit["paired_differences"]["tied_minus_untied_propagation"]["differences"]
            actual = [scores[mask, depth, seed, "tied"] -
                      scores[mask, depth, seed, "untied_propagation"] for seed in SEEDS]
            assert np.allclose(stored, actual, atol=1e-6, rtol=0)
    assert checks == 48
    paired = [{"mask": mask, "depth": depth, "seed": seed,
               "tied_minus_untied_accuracy_pp":
               100 * (scores[mask, depth, seed, "tied"] -
                      scores[mask, depth, seed, "untied_propagation"])}
              for mask in MASKS for depth in DEPTHS for seed in SEEDS]
    assert len(paired) == 24
    out = {"status": "COMPLETE_48_CELL_LOCAL_SCORE_AUDIT_PASS",
           "scope": "post hoc masks 1-4 of one Roman graph, two depths, three seeds per depth/mask",
           "freeze_manifest_sha256": sha(freeze_path),
           "complete_grid_audit_sha256": sha(complete_path),
           "public_npz_checked": cli.public_npz is not None,
           "paired": paired,
           "mean_paired_difference_pp": float(np.mean([r["tied_minus_untied_accuracy_pp"] for r in paired]))}
    (ROOT / "local_compact_score_audit.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("COMPLETE_48_CELL_LOCAL_SCORE_AUDIT_PASS", checks,
          "paired", len(paired), "mean_pp", out["mean_paired_difference_pp"])


if __name__ == "__main__":
    main()
