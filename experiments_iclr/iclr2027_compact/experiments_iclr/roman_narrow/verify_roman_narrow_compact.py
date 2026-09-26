"""Independently check the anonymous six-cell narrow-control evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEPTHS = (2, 5)
SEEDS = (0, 1, 2)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    z = logits.astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=-1)) -
                         z[np.arange(len(labels)), labels]))


def trace_check(path: Path, row: dict) -> None:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1000
    best_acc, best_ce, best_epoch = -1.0, math.inf, 0
    for epoch, item in enumerate(rows, 1):
        assert int(item["epoch"]) == epoch
        acc, loss = float(item["valid_accuracy"]), float(item["valid_ce"])
        assert math.isfinite(acc) and math.isfinite(loss) and 0 <= acc <= 1
        selected = acc > best_acc or (acc == best_acc and loss < best_ce)
        assert int(item["selected_now"]) == int(selected)
        if selected:
            best_acc, best_ce, best_epoch = acc, loss, epoch
    assert best_epoch == row["selected_epoch"]
    assert abs(best_acc - row["valid_accuracy"]) < 1e-7
    assert abs(best_ce - row["valid_ce"]) < 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roman24", type=Path, required=True)
    args = parser.parse_args()
    roman24 = args.roman24.resolve()
    manifest = read(ROOT / "derivation_manifest.json")
    freeze = read(ROOT / "SOURCE_FREEZE.json")
    amendment = read(ROOT / "INITIAL_DECISION_AMENDMENT_FREEZE.json")
    diagnosis = read(ROOT / "initial_replay_all6_diagnostic.json")
    lock = read(ROOT / "validation_lock.json")
    audit = read(ROOT / "complete_score_audit.json")
    source_sha = sha(ROOT / "SOURCE_FREEZE.json")
    assert manifest["status"] == "COMPLETE_6_CELL_NARROW_COMPACT_DERIVATION_PASS"
    assert source_sha == manifest["source_freeze_sha256"] == audit["source_freeze_sha256"]
    assert freeze["source_sha256"] == {name: sha(ROOT / name) for name in
                                      ("roman_narrow_untied.py", "verify_roman_narrow_untied.py",
                                       "ROMAN_NARROW_UNTIED_PROTOCOL.md")}
    assert sha(ROOT / "ROMAN_NARROW_UNTIED_WIDTH_LOCK.json") == freeze["width_lock_sha256"] == manifest["width_lock_sha256"]
    assert sha(ROOT / "verify_roman_narrow_initial_amendment.py") == amendment["amended_verifier_sha256"] == manifest["amendment_source_sha256"]
    assert sha(ROOT / "initial_replay_all6_diagnostic.json") == amendment["all_six_validation_only_diagnosis_sha256"]
    assert diagnosis["test_ids_or_labels_accessed"] is False and len(diagnosis["records"]) == 6
    assert amendment["original_validation_failure_log_sha256"] == manifest["original_failure_log_sha256"]
    assert sha(ROOT / "original_validation_failure_redacted.log") == manifest["redacted_failure_log_sha256"]
    assert sha(ROOT / "prepare_roman_narrow_compact.py") == manifest["derivation_script_sha256"]
    assert sha(ROOT / "validation_lock.json") == manifest["validation_lock_sha256"] == audit["validation_lock_sha256"]
    assert sha(ROOT / "complete_score_audit.json") == manifest["complete_score_audit_sha256"]
    assert lock["status"] == "COMPLETE_6_CELL_NARROW_VALIDATION_REPLAY_LOCK" and len(lock["records"]) == 6
    assert lock["test_scored"] is False
    assert lock["initial_decision_amendment_sha256"] == sha(ROOT / "INITIAL_DECISION_AMENDMENT_FREEZE.json")
    assert audit["status"] == "COMPLETE_6_CELL_NARROW_CUDA_TEST_REPLAY_PASS"
    assert len(audit["records"]) == len(audit["paired"]) == len(manifest["records"]) == 6
    assert sha(roman24 / "validation_lock.json") == freeze["original_validation_lock_sha256"]
    assert sha(roman24 / "complete_score_audit.json") == freeze["original_test_audit_sha256"]
    assert sha(roman24 / "official_labels_mask0.npz") == manifest["roman24_anchor_sha256"]
    with np.load(roman24 / "official_labels_mask0.npz", allow_pickle=False) as p:
        labels = p["node_labels"].copy()
        ids = {part: np.flatnonzero(p[f"{part}_mask"])
               for part in ("val", "test")}
    keyed_lock = {(r["depth"], r["seed"]): r for r in lock["records"]}
    keyed_audit = {(r["depth"], r["seed"]): r for r in audit["records"]}
    keyed_pair = {(r["depth"], r["seed"]): r for r in audit["paired"]}
    keyed_manifest = {(r["depth"], r["seed"]): r for r in manifest["records"]}
    assert all(len(x) == 6 for x in
               (keyed_lock, keyed_audit, keyed_pair, keyed_manifest))
    all_contrasts = []
    for depth in DEPTHS:
        for seed in SEEDS:
            run = Path("results/roman") / f"depth{depth}" / f"seed{seed}" / "untied"
            test = Path("test_scores") / f"depth{depth}" / f"seed{seed}" / "untied"
            row, score = read(ROOT / run / "result.json"), read(ROOT / test / "test_result.json")
            evidence = ROOT / run / "selected_evidence.npz"
            record = keyed_manifest[depth, seed]
            assert sha(ROOT / run / "result.json") == record["result_sha256"] == keyed_lock[depth, seed]["result_sha256"]
            assert sha(ROOT / test / "test_result.json") == record["test_result_sha256"] == keyed_audit[depth, seed]["narrow_test_result_sha256"]
            assert sha(evidence) == record["derived_evidence_sha256"]
            assert row["artifact_sha256"]["selected_validation.npz"] == record["original_validation_sha256"]
            assert score["test_predictions_sha256"] == record["original_test_sha256"]
            assert score["validation_lock_sha256"] == manifest["validation_lock_sha256"]
            trace_check(ROOT / run / "validation_trace.csv", row)
            with np.load(evidence, allow_pickle=False) as p:
                assert set(p.files) == {"valid_pool_pred", "test_pool_logits",
                                        "valid_member_pred", "test_member_pred"}
                for part in ("valid", "test"):
                    member = p[f"{part}_member_pred"]
                    y = labels[ids["val" if part == "valid" else "test"]]
                    assert member.shape == (4, len(y)) and np.issubdtype(member.dtype, np.integer)
                    assert np.all(member < 18)
                    if part == "valid":
                        pred = p["valid_pool_pred"]
                        assert pred.shape == (len(y),) and np.all(pred < 18)
                    else:
                        pool = p["test_pool_logits"]
                        assert pool.dtype == np.float32 and pool.shape == (len(y), 18)
                        assert np.isfinite(pool).all()
                        pred = pool.argmax(-1)
                        assert abs(ce(pool, y) - score["test_ce"]) < 1e-4
                    acc = float(np.mean(pred == y))
                    assert abs(acc - (row if part == "valid" else score)[f"{part}_accuracy"]) < 1e-6
            baseline = freeze["baseline_tied128_artifact_sha256"][f"{depth}/{seed}"]
            tied_run = roman24 / run.parent / "tied" / "result.json"
            tied_test = roman24 / test.parent / "tied" / "test_result.json"
            assert sha(tied_run) == baseline["result_sha256"]
            assert sha(tied_test) == baseline["test_result_sha256"]
            tied_acc = read(tied_test)["test_accuracy"]
            contrast = 100 * (tied_acc - score["test_accuracy"])
            assert abs(contrast - keyed_pair[depth, seed]["tied128_minus_narrow_untied_pp"]) < 1e-9
            all_contrasts.append((depth, seed, contrast))
    print(json.dumps({"status": "NARROW_COMPACT_6_CELL_PASS",
                      "paired_tied128_minus_narrow_untied_pp": all_contrasts},
                     indent=2))


if __name__ == "__main__":
    main()
