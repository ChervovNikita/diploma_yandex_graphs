"""Supplemental post-freeze exact-decision replay of allowed test scores.

Prepared before any test scoring. The frozen verifier checks all 432
validation cells and numeric selected/default replay. This script adds an
explicit per-node pooled argmax decision comparison across runtime versions.
It cannot change the validation lock or choose a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import tuning as study
import verify_tuning as audit


ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise RuntimeError(label)


def main():
    freeze_sha = study.check_freeze()
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    final_path = ROOT / "FINAL_SCORE_AUDIT.json"
    lock, final = json.loads(lock_path.read_text()), json.loads(final_path.read_text())
    check(lock["freeze_sha256"] == freeze_sha and len(lock["cells"]) == 432,
          "Incomplete or mismatched validation lock")
    check(final["selection_lock_sha256"] == sha(lock_path), "Final audit/lock hash mismatch")
    rows = {}
    expected = set()
    device = torch.device("cuda")
    for dataset in audit.DATASETS:
        bundle, _ = study.load_graph(dataset, device, include_test=True)
        for arm in audit.ARMS:
            selected = tuple(lock["selections"][dataset][arm]["selected_candidate"])
            for lr, wd in dict.fromkeys((selected, audit.DEFAULT)):
                for seed in audit.SEEDS:
                    key = audit.cell_key(dataset, arm, lr, wd, seed)
                    expected.add(key)
                    check(key in final["scores"], f"Missing final score audit entry: {key}")
                    folder = audit.cell_dir(dataset, arm, lr, wd, seed)
                    score = ROOT / "scores" / key
                    check(sha(folder / "checkpoint.pt") == lock["cells"][key]["checkpoint_sha256"],
                          f"Checkpoint changed: {key}")
                    study.seed_all(seed)
                    model, _ = study.make_model(arm, bundle, device)
                    saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                    model.load_state_dict(saved["state_dict"], strict=True)
                    va, vc, vp, _ = audit.independent_metrics(
                        model, arm, bundle, bundle.valid_idx, bundle.valid_y)
                    ta, tc, tp, _ = audit.independent_metrics(
                        model, arm, bundle, bundle.test_idx, bundle.test_y)
                    with np.load(score / "predictions.npz", allow_pickle=False) as earlier:
                        earlier_valid = earlier["valid_pooled_logits"]
                        earlier_test = earlier["test_pooled_logits"]
                    valid_new, test_new = vp.numpy(), tp.numpy()
                    valid_diff = float(np.max(np.abs(earlier_valid - valid_new)))
                    test_diff = float(np.max(np.abs(earlier_test - test_new)))
                    valid_decisions = int(np.count_nonzero(
                        earlier_valid.argmax(axis=-1) != valid_new.argmax(axis=-1)))
                    test_decisions = int(np.count_nonzero(
                        earlier_test.argmax(axis=-1) != test_new.argmax(axis=-1)))
                    check(np.allclose(earlier_valid, valid_new, rtol=1e-5, atol=1e-5) and
                          np.allclose(earlier_test, test_new, rtol=1e-5, atol=1e-5) and
                          valid_decisions == 0 and test_decisions == 0,
                          f"Numeric or decision replay failed: {key}")
                    rows[key] = {"valid_max_abs_diff": valid_diff,
                                 "test_max_abs_diff": test_diff,
                                 "valid_decision_mismatches": valid_decisions,
                                 "test_decision_mismatches": test_decisions,
                                 "valid_accuracy": va, "valid_ce": vc,
                                 "test_accuracy": ta, "test_ce": tc}
    check(set(rows) == expected and set(rows) == set(final["scores"]),
          "Replay does not cover exactly the allowed score set")
    report = {"freeze_sha256": freeze_sha,
              "selection_lock_sha256": sha(lock_path),
              "final_score_audit_sha256": sha(final_path),
              "script_sha256": sha(Path(__file__)),
              "replayed_allowed_checkpoints": len(rows),
              "max_valid_abs_diff": max(r["valid_max_abs_diff"] for r in rows.values()),
              "max_test_abs_diff": max(r["test_max_abs_diff"] for r in rows.values()),
              "total_valid_decision_mismatches": 0,
              "total_test_decision_mismatches": 0,
              "per_checkpoint": rows}
    study.write_json(ROOT / "STRICT_DECISION_AUDIT.json", report)
    print(json.dumps({"replayed_allowed_checkpoints": len(rows),
                      "report_sha256": sha(ROOT / "STRICT_DECISION_AUDIT.json")}))


if __name__ == "__main__":
    main()
