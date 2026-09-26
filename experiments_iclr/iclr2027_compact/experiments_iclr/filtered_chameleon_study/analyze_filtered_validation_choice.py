"""Apply the frozen two-placement validation rule on filtered Chameleon."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("tied", "untied_propagation", "private_first", "private_last")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def mean(values):
    return sum(values) / len(values)


def main():
    base = ROOT / "position_results/chameleon_filtered"
    freeze_path = ROOT / "POSITION_PRETRAIN_FREEZE.json"
    freeze = read(freeze_path)
    audit_path = base / "completion_audit.json"
    audit = read(audit_path)
    assert freeze["status"] == "FILTERED_POSITION_FROZEN_BEFORE_TRAINING"
    assert freeze["source_manifest_sha256"] == sha(base / "source_manifest.json")
    assert audit["status"] == "COMPLETE_CUDA_REPLAY_PASS"
    assert audit["equal_private_parameter_counts_per_seed"]
    records = {(r["seed"], r["arm"]): r for r in audit["records"]}
    assert len(audit["records"]) == 12
    assert set(records) == {(seed, arm) for seed in range(3) for arm in ARMS}
    stats = {}
    hashes = {}
    for arm in ARMS:
        rows = []
        for seed in range(3):
            path = base / f"seed{seed}" / arm / "result.json"
            row = read(path)
            full = records[seed, arm]
            assert row["dataset"] == "chameleon_filtered" and row["arm"] == arm
            assert row["optimization_seed"] == seed
            assert row["source_manifest_sha256"] == freeze["source_manifest_sha256"]
            assert row["selected_epoch"] == full["selected_epoch"]
            assert row["parameter_count"] == full["parameter_count"]
            assert abs(row["valid_accuracy"] - full["valid_accuracy"]) < 1e-6
            assert abs(row["test_accuracy"] - full["test_accuracy"]) < 1e-6
            hashes[str(path.relative_to(ROOT))] = sha(path)
            rows.append(row)
        stats[arm] = {
            "mean_validation_accuracy": mean([r["valid_accuracy"] for r in rows]),
            "mean_validation_ce": mean([r["valid_ce"] for r in rows]),
            "mean_test_accuracy": mean([r["test_accuracy"] for r in rows]),
            "test_accuracy_by_seed": [r["test_accuracy"] for r in rows],
            "selected_epoch_by_seed": [r["selected_epoch"] for r in rows],
            "parameter_count_by_seed": [r["parameter_count"] for r in rows],
        }
    assert all(stats["private_first"]["parameter_count_by_seed"][s] ==
               stats["private_last"]["parameter_count_by_seed"][s]
               for s in range(3))
    first, last = stats["private_first"], stats["private_last"]
    if first["mean_validation_accuracy"] > last["mean_validation_accuracy"]:
        chosen = "private_first"
    elif last["mean_validation_accuracy"] > first["mean_validation_accuracy"]:
        chosen = "private_last"
    elif first["mean_validation_ce"] < last["mean_validation_ce"]:
        chosen = "private_first"
    else:
        chosen = "private_last"
    output = {
        "status": "FROZEN_FILTERED_VALIDATION_CHOICE_APPLIED",
        "selection_rule": freeze["secondary_validation_choice"]["rule"],
        "unit_of_replication": "optimizer seed on official filtered-Chameleon split 0",
        "freeze_sha256": sha(freeze_path),
        "completion_audit_sha256": sha(audit_path),
        "result_sha256": hashes,
        "arms": stats,
        "selected_by_validation": chosen,
        "private_first_minus_private_last_test_pp_by_seed": [
            100 * (first["test_accuracy_by_seed"][s] -
                   last["test_accuracy_by_seed"][s]) for s in range(3)],
        "private_first_minus_private_last_mean_test_pp": 100 * (
            first["mean_test_accuracy"] - last["mean_test_accuracy"]),
        "retrospective_oracle_regret_pp": 100 * (
            max(first["mean_test_accuracy"], last["mean_test_accuracy"]) -
            stats[chosen]["mean_test_accuracy"]),
    }
    (ROOT / "FILTERED_VALIDATION_CHOICE.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: output[k] for k in
                      ("selected_by_validation", "private_first_minus_private_last_mean_test_pp",
                       "retrospective_oracle_regret_pp")}, sort_keys=True))


if __name__ == "__main__":
    main()
