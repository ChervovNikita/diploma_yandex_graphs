"""Descriptive tables from the independently locked and replayed study.

This reporting script was prepared after the training source freeze; it is not
part of the prospective model/selection source. It refuses to summarize test
scores without the complete final score audit.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import verify_tuning as audit


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    final_path = ROOT / "FINAL_SCORE_AUDIT.json"
    require(lock_path.is_file() and final_path.is_file(),
            "Complete validation lock and fresh final score audit required")
    lock, final = json.loads(lock_path.read_text()), json.loads(final_path.read_text())
    require(len(lock["cells"]) == 216 and
            final["selection_lock_sha256"] == audit.sha(lock_path) and
            final["freeze_sha256"] == lock["freeze_sha256"],
            "Audit/freeze/lock mismatch")
    rows, paired, validation = [], [], []
    for dataset in audit.DATASETS:
        selected_accuracy = {}
        for arm in audit.ARMS:
            group = lock["selections"][dataset][arm]
            selected = tuple(group["selected_candidate"])
            candidate_table = group["candidate_table"]
            require(len(candidate_table) == 6, "Incomplete candidate table")
            for cand in candidate_table:
                validation.append({"dataset": dataset, "arm": arm,
                                   "lr": cand["lr"], "weight_decay": cand["weight_decay"],
                                   "valid": cand["valid"],
                                   "mean_validation_accuracy": cand["mean_valid_accuracy"],
                                   "mean_validation_ce": cand["mean_valid_ce"],
                                   "selected": (cand["lr"], cand["weight_decay"]) == selected})
            configuration_results = {}
            for role, configuration in (("selected", selected), ("default", audit.DEFAULT)):
                lr, wd = configuration
                accuracies, epochs, times, counts = [], [], [], []
                for seed in audit.SEEDS:
                    key = audit.cell_key(dataset, arm, lr, wd, seed)
                    require(key in final["scores"], f"Missing score audit entry {key}")
                    path = audit.cell_dir(dataset, arm, lr, wd, seed)
                    score = json.loads((ROOT / "scores" / key / "score.json").read_text())
                    train = json.loads((path / "result.json").read_text())
                    # Match the independently audited float32 replay tolerance.
                    require(abs(score["test_accuracy"] - final["scores"][key]["test_accuracy"]) <= 1e-7,
                            f"Score differs from independent final audit: {key}")
                    accuracies.append(score["test_accuracy"])
                    epochs.append(train["selected_epoch"])
                    times.append(train["training_seconds"])
                    counts.append(train["initialization"]["parameter_count"])
                require(len(set(counts)) == 1, "Parameter count varies across seeds")
                configuration_results[role] = accuracies
                rows.append({
                    "dataset": dataset, "arm": arm, "role": role,
                    "lr": lr, "weight_decay": wd,
                    "mean_validation_accuracy": next(c["mean_valid_accuracy"] for c in candidate_table
                                                     if (c["lr"], c["weight_decay"]) == configuration),
                    "test_seed0": accuracies[0], "test_seed1": accuracies[1],
                    "test_seed2": accuracies[2],
                    "mean_test_accuracy": statistics.mean(accuracies),
                    "sd_test_accuracy_across_optimizer_seeds": statistics.stdev(accuracies),
                    "parameter_count": counts[0],
                    "selected_epoch_seed0": epochs[0], "selected_epoch_seed1": epochs[1],
                    "selected_epoch_seed2": epochs[2],
                    "training_seconds_all_three_seeds": sum(times),
                    "partial_family_selected": (
                        arm == lock["partial_family_choice"][dataset]["selected_arm"]),
                })
            selected_accuracy[arm] = configuration_results["selected"]
        for arm in audit.ARMS:
            for comparator in ("base", "ens", "tied", "untied"):
                if arm == comparator:
                    continue
                differences = [(a - b) * 100 for a, b in
                               zip(selected_accuracy[arm], selected_accuracy[comparator])]
                paired.append({"dataset": dataset, "arm": arm, "comparator": comparator,
                               "seed0_difference_pp": differences[0],
                               "seed1_difference_pp": differences[1],
                               "seed2_difference_pp": differences[2],
                               "mean_difference_pp": statistics.mean(differences)})
    output = ROOT / "summary"
    output.mkdir(exist_ok=True)
    for name, content in (("selected_default.csv", rows), ("validation_grid.csv", validation),
                          ("paired_seed_differences.csv", paired)):
        with (output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    (output / "provenance.json").write_text(json.dumps({
        "freeze_sha256": lock["freeze_sha256"],
        "selection_lock_sha256": audit.sha(lock_path),
        "final_score_audit_sha256": audit.sha(final_path),
        "selected_default_rows": len(rows), "validation_rows": len(validation),
        "paired_rows": len(paired),
        "interpretation": "Descriptive tuned accuracy/storage comparison on one published split per graph; optimizer seed variation only",
        "partial_family_search_cost": "12 configurations versus 6 per individual comparator arm",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary_folder": str(output), "rows": len(rows),
                      "validation_candidates": len(validation)}))


if __name__ == "__main__":
    main()
