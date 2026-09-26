"""Descriptive report after both validation locks and independent test audit.

This script never selects a checkpoint or candidate. It runs only after
verify_mechanism.py has replayed every allowed test score.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("tied", "untied", "sync")
DEFAULT = (0.001, 0.0)
SEEDS = (0, 1, 2)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(name: str) -> dict:
    path = ROOT / name
    require(path.is_file(), f"Required audit artifact missing: {name}")
    return json.loads(path.read_text())


def candidate_name(lr: float, decay: float) -> str:
    return f"lr{lr:g}_wd{decay:g}"


def key(dataset: str, arm: str, lr: float, decay: float, seed: int) -> str:
    return f"{dataset}/{arm}/{candidate_name(lr, decay)}/seed{seed}"


def distribution(values: list[float]) -> dict:
    require(len(values) == 3, "Expected three paired optimizer seeds")
    return {"seeds_0_1_2": values, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values)}


def main() -> None:
    freeze = load("MECHANISM_FREEZE.json")
    lock = load("MECHANISM_VALIDATION_LOCK.json")
    original_lock = load("ORIGINAL_432_VALIDATION_SELECTION_LOCK.json")
    final_audit = load("MECHANISM_FINAL_SCORE_AUDIT.json")
    require(len(lock["cells"]) == 96 and len(original_lock["cells"]) == 432,
            "Both validation locks must be complete")
    require(lock["freeze_sha256"] == digest(ROOT / "MECHANISM_FREEZE.json") and
            original_lock["freeze_sha256"] == digest(ROOT / "FROZEN_STUDY.json") and
            freeze["base_validation_grid_freeze_sha256"] ==
            digest(ROOT / "FROZEN_STUDY.json"),
            "Frozen study and validation locks differ")
    require(final_audit["freeze_sha256"] == lock["freeze_sha256"] and
            final_audit["mechanism_validation_lock_sha256"] ==
            digest(ROOT / "MECHANISM_VALIDATION_LOCK.json") and
            final_audit["original_432_validation_lock_sha256"] ==
            digest(ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"),
            "Independent score audit does not match both validation locks")

    scored = final_audit["scores"]
    expected = set()
    for dataset in DATASETS:
        selected = tuple(lock["sync_selections"][dataset]["selected_candidate"])
        for arm in ARMS:
            for lr, decay in (dict.fromkeys((DEFAULT, selected)) if arm == "sync"
                              else (DEFAULT,)):
                for seed in SEEDS:
                    expected.add(key(dataset, arm, lr, decay, seed))
    require(set(scored) == expected, "Independent test audit has an unexpected score set")

    score_cache = {}
    for cell in sorted(expected):
        path = ROOT / "scores" / cell / "score.json"
        require(path.is_file() and digest(path) == scored[cell]["score_sha256"],
                f"Score changed after independent audit: {cell}")
        row = json.loads(path.read_text())
        require(row["checkpoint_sha256"] == lock["cells"][cell]["checkpoint_sha256"],
                f"Score checkpoint differs from validation lock: {cell}")
        score_cache[cell] = row

    table = []
    candidate_rows = []
    for dataset in DATASETS:
        selection = lock["sync_selections"][dataset]
        chosen = tuple(selection["selected_candidate"])
        for row in selection["candidate_table"]:
            candidate_rows.append({"dataset": dataset, "learning_rate": row["lr"],
                                   "weight_decay": row["weight_decay"],
                                   "valid_three_seeds": row["valid"],
                                   "mean_selected_validation_accuracy": row["mean_valid_accuracy"],
                                   "mean_selected_validation_ce": row["mean_valid_ce"],
                                   "selected": (row["lr"], row["weight_decay"]) == chosen})

        def score_values(arm: str, candidate: tuple[float, float]) -> list[float]:
            return [score_cache[key(dataset, arm, *candidate, seed)]["test_accuracy"]
                    for seed in SEEDS]

        def val_values(arm: str, candidate: tuple[float, float]) -> list[float]:
            return [lock["cells"][key(dataset, arm, *candidate, seed)][
                "selected_valid_accuracy"] for seed in SEEDS]

        tied = score_values("tied", DEFAULT)
        untied = score_values("untied", DEFAULT)
        sync = score_values("sync", DEFAULT)
        chosen_sync = score_values("sync", chosen)
        rows = (("TIED default", "tied", DEFAULT, tied),
                ("UNTIED default", "untied", DEFAULT, untied),
                ("SYNC default", "sync", DEFAULT, sync),
                ("SYNC selected", "sync", chosen, chosen_sync))
        for label, arm, candidate, scores in rows:
            table.append({"dataset": dataset, "arm": label,
                          "learning_rate": candidate[0], "weight_decay": candidate[1],
                          "validation_accuracy": distribution(val_values(arm, candidate)),
                          "test_accuracy": distribution(scores),
                          "paired_test_difference_vs_tied_default":
                          (distribution([a - b for a, b in zip(scores, tied)])
                           if label.startswith("SYNC") else None),
                          "paired_test_difference_vs_untied_default":
                          (distribution([a - b for a, b in zip(scores, untied)])
                           if label.startswith("SYNC") else None),
                          "selected_from_validation_grid": label == "SYNC selected"})

    report = {"protocol": "optimizer_aggregation_mechanism_v1",
              "mechanism_freeze_sha256": digest(ROOT / "MECHANISM_FREEZE.json"),
              "mechanism_validation_lock_sha256": digest(ROOT / "MECHANISM_VALIDATION_LOCK.json"),
              "original_432_validation_lock_sha256":
              digest(ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"),
              "independent_score_audit_sha256": digest(ROOT / "MECHANISM_FINAL_SCORE_AUDIT.json"),
              "note": "Three optimizer seeds on one fixed split per graph; means and sample SDs are descriptive. Only the three default arms form a matched optimizer intervention. Selected SYNC has six-candidate validation search.",
              "rows": table, "sync_candidate_validation": candidate_rows}
    (ROOT / "MECHANISM_SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "mechanism_main_table.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset", "arm", "learning_rate", "weight_decay",
                         "validation_accuracy_mean", "test_accuracy_mean", "test_accuracy_sample_sd",
                         "paired_test_minus_tied_mean", "paired_test_minus_untied_mean"))
        for row in table:
            writer.writerow((row["dataset"], row["arm"], row["learning_rate"],
                             row["weight_decay"], row["validation_accuracy"]["mean"],
                             row["test_accuracy"]["mean"], row["test_accuracy"]["sample_sd"],
                             row["paired_test_difference_vs_tied_default"]["mean"]
                             if row["paired_test_difference_vs_tied_default"] else "",
                             row["paired_test_difference_vs_untied_default"]["mean"]
                             if row["paired_test_difference_vs_untied_default"] else ""))
    with (ROOT / "mechanism_candidate_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    print(json.dumps({"summary_sha256": digest(ROOT / "MECHANISM_SUMMARY.json"),
                      "rows": len(table), "candidate_rows": len(candidate_rows),
                      "audited_test_scores": len(scored)}), flush=True)


if __name__ == "__main__":
    main()
