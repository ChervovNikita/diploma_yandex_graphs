"""Report every fixed default NORM-SYNC case after independent score audits.

This script selects no candidate or checkpoint and reads no raw test labels.
It requires all three complete validation locks and both independent score
audits before reporting held-out scores.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
DEFAULT = "lr0.001_wd0"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(name: str) -> dict:
    path = ROOT / name
    require(path.is_file(), f"Required audited artifact missing: {name}")
    return json.loads(path.read_text())


def distribution(values: list[float]) -> dict:
    require(len(values) == 3, "Expected three paired optimizer seeds")
    return {"seeds_0_1_2": values, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values)}


def main() -> None:
    norm_freeze = load("NORM_SYNC_V2_FREEZE.json")
    v1_diag = load("NORM_SYNC_ROUNDING_DIAG_V1_CORA_SEED0.json")
    require(v1_diag["failed_epoch"] == 510 and v1_diag["test_metrics_computed"] is False,
            "V1 failure chronology differs")
    mechanism_freeze = load("MECHANISM_FREEZE.json")
    original_freeze = load("FROZEN_STUDY.json")
    norm_lock = load("NORM_SYNC_V2_VALIDATION_LOCK.json")
    mechanism_lock = load("MECHANISM_VALIDATION_LOCK.json")
    original_lock = load("ORIGINAL_432_VALIDATION_SELECTION_LOCK.json")
    norm_audit = load("NORM_SYNC_V2_FINAL_SCORE_AUDIT.json")
    mechanism_audit = load("MECHANISM_FINAL_SCORE_AUDIT.json")
    require(norm_freeze["protocol"] == norm_lock["protocol"] ==
            norm_audit["protocol"] == "norm_matched_sync_v2" and
            norm_lock["freeze_sha256"] == norm_audit["freeze_sha256"] ==
            sha(ROOT / "NORM_SYNC_V2_FREEZE.json"),
            "NORM-SYNC freeze, lock, or score audit differs")
    require(mechanism_lock["freeze_sha256"] ==
            mechanism_audit["freeze_sha256"] ==
            sha(ROOT / "MECHANISM_FREEZE.json") and
            original_lock["freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            norm_freeze["mechanism_freeze_sha256"] ==
            sha(ROOT / "MECHANISM_FREEZE.json") and
            norm_freeze["base_freeze_sha256"] ==
            sha(ROOT / "FROZEN_STUDY.json") and
            mechanism_freeze["base_validation_grid_freeze_sha256"] ==
            sha(ROOT / "FROZEN_STUDY.json") and
            original_freeze["protocol"] == original_lock["protocol"],
            "Source/data freezes and validation locks differ")
    require(len(norm_lock["cells"]) == 12 and
            len(mechanism_lock["cells"]) == 96 and
            len(original_lock["cells"]) == 432 and
            norm_audit["norm_validation_lock_sha256"] ==
            sha(ROOT / "NORM_SYNC_V2_VALIDATION_LOCK.json") and
            norm_audit["mechanism_validation_lock_sha256"] ==
            sha(ROOT / "MECHANISM_VALIDATION_LOCK.json") and
            norm_audit["original_432_validation_lock_sha256"] ==
            sha(ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json") and
            mechanism_audit["mechanism_validation_lock_sha256"] ==
            sha(ROOT / "MECHANISM_VALIDATION_LOCK.json") and
            mechanism_audit["original_432_validation_lock_sha256"] ==
            sha(ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"),
            "Complete validation or independent score audit gate differs")

    expected_norm = {f"{dataset}/seed{seed}"
                     for dataset in DATASETS for seed in SEEDS}
    require(set(norm_audit["scores"]) == expected_norm,
            "NORM-SYNC score audit is incomplete or contains extra cells")
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            norm_key = f"{dataset}/seed{seed}"
            norm_score_file = ROOT / "scores_norm_sync_v2" / norm_key / "score.json"
            norm_result_file = ROOT / "results_norm_sync_v2" / norm_key / "result.json"
            require(sha(norm_score_file) == norm_audit["scores"][norm_key]["score_sha256"] and
                    sha(norm_result_file) == norm_lock["cells"][norm_key]["result_sha256"],
                    f"NORM-SYNC score or validation result changed: {norm_key}")
            norm_score = json.loads(norm_score_file.read_text())
            norm_result = json.loads(norm_result_file.read_text())
            arm_scores = {}
            arm_validation = {}
            for arm in ("tied", "sync"):
                key = f"{dataset}/{arm}/{DEFAULT}/seed{seed}"
                require(key in mechanism_audit["scores"] and
                        key in mechanism_lock["cells"],
                        f"Default mechanism cell not audited: {key}")
                score_file = ROOT / "scores" / key / "score.json"
                require(sha(score_file) ==
                        mechanism_audit["scores"][key]["score_sha256"],
                        f"Default mechanism score changed: {key}")
                scored = json.loads(score_file.read_text())
                arm_scores[arm] = scored["test_accuracy"]
                arm_validation[arm] = mechanism_lock["cells"][key][
                    "selected_valid_accuracy"]
            rows.append({
                "dataset": dataset, "seed": seed,
                "tied_default_valid_accuracy": arm_validation["tied"],
                "sync_default_valid_accuracy": arm_validation["sync"],
                "norm_sync_default_valid_accuracy": norm_result[
                    "selected_valid_accuracy"],
                "tied_default_test_accuracy": arm_scores["tied"],
                "sync_default_test_accuracy": arm_scores["sync"],
                "norm_sync_default_test_accuracy": norm_score["test_accuracy"],
                "norm_minus_tied_test_accuracy":
                    norm_score["test_accuracy"] - arm_scores["tied"],
                "norm_minus_sync_test_accuracy":
                    norm_score["test_accuracy"] - arm_scores["sync"],
                "max_relative_graph_step_norm_error":
                    norm_result["max_relative_norm_error"],
                "rounding_corrected_steps": norm_result["rounding_corrected_steps"],
                "max_abs_relative_scale_correction": norm_result["max_abs_relative_scale_correction"],
                "min_graph_step_scale": norm_result["min_norm_scale"],
                "max_graph_step_scale": norm_result["max_norm_scale"],
                "norm_score_sha256": sha(norm_score_file),
                "norm_result_sha256": sha(norm_result_file),
            })
    require(len(rows) == 12, "Expected the complete fixed 12-cell matrix")
    aggregates = []
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        aggregates.append({
            "dataset": dataset,
            "tied_default_test_accuracy": distribution([
                row["tied_default_test_accuracy"] for row in subset]),
            "sync_default_test_accuracy": distribution([
                row["sync_default_test_accuracy"] for row in subset]),
            "norm_sync_default_test_accuracy": distribution([
                row["norm_sync_default_test_accuracy"] for row in subset]),
            "norm_minus_tied_test_accuracy": distribution([
                row["norm_minus_tied_test_accuracy"] for row in subset]),
            "norm_minus_sync_test_accuracy": distribution([
                row["norm_minus_sync_test_accuracy"] for row in subset]),
            "maximum_relative_graph_step_norm_error": max(
                row["max_relative_graph_step_norm_error"] for row in subset),
            "rounding_corrected_steps": sum(row["rounding_corrected_steps"] for row in subset),
            "maximum_abs_relative_scale_correction": max(
                row["max_abs_relative_scale_correction"] for row in subset),
        })
    report = {
        "protocol": "norm_matched_sync_descriptive_summary_v2",
        "norm_freeze_sha256": sha(ROOT / "NORM_SYNC_V2_FREEZE.json"),
        "norm_validation_lock_sha256": sha(ROOT / "NORM_SYNC_V2_VALIDATION_LOCK.json"),
        "mechanism_validation_lock_sha256":
            sha(ROOT / "MECHANISM_VALIDATION_LOCK.json"),
        "original_432_validation_lock_sha256":
            sha(ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"),
        "norm_independent_score_audit_sha256":
            sha(ROOT / "NORM_SYNC_V2_FINAL_SCORE_AUDIT.json"),
        "mechanism_independent_score_audit_sha256":
            sha(ROOT / "MECHANISM_FINAL_SCORE_AUDIT.json"),
        "scope": "All 12 fixed default norm-matched cells against paired default TIED and SYNC; three optimizer seeds on one split per graph; no candidate selection",
        "interpretation_limit": "Matching one global graph-step norm does not equalize update direction, coordinatewise AdamW moments, shared boundary trajectories, or later graph weights",
        "v1_failure_chronology": {
            "failed_cell": "cora/seed0", "failed_epoch": 510,
            "v1_source_freeze_sha256": sha(ROOT / "NORM_SYNC_FREEZE.json"),
            "training_only_rounding_diagnostic_sha256": sha(ROOT / "NORM_SYNC_ROUNDING_DIAG_V1_CORA_SEED0.json"),
            "v1_test_metrics_computed": False,
            "v2_frozen_after_v1_training_failure": True,
        },
        "rows": rows,
        "graphs": aggregates,
    }
    out = ROOT / "NORM_SYNC_V2_SUMMARY.json"
    csv_file = ROOT / "norm_sync_v2_all12_table.csv"
    require(not out.exists() and not csv_file.exists(),
            "Refusing to overwrite existing NORM-SYNC descriptive summary")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with csv_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "graphs": len(aggregates),
                      "summary_sha256": sha(out), "csv_sha256": sha(csv_file)}),
          flush=True)


if __name__ == "__main__":
    main()
