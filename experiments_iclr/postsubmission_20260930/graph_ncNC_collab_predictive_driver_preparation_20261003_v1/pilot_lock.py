"""Metadata-only full-family closure; no model/data imports or TEST stage."""
from pathlib import Path
import json
import statistics
from pilot_common import UNITS, require, read_bound_json, file_sha, atomic_json, utc

ARMS = ("native_single_64", "independent_native_4", "factorized_private_4", "factorized_pooled_after_clamp_4", "native_single_70")
EXPECTED_ARMS = {"native_bank4": ARMS[:2], "factor_private4": (ARMS[2],), "factor_pooled4": (ARMS[3],), "native70": (ARMS[4],)}


def checked_local(root, receipt):
    relative = receipt["path"]
    require(Path(relative).name == relative, "Checkpoint receipt path escaped unit")
    path = root / relative
    require(path.stat().st_size == receipt["bytes"] and file_sha(path) == receipt["sha256"], "Selected checkpoint custody differs")
    return path


def run(context, output, attempts):
    bindings = context["release"].get("family_inputs", [])
    expected = {(unit, seed) for unit in UNITS for seed in range(5)}
    require(len(bindings) == 20 and {(b["unit"], b["base_seed"]) for b in bindings} == expected, "Exactly all20 complete-or-terminal-failed family units required before outcome access")
    units = {}
    terminal_failures = []
    paths = set()
    attempts.phase("complete_family_metadata_and_checkpoint_custody")
    # First pass checks every complete unit before opening any private selection.
    for binding in bindings:
        root = Path(binding["output_directory"]).resolve()
        require(root not in paths, "One execution output is reused by different units")
        require(not (root / "FAMILY_CLOSURE.json").exists(), "Family unit has already been immutably closed")
        paths.add(root)
        key = (binding["unit"], binding["base_seed"])
        disposition = binding.get("disposition", "COMPLETE")
        require(disposition in ("COMPLETE", "TERMINAL_FAILED"), "Unknown family disposition")
        if disposition == "TERMINAL_FAILED":
            require(binding.get("terminal_failure_immutable") is True and binding.get("terminal_failure_authorization_reference"), "Failed unit retirement must be explicit and immutable")
            failed = read_bound_json(root / "FAILED.json", binding["failure_sha256"])
            require(failed["schema"] == "ncnc-pilot-failed-unit-v1" and failed["identity"] == context["identity"] and failed["stage"] == "fit" and
                    (failed["unit"], failed["base_seed"]) == key and failed["status"] == "FAILED" and failed["test_file_opened"] is False, "Terminal failure contract differs")
            checked_local(root, failed["attempts_receipt"])
            ledger = read_bound_json(root / "ATTEMPTS.json", failed["attempts_receipt"]["sha256"])
            require(ledger["identity"] == context["identity"] and ledger["attempts"][-1]["status"] == "FAILED", "Failure ledger is incomplete/different")
            if failed["journal_receipt"] is not None:
                checked_local(root, failed["journal_receipt"])
            terminal_failures.append({"unit": key[0], "base_seed": key[1], "failure": failed["failure"],
                "failure_sha256": binding["failure_sha256"], "authorization_reference": binding["terminal_failure_authorization_reference"],
                "missing_arms": list(EXPECTED_ARMS[key[0]])})
            units[key] = (root, None, ledger)
            continue
        complete = read_bound_json(root / "COMPLETE.json", binding["complete_sha256"])
        require(complete["schema"] == "ncnc-pilot-complete-unit-v1" and complete["identity"] == context["identity"], "Different complete family contract")
        require((complete["unit"], complete["base_seed"]) == key and complete["epochs"] == 100, "Incomplete/different family unit")
        count = 4 if key[0] == "native_bank4" else 1
        require(complete["unique_fits"] == count and complete["completed_optimizer_steps"] == 1700 * count, "Incomplete scientific fit budget")
        require(complete["scientific_initialization"] == "fresh" and complete["resource_state_donor"] is False and complete["test_file_opened"] is False, "Unadmitted donor/TEST authority")
        require(set(complete["selected_checkpoints"]) == set(EXPECTED_ARMS[key[0]]), "Missing served arm selected state")
        require(complete["independent_candidate_count"] == (101 if count == 4 else 0) and complete["candidate101_extra_VALID_evaluations"] == (4 if count == 4 else 0), "Missing ordered bank selection/evaluation work")
        require(len(complete["member_metadata"]) == count, "Member metadata cardinality differs")
        for member, record in enumerate(complete["member_metadata"]):
            expected_seed = key[1] + 5 * member if count == 4 else key[1]
            require(record["member"] == member and record["fit_seed"] == expected_seed, "Independent/native seeds differ")
            require([e["epoch"] for e in record["epochs"]] == list(range(1, 101)), "Not all100 ordered epochs")
            for epoch in record["epochs"]:
                require(epoch["full_batches"] == epoch["optimizer_steps"] == 17 and epoch["positive_queries"] == 60084 and epoch["negative_queries"] == 100000, "Reduced TRAIN/VALID coverage")
                stream = epoch["stream"]
                require(stream["full_batches"] == 17 and stream["supervised_records"] == 1114112 and stream["dropped_tail_records"] == 64940 and stream["negative_rows_drawn"] >= 1179052, "Native stream budget differs")
        for receipt in complete["selected_checkpoints"].values():
            checked_local(root, receipt)
        for receipt in complete["selection_artifacts"].values():
            checked_local(root, receipt)
        ledger = json.loads((root / "ATTEMPTS.json").read_text())
        require(ledger["identity"] == context["identity"] and ledger["attempts"][-1]["status"] == "COMPLETE", "Unclosed final execution accounting")
        units[key] = (root, complete, ledger)
    completed_unique_fits = sum(v[1]["unique_fits"] for v in units.values() if v[1] is not None)
    full_success = not terminal_failures
    require(not full_success or completed_unique_fits == 35, "Full35-fit successful family required")
    paired_complete_seeds = []
    for seed in range(5):
        if units[("factor_private4", seed)][1] is None or units[("factor_pooled4", seed)][1] is None:
            continue
        paired_complete_seeds.append(seed)
        private = units[("factor_private4", seed)][1]["member_metadata"][0]
        pooled = units[("factor_pooled4", seed)][1]["member_metadata"][0]
        require(private["initial_state_sha256"] == pooled["initial_state_sha256"] and private["initial_rng_sha256"] == pooled["initial_rng_sha256"], "F4 twins did not share the exact fresh initial model/emptyAdam/flags/RNG")
        for p, q in zip(private["epochs"], pooled["epochs"]):
            require(p["stream"] == q["stream"] and p["start_rng_sha256"] == q["start_rng_sha256"] and p["end_rng_sha256"] == q["end_rng_sha256"], "F4 paired native sampler/permutation/dropout schedule differs")
    attempts.phase("complete_family_private_selection_disclosure")
    cells = []
    for (unit, seed), (root, complete, ledger) in sorted(units.items()):
        for arm in EXPECTED_ARMS[unit]:
            if complete is None:
                cells.append({"arm": arm, "base_seed": seed, "status": "MISSING_TERMINAL_FAILED",
                              "served_VALID_hits50": None, "checkpoint": None})
                continue
            relative = complete["selection_artifacts"][arm]["path"]
            require(relative == "PRIVATE_SELECTION_" + arm + ".json", "Selection artifact path differs")
            selected = json.loads((root / relative).read_text())
            require(selected["identity"] == context["identity"] and selected["arm"] == arm and selected["seed"] == seed and selected["checkpoint"] == complete["selected_checkpoints"][arm], "Selection/checkpoint provenance differs")
            candidate = selected["selection"]
            require(type(candidate["hits50"]) in (float, int) and 0 <= candidate["hits50"] <= 1, "Invalid complete served VALID metric")
            require(1 <= candidate["order"] <= (101 if arm == ARMS[1] else 100), "Selected candidate outside frozen order")
            cells.append({"arm": arm, "base_seed": seed, "status": "COMPLETE", "served_VALID_hits50": candidate["hits50"],
                          "selection": candidate, "checkpoint": {"output_directory": str(root), **selected["checkpoint"]},
                          "native64_exact_member0_donor_reuse": arm == ARMS[0]})
    require(len(cells) == 25, "All25 served cells required")
    values = {(c["arm"], c["base_seed"]): c["served_VALID_hits50"] for c in cells}
    differences = [values[(ARMS[2], seed)] - values[(ARMS[3], seed)] for seed in range(5)] if full_success else None
    arm_summaries = {arm: {"mean_VALID_hits50": statistics.mean(values[(arm, s)] for s in range(5)),
                           "sample_sd": statistics.stdev(values[(arm, s)] for s in range(5))} for arm in ARMS} if full_success else None
    unit_costs = []
    for (unit, seed), (root, complete, ledger) in sorted(units.items()):
        attempts_rows = ledger["attempts"]
        closed = [a["inclusive_wall_seconds"] for a in attempts_rows if a.get("inclusive_wall_seconds") is not None]
        unit_costs.append({"unit": unit, "base_seed": seed, "status": "COMPLETE" if complete else "TERMINAL_FAILED",
            "unique_fits_completed": complete["unique_fits"] if complete else 0, "unique_fits_planned": 4 if unit == "native_bank4" else 1,
            "all_closed_attempt_wall_seconds": sum(closed), "total_cost_exact": all(a.get("inclusive_wall_seconds") is not None for a in attempts_rows),
            "terminal_accounting_write_tail_measured": False,
            "failed_or_interrupted_attempts": sum(a["status"] in ("FAILED", "INTERRUPTED_UNCLOSED") for a in attempts_rows),
            "completed_TRAIN_wall_seconds": sum(e["train_wall_seconds"] for m in complete["member_metadata"] for e in m["epochs"]) if complete else None,
            "completed_epoch_VALID_wall_seconds": sum(e["valid_wall_seconds"] for m in complete["member_metadata"] for e in m["epochs"]) if complete else None,
            "extra_bank_VALID_evaluations": complete["candidate101_extra_VALID_evaluations"] if complete else None,
            "cost_attribution": "N64_reuses_I4_member0_exact_fit; all_four_fit_and_101candidate_work_is_charged_to_native_bank4" if unit == "native_bank4" else "one_unique_fit"})
    return {"schema": "ncnc-pilot-full-family-lock-v1", "identity": context["identity"], "status": "COMPLETE_FAMILY_LOCKED" if full_success else "TERMINAL_FAILED_FAMILY_LOCKED",
            "unique_fits_planned": 35, "unique_fits_completed": completed_unique_fits, "served_arm_seed_cells": 25,
            "complete_served_cells": sum(c["status"] == "COMPLETE" for c in cells), "cells": cells, "arm_summaries": arm_summaries,
            "primary_development_pilot": {"contrast": "separately_VALID_selected_private_minus_pooled",
                "paired_base_seeds": list(range(5)), "paired_VALID_hits50_differences": differences,
                "mean_difference": statistics.mean(differences) if full_success else None, "sample_sd": statistics.stdev(differences) if full_success else None,
                "range": [min(differences), max(differences)] if full_success else None,
                "sign_count": {"private_greater": sum(d > 0 for d in differences), "equal": sum(d == 0 for d in differences),
                               "pooled_greater": sum(d < 0 for d in differences)} if full_success else None,
                "paired_seed_meaning": "training_randomness_conditional_on_one_fixed_graph_and_time_split",
                "interpretation": "development_pilot; VALID_is_used_for_selection_and_this_contrast"},
            "baseline_quality_and_cost_comparisons": "exploratory", "unit_costs": unit_costs,
            "terminal_failures": terminal_failures, "no_success_only_subset_summary": True,
            "paired_F4_initialization_and_all100_epoch_stream_RNG_checked_seeds": paired_complete_seeds,
            "native64_exact_member0_fit_reuse_disclosed": True, "test_file_opened": False,
            "TEST_execution_authorized": False, "novelty_or_baseline_superiority_claim": False,
            "inputs": bindings, "UTC": utc()}


def commit_closure_markers(context, output, result):
    """Retire all bound units after the immutable complete-or-failed lock."""
    lock_path = Path(output) / "FAMILY_LOCK.json"
    marker = {"schema": "ncnc-pilot-immutable-family-closure-v1", "identity": context["identity"],
              "family_lock_path": str(lock_path), "family_lock_sha256": file_sha(lock_path),
              "status": result["status"], "resume_or_seed_replacement_permitted": False}
    for binding in result["inputs"]:
        target = Path(binding["output_directory"]) / "FAMILY_CLOSURE.json"
        require(not target.exists(), "Unit already belongs to an immutable closure")
        atomic_json(target, marker)
