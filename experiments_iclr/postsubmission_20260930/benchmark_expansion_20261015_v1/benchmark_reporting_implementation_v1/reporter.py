"""Strict metadata reporting for the accepted ten-cohort exploratory protocol.

Reads JSON only. NumPy is used only for the specified descriptive bootstrap.
No tensor/model/evaluator imports, prediction reconstruction or score selection.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

HASH_FIELDS = ("native_source", "data", "controller", "evaluator", "readout")
IDENTITY_FIELDS = ("dataset", "preset_id", "track", "cohort_id", "cohort_seed", "split_index",
                   "literal_split_seed", "scientific_method_id", "runtime_procedure_id", "readout", "fit_id", "actual_seed")
OFFSETS = (0, 100003, 200003, 300007)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def authority(value):
    return isinstance(value, dict) and isinstance(value.get("path"), str) and type(value.get("bytes")) is int \
        and value["bytes"] > 0 and isinstance(value.get("sha256"), str) and len(value["sha256"]) == 64 \
        and all(c in "0123456789abcdef" for c in value["sha256"])


def row_key(row):
    return tuple(row[k] for k in ("preset_id", "track", "cohort_id", "scientific_method_id", "readout"))


def extract_runtime(descriptor, result, identity, binding_evidence=None):
    """Normalize actual run_fit RESULT metadata; never fills missing authority."""
    require(result.get("schema") == "native-full-epoch-fit-result-v1", "Unsupported runtime RESULT schema")
    require(result["job_id"] == descriptor["id"] == identity["fit_id"], "Fit/job identity mismatch")
    require(result["job_hash"] == digest(descriptor), "Runtime job hash differs from descriptor")
    require(result["cohort_id"] == descriptor["cohort_id"] == identity["cohort_id"], "Cohort identity mismatch")
    for target, value in (("dataset", descriptor["recipe"]["dataset"]), ("preset_id", descriptor["recipe"]["id"]),
                          ("actual_seed", descriptor["seed"]), ("split_index", descriptor.get("split", 0)),
                          ("runtime_procedure_id", descriptor["procedure"])):
        require(identity[target] == value, "Descriptor identity mismatch: " + target)
    require(result["dataset"] == identity["dataset"] and result["seed"] == identity["actual_seed"]
            and result.get("split", 0) == identity["split_index"] and result["procedure"] == identity["runtime_procedure_id"],
            "RESULT identity differs from descriptor")
    require(identity["readout"] == "actual", "Runtime RESULT exposes actual readout only; use an authenticated mean sidecar")
    require(result.get("TEST_accessed") is False, "Only closed-TEST TRAIN/VALID metadata is supported")
    policy_track = descriptor["policy"]["track"]
    expected_track = "source_reference" if policy_track == "source_reference" else "matched_extended"
    require(identity["track"] == expected_track, "Source/matched track mismatch")
    value = result.get("selected_VALID") or {}
    row = copy.deepcopy(identity)
    row["declared_bindings"] = row.pop("bindings", {})
    row["bindings"] = {}
    if binding_evidence is not None:
        require(binding_evidence.get("job_hash") == result["job_hash"]
                and binding_evidence.get("selected_export_sha256") == (result.get("selected_VALID_export") or {}).get("sha256"),
                "Binding evidence does not anchor the actual job/export")
        require(authority(binding_evidence.get("producer_receipt")), "Binding evidence requires its producer receipt")
        row["bindings"] = copy.deepcopy(binding_evidence["bindings"])
        row["binding_evidence"] = copy.deepcopy(binding_evidence)
    row.update(schema="exploratory-predictor-row-v1", status=result["status"],
        selected_served_metric=(value.get("metrics") or {}).get(identity["endpoint"]["name"]),
        native_proper_loss=value.get("loss"), selected_epoch=result.get("selected_epoch"),
        trained_epochs=result.get("completed_epochs"), stop_reason=result.get("stop_reason"),
        native_proper_loss_definition=identity.get("native_proper_loss_definition"),
        auxiliary_support_and_zero_support_counts=result.get("auxiliary_support_and_zero_support_counts"),
        member_and_pooling_diagnostics=result.get("member_and_pooling_diagnostics"),
        convergence_status="declared_matched_patience_satisfied" if expected_track == "matched_extended" and result.get("stop_reason") == "patience"
            else "source_horizon_separate" if expected_track == "source_reference" else "unresolved",
        selected_checkpoint=result.get("selected_checkpoint"), selected_export=result.get("selected_VALID_export"),
        own_selection=(value.get("role") == "VALID" and value.get("complete_source_VALID") is True
                       and value.get("TEST_accessed") is False),
        descriptor_hash=digest(descriptor), runtime_binding=copy.deepcopy(result.get("runtime_binding")),
        costs={"physical_fit_count": 1, "optimizer_calls": result.get("optimizer_calls"),
               "latest_invocation_wall_seconds": result.get("invocation_wall_seconds"),
               "resources": result.get("resources"), "elapsed_wall_seconds": None,
               "full_attempt_history_cost": None, "preprocessing_and_cache_cost": None,
               "selected_inference_and_export_cost": None, "engineering_resume_failure_cost": None,
               "storage": None, "historical_reuse_cost": None},
        raw_result=copy.deepcopy(result))
    row["scope"] = "expanded_exploratory" if policy_track in ("matched_extended", "source_reference") \
        and descriptor.get("engineering_only") is not True else "noncampaign_pilot_or_engineering"
    return row


def extract_i4(result, identity, components, binding_evidence=None):
    """Consume the existing JSON aggregate RESULT and four normalized bodies."""
    require(result.get("schema") == "independent-own-selected-VALID-bank-result-v1", "Unsupported I4 RESULT schema")
    require(result.get("cohort_id") == identity["cohort_id"] and result.get("TEST_accessed") is False,
            "I4 cohort/role mismatch")
    require(result.get("extra_training_fits") == 0 and result.get("aggregation") == "mean_raw_logits_after_four_own_selectors",
            "I4 aggregation or extra-fit identity changed")
    require(len(components) == 4 and len(result.get("members", [])) == 4, "Exactly four I4 bodies required")
    members = sorted(result["members"], key=lambda r: r["body"])
    require([m["body"] for m in members] == [0, 1, 2, 3], "I4 body identities incomplete/duplicate")
    by_fit = {r["fit_id"]: r for r in components}
    require(len(by_fit) == 4, "I4 bodies must be distinct physical fits")
    resolved = []
    for body, member in enumerate(members):
        require(member["job_id"] in by_fit, "I4 references an unknown body")
        row = by_fit[member["job_id"]]
        require(member["seed"] == identity["cohort_seed"] + OFFSETS[body] == row["actual_seed"], "I4 seed offset mismatch")
        for key in ("preset_id", "dataset", "cohort_id", "cohort_seed", "split_index", "literal_split_seed", "track"):
            require(row[key] == identity[key], "I4 component pairing mismatch: " + key)
        require(row["runtime_procedure_id"] == "native_single" and row["readout"] == "actual", "I4 body is not a native actual predictor")
        require(row.get("descriptor_hash") == member["job_hash"], "I4 component job hash mismatch")
        require(row.get("selected_epoch") == member["selected_epoch"] and row.get("selected_checkpoint") == member["selected_checkpoint"],
                "I4 component selected state mismatch")
        resolved.append(copy.deepcopy(row))
    require(result["M1_exact_member0_job_hash"] == members[0]["job_hash"], "I4 M1/body0 hash mismatch")
    row = copy.deepcopy(identity)
    row["declared_bindings"] = row.pop("bindings", {})
    row["bindings"] = {}
    for field in ("native_source", "data", "controller", "evaluator"):
        values = [part.get("bindings", {}).get(field) for part in resolved]
        if all(v is not None and v == values[0] for v in values):
            row["bindings"][field] = values[0]
    if binding_evidence is not None:
        require(binding_evidence.get("aggregate_bank_sha256") == (result.get("bank_receipt") or {}).get("sha256")
                and authority(binding_evidence.get("producer_receipt")), "I4 binding evidence does not anchor the actual aggregate export")
        row["bindings"] = copy.deepcopy(binding_evidence["bindings"])
        row["binding_evidence"] = copy.deepcopy(binding_evidence)
    row.update(schema="exploratory-predictor-row-v1", status=result["status"],
        selected_served_metric=(result.get("metrics") or {}).get(identity["endpoint"]["name"]),
        native_proper_loss=result.get("loss"), selected_epoch=None, trained_epochs=None,
        stop_reason="four_own_selected_components", own_selection=True, selected_checkpoint=None,
        selected_export=result.get("bank_receipt"), components=resolved,
        aggregation=result.get("aggregation"), extra_training_fits=result.get("extra_training_fits"),
        M1_exact_member0_fit_id=members[0]["job_id"],
        costs={"physical_fit_count": 0, "aggregation_export_cost": None}, raw_result=copy.deepcopy(result))
    row["scope"] = "expanded_exploratory" if all(p.get("scope") == "expanded_exploratory" for p in resolved) else "noncampaign_pilot_or_engineering"
    return row


def unresolved(row):
    reasons = []
    if row.get("scope") != "expanded_exploratory":
        reasons.append("outside_expanded_campaign_scope")
    if row.get("status") != "complete":
        reasons.append("status_" + str(row.get("status")))
    if row.get("own_selection") is not True or not authority(row.get("selected_export")):
        reasons.append("own_selected_export_unresolved")
    if row["scientific_method_id"] == "native_I4":
        if len(row.get("components", [])) != 4:
            reasons.append("four_I4_components_unresolved")
        for part in row.get("components", []):
            reasons.extend("I4_body:" + part["fit_id"] + ":" + reason for reason in unresolved(part))
    elif not authority(row.get("selected_checkpoint")) or type(row.get("selected_epoch")) is not int \
            or row["selected_epoch"] < 1 or row.get("trained_epochs", -1) < row["selected_epoch"]:
        reasons.append("own_selected_checkpoint_epoch_unresolved")
    if row["track"] == "matched_extended" and row["scientific_method_id"] != "native_I4" \
            and row.get("stop_reason") != "patience":
        reasons.append("matched_controller_or_convergence_unresolved:" + str(row.get("stop_reason")))
    for field in HASH_FIELDS:
        value = row.get("bindings", {}).get(field)
        if not isinstance(value, str) or len(value) != 64:
            reasons.append("binding_unresolved:" + field)
    return sorted(set(reasons))


def describe(values):
    return {"mean": statistics.mean(values) if values else None,
            "sample_SD_ddof1": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values) if values else None}


def compile_contrast_plan(facts):
    """Resolve the two documented task aliases; no score-dependent control choice."""
    output, cohort_keys = {}, {}
    for preset in facts["presets"]:
        preset_id, family = preset["id"], preset["task_family"]
        methods = {m["id"] for m in facts["methods"] if preset_id in m["presets"]}
        rows = []
        for method in facts["methods"]:
            if method["id"] not in methods:
                continue
            for control in method["controls"]:
                readout = "actual"
                if control == "same_selected_bank_fixed_mean":
                    target, readout = method["id"], "same_selected_bank_mean"
                elif control == "task_geometry_alignment_variance_or_link_endpoint":
                    target = "link_same_query_endpoint" if family == "link" else "task_geometry_alignment_variance"
                elif control == "structural_class_quartile_private_or_link_endpoint_v2":
                    target = "link_endpoint_responsibility_v2" if family == "link" else "structural_class_quartile_private_v1"
                else:
                    target = control
                require(target in methods or target in ("native_M1", "native_I4"), "Applicable method/control alias unresolved: " + target)
                rows.append({"id": method["id"] + "__vs__" + target + "__" + readout,
                    "lhs": {"method_id": method["id"], "readout": "actual"},
                    "rhs": {"method_id": target, "readout": readout}})
        rows.append({"id": "native_M1__vs__native_I4__actual", "lhs": {"method_id": "native_M1", "readout": "actual"},
                     "rhs": {"method_id": "native_I4", "readout": "actual"}})
        output[preset_id] = rows
        cohort_keys[preset_id] = [{"seed": c["seed"], "split_index": c["split"],
            "literal_split_seed": preset["source_literal_split_seeds"][str(c["split"]) ]}
            for c in preset["matched_cohorts"]]
    return {"schema": "expanded-prescribed-contrast-plan-v1", "facts_content_hash": digest(facts), "presets": output,
            "cohort_keys": cohort_keys, "outcomes_used": False, "same_bank_mean_is_diagnostic": True}


def report(registry, observations, cost_ledger=None, contrast_plan=None):
    require(registry.get("schema") == "expanded-ten-cohort-report-registry-v1", "Explicit frozen campaign registry required")
    require(contrast_plan is not None and contrast_plan.get("schema") == "expanded-prescribed-contrast-plan-v1",
            "Pinned prescribed contrast plan required")
    require({p["id"] for p in registry["presets"]} == set(contrast_plan["presets"]), "Frozen preset inventory omitted/replaced")
    for preset in registry["presets"]:
        require(preset["contrasts"] == contrast_plan["presets"][preset["id"]], "Prescribed method/control contrasts omitted/replaced")
        keys = [{k: c[k] for k in ("seed", "split_index", "literal_split_seed")} for c in preset["matched_cohorts"]]
        require(sorted(keys, key=lambda c: (c["seed"], c["split_index"])) ==
                sorted(contrast_plan["cohort_keys"][preset["id"]], key=lambda c: (c["seed"], c["split_index"])),
                "Frozen cohort seed/split registry omitted/replaced")
    expected = {row_key(r): r for r in registry["expected_predictors"]}
    require(len(expected) == len(registry["expected_predictors"]), "Duplicate expected predictor identity")
    indexed = {}
    for row in observations:
        key = row_key(row)
        require(key not in indexed, "Duplicate observation; explicit attempt disposition required")
        require(key in expected, "Unexpected predictor identity")
        declared = expected[key]
        for field in IDENTITY_FIELDS + ("endpoint",):
            require(row.get(field) == declared.get(field), "Frozen pairing/identity mismatch: " + field)
        for field in HASH_FIELDS:
            value = row.get("bindings", {}).get(field)
            if value is not None:
                require(value == declared["bindings"].get(field), "Frozen authority mismatch: " + field)
        metric = row.get("selected_served_metric")
        require(metric is None or numeric(metric), "Nonfinite/non-numeric endpoint must be explicitly missing")
        indexed[key] = copy.deepcopy(row)

    # Validate I4/body0 and diagnostic bank identity even if no contrast uses it.
    for key, row in indexed.items():
        if row["scientific_method_id"] == "native_I4":
            m1_key = (row["preset_id"], row["track"], row["cohort_id"], "native_M1", "actual")
            m1 = indexed.get(m1_key)
            require(m1 is not None, "I4 requires its exact M1/body0 observation")
            require(row.get("M1_exact_member0_fit_id") == m1["fit_id"], "I4 body0 is not exact matched M1")
            parts = row.get("components", [])
            require(len(parts) == 4 and parts[0]["fit_id"] == m1["fit_id"], "I4 component identity/order mismatch")
            require(len({p["fit_id"] for p in parts}) == 4
                    and [p["fit_id"] for p in parts] == expected[key]["component_fit_ids"], "I4 frozen component-fit identities mismatch")
            require(row.get("aggregation") == "mean_raw_logits_after_four_own_selectors"
                    and row.get("extra_training_fits") == 0, "I4 readout/training-fit identity mismatch")
            for body, part in enumerate(parts):
                require(part["actual_seed"] == row["cohort_seed"] + OFFSETS[body]
                        and part["runtime_procedure_id"] == "native_single", "I4 native seed-offset identity mismatch")
                for field in ("preset_id", "dataset", "cohort_id", "cohort_seed", "split_index", "literal_split_seed", "track"):
                    require(part[field] == row[field], "I4 component pairing mismatch: " + field)
            require(parts[0].get("descriptor_hash") == m1.get("descriptor_hash")
                    and parts[0].get("selected_checkpoint") == m1.get("selected_checkpoint"), "I4 M1 checkpoint/hash mismatch")
        if row["readout"] == "same_selected_bank_mean":
            actual = indexed.get((row["preset_id"], row["track"], row["cohort_id"], row["scientific_method_id"], "actual"))
            require(actual is not None, "Mean diagnostic requires its own actual predictor")
            for field in ("fit_id", "selected_epoch", "selected_checkpoint"):
                require(row.get(field) == actual.get(field), "Mean diagnostic reselected/changed bank: " + field)

    contrasts = []
    bootstrap_arrays = {}
    numpy_version = None
    for preset in registry["presets"]:
        cohorts = preset["matched_cohorts"]
        require(len(cohorts) == 10 and len({(c["seed"], c["split_index"], c["literal_split_seed"]) for c in cohorts}) == 10,
                "Exactly ten distinct frozen seed/split cohorts required")
        cohorts = sorted(cohorts, key=lambda c: (c["seed"], c["split_index"]))
        for contrast in preset["contrasts"]:
            rows, differences, wtl = [], [], [0, 0, 0]
            unresolved_count = missing_count = 0
            for cohort in cohorts:
                pair = []
                for side in ("lhs", "rhs"):
                    predictor = contrast[side]
                    key = (preset["id"], "matched_extended", cohort["cohort_id"], predictor["method_id"], predictor["readout"])
                    require(key in expected, "Frozen contrast has no expected predictor: " + str(key))
                    decl = expected[key]
                    require((decl["cohort_seed"], decl["split_index"], decl["literal_split_seed"]) ==
                            (cohort["seed"], cohort["split_index"], cohort["literal_split_seed"]), "Registry cohort pairing mismatch")
                    pair.append(indexed.get(key))
                lhs, rhs = pair
                reason = (unresolved(lhs) if lhs else ["lhs_missing"]) + (unresolved(rhs) if rhs else ["rhs_missing"])
                if lhs and rhs:
                    require(lhs["endpoint"] == rhs["endpoint"], "Contrasted endpoint name/units/direction mismatch")
                    for field in ("native_source", "data", "controller", "evaluator"):
                        a, b = lhs.get("bindings", {}).get(field), rhs.get("bindings", {}).get(field)
                        require(a is None or b is None or a == b, "Method/control pairing authority mismatch: " + field)
                scored = lhs is not None and rhs is not None and numeric(lhs.get("selected_served_metric")) and numeric(rhs.get("selected_served_metric"))
                difference = lhs["selected_served_metric"] - rhs["selected_served_metric"] if scored else None
                if scored:
                    differences.append(difference)
                    direction = lhs["endpoint"]["direction"]
                    require(direction in ("max", "min"), "Explicit native metric direction required")
                    signed = difference if direction == "max" else -difference
                    wtl[0 if signed > 0 else 1 if signed == 0 else 2] += 1
                    unresolved_count += bool(reason)
                else:
                    missing_count += 1
                rows.append({"cohort": copy.deepcopy(cohort), "lhs_fit_id": None if lhs is None else lhs["fit_id"],
                    "rhs_fit_id": None if rhs is None else rhs["fit_id"], "lhs_predictor": contrast["lhs"], "rhs_predictor": contrast["rhs"],
                    "control_fit_ids": [] if rhs is None else [rhs["fit_id"]],
                    "I4_control_component_fit_ids": [p["fit_id"] for p in (rhs or {}).get("components", [])],
                    "lhs_score": None if lhs is None else lhs.get("selected_served_metric"),
                    "rhs_score": None if rhs is None else rhs.get("selected_served_metric"),
                    "signed_lhs_minus_rhs": difference, "scored": scored, "resolved": scored and not reason,
                    "unresolved_reasons": reason, "lhs_status": None if lhs is None else lhs.get("status"),
                    "rhs_status": None if rhs is None else rhs.get("status")})
            complete = len(differences) == 10 and unresolved_count == 0
            interval = None
            if complete:
                import numpy as np
                numpy_version = np.__version__
                if preset["id"] not in bootstrap_arrays:
                    bootstrap_arrays[preset["id"]] = np.random.Generator(np.random.PCG64(130102026)).integers(0, 10, size=(10000, 10))
                samples = np.asarray(differences, dtype=np.float64)[bootstrap_arrays[preset["id"]]].mean(axis=1)
                interval = np.quantile(samples, [0.025, 0.975], method="linear").tolist()
            contrasts.append({"preset_id": preset["id"], "contrast_id": contrast["id"],
                "endpoint": expected[(preset["id"], "matched_extended", cohorts[0]["cohort_id"],
                    contrast["lhs"]["method_id"], contrast["lhs"]["readout"])]["endpoint"],
                "lhs": contrast["lhs"], "rhs": contrast["rhs"], "n_planned": 10, "n_score_pairs": len(differences),
                "n_missing_pairs": missing_count, "n_resolved_pairs": len(differences) - unresolved_count,
                "n_unresolved_scored_pairs": unresolved_count, "wins_ties_losses_missing": wtl + [missing_count],
                "complete_preset_summary": complete, "available_pair_summary_incomplete": not complete,
                **describe(differences), "descriptive_marginal_95pct_paired_percentile_interval": interval,
                "interval_withheld_reason": None if complete else "All ten scored and resolved own-selected pairs required; no survivor interval.",
                "rows": rows})
    ledger = copy.deepcopy(cost_ledger or [])
    require(len({r["attempt_id"] for r in ledger}) == len(ledger), "Duplicate cost-ledger attempt ID")
    known_wall = [r["worker_wall_seconds"] for r in ledger if numeric(r.get("worker_wall_seconds"))]
    return {"schema": "expanded-exploratory-metadata-report-v1", "registry_hash": digest(registry),
        "prescribed_contrast_plan_hash": digest(contrast_plan),
        "raw_rows": copy.deepcopy(observations), "source_reference_rows": [r for r in observations if r["track"] == "source_reference"],
        "all_expected_predictor_statuses": [{"identity": copy.deepcopy(r), "observed": key in indexed,
            "unresolved": unresolved(indexed[key]) if key in indexed else ["missing"]} for key, r in expected.items()],
        "contrasts": contrasts, "cost_ledger": ledger,
        "cost_summary": {"known_attempt_worker_wall_seconds_sum": sum(known_wall), "attempts_with_known_wall": len(known_wall),
            "attempts_with_unknown_wall": len(ledger) - len(known_wall), "elapsed_wall_seconds": None,
            "row_costs_are_preserved_not_summed_as_lifetime_cost": True},
        "bootstrap": {"generator": "NumPy Generator(PCG64)", "seed": 130102026, "resamples": 10000,
            "quantiles": [0.025, 0.975], "quantile_method": "linear", "numpy_version": numpy_version,
            "array_reused_within_preset": True, "coverage": "descriptive approximate marginal95%; not simultaneous/confirmatory"},
        "significance_or_superiority_claims": False, "winner_selection": False, "imputation": False,
        "TEST_access": False, "full_campaign_GO": False}


def extract_compact_molecular(value):
    """Audit old completed three-seed metadata without campaign relabeling."""
    require(len(value["fits"]) == 24, "Expected preserved MolBACE24 compact report")
    seeds = sorted({r["seed"] for r in value["fits"]})
    require(seeds == [101, 202, 303], "Original pilot cohort identities changed")
    rows = []
    for aggregate in value["aggregate"]:
        require(len(aggregate["AUC_percent"]["values"]) == len(aggregate["BCE"]["values"]) == 3, "Pilot aggregation shape changed")
        for i, seed in enumerate(seeds):
            method = aggregate["method"]
            fits = [r for r in value["fits"] if r["seed"] == seed and
                    ((method in ("M1", "I4") and r["procedure"] == "native_single") or r["procedure"] == method)]
            if method == "M1":
                fits = [r for r in fits if r["body"] == 0]
            require(len(fits) == (4 if method == "I4" else 1), "Pilot fit/aggregate identity mismatch")
            if method != "I4":
                require(fits[0]["selected_AUC_percent"] == aggregate["AUC_percent"]["values"][i], "Raw selected pilot score differs")
            rows.append({"scope": "existing_three_seed_validation_only_pilot_not_ten_cohort_benchmark",
                "seed": seed, "method": method, "fit_ids": [r["job_id"] for r in fits],
                "selected_epochs": [r["selected_epoch"] for r in fits], "trained_epochs": [r["trained_epochs"] for r in fits],
                "AUC_percent": aggregate["AUC_percent"]["values"][i], "BCE": aggregate["BCE"]["values"][i],
                "selected_checkpoint_receipts_available_in_this_compact_view": False})
    return {"schema": "MolBACE24-compact-extraction-audit-v1", "cohorts": seeds, "n_cohorts": 3,
        "eligible_for_ten_cohort_bootstrap": False, "rows": rows, "raw_fit_rows": copy.deepcopy(value["fits"]),
        "costs": copy.deepcopy(value["costs"]), "pilot_elapsed_seconds": value["pilot_elapsed_seconds"],
        "old_intervals_not_recomputed_or_promoted": True, "past_scores_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["report", "compile-contrasts", "extract-runtime", "extract-I4", "extract-compact-molecular"])
    parser.add_argument("--registry"); parser.add_argument("--observations"); parser.add_argument("--cost-ledger")
    parser.add_argument("--descriptor"); parser.add_argument("--result"); parser.add_argument("--identity"); parser.add_argument("--components")
    parser.add_argument("--binding-evidence")
    parser.add_argument("--contrast-plan")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reads = []
    def load(path):
        require(path is not None, "Required command argument missing")
        require(Path(path).suffix == ".json", "JSON metadata only")
        reads.append(pin(path)); return json.loads(Path(path).read_text())
    if args.command == "report":
        value = report(load(args.registry), load(args.observations), load(args.cost_ledger) if args.cost_ledger else None,
                       load(args.contrast_plan))
    elif args.command == "compile-contrasts":
        value = compile_contrast_plan(load(args.result))
    elif args.command == "extract-runtime":
        value = extract_runtime(load(args.descriptor), load(args.result), load(args.identity),
                                load(args.binding_evidence) if args.binding_evidence else None)
    elif args.command == "extract-I4":
        value = extract_i4(load(args.result), load(args.identity), load(args.components),
                           load(args.binding_evidence) if args.binding_evidence else None)
    else:
        value = extract_compact_molecular(load(args.result))
    value["metadata_inputs"] = reads
    destination = Path(args.output)
    require(not destination.exists(), "Fresh output only; preserve earlier reports")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"schema": value["schema"], "output": pin(destination)}))


if __name__ == "__main__":
    main()
