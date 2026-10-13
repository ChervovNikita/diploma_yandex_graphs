"""Freeze the reference-backbone proposal; no learning imports or execution."""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ENGINE = HERE / "engineering" / "campaign_compiler.py"
SPEC = importlib.util.spec_from_file_location("reference_campaign_compiler", ENGINE)
COMPILER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPILER)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(name, rows):
    keys = sorted({key for row in rows for key in row})
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def numerical_config(settings):
    result = {}
    for key, value in settings.items():
        if isinstance(value, str):
            try:
                value = float(value) if any(character in value.lower() for character in ".e") else int(value)
            except ValueError:
                pass
        result[key] = value
    return result


def source_receipt(relative):
    path = (PHASE / relative).resolve()
    if not path.is_relative_to(PHASE) or not path.is_file():
        raise ValueError("Missing confined native source: " + relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha(path)}


def source_training(cell):
    config = numerical_config(cell["settings"])
    metric = "accuracy" if cell["reference_metric"] == "accuracy_percent" else (
        "rocauc" if cell["reference_metric"] == "rocauc_percent" else "hits@50")
    optimizer = {"type": "Adam", "betas": [0.9, 0.999], "epsilon": 1e-8,
                 "weight_decay": config.get("weight_decay", 0),
                 "group_parameters_and_native_regularizers": "unchanged pinned source"}
    if cell["architecture"] == "PolyFormer":
        optimizer["groups"] = [
            {"role": "ordinary", "lr": config["lr"], "weight_decay": config["weight_decay"]},
            {"role": "attention", "lr": config["attn_lr"], "weight_decay": config["attn_wd"]},
        ]
    elif cell["architecture"] in ("NCN", "NCNC"):
        optimizer["groups"] = [
            {"role": "encoder", "lr": config["encoder_lr"], "weight_decay": 0},
            {"role": "predictor", "lr": config["predictor_lr"], "weight_decay": 0},
        ]
    else:
        optimizer["lr"] = config["lr"]
    return {
        "unit": "epochs", "maximum": int(cell["epochs"]),
        "optimizer": optimizer,
        "selection": {"components": [{"metric": metric, "direction": "max"}],
                      "ties": "first", "checkpoint0": False,
                      "validation_interval_epochs": 1,
                      "native_callback_authority": cell["source_program"]},
        "native_schedule": {"type": "constant native optimizer groups", "warmup": int(cell["warmup"]),
                            "source_program": cell["source_program"]},
        "native_stopping": {"patience": cell["patience"], "source_callback_unchanged": True},
        "loss_masks_reductions_training_tail": "unchanged native source",
        "native_TEST_diagnostics": "retain source RNG semantics; quarantine any source TEST diagnostics; never select or amend from them",
    }


def wrappers(task, methods):
    initializer = {
        "native_weights_biases_and_buffers": "unchanged native constructor and source initialization",
        "input_boundary_R": "independent Rademacher rows from isolated deterministic factor RNG",
        "other_R_and_all_S": "one",
        "private_additive_bias_residual": "zero",
        "neutral_function_check": "R/S one and residual zero recover native forward; actual signed start is a declared wrapper change",
        "initialization_search": False,
    }
    result = [{
        "id": "gnnm_boundary_native", "factory": "runtime.native_boundary_adapter:make_original_boundary",
        "tracks": ["long_convergence"], "members": 4,
        "native_operator_preservation": True,
        "training_objective": "mean of four native task losses; native regularizers once under their original reduction",
        "delta": "boundary R/S and additive private bias residual only; retain all native W/b and operators",
        "serving": "mean native raw logits; preserve native output transform where required",
        "implementation_status": "native_preserving_boundary_adapter_not_yet_implemented",
        "fixed_wrapper_initializer": initializer,
    }]
    for method in methods:
        if method["id"] == "boundary_own_within_route_supcon" and task != "node_classification":
            continue
        if method["id"] == "boundary_private_fixed_responsibility" and task == "link_prediction":
            continue
        result.append({
            "id": method["id"], "factory": "runtime.native_boundary_adapter:" + method["id"],
            "tracks": ["long_convergence"], "members": 4,
            "native_operator_preservation": True,
            "training_objective": {key: method[key] for key in
                                   ("shared_gradient", "boundary_private_gradient", "fixed_coefficients")},
            "source_primitive_status": method["status"],
            "implementation_status": "boundary_only_source_port_not_yet_implemented",
            "fixed_wrapper_initializer": copy.deepcopy(initializer),
            "serving": "same mean-native-logit rule as base GNNM and ordinary I4",
        })
    return result


def build():
    source_path = HERE / "source_protocols" / "paper_setting_roster.json"
    source = json.loads(source_path.read_text())
    method_path = HERE / "methods" / "WRAPPER_METHODS.json"
    methods = json.loads(method_path.read_text())["methods"]
    adopted, deferred = [], []
    for row in source["cells"]:
        row = copy.deepcopy(row)
        if row["architecture"].startswith("Polynormer"):
            row["disposition"] = "deferred_paper_release_heads_and_horizon_conflict"
            row["reason"] = "Paper and released WikiCS head count/horizon differ; fully matched native preset must be resolved before admission. No inferred 512/8 per-head mapping."
            deferred.append(row)
        elif row["roster_role"] == "excluded_hidden_online_test_evidence":
            row["disposition"] = "excluded_until_official_hidden_evaluation_is_available"
            row["reason"] = "Local public-label holdout cannot be compared as official hidden TEST."
            deferred.append(row)
        else:
            row["disposition"] = "adopted_fixed_source_preset_pending_runtime_port"
            adopted.append(row)
    if len(adopted) != 27:
        raise ValueError("Closed proposal expects 8 PolyFormer + 16 OGB + 3 Collab presets")

    # Bind fresh exact snapshots, never overwrite historical author captures.
    source_programs = {
        "PolyFormer": "polyformer/node_classification/training.py",
        "NCN": "ncn/ncn__NeighborOverlap.py",
        "NCNC": "ncn/ncn__NeighborOverlap.py",
        "BUDDY": "buddy/buddy_src_runners_run.py",
    }
    for row in adopted:
        row["audited_original_source_program"] = row["source_program"]
        suffix = source_programs.get(row["architecture"], "ogb/main_pyg.py")
        row["source_program"] = str(HERE.relative_to(PHASE) / "native_sources" / suffix)
        assert sha(PHASE / row["audited_original_source_program"]) == sha(PHASE / row["source_program"])

    # Fresh primary-paper readings; these are historical reference numbers, not project outcomes.
    references = {
        "collab-ncn": (64.76, 0.87, "ncn_paper.html", "Table 2/Collab/NCN"),
        "collab-ncnc": (66.61, 0.71, "ncn_paper.html", "Table 2/Collab/NCNC"),
        "collab-buddy": (65.94, 0.58, "buddy_candidate_paper.html", "Table 2/Collab/BUDDY"),
    }
    for row in adopted:
        if row["cell_id"] in references:
            mean, standard_deviation, filename, location = references[row["cell_id"]]
            row.update(reference_mean=mean, reference_std=standard_deviation,
                       reference_metric="hits_at_50_percent", reference_score_null_reason="",
                       reference_dispersion_type="standard_deviation",
                       score_location="references/" + filename + " " + location)
            row["reference_source_sha256"] = sha(HERE / "references" / filename)

    recipes = []
    split_seeds = [1941488137, 4198936517, 983997847, 4023022221, 4019585660, 2108550661,
                   1648766618, 629014539, 3212139042, 2424918363]
    cost_factor = {
        "gnnm_boundary_native": 4, "boundary_global_half_pool": 4,
        "boundary_private_half_pool": 6, "boundary_private_fixed_responsibility": 6,
        "boundary_own_within_route_supcon": 5,
    }
    for cell in adopted:
        is_poly = cell["architecture"] == "PolyFormer"
        task = "node_classification" if is_poly else (
            "graph_classification" if cell["dataset"].startswith("ogbg-") else "link_prediction")
        base = source_training(cell)
        extended = copy.deepcopy(base)
        extended["maximum"] = 10000 if is_poly else 1000
        extended["native_stopping"] = {"replaced_by_declared_long_controller": True}
        extended["stopping"] = {
            "minimum_units": int(cell["epochs"]),
            "patience_validation_checks": 100 if is_poly else 50,
            "required_gates": ["native_warmup_and_phase_transitions_complete"],
            "counting": "start staleness after minimum/source phases; reset on strict validation improvement",
        }
        extended["schedule_adaptations"] = [
            "Same constant Adam groups and every native LR/WD; larger cap and declared gated patience only."
        ]
        split_ids = list(range(10)) if is_poly else [0]
        pairs = [{"split": index if is_poly else 0, "seed": index} for index in range(10)]
        bindings = {str(index): {"source_policy": cell["split_policy"],
                                "source_split_seed": split_seeds[index] if is_poly else None,
                                "whole_native_VALID": True, "CAL_created": False,
                                "materialization_status": "author_provider_port_pending"}
                    for index in split_ids}
        selected_wrappers = wrappers(task, methods)
        weights = {"native_replica/native_single": {"value": cell["epochs"],
                   "basis": "source epoch cap in unmeasured relative native units", "measured": False},
                   "long_convergence/native_single": {"value": extended["maximum"],
                   "basis": "long cap in unmeasured relative native units", "measured": False}}
        for wrapper in selected_wrappers:
            weights["long_convergence/" + wrapper["id"]] = {
                "value": extended["maximum"] * cost_factor[wrapper["id"]],
                "basis": "cap times declared four-route/gradient-work proxy; dataset and GPU speed unmeasured",
                "measured": False,
            }
        recipes.append({
            "id": cell["cell_id"], "dataset": cell["dataset"], "architecture": cell["architecture"],
            "task": task, "source_bindings": {"native_program": source_receipt(cell["source_program"]),
                                             "author_snapshot_manifest": source_receipt(str(HERE.relative_to(PHASE) / "native_sources" / "MANIFEST.json")),
                                             "snapshot_scope": "exact saved author-source files; native dependency/provider qualification pending",
                                             "literature_and_release_row": cell},
            "native_factory": "runtime.source_native_adapter:" + cell["architecture"].lower().replace("-", "_"),
            "native_fixed_config": numerical_config(cell["settings"]),
            "native_seed_offsets": [0, 100003, 200003, 300007],
            "independent_seed_policy": "new deterministic own-body initialization streams; no fitted state or parameter sharing",
            "cohorts_by_track": {"native_replica": pairs, "long_convergence": copy.deepcopy(pairs)},
            "official_split_bindings": bindings,
            "official_evaluator": {"source_metric": cell["reference_metric"], "source_program": cell["source_program"],
                                   "complete_source_queries_and_negative_pool": True,
                                   "source_TEST_topology": cell["split_policy"],
                                   "published_value_is_context_not_paired_measurement": True},
            "published_native_training": base,
            "tracks": {"native_replica": base, "long_convergence": extended},
            "wrappers": selected_wrappers,
            "source_seed_fidelity": {
                "status": "source_settings_and_horizon_pinned_with_declared_repetition_adaptation",
                "source_global_rng_known": is_poly,
                "source_seed_policy_known": is_poly or task == "link_prediction",
                "winner_dropout_known": not cell["dataset"].startswith("ogbg-"),
                "unresolved": [cell["seed_policy"], "New native source adapter and official input binding are not qualified yet."]
                              + (["Published OGB winning dropout unspecified; both paper-declared presets reported separately."]
                                 if cell["dataset"].startswith("ogbg-") else []),
            },
            "fit_relative_cost_weights": weights,
            "resources": {}, "endpoint_resources": {},
            "runtime_status": "native_provider_boundary_optimizer_and_checkpoint_ports_pending",
            "GO": False,
        })
    roster = {
        "schema": "reference-wrapper-roster-v2", "scientific_launch_enabled": False,
        "slots_per_gpu": 2, "recipes": recipes,
        "no_HPO": True, "no_cross_preset_winner_selection": True,
        "source_audit_sha256": sha(source_path), "method_definitions_sha256": sha(method_path),
        "source_track": "native M1 only at source horizon; seed/protocol limits expressly disclosed",
        "main_track": "native M1 + independent I4 + boundary GNNM and declared training-only variants",
        "all_native_parameters_and_source_operators_preserved": True,
        "additional_reference_arms": [], "deferred_source_cells": deferred,
        "historical_scores_or_checkpoints_reused": False,
        "current_host_authorization": "allocation only; 77 assignments are future preparation, not access authorization",
        "reporting_policy": {
            "unit": "ten declared paired trials per preset, never pooled across different presets",
            "primary_contrasts": "each declared GNNM procedure minus matched M1 and minus matched I4",
            "mean_difference_interval": "paired percentile bootstrap, 10000 resamples, 95% marginal interval, fixed analysis seed 130102026",
            "multiplicity": "Holm across all 226 primary paired contrasts; adjusted exploratory evidence, not an independent confirmation study",
            "test": "two-sided complete sign-flip mean-difference test, 2^10 patterns; sign-symmetry assumption stated",
            "descriptive_outputs": "per-trial native scores, signed differences, all preset tables, failures, epochs and costs",
            "scope": "OGB fixed-split seed uncertainty describes training randomness, not uncertainty over unseen graph/scaffold splits",
            "invalid_or_missing_trials": "no automatic refits or survivor-only inference; report incomplete preset and failure reasons",
            "primary_contrast_count": 226,
            "historical_paper_numbers_in_paired_tests": False,
        },
    }
    plan, summary = COMPILER.compile_campaign(roster)
    plan["reporting_policy"] = copy.deepcopy(roster["reporting_policy"])
    summary["plan_hash"] = COMPILER.digest(plan)
    summary.update({
        "datasets": sorted({cell["dataset"] for cell in adopted}),
        "dataset_count": len({cell["dataset"] for cell in adopted}),
        "architectures": sorted({cell["architecture"] for cell in adopted}),
        "architecture_count": len({cell["architecture"] for cell in adopted}),
        "fits_by_task_and_track": dict(Counter(recipe["task"] + "/" + track
            for recipe in recipes for track in COMPILER.TRACKS
            for _ in range(len(recipe["cohorts_by_track"][track]) *
                           (1 if track == "native_replica" else 4 + len(recipe["wrappers"]))))),
        "deferred_cells": len(deferred), "scientific_runtime_ready": False,
        "adapters_and_resources_pending": True,
    })
    if summary["fits"] != 2480 or summary["fits_by_track"] != {"native_replica": 270, "long_convergence": 2210}:
        raise ValueError("Unexpected closed proposal count")
    save("ROSTER.json", roster)
    save("PLAN.json", plan)
    # Deterministic compact publication; the allocation also keeps raw JSON.
    (HERE / "PLAN.json.gz").write_bytes(gzip.compress((HERE / "PLAN.json").read_bytes(),
                                                     compresslevel=9, mtime=0))
    save("SUMMARY.json", summary)
    save("ADOPTED_SOURCE_CELLS.json", {"cells": adopted, "deferred": deferred})
    write_csv("SOURCE_CELLS.csv", adopted)
    write_csv("DEFERRED_CELLS.csv", deferred)
    write_csv("FIT_ROSTER.csv", [{"fit_id": job["id"], "recipe": job["contract"]["recipe_id"],
               "track": job["contract"]["track"], "split": job["contract"]["split"],
               "seed": job["contract"]["cohort_seed"], "body": job["contract"]["body"],
               "method": job["contract"]["model_procedure"], "contract_hash": job["contract_hash"]}
              for job in plan["jobs"] if job["kind"] == "fit"])
    save("FREEZE.json", {"canonical_plan_sha256": COMPILER.digest(plan),
                          "raw_plan_sha256": sha(HERE / "PLAN.json"),
                          "compressed_plan_sha256": sha(HERE / "PLAN.json.gz"),
                          "roster_sha256": sha(HERE / "ROSTER.json"),
                          "summary_sha256": sha(HERE / "SUMMARY.json"),
                          "portable_runtime_static_receipt_sha256": sha(HERE / "runtime" / "STATIC_CHECKS.json"),
                          "preservation_contract_sha256": sha(HERE / "methods" / "PRESERVATION_CONTRACT.json"),
                          "source_port_requirements_sha256": sha(HERE / "runtime" / "PORT_REQUIREMENTS.json"),
                          "author_source_snapshot_manifest_sha256": sha(HERE / "native_sources" / "MANIFEST.json"),
                          "scientific_fits_executed": 0, "GO": False,
                          "readiness": "finite proposed design frozen; executable source adapters are pending"})
    print(json.dumps({key: value for key, value in summary.items() if key != "method_count_per_long_cohort"}, indent=2))


if __name__ == "__main__":
    build()
