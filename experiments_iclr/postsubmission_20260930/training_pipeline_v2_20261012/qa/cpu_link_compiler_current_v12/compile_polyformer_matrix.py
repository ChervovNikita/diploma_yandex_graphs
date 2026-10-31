"""Disabled pinned PolyFormer mono matrix; no numerical data/model imports.

Scoped author-native points anchor adapted standardized four-role studies.
Exact author reproduction is separately blocked, with zero author fits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters.polyformer import (AUTHOR_COMMIT, AUTHOR_REPOSITORY, METHODS,
                                SOURCE_FILES, _arguments)
from adapters.polyformer_endpoints import independent_contracts
from compile_native_matrix import bind_custody, file_binding, product_union
from compile_molecular_matrix import make_plan
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate

SEARCH_SEEDS = (70101, 70103)
COMPARATIVE_SEEDS = (70201, 70203, 70207)
DEFAULT_ENGINE = "training_pipeline_v2_20261012"
GEOMETRIES = (
    ("native", "polyformer_single", 1, "unit"),
    ("equipped", "polyformer_factorized_single", 1, "unit"),
    ("equipped_first", "polyformer_factorized_single", 1, "first_rademacher"),
    ("shared4", "polyformer_internal_be", 4, "unit"),
    ("shared4_first", "polyformer_internal_be", 4, "first_rademacher"),
)
LOSSES = (
    ("own", "own", {}, 1),
    ("supcon", "within_route_supcon", {"supcon_weight": .05, "temperature": .2}, 1),
    ("paired_own", "two_view_class_full_cross_view", {"alignment_weight": 0., "temperature": .2}, 2),
    ("paired_class_full", "two_view_class_full_cross_view", {"alignment_weight": .05, "temperature": .2}, 2),
    ("paired_supcon_eq2", "two_view_supcon_eq2", {"supcon_weight": .05, "temperature": .2}, 2),
)
SHARED_LOSSES = (
    ("gncl", "gncl", {"pool_weight": .5}, 1),
    ("gncl_supcon", "gncl_within_route_supcon", {"pool_weight": .5, "supcon_weight": .05, "temperature": .2}, 1),
    ("paired_alignment_residual", "two_view_internal_be", {"alignment_weight": .05, "residual_weight": .05, "temperature": .2}, 2),
)


def templates():
    rows = []
    for geometry, mode, members, initialization in GEOMETRIES:
        losses = list(LOSSES) + (list(SHARED_LOSSES) if members == 4 else [("own_mean4_dropout", "own", {}, 4)])
        for label, variant, coefficients, views in losses:
            rows.append({"arm": geometry+"_"+label, "method": "polyformer_"+geometry+"_"+label,
                         "polyformer_mode": mode, "members": members, "factor_initialization": initialization,
                         "loss": {"variant": variant, "coefficients": coefficients}, "train_views": views,
                         "matched_paired_own_control": label == "paired_own"})
    return rows


def coefficient_domain(template, policy):
    native = template["loss"]["coefficients"]
    if policy not in {"native_fixed", "neighbors"}:
        raise ValueError("Declare a finite coefficient policy")
    if not native or policy == "native_fixed" or template["matched_paired_own_control"]:
        return [copy.deepcopy(native)]
    fields = {key: ([.25, .5, .75] if key == "pool_weight" else [.1, .2, .4] if key == "temperature"
                    else [.025, .05, .1]) for key in native}
    return product_union([fields])


def parameter_domain(recipe, template, policy):
    if recipe["operator"]["base"] != "mono" or recipe["configuration_count"] != 80:
        raise ValueError("The pinned adapter admits only the scoped 80-point mono domain")
    points = product_union(recipe["numeric_search_blocks"])
    if len(points) != 80 or recipe["native_point"] not in points:
        raise ValueError("Exactly80 effective base configurations including the native anchor required")
    result = []
    for point in points:
        if set(point) != {"learning_rate_groups", "weight_decay_groups", "width", "depth", "dropout"}:
            raise ValueError("Unsupported or ineffective PolyFormer numeric axis")
        native = {"K": recipe["operator"]["polynomial_order_K"], "q": recipe["operator"]["q"],
            "multi": recipe["operator"]["multi"], "hidden": point["width"], "nlayer": point["depth"],
            "dropout": point["dropout"], "n_head": recipe["locks"]["heads"],
            "d_ffn": recipe["locks"]["d_ffn"], "dprate": recipe["locks"]["dprate"]}
        _arguments(native, 1, 3, "schema-only")
        for coefficients in coefficient_domain(template, policy):
            result.append({"native": native, "learning_rate": point["learning_rate_groups"]["other"],
                "attn_lr": point["learning_rate_groups"]["attnmodule"],
                "weight_decay": point["weight_decay_groups"]["other"],
                "attn_weight_decay": point["weight_decay_groups"]["attnmodule"],
                "loss": {"variant": template["loss"]["variant"], "coefficients": coefficients}})
    return result


def resource_for(profiles, dataset):
    value = profiles.get(dataset+"/PolyFormer", profiles.get("*"))
    if not isinstance(value, dict):
        raise ValueError("Explicit provisional/qualified resource profile required")
    for key in ("reservation_mib", "wall_seconds"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError("Positive resource declaration required: "+key)
    if type(value.get("estimated_seconds_per_update")) not in (int, float) or not math.isfinite(value["estimated_seconds_per_update"]) or value["estimated_seconds_per_update"] <= 0 or not value.get("evidence"):
        raise ValueError("Finite positive scheduling estimate and its source/assumption evidence required")
    return value


def author_status(recipe):
    return {"recipe_id": recipe["recipe_id"], "repository": AUTHOR_REPOSITORY, "commit": AUTHOR_COMMIT,
            "status": "BLOCKED_EXACT_AUTHOR_REPRODUCTION", "author_model_fit_count": 0,
            "source_native_point": recipe["native_point"], "maximum_horizon": 2000, "native_bad_validation_patience": 250,
            "source_split": "60/20/20 class-balanced TRAIN/random remainder for homophilic source; ten fixed filtered masks for squirrel",
            "source_runs": 10, "source_selector": "strict_first_val_accuracy_maximum_above_initial_zero",
            "blockers": ["four_role_provider_splits_and_graph_preprocessing_differ_from_author",
                "standardized_full2000_horizon_does_not_implement_native250_bad_validation_stopping",
                "AST_definition_loader_omits_author_module_global_rng_and_transitive_imports",
                "author_training_reads_TEST_each_epoch; exact_deferred_TEST_protocol_not_implemented"],
            "standardized_native_anchor_is_not_exact_author_replication": True}


def make_study(root, custody, data_dir, recipe, sources, profile, policy="neighbors", engine_prefix=DEFAULT_ENGINE):
    dataset = custody["dataset_id"]
    if recipe["dataset_scope"] != [dataset] or recipe["operator"]["base"] != "mono":
        raise ValueError("A dataset-specific supported mono recipe is required")
    horizon = recipe["training"]["active_maximum"]
    if horizon != 2000 or recipe["training"]["policy"] != "full_budget":
        raise ValueError("This packet declares the complete source maximum2000 horizon")
    splits = {r["split_id"]: r for r in custody["splits"]}
    if 0 not in splits or any(r["evaluate_purpose"] != "development" for r in splits.values()):
        raise ValueError("Search split0 and development-only prepared roles required")
    if any(r["role_counts"]["train"] > 512 for r in splits.values()):
        raise ValueError("Full-TRAIN contrastive ports above512 need a separately declared bounded panel/resource adapter")
    study_id = "polyformer-v2-"+dataset+"-mono"
    budget = {"unit": "optimizer_updates", "policy": "full_budget", "maximum": horizon}
    selection = {"metric": "accuracy", "direction": "max", "ties": "first", "checkpoint0": False,
                 "evaluation_interval": 1, "save_every": 1, "patience_steps": horizon}
    rosters, runs, banks, choices, domains = templates(), [], [], {}, {}
    data_dir, root = Path(data_dir), Path(root)
    def inputs(row, field):
        return {key: str((data_dir/value).resolve().relative_to(root.resolve())) for key,value in row[field].items()}
    def add(identifier, kind, method, split, seed, body, adapter, config, parents=()):
        row = {"run_id": identifier, "kind": kind, "method": method, "split": split, "seed": seed, "body": body,
               "adapter": adapter, "config": config, "parents": list(parents),
               "budget": copy.deepcopy(budget) if kind == "model_fit" else {},
               "selection": copy.deepcopy(selection) if kind == "model_fit" else {}}
        runs.append(row)
        return identifier
    def fit_config(template, row, seed, body, stage, parameters, group=None):
        slots = max(template["members"], template["train_views"])
        config = {**copy.deepcopy(parameters), "polyformer_mode": template["polyformer_mode"],
            "node_inputs": inputs(row, "node_inputs"), "classes": custody["model_graph"]["classes"],
            "preprocessing": "pinned_mono_base", "fusion": "mean_probabilities",
            "train_views": template["train_views"], "members": template["members"],
            "initialization_seed": seed+1000003*body,
            "dropout_seeds": [seed*17+body*10000019+slot*1009+300001 for slot in range(slots)],
            "factor_initialization": template["factor_initialization"], "stage": stage,
            "reservation_mib": profile["reservation_mib"], "maximum_cuda_bytes": profile["reservation_mib"]*1024**2,
            "estimated_seconds": profile["estimated_seconds_per_update"]*horizon*template["members"]*template["train_views"],
            "recipe_id": recipe["recipe_id"], "recipe_scope": "standardized_pinned_mono_four_roles_full_source_maximum",
            "exact_author_replication": False, "source_native_patience": 250}
        if template["factor_initialization"] == "first_rademacher":
            config["factor_seed"] = config["initialization_seed"]+900001
        if group:
            config["independence_group"] = group
        return config
    for template in rosters:
        arm = template["arm"]
        parameters = parameter_domain(recipe, template, policy)
        domains[arm] = {"configuration_count": len(parameters), "members_per_fit": template["members"],
                       "train_views": template["train_views"], "search_seed_ids": list(SEARCH_SEEDS)}
        candidates = []
        for parameter in parameters:
            config_id = fingerprint(parameter)[:20]
            ids = []
            for seed in SEARCH_SEEDS:
                identifier = "search-"+arm+"-"+config_id+"-s"+str(seed)
                ids.append(add(identifier, "model_fit", template["method"], 0, seed, 0,
                    engine_prefix+"/adapters/polyformer.py:make_adapter",
                    fit_config(template, splits[0], seed, 0, "search", parameter)))
            candidates.append({"config_id": config_id, "parameters": parameter, "run_ids": ids})
        choices[arm] = add("select-"+arm, "selection", template["method"], 0, 0, 0,
            engine_prefix+"/adapters/search.py:select_configuration",
            {"stage": "search_selection", "ranking_metric": "accuracy", "ranking_direction": "max",
             "configurations": candidates, "reservation_mib": 256, "estimated_seconds": 1.},
            [identifier for candidate in candidates for identifier in candidate["run_ids"]])
    for split, row in sorted(splits.items()):
        for seed in COMPARATIVE_SEEDS:
            for template in rosters:
                arm = template["arm"]
                group = "compare-sp"+str(split)+"-"+arm+"-s"+str(seed)
                fits = []
                anchor = parameter_domain(recipe, template, "native_fixed")[0]
                for body in range(4 if template["members"] == 1 else 1):
                    config = fit_config(template, row, seed, body, "comparative", anchor,
                                        group if template["members"] == 1 else None)
                    config["selected_config_from"] = choices[arm]
                    fits.append(add(group+"-b"+str(body), "model_fit", template["method"], split, seed, body,
                        engine_prefix+"/adapters/polyformer.py:make_adapter", config, [choices[arm]]))
                served = [(arm, fits[:1])]+([(arm+"_I4", fits)] if template["members"] == 1 else [])
                for bank_arm, parents in served:
                    bank_id = "bank-sp"+str(split)+"-"+bank_arm+"-s"+str(seed)
                    common = {"endpoint_inputs": inputs(row, "endpoint_inputs"), "fusion": "mean_probabilities"}
                    exported = add(bank_id, "ensemble_assembly", "polyformer_"+bank_arm, split, seed, 0,
                        engine_prefix+"/adapters/polyformer_endpoints.py:export_bank",
                        {**common, "stage": "bank_export", "fit_parents": parents,
                         "singleton_reference": "fixed_body0" if template["members"] == 1 and len(parents) == 1 else None,
                         "reservation_mib": profile["reservation_mib"],
                         "estimated_seconds": profile["estimated_seconds_per_update"]*max(len(parents),template["members"])}, parents)
                    cal = add(bank_id+"-cal", "calibration", "polyformer_"+bank_arm, split, seed, 0,
                        engine_prefix+"/adapters/polyformer_endpoints.py:scalar_calibration",
                        {**common, "stage": "calibration", "export_parent": exported, "placement": "after_serving_pool",
                         "steps": 500, "learning_rate": .01, "reservation_mib": 512, "estimated_seconds": 5.}, [exported])
                    evaluated = add(bank_id+"-evaluate", "evaluation", "polyformer_"+bank_arm, split, seed, 0,
                        engine_prefix+"/adapters/polyformer_endpoints.py:evaluate_bank",
                        {**common, "stage": "evaluation", "export_parent": exported, "calibration_parent": cal,
                         "reservation_mib": 512, "estimated_seconds": 5.}, [exported,cal])
                    banks.append({"split": split, "seed": seed, "arm": bank_arm, "fit_parents": parents,
                        "export": exported, "calibration": cal, "evaluation": evaluated,
                        "routes": max(len(parents),template["members"]), "body0_reused": len(parents) == 4,
                        "single_is_fixed_body0": template["members"] == 1 and len(parents) == 1})
    roles = {role: {str(split): inputs(row,"node_inputs" if role in {"train","select"} else "endpoint_inputs")[role if role in {"train","select"} else role+"_targets"]
             for split,row in sorted(splits.items())} for role in ("train","select","cal","evaluate")}
    spec = {"schema_version": 1, "study_id": study_id, "stage": "development", "dataset": dataset,
        "hypothesis": "Pinned PolyFormer mono shared/equipped/independent mechanisms under matched source geometry/Adam and prospectively fixed standardized roles and full source horizon.",
        "data": {"task": "node_classification", "roles": roles, "preprocessing": custody["request"]["graph_policy"],
            "prior_exposure": custody["request"]["evaluation_exposure"], "evidence": {
                "provider_files": custody["provider_files"], "model_graph": custody["model_graph"],
                "custody_request_hash": custody["request_hash"], "split_roster": sorted(splits), "search_split": 0,
                "mono_policy": "literal_pinned_mono_on_prepared_graph; no_additional_feature_transform",
                "unused_confirmation_claimed": False, "author_replication": False}}, "source": sources, "runs": runs,
        "comparisons": [{"references": ["native_own", "equipped_own", "native_own_I4", "equipped_own_I4", "native_own_mean4_dropout"],
            "same_equipment_controls": "unit/first_rademacher M1+I4 and shared own/paired-own under each initialization",
            "original_GNNM_reference": "separate_native_SAGE_packet_on_same_custody; not_attached_to_PolyFormer",
            "report": "all arm/split/seed outcomes, failures, costs and paired deltas; splits not independent datasets",
            "acceptance_guarantee": False}],
        "release": {"policy": "whole_roster", "primary_endpoint": "all_closed_raw_and_scalar_CAL_PolyFormer_bank_evaluations", "global_plan_gates_required": True},
        "resources": {"wall_seconds": profile["wall_seconds"], "profile": profile,
                      "status": "PREPARED_ESTIMATES_REQUIRE_RUNTIME_QUALIFICATION"}}
    validate(spec)
    fits = [r for r in runs if r["kind"] == "model_fit"]
    grouped = {}
    for fit in fits:
        group = fit["config"].get("independence_group")
        if group:
            grouped.setdefault(group,[]).append(fit)
    for cohort in grouped.values():
        independent_contracts(spec,cohort)
    starts = dict(Counter(r["config"]["stage"] for r in fits))
    counts = {"optimizer_starts_by_stage": starts, "initial_training_segments": len(fits),
        "continuation_segments": 0, "extension_segments": 0, "physical_body_instances_by_stage": starts,
        "coupled_optimizer_fits_by_stage": dict(Counter(r["config"]["stage"] for r in fits if r["config"]["members"] == 4)),
        "independent_comparative_body_fits": sum(r["config"]["stage"] == "comparative" and r["config"]["members"] == 1 for r in fits),
        "full_budget_optimizer_updates": len(fits)*horizon,
        "planned_TRAIN_route_forwards": sum(r["config"]["members"]*r["config"]["train_views"]*horizon for r in fits),
        "planned_validation_route_forwards": sum(r["config"]["members"]*horizon for r in fits),
        "search_selection_jobs": len(rosters), "comparative_served_banks": len(banks), "bank_exports": len(banks),
        "scalar_CAL_fits": len(banks), "evaluation_jobs": len(banks), "genuine_I4_banks": sum(len(b["fit_parents"]) == 4 for b in banks),
        "reused_I4_body0_artifacts": sum(b["body0_reused"] for b in banks), "comparative_served_prediction_views": sum(b["routes"] for b in banks),
        "source_stage_fits": 0, "author_replication_fits": 0, "all_run_records": len(runs), "comparative_split_count": len(splits)}
    return spec, counts, banks, domains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root","resources","host-bindings","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--custody",nargs="+",required=True)
    parser.add_argument("--scalar-coefficients",choices=["neighbors","native_fixed"],default="neighbors")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    verify_allocation()
    root = Path(args.project_root).resolve()
    package = Path(__file__).resolve().parent
    engine_prefix = str(package.relative_to(root))
    def local(value):
        path=Path(value)
        return confined(root,str(path.resolve().relative_to(root)) if path.is_absolute() else str(path))
    output=local(args.output)
    if output.exists():
        raise ValueError("Retain previous packet; use fresh output")
    matrix_path=package/"docs/MATRIX_AXES_DRAFT.json"
    matrix=json.loads(matrix_path.read_text())
    recipes=[r for r in matrix["baseline_recipe_catalogue"]["rows"] if r["recipe_id"].startswith("polyformer_mono_")]
    resource_path,host_path=local(args.resources),local(args.host_bindings)
    profiles=json.loads(resource_path.read_text())["profiles"]
    hosts=json.loads(host_path.read_text())
    programs=[p for p in package.rglob("*.py") if "__pycache__" not in p.parts]
    programs += [matrix_path,resource_path,host_path]
    for row in SOURCE_FILES.values():
        observed=file_binding(root,root/row["path"])
        if observed != row:
            raise ValueError("Pinned PolyFormer source changed")
        programs.append(root/row["path"])
    for recipe in recipes:
        for row in recipe["source_bindings"]:
            observed=file_binding(root,root/row["path"])
            if observed["sha256"] != row["sha256"]:
                raise ValueError("Scoped author recipe bytes changed")
            programs.append(root/row["path"])
    common=[file_binding(root,p) for p in dict.fromkeys(programs)]
    commit=args.source_commit or subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    prepared,blocked=[],[]
    seen=set()
    for path in args.custody:
        custody,data_dir,bindings=bind_custody(root,local(path))
        dataset=custody["dataset_id"]
        if dataset in seen:raise ValueError("One real custody per dataset")
        seen.add(dataset)
        matching=[r for r in recipes if r["dataset_scope"]==[dataset]]
        if len(matching)!=1:
            blocked.append({"dataset":dataset,"status":"BLOCKED_NO_SCOPED_MONO_RECIPE","fits":0})
            continue
        source={"git_commit":commit,"files":list({r["path"]:r for r in common+bindings}.values()),
            "author_repository":AUTHOR_REPOSITORY,"author_commit":AUTHOR_COMMIT,
            "working_tree_fingerprints_authoritative":True,"official_author_protocol_replication":False}
        prepared.append(make_study(root,custody,data_dir,matching[0],source,
            resource_for(profiles,dataset),args.scalar_coefficients,engine_prefix))
    relative=str(output.relative_to(root))
    plan=make_plan(prepared,relative,hosts)
    plan["immutable_source_bundle"]={"package":engine_prefix,"launcher_entrypoint":engine_prefix+"/launcher.py",
        "model_worker_entrypoint":engine_prefix+"/run_study.py","endpoint_worker_entrypoint":engine_prefix+"/run_endpoint.py",
        "requires_root_runtime_resource_qualification_before_GO":True}
    output.mkdir();(output/"studies").mkdir()
    summaries=[]
    for spec,counts,banks,domains in prepared:
        path=output/"studies"/(spec["study_id"]+".json.gz")
        path.write_bytes(gzip.compress((canonical(spec)+"\n").encode(),mtime=0))
        plan["studies"].append({"path":str(path.relative_to(root)),"spec_hash":fingerprint(spec)})
        summaries.append({"study_id":spec["study_id"],"counts":counts,"served_banks":banks,"search_domains":domains})
    (output/"PLAN.json").write_text(json.dumps(plan,indent=2,allow_nan=False)+"\n")
    summary={"schema":"polyformer-matrix-preparation-v1","plan_hash":fingerprint(plan),"prepared_studies":len(prepared),
        "per_study":summaries,"blocked_datasets":blocked,"coefficient_policy":args.scalar_coefficients,
        "author_reproduction_contracts":[author_status(r) for r in recipes],"author_replication_fits":0,
        "scientific_training_started":False,"GO_created":False,"resource_status":"PROVISIONAL_REQUIRES_ROOT_QUALIFICATION"}
    (output/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"per_study","author_reproduction_contracts"}},indent=2))


if __name__ == "__main__":
    main()
