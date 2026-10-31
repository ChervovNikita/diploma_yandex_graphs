"""Compile disabled MolHIV/MolBACE GINE packets from closed data custody.

Pure JSON/source compilation: no numerical provider, graph tensor, model, fit,
score, download or GO receipt. The CLI requires the authorized allocation.
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
from adapters.molecular import (ARMS, bound_file, validate_config,
                                validate_independent_roster, validate_role_authorities)
from compile_native_matrix import file_binding, product_union
from partition import assign
from prepare_molecular_data import DATASETS, validate_roles
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate

ENGINE = "training_pipeline_v2_20261012"
SOURCE = "portable_internal_be_public_interface_20261007_v2/core"
SEARCH_SEEDS = (70101, 70103)
COMPARATIVE_SEEDS = (70201, 70203, 70207)
EPOCH_DOMAINS = (100, 300)
BASE_ARMS = ("single", "single_contrastive", "factorized_single", "factorized_single_contrastive", "be_unit", "be_init",
             "be_unit_contrastive", "be_init_contrastive")
I4_SEARCH_MEMBER = {"independent4_body": "single",
                    "independent4_body_contrastive": "single_contrastive",
                    "factorized_independent4_body": "factorized_single",
                    "factorized_independent4_body_contrastive": "factorized_single_contrastive"}
SOURCE_PINS = {
    "models": (8758, "2be6d872e962d9fdd341883b0d1060170fe0a176f7a8d6829d988fc6f8e7f9c3"),
    "factors": (3286, "9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3"),
    "objectives": (4824, "9208e677ab5cfb286862216991453f8592a1d373120019a6e2ff14c38fad5bcf")}


def source_receipts(root):
    result = {key: file_binding(root, root / SOURCE / (key+".py")) for key in SOURCE_PINS}
    if any((row["bytes"], row["sha256"]) != SOURCE_PINS[key] for key, row in result.items()):
        raise ValueError("Molecular archived source differs from the packet's exact pins")
    return result


def bind_custody(root, path):
    """Verify custody JSON and file hashes without decoding any role tensor."""
    root = Path(root).resolve()
    path = confined(root, str(Path(path).resolve().relative_to(root)))
    observed = file_binding(root, path)
    value = json.loads(path.read_text())
    if (value.get("schema_version") != 1 or value.get("dataset") not in DATASETS
            or value.get("complete") is not True or value.get("task") != "graph_binary_classification"
            or value.get("target_policy") != "complete_binary"
            or any(value.get(k) is not False for k in ("model_constructed", "optimizer_created", "training_launched", "scoring_performed", "scientific_launch_allowed", "author_protocol_replication"))):
        raise ValueError("An actual closed, no-fit/no-score molecular MANIFEST.json is required")
    if set(value["roles"]) != {"train", "select", "cal", "evaluate"}:
        raise ValueError("Molecular custody must declare all four roles")
    if set(value.get("endpoint_inputs", {})) != {"cal_graphs", "evaluate_graphs", "cal_targets", "evaluate_targets"}:
        raise ValueError("Fresh graph-only/target-only endpoint custody is required")
    rows = list(value["roles"].values()) + list(value["endpoint_inputs"].values())
    rows += [value[k] for k in ("split_receipt", "provider_custody_receipt", "preparation_program")]
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Every molecular custody artifact must have its own authority")
    if value.get("cache_ancestry_receipt") is not None:
        rows.append(value["cache_ancestry_receipt"])
        ancestry = json.loads(bound_file(root, value["cache_ancestry_receipt"]).read_text())
        rows.extend(ancestry.get("retained_official_archives", []))
    files = [observed]
    for row in rows:
        files.append(file_binding(root, bound_file(root, row)))
    split = json.loads(bound_file(root, value["split_receipt"]).read_text())
    validate_roles(split["roles"], value["dataset_graphs"])
    if (split["dataset"] != value["dataset"] or split.get("author_protocol_replication") is not False
            or split.get("predictions_or_scores_used") is not False
            or split["roles"]["train"] != split["official_indices"]["train"]
            or split["roles"]["evaluate"] != split["official_indices"]["test"]
            or set(split["roles"]["select"] + split["roles"]["cal"]) != set(split["official_indices"]["valid"])):
        raise ValueError("Molecular scaffold/four-role protocol changed")
    provider = json.loads(bound_file(root, value["provider_custody_receipt"]).read_text())
    if provider["dataset_name"] != value["dataset"] or provider["split_type"] != "scaffold" or provider["num_tasks"] != 1:
        raise ValueError("Provider does not match the scalar scaffold molecular recipe")
    for row in provider["cache_files"]:
        files.append(file_binding(root, bound_file(root, row)))
    return value, list({row["path"]: row for row in files}.values())


def resource_for(profiles, dataset, epochs):
    value = profiles.get(dataset+"/GINE/"+str(epochs), profiles.get(dataset+"/GINE", profiles.get("*")))
    if not isinstance(value, dict):
        raise ValueError("Explicit molecular resource profile required")
    for key in ("reservation_mib", "wall_seconds"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError("Positive integer resource declaration required: " + key)
    for key in ("estimated_seconds_per_epoch", "estimated_export_seconds", "estimated_CAL_seconds", "estimated_evaluation_seconds"):
        if type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("Positive finite molecular scheduling estimate required: " + key)
    if not value.get("evidence"):
        raise ValueError("Resource estimates require their source/assumption evidence")
    return value


def validate_practical_policy(policy, recipe):
    """An explicit nine-point source-shape study, separate from the full grid."""
    expected = {'schema', 'datasets', 'epoch_domains', 'coefficient_policy', 'numeric_search_points'}
    if (not isinstance(policy, dict) or set(policy) != expected
            or policy['schema'] != 'molecular-practical-policy-v1'
            or policy['datasets'] != ['ogbg-molhiv', 'ogbg-molbace']
            or policy['epoch_domains'] != [300] or policy['coefficient_policy'] != 'native_fixed'):
        raise ValueError('Expected the complete explicit two-dataset 300-epoch molecular practical policy')
    points, anchor = policy['numeric_search_points'], recipe['native_point']
    if not isinstance(points, list) or len(points) != 9 or len({canonical(p) for p in points}) != 9:
        raise ValueError('Exactly nine distinct practical molecular configurations are required')
    for point in points:
        if (not isinstance(point, dict) or set(point) != set(anchor)
                or any(point[key] != anchor[key] for key in ('width', 'depth', 'weight_decay'))
                or type(point['learning_rate']) not in (int, float)
                or not math.isfinite(point['learning_rate']) or point['learning_rate'] <= 0
                or type(point['dropout']) not in (int, float)
                or not math.isfinite(point['dropout']) or not 0 <= point['dropout'] < 1):
            raise ValueError('Practical molecular points retain source shape/decay and finite LR/dropout')
    rates, drops = {p['learning_rate'] for p in points}, {p['dropout'] for p in points}
    if len(rates) != 3 or len(drops) != 3 or anchor not in points:
        raise ValueError('Practical domain is three learning rates by three dropouts including the source anchor')
    return copy.deepcopy(sorted(points, key=canonical))


def parameter_domain(recipe, arm, coefficient_policy, practical_policy=None):
    """Expand effective source controls; singleton residual has no HPO axis."""
    base = (product_union(recipe["numeric_search_blocks"]) if practical_policy is None else
            validate_practical_policy(practical_policy, recipe))
    if practical_policy is None and (len(base) != 40 or recipe["configuration_count"] != 40):
        raise ValueError("GINE requires its exact deduplicated 40-point base domain")
    if practical_policy is not None and coefficient_policy != practical_policy['coefficient_policy']:
        raise ValueError('Practical molecular preparation cannot introduce a coefficient grid')
    if coefficient_policy not in {"native_fixed", "neighbors"}:
        raise ValueError("Declare a finite coefficient policy")
    native = {"max_objects": 512, "alignment_weight": .05, "residual_weight": .05, "temperature": .2}
    coefficients = [native]
    if arm.endswith("_contrastive") and coefficient_policy == "neighbors":
        coefficients = product_union([{"max_objects": [512], "alignment_weight": [.025, .05, .1],
            "residual_weight": [.025, .05, .1] if arm.startswith("be_") else [.05], "temperature": [.1, .2, .4]}])
    values = []
    for point in base:
        if set(point) != {"width", "depth", "dropout", "learning_rate", "weight_decay"}:
            raise ValueError("Molecular base domain contains an unsupported or ineffective knob")
        for coefficient in coefficients:
            values.append({"model": {"hidden": point["width"], "layers": point["depth"], "dropout": point["dropout"], "outputs": 1},
                "optimizer": {"name": "Adam", "learning_rate": point["learning_rate"], "weight_decay": point["weight_decay"], "eps": 1e-8, "betas": [.9, .999]},
                "contrastive": copy.deepcopy(coefficient)})
    native_point = recipe["native_point"]
    if not any(v["model"] == {"hidden": native_point["width"], "layers": native_point["depth"], "dropout": native_point["dropout"], "outputs": 1}
               and v["optimizer"]["learning_rate"] == native_point["learning_rate"] and v["contrastive"] == native for v in values):
        raise ValueError("Molecular native anchor is absent from the search domain")
    return values


def make_study(root, custody, recipe, sources, molecular_source, profile, epochs,
               exposure, coefficient_policy="neighbors", engine_prefix=ENGINE, practical_policy=None):
    """Materialize a future development domain, with genuine I4 own selectors."""
    if epochs not in EPOCH_DOMAINS:
        raise ValueError("Only separately declared 100/300 full-epoch domains are supported")
    if exposure.get("evaluate_purpose") != "development" or not exposure.get("prior_exposure"):
        raise ValueError("Protected molecular confirmation requires root's separately frozen activation packet")
    if exposure.get("protected_confirmation") is not False:
        raise ValueError("Molecular development requires an explicit non-protected exposure decision")
    dataset = custody["dataset"]
    if practical_policy is not None:
        validate_practical_policy(practical_policy, recipe)
        if dataset not in practical_policy['datasets'] or epochs not in practical_policy['epoch_domains']:
            raise ValueError('Dataset/horizon is outside the explicit practical molecular policy')
    study_id = "molecular-v2-"+dataset+"-e"+str(epochs)
    if practical_policy is not None:
        study_id = 'practical-' + study_id
    budget = {"unit": "epochs", "policy": "full_budget", "maximum": epochs}
    selector = {"metric": "rocauc", "direction": "max", "ties": "first", "evaluation_interval": 1,
                "checkpoint0": False, "save_every": 1}
    runs, selections, banks, domains = [], {}, [], {}

    def add(identifier, kind, arm, seed, body, adapter, config, parents=()):
        row = {"run_id": identifier, "kind": kind, "method": "molecular_"+arm, "split": 0,
               "seed": seed, "body": body, "adapter": adapter, "config": config,
               "budget": copy.deepcopy(budget) if kind == "model_fit" else {},
               "selection": copy.deepcopy(selector) if kind == "model_fit" else {}, "parents": list(parents)}
        runs.append(row)
        return identifier

    def fit_config(arm, seed, body, stage, parameters, group=None):
        config = {"molecular_inputs": {r: custody["roles"][r] for r in ("train", "select")},
            "molecular_source": molecular_source, **copy.deepcopy(parameters), "arm": arm,
            "fusion": "mean_logits", "initialization_seed": seed*10+body,
            "dropout_seed": seed*10+body+300001, "shuffle_seed": seed*10+body+600001,
            "head_policy": "archived_scalar", "target_policy": custody["target_policy"],
            "training": {"batch_size": 128, "eval_batch_size": 128, "views_per_update": 2},
            "stage": stage, "reservation_mib": profile["reservation_mib"],
            "maximum_cuda_bytes": profile["reservation_mib"]*1024**2,
            "estimated_seconds": profile["estimated_seconds_per_epoch"]*epochs*(4 if arm.startswith("be_") else 1)}
        if group is not None:
            config["independence_group"] = group
        return config

    # Every I4 recipe inherits its objective-equivalent complete M1 search.
    # Singleton residual is identically zero; it adds no residual-weight axis.
    searchable = list(BASE_ARMS)
    for arm in searchable:
        parameters = parameter_domain(recipe, arm, coefficient_policy, practical_policy)
        domains[arm] = {"configuration_count": len(parameters), "search_seed_ids": list(SEARCH_SEEDS),
            "bodies_per_search_seed": 1,
            "ranking_law": "mean_own_selected_SELECT_ROCAUC; proper_loss_then_stable_config_id_ties"}
        candidates = []
        for parameter in parameters:
            config_id = fingerprint(parameter)[:20]
            identifiers = []
            for seed in SEARCH_SEEDS:
                group = "search-"+arm+"-"+config_id+"-s"+str(seed)
                for body in range(4 if "independent4_body" in arm else 1):
                    identifier = group+"-b"+str(body)
                    identifiers.append(add(identifier, "model_fit", arm, seed, body,
                        engine_prefix+"/adapters/molecular.py:make_adapter",
                        fit_config(arm, seed, body, "search", parameter, group if "independent4_body" in arm else None)))
            candidates.append({"config_id": config_id, "parameters": parameter, "run_ids": identifiers})
        select_id = "select-"+arm
        selections[arm] = add(select_id, "selection", arm, 0, 0,
            engine_prefix+"/adapters/search.py:select_configuration",
            {"stage": "search_selection", "ranking_metric": "rocauc", "ranking_direction": "max",
             "configurations": candidates, "reservation_mib": 256, "estimated_seconds": 1},
            [identifier for candidate in candidates for identifier in candidate["run_ids"]])
    anchor = parameter_domain(recipe, "single", "native_fixed", practical_policy)[0]
    for seed in COMPARATIVE_SEEDS:
        fitted = {}
        member_to_i4 = {member: i4 for i4, member in I4_SEARCH_MEMBER.items()}
        # A single-member reference is prospectively the typed I4 group's
        # fixed body0, with the same run/config/checkpoint identity. It is
        # never the best member selected from I4 after observing outcomes.
        for arm in BASE_ARMS:
            fit_arm = member_to_i4.get(arm, arm)
            group = "compare-"+fit_arm+"-s"+str(seed)
            fits = []
            for body in range(4 if fit_arm in I4_SEARCH_MEMBER else 1):
                config = fit_config(fit_arm, seed, body, "comparative", anchor, group if fit_arm in I4_SEARCH_MEMBER else None)
                config["selected_config_from"] = selections[arm]
                fits.append(add(group+"-b"+str(body), "model_fit", fit_arm, seed, body,
                    engine_prefix+"/adapters/molecular.py:make_adapter", config, [selections[arm]]))
            fitted[arm] = fits[:1]
            if fit_arm in I4_SEARCH_MEMBER:
                fitted[fit_arm] = fits
        for arm, fits in fitted.items():
            search_arm = I4_SEARCH_MEMBER.get(arm, arm)
            endpoint = {"endpoint_inputs": custody["endpoint_inputs"], "fusion": "mean_logits",
                        "target_policy": custody["target_policy"]}
            bank_id = "bank-"+arm+"-s"+str(seed)
            reference = {"single_reference_arm": arm} if arm in member_to_i4 else {}
            exported = add(bank_id, "ensemble_assembly", arm, seed, 0,
                engine_prefix+"/adapters/molecular_endpoints.py:export_bank",
                {**endpoint, **reference, "stage": "bank_export", "fit_parents": fits,
                 "reservation_mib": profile["reservation_mib"],
                 "estimated_seconds": profile["estimated_export_seconds"]*(4 if arm.startswith("be_") or arm in I4_SEARCH_MEMBER else 1)}, fits)
            cal_id = add(bank_id+"-cal", "calibration", arm, seed, 0,
                engine_prefix+"/adapters/molecular_endpoints.py:scalar_calibration",
                {**endpoint, "stage": "calibration", "export_parent": exported, "placement": "after_serving_pool",
                 "steps": 500, "learning_rate": .01, "reservation_mib": 512,
                 "estimated_seconds": profile["estimated_CAL_seconds"]}, [exported])
            evaluation = add(bank_id+"-evaluate", "evaluation", arm, seed, 0,
                engine_prefix+"/adapters/molecular_endpoints.py:evaluate_bank",
                {**endpoint, "stage": "evaluation", "export_parent": exported, "calibration_parent": cal_id,
                 "reservation_mib": 512, "estimated_seconds": profile["estimated_evaluation_seconds"]}, [exported, cal_id])
            banks.append({"arm": arm, "seed": seed, "split": 0, "fit_parents": fits, "export": exported,
                          "calibration": cal_id, "evaluation": evaluation,
                          "routes": 4 if arm.startswith("be_") or arm in I4_SEARCH_MEMBER else 1,
                          "independently_paid_fits": len(fits) if arm in I4_SEARCH_MEMBER else 0,
                          "own_selected_checkpoints": len(fits), "body0_reused": arm in I4_SEARCH_MEMBER,
                          "singleton_reference_is_fixed_body0": arm in member_to_i4,
                          "selected_configuration_source": selections[search_arm]})
    spec = {"schema_version": 1, "study_id": study_id, "stage": "development", "dataset": dataset,
        "hypothesis": "Prospective standardized raw-logit GINE/internal-BE and genuine independently selected molecular comparisons at a fixed complete epoch domain.",
        "data": {"task": custody["task"], "target_policy": custody["target_policy"],
            "preprocessing": "original_OGB_atom9_bond3; scaffold_TRAIN; fixed_hash_class_stratified_VALID_to_SELECT_CAL; official_TEST_evaluate",
            "roles": custody["roles"], "prior_exposure": exposure["prior_exposure"],
            "evidence": {"endpoint_inputs": custody["endpoint_inputs"], "preparation_manifest_hash": fingerprint(custody),
                "split_receipt": custody["split_receipt"], "provider_custody_receipt": custody["provider_custody_receipt"],
                "pack_descriptives": custody["pack_descriptives"], "exposure_decision": exposure,
                "author_protocol_replication": False}}, "source": sources, "runs": runs,
        "comparisons": [{"references": ["single", "factorized_single", "independent4_body", "factorized_independent4_body"],
             "candidates": list(BASE_ARMS[1:])+list(I4_SEARCH_MEMBER), "seed_ids": list(COMPARATIVE_SEEDS),
             "report": "all failures/costs; paired seed deltas; raw and post-pool scalar CAL proper losses; macro eligible-task ROC-AUC",
             "original_GNNM_reference": "not_implemented_for_this_binary_graph_adapter; comparison_unresolved",
             "acceptance_guarantee": False}],
        "release": {"policy": "whole_roster", "primary_endpoint": "all_closed_raw_and_calibrated_molecular_bank_evaluations",
                    "global_plan_gates_required": True},
        "resources": {"wall_seconds": profile["wall_seconds"], "profile": profile,
                      "status": "PREPARED_ESTIMATES_REQUIRE_RUNTIME_QUALIFICATION"}}
    validate(spec)
    if set(BASE_ARMS) | set(I4_SEARCH_MEMBER) != ARMS:
        raise ValueError("Molecular compiler arm coverage differs from the actual adapter")
    for row in runs:
        if row["kind"] == "model_fit":
            validate_config(row["config"], row)
            validate_role_authorities(spec, row["config"])
            validate_independent_roster(spec, row)
    fit_rows = [r for r in runs if r["kind"] == "model_fit"]
    starts = dict(Counter(r["config"]["stage"] for r in fit_rows))
    counts = {"optimizer_starts_by_stage": starts, "initial_training_segments": len(fit_rows),
        "continuation_segments": 0, "extension_segments": 0, "fresh_source_stage_fits": 0,
        "physical_body_instances_by_stage": copy.deepcopy(starts),
        "independently_trained_member_fits_by_stage": dict(Counter(r["config"]["stage"] for r in fit_rows if "independent4_body" in r["config"]["arm"])),
        "coupled_optimizer_fits_by_stage": dict(Counter(r["config"]["stage"] for r in fit_rows if r["config"]["arm"].startswith("be_"))),
        "full_budget_epochs": len(fit_rows)*epochs, "optimizer_update_total": "dataset_and_tail_minibatch_dependent; measured_by_engine",
        "search_selection_jobs": len(selections), "comparative_served_banks": len(banks), "bank_exports": len(banks),
        "scalar_CAL_fits": len(banks), "evaluation_jobs": len(banks), "genuine_I4_banks": sum(len(b["fit_parents"]) == 4 for b in banks),
        "comparative_served_prediction_views": sum(b["routes"] for b in banks), "reused_I4_body0_artifacts": sum(b["body0_reused"] for b in banks),
        "reused_member_search_selection_rules": 4*len(COMPARATIVE_SEEDS), "all_run_records": len(runs)}
    return spec, counts, banks, domains


def make_plan(prepared, relative_output, host_bindings):
    """Use root's existing dependency registry and phase-gate receipt contract."""
    groups = [{"id": spec["study_id"], "estimated_seconds": sum(r["config"]["estimated_seconds"] for r in spec["runs"]),
               "maximum_cuda_bytes": max(r["config"]["reservation_mib"] for r in spec["runs"])*1024**2,
               "run_keys": [spec["study_id"]+"/"+r["run_id"] for r in spec["runs"]]} for spec, *_ in prepared]
    # Current scheduling authorization is allocation-only. The root can merge
    # these groups into a larger approved plan after verifying its hosts.
    assigned = assign(groups, capacities={"allocation/gpu0": None})
    plan = {"schema_version": 1, "slots_per_gpu": 2, "torch_threads_per_fit": 2,
            "output_root": relative_output+"/execution", "studies": [], "groups": assigned["groups"],
            "estimated_gpu_seconds": assigned["estimated_gpu_seconds"], "estimates_are_measurements": False,
            "scientific_launch_enabled": False, "human_GO_required": True,
            "host_bindings": host_bindings, "global_gates": {}, "barriers": {}}
    for name, sources, required in (("whole_search", ["search", "search_selection"], ["comparative"]),
            ("whole_comparative", ["comparative"], ["bank_export", "calibration", "evaluation"]),
            ("whole_calibration", ["calibration"], ["evaluation"])):
        plan["global_gates"][name] = {"path": relative_output+"/gate_receipts/"+name+".json",
            "source_stages": sources, "required_for_stages": required, "required_hosts": ["allocation"],
            "receipt_contract": "status=complete;plan_hash=current_plan_hash;closed_phase=gate_name;host_receipts_keys=required_hosts"}
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "resources", "exposure-decisions", "host-bindings", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--custody", nargs="+", required=True, help="Closed molecular MANIFEST.json paths")
    parser.add_argument("--source-commit")
    parser.add_argument("--engine-prefix", help="Project-relative immutable package source directory; defaults to this compiler's directory")
    parser.add_argument("--epochs", nargs="+", type=int, choices=EPOCH_DOMAINS, default=list(EPOCH_DOMAINS))
    parser.add_argument("--scalar-coefficients", choices=["neighbors", "native_fixed"], default="neighbors")
    parser.add_argument("--plain-json", action="store_true")
    parser.add_argument("--practical-policy", help="Exact separately bound finite policy; leaves the unrestricted grid unchanged")
    args = parser.parse_args()
    verify_allocation()
    root = Path(args.project_root).resolve()
    def local(value):
        path = Path(value)
        return confined(root, str(path.resolve().relative_to(root)) if path.is_absolute() else str(path))
    output = local(args.output)
    if output.exists():
        raise ValueError("Compiler output exists; retain it and use a fresh packet path")
    engine_path = local(args.engine_prefix) if args.engine_prefix else Path(__file__).resolve().parent
    if not engine_path.is_relative_to(root):
        raise ValueError("Compiler package must lie inside the declared project root")
    engine_prefix = str(engine_path.relative_to(root))
    if engine_path / "compile_molecular_matrix.py" != Path(__file__).resolve():
        raise ValueError("Execute the compiler from the immutable package selected by engine-prefix")
    resources_path, exposure_path, host_path = local(args.resources), local(args.exposure_decisions), local(args.host_bindings)
    profiles = json.loads(resources_path.read_text())["profiles"]
    exposure_decisions = json.loads(exposure_path.read_text())["datasets"]
    hosts = json.loads(host_path.read_text())
    allocation = hosts.get("allocation", {})
    if allocation.get("hostname") != "anogena-2-0" or allocation.get("gpu_uuids") != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Host receipt does not bind the authorized allocation")
    matrix_path = engine_path / "docs/MATRIX_AXES_DRAFT.json"
    matrix = json.loads(matrix_path.read_text())
    recipes = [r for r in matrix["baseline_recipe_catalogue"]["rows"] if r["recipe_id"] == "gine_residual_virtualnode_molhiv"]
    if len(recipes) != 1:
        raise ValueError("Exactly one pinned GINE numeric recipe required")
    recipe = recipes[0]
    practical_policy = None
    if args.practical_policy:
        practical_path = local(args.practical_policy)
        practical_policy = json.loads(practical_path.read_text())
        validate_practical_policy(practical_policy, recipe)
        if sorted(set(args.epochs)) != practical_policy['epoch_domains'] or args.scalar_coefficients != practical_policy['coefficient_policy']:
            raise ValueError('CLI horizons and coefficient policy differ from the frozen practical policy')
    molecular_source = source_receipts(root)
    programs = sorted(p for p in engine_path.rglob("*.py") if "__pycache__" not in p.parts)
    programs += [matrix_path, resources_path, exposure_path, host_path]
    if practical_policy is not None:
        programs.append(practical_path)
    for row in recipe["source_bindings"]:
        path = bound_file(root, {**row, "bytes": (root / row["path"]).stat().st_size})
        programs.append(path)
    programs += [root / row["path"] for row in molecular_source.values()]
    common = [file_binding(root, p) for p in dict.fromkeys(programs)]
    commit = args.source_commit or subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if not commit:
        raise ValueError("Bind a source commit alongside the authoritative file fingerprints")
    prepared, blocked = [], []
    seen = set()
    for custody_path in args.custody:
        custody, bindings = bind_custody(root, local(custody_path))
        dataset = custody["dataset"]
        if dataset in seen:
            raise ValueError("One custody roster per molecular dataset in this packet")
        seen.add(dataset)
        exposure = exposure_decisions.get(dataset)
        if not isinstance(exposure, dict) or not exposure.get("prior_exposure"):
            raise ValueError("Missing explicit evidence exposure decision for " + dataset)
        if exposure.get("protected_confirmation") is not False or exposure.get("evaluate_purpose") != "development":
            blocked.append({"dataset": dataset, "epoch_domains": sorted(set(args.epochs)),
                "reason": "protected_confirmation_requires_root_frozen_activation_packet", "model_fit_count": 0,
                "custody_receipt": file_binding(root, local(custody_path)), "exposure_decision": exposure})
            continue
        source = {"git_commit": commit, "files": list({r["path"]:r for r in common+bindings}.values()),
                  "working_tree_fingerprints_authoritative": True, "official_author_protocol_replication": False}
        if practical_policy is not None:
            source['practical_policy'] = {'receipt': file_binding(root, practical_path), 'contract': practical_policy,
                                         'unrestricted_grid_unchanged': True}
        for epochs in sorted(set(args.epochs)):
            prepared.append(make_study(root, custody, recipe, source, molecular_source,
                resource_for(profiles, dataset, epochs), epochs, exposure, args.scalar_coefficients, engine_prefix, practical_policy))
    if practical_policy is not None and seen != set(practical_policy['datasets']):
        raise ValueError('Practical molecular policy requires both explicitly named dataset custodies')
    relative = str(output.relative_to(root))
    plan = make_plan(prepared, relative, hosts)
    plan["immutable_source_bundle"] = {"package": engine_prefix,
        "launcher_entrypoint": engine_prefix+"/launcher.py",
        "model_worker_entrypoint": engine_prefix+"/run_study.py",
        "endpoint_worker_entrypoint": engine_prefix+"/run_endpoint.py",
        "requires_root_runtime_resource_qualification_before_GO": True}
    output.mkdir(parents=True, exist_ok=False)
    (output / "studies").mkdir()
    summaries = []
    for spec, counts, banks, domains in prepared:
        suffix = ".json" if args.plain_json else ".json.gz"
        path = output / "studies" / (spec["study_id"]+suffix)
        raw = (canonical(spec)+"\n").encode()
        path.write_bytes(raw if args.plain_json else gzip.compress(raw, mtime=0))
        plan["studies"].append({"path": str(path.relative_to(root)), "spec_hash": fingerprint(spec)})
        summaries.append({"study_id": spec["study_id"], "dataset": spec["dataset"],
                          "counts": counts, "banks": banks, "search_domains": domains})
    (output / "PLAN.json").write_text(json.dumps(plan, indent=2, allow_nan=False)+"\n")
    summary = {"schema": "molecular-matrix-preparation-summary-v1", "plan_hash": fingerprint(plan),
        "status": "PREPARED_ESTIMATES_REQUIRE_RUNTIME_QUALIFICATION", "epoch_domains": sorted(set(args.epochs)),
        "coefficient_policy": args.scalar_coefficients, "prepared_studies": len(prepared), "per_study": summaries,
        "blocked_dataset_packets": blocked, "all_implemented_molecular_adapter_arms_materialized": bool(prepared),
        "original_GNNM_binary_graph_reference": "unimplemented; scientific_comparison_unresolved",
        "author_replication": "not_created; standardized_four_role_protocol_differs",
        "halving_and_extension_segments": 0, "benchmark_training_started": False, "GO_receipt_created": False}
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "per_study"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
