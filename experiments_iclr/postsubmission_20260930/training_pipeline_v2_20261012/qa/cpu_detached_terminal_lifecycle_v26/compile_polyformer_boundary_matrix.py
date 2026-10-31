"""Fresh disabled standardized PolyFormer v2 from immutable v1 metadata.

Retains every prior learning contract and appends original boundary M4 and
matched source boundary M1/I4. No tensor decoding or scientific fit occurs.
"""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path

from adapters.polyformer_boundary import BOUNDARY_SOURCE, TRAIN_SOURCE, M4, M1, validate_config
from adapters.polyformer_endpoints import independent_contracts
from compile_native_matrix import file_binding
from compile_molecular_matrix import make_plan
from compile_polyformer_matrix import SEARCH_SEEDS, COMPARATIVE_SEEDS
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate


def receipt(root, path):
    return file_binding(root, confined(root, str(path)))


def verify_receipt(root, binding):
    if receipt(root, binding["path"]) != binding:
        raise ValueError("Immutable PolyFormer source/custody receipt changed")


def boundary_domain(original):
    selection = next(r for r in original["runs"] if r["kind"] == "selection" and r["method"] == "polyformer_native_own")
    values = {}
    for candidate in selection["config"]["configurations"]:
        parameters = copy.deepcopy(candidate["parameters"])
        parameters.update(weight_decay=0, attn_weight_decay=0,
                          optimizer={"name": "AdamW", "betas": [.9, .999], "eps": 1e-8,
                                     "grouping": "attnmodule_vs_other", "weight_decay": 0})
        if parameters["loss"] != {"variant": "own", "coefficients": {}}:
            raise ValueError("Boundary source-family domain requires unmodified own CE")
        values[fingerprint(parameters)] = parameters
    if len(values) != 40:
        raise ValueError("Exactly40 effective joint-LR/width/depth/dropout boundary configurations required")
    return list(values.values())


def append_boundary(original, previous, runtime_prefix, source_files):
    value = copy.deepcopy(original)
    value["study_id"] = "polyformer-standardized-v2-" + value["dataset"] + "-mono"
    old_prefix = value["runs"][0]["adapter"].split("/adapters/")[0]
    for row in value["runs"]:
        if not row["adapter"].startswith(old_prefix + "/"):
            raise ValueError("Every prior adapter must belong to its frozen package")
        row["adapter"] = runtime_prefix + row["adapter"][len(old_prefix):]
    value["source"]["v1_ancestry"] = {"study_id": original["study_id"], "spec_hash": fingerprint(original),
        "prior_learning_contracts_unchanged": True, "comparison_labels_corrected_to_actual_export_methods": True}
    value["source"]["files"] = [r for r in value["source"]["files"] if not r["path"].startswith(old_prefix + "/")]
    value["source"]["files"] += source_files + [BOUNDARY_SOURCE, TRAIN_SOURCE]
    unique = {}
    for row in value["source"]["files"]:
        if row["path"] in unique and unique[row["path"]] != row:
            raise ValueError("Conflicting immutable PolyFormer source binding")
        unique[row["path"]] = row
    value["source"]["files"] = list(unique.values())
    value["source"]["original_boundary_adaptation"] = {
        "sites": ["lin1", "lin3"], "interior": "plain_shared_attention_FFN_lin2_no_internal_factors",
        "source_initializer": "first_R_Rademacher_later_R_and_all_S1_private_B0_shared_biasfree_Xavier_W",
        "optimizer": "one_grouped_AdamW_zero_decay_joint_source_LR_pair",
        "objective": "CE/M_separate_backward_old_parameters_then_one_update",
        "fusion": "mean_logits", "published_task_replication": False,
        "declared_changes": ["PolyFormer_backbone_and_LR_grouping", "private_constructor_dropout_RNG",
                             "four_roles_full2000_horizon_first_SELECT_accuracy_selector"]}
    runs = value["runs"]
    prior_by_id = {r["run_id"]: r for r in runs}
    native_fits = [r for r in runs if r["kind"] == "model_fit" and r["method"] == "polyformer_native_own"]
    anchor = next(r for r in native_fits if r["config"]["stage"] == "search")
    horizon = anchor["budget"]["maximum"]
    if horizon != 2000:
        raise ValueError("Boundary packet preserves exact standardized2000 update budget")
    profile = value["resources"]["profile"]
    domains = boundary_domain(original)
    choices, banks = {}, copy.deepcopy(previous["served_banks"])

    def add(identifier, kind, method, split, seed, body, config, parents=()):
        endpoint = {"selection": "search.py:select_configuration", "ensemble_assembly": "polyformer_endpoints.py:export_bank",
                    "calibration": "polyformer_endpoints.py:scalar_calibration", "evaluation": "polyformer_endpoints.py:evaluate_bank"}
        row = {"run_id": identifier, "kind": kind, "method": method, "split": split, "seed": seed, "body": body,
            "adapter": runtime_prefix + "/adapters/" + ("polyformer_boundary.py:make_adapter" if kind == "model_fit" else endpoint[kind]),
            "config": config, "parents": list(parents), "budget": copy.deepcopy(anchor["budget"]) if kind == "model_fit" else {},
            "selection": copy.deepcopy(anchor["selection"]) if kind == "model_fit" else {}}
        if identifier in prior_by_id:
            raise ValueError("Fresh boundary run ID collision")
        runs.append(row)
        return row

    def fit_config(template, mode, members, seed, body, stage, parameters, group=None):
        config = copy.deepcopy(template["config"])
        for key in ("factor_seed", "independence_group", "selected_config_from", "resolved_selection"):
            config.pop(key, None)
        config.update(copy.deepcopy(parameters))
        config.update(polyformer_mode=mode, members=members, fusion="mean_logits", train_views=1,
            boundary_source=copy.deepcopy(BOUNDARY_SOURCE), is_original_gnnm=mode == M4,
            factor_initialization="source_rsb_xavier", factor_seed=None, stage=stage,
            initialization_seed=seed + 1000003 * body,
            dropout_seeds=[seed * 17 + body * 10000019 + slot * 1009 + 300001 for slot in range(members)],
            estimated_seconds=profile["estimated_seconds_per_update"] * horizon * members,
            recipe_scope="standardized_same_PolyFormer_source_boundary_GNNM_adaptation")
        if group:
            config["independence_group"] = group
        return config

    for mode, members in ((M4, 4), (M1, 1)):
        candidates = []
        for parameter in domains:
            identifier = fingerprint(parameter)[:20]
            fits = []
            for seed in SEARCH_SEEDS:
                row = add("search-" + mode + "-" + identifier + "-s" + str(seed), "model_fit", mode, 0, seed, 0,
                          fit_config(anchor, mode, members, seed, 0, "search", parameter))
                fits.append(row["run_id"])
            candidates.append({"config_id": identifier, "parameters": parameter, "run_ids": fits})
        selector = add("select-" + mode, "selection", mode, 0, 0, 0,
            {"stage": "search_selection", "ranking_metric": "accuracy", "ranking_direction": "max",
             "configurations": candidates, "reservation_mib": 256, "estimated_seconds": 1.},
            [i for c in candidates for i in c["run_ids"]])
        choices[mode] = selector["run_id"]

    for split in value["data"]["evidence"]["split_roster"]:
        for seed in COMPARATIVE_SEEDS:
            template = next(r for r in native_fits if r["split"] == split and r["seed"] == seed
                            and r["body"] == 0 and r["config"]["stage"] == "comparative")
            old_bank = next(r for r in runs if r["kind"] == "ensemble_assembly" and r["method"] == "polyformer_native_own"
                            and r["split"] == split and r["seed"] == seed)
            for mode, members in ((M4, 4), (M1, 1)):
                group = "compare-sp" + str(split) + "-" + mode + "-s" + str(seed)
                fits = []
                for body in range(4 if members == 1 else 1):
                    config = fit_config(template, mode, members, seed, body, "comparative", domains[0], group if members == 1 else None)
                    config["selected_config_from"] = choices[mode]
                    fits.append(add(group + "-b" + str(body), "model_fit", mode, split, seed, body, config,
                                    [choices[mode]])["run_id"])
                for served_mode, parents in [(mode, fits[:1])] + ([(mode + "_I4", fits)] if members == 1 else []):
                    bank_id = "bank-sp" + str(split) + "-" + served_mode + "-s" + str(seed)
                    common = {"endpoint_inputs": copy.deepcopy(old_bank["config"]["endpoint_inputs"]), "fusion": "mean_logits"}
                    export = add(bank_id, "ensemble_assembly", served_mode, split, seed, 0,
                        {**common, "stage": "bank_export", "fit_parents": parents,
                         "singleton_reference": "fixed_body0" if members == 1 and len(parents) == 1 else None,
                         "reservation_mib": profile["reservation_mib"],
                         "estimated_seconds": profile["estimated_seconds_per_update"] * max(members, len(parents))}, parents)
                    cal = add(bank_id + "-cal", "calibration", served_mode, split, seed, 0,
                        {**common, "stage": "calibration", "export_parent": bank_id, "placement": "after_serving_pool",
                         "steps": 500, "learning_rate": .01, "reservation_mib": 512, "estimated_seconds": 5.}, [bank_id])
                    evaluation = add(bank_id + "-evaluate", "evaluation", served_mode, split, seed, 0,
                        {**common, "stage": "evaluation", "export_parent": bank_id, "calibration_parent": cal["run_id"],
                         "reservation_mib": 512, "estimated_seconds": 5.}, [bank_id, cal["run_id"]])
                    banks.append({"split": split, "seed": seed, "arm": served_mode, "fit_parents": parents,
                        "export": export["run_id"], "calibration": cal["run_id"], "evaluation": evaluation["run_id"],
                        "routes": max(members, len(parents)), "body0_reused": len(parents) == 4,
                        "single_is_fixed_body0": members == 1 and len(parents) == 1})
    methods = sorted({r["method"] for r in runs if r["kind"] == "ensemble_assembly"})
    for comparison in value["comparisons"]:
        for field in ("references", "candidates", "equipped_I4"):
            if field in comparison:
                comparison[field] = [name if name.startswith("polyformer_") else "polyformer_" + name for name in comparison[field]]
        comparison["references"] = list(dict.fromkeys(comparison.get("references", []) + [M4, M1, M1 + "_I4"]))
        comparison["candidates"] = methods
        comparison["original_GNNM_reference"] = M4
        comparison["same_boundary_equipment_controls"] = [M1, M1 + "_I4"]
    value["hypothesis"] += " Source boundary GNNM M4/M1/I4 are matched same-backbone adaptations."
    for row in runs[len(original["runs"]):]:
        if row["kind"] == "model_fit":
            validate_config(row["config"], row)
    groups = {}
    for row in runs:
        if row["kind"] == "model_fit" and row["config"].get("independence_group"):
            groups.setdefault(row["config"]["independence_group"], []).append(row)
    for cohort in groups.values():
        independent_contracts(value, cohort)
    for comparison in value["comparisons"]:
        cohorts = {(r["split"], r["seed"]) for r in runs if r["kind"] == "ensemble_assembly"}
        for name in comparison["references"] + comparison["candidates"]:
            if {(r["split"], r["seed"]) for r in runs if r["kind"] == "ensemble_assembly" and r["method"] == name} != cohorts:
                raise ValueError("Actual comparison method lacks a full matched split/seed cohort: " + name)
    validate(value)
    fits = [r for r in runs if r["kind"] == "model_fit"]
    counts = copy.deepcopy(previous["counts"])
    counts.update(optimizer_starts_by_stage=dict(Counter(r["config"]["stage"] for r in fits)), initial_training_segments=len(fits),
        physical_body_instances_by_stage=dict(Counter(r["config"]["stage"] for r in fits)),
        coupled_optimizer_fits_by_stage=dict(Counter(r["config"]["stage"] for r in fits if r["config"]["members"] == 4)),
        independent_comparative_body_fits=sum(r["config"]["stage"] == "comparative" and r["config"]["members"] == 1 for r in fits),
        full_budget_optimizer_updates=sum(r["budget"]["maximum"] for r in fits),
        planned_TRAIN_route_forwards=sum(r["config"]["members"] * r["config"]["train_views"] * r["budget"]["maximum"] for r in fits),
        planned_validation_route_forwards=sum(r["config"]["members"] * r["budget"]["maximum"] for r in fits),
        search_selection_jobs=sum(r["kind"] == "selection" for r in runs), comparative_served_banks=len(banks), bank_exports=len(banks),
        scalar_CAL_fits=len(banks), evaluation_jobs=len(banks), genuine_I4_banks=sum(len(b["fit_parents"]) == 4 for b in banks),
        reused_I4_body0_artifacts=sum(b["body0_reused"] for b in banks),
        comparative_served_prediction_views=sum(b["routes"] for b in banks), all_run_records=len(runs))
    domain_meta = copy.deepcopy(previous["search_domains"])
    for mode, members in ((M4, 4), (M1, 1)):
        domain_meta[mode] = {"configuration_count": 40, "members_per_fit": members, "train_views": 1,
            "search_seed_ids": list(SEARCH_SEEDS), "effective_axes": "5jointLR_x2width_x2depth_x2dropout",
            "weight_decay_locked_zero": True, "factor_seed_searched": False}
    return value, counts, banks, domain_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "v1-packet", "runtime-prefix", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); verify_allocation(); root = Path(args.project_root).resolve()
    old = confined(root, args.v1_packet)
    old_plan = json.loads((old / "matrix/PLAN.json").read_text()); old_summary = json.loads((old / "matrix/SUMMARY.json").read_text())
    if fingerprint(old_plan) != old_summary["plan_hash"] or old_plan["scientific_launch_enabled"] or old_summary["GO_created"]:
        raise ValueError("Only immutable disabled PolyFormer v1 may be expanded")
    runtime = confined(root, args.runtime_prefix)
    files = [receipt(root, p.relative_to(root)) for p in sorted(runtime.rglob("*")) if p.is_file() and p.suffix in {".py", ".json", ".md"}]
    files += [receipt(root, (old / name).relative_to(root)) for name in ("matrix/PLAN.json", "matrix/SUMMARY.json")]
    for row in (BOUNDARY_SOURCE, TRAIN_SOURCE):
        verify_receipt(root, row)
    prepared = []
    for row in old_plan["studies"]:
        path = confined(root, row["path"]); original = json.loads(gzip.decompress(path.read_bytes()))
        if fingerprint(original) != row["spec_hash"]:
            raise ValueError("Frozen v1 study changed")
        for binding in original["source"]["files"]:
            verify_receipt(root, binding)
        previous = next(s for s in old_summary["per_study"] if s["study_id"] == original["study_id"])
        prepared.append(append_boundary(original, previous, args.runtime_prefix, files + [receipt(root, path.relative_to(root))]))
    output = confined(root, args.output)
    if output.exists():
        raise ValueError("Preserve earlier packet; use fresh v2 output")
    output.mkdir(); (output / "studies").mkdir()
    plan = make_plan(prepared, args.output, old_plan["host_bindings"])
    plan["immutable_source_bundle"] = {"package": args.runtime_prefix, "launcher_entrypoint": args.runtime_prefix + "/launcher.py",
        "model_worker_entrypoint": args.runtime_prefix + "/run_study.py", "endpoint_worker_entrypoint": args.runtime_prefix + "/run_endpoint.py",
        "requires_root_runtime_resource_qualification_before_GO": True}
    for value, *_ in prepared:
        path = output / "studies" / (value["study_id"] + ".json.gz")
        path.write_bytes(gzip.compress((canonical(value) + "\n").encode(), mtime=0))
        plan["studies"].append({"path": str(path.relative_to(root)), "spec_hash": fingerprint(value)})
    totals = {key: sum(c[key] for _, c, _, _ in prepared) for key in ("initial_training_segments", "all_run_records",
        "full_budget_optimizer_updates", "planned_TRAIN_route_forwards", "planned_validation_route_forwards", "search_selection_jobs",
        "bank_exports", "scalar_CAL_fits", "evaluation_jobs", "genuine_I4_banks", "reused_I4_body0_artifacts", "comparative_served_prediction_views")}
    summary = {"schema": "polyformer-boundary-v2-disabled-preparation-v1", "plan_hash": fingerprint(plan), "totals": totals,
        "per_study": [{"study_id": s["study_id"], "counts": c, "served_banks": b, "search_domains": d} for s, c, b, d in prepared],
        "scientific_training_started": False, "GO_created": False, "scientific_launch_enabled": False,
        "v1_preserved_immutable": True, "v1_data_reused_by_exact_hash": True, "heldout_target_tensors_decoded": False,
        "published_author_protocol_replication": False, "resource_status": "PROVISIONAL_REQUIRES_ROOT_QUALIFICATION"}
    (output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(canonical({k: v for k, v in summary.items() if k != "per_study"}))


if __name__ == "__main__":
    main()
