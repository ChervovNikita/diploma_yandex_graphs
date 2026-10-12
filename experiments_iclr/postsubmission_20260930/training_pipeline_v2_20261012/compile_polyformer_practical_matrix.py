"""Explicit practical PolyFormer policy; rebuild all36 procedures, no row prune."""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path

from adapters.polyformer_boundary import M4, M1, validate_config
from adapters.polyformer_endpoints import independent_contracts
from compile_polyformer_boundary_matrix import receipt, verify_receipt
from compile_polyformer_matrix import templates, SEARCH_SEEDS, COMPARATIVE_SEEDS
from compile_molecular_matrix import make_plan
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate


POLICY = {"schema": "polyformer-practical-policy-v1", "id": "polyformer-practical-20261012-v1",
    "datasets": {"citeseer": {"hidden": 256, "nlayer": 1, "learning_rate": .0005, "attn_lr": .001,
                                "dropout": [.7, .8, .9], "source_anchor_dropout": .9},
                 "pubmed": {"hidden": 256, "nlayer": 2, "learning_rate": .005, "attn_lr": .0005,
                            "dropout": [.3, .5, .7], "source_anchor_dropout": .5}},
    "joint_LR_multipliers": [.5, 1., 2.], "scalar_coefficients": "native_fixed",
    "templates": "all34_existing_plus_boundary_M4_and_matched_boundary_M1",
    "operator_heads_FFN_dprate": "source_scoped_locked", "ordinary_internal_decay": "native_scoped_groups",
    "boundary_decay": "all_groups_zero_AdamW", "search_splits": [0], "comparative_splits": [0],
    "search_seeds": list(SEARCH_SEEDS), "comparative_seeds": list(COMPARATIVE_SEEDS),
    "horizon": 2000, "source_anchor": "tagged_inside9_no_extra_duplicate_fit",
    "I4": "four_paid_M1_bodies_four_own_selectors_fixed_body0_reuse",
    "scientific_launch_enabled": False, "human_GO_required": True}

# Separate prospective Coauthor coverage policy; the frozen CiteSeer/PubMed
# policy and its packets retain their exact declarations.
COAUTHOR_POLICY = {**copy.deepcopy(POLICY), "schema": "polyformer-coauthor-practical-policy-v1",
    "id": "polyformer-coauthor-practical-20261012-v1", "study_prefix": "polyformer-coauthor-practical-v1-",
    "datasets": {"coauthor-cs": {"hidden": 128, "nlayer": 1, "learning_rate": .001, "attn_lr": .005,
                                 "dropout": [0., .1, .2], "source_anchor_dropout": 0.},
                 "coauthor-physics": {"hidden": 128, "nlayer": 1, "learning_rate": .001, "attn_lr": .0001,
                                      "dropout": [.8, .9, .95], "source_anchor_dropout": .9}},
    "track": "standardized_development", "author_score_protocol_equality": False,
    "comparative_splits": list(range(10)),
    "mask_policy": "search_mask0_all10_prepared_comparative_masks_reconstructed_from_source_compiler_APIs",
    "policy_basis": "prospective_three_dropout_values_times_three_joint_LR_multipliers; native_anchor_included_once; no_outcome_selection"}


def practical_domain(recipe, method, policy):
    if policy not in (POLICY, COAUTHOR_POLICY):
        raise ValueError("Practical policy differs from its explicit frozen contract")
    dataset = recipe["dataset_scope"][0]; axes = policy["datasets"][dataset]; anchor = recipe["native_point"]
    if (axes["hidden"] != anchor["width"] or axes["nlayer"] != anchor["depth"]
            or axes["learning_rate"] != anchor["learning_rate_groups"]["other"]
            or axes["attn_lr"] != anchor["learning_rate_groups"]["attnmodule"]
            or axes["source_anchor_dropout"] != anchor["dropout"]):
        raise ValueError("Practical source anchor must equal the pinned native recipe")
    old = {t["method"]: t for t in templates()}
    if method not in old and method not in (M4, M1):
        raise ValueError("Practical packet requires an implemented exact procedure")
    loss = old[method]["loss"] if method in old else {"variant": "own", "coefficients": {}}
    domain = []
    for multiplier in policy["joint_LR_multipliers"]:
        for dropout in axes["dropout"]:
            native = {"K": recipe["operator"]["polynomial_order_K"], "q": recipe["operator"]["q"],
                "multi": recipe["operator"]["multi"], "hidden": axes["hidden"], "nlayer": axes["nlayer"],
                "dropout": dropout, "n_head": recipe["locks"]["heads"], "d_ffn": recipe["locks"]["d_ffn"],
                "dprate": recipe["locks"]["dprate"]}
            parameter = {"native": native, "learning_rate": axes["learning_rate"] * multiplier,
                "attn_lr": axes["attn_lr"] * multiplier,
                "weight_decay": 0 if method in (M4, M1) else anchor["weight_decay_groups"]["other"],
                "attn_weight_decay": 0 if method in (M4, M1) else anchor["weight_decay_groups"]["attnmodule"],
                "loss": copy.deepcopy(loss)}
            if method in (M4, M1):
                parameter["optimizer"] = {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8,
                    "grouping": "attnmodule_vs_other", "weight_decay": 0}
            domain.append({"parameters": parameter, "source_anchor": multiplier == 1. and dropout == axes["source_anchor_dropout"]})
    if len({fingerprint(p["parameters"]) for p in domain}) != 9 or sum(p["source_anchor"] for p in domain) != 1:
        raise ValueError("Nine effective points with one in-domain source anchor required")
    return domain


def practical_study(original, previous, recipe, policy, runtime_prefix, source_files):
    value = copy.deepcopy(original); value["study_id"] = policy.get("study_prefix", "polyformer-practical-v1-") + value["dataset"] + "-mono"
    if value["data"]["evidence"]["split_roster"] != policy["comparative_splits"] or value["dataset"] not in policy["datasets"]:
        raise ValueError("Practical stage requires the declared dataset and exact prepared comparative masks")
    old_prefix = value["runs"][0]["adapter"].split("/adapters/")[0]
    value["source"]["unrestricted_v2_ancestry"] = {"study_id": original["study_id"], "spec_hash": fingerprint(original),
                                                 "unrestricted_packet_modified": False}
    value["source"]["practical_policy"] = {"policy": policy, "policy_hash": fingerprint(policy),
        "all_procedures_rebuilt_from_explicit_source_locked_9_point_policy": True, "arbitrary_row_pruning": False}
    value["source"]["files"] = [r for r in value["source"]["files"] if not r["path"].startswith(old_prefix + "/")]
    value["source"]["files"] += source_files
    unique = {}
    for binding in value["source"]["files"]:
        if binding["path"] in unique and unique[binding["path"]] != binding:
            raise ValueError("Conflicting practical source receipt")
        unique[binding["path"]] = binding
    value["source"]["files"] = list(unique.values())
    selectors = [r for r in value["runs"] if r["kind"] == "selection"]
    if len(selectors) != 36 or {r["method"] for r in selectors} != {t["method"] for t in templates()} | {M4, M1}:
        raise ValueError("Exactly all36 actual standardized procedures required")
    oldfits = [r for r in value["runs"] if r["kind"] == "model_fit" and r["config"]["stage"] == "search"]
    retained = [r for r in value["runs"] if r["config"]["stage"] not in {"search", "search_selection"}]
    runs, choices, domainmeta = [], {}, {}
    profile = value["resources"]["profile"]
    for oldselector in selectors:
        method = oldselector["method"]; template = next(r for r in oldfits if r["method"] == method)
        domain = practical_domain(recipe, method, policy); candidates = []
        for point in domain:
            parameters = point["parameters"]; config_id = fingerprint(parameters)[:20]; identifiers = []
            for seed in policy["search_seeds"]:
                row = copy.deepcopy(template); config = row["config"]
                config.update(copy.deepcopy(parameters)); config.update(practical_policy_hash=fingerprint(policy),
                    initialization_seed=seed, dropout_seeds=[seed * 17 + slot * 1009 + 300001
                    for slot in range(max(config["members"], config["train_views"]))])
                if config["factor_initialization"] == "first_rademacher": config["factor_seed"] = seed + 900001
                config["estimated_seconds"] = profile["estimated_seconds_per_update"] * 2000 * config["members"] * config["train_views"]
                row.update(run_id="search-practical-" + method + "-" + config_id + "-s" + str(seed), seed=seed,
                           adapter=runtime_prefix + row["adapter"][len(old_prefix):])
                runs.append(row); identifiers.append(row["run_id"])
            candidates.append({"config_id": config_id, "parameters": parameters, "run_ids": identifiers,
                               "source_anchor": point["source_anchor"]})
        selector = copy.deepcopy(oldselector); selector["run_id"] = "select-practical-" + method
        selector["adapter"] = runtime_prefix + selector["adapter"][len(old_prefix):]
        selector["config"].update(configurations=candidates, practical_policy_hash=fingerprint(policy))
        selector["parents"] = [i for c in candidates for i in c["run_ids"]]
        runs.append(selector); choices[oldselector["run_id"]] = selector["run_id"]
        domainmeta[method] = {"configuration_count": 9, "members_per_fit": template["config"]["members"],
            "train_views": template["config"]["train_views"], "search_seed_ids": policy["search_seeds"],
            "source_anchor_config_id": next(c["config_id"] for c in candidates if c["source_anchor"]),
            "parameter_points": candidates}
    for row in retained:
        row["adapter"] = runtime_prefix + row["adapter"][len(old_prefix):]
        row["parents"] = [choices.get(i, i) for i in row["parents"]]
        row["config"]["practical_policy_hash"] = fingerprint(policy)
        if row["kind"] == "model_fit":
            domain = practical_domain(recipe, row["method"], policy)
            row["config"].update(copy.deepcopy(next(p["parameters"] for p in domain if p["source_anchor"])))
            row["config"]["selected_config_from"] = choices[row["config"]["selected_config_from"]]
        runs.append(row)
    value["runs"] = runs; value["hypothesis"] += " Explicit practical nine-point policy with native scalar coefficients."
    if any(r["kind"] == "model_fit" and (r["budget"]["maximum"] != policy["horizon"] or
             r["seed"] not in policy["search_seeds"] + policy["comparative_seeds"]) for r in runs):
        raise ValueError("Practical horizon/seed binding changed")
    groups = {}
    for row in runs:
        if row["kind"] == "model_fit":
            if row["method"] in (M4, M1): validate_config(row["config"], row)
            if row["config"].get("independence_group"): groups.setdefault(row["config"]["independence_group"], []).append(row)
    for cohort in groups.values(): independent_contracts(value, cohort)
    validate(value)
    fits = [r for r in runs if r["kind"] == "model_fit"]; banks = copy.deepcopy(previous["served_banks"])
    counts = copy.deepcopy(previous["counts"])
    counts.update(optimizer_starts_by_stage=dict(Counter(r["config"]["stage"] for r in fits)), initial_training_segments=len(fits),
        physical_body_instances_by_stage=dict(Counter(r["config"]["stage"] for r in fits)),
        coupled_optimizer_fits_by_stage=dict(Counter(r["config"]["stage"] for r in fits if r["config"]["members"] == 4)),
        full_budget_optimizer_updates=sum(r["budget"]["maximum"] for r in fits),
        planned_TRAIN_route_forwards=sum(r["config"]["members"] * r["config"]["train_views"] * r["budget"]["maximum"] for r in fits),
        planned_validation_route_forwards=sum(r["config"]["members"] * r["budget"]["maximum"] for r in fits), all_run_records=len(runs))
    return value, counts, banks, domainmeta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "v2-packet", "runtime-prefix", "policy", "output"): parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); verify_allocation(); root = Path(args.project_root).resolve()
    old = confined(root, args.v2_packet); runtime = confined(root, args.runtime_prefix); policy_path = confined(root, args.policy)
    policy = json.loads(policy_path.read_text())
    if policy != POLICY: raise ValueError("Explicit frozen practical policy required")
    old_plan = json.loads((old / "matrix/PLAN.json").read_text()); old_summary = json.loads((old / "matrix/SUMMARY.json").read_text())
    if fingerprint(old_plan) != old_summary["plan_hash"] or old_plan["scientific_launch_enabled"] or old_summary["GO_created"]:
        raise ValueError("Unrestricted v2 must remain disabled and immutable")
    matrix = json.loads((runtime / "docs/MATRIX_AXES_DRAFT.json").read_text())
    files = [receipt(root, path.relative_to(root)) for path in sorted(runtime.rglob("*")) if path.is_file() and path.suffix in {".py", ".json", ".md"}]
    files += [receipt(root, policy_path.relative_to(root))] + [receipt(root, (old / name).relative_to(root)) for name in ("matrix/PLAN.json", "matrix/SUMMARY.json")]
    prepared = []
    for row in old_plan["studies"]:
        path = confined(root, row["path"]); original = json.loads(gzip.decompress(path.read_bytes()))
        if fingerprint(original) != row["spec_hash"]: raise ValueError("Unrestricted v2 spec changed")
        if original["dataset"] not in policy["datasets"]: continue
        for binding in original["source"]["files"]: verify_receipt(root, binding)
        recipe = next(r for r in matrix["baseline_recipe_catalogue"]["rows"] if r["recipe_id"] == "polyformer_mono_" + original["dataset"])
        previous = next(r for r in old_summary["per_study"] if r["study_id"] == original["study_id"])
        prepared.append(practical_study(original, previous, recipe, policy, args.runtime_prefix, files + [receipt(root, path.relative_to(root))]))
    if {s["dataset"] for s, *_ in prepared} != set(policy["datasets"]): raise ValueError("Both prepared practical datasets required")
    output = confined(root, args.output)
    if output.exists(): raise ValueError("Use fresh practical output")
    output.mkdir(); (output / "studies").mkdir(); plan = make_plan(prepared, args.output, old_plan["host_bindings"])
    plan["immutable_source_bundle"] = {"package": args.runtime_prefix, "launcher_entrypoint": args.runtime_prefix + "/launcher.py",
        "model_worker_entrypoint": args.runtime_prefix + "/run_study.py", "endpoint_worker_entrypoint": args.runtime_prefix + "/run_endpoint.py",
        "requires_root_runtime_resource_qualification_before_GO": True}
    for value, *_ in prepared:
        path = output / "studies" / (value["study_id"] + ".json.gz"); path.write_bytes(gzip.compress((canonical(value) + "\n").encode(), mtime=0))
        plan["studies"].append({"path": str(path.relative_to(root)), "spec_hash": fingerprint(value)})
    keys = ("initial_training_segments", "all_run_records", "full_budget_optimizer_updates", "planned_TRAIN_route_forwards",
        "planned_validation_route_forwards", "search_selection_jobs", "bank_exports", "scalar_CAL_fits", "evaluation_jobs",
        "genuine_I4_banks", "reused_I4_body0_artifacts", "comparative_served_prediction_views")
    summary = {"schema": "polyformer-practical-disabled-preparation-v1", "plan_hash": fingerprint(plan), "policy_hash": fingerprint(policy),
        "totals": {k: sum(c[k] for _, c, _, _ in prepared) for k in keys},
        "per_study": [{"study_id": s["study_id"], "counts": c, "served_banks": b, "search_domains": d} for s, c, b, d in prepared],
        "scientific_training_started": False, "scientific_launch_enabled": False, "GO_created": False,
        "unrestricted_v2_preserved_immutable": True, "heldout_target_tensors_decoded": False,
        "resource_status": "PROVISIONAL_REQUIRES_ROOT_QUALIFICATION"}
    (output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n"); (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(canonical({k: v for k, v in summary.items() if k != "per_study"}))


if __name__ == "__main__": main()
