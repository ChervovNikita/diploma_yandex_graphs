"""Source-specific disabled continuous-affinity node packet and TRAIN-only custody.

Main verifies the literal allocation before project reads in either CLI mode.
Target preparation reads GRAPH/TRAIN only. Compilation hashes opaque artifacts
and emits no model fits, scores, GO, target endpoint decode or automatic launch.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess

from adapters.continuous_affinity import (ARMS, SOURCE_ARMS, SOURCE_FILES, SOURCE_CONTRACT,
    objective_module, read_bound, receipt, source_paths, target_payload)
from compile_native_matrix import bind_custody, file_binding, RUNTIME_FILES, resource_for
from prepare_node_data import verify_allocation
from spec import canonical, confined, fingerprint, validate

SEARCH_SEEDS = (75101, 75103)
COMPARATIVE_SEEDS = (75201, 75203, 75207)
DOMAIN = [{"learning_rate": lr, "dropout": dropout} for lr in (.0005, .001, .002) for dropout in (.1, .2, .4)]
SOURCE_ANCHOR = {"learning_rate": .001, "dropout": .2}
REFERENCES = ("ordinary_m1", "factorized_m1", "shared4_unchanged", "original_boundary")
I4 = {"ordinary_i4": "ordinary_m1", "factorized_i4": "factorized_m1",
      "genuinefactorizedI4_weighted_mixture": "factorizedM1_weighted_mixture"}
METHODS = {arm: "continuous_context_affinity_" + arm.lower() for arm in ARMS}
METHODS.update(ordinary_m1="ordinary_native_single", factorized_m1="allmap_factor_single",
    shared4_unchanged="allmap_factor_shared4", original_boundary="original_boundary_GNNM",
    ordinary_i4="ordinary_genuine_independent4", factorized_i4="allmap_factor_genuine_independent4")


def validate_custody(custody):
    expected = {"wikics": 20, "amazon-photo": 10}
    dataset = custody["dataset_id"]
    if dataset not in expected or sorted(r["split_id"] for r in custody["splits"]) != list(range(expected[dataset])):
        raise ValueError("Exact WikiCS20 or Photo10 provider roster required")
    if custody["status"] != "prepared" or custody["benchmark_fits"] != 0 or custody["predictive_scores"] != 0:
        raise ValueError("Actual no-fit/no-score provider custody required")
    for row in custody["splits"]:
        counts = row["role_class_counts"]["train"]
        if set(counts) != {str(i) for i in range(custody["model_graph"]["classes"])} or any(v < 2 for v in counts.values()):
            raise ValueError("Each declared TRAIN class needs a nonself positive; no class omission")
        if row["evaluate_purpose"] != "development":
            raise ValueError("No unused confirmation claim is admitted")


def prepare_target_custody(root, custody_path, source_bindings, output):
    """Called only after main's allocation guard; no heldout tensor decoder."""
    import numpy as np
    import torch
    root = Path(root).resolve();output = Path(output)
    if output.exists():raise ValueError("Preserve previous target custody; fresh output required")
    custody = json.loads(Path(custody_path).read_text());validate_custody(custody)
    paths = source_paths(root, source_bindings);helper = objective_module(paths["objective"])
    directory = Path(custody_path).parent
    file_rows = {r["path"]: r for r in custody["files"]}
    def binding(relative):
        actual = receipt(root, str((directory / relative).resolve().relative_to(root)))
        if {k: actual[k] for k in ("bytes", "sha256")} != {k:file_rows[relative][k] for k in ("bytes", "sha256")}:
            raise ValueError("GRAPH/TRAIN changed after provider custody")
        return actual
    output.mkdir(parents=True)
    entries = []
    for row in custody["splits"]:
        graph_binding, train_binding = (binding(row["node_inputs"][k]) for k in ("graph", "train"))
        graph = torch.load(read_bound(root, graph_binding), map_location="cpu", weights_only=True)
        train = torch.load(read_bound(root, train_binding), map_location="cpu", weights_only=True)
        if set(graph) != {"x", "edge_index"} or set(train) != {"ids", "y"}:
            raise ValueError("Target custodian opens exactly target-free GRAPH and TRAIN")
        ancestry = {"graph": graph_binding, "train": train_binding,
                    "objective_sha256": SOURCE_FILES["objective"][1]}
        pack = target_payload(torch, np, helper, graph, train, ancestry)
        path = output / ("split_" + str(row["split_id"]) + ".pt")
        torch.save(pack, path)
        path.chmod(0o444)
        entries.append({"split_id": row["split_id"], "target": receipt(root, str(path.relative_to(root))),
                        "preparation": pack["preparation"], "tensor_hashes": pack["tensor_hashes"]})
    value = {"schema": "continuous-affinity-target-custody-v1", "dataset_id": custody["dataset_id"],
             "provider_custody": receipt(root, str(Path(custody_path).relative_to(root))),
             "request_hash": custody["request_hash"], "source_bindings": source_bindings,
             "source_contract": SOURCE_CONTRACT, "splits": entries, "model_constructed": False,
             "optimizer_created": False, "SELECT_CAL_TEST_tensors_opened": False,
             "scientific_fits": 0, "GO_created": False}
    (output / "CUSTODY.json").write_text(canonical(value) + "\n")
    (output / "CUSTODY.json").chmod(0o444)
    return value


def make_study(root, custody, data_dir, targets, sources, source_bindings, profiles, runtime, validator=validate):
    validate_custody(custody)
    root, data_dir = Path(root).resolve(), Path(data_dir).resolve()
    dataset = custody["dataset_id"]
    splits = {r["split_id"]: r for r in custody["splits"]}
    target_rows = {r["split_id"]: r for r in targets["splits"]}
    if (targets["schema"] != "continuous-affinity-target-custody-v1" or targets["dataset_id"] != dataset
            or len(target_rows) != len(targets["splits"]) or set(target_rows) != set(splits)
            or targets["source_contract"] != SOURCE_CONTRACT or targets["source_bindings"] != source_bindings
            or targets["request_hash"] != custody["request_hash"]
            or any(targets[key] is not False for key in ("SELECT_CAL_TEST_tensors_opened", "model_constructed", "optimizer_created"))):
        raise ValueError("Exact shared TRAIN-only target custody required")
    known = {r["path"]: r for r in custody["files"]}
    def inputs(split, field):
        return {key: {**known[path], "path": str((data_dir / path).relative_to(root))}
                for key, path in splits[split][field].items()}
    runs, selections, banks = [], {}, []
    def add(identifier, kind, arm, split, seed, body, config, parents=()):
        entrypoint = ("adapters/continuous_affinity.py:make_adapter" if kind == "model_fit" else
            "adapters/search.py:select_configuration" if kind == "selection" else
            "adapters/node_endpoints.py:" + {"ensemble_assembly": "export_bank", "calibration": "scalar_calibration", "evaluation": "evaluate_bank"}[kind])
        row = {"run_id": identifier, "kind": kind, "method": METHODS[arm], "split": split, "seed": seed,
            "body": body, "adapter": runtime + "/" + entrypoint, "config": config, "parents": list(parents),
            "budget": {"unit": "optimizer_updates", "policy": "full_budget", "maximum": 5000} if kind == "model_fit" else {},
            "selection": {"metric": "accuracy", "direction": "max", "ties": "first", "evaluation_interval": 1, "checkpoint0": False} if kind == "model_fit" else {}}
        runs.append(row)
        return row["run_id"]
    def config(arm, split, stage, point):
        architecture, members, mode = SOURCE_ARMS[arm]
        profile = resource_for(profiles, dataset, "continuous-affinity-" + arm)
        return {"affinity_arm": arm, "architecture": architecture, "members": members, "affinity_mode": mode,
            "backbone": "SAGE", "width": 128, "depth": 2, "classes": custody["model_graph"]["classes"],
            "weight_decay": 0., "affinity_source": source_bindings, "node_inputs": inputs(split, "node_inputs"),
            "affinity_targets": target_rows[split]["target"] if mode != "none" else None,
            "context_assignment_law": SOURCE_CONTRACT["independent_assignment"] if mode == "weighted_assigned" else None,
            "fusion": "mean_logits" if architecture == "original_gnnm" else "mean_probabilities",
            "stage": stage, "is_original_gnnm": architecture == "original_gnnm", "source_optimizer_count": 1,
            "reservation_mib": profile["reservation_mib"], "estimated_seconds": profile["estimated_seconds_per_update"] * 5000,
            "recipe_scope": "pinned_affinity_mechanism_standardized_new_roles_full5000", **point}
    search_arms = [a for a in ARMS if a not in I4] + list(REFERENCES)
    for arm in search_arms:
        candidates = []
        for point in DOMAIN:
            identity = fingerprint(point)[:20]
            ids = [add("search-" + arm + "-" + identity + "-s" + str(seed) + "-b" + str(body),
                "model_fit", arm, 0, seed, body, config(arm, 0, "search", point))
                for seed in SEARCH_SEEDS for body in range(4 if arm == ARMS[3] else 1)]
            candidates.append({"config_id": identity, "parameters": point, "run_ids": ids,
                               "contains_source_anchor": point == SOURCE_ANCHOR})
        selections[arm] = add("select-" + arm, "selection", arm, 0, 0, 0,
            {"stage": "search_selection", "configurations": candidates, "ranking_metric": "accuracy",
             "ranking_direction": "max", "ranking_scope": "mean_of_own_selected_body_and_seed_accuracy;all_four_assigned_contexts_paid",
             "reservation_mib": 256, "estimated_seconds": 60}, [i for candidate in candidates for i in candidate["run_ids"]])
    for split in sorted(splits):
        for seed in COMPARATIVE_SEEDS:
            fitted = {}
            for arm in search_arms:
                fitted[arm] = []
                for body in range(4 if arm == ARMS[3] else 1):
                    cfg = config(arm, split, "comparative", DOMAIN[0])
                    cfg["selected_config_from"] = selections[arm]
                    fitted[arm].append(add("compare-sp" + str(split) + "-" + arm + "-s" + str(seed) + "-b" + str(body),
                        "model_fit", arm, split, seed, body, cfg, [selections[arm]]))
            for bank, member in I4.items():
                fitted[bank] = list(fitted[member])
                for body in (1, 2, 3):
                    cfg = config(member, split, "comparative", DOMAIN[0])
                    cfg["selected_config_from"] = selections[member]
                    fitted[bank].append(add("compare-sp" + str(split) + "-" + bank + "-s" + str(seed) + "-b" + str(body),
                        "model_fit", member, split, seed, body, cfg, [selections[member]]))
            endpoints = {key: row["path"] for key, row in inputs(split, "endpoint_inputs").items()}
            for arm, parents in fitted.items():
                member = I4.get(arm, arm)
                profile = resource_for(profiles, dataset, "continuous-affinity-" + member)
                fusion = config(member, split, "comparative", DOMAIN[0])["fusion"]
                prefix = "bank-sp" + str(split) + "-" + arm + "-s" + str(seed)
                export = add(prefix, "ensemble_assembly", arm, split, seed, 0,
                    {"stage": "bank_export", "fit_parents": parents, "endpoint_inputs": endpoints, "fusion": fusion,
                     "reservation_mib": profile["reservation_mib"], "estimated_seconds": profile.get("estimated_export_seconds", 60)}, parents)
                cal = add(prefix + "-cal", "calibration", arm, split, seed, 0,
                    {"stage": "calibration", "export_parent": export, "endpoint_inputs": endpoints, "fusion": fusion,
                     "placement": "after_serving_pool", "steps": 500, "learning_rate": .01, "reservation_mib": 512,
                     "estimated_seconds": profile.get("estimated_scalar_cal_seconds", 10)}, [export])
                evaluate = add(prefix + "-evaluate", "evaluation", arm, split, seed, 0,
                    {"stage": "evaluation", "export_parent": export, "calibration_parent": cal, "endpoint_inputs": endpoints,
                     "fusion": fusion, "reservation_mib": 512, "estimated_seconds": profile.get("estimated_evaluation_seconds", 10)}, [export, cal])
                banks.append({"arm": arm, "split": split, "seed": seed, "fit_parents": parents, "export": export,
                    "calibration": cal, "evaluation": evaluate, "genuine_I4": len(parents) == 4,
                    "body0_reused": arm in I4, "source_bound_context_assignment": arm == ARMS[3]})
    roles = {role: {str(split): inputs(split, "node_inputs" if role in ("train", "select") else "endpoint_inputs")
        [role if role in ("train", "select") else role + "_targets"] for split in splits} for role in ("train", "select", "cal", "evaluate")}
    study = {"schema_version": 1, "study_id": "continuous-affinity-standardized-v1-" + dataset, "stage": "development", "dataset": dataset,
        "hypothesis": "Exact continuous full-support weighted affinity and target assignment on source SAGE versus own-only, equipped independent and actual boundary original GNNM references.",
        "data": {"task": "node_classification", "preprocessing": custody["request"]["graph_policy"], "roles": roles,
            "prior_exposure": custody["request"]["evaluation_exposure"], "evidence": {"custody_request_hash": custody["request_hash"],
                "model_graph": custody["model_graph"], "source_contract": SOURCE_CONTRACT, "source_exact_training_recipe": False,
                "prepared_provider_splits": sorted(splits), "source_history": "closed_private_recipe_failed_raw_attribution_and_I4_mean_NLL_protection;not_erased_by_migration",
                "source_target_preparations_once_per_split": len(splits), "protected_confirmation": False}},
        "source": sources, "runs": runs,
        "comparisons": [{"references": [METHODS[a] for a in (*REFERENCES, "ordinary_i4", "factorized_i4")],
            "candidates": [METHODS[a] for a in ARMS], "split_ids": sorted(splits), "seed_ids": list(COMPARATIVE_SEEDS),
            "private_attribution_required_controls": [METHODS[a] for a in ("shared4_unchanged", ARMS[2], ARMS[3], ARMS[5], ARMS[6])],
            "all_seven_canonical_records_preserved": True}],
        "release": {"policy": "whole_roster", "primary_endpoint": "closed_raw_and_scalar_CAL_banks", "global_plan_gates_required": True},
        "resources": {"wall_seconds": max(p["wall_seconds"] for p in profiles.values()), "status": "PROVISIONAL_DISABLED_REQUIRES_ROOT_RESOURCE_REFERENCE_QA_AND_GO"}}
    validator(study)
    fits = [r for r in runs if r["kind"] == "model_fit"]
    counts = {"atomic_fit_groups": len(fits), "paid_optimizer_objects": len(fits), "physical_model_instances": len(fits),
        "full_budget_optimizer_updates": 5000 * len(fits), "optimizer_starts_by_stage": dict(Counter(r["config"]["stage"] for r in fits)),
        "TRAIN_native_route_forwards": 5000 * sum(r["config"]["members"] for r in fits),
        "source_target_preparations": len(splits), "search_selection_jobs": len(selections), "comparative_banks": len(banks),
        "bank_exports": len(banks), "scalar_CAL_fits": len(banks), "registered_scalar_CAL_updates": 500 * len(banks),
        "evaluation_jobs": len(banks), "genuine_I4_banks": sum(b["genuine_I4"] for b in banks),
        "reused_exact_I4_body0": sum(b["body0_reused"] for b in banks), "all_run_records": len(runs),
        "actual_model_fits": 0, "actual_optimizer_updates": 0, "actual_predictive_scores": 0}
    return study, counts, banks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "source-bindings", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--custody", nargs="+", required=True)
    parser.add_argument("--prepare-targets", action="store_true")
    parser.add_argument("--targets-custody", nargs="+")
    for name in ("runtime-package", "resources", "source-commit"):
        parser.add_argument("--" + name)
    parser.add_argument("--fixture-receipts", nargs="+")
    args = parser.parse_args()
    verify_allocation()  # Before any project metadata/array read in either mode.
    root = Path(args.project_root).resolve()
    def path(value):
        actual = Path(value)
        return confined(root, str(actual.resolve().relative_to(root)) if actual.is_absolute() else str(actual))
    bindings_path, output = path(args.source_bindings), path(args.output)
    source_bindings = json.loads(bindings_path.read_text())
    source_paths(root, source_bindings)
    if args.prepare_targets:
        if len(args.custody) != 1:
            raise ValueError("Each target custody has exactly one provider parent")
        value = prepare_target_custody(root, path(args.custody[0]), source_bindings, output)
        print(canonical({"target_splits": len(value["splits"]), "models": 0, "optimizers": 0, "scientific_fits": 0, "GO_created": False}))
        return
    if not all((args.targets_custody, args.runtime_package, args.resources, args.fixture_receipts)):
        raise ValueError("Prepared targets, frozen runtime, explicit resources and meaningful CPU QA receipts are required")
    if len(args.targets_custody) != len(args.custody) or output.exists():
        raise ValueError("Pair every provider and target custody; preserve prior output")
    runtime = path(args.runtime_package)
    if runtime.is_relative_to(root / "training_pipeline_v2_20261012"):
        raise ValueError("Use an immutable runtime outside the live source package")
    resource_path = path(args.resources)
    resources = json.loads(resource_path.read_text())
    capacities = resources["worker_capacities_cuda_bytes"]
    if set(capacities) != {"allocation/gpu0"} or type(capacities["allocation/gpu0"]) is not int or capacities["allocation/gpu0"] <= 0:
        raise ValueError("Allocation-only disabled source preparation; no server77 assignment")
    runtime_files = list(RUNTIME_FILES) + ["adapters/continuous_affinity.py", "compile_continuous_affinity_matrix.py",
                                        "compile_native_matrix.py", "prepare_node_data.py"]
    sources = [file_binding(root, runtime / name) for name in runtime_files]
    sources += list(source_bindings.values()) + [file_binding(root, bindings_path), file_binding(root, resource_path)]
    sources += [file_binding(root, path(name)) for name in args.fixture_receipts]
    schema_spec = importlib.util.spec_from_file_location("affinity_frozen_schema", runtime / "spec.py")
    schema = importlib.util.module_from_spec(schema_spec)
    schema_spec.loader.exec_module(schema)
    commit = args.source_commit or subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    prepared, observed_datasets = [], set()
    for custody_argument, target_argument in zip(args.custody, args.targets_custody):
        custody_path, target_path = path(custody_argument), path(target_argument)
        custody, data_dir, data_bindings = bind_custody(root, custody_path)
        validate_custody(custody)
        if custody["dataset_id"] in observed_datasets:
            raise ValueError("Repeated provider study")
        observed_datasets.add(custody["dataset_id"])
        targets = json.loads(target_path.read_text())
        if targets["provider_custody"] != file_binding(root, custody_path):
            raise ValueError("Target/provider custody ancestry differs")
        target_bindings = [file_binding(root, target_path)]
        for row in targets["splits"]:
            if file_binding(root, path(row["target"]["path"])) != row["target"]:
                raise ValueError("Prepared target pack changed")
            target_bindings.append(row["target"])
        source = {"git_commit": commit, "files": list({r["path"]: r for r in sources + data_bindings + target_bindings}.values()),
                  "frozen_runtime_package": str(runtime.relative_to(root)), "working_tree_fingerprints_authoritative": True,
                  "source_archived_recipe_reproduction": False}
        prepared.append(make_study(root, custody, data_dir, targets, source, source_bindings, resources["profiles"],
                                   str(runtime.relative_to(root)), schema.validate))
    if observed_datasets != {"wikics", "amazon-photo"}:
        raise ValueError("The complete declared affinity packet requires WikiCS20 and Photo10")
    if any(r["config"]["reservation_mib"] * 1024**2 > capacities["allocation/gpu0"] for study, _, _ in prepared for r in study["runs"]):
        raise ValueError("A declared fit or endpoint reservation exceeds the planning bound")
    output.mkdir(parents=True)
    (output / "studies").mkdir()
    prefix = str(output.relative_to(root))
    plan = {"schema_version": 1, "slots_per_gpu": 2, "torch_threads_per_fit": 2,
        "output_root": prefix + "/execution", "runtime_package": str(runtime.relative_to(root)),
        "runtime_launcher": str((runtime / "launcher.py").relative_to(root)), "worker_capacities_cuda_bytes": capacities,
        "studies": [], "groups": [], "barriers": {}, "global_gates": {}, "scientific_launch_enabled": False,
        "human_GO_required": True, "host_bindings": {"allocation": {"status": "verified_at_compilation", "hostname": "anogena-2-0",
        "gpu_uuids": ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]}}}
    summaries, combined = {}, Counter()
    for study, counts, banks in prepared:
        relative = prefix + "/studies/" + study["study_id"] + ".json.gz"
        (root / relative).write_bytes(gzip.compress((canonical(study) + "\n").encode(), mtime=0))
        (root / relative).chmod(0o444)
        plan["studies"].append({"path": relative, "spec_hash": fingerprint(study)})
        plan["groups"].append({"id": study["study_id"], "gpu_assignment": "allocation/gpu0", "allowed_workers": ["allocation/gpu0"],
            "maximum_cuda_bytes": max(r["config"]["reservation_mib"] for r in study["runs"]) * 1024**2,
            "estimated_seconds": sum(r["config"]["estimated_seconds"] for r in study["runs"]),
            "run_keys": [study["study_id"] + "/" + r["run_id"] for r in study["runs"]]})
        summaries[study["dataset"]] = {"counts": counts, "banks": banks, "study_hash": fingerprint(study)}
        combined.update({key: value for key, value in counts.items() if type(value) is int})
    for name, stages, required in (("whole_search", ["search", "search_selection"], ["comparative"]),
        ("whole_comparative", ["comparative"], ["bank_export", "calibration", "evaluation"]),
        ("whole_calibration", ["calibration"], ["evaluation"])):
        plan["global_gates"][name] = {"path": prefix + "/gate_receipts/" + name + ".json", "source_stages": stages,
            "required_for_stages": required, "required_hosts": ["allocation"]}
    summary = {"schema": "disabled-continuous-affinity-standardized-summary-v1", "plan_hash": fingerprint(plan),
        "counts": dict(combined), "per_dataset": summaries, "source_contract": SOURCE_CONTRACT, "domain": DOMAIN,
        "source_anchor": SOURCE_ANCHOR, "source_canonical_methods": list(METHODS[a] for a in ARMS),
        "source_mechanism_port": True, "source_exact_training_recipe": False, "full_249_coverage": False,
        "resources_provisional": True, "scientific_training_started": False, "GO_created": False,
        "status": "DISABLED_PENDING_ROOT_RESOURCE_REFERENCE_QA_MASTER_AND_HUMAN_GO"}
    (output / "PLAN.json").write_text(canonical(plan) + "\n")
    (output / "SUMMARY.json").write_text(canonical(summary) + "\n")
    (output / "PLAN.json").chmod(0o444)
    (output / "SUMMARY.json").chmod(0o444)
    print(canonical({"status": summary["status"], "plan_hash": fingerprint(plan), "counts": dict(combined), "scientific_fits": 0, "GO_created": False}))


if __name__ == "__main__":
    main()
