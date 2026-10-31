"""Materialize eighty independent seeded OGB CLI processes in a fresh disabled packet."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil

from author_cli_spec import receipt, verify_receipt
from ogb_author_cli_spec import (AUTHOR_FAMILY, BUDGET, DATASETS, ENVIRONMENT, MODELS,
                                 SEEDS, SEED_POLICY, SELECTION, SOURCE_COMMIT,
                                 SOURCE_PINS, bind_ogb_author_files, ogb_author_counts,
                                 seeded_argv, validate_ogb_author_spec)
from run_author_cli import verify_allocation
from spec import canonical, confined, fingerprint

RUNTIME_FILES = ["ogb_author_cli_spec.py", "run_ogb_author_cli.py", "ogb_declared_seed_entry.py",
                 "compile_ogb_author_study.py", "author_cli_spec.py", "run_author_cli.py",
                 "registry.py", "spec.py", "run_study.py"]


def materialize_ogb_author_study(root, preparation, runtime_prefix, output, resource_profiles):
    """Source/raw copies and JSON only; no numerical import, provider load or registry registration."""
    root = Path(root).resolve(); prepared = confined(root, preparation)
    target = confined(root, output)
    if target.exists():
        raise ValueError("Use a fresh disabled OGB source packet; never overwrite/retry prior runs")
    plan_path = prepared / "DISABLED_AUTHOR_REFERENCE_PLAN.json"
    prepared_plan = json.loads(plan_path.read_text())
    if (prepared_plan["author_repository"] != "snap-stanford/ogb"
            or prepared_plan["author_commit"] != SOURCE_COMMIT
            or prepared_plan["recipe_cell_count"] != 8
            or prepared_plan["scientific_launch_enabled"] or prepared_plan["GO_created"]
            or prepared_plan["fits_started"] or prepared_plan["model_constructions"]):
        raise ValueError("Only the exact disabled eight-recipe source/raw preparation is admitted")
    help_path = prepared / "engineering/CLI_HELP_IMPORT_RECEIPT.json"
    help_proof = json.loads(help_path.read_text())
    if (help_proof["exit_code"] != 0 or help_proof["CUDA_VISIBLE_DEVICES"] != ""
            or not help_proof["no_fits_or_model_forwards_or_updates"]):
        raise ValueError("Bind the CPU-visible native help/import engineering receipt")
    target.mkdir(); (target / "sources").mkdir(); (target / "quarantine").mkdir()
    source = {Path(row["path"]).name: row for row in prepared_plan["source_receipts"]}
    bindings = [receipt(root, plan_path.relative_to(root)), receipt(root, help_path.relative_to(root))]
    for filename, pin in SOURCE_PINS.items():
        row = source[filename]
        if (row["bytes"], row["sha256"]) != pin:
            raise ValueError("Upstream OGB source pin mismatch")
        path = verify_receipt(root, row)
        shutil.copyfile(path, target / "sources" / filename)
        bindings.append(receipt(root, (target / "sources" / filename).relative_to(root)))
    wrapper = confined(root, Path(runtime_prefix) / "ogb_declared_seed_entry.py")
    seed_entry = receipt(root, wrapper.relative_to(root))
    shutil.copyfile(wrapper, target / "sources/ogb_declared_seed_entry.py")
    bindings.append(receipt(root, (target / "sources/ogb_declared_seed_entry.py").relative_to(root)))
    for filename in RUNTIME_FILES:
        bindings.append(receipt(root, Path(runtime_prefix) / filename))
    custody = {}
    for dataset, entry in prepared_plan["provider_custody"].items():
        verify_receipt(root, entry["custody_receipt"])
        for row in entry["raw_files"]:
            verify_receipt(root, row)
        custody[dataset] = {"receipt": entry["custody_receipt"], "raw_files": entry["raw_files"],
                            "processed_cache_policy": "fresh_empty_per_process"}
    recipes = {(row["dataset"], row["source_model"]): row for row in prepared_plan["recipe_cells"]}
    if set(recipes) != {(ds, model) for ds in DATASETS for model in MODELS}:
        raise ValueError("Complete eight-recipe preparation required")
    runs = []
    for dataset in DATASETS:
        for model in MODELS:
            recipe = recipes[dataset, model]
            repo = Path(recipe["python_executable"]).parents[2]
            # .venv/bin/python -> repository; runtime files are repo-relative,
            # while source/data/workdir receipts remain phase-relative.
            if not root.is_relative_to(repo):
                raise ValueError("Prepared runtime repository does not contain the phase")
            provider_rows = []
            for row in prepared_plan["provider_programs"]:
                path = Path(row["path"])
                path = path if path.is_absolute() else root / path
                if not path.resolve().is_relative_to(repo.resolve()):
                    raise ValueError("Provider runtime source escaped the declared repository")
                provider_rows.append(receipt(repo, path.relative_to(repo)))
            profile = resource_profiles.get(dataset + "-" + model)
            if not profile or not profile.get("resource_evidence"):
                raise ValueError("Declare an explicit provisional/qualified per-recipe resource profile")
            for seed in SEEDS:
                identifier = dataset + "-" + model + "-seed" + str(seed)
                work = target / "workdirs" / identifier; work.mkdir(parents=True)
                for filename in SOURCE_PINS:
                    shutil.copyfile(target / "sources" / filename, work / filename)
                shutil.copyfile(wrapper, work / "declared_seed_entry.py")
                cache = work / "dataset" / dataset.replace("-", "_")
                raw_files = []
                for row in custody[dataset]["raw_files"]:
                    origin = verify_receipt(root, row)
                    parts = Path(row["path"]).parts; pivot = parts.index(dataset.replace("-", "_"))
                    destination = cache / Path(*parts[pivot+1:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(origin, destination)
                    bound = receipt(root, destination.relative_to(root))
                    if (bound["bytes"], bound["sha256"]) != (row["bytes"], row["sha256"]):
                        raise ValueError("Copied official raw file changed")
                    raw_files.append(bound)
                processed = cache / "processed"; processed.mkdir(exist_ok=False)
                source_files = [receipt(root, (work / filename).relative_to(root))
                                for filename in [*SOURCE_PINS, "declared_seed_entry.py"]]
                runs.append({"run_id": identifier, "kind": "author_cli_group",
                    "method": "OGB-BatchNorm-" + model + "-declared-seed-replication",
                    "split": "official_scaffold", "seed": seed, "body": 0, "parents": [],
                    "config": {"stage": "author_cli", "author_family": AUTHOR_FAMILY,
                        "dataset": dataset, "source_model": model,
                        "reservation_mib": profile["reservation_mib"], "estimated_seconds": profile["estimated_seconds"],
                        "wall_seconds": profile["wall_seconds"], "resource_evidence": profile["resource_evidence"],
                        "max_train_batches_per_epoch": DATASETS[dataset]["max_train_batches"],
                        "source_code_path_replication": True, "published_score_equality_claim": False},
                    "budget": dict(BUDGET), "selection": dict(SELECTION),
                    "cli": {"cwd": str(work.relative_to(root)), "argv": seeded_argv(dataset, model, seed),
                        "python_executable": recipe["python_executable"], "execution_repository": str(repo),
                        "source_files": source_files, "raw_files": raw_files, "environment": dict(ENVIRONMENT),
                        "runtime_versions": prepared_plan["runtime_versions"], "runtime_provider_files": provider_rows,
                        "quarantine": {"stdout": str((target / "quarantine" / (identifier + ".stdout.log")).relative_to(root)),
                                       "stderr": str((target / "quarantine" / (identifier + ".stderr.log")).relative_to(root))},
                        "fresh_processed_dir": str(processed.relative_to(root)), "protocol": dict(SEED_POLICY)}})
    value = {"schema_version": 1, "study_id": "ogb-batchnorm-declared-seed-author-20261012-v1",
        "stage": "development", "hypothesis": "Replicate the eight pinned OGB source code paths under declared independent seeds/runtime and whole-scaffold validation; no published-score equality claim.",
        "dataset": "ogb-two-molecule-native-author-roster",
        "data": {"task": "graph_binary_classification", "preprocessing": "unchanged OGB categorical raw provider; fresh per-process native cache processing",
            "roles": {"train": "official scaffold TRAIN", "select": "whole official scaffold VALID",
                      "evaluate": "official scaffold TEST diagnostics each epoch, quarantined until whole-author closure"},
            "role_protocol": "official_ogb_scaffold_train_whole_valid_test_no_cal",
            "prior_exposure": "development evidence; root project exposure audit determines any future confirmation authority",
            "evidence": [entry["receipt"] for entry in custody.values()], "dataset_custody": custody,
            "train_graph_count_authority": DATASETS, "raw_arrays_or_labels_decoded_by_compiler": False},
        "source": {"author_family": AUTHOR_FAMILY, "repository": "snap-stanford/ogb", "git_commit": SOURCE_COMMIT,
            "files": bindings, "seed_entry": seed_entry, "packet_root": str(target.relative_to(root)),
            "worker_entrypoint": str(Path(runtime_prefix) / "run_ogb_author_cli.py"),
            "replication_seed_policy": dict(SEED_POLICY), "native_CLI_modified": False,
            "published_score_equality_claim": False},
        "runs": runs,
        "comparisons": [{"reporting": "ten independently seeded process reports per frozen recipe after whole-author terminal closure",
                         "rule": "Retain every failure/cost; no selected seed/exclusion/rescue rerun; not published-score equality",
                         "compatible_architecture_baseline_gate": "Matched original-GNNM/M1/genuine-I4 source adaptation scope remains root admission work"}],
        "release": {"policy": "whole_roster", "primary_endpoint": "source validation-curve selection; TEST report quarantine released only after whole-author closure",
            "global_plan_gates_required": True, "required_before_launch": ["whole_search"], "author_output_gate": "whole_author",
            "release_failure_rule": "All required author runs terminal; failed processes retained and never restarted"},
        "resources": {"wall_seconds": sum(run["config"]["wall_seconds"] for run in runs),
            "scope": "Explicit scheduling placeholders unless root-qualified; no model memory/time evidence from CLI help/import"}}
    validate_ogb_author_spec(value); bind_ogb_author_files(root, value)
    (target / "studies").mkdir()
    study_path = target / "studies" / (value["study_id"] + ".json.gz")
    study_path.write_bytes(gzip.compress((canonical(value) + "\n").encode(), mtime=0))
    keys = [value["study_id"] + "/" + run["run_id"] for run in runs]
    disabled = {"schema_version": 1, "slots_per_gpu": 1,
        "studies": [{"path": str(study_path.relative_to(root)), "spec_hash": fingerprint(value)}],
        "groups": [{"id": key.replace("/", "--"), "gpu_assignment": "allocation/gpu0", "run_keys": [key]} for key in keys],
        "barriers": {}, "global_gates": {
            "whole_search": {"path": str((target / "gate_receipts/whole_search.json").relative_to(root)),
                "source_stages": ["search", "search_selection"], "required_for_stages": ["author_cli"],
                "required_hosts": [], "all_packet_source_run_keys": [], "external_complete_master_HPO_cohort_required": True},
            "whole_author": {"path": str((target / "gate_receipts/whole_author.json").relative_to(root)),
                "source_stages": ["author_cli"], "required_for_stages": [], "required_hosts": ["allocation"],
                "all_packet_source_run_keys": sorted(keys), "terminal_failures_preserved": True}},
        "output_root": str((target / "execution").relative_to(root)),
        "runtime_author_worker": value["source"]["worker_entrypoint"],
        "scientific_launch_enabled": False, "human_GO_required": True, "approval_receipt_created": False,
        "master_integration_required": True}
    summary = {"schema": "disabled-ogb-author-source-packet-v1", "plan_hash": fingerprint(disabled),
        "study_hash": fingerprint(value), "counts": ogb_author_counts(value),
        "declared_seed_policy": SEED_POLICY, "source_code_path_replication": True,
        "published_score_equality_claim": False, "scientific_launch_enabled": False,
        "GO_created": False, "models_or_forwards_or_updates": 0, "actual_author_fits": 0,
        "standalone_plan_cannot_satisfy_whole_master_HPO_gate": True,
        "root_freeze_successor_and_master_integration_pending": True}
    (target / "PLAN.json").write_text(json.dumps(disabled, indent=2, sort_keys=True) + "\n")
    (target / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return value, disabled, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project-root", "preparation", "runtime-prefix", "output", "resources"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); verify_allocation()
    root = Path(args.project_root).resolve()
    profiles = json.loads(confined(root, args.resources).read_text())
    value, plan, summary = materialize_ogb_author_study(root, args.preparation, args.runtime_prefix, args.output, profiles)
    print(canonical({"plan_hash": fingerprint(plan), "study_hash": fingerprint(value), "counts": summary["counts"],
                     "scientific_launch_enabled": False, "GO_created": False}))


if __name__ == "__main__":
    main()
