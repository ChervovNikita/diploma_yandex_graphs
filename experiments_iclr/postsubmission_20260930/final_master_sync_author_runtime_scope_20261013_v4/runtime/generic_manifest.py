"""Successor manifest union for exact final masters including both author families.

Derived from frozen v16 prepare_sync_bundle.manifest; extends author provider
receipts and family-specific empty directories without editing controllers.
Origin SHA256: 2739aaf4ca390a4e840413f64d9b4bc1bdeaf1c6f302cf1e02eaed27e41421a0
"""
from collections import Counter
from pathlib import Path, PurePosixPath
import re
from prepare_sync_bundle import (read_json, checked_receipt, safe, receipt,
    fingerprint, nested_receipts, ORIGINAL_FIVE, compatibility_requirements)

def manifest(root, plan_relative, environment, tooling_receipts=()):
    plan = read_json(root, plan_relative)
    if plan.get("scientific_launch_enabled") is not False or plan.get("human_GO_required") is not True:
        raise ValueError("Only disabled human-GO master plans may be synchronized")
    if not plan.get("runtime_package") or not plan.get("runtime_launcher"):
        raise ValueError("A consolidated frozen master controller is required")
    files, reasons, specs, author, empty, aliases = {}, {}, [], [], set(), {}
    allocation_runtime_providers = {}
    def add(row, reason):
        row = checked_receipt(row)
        if row["path"] in files:
            if files[row["path"]] != row: raise ValueError("Conflicting union receipts: " + row["path"])
            reasons[row["path"]].add(reason); return
        safe(root, row["path"])
        files[row["path"]] = row; reasons.setdefault(row["path"], set()).add(reason)
    def add_path(path, reason):
        if path in files: reasons[path].add(reason)
        else: add(receipt(root, path), reason)
    add(receipt(root, plan_relative), "master_plan")
    for row in tooling_receipts: add(row, "synchronization_tooling")
    runtime = plan["runtime_package"]
    if plan["runtime_launcher"] != runtime + "/launcher.py":
        raise ValueError("Frozen launcher path differs from its runtime package")
    add(receipt(root, plan["runtime_launcher"]), "controller_entrypoint")
    add(receipt(root, runtime + "/start_massive.sh"), "existing_startup_requires_Git_checkout")
    add(receipt(root, "AGENTS.md"), "required_phase_marker")
    rendezvous = plan.get("phase_rendezvous", {})
    if (not rendezvous.get("remote_name") or not re.fullmatch("[0-9a-f]{64}", rendezvous.get("remote_url_sha256", ""))):
        raise ValueError("Plan-bound existing Git receipt remote digest required")
    run_keys, author_keys = set(), set(); assigned = {}
    for entry in plan["studies"]:
        value = read_json(root, entry["path"])
        if fingerprint(value) != entry["spec_hash"]:
            raise ValueError("Master study changed")
        normalize = lambda s: "".join(c for c in str(s).lower() if c.isalnum())
        dataset = value["dataset"]
        scope = value["data"].get("dataset_scope", [])
        if normalize(dataset) in ORIGINAL_FIVE or any(normalize(s) in ORIGINAL_FIVE for s in scope):
            raise ValueError("Original five paper datasets are hard excluded")
        add(receipt(root, entry["path"]), "master_study_spec")
        specs.append({"study_id": value["study_id"], "dataset": dataset, "path": entry["path"], "spec_hash": entry["spec_hash"],
                      "run_count": len(value["runs"])})
        for row in nested_receipts(value["source"]): add(row, "study_source_binding")
        evidence = value["data"].get("evidence", {})
        for row in nested_receipts(evidence.get("provider_files", []) if isinstance(evidence, dict) else []):
            if PurePosixPath(row["path"]).is_absolute():
                original = dict(row)
                try: relative = Path(row["path"]).relative_to(Path(root).absolute()).as_posix()
                except ValueError: raise ValueError("Absolute data provenance lies outside authorized phase; not accessed")
            else:
                original, relative = dict(row), row["path"]
            safe(root, relative, file=False)
            aliases[original["path"]] = {"original_provenance_receipt": original, "phase_relative_mirror_receipt": dict(row, path=relative),
                "spec_rewritten": False, "execution_uses_absolute_provenance_path": False,
                "provenance_only_optional_copy_excluded": True, "reason": "data.evidence.provider_files_is_metadata_not_loader_input"}
        required_data = {key: item for key, item in value["data"].items() if key != "evidence"}
        required_data["evidence"] = ({key: item for key, item in evidence.items() if key != "provider_files"}
                                     if isinstance(evidence, dict) else evidence)
        for row in nested_receipts(required_data): add(row, "required_study_data_receipt")
        for run in value["runs"]:
            key = value["study_id"] + "/" + run["run_id"]
            if key in run_keys: raise ValueError("Repeated study/run identity")
            run_keys.add(key)
            for row in nested_receipts(run.get("config", {})): add(row, "run_input_receipt")
            if run["kind"] != "author_cli_group":
                adapter = run["adapter"].split(":", 1)[0]
                add_path(adapter, "frozen_factory_entrypoint")
                # Some standardized contracts use relative strings for role files.
                for field in ("node_inputs", "endpoint_inputs"):
                    for val in run["config"].get(field, {}).values():
                        if isinstance(val, str): add_path(val, "relative_role_input")
                continue
            author_keys.add(key); cli = run["cli"]
            if "--dataset" in cli.get("argv", []):
                index = cli["argv"].index("--dataset")
                if index + 1 >= len(cli["argv"]) or normalize(cli["argv"][index + 1]) in ORIGINAL_FIVE:
                    raise ValueError("Original five author datasets are hard excluded")
            for row in cli["source_files"] + cli["raw_files"]: add(row, "author_all_host_hash_validation_only")
            # run_ogb_author_cli verifies these receipts against the declared
            # repository, not the research phase. They describe its existing
            # allocation-only environment and must never become phase paths.
            repository = Path(cli["execution_repository"]).resolve()
            if not Path(root).resolve().is_relative_to(repository):
                raise ValueError("Author runtime repository must contain the authorized phase")
            for provider in cli.get("runtime_provider_files", []):
                row = checked_receipt(provider)
                if not row["path"].startswith(".venv/lib/"):
                    raise ValueError("Only declared repository venv library providers are external to phase")
                if receipt(repository, row["path"]) != row:
                    raise ValueError("Allocation author runtime provider changed")
                prior = allocation_runtime_providers.get(row["path"])
                if prior is not None and prior["receipt"] != row:
                    raise ValueError("Conflicting allocation author runtime provider")
                allocation_runtime_providers[row["path"]] = {"receipt": row,
                    "authority": "allocation_author_execution_repository",
                    "scientific_execution_host": "allocation",
                    "phase_relative_file": False, "bundled": False,
                    "actual_author_preflight_rechecks_repository_receipt": True}
            empty.update(cli[field] for field in ("basis_dir", "processed_dir", "fresh_processed_dir") if field in cli)
            for path in cli["quarantine"].values():
                safe(root, path, file=False, forbid=False)
                if (Path(root) / path).exists(): raise ValueError("Author quarantine already exists; never include or repeat")
            author.append({"run_key": key, "execution_host": "allocation", "absolute_execution_repository": cli["execution_repository"],
                "absolute_python_executable": cli["python_executable"], "absolute_environment": cli["environment"],
                "runtime_versions": cli["runtime_versions"], "private_dependency_dir": cli.get("private_dependency_dir"),
                "runtime_provider_files": cli.get("runtime_provider_files", []),
                "source_and_raw_files_must_exist_on_all_hosts_for_hash_validation": True,
                "execute_or_rewrite_author_contract_on_server77": False})
    for group in plan["groups"]:
        for key in group["run_keys"]:
            if key not in run_keys or key in assigned: raise ValueError("Unknown/repeated placement")
            assigned[key] = group["gpu_assignment"]
    if set(assigned) != run_keys: raise ValueError("Missing master placement")
    if any(not assigned[k].startswith("allocation/") for k in author_keys):
        raise ValueError("Absolute author CLI contract must remain allocated to allocation")
    for path in empty:
        directory = safe(root, path, file=False)
        if not directory.is_dir() or any(directory.iterdir()): raise ValueError("Author initial empty workdir changed")
    for row in files.values():
        if receipt(root, row["path"]) != row: raise ValueError("Required artifact bytes changed: " + row["path"])
    sizes = Counter()
    for path, row in files.items(): sizes[PurePosixPath(path).parts[0]] += row["bytes"]
    return {"schema": "plan-bound-sync-manifest-v1", "plan": files[plan_relative], "plan_hash": fingerprint(plan),
        "phase_relative_layout_preserved": True, "study_ids_and_run_ids_preserved": True,
        "files": [dict(row, reasons=sorted(reasons[path])) for path, row in sorted(files.items())],
        "empty_directories": sorted(empty), "studies": specs, "runtime_requirements": environment,
        "runtime_compatibility_policy": compatibility_requirements(environment),
        "allocation_only_author_contracts": author, "host_bindings": plan.get("host_bindings", {}),
        "provider_provenance_receipts_retained_without_optional_copy": [dict(aliases[name],
            required_by_other_source_or_runtime_binding=aliases[name]["phase_relative_mirror_receipt"]["path"] in files)
            for name in sorted(aliases)],
        "Git_overlay_contract": {"existing_or_later_authorized_cloned_checkout_required": True,
            "git_directory_bundled": False, "remote_name": rendezvous["remote_name"],
            "remote_url_sha256": rendezvous["remote_url_sha256"], "source_phase_relative_to_repository": "experiments_iclr/postsubmission_20260930",
            "artifact_layout_relative_to_destination_phase_must_match": True,
            "remote_URL_or_credentials_in_manifest": False},
        "statistics": {"files": len(files), "opaque_bytes": sum(r["bytes"] for r in files.values()),
                       "bytes_by_top_directory": dict(sorted(sizes.items())), "studies": len(specs), "registered_runs": len(run_keys),
                       "author_groups": len(author), "provider_provenance_receipts_retained": len(aliases),
                       "provenance_only_files_not_bundled": sum(a["phase_relative_mirror_receipt"]["path"] not in files for a in aliases.values()),
                       "provenance_only_bytes_not_bundled": sum(a["original_provenance_receipt"]["bytes"] for a in aliases.values()
                           if a["phase_relative_mirror_receipt"]["path"] not in files)},
        "allocation_only_author_runtime_providers": [allocation_runtime_providers[k] for k in sorted(allocation_runtime_providers)],
        "all_study_bindings_needed_on_every_host": True, "tensor_deserialization_performed": False,
        "environment_ready_on_server77": False, "server77_access_authorized": False,
        "scientific_launch_enabled": False, "GO_created": False,
        "portability_requirements": ["destination_phase_relative_layout_exact", "exact_standardized_runtime_environment",
            "verified_plan_hostname_and_GPU_UUIDs", "all_manifest_hashes", "allocation_only_author_absolute_environment",
            "existing_Git_checkout_and_plan_bound_remote_digest_for_start_massive_and_phase_sidecar"],
        "time_estimates": "provisional_placeholder_fit_values_are_not_ETA;accurate_window_requires_later_resource_qualification"}
