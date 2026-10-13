"""Prepare an exact disabled master sync manifest; metadata and opaque hashes only."""
import argparse
from collections import Counter
import importlib
import os
from pathlib import Path
import sys
import time

from runtime_requirements import contracts
from sync_common import allocation_guard, bound_master, checked, fingerprint, fresh, read_json, receipt, safe, verify_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--master", required=True, help="Exact project-relative master PLAN.json")
    parser.add_argument("--master-hash", required=True, help="Exact canonical frozen plan SHA256")
    parser.add_argument("--summary", required=True, help="Bound compact/full summary with typed counts")
    parser.add_argument("--summary-sha256", required=True, help="Raw summary bytes SHA256")
    parser.add_argument("--tooling-inventory", required=True)
    parser.add_argument("--tooling-inventory-sha256", required=True)
    parser.add_argument("--output", required=True, help="Fresh project-relative manifest directory")
    args = parser.parse_args()
    allocation_guard()  # Before any project read or directory operation.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("Preparation requires CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1")
    root = Path(args.project_root).absolute()
    plan, summary, plan_receipt, summary_receipt = bound_master(
        root, args.master, args.master_hash, args.summary, args.summary_sha256)
    inventory, inventory_receipt = read_json(root, args.tooling_inventory, sha256=args.tooling_inventory_sha256)
    stage = safe(root, args.output, file=False)
    if stage.exists():
        raise FileExistsError("Retain prior manifest and failures; choose a fresh output")
    stage.mkdir()
    cache, calls = {args.master: plan_receipt, args.summary: summary_receipt,
                    args.tooling_inventory: inventory_receipt}, Counter()
    external_allowed, external_cache = {}, {}
    def cached(project_root, relative):
        relative = str(relative)
        authority = Path(project_root).absolute()
        if authority != root:
            key = (str(authority), relative)
            if key not in external_allowed:
                raise ValueError("Receipt root/path is not an exact declared allocation author runtime provider")
            if key not in external_cache:
                calls["author_runtime_repository:" + str(authority) + ":" + relative] += 1
                external_cache[key] = receipt(authority, relative)
            if external_cache[key] != external_allowed[key]:
                raise ValueError("Exact declared allocation author runtime provider changed")
            return dict(external_cache[key])
        if relative not in cache:
            calls[relative] += 1
            cache[relative] = receipt(root, relative)
        return dict(cache[relative])
    controller_files = [checked(row) for row in plan["immutable_source_bundle"]["files"]]
    for row in controller_files:
        if cached(root, row["path"]) != row:
            raise ValueError("Frozen controller changed: " + row["path"])
    controller = safe(root, plan["runtime_package"], file=False)
    for name in ("prepare_sync_bundle.py", "compile_master_plan.py", "launcher.py"):
        relative = plan["runtime_package"] + "/" + name
        if relative not in {row["path"] for row in controller_files}:
            raise ValueError("Controller source bundle omits required metadata helper: " + name)
    sys.path.insert(0, str(controller))
    sync = importlib.import_module("prepare_sync_bundle")
    compiler = importlib.import_module("compile_master_plan")
    generic = importlib.import_module("generic_manifest")
    for module in (sync, compiler):
        if Path(module.__file__).resolve().parent != controller.resolve():
            raise ValueError("Wrong frozen controller module imported")
    studies = [read_json(root, entry["path"], canonical_hash=entry["spec_hash"])[0] for entry in plan["studies"]]
    counts = verify_counts(plan, summary, studies, compiler)
    for study in studies:
        for run in study["runs"]:
            if run["kind"] != "author_cli_group":continue
            cli = run["cli"]
            repository = Path(cli["execution_repository"]).absolute()
            if repository == root or not root.resolve().is_relative_to(repository.resolve()):
                raise ValueError("Declared allocation author runtime repository must be an ancestor distinct from the phase")
            for provider in cli.get("runtime_provider_files", []):
                row = checked(provider)
                if not row["path"].startswith(".venv/lib/"):
                    raise ValueError("Only exact declared repository venv library provider source receipts are external")
                key = (str(repository), row["path"])
                if key in external_allowed and external_allowed[key] != row:
                    raise ValueError("Conflicting declared allocation author runtime provider")
                external_allowed[key] = row
    explicit = [plan_receipt, summary_receipt, inventory_receipt, *controller_files]
    for row in inventory:
        row = checked(row)
        if cached(root, row["path"]) != row:
            raise ValueError("Owned tooling bytes changed")
        explicit.append(row)
    started = time.monotonic()
    original = generic.receipt
    generic.receipt = cached
    try:
        manifest = generic.manifest(root, args.master, plan["runtime_requirements"], explicit)
    finally:
        generic.receipt = original
    manifest_seconds = time.monotonic() - started
    if manifest["statistics"]["studies"] != counts["studies"] or manifest["statistics"]["registered_runs"] != counts["registered_records"]:
        raise ValueError("Manifest quantities differ from frozen typed summary")
    union = {row["path"] for row in manifest["files"]}
    if not {row["path"] for row in controller_files} <= union:
        raise ValueError("Frozen controller omitted from manifest union")
    runtime = contracts(root, plan, studies, manifest["files"], receipt_function=cached)
    # Scoped cache provider files are small source bytes, never old array copies.
    # Keep their exact declaration and isolated overlay beside the cache recipe.
    for provider in runtime["provider_contracts"]:
        if provider["kind"] != "isolated_BUDDY_cache_regeneration_provider":continue
        for row in [provider["source_provider_manifest"], *provider["provider_receipts"]]:
            row = checked(row)
            if cached(root, row["path"]) != row:raise ValueError("BUDDY scoped cache provider bytes changed")
            existing = next((r for r in manifest["files"] if r["path"] == row["path"]), None)
            if existing:
                if checked(existing) != row:raise ValueError("Conflicting BUDDY provider receipt")
                continue
            manifest["files"].append(dict(row, reasons=["isolated_BUDDY_cache_regeneration_provider"]))
            manifest["statistics"]["files"] += 1
            manifest["statistics"]["opaque_bytes"] += row["bytes"]
            by_top = manifest["statistics"]["bytes_by_top_directory"]
            top = Path(row["path"]).parts[0];by_top[top] = by_top.get(top, 0) + row["bytes"]
    fresh(stage / "TRANSITIVE_RUNTIME_REQUIREMENTS.json", runtime)
    runtime_receipt = receipt(root, args.output + "/TRANSITIVE_RUNTIME_REQUIREMENTS.json")
    manifest["files"].append(dict(runtime_receipt, reasons=["transitive_runtime_and_provider_requirements"]))
    manifest["files"].sort(key=lambda row: row["path"])
    manifest["statistics"]["files"] += 1
    manifest["statistics"]["opaque_bytes"] += runtime_receipt["bytes"]
    by_top = manifest["statistics"]["bytes_by_top_directory"]
    top = Path(runtime_receipt["path"]).parts[0]
    by_top[top] = by_top.get(top, 0) + runtime_receipt["bytes"]
    manifest["exact_summary_binding"] = summary_receipt
    manifest["expected_counts"] = counts
    manifest["transitive_runtime_requirements"] = runtime_receipt
    manifest["transitive_runtime_requirements_hash"] = fingerprint(runtime)
    manifest["preparation_status"] = "EXACT_DISABLED_MASTER_METADATA_BOUND_AWAITING_SOURCE_AUDIT_AND_GO"
    manifest["immutable_data_receipts_preserved"] = True
    manifest["no_input_data_copy_performed"] = True
    manifest["hash_policy"] = "one_full_opaque_hash_per_unique_manifest_input_during_build;actual_launcher_independently_checks_source_bindings"
    manifest["runtime_admission_policy"] = {
        "base": "plan-bound numerical and ABI locks plus native CPU compatibility probe before scientific launch",
        "providers": "transitive metadata locks and scoped child/adapter environments; destination native ABI probe required",
        "source_AST_inventory": "conservative_import_inventory;optional_and_unresolved_branches_not_claimed_qualified",
        "existing_environment_recreation_mandatory": False}
    fresh(stage / "MANIFEST.json", manifest)
    result = {"schema": "generic-final-master-sync-preparation-summary-v1", "plan_hash": args.master_hash,
              "master": plan_receipt, "master_summary": summary_receipt,
              "manifest_hash": fingerprint(manifest), "manifest": receipt(root, args.output + "/MANIFEST.json"),
              "statistics": manifest["statistics"], "expected_counts": counts,
              "transitive_runtime_requirements": runtime_receipt,
              "isolated_provider_contracts": len(runtime["provider_contracts"]),
              "manifest_wall_seconds": manifest_seconds,
              "unique_inputs_fully_hashed": len(calls), "maximum_full_hashes_per_unique_input": max(calls.values(), default=0),
              "actual_final_launcher_source_audit_pending": True,
              "target_host_runtime_and_provider_ABI_admission_pending": True,
              "tensor_arrays_or_targets_deserialized": False, "tar_created": False, "data_copies": 0,
              "package_installs": 0, "server77_access": False, "scientific_fits": 0, "GO_created": False}
    fresh(stage / "SUMMARY.json", result)
    fresh(stage / "COMPACT_ROSTER.json", {"plan_hash": args.master_hash, "studies": manifest["studies"],
                                         "expected_counts": counts, "controller_source_files": controller_files})
    print(__import__("json").dumps({k:v for k,v in result.items() if k != "expected_counts"}, sort_keys=True))


if __name__ == "__main__":
    main()
