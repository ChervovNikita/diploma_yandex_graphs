"""Bind the existing disabled v3 master to compatible sync inputs, without tar."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

MASTER = "massive_practical_master_20261012_v3/PLAN.json"
MASTER_HASH = "081b1358d83468c8062844288f28c9f741984176d441466df8c7daded82a57ef"
CONTROLLER = "master_controller_source_20261012_v2/runtime"
COMPAT = "master_sync_runtime_compatibility_20261012_v2"
ENV_HASH = "2dc6abc0ec496c6351f2842d7ffce2c2f081eeb6465fb53eb8eb61aacaef48d9"
RECIPE_HASH = "776d82c8ded92f63e82183ce0f000d659a45f12187c14252462f5168fc1466d5"
POLICY_HASH = "10832a5ddf4cb71225706dfbce7844a062f501c314618947e4f4334e2eb489cf"


def fresh(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    stage = root / args.output
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    sys.path.insert(0, str(root / CONTROLLER))
    import prepare_sync_bundle as sync
    from launcher import runtime_preflight
    sync.allocation_guard()
    plan = sync.read_json(root, MASTER)
    assert sync.fingerprint(plan) == MASTER_HASH
    assert plan["runtime_package"] == CONTROLLER and plan["runtime_compatibility_required"] is True
    assert plan["scientific_launch_enabled"] is False and plan["human_GO_required"] is True
    assert len(plan["studies"]) == 15 and len(plan["immutable_source_bundle"]["files"]) == 17
    expected = sync.read_json(root, COMPAT + "/COMPATIBLE_ENVIRONMENT.json")
    policy = sync.read_json(root, COMPAT + "/REQUIRED_COMPATIBILITY_POLICY.json")
    recipe = sync.read_json(root, COMPAT + "/FINAL_COMPATIBLE_RECIPE.json")
    assert expected == plan["runtime_requirements"]
    assert sync.fingerprint(expected) == ENV_HASH and sync.fingerprint(policy) == POLICY_HASH
    assert sync.fingerprint(recipe) == RECIPE_HASH and recipe["environment_hash"] == ENV_HASH
    assert recipe["source_environment"] == expected and recipe["required_compatibility"] == policy
    assert sync.compatibility_requirements(expected) == policy
    metadata_check = sync.environment_verify(expected)
    runtime_check = runtime_preflight(plan, sys.executable)
    assert runtime_check["source_environment_hash"] == ENV_HASH
    assert runtime_check["mandatory_numerical_pins_verified"] and runtime_check["native_binary_compatibility_verified"]
    assert runtime_check["CUDA_initialized"] is False
    fresh(stage / "ACTUAL_BOUND_RUNTIME_COMPATIBILITY_RECEIPT.json", runtime_check)
    prior_inventory = sync.read_json(root, COMPAT + "/SOURCE_INVENTORY.json")
    for name in ("prepare_sync_bundle.py", "prepare_sync_environment.py"):
        assert (root / CONTROLLER / name).read_bytes() == (root / COMPAT / "runtime" / name).read_bytes()
    master_summary = sync.read_json(root, "massive_practical_master_20261012_v3/COMPACT_SUMMARY.json")
    assert master_summary["plan_hash"] == MASTER_HASH
    assert master_summary["counts"]["totals"]["run_records"] == 26770
    assert master_summary["counts"]["totals"]["atomic_fit_run_groups"] == 11581
    paths = [COMPAT + "/" + name for name in (
        "COMPATIBLE_ENVIRONMENT.json", "FINAL_COMPATIBLE_RECIPE.json", "REQUIRED_COMPATIBILITY_POLICY.json",
        "SOURCE_INVENTORY.json", "COMPATIBLE_ENVIRONMENT_MAP_PROOF.json", "FINAL_HANDOFF.md",
        "ACTUAL_ALLOCATION_STARTUP_API_RECEIPT.json", "CPU_QA_INHERITANCE.json")]
    paths += ["master_sync_runtime_compatibility_20261012_v1/REQUIRED_RUNTIME_COMPATIBILITY_CPU_QA_RESULT.json",
              expected["source_observation_snapshot"]["path"]]
    paths += ["massive_practical_master_20261012_v3/" + name for name in (
        "COMPACT_SUMMARY.json", "SUMMARY.json", "READINESS.json", "ACTUAL_CONTROLLER_LOAD_VERIFICATION.json", "WORKER_COSTS.csv")]
    paths += [args.output + "/" + name for name in (
        "SOURCE_INVENTORY.json", "ABSENT_PROVIDER_FIXTURE_RESULT.json", "ACTUAL_BOUND_RUNTIME_COMPATIBILITY_RECEIPT.json")]
    paths += [row["path"] for row in sync.read_json(root, args.output + "/SOURCE_INVENTORY.json")]
    cache, calls = {}, Counter()
    original_receipt = sync.receipt

    def cached_receipt(project_root, relative):
        assert Path(project_root).resolve() == root
        if relative not in cache:
            calls[relative] += 1
            cache[relative] = original_receipt(project_root, relative)
        return dict(cache[relative])

    sync.receipt = cached_receipt
    started = time.monotonic()
    explicit = [cached_receipt(root, name) for name in sorted(set(paths))]
    explicit += prior_inventory + plan["immutable_source_bundle"]["files"]
    manifest = sync.manifest(root, MASTER, expected, explicit)
    sync.receipt = original_receipt
    elapsed = time.monotonic() - started
    assert max(calls.values()) == 1 and len(cache) == manifest["statistics"]["files"]
    assert manifest["statistics"]["studies"] == 15 and manifest["statistics"]["registered_runs"] == 26770
    assert manifest["statistics"]["author_groups"] == 8
    union = {row["path"] for row in manifest["files"]}
    assert {row["path"] for row in plan["immutable_source_bundle"]["files"]} <= union
    assert all(row["execution_host"] == "allocation" for row in manifest["allocation_only_author_contracts"])
    # Clarify the historical wording retained in frozen helper bytes using v5 metadata.
    manifest["preparation_status"] = "DISABLED_V3_SNAPSHOT_RESOURCES_AND_MIGRATIONS_PENDING_NOT_FINAL_ALL"
    manifest["runtime_admission_policy"] = {
        "mandatory": "runtime_compatibility_policy;exact_numerical_and_ABI_pins;stdlib_capabilities;CPU_native_probe",
        "provenance_only": policy["provenance_only_version_fields"],
        "existing_environment_recreation_mandatory": False,
        "legacy_exact_standardized_runtime_environment_label_is_not_a_micro_or_system_library_equality_gate": True}
    manifest["compatible_environment_binding"] = cached_receipt(root, COMPAT + "/COMPATIBLE_ENVIRONMENT.json")
    manifest["final_compatible_recipe_binding"] = cached_receipt(root, COMPAT + "/FINAL_COMPATIBLE_RECIPE.json")
    manifest["final_compatible_recipe_canonical_hash"] = RECIPE_HASH
    fresh(stage / "MANIFEST.json", manifest)
    summary = {"schema": "disabled-v3-sync-preparation-summary-v1", "manifest_hash": sync.fingerprint(manifest),
               "plan_hash": MASTER_HASH, "environment_hash": ENV_HASH, "recipe_hash": RECIPE_HASH,
               "required_compatibility_policy_hash": POLICY_HASH, "statistics": manifest["statistics"],
               "model_groups": 11581, "controller_files": 17, "manifest_wall_seconds": elapsed,
               "unique_files_opaque_hashed": len(cache), "maximum_full_hashes_per_path": max(calls.values()),
               "source_data_arrays_or_targets_deserialized": False, "resources_pending": True,
               "inventory_migrations_pending": True, "final_ALL_plan": False, "tar_created": False,
               "server77_access": False, "installs": 0, "scientific_fits": 0, "GO_created": False}
    fresh(stage / "SUMMARY.json", summary)
    roster = {"plan_hash": MASTER_HASH, "studies": manifest["studies"],
              "model_groups": 11581, "registered_records": 26770, "controller_source_files": plan["immutable_source_bundle"]["files"]}
    fresh(stage / "COMPACT_ROSTER.json", roster)
    proof = {"plan_hash": MASTER_HASH, "bound_compatible_environment_equals_plan_inline_map": True,
             "final_recipe_binds_exact_compatible_map": True, "controller_helpers_match_compatible_v2": True,
             "all17_controller_files_in_union": True, "all_study_source_and_required_data_receipts_verified": True,
             "opaque_hash_count": len(cache), "maximum_full_hashes_per_path": max(calls.values()),
             "environment_metadata_check": metadata_check, "actual_launcher_runtime_preflight_passed": True,
             "author_absolute_execution_still_allocation_only": True,
             "data_and_target_deserialization": False, "tar_created": False, "installs": 0,
             "server77_access": False, "scientific_fits": 0, "GO_created": False}
    fresh(stage / "MANIFEST_BINDING_PROOF.json", proof)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
