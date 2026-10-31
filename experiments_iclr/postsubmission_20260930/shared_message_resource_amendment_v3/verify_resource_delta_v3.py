"""Narrow source/metadata correspondence for the prospective resource-only v3."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
OLD = PHASE / "shared_message_predictive_v2"
NEW = PHASE / "shared_message_predictive_v3"
OLD_DESIGN = "shared_message_study_design_v1/PLAN_ROOT_before_validation_v2.json"
NEW_DESIGN = "shared_message_resource_amendment_v3/PLAN_ROOT_before_validation_v3.json"
OLD_DESIGN_SHA = "021c3d5f6c09d3f9eee181bc9767a56b6647a8a1ff3a1b5c3c7a932a489d9af0"
NEW_DESIGN_SHA = "66b242e582c61205e19264cdcaad2d17ad8e7611f39706b9b1ba648b68e0f198"
OLD_MANIFEST_SHA = "b9a348bd41c456fcd974b257ef65a19655163022d5f18bf0dc1a4429f57ccec4"
NEW_MANIFEST_SHA = "e4d1b4b7d96ae07e7588a854cb8a0d70e9af6612ac72d1df583c3c637c376c9b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(relative):
    return dict(path=relative, sha256=sha(PHASE / relative))


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def normalized(name, text):
    value = text.replace("shared_message_predictive_v3", "shared_message_predictive_v2")
    if name == "contract.py":
        value = value.replace(NEW_DESIGN, OLD_DESIGN).replace(NEW_DESIGN_SHA, OLD_DESIGN_SHA).replace("FORECAST_CAP = 18000", "FORECAST_CAP = 16200")
    elif name == "resource_forecast.py":
        value = value.replace("18000", "16200")
    elif name == "seal_source.py":
        value = value.replace(NEW_DESIGN, OLD_DESIGN)
    elif name == "verify_source.py":
        value = value.replace('contract.DESIGN).name.endswith("v3.json")', 'contract.DESIGN).name.endswith("v2.json")')
    return value


def main():
    checks, rows = [], []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True))
    check("Exact preserved v2 and new v3 source manifests", sha(OLD / "SOURCE_MANIFEST_v1.json") == OLD_MANIFEST_SHA
          and sha(NEW / "SOURCE_MANIFEST_v1.json") == NEW_MANIFEST_SHA)
    check("Exact original and new prospective design refs", sha(PHASE / OLD_DESIGN) == OLD_DESIGN_SHA and sha(PHASE / NEW_DESIGN) == NEW_DESIGN_SHA)
    old_design = json.loads((PHASE / OLD_DESIGN).read_text())
    new_design = json.loads((PHASE / NEW_DESIGN).read_text())
    restored = copy.deepcopy(new_design)
    restored.pop("resource_only_amendment_v3")
    restored["resource_policy_proposed"]["forecast_ceiling_seconds"] = 16200
    check("Design differs only by forecast ceiling and explicit amendment metadata", restored == old_design)
    check("All prospective launch flags remain closed", new_design["scientific_training_authorized"] is False
        and new_design["validation_target_export_authorized"] is False and new_design["test_target_export_or_evaluation_authorized"] is False
        and new_design["resource_only_amendment_v3"]["scientific_training_or_validation_export_admitted"] is False)
    old_manifest = json.loads((OLD / "SOURCE_MANIFEST_v1.json").read_text())
    new_manifest = json.loads((NEW / "SOURCE_MANIFEST_v1.json").read_text())
    old_files = {row["relative_path"]: row for row in old_manifest["files"]}
    new_files = {row["relative_path"]: row for row in new_manifest["files"]}
    check("Exact local source file set retained", set(old_files) == set(new_files))
    for name in sorted(old_files):
        old, new = OLD / name, NEW / name
        check("Sealed old/new file identity " + name, sha(old) == old_files[name]["sha256"] and sha(new) == new_files[name]["sha256"]
            and old.stat().st_size == old_files[name]["bytes"] and new.stat().st_size == new_files[name]["bytes"])
        row = dict(name=name, old_sha256=sha(old), new_sha256=sha(new), old_bytes=old.stat().st_size, new_bytes=new.stat().st_size,
                   byte_identical=old.read_bytes() == new.read_bytes())
        if name.endswith(".py"):
            source = new.read_text()
            restored = normalized(name, source)
            check("Exact allowed normalized Python delta " + name, restored.encode() == old.read_bytes())
            row["allowed_normalized_byte_identical"] = True
            row["normalized_AST_identical"] = ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(ast.parse(old.read_text()), include_attributes=False)
        rows.append(row)
    check("Computational core metrics artifact accounting and import guards byte-identical", all((OLD / name).read_bytes() == (NEW / name).read_bytes()
          for name in ("core.py", "metrics.py", "artifact_plan.py", "import_guard.py", "GENERATED_MODULE_POLICY_v1.json")))
    old_bind = json.loads((OLD / "SOURCE_BINDINGS.json").read_text())["sha256"]
    new_bind = json.loads((NEW / "SOURCE_BINDINGS.json").read_text())["sha256"]
    mapped = {path.replace("shared_message_predictive_v2/", "shared_message_predictive_v3/")
              .replace(OLD_DESIGN, NEW_DESIGN) for path in old_bind}
    check("Closure membership changes only by self relocation and exact design ref", mapped == set(new_bind))
    upstream = {path: wanted for path, wanted in old_bind.items() if not path.startswith("shared_message_predictive_v2/") and path != OLD_DESIGN}
    check("Every numerical/model/upstream dependency remains byte-identical", all(new_bind[path] == wanted and sha(PHASE / path) == wanted for path, wanted in upstream.items()))
    policy_old = json.loads((OLD / "RUNTIME_IMPORT_POLICY_v1.json").read_text())
    policy_new = json.loads((NEW / "RUNTIME_IMPORT_POLICY_v1.json").read_text())
    normalized_policy = json.loads(json.dumps(policy_new).replace("shared_message_predictive_v3", "shared_message_predictive_v2"))
    check("Runtime tree/schema/package directories preserve exact relocated interface", all(normalized_policy[key] == policy_old[key] for key in policy_old if key != "importable_sha256"))
    check("Importable source set retained under exact relocation", set(normalized_policy["importable_sha256"]) == set(policy_old["importable_sha256"]))
    spec = importlib.util.spec_from_file_location("resource_v3_guard_source_check", NEW / "import_guard.py")
    guard = importlib.util.module_from_spec(spec)
    exec(compile((NEW / "import_guard.py").read_bytes(), str(NEW / "import_guard.py"), "exec", dont_inherit=True), guard.__dict__)
    actual_policy, tree = guard.verify_packet(NEW_MANIFEST_SHA)
    check("Exact new sealed source closure and merged import tree verify", tree == policy_new["exact_source_importable_tree"])
    check("No legacy v2 source import is admitted as v3", "shared_message_predictive_v2" not in actual_policy["package_roots"]
        and not any(path.startswith("shared_message_predictive_v2/") for path in actual_policy["importable_sha256"]))
    check("No scientific third-party package imported by correspondence verifier", not any(name == "torch" or name.startswith(("torch.", "numpy", "torch_geometric", "ogb")) for name in sys.modules))
    counts_old = dict(files=len(old_bind), total_bytes=sum((PHASE / path).stat().st_size for path in old_bind),
        maximum_file_bytes=max((PHASE / path).stat().st_size for path in old_bind),
        path_component_count=sum(len(Path(path).parts) for path in old_bind))
    counts_new = dict(files=len(new_bind), total_bytes=sum((PHASE / path).stat().st_size for path in new_bind),
        maximum_file_bytes=max((PHASE / path).stat().st_size for path in new_bind),
        path_component_count=sum(len(Path(path).parts) for path in new_bind))
    ratios = {key: counts_new[key] / counts_old[key] for key in counts_old}
    helper = json.loads((PHASE / "shared_message_cost_admission_v2/fixed_overhead_authorized1gpu_run01/RESULT.json").read_text())
    old_max = max(helper["measured_seconds"]["16_exact_source_closure_full_hash_pass_samples_seconds"])
    old_charge = old_max * 2048 * 1.25
    scale = max(1.0, *ratios.values())
    new_charge = old_charge * scale
    check("Unchanged source count and path-stat work permit narrow conservative byte scaling", counts_new["files"] == counts_old["files"]
        and counts_new["path_component_count"] == counts_old["path_component_count"] and counts_new["maximum_file_bytes"] == counts_old["maximum_file_bytes"]
        and ratios["total_bytes"] < 1.02)
    source_cost = dict(old_closure=counts_old, new_closure=counts_new, ratios=ratios, conservative_size_and_count_multiplier=scale,
        source_hash_worst_sample_seconds=old_max, closure_passes=2048, retained_engineering_margin=1.25,
        old_2048pass_source_hash_charge_seconds=old_charge, suggested_v3_2048pass_source_hash_charge_seconds=new_charge,
        suggested_additional_source_hash_reserve_seconds=new_charge-old_charge,
        justification="Same117files, same path depth/stat operation count, unchanged maximum file size and same hashing algorithm/filesystem scope; scale retained observed worst-sample2048pass1.25margin charge by maximum file/byte/depth ratio.",
        new_runtime_source_hash_measurement_performed=False, statistical_or_absolute_runtime_upper_bound_claimed=False,
        root_cost_correspondence_adoption_required=True,
        narrow_fresh_sample_required_if_root_rejects_proxy_correspondence=True)
    receipt = dict(schema="gnnm-prospective-resource-only-source-correspondence-v3", UTC=datetime.now(timezone.utc).isoformat(),
        status="narrow_normalized_source_and_guard_correspondence_verified_prospective_root_resource_admission_pending",
        checks=checks, check_count=len(checks), old_design=ref(OLD_DESIGN), new_design=ref(NEW_DESIGN),
        old_manifest=ref("shared_message_predictive_v2/SOURCE_MANIFEST_v1.json"), old_bindings=ref("shared_message_predictive_v2/SOURCE_BINDINGS.json"),
        new_manifest=ref("shared_message_predictive_v3/SOURCE_MANIFEST_v1.json"), new_bindings=ref("shared_message_predictive_v3/SOURCE_BINDINGS.json"),
        source_file_correspondence=rows, byte_identical_upstream_dependency_count=len(upstream),
        operational_schemas_unchanged=True, computational_function_bodies_unchanged_after_exact_self_relocation=True,
        model_quality_v3_requalification_recommended=False, model_source_or_math_or_precision_or_artifact_behavior_changed=False,
        forecast_ceiling_only_change=dict(old=16200, new=18000, hard_wall_unchanged=21600, memory_unchanged=60*2**30, disk_unchanged=200*2**30),
        source_cost_correspondence=source_cost,
        old_measurements_keep_their_actual_v2_source_runtime_import_and_outer_chains=True,
        old_measurements_must_be_verified_by_actual_v2_verifier_and_audit=True,
        no_v3_guard_weakening_or_legacy_import_identity_claim=True,
        Torch_models_data_targets_checkpoints_GPU_remote_or_prior_scores_accessed=False,
        old_sealed_packets_or_finalized_receipts_modified=False, scientific_or_resource_launch_authorized=False)
    write(HERE / "SOURCE_CORRESPONDENCE_v3.json", receipt)
    adoption = dict(schema="gnnm-prospective-v3-adoption-of-completed-v2-cost-evidence-draft-v1", qualified=False,
        source_correspondence=ref("shared_message_resource_amendment_v3/SOURCE_CORRESPONDENCE_v3.json"),
        scientific_source_manifest=receipt["new_manifest"], scientific_source_bindings=receipt["new_bindings"], design=receipt["new_design"],
        preserved_v2_cost_source_manifest=receipt["old_manifest"], preserved_v2_cost_source_bindings=receipt["old_bindings"],
        actual_completed_v2_cost_audit=ref("shared_message_predictive_v2/COMPLETED_TEN_COST_PREFLIGHT_ROOT_AUDIT_authorized1gpu_run01.json"),
        actual_completed_helpers_audit=ref("shared_message_completed_cost_audit_v1/AUDIT_RECEIPT_v1.json"),
        preliminary_root_resource_calculation=ref("protocols/SHARED_MESSAGE_COMPLETE_RESOURCE_CALCULATION_ROOT_before_v3_v1.json"),
        unchanged_model_source_manifest=dict(path="shared_message_quality_v3/SOURCE_MANIFEST_v1.json", sha256="607c487f83696a21d5792c3f22b408491f9fb55c8f7e0b1dd1527b8f68653dcc"),
        source_cost_correspondence=source_cost, resource_ledger=None, root_approval=None,
        scientific_training_authorized=False, validation_target_export_authorized=False, test_target_access_authorized=False,
        required_root_actions=["Independently accept the narrow exact source/closure correspondence.",
            "Retain actual v2 completed measurement verification; never label its old imported tree as v3.",
            "Adopt the additional source-hash engineering reserve or obtain a narrow fresh metadata timing sample.",
            "Update conservative phase bytes and all prior/query/audit charges after source publication.",
            "Finalize the actual resource ledger/approval and validation permissions under distinct v3 source refs."])
    write(HERE / "DRAFT_RESOURCE_ADOPTION_v3.json", adoption)
    print(json.dumps(dict(checks=len(checks), old_closure=counts_old, new_closure=counts_new,
        additional_source_seconds=new_charge-old_charge, launch_closed=True)))


if __name__ == "__main__":
    main()
