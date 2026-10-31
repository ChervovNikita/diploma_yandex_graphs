"""Review the resource-only v3 delta with stdlib text/AST/hash/JSON only."""
import ast
import copy
import difflib
import hashlib
import json
import math
import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PHASE = OUT.parents[1]
OLD = "shared_message_predictive_v2"
NEW = "shared_message_predictive_v3"
OLD_DESIGN = "shared_message_study_design_v1/PLAN_ROOT_before_validation_v2.json"
NEW_DESIGN = "shared_message_resource_amendment_v3/PLAN_ROOT_before_validation_v3.json"
OLD_DESIGN_SHA = "021c3d5f6c09d3f9eee181bc9767a56b6647a8a1ff3a1b5c3c7a932a489d9af0"
NEW_DESIGN_SHA = "66b242e582c61205e19264cdcaad2d17ad8e7611f39706b9b1ba648b68e0f198"
CALC = "protocols/SHARED_MESSAGE_COMPLETE_RESOURCE_CALCULATION_ROOT_before_v3_v1.json"
checks, reads, correspondences, differences = [], {}, [], []


def raw(rel):
    path = PHASE / rel
    assert not path.is_symlink() and path.resolve().is_relative_to(PHASE)
    assert path.suffix in (".py", ".json", ".md", ".sh", ".jinja", ".log"), "Non-source/metadata access refused"
    data = path.read_bytes()
    reads[rel] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return data


def value(rel):
    return json.loads(raw(rel))


def check(name, condition, detail=None):
    checks.append(dict(check=name, passed=bool(condition), detail=detail))
    assert condition, name


def ref_ok(ref):
    return set(ref) == {"path", "sha256"} and hashlib.sha256(raw(ref["path"])).hexdigest() == ref["sha256"]


def rename(rel):
    return rel.replace(OLD, NEW)


def rename_value(v):
    if isinstance(v, str):
        return rename(v)
    if isinstance(v, list):
        return [rename_value(x) for x in v]
    if isinstance(v, dict):
        return {rename(k): rename_value(x) for k, x in v.items()}
    return v


manifests = {p: value(p + "/SOURCE_MANIFEST_v1.json") for p in (OLD, NEW)}
check("old_manifest_exact", reads[OLD + "/SOURCE_MANIFEST_v1.json"]["sha256"] == "b9a348bd41c456fcd974b257ef65a19655163022d5f18bf0dc1a4429f57ccec4")
check("new_manifest_exact", reads[NEW + "/SOURCE_MANIFEST_v1.json"]["sha256"] == "e4d1b4b7d96ae07e7588a854cb8a0d70e9af6612ac72d1df583c3c637c376c9b")
for package, manifest in manifests.items():
    for row in manifest["files"]:
        data = raw(package + "/" + row["relative_path"])
        check("seal:" + package + "/" + row["relative_path"], len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"])
    check("binding_link:" + package, manifest["source_bindings_sha256"] == reads[package + "/SOURCE_BINDINGS.json"]["sha256"])
check("same_packet_filenames", [r["relative_path"] for r in manifests[OLD]["files"]] == [r["relative_path"] for r in manifests[NEW]["files"]])
bindings = {p: value(p + "/SOURCE_BINDINGS.json") for p in (OLD, NEW)}
check("new_bindings_exact", reads[NEW + "/SOURCE_BINDINGS.json"]["sha256"] == "b07650322ffbf24a7d979d8ac9f8cb5fe6778e3d5921c6f30ebf8499bc17e618")
for package, binding in bindings.items():
    check("closure_count:" + package, len(binding["sha256"]) == 117)
    for path, wanted in binding["sha256"].items():
        check("closure:" + package + ":" + path, hashlib.sha256(raw(path)).hexdigest() == wanted)
    check("closure_launch_closed:" + package, binding["actual_data_runtime_resource_admission_established"] is False and binding["scientific_training_validation_export_or_cost_launch_authorized"] is False)
old_external = {p: h for p, h in bindings[OLD]["sha256"].items() if not p.startswith(OLD + "/")}
new_external = {p: h for p, h in bindings[NEW]["sha256"].items() if not p.startswith(NEW + "/")}
expected_external = dict(old_external)
del expected_external[OLD_DESIGN]
expected_external[NEW_DESIGN] = NEW_DESIGN_SHA
check("external_closure_only_design_swap", new_external == expected_external)
check("unchanged_qualified_models", bindings[OLD]["sha256"]["shared_message_quality_v3/models.py"] == bindings[NEW]["sha256"]["shared_message_quality_v3/models.py"] and bindings[OLD]["sha256"]["shared_message_quality_v3/SOURCE_MANIFEST_v1.json"] == "607c487f83696a21d5792c3f22b408491f9fb55c8f7e0b1dd1527b8f68653dcc")

same = {"__init__.py", "core.py", "metrics.py", "artifact_plan.py", "import_guard.py"}
imports_only = {"labels.py", "runner.py", "preflight.py"}
python_names = {r["relative_path"] for r in manifests[OLD]["files"] if r["relative_path"].endswith(".py")}
check("python_delta_family_exact", python_names == same | imports_only | {"contract.py", "resource_forecast.py", "seal_source.py", "verify_source.py"})
for name in sorted(python_names):
    a, b = raw(OLD + "/" + name), raw(NEW + "/" + name)
    ast.parse(a.decode(), filename=OLD + "/" + name)
    ast.parse(b.decode(), filename=NEW + "/" + name)
    expected = a.decode()
    kind = "byte_identical"
    if name in imports_only:
        expected = expected.replace(OLD, NEW)
        kind = "self_package_imports_only"
    elif name == "contract.py":
        expected = expected.replace(OLD_DESIGN, NEW_DESIGN).replace(OLD_DESIGN_SHA, NEW_DESIGN_SHA).replace("FORECAST_CAP = 16200", "FORECAST_CAP = 18000")
        kind = "exact_design_reference_and_forecast_cap_only"
    elif name == "resource_forecast.py":
        expected = expected.replace("16200", "18000")
        kind = "forecast_ceiling_and_comparison_only"
    elif name == "seal_source.py":
        expected = expected.replace(OLD_DESIGN, NEW_DESIGN)
        kind = "exact_design_closure_path_only"
    elif name == "verify_source.py":
        expected = expected.replace(OLD, NEW).replace('.name.endswith("v2.json")', '.name.endswith("v3.json")')
        kind = "self_package_check_and_design_suffix_only"
    check("normalized_python:" + name, expected.encode() == b)
    correspondences.append(dict(file=name, kind=kind, old=reads[OLD + "/" + name], new=reads[NEW + "/" + name], normalized_equivalent=True))
    differences.extend(difflib.unified_diff(a.decode().splitlines(True), b.decode().splitlines(True), fromfile=OLD + "/" + name, tofile=NEW + "/" + name))

old_design, new_design = value(OLD_DESIGN), value(NEW_DESIGN)
expected_design = copy.deepcopy(old_design)
expected_design["resource_policy_proposed"]["forecast_ceiling_seconds"] = 18000
amendment = new_design["resource_only_amendment_v3"]
new_without_annotation = copy.deepcopy(new_design)
del new_without_annotation["resource_only_amendment_v3"]
check("design_only_forecast_change", new_without_annotation == expected_design)
check("design_reference_hash", reads[NEW_DESIGN]["sha256"] == NEW_DESIGN_SHA and reads[OLD_DESIGN]["sha256"] == OLD_DESIGN_SHA)
check("amendment_preserves_old_design", amendment["from_design"] == dict(path=OLD_DESIGN, sha256=OLD_DESIGN_SHA) and amendment["v2_preserved_immutable"] is True)
check("amendment_is_prospective", amendment["before_any_predictive_fit_or_validation_export"] is True and amendment["quality_outcomes_scored"] is False and amendment["scientific_training_or_validation_export_admitted"] is False and amendment["root_finalization_and_resource_admission_pending"] is True)

policies = {p: value(p + "/RUNTIME_IMPORT_POLICY_v1.json") for p in (OLD, NEW)}
expected_policy = rename_value(policies[OLD])
expected_policy["importable_sha256"] = {rename(path): bindings[NEW]["sha256"][rename(path)] for path in policies[OLD]["importable_sha256"]}
check("import_policy_relocation_and_hashes_only", policies[NEW] == expected_policy)
check("generated_policy_byte_identical", raw(OLD + "/GENERATED_MODULE_POLICY_v1.json") == raw(NEW + "/GENERATED_MODULE_POLICY_v1.json"))
check("guard_uses_dynamic_local_identity", raw(NEW + "/import_guard.py") == raw(OLD + "/import_guard.py") and b"HERE = Path(__file__).resolve().parent" in raw(NEW + "/import_guard.py") and b"for package in policy[\"package_roots\"]" in raw(NEW + "/import_guard.py"))
policy = policies[NEW]
suffixes = {".py", ".pyw", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}
stems = {str(Path(p).with_suffix("")) for p in policy["importable_sha256"]}
for package in policy["package_roots"]:
    if package != NEW:
        continue
    actual = []
    for directory, dirs, files in os.walk(PHASE / package, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            rel = str(path.relative_to(PHASE))
            assert not path.is_symlink() and name != "__pycache__" and not (rel in stems and rel not in policy["package_directories"]), rel
        for name in files:
            path = Path(directory) / name
            assert not path.is_symlink(), str(path.relative_to(PHASE))
            if path.suffix.lower() in suffixes:
                actual.append(str(path.relative_to(PHASE)))
    expected = sorted(path for path in policy["importable_sha256"] if Path(path).parts[0] == package)
    check("import_tree_exact:" + package, sorted(actual) == expected and sorted(actual) == policy["exact_source_importable_tree"][package])
check("no_old_package_in_new_python", all(OLD.encode() not in raw(NEW + "/" + name) for name in python_names))

calculation = value(CALC)
check("amendment_calculation_pin", amendment["completed_engineering_resource_calculation"] == dict(path=CALC, sha256=reads[CALC]["sha256"]))
for key in ("completed_resource_receipt", "prior_accounting", "ten_cost_root_audit"):
    check("resource_metadata_pin:" + key, ref_ok(calculation[key]))
prior = value(calculation["prior_accounting"]["path"])
ten = value(calculation["ten_cost_root_audit"]["path"])
measurement = value(calculation["completed_resource_receipt"]["path"])
check("completed_measurement_source_only", measurement["data_targets_predictions_models_or_checkpoints_opened"] is False)
check("actual_v2_preflight_identity", ten["request"]["path"].startswith(OLD + "/") and ref_ok(ten["request"]))
old_request = value(ten["request"]["path"])
check("actual_v2_preflight_manifest", old_request["source_manifest_sha256"] == reads[OLD + "/SOURCE_MANIFEST_v1.json"]["sha256"])
check("root_core_and_memory_match_cost_audit", calculation["core_30_seconds"] == ten["future_all30_core_terms_with125percent_margin_seconds"] and calculation["peak_reserved_bytes"] == ten["maximum_reserved_bytes"])
check("root_prior_wall_matches_receipt", calculation["fixed_costs"]["prior_GPU_wall_seconds"] == prior["all_job_wall_seconds_conservatively_charged_to_allocation_and_CPU_wall"])
check("root_forecast_arithmetic", math.isclose(calculation["total_GPU_wall_seconds_forecast"], calculation["core_30_seconds"] + sum(calculation["fixed_costs"].values()), rel_tol=0, abs_tol=1e-9))
check("root_disk_arithmetic", calculation["additional_phase_disk_bytes_forecast"] == prior["phase_disk_conservative_prior_bytes"] + calculation["artifact_plan"]["scientific_artifact_bytes_bound"] + calculation["validation_export_tensor_bytes_bound"])
check("old_ceiling_not_admitted_new_caps_fit", 16200 < calculation["total_GPU_wall_seconds_forecast"] < 18000 < 21600 and calculation["peak_reserved_bytes"] < 60 * 2**30 and calculation["additional_phase_disk_bytes_forecast"] < 200 * 2**30)
check("resource_record_preserves30_9000", calculation["fit_count"] == 30 and calculation["updates"] == 9000 and calculation["qualified"] is False and calculation["scientific_training_authorized"] is False)
for row in calculation["all_helper_chains"]:
    check("helper_start_pin:" + row["job"], ref_ok(row["outer_start"]))
    check("helper_terminal_pin:" + row["job"], ref_ok(row["outer_terminal"]))
    terminal = value(row["outer_terminal"]["path"])
    check("helper_summary:" + row["job"], terminal["whole_supervised_seconds"] == row["seconds"] and terminal["complete"] == row["complete"] and terminal["child_exit_code"] == row["exit_code"])

companion = None
source_cost = None
prospective_final_forecast = None
if len(sys.argv) > 1:
    rel = sys.argv[1]
    declared = value(rel)
    check("companion_correspondence_exact", reads[rel]["sha256"] == "8530ca171e3f783de830a72f89069ead4a655ff69a8d57ac21068f1aa72c0925")
    amendment_manifest_path = "shared_message_resource_amendment_v3/SOURCE_MANIFEST_v3.json"
    amendment_manifest = value(amendment_manifest_path)
    check("amendment_manifest_exact", reads[amendment_manifest_path]["sha256"] == "a96a1a5f89503fdf0203ff1fd56069a47f76a684b491c4c3999cc3da65fb84f0")
    for row in amendment_manifest["files"]:
        data = raw("shared_message_resource_amendment_v3/" + row["relative_path"])
        check("amendment_seal:" + row["relative_path"], len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"])
    for key, path in (("old_manifest", OLD + "/SOURCE_MANIFEST_v1.json"), ("new_manifest", NEW + "/SOURCE_MANIFEST_v1.json"), ("old_bindings", OLD + "/SOURCE_BINDINGS.json"), ("new_bindings", NEW + "/SOURCE_BINDINGS.json"), ("old_design", OLD_DESIGN), ("new_design", NEW_DESIGN)):
        check("companion_identity:" + key, declared[key] == dict(path=path, sha256=reads[path]["sha256"]))
    for row in declared["source_file_correspondence"]:
        name = row["name"]
        a, b = reads[OLD + "/" + name], reads[NEW + "/" + name]
        check("companion_local_file:" + name, row["old_sha256"] == a["sha256"] and row["new_sha256"] == b["sha256"] and row["old_bytes"] == a["bytes"] and row["new_bytes"] == b["bytes"] and row["byte_identical"] == (a == b))
    source_cost = declared["source_cost_correspondence"]
    summaries = {}
    for package in (OLD, NEW):
        paths = bindings[package]["sha256"]
        sizes = [reads[path]["bytes"] for path in paths]
        summaries[package] = dict(files=len(paths), total_bytes=sum(sizes), maximum_file_bytes=max(sizes), path_component_count=sum(len(Path(path).parts) for path in paths))
    check("closure_cost_counts_independently_match", source_cost["old_closure"] == summaries[OLD] and source_cost["new_closure"] == summaries[NEW])
    ratios = {k: summaries[NEW][k] / summaries[OLD][k] for k in summaries[OLD]}
    check("closure_scaling_ratios", source_cost["ratios"] == ratios and source_cost["conservative_size_and_count_multiplier"] == max(1.0, *ratios.values()))
    helper = value("shared_message_cost_admission_v2/fixed_overhead_authorized1gpu_run01/RESULT.json")
    old_worst = max(helper["measured_seconds"]["16_exact_source_closure_full_hash_pass_samples_seconds"])
    old_charge = old_worst * 2048 * 1.25
    new_charge = old_charge * max(1.0, *ratios.values())
    check("retained_measured_source_cost", source_cost["source_hash_worst_sample_seconds"] == old_worst and source_cost["old_2048pass_source_hash_charge_seconds"] == old_charge and source_cost["closure_passes"] == 2048 and source_cost["retained_engineering_margin"] == 1.25)
    check("additional_source_cost_arithmetic", source_cost["suggested_v3_2048pass_source_hash_charge_seconds"] == new_charge and source_cost["suggested_additional_source_hash_reserve_seconds"] == new_charge-old_charge)
    check("source_correspondence_not_a_new_runtime_claim", source_cost["new_runtime_source_hash_measurement_performed"] is False and source_cost["statistical_or_absolute_runtime_upper_bound_claimed"] is False and source_cost["root_cost_correspondence_adoption_required"] is True and declared["old_measurements_keep_their_actual_v2_source_runtime_import_and_outer_chains"] is True and declared["no_v3_guard_weakening_or_legacy_import_identity_claim"] is True)
    adoption = value("shared_message_resource_amendment_v3/DRAFT_RESOURCE_ADOPTION_v3.json")
    for key in ("actual_completed_helpers_audit", "actual_completed_v2_cost_audit", "preliminary_root_resource_calculation", "preserved_v2_cost_source_bindings", "preserved_v2_cost_source_manifest", "scientific_source_bindings", "scientific_source_manifest", "unchanged_model_source_manifest", "design", "source_correspondence"):
        check("adoption_pin:" + key, ref_ok(adoption[key]))
    check("adoption_explicit_actual_generations", adoption["preserved_v2_cost_source_manifest"] == declared["old_manifest"] and adoption["scientific_source_manifest"] == declared["new_manifest"] and adoption["source_cost_correspondence"] == source_cost and adoption["source_correspondence"] == dict(path=rel, sha256=reads[rel]["sha256"]))
    check("adoption_is_root_owned_draft", adoption["qualified"] is False and adoption["resource_ledger"] is None and adoption["root_approval"] is None and adoption["scientific_training_authorized"] is False and adoption["validation_target_export_authorized"] is False and adoption["test_target_access_authorized"] is False)
    prospective_final_forecast = calculation["total_GPU_wall_seconds_forecast"] + source_cost["suggested_additional_source_hash_reserve_seconds"] + 30
    check("root_declared_final_allowances_fit_ceiling", prospective_final_forecast < 18000)
    companion = dict(reference=dict(path=rel, sha256=reads[rel]["sha256"]), declared_receipt=declared)

receipt = dict(schema="gnnm-independent-scientific-resource-only-delta-audit-v3", verdict="no_concrete_blocker_in_resource_only_v3_delta", source_only=True, old_manifest=dict(path=OLD + "/SOURCE_MANIFEST_v1.json", sha256=reads[OLD + "/SOURCE_MANIFEST_v1.json"]["sha256"]), new_manifest=dict(path=NEW + "/SOURCE_MANIFEST_v1.json", sha256=reads[NEW + "/SOURCE_MANIFEST_v1.json"]["sha256"]), new_bindings_sha256=reads[NEW + "/SOURCE_BINDINGS.json"]["sha256"], amended_design=dict(path=NEW_DESIGN, sha256=NEW_DESIGN_SHA), independent_checks=checks, python_file_correspondence=correspondences, root_calculation=dict(path=CALC, sha256=reads[CALC]["sha256"]), resource_arithmetic=dict(total_GPU_wall_seconds_forecast=calculation["total_GPU_wall_seconds_forecast"], new_forecast_ceiling_seconds=18000, hard_wall_seconds=21600, forecast_headroom_seconds=18000-calculation["total_GPU_wall_seconds_forecast"], hard_wall_headroom_seconds=21600-calculation["total_GPU_wall_seconds_forecast"], peak_reserved_bytes=calculation["peak_reserved_bytes"], additional_phase_disk_bytes_forecast=calculation["additional_phase_disk_bytes_forecast"]), companion_correspondence=companion, reviewed_reads=reads, author_source_executed_or_imported=False, scientific_or_numerical_code_executed=False, dataset_provider_target_checkpoint_logit_bytes_opened=False, new_numerical_or_unchanged_model_gate_required=False, v2_costs_remain_actual_v2_evidence=True, old_v2_generation_remains_not_resource_admitted=True, root_approval_resource_admission_or_launch_issued=False)
receipt["source_cost_correspondence_arithmetic"] = source_cost
receipt["root_declared_prospective_final_forecast_seconds"] = prospective_final_forecast
receipt["root_declared_publication_and_final_query_allowance_seconds"] = 30
receipt["discovery_missing_paths"] = ["shared_message_resource_amendment_v3/SOURCE_MANIFEST_v1.json", "shared_message_resource_amendment_v3/FINAL_CHECKS_v3.json"]
if len(sys.argv) > 1:
    for name, payload in (("AUDIT_RECEIPT.json", receipt), ("SOURCE_CORRESPONDENCE.json", correspondences)):
        with (OUT / name).open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    with (OUT / "PYTHON_DELTA.diff").open("x") as stream:
        stream.write("".join(differences))
print(json.dumps(dict(checks=len(checks), verdict=receipt["verdict"], companion_bound=companion is not None, receipt_written=len(sys.argv)>1, arithmetic=receipt["resource_arithmetic"]), sort_keys=True))
