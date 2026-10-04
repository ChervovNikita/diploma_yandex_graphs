"""Capture this bounded result and source immutability; no source execution."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json

OUT = Path(__file__).resolve().parent
auth = json.loads((OUT / "AUTHENTICATION.json").read_text())
rows = []
for packet in [auth["source"], auth["review"], *auth["original_source_packets"].values()]:
    rows.extend([packet["manifest"], *packet["files"]])
rows.extend([auth["source_seal"], auth["review_seal"], auth["runtime_authority_metadata"],
             auth["old_release_metadata"], *auth["fixture_builder_lineage_source"]])
for row in rows:
    value = Path(row["path"]).read_bytes()
    assert len(value) == row["bytes"] and sha256(value).hexdigest() == row["sha256"]
helper = json.loads((OUT / "STDLIB_HELPER_RESULT.json").read_text())
assert helper["status"] == "PASS_BOUNDED_STDLIB_HELPERS_ONLY" and len(helper["cases"]) == 4
candidate = json.loads((OUT / "ROOT_SYNTHETIC_RELEASE_DISABLED_CANDIDATE.json").read_text())
assert candidate["execution_enabled"] is False and candidate["authorized_stages"] == candidate["authorized_invocations"] == []
assert candidate["root_authorization_reference"] == "" and candidate["existing_fabricated_fixture"] is None
plan = json.loads((OUT.parent / "ncnc_selected_checkpoint_metric_audit_v4_source_20261004/FABRICATED_QUALIFICATION_PLAN.json").read_text())
def write(name, value):
    path = OUT / name
    if path.exists(): raise RuntimeError("Fresh result already exists: " + str(path))
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
write("RESULT.json", {"schema": "ncnc-v4-bounded-offline-qualification-preparation-v1",
    "UTC": datetime.now(timezone.utc).isoformat(), "status": "PARTIAL_OFFLINE_HELPER_PASS_GPU_QUALIFICATION_NOT_RUN",
    "source_manifest_sha256": auth["source"]["manifest"]["sha256"],
    "local_helper_passed": 4, "local_helper_failed": 0, "local_helper_attempts": 1,
    "sealed_suite_status": "NOT_RUN_REMOTE_RUNTIME_AND_EXISTING_FIXTURE_REQUIRED",
    "sealed_cases": [{"case": name, "status": "NOT_ATTEMPTED", "work": {"planned": 120 if name in plan["full_core_cases"] else 0,
        "attempted": 0, "entered_original_scorer": 0, "returned": 0, "completed_validated": 0}}
        for name in plan["shared_contract_cases"] + plan["full_core_cases"]],
    "sealed_case_count": 22, "sealed_suite_planned_slots": 1920,
    "actual_scorer_work": helper["actual_scorer_work"], "new_training_updates": 0, "fixture_creation_calls": 0,
    "post_execution_source_metadata_pin_recheck": "PASS", "rechecked_descriptor_count": len(rows),
    "CUDA_invocation": False, "remote_access": False, "scientific_data_checkpoint_score_TEST_read": False,
    "enabled_release_issued": False, "actual_all25_audit_authorized": False, "actual_all25_audit_run": False,
    "deviation": "Root requested no CUDA invocation; full original-runtime qualification requires root-authenticated GPU77 dependencies. Four fabricated stdlib helper checks are separately labeled and cannot qualify ordinary admission.",
    "root_required_next_steps": ["authenticate/stage exact source/review metadata and recheck original runtime dependencies",
        "authenticate existing fabricated fixture bytes and bind fresh owned-copy manifest/descriptor without rebuilding",
        "perform dispatch resource/fresh-output recheck and issue separate root fabricated-only enabled release",
        "run exact22-case fabricated qualification once and preserve every actual count/failure"]})
write("EXECUTION_LOG.json", {"runs": [
    {"shell_command": "python3 -B postsubmission_research_20260930/ncnc_v4_offline_qualification_20261004_v1/prepare_local.py", "exit_code": 0,
     "result": "SOURCE_METADATA_PINS_AUTHENTICATED_DISABLED_CANDIDATE_PREPARED", "payload_count": 22, "payload_bytes": 308604},
    {"shell_command": "python3 -B postsubmission_research_20260930/ncnc_v4_offline_qualification_20261004_v1/stdlib_fabricated_checks.py", "exit_code": 0,
     "result": helper["status"], "local_helper_cases": 4, "retry": False}],
    "working_directory": str(OUT.parent.parent), "all_scorer_calls": 0, "all_training_updates": 0,
    "execution_failures": [], "GPU77_candidate_executed": False,
    "local_runtime": {"executable": helper["actual_python_executable"], "version": helper["actual_python_version"],
        "interpreter_binary_sha256": None, "reason_binary_pin_uncollected": "Executable lies outside bounded project filesystem read scope; no original-runtime qualification claimed."}})
files = []
for path in sorted(OUT.iterdir()):
    if path.is_file() and path.name not in ("MANIFEST.json", "SEAL.json"):
        value = path.read_bytes()
        files.append({"path": path.name, "bytes": len(value), "sha256": sha256(value).hexdigest()})
write("MANIFEST.json", {"schema": "ncnc-v4-bounded-offline-qualification-preparation-manifest-v1",
    "status": "PARTIAL_OFFLINE_HELPER_PASS_GPU_QUALIFICATION_NOT_RUN", "files": files,
    "payload_count": len(files), "payload_bytes": sum(row["bytes"] for row in files),
    "source_manifest_sha256": auth["source"]["manifest"]["sha256"], "execution_authorized": False})
value = (OUT / "MANIFEST.json").read_bytes()
write("SEAL.json", {"schema": "ncnc-v4-bounded-offline-qualification-preparation-seal-v1",
    "status": "SEALED_BOUNDED_RESULT_NOT_GPU_QUALIFICATION_OR_EXECUTION_RELEASE",
    "manifest": {"path": "MANIFEST.json", "bytes": len(value), "sha256": sha256(value).hexdigest()},
    "payload_count": len(files), "payload_bytes": sum(row["bytes"] for row in files),
    "sealed_qualification_executed": False, "execution_enabled": False})
print(json.dumps({"status": "SEALED_BOUNDED_RESULT", "source_metadata_recheck": "PASS", "helper_checks_PASS": 4,
    "sealed22_cases_executed": 0, "scorer_calls": 0, "manifest_sha256": sha256(value).hexdigest()}))
