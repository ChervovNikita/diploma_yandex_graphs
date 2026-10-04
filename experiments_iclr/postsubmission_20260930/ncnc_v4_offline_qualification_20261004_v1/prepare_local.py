"""Bounded source/metadata authentication and disabled-command preparation only."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import os
import platform
import shlex
import sys

OUT = Path(__file__).resolve().parent
P = OUT.parent
SOURCE = P / "ncnc_selected_checkpoint_metric_audit_v4_source_20261004"
REVIEW = P / "ncnc_selected_metric_audit_v4_fresh_source_review_20261004_v1"
SOURCE_SHA = "fd59e048a53695c945c6cfb613a77ce0e4e21c2631ce56085348c3941f5ec6d6"
REVIEW_SHA = "5010ccb9dc39f3e9af69d608fd7c4ada4e9a551f9fa55c25a27e61b329699e9f"
REMOTE = Path("/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930")


def pin(path):
    value = path.read_bytes()
    return {"path": str(path), "bytes": len(value), "sha256": sha256(value).hexdigest()}


def write(name, value):
    target = OUT / name
    if target.exists():
        raise RuntimeError("Fresh artifact already exists: " + str(target))
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def packet(root, expected):
    manifest_pin = pin(root / "MANIFEST.json")
    if manifest_pin["sha256"] != expected:
        raise RuntimeError("Manifest differs: " + str(root))
    manifest = json.loads((root / "MANIFEST.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = root / row["path"]
        if path.resolve() != path or not path.is_relative_to(root):
            raise RuntimeError("Source path escaped: " + str(path))
        actual = pin(path)
        if actual["bytes"] != row.get("bytes", row.get("size")) or actual["sha256"] != row["sha256"]:
            raise RuntimeError("Source payload differs: " + str(path))
        rows.append(actual)
    return {"manifest": manifest_pin, "files": rows, "payload_count": len(rows),
            "payload_bytes": sum(row["bytes"] for row in rows)}


source = packet(SOURCE, SOURCE_SHA)
review = packet(REVIEW, REVIEW_SHA)
seal = json.loads((SOURCE / "SEAL.json").read_text())
assert seal["manifest"] == {**source["manifest"], "path": "MANIFEST.json"}
assert source["payload_count"] == 22 and source["payload_bytes"] == 308604
review_seal = json.loads((REVIEW / "SEAL.json").read_text())
assert review_seal["manifest"] == {**review["manifest"], "path": "MANIFEST.json"}
receipt = json.loads((REVIEW / "REVIEW_RECEIPT.json").read_text())
assert receipt["sidecar_manifest_sha256"] == SOURCE_SHA and receipt["status"] == "PASS"
assert receipt["execution_authorized"] is False
bindings = json.loads((SOURCE / "SOURCE_BINDINGS.json").read_text())
originals = {}
for role, spec in bindings["original_manifests"].items():
    originals[role] = packet(P / spec["local_root"], spec["manifest_sha256"])
runtime_path = P / "graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json"
runtime_pin = pin(runtime_path)
assert runtime_pin["sha256"] == "e63f602baa8a91e129fb4a0c0debd6ec282e7cd132be87eb66cd8c5cf0b7c882"
runtime = json.loads(runtime_path.read_text())
predecessor_path = P / "graph_ncNC_complete_family_selected_state_replay_preparation_20261004_v2"
predecessor_manifest = json.loads((predecessor_path / "MANIFEST.json").read_text())
assert pin(predecessor_path / "MANIFEST.json")["sha256"] == "0dad46a766c82b689def0b34a411e499d9940c07e68e136ac7fcaa57f5a31c6b"
lineage_files = []
for name in ("replay_synthetic.py", "replay_synthetic_fixtures.py"):
    expected = next(row for row in predecessor_manifest["files"] if row["path"] == name)
    actual = pin(predecessor_path / name)
    assert actual["bytes"] == expected.get("bytes", expected.get("size")) and actual["sha256"] == expected["sha256"]
    lineage_files.append(actual)
old_release_path = P / "ncnc_selected_replay_synthetic_execution_root_20261004_v1/ROOT_SYNTHETIC_RELEASE.json"
old_release = json.loads(old_release_path.read_text())
assert old_release["sidecar_manifest_sha256"] == "0dad46a766c82b689def0b34a411e499d9940c07e68e136ac7fcaa57f5a31c6b"
fixture_remote = Path(old_release["authorized_invocations"][0]["output_directory"]) / "fabricated_reference"
fixture_local_candidate = P / fixture_remote.relative_to(REMOTE)
plan = json.loads((SOURCE / "FABRICATED_QUALIFICATION_PLAN.json").read_text())
slots = json.loads((SOURCE / "SLOT_PLAN.json").read_text())["slots"]
assert len(slots) == 120 and plan["case_count"] == 22 and plan["whole_suite_predeclared_slots"] == 1920
assert len(plan["full_core_cases"]) == 16 and len(plan["shared_contract_cases"]) == 6
write("AUTHENTICATION.json", {"UTC": datetime.now(timezone.utc).isoformat(), "status": "PASS_LOCAL_SOURCE_METADATA_BYTES_ONLY",
    "source": source, "source_seal": pin(SOURCE / "SEAL.json"), "review": review,
    "review_seal": pin(REVIEW / "SEAL.json"), "original_source_packets": originals,
    "runtime_authority_metadata": runtime_pin, "runtime_actual_binary_authentication": "NOT_PERFORMED_REMOTE_DEPENDENCY",
    "fixture_builder_lineage_source": lineage_files, "old_release_metadata": pin(old_release_path),
    "scientific_data_checkpoint_score_read": False, "remote_access": False})
write("BOUNDED_RATIONALE.json", {"authority": "Root delegated fabricated local qualification and source/lineage work; later root instruction requires no CUDA invocation and a disabled remote candidate.",
    "planned_local_execution": "stdlib_fabricated_checks.py imports only authenticated v4 gate/contract/run/numeric module definitions and exercises SlotLedger, private_receipt, terminal precedence and null-output helpers using fabricated Python inputs.",
    "why_useful": "Direct local helper behavior checks cover literal late-fault ledger tuples, malformed receipt preservation and failure suppression without Torch/CUDA, original scorer calls or model payloads.",
    "local_helper_checks_are_22_case_qualification": False, "source_review_is_execution_authority": False,
    "scope_exclusions": ["remote access", "CUDA invocation", "scientific data/checkpoint/score access", "fixture rebuilding", "actual all25 audit", "old failed tree mutation"],
    "helper_command": [sys.executable, "-B", str(OUT / "stdlib_fabricated_checks.py")],
    "local_runtime": {"executable": sys.executable, "resolved_executable": str(Path(sys.executable).resolve()), "version": sys.version, "platform": platform.platform()},
    "local_runtime_is_original_admitted_runtime": False,
    "local_environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")},
    "exact_code_pin": pin(OUT / "stdlib_fabricated_checks.py")})
write("FIXTURE_LINEAGE.json", {"status": "REMOTE_FIXTURE_BYTES_NOT_AUTHENTICATED",
    "existing_fixture_expected_remote_root": str(fixture_remote), "source_basis": lineage_files,
    "old_release_metadata": pin(old_release_path), "local_corresponding_root": str(fixture_local_candidate),
    "local_corresponding_root_exists": fixture_local_candidate.exists(),
    "local_descriptor_found": False, "fixture_payload_opened": False,
    "historical_fixture_creation_metadata": {"tiny_engineering_updates": 40, "reference_scorer_calls": 44, "scientific_updates": 0,
        "metadata_100_epochs_not_executed": True, "basis": "authenticated v2 builder source; no old outcome opened"},
    "v4_new_fixture_updates": 0, "v4_fixture_creation_scorer_calls": 0,
    "next_root_action": "Authenticate existing fabricated_reference bytes on GPU77 without deserialization; copy the exact existing bytes into a fresh owned root, bind an all-file manifest and v3 fixture descriptor, leave the original FAILED v2 tree immutable. No builder invocation or reconstruction is admitted."})

remote_out = REMOTE / OUT.name
invocation = {"stage": "synthetic_checkpoint_metric_qualification", "output_directory": str(remote_out / "qualification/run01"),
              "cuda_visible_devices": old_release["cuda_visible_devices"]}
candidate = {"schema": "ncnc-selected-checkpoint-metric-audit-synthetic-root-release-v4", "execution_enabled": False,
    "root_authorization_reference": "", "authorized_stages": [], "authorized_invocations": [],
    "candidate_invocation_not_authorized": invocation, "sidecar_manifest_sha256": SOURCE_SHA,
    "cuda_visible_devices": invocation["cuda_visible_devices"], "minimum_GPU_free_MiB": 16384,
    "minimum_host_MemAvailable_bytes": 17179869184, "original_identity": receipt["identity"],
    "runtime_authority": {**runtime_pin, "path": str(REMOTE / runtime_path.relative_to(P))},
    "independent_source_review": {**pin(REVIEW / "REVIEW_RECEIPT.json"), "path": str(REMOTE / REVIEW.name / "REVIEW_RECEIPT.json")},
    "existing_fabricated_fixture": None, "profile_contract": {
        "labels": ["False_anchor", "True1", "True2"], "strict_candidates": ["True1", "True2"], "cublas_workspace_config": ":4096:8",
        "warn_only": False, "dtype": "torch.float32", "TF32": False, "autocast_cuda": False, "autocast_cpu": False,
        "float32_matmul_precision": "highest", "cudnn_benchmark": False, "cudnn_deterministic": False,
        "anchor_metric_acceptance_authority": False, "engineering_128eps_acceptance_authority": False},
    "unresolved_before_root_enabled_release": ["existing fabricated fixture all-file manifest and descriptor authenticated on GPU77",
        "remote source/review/runtime byte custody", "fresh output admission", "dispatch GPU free >=16384 MiB and MemAvailable >=17179869184", "root authorization"]}
for role, spec in bindings["original_manifests"].items():
    candidate[role + "_root"] = str(REMOTE / spec["local_root"])
    candidate[role + "_manifest_sha256"] = spec["manifest_sha256"]
write("ROOT_SYNTHETIC_RELEASE_DISABLED_CANDIDATE.json", candidate)
argv = [runtime["interpreter_path"], "-B", str(REMOTE / SOURCE.name / "replay_synthetic.py"), "--root-release",
        str(remote_out / "ROOT_SYNTHETIC_RELEASE_DISABLED_CANDIDATE.json"), "--output", invocation["output_directory"]]
env = {"CUDA_VISIBLE_DEVICES": invocation["cuda_visible_devices"], "PYTHONPATH": runtime["project_PYTHONPATH"],
       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
command = shlex.join(["env", *[key + "=" + value for key, value in env.items()], *argv])
(OUT / "GPU77_COMMAND_DISABLED.txt").write_text("NOT EXECUTED. This exact candidate command targets a disabled release and must fail admission. Root must issue a separate authenticated enabled release and repin the command before a fabricated-only GPU77 run.\n\n" + command + "\n")
write("INVOCATION_AND_DEPENDENCIES.json", {"execution_performed": False, "argv_candidate": argv, "environment_candidate": env,
    "interpreter_expected_sha256": runtime["interpreter_sha256"], "runtime_authority": runtime_pin,
    "distribution_versions": runtime["distribution_versions"], "cuda_version": runtime["cuda_version"],
    "runtime_source_pins": runtime["runtime_source_pins"], "runtime_binary_files": runtime["runtime_binary_files"],
    "negative_sampler": runtime["negative_sampler"], "ogb_evaluator": json.loads((SOURCE / "FABRICATED_AUTHORITY.json").read_text())["ogb_evaluator"],
    "GPU_name": runtime["gpu_name"], "GPU_total_memory_bytes": runtime["gpu_total_memory_bytes"],
    "strict_profile": candidate["profile_contract"], "command_is_portable_cpu_qualification": False,
    "remote_only_required_dependencies": ["admitted Python3.12 bytes/distributions", "CUDA12.6 and physical A100 UUID", "pinned sparse/scatter extensions, PyG/sampler and OGB evaluator", "existing fabricated fixture payload bytes plus new authenticated descriptor"]})
stage = []
for role, root, evidence in (("v4_source", SOURCE, source), ("v4_review", REVIEW, review)):
    for actual in [evidence["manifest"], *evidence["files"], pin(root / "SEAL.json")]:
        stage.append({**actual, "role": role, "remote_path": str(REMOTE / Path(actual["path"]).relative_to(P)), "remote_present_now": "UNVERIFIED_NO_REMOTE_ACCESS"})
write("STAGING_INVENTORY.json", {"status": "EXACT_LOCAL_PINS_REMOTE_PRESENCE_UNVERIFIED", "candidate_stage_files": stage,
    "v4_source_file_count": 24, "v4_review_file_count": 5,
    "remote_dependency_recheck_only": {"original_source_packets": originals, "runtime_authority": runtime_pin},
    "local_pin_authentication_is_remote_authentication": False,
    "bounded_staging_note": "Only source/metadata packets may be staged. Existing original source/runtime dependencies must be rechecked; current remote missing/present state is unknown. Fabricated fixture descriptor is unresolved. No scientific input, checkpoint, score or actual family lock is part of this staging inventory."})
write("SOURCE_READING_SCOPE.json", {"scope": "Independent bounded harness/dependency/lineage inspection, not a replacement for the fresh source review.",
    "complete_v4_python_source_text_read": [row for row in source["files"] if row["path"].endswith(".py")],
    "source_metadata_read": ["README.md", "MANIFEST.json", "SEAL.json", "ROOT_RELEASE_SCHEMA.md", "FABRICATED_AUTHORITY.json", "FABRICATED_QUALIFICATION_PLAN.json"],
    "source_metadata_json_parsed_for_bindings_counts_only": ["SOURCE_BINDINGS.json", "SLOT_PLAN.json"],
    "other_v4_payloads": "Hash/length only; no claim of complete semantic reading.",
    "independent_review_text_read": pin(REVIEW / "REVIEW.txt"), "independent_review_receipt_read": pin(REVIEW / "REVIEW_RECEIPT.json"),
    "review_SOURCE_EVIDENCE_scope": "Hash/length only; initial combined display was truncated and no complete text-read claim is made.",
    "additional_source_text_read": [
        {**pin(P / "graph_ncNC_collab_predictive_driver_preparation_20261003_v1/pilot_model.py"), "scope": "complete"},
        {**pin(P / "graph_ncNC_collab_predictive_driver_preparation_20261003_v1/pilot_evaluate.py"), "scope": "complete"},
        {**pin(P / "graph_ncNC_collab_predictive_driver_preparation_20261003_v1/pilot_common.py"), "scope": "lines1-155"},
        {**lineage_files[0], "scope": "lines55-107 plus fixture/import/main text searches"},
        {**lineage_files[1], "scope": "complete"}],
    "metadata_only_read": [runtime_pin, pin(old_release_path)],
    "actual_fixture_and_scientific_payloads_read": False, "sealed_source_mutations": False})
print(json.dumps({"status": "SOURCE_METADATA_PINS_AUTHENTICATED_DISABLED_CANDIDATE_PREPARED",
    "v4_payload_count": source["payload_count"], "v4_payload_bytes": source["payload_bytes"],
    "qualification_cases_planned": 22, "qualification_slots_planned": 1920,
    "local_existing_fixture_present": fixture_local_candidate.exists(), "remote_execution": False}))
