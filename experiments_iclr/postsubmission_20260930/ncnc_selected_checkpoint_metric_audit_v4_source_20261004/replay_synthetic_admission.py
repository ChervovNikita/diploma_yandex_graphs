"""Separate stdlib admission for fabricated qualification; no study input access."""
from pathlib import Path
import json
import os
from replay_gate import (HERE, DRIVER_SHA, DESIGN_SHA, PROTOTYPE_SHA, RESOURCE_SHA,
                         DATA_SHA, RUNTIME_SHA, require, canonical, bound, file_sha,
                         verify_manifest, fresh_output_gate, runtime_and_data_custody)


def admission_context(release_path, output, *, expected_release_sha256=None):
    release_path = canonical(str(Path(release_path).expanduser()))
    require(expected_release_sha256 is None or file_sha(release_path) == expected_release_sha256, "Admitted synthetic root release bytes changed")
    release = json.loads(release_path.read_text())
    require(release.get("schema") == "ncnc-selected-checkpoint-metric-audit-synthetic-root-release-v4" and release.get("execution_enabled") is True and release.get("root_authorization_reference"), "Synthetic qualification lacks separate root authorization")
    require(release["authorized_stages"] == ["synthetic_checkpoint_metric_qualification"], "Only fabricated qualification may be authorized here")
    require(not any(k in release for k in ("family_lock", "unit_custody", "data_authority", "dataset_root", "ordinary_runtime_synthetic_qualification")), "Synthetic entry cannot accept study inputs or require its own qualification")
    from audit_contract import PROFILE_CONTRACT
    require(release["profile_contract"] == PROFILE_CONTRACT, "Exact successor synthetic profile required")
    sidecar_sha = file_sha(HERE / "MANIFEST.json")
    require(release["sidecar_manifest_sha256"] == sidecar_sha, "Synthetic release targets another sidecar")
    verify_manifest(HERE, sidecar_sha)
    paths = {k: canonical(release[k]) for k in ("driver_root", "design_root", "prototype_root", "resource_root")}
    for key, expected in (("driver_root", DRIVER_SHA), ("design_root", DESIGN_SHA), ("prototype_root", PROTOTYPE_SHA), ("resource_root", RESOURCE_SHA)):
        require(release[key.replace("_root", "_manifest_sha256")] == expected, "Synthetic source identity differs")
        verify_manifest(paths[key], expected)
    runtime = bound(release["runtime_authority"], decode=True)
    require(release["runtime_authority"]["sha256"] == RUNTIME_SHA and runtime["schema"] == "ncnc-collab-predictive-runtime-authority-v1" and runtime["ordinary_host_execution"] is True and runtime["deterministic_algorithms"] is False and runtime["TF32"] is False and runtime["mixed_precision"] is False, "Original ordinary synthetic runtime differs")
    authority = json.loads((HERE / "FABRICATED_AUTHORITY.json").read_text())
    require(authority["schema"] == "ncnc-selected-state-replay-fabricated-authority-v1" and authority["fabricated_inputs_only"] is True and authority["files"] == {} and authority["test_file_opened"] is False, "Synthetic authority is not fabricated-only")
    original_identity = release["original_identity"]
    require(set(original_identity) == {"driver_manifest_sha256", "design_manifest_sha256", "prototype_manifest_sha256", "resource_manifest_sha256", "data_authority_sha256", "runtime_authority_sha256", "family_id", "family_lock_output_directory"}, "Original identity fields differ")
    for key, expected in (("driver_manifest_sha256", DRIVER_SHA), ("design_manifest_sha256", DESIGN_SHA), ("prototype_manifest_sha256", PROTOTYPE_SHA), ("resource_manifest_sha256", RESOURCE_SHA), ("data_authority_sha256", DATA_SHA), ("runtime_authority_sha256", RUNTIME_SHA)):
        require(original_identity[key] == expected, "Original identity pin differs")
    review = bound(release["independent_source_review"], decode=True)
    require(review["schema"] == "ncnc-selected-checkpoint-metric-audit-source-review-v4" and review["status"] == "PASS" and review["sidecar_manifest_sha256"] == sidecar_sha and review["identity"] == original_identity, "Exact successor independent source review is required")
    require(review["reviewed_claim"] == "all25_checkpoint_metric_consistency_strict_True_observed_repeat_bytes" and review["all25_public_endpoint_reviewed"] is True and review["execution_authorized"] is False, "Exact successor source and public endpoint review required")
    output = canonical(str(Path(output).expanduser()))
    invocation = {"stage": "synthetic_checkpoint_metric_qualification", "output_directory": str(output), "cuda_visible_devices": release["cuda_visible_devices"]}
    require(release["authorized_invocations"] == [invocation] and release["cuda_visible_devices"] in authority["physical_GPU_UUIDs"] and os.environ.get("CUDA_VISIBLE_DEVICES") == release["cuda_visible_devices"], "Exactly one synthetic output/GPU invocation required")
    require(release["minimum_GPU_free_MiB"] == 16384 and release["minimum_host_MemAvailable_bytes"] == 16 * 1024**3, "Reviewed synthetic resource floor differs")
    bound(release["existing_fabricated_fixture"], decode=True)
    return {"release": release, "admission_release": release, "release_path": release_path, "release_sha256": file_sha(release_path), "sidecar_manifest_sha256": sidecar_sha, "paths": paths, "plan": json.loads((paths["design_root"] / "PILOT_PLAN.json").read_text()), "authority": authority, "runtime": runtime, "identity": original_identity, "original_identity": original_identity, "output": output, "admission_output": output, "synthetic_only": True}


def preflight(release_path, output):
    context = admission_context(release_path, output)
    fresh_output_gate(context)
    return context


def admitted_input_custody(context):
    repeated = admission_context(context["release_path"], context["admission_output"], expected_release_sha256=context["release_sha256"])
    require(repeated["release"] == context["admission_release"], "Admitted synthetic release context changed")
    for key in ("paths", "plan", "authority", "runtime", "original_identity", "sidecar_manifest_sha256"):
        require(repeated[key] == context[key], "Admitted synthetic source/runtime/review context changed")
    runtime_and_data_custody(context)
    return {"source_release_authority_review_fabricated_tensor_runtime_custody": "PASS", "study_data_accessed": False, "output_freshness_rechecked": False}
