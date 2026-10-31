"""Read-only, root-gated caller of the existing complete-family consumer.

This separate prerequisite hashes scientific artifacts through the established
consumer. It never deserializes tensors and is not the64-input diagnostic.
"""
import os
import sys
sys.dont_write_bytecode = True
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
SCIENCE = {"path": "shared_message_predictive_v3/SOURCE_MANIFEST_v1.json", "sha256": "e4d1b4b7d96ae07e7588a854cb8a0d70e9af6612ac72d1df583c3c637c376c9b"}
FUNCTIONS = ["shared_message_predictive_v3.contract.binding", "shared_message_predictive_v3.core.validation_freeze", "shared_message_predictive_v3.core.audit_family_lock"]
CALLER_MAX_CAPS = {"whole_wall_seconds": 1800, "host_RSS_bytes": 2**31, "new_output_disk_bytes": 16 * 2**20, "GPU_count": 0}


def common(request):
    """Load only exact manifest-bound local stdlib source, without registration."""
    manifest_path = HERE / "SOURCE_MANIFEST_v1.json"
    manifest_raw = manifest_path.read_bytes()
    if request.get("source_manifest", {}).get("path") != str(manifest_path.relative_to(PHASE)) or hashlib.sha256(manifest_raw).hexdigest() != request["source_manifest"]["sha256"]:
        raise ValueError("Caller source manifest not bound to the request")
    manifest = json.loads(manifest_raw)
    path = HERE / "decision_analysis.py"
    rows = [r for r in manifest["files"] if r["relative_path"] == path.name]
    if path.is_symlink() or len(rows) != 1 or hashlib.sha256(path.read_bytes()).hexdigest() != rows[0]["sha256"]:
        raise ValueError("Exact local admission helpers required")
    namespace = {"__file__": str(path), "__name__": "_decision_v2_common"}
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), namespace)
    namespace["verify_plan_and_source"](request)
    return namespace


def bootstrap(request, c):
    """Augment only the caller boundary; original scientific policy is intact."""
    c["require"](request.get("scientific_source_manifest") == SCIENCE, "Exact scientific-v3 generation required")
    manifest_path = c["checked"](SCIENCE)
    manifest = json.loads(manifest_path.read_text())
    guard_path = PHASE / "shared_message_predictive_v3/import_guard.py"
    rows = [r for r in manifest["files"] if r["relative_path"] == "import_guard.py"]
    c["require"](len(rows) == 1 and c["sha"](guard_path) == rows[0]["sha256"], "Exact scientific import guard required")
    guard = {"__file__": str(guard_path), "__name__": "_complete30_consumer_guard_v2"}
    exec(compile(guard_path.read_bytes(), str(guard_path), "exec", dont_inherit=True), guard)
    original, tree = guard["verify_packet"](SCIENCE["sha256"])
    policy = dict(original)
    policy["importable_sha256"] = dict(original["importable_sha256"])
    caller_manifest = json.loads(c["checked"](request["source_manifest"]).read_text())
    for row in caller_manifest["files"]:
        if row["relative_path"].endswith(".py"):
            path = c["confined"](HERE / row["relative_path"], HERE)
            relative = str(path.relative_to(PHASE))
            c["require"](relative not in policy["importable_sha256"], "Caller augmentation must not override scientific source")
            policy["importable_sha256"][relative] = row["sha256"]
    guard["installed_path_only"]()
    guard["verify_generator_installation"](policy)
    guard["imported_modules"](policy)
    sys.meta_path.insert(0, guard["PhaseFinder"](policy))
    return guard, policy, tree


def receipts(guard, policy):
    rows = guard["imported_modules"](policy)
    guard["verify_imported_receipts"](rows, policy)
    if any(n == "torch" or n.startswith(("torch.", "torch_geometric.", "ogb.")) or n in {"torch_geometric", "ogb"} for n in sys.modules):
        raise ValueError("Read-only consumer imported a numerical/provider package")
    return rows


def verify_witness(consumer, admitted, freeze, family, request, c, budget):
    """Reuse successful consumer supervision; do not repeat its artifact audit."""
    witness_ref = consumer["completed_family_root_audit"]
    witness_path = c["checked"](witness_ref, budget, c["MAX_METADATA_BYTES"])
    witness = json.loads(witness_path.read_text())
    c["require"](witness.get("schema") == "gnnm-shared-message-complete30-consumer-result-v1" and witness.get("status") == "passed_existing_complete_family_consumer", "Actual established consumer result required")
    c["require"](witness.get("invoked_consumers") == FUNCTIONS and witness.get("scientific_source_manifest") == SCIENCE, "Completed consumer source/function generation differs")
    c["require"](c["same_ref"](witness.get("source_manifest"), request["source_manifest"]) and witness.get("tensor_deserialization_performed") is False and witness.get("Torch_imported") is False, "Witness caller/read-only identity differs")
    for key in ("binding", "validation_freeze", "family_lock"):
        c["require"](c["same_ref"](witness.get(key), consumer[key]), "Witness qualified a different scientific family")
    c["require"](witness.get("locked_rows_sha256") == family["locked_rows_sha256"] and witness.get("all30_fits_and_replays_qualified") is True and witness.get("test_targets_loaded") is False, "Witness family/qualification differs")
    caller_request = c["read_json"](witness["request"], budget)
    c["require"](caller_request.get("schema") == "gnnm-shared-message-complete30-consumer-request-v1" and caller_request.get("root_authorized") is True, "Authorized actual consumer invocation required")
    c["require"](c["same_ref"](caller_request.get("source_manifest"), request["source_manifest"]) and caller_request.get("scientific_source_manifest") == SCIENCE, "Consumer caller source differs")
    for key in ("binding", "validation_freeze", "family_lock"):
        c["require"](c["same_ref"](caller_request.get(key), consumer[key]), "Consumer request belongs to another family")
    output = c["confined"](PHASE / caller_request["output_relative_to_phase"], HERE)
    c["require"](witness_path == output / "CONSUMER_RESULT.json", "Consumer result not at its prescribed output")
    guard, policy, tree = bootstrap(caller_request, c)
    from shared_message_predictive_v3.contract import completed
    supervisor = c["confined"](PHASE / witness["live"]["supervisor_relative_to_project"])
    chain = completed(supervisor, witness["live"]["outer_start"])
    argv = [str(Path(json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())["repository"]) / ".venv/bin/python"), str(HERE / "family_consumer.py"), "--request", str(c["checked"](witness["request"], budget)), "--supervisor", str(supervisor)]
    c["require"](chain["argv"] == argv and chain["live"] == witness["live"] and chain["gpu_uuid"] == admitted["gpu_uuid"], "Unsuccessful or unrelated actual consumer supervision")
    terminal = c["read_json"](chain["outer_terminal"], budget)
    outer = c["read_json"](chain["outer_start"], budget)
    c["require"](outer["whole_cap_seconds"] == caller_request["caps"]["whole_wall_seconds"] and c["same_ref"](outer["root_request"], witness["request"]), "Consumer request/deadline/terminal linkage differs")
    approval = c["read_json"](caller_request["root_approval"], budget)
    c["require"](approval.get("schema") == "gnnm-shared-message-complete30-consumer-root-approval-v1" and approval.get("root_authorized") is True and approval.get("approved_request_payload_sha256") == c["digest"]({k: v for k, v in caller_request.items() if k != "root_approval"}), "Actual consumer root payload approval differs")
    c["require"](witness.get("scientific_import_tree") == tree and witness.get("caller_import_policy_sha256") == c["digest"](policy["importable_sha256"]), "Actual caller/scientific import boundary differs")
    guard["verify_imported_receipts"](witness["imported_phase_modules"], policy)
    c["require"]({"shared_message_predictive_v3.contract", "shared_message_predictive_v3.core"}.issubset({r["module"] for r in witness["imported_phase_modules"]}), "Actual scientific consumer import receipts missing")
    for row in witness["imported_phase_modules"]:
        c["require"](row.get("source_kind") != "pinned_PyG_generated_tmp_source", "Read-only caller must not instantiate PyG")
        c["checked"]({"path": row["relative_to_phase"], "sha256": row["sha256"]}, budget)
    artifact_ref = {"path": str((output / "ARTIFACT_MANIFEST.json").relative_to(PHASE)), "sha256": c["sha"](output / "ARTIFACT_MANIFEST.json", budget)}
    artifacts = c["read_json"](artifact_ref, budget)
    c["require"](artifacts.get("schema") == "gnnm-shared-message-complete30-consumer-artifacts-v1" and artifacts.get("files", {}).get("CONSUMER_RESULT.json") == {"sha256": witness_ref["sha256"], "bytes": witness_path.stat().st_size}, "Consumer result artifact binding differs")
    receipts(guard, policy)
    refs = [witness["request"], caller_request["root_approval"], caller_request["resource_approval"], caller_request["implementation_review"], chain["outer_start"], chain["outer_terminal"], artifact_ref]
    refs += [c["reference"](supervisor / name) for name in ("authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log")]
    return refs, policy["importable_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    args = parser.parse_args()
    request_path = Path(os.path.abspath(args.request))
    if request_path == HERE or not request_path.is_relative_to(HERE):
        raise ValueError("Consumer request must remain in this source packet")
    current = HERE
    for name in request_path.relative_to(HERE).parts:
        current /= name
        if current.is_symlink():
            raise ValueError("Consumer request symlinks forbidden")
    request = json.loads(request_path.read_text())
    if request.get("schema") != "gnnm-shared-message-complete30-consumer-request-v1" or request.get("root_authorized") is not True:
        raise ValueError("Separate root-approved read-only consumer request required")
    if request.get("scientific_source_manifest") != SCIENCE or request.get("invoked_consumers") != FUNCTIONS or request.get("scientific_artifact_byte_hashing_authorized") is not True or any(request.get(k) is not False for k in ("tensor_deserialization_authorized", "test_target_access_authorized", "GPU_operations_authorized", "new_learning_or_inference_authorized")):
        raise ValueError("Only the established read-only consumer is authorized")
    caps = request["caps"]
    if set(caps) != set(CALLER_MAX_CAPS) or any(type(caps[k]) is not int or not 0 < caps[k] <= CALLER_MAX_CAPS[k] for k in caps if k != "GPU_count") or caps["GPU_count"] != 0:
        raise ValueError("Separately bounded consumer allowance required")
    c = common(request)
    request_path = c["confined"](request_path, HERE)
    request_ref = c["reference"](request_path)
    review = c["read_json"](request["implementation_review"])
    c["require"](review.get("schema") == "gnnm-shared-message-decision-analysis-independent-review-v1" and review.get("status") == "approved_for_bounded_execution" and review.get("independent_of_implementation_author") is True and review.get("unresolved_blocking_issues") == [] and review.get("numerical_execution_performed") is False and c["same_ref"](review.get("source_manifest"), request["source_manifest"]), "Independent reviewed v2 consumer/diagnostic packet required")
    approval = c["read_json"](request["root_approval"])
    c["require"](approval.get("schema") == "gnnm-shared-message-complete30-consumer-root-approval-v1" and approval.get("root_authorized") is True and approval.get("approved_request_payload_sha256") == c["digest"]({k: v for k, v in request.items() if k != "root_approval"}), "Consumer root payload approval required")
    allocation = c["read_json"](request["resource_approval"])
    c["require"](allocation.get("schema") == "gnnm-shared-message-complete30-consumer-resource-approval-v1" and allocation.get("root_approved") is True and allocation.get("reserved_caps") == caps and allocation.get("separate_actual_wall_and_disk_charged") is True and allocation.get("frozen_training_allocation_extended") is False, "Separate consumer allocation reconciliation required")
    guard, policy, tree = bootstrap(request, c)
    from shared_message_predictive_v3.contract import binding, verify_outer_start
    from shared_message_predictive_v3.core import validation_freeze, audit_family_lock
    from propagation_cost_engineering_v1.contract import live_context
    supervisor = c["confined"](args.supervisor)
    outer_path = c["confined"](os.environ["GNNM_BOUND_START_JSON"])
    argv = json.loads((supervisor / "command.json").read_text())["argv"]
    expected = [argv[0], str(HERE / "family_consumer.py"), "--request", str(request_path), "--supervisor", str(supervisor)]
    c["require"](argv == expected, "Exact consumer child arguments required")
    outer_ref, outer = verify_outer_start(outer_path, request_path, supervisor, argv)
    c["require"](outer["whole_cap_seconds"] == caps["whole_wall_seconds"], "Consumer cap differs")
    admitted, binding_ref = binding(c["checked"](request["binding"]))
    c["require"](c["same_ref"](binding_ref, request["binding"]) and admitted["source_manifest_sha256"] == SCIENCE["sha256"], "Consumer admission generation differs")
    context = {**live_context(supervisor, Path(__file__)), "gpu_uuid": admitted["gpu_uuid"], "outer_start": outer_ref}
    output = c["confined"](PHASE / request["output_relative_to_phase"], HERE)
    c["require"](output.parent == HERE / "consumer_runs" and not output.exists(), "Exclusive consumer output required")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.fromisoformat(outer["start_UTC"])

    def cap():
        seconds = (datetime.now(timezone.utc) - started).total_seconds()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform.startswith("linux") else 1)
        c["require"](0 <= seconds < outer["normal_child_budget_seconds"] and peak <= caps["host_RSS_bytes"], "Consumer CPU wall/RSS limit exceeded")
        c["require"](sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= caps["new_output_disk_bytes"], "Consumer output cap exceeded")

    c["write_json"](output / "START.json", {"request": request_ref, "live": context, "read_only_existing_consumer": True})
    try:
        cap()
        freeze, freeze_ref = validation_freeze(c["checked"](request["validation_freeze"]), admitted, binding_ref)
        family_path = c["checked"](request["family_lock"])
        value = audit_family_lock(family_path, admitted, binding_ref, freeze_ref)
        cap()
        imports = receipts(guard, policy)
        for key in ("binding", "validation_freeze", "family_lock", "source_manifest", "scientific_source_manifest", "root_approval", "resource_approval", "implementation_review"):
            c["checked"](request[key])
        c["write_json"](output / "CONSUMER_RESULT.json", {"schema": "gnnm-shared-message-complete30-consumer-result-v1", "status": "passed_existing_complete_family_consumer", "request": request_ref, "source_manifest": request["source_manifest"], "scientific_source_manifest": SCIENCE, "invoked_consumers": FUNCTIONS, "binding": binding_ref, "validation_freeze": freeze_ref, "family_lock": c["reference"](family_path), "locked_rows_sha256": value["locked_rows_sha256"], "all30_fits_and_replays_qualified": True, "test_targets_loaded": False, "tensor_deserialization_performed": False, "Torch_imported": False, "scientific_import_tree": tree, "caller_import_policy_sha256": c["digest"](policy["importable_sha256"]), "imported_phase_modules": imports, "live": context, "external_completion_required": True})
        inventory = {p.name: {"sha256": c["sha"](p), "bytes": p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
        c["write_json"](output / "ARTIFACT_MANIFEST.json", {"schema": "gnnm-shared-message-complete30-consumer-artifacts-v1", "files": inventory})
        cap()
        print(json.dumps({"status": "passed_existing_consumer_pending_external_terminal", "output": str(output.relative_to(PHASE))}), flush=True)
    except BaseException as error:
        for name in ("CONSUMER_RESULT.json", "ARTIFACT_MANIFEST.json"):
            path = output / name
            if path.is_file() and not path.is_symlink():
                path.unlink()
        c["write_json"](output / "FAILURE.json", {"exception": type(error).__name__, "reason": str(error)[:500], "consumer_qualified": False})
        raise


if __name__ == "__main__":
    main()
