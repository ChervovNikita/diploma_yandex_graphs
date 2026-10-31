"""Separately supervised validation-only target export from exact public storage.

Public provider pickle loading materializes all stored labels. Only training
and requested official validation target IDs are indexed. This is controlled
scope, not cryptographic isolation of test labels.
"""
import sys
import os
# Discard script/cwd/PYTHONPATH before stdlib or phase-package imports.
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bootstrap(request_path):
    """Validate the request-bound guard before the first phase-package import."""
    request_path = Path(request_path)
    request_value = json.loads(request_path.read_text())
    manifest_path = HERE / "SOURCE_MANIFEST_v1.json"
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != request_value.get("source_manifest_sha256"):
        raise ValueError("Bootstrap source manifest differs from the exact root request")
    manifest = json.loads(raw)
    guards = [r for r in manifest["files"] if r["relative_path"] == "import_guard.py"]
    guard_path = HERE / "import_guard.py"
    if guard_path.is_symlink() or len(guards) != 1 or hashlib.sha256(guard_path.read_bytes()).hexdigest() != guards[0]["sha256"]:
        raise ValueError("Bootstrap import guard source changed")
    spec = importlib.util.spec_from_file_location("_shared_message_predictive_guard_v2", guard_path)
    guard = importlib.util.module_from_spec(spec)
    exec(compile(guard_path.read_bytes(), str(guard_path), "exec", dont_inherit=True), guard.__dict__)
    return guard, *guard.install(request_value["source_manifest_sha256"])


def load_validation(ref, task, export, admitted, binding_ref):
    from shared_message_predictive_v2.contract import (PHASE, audit_runtime_import_receipt, checked,
        completed, prescribed)
    import torch
    if checked(ref["bundle"]).parent != prescribed(admitted, "targets", task=task) or checked(ref["receipt"]) != prescribed(admitted, "targets", task=task) / "TARGET_ADMISSION.json":
        raise ValueError("Alternative validation evidence location refused")
    receipt = json.loads(checked(ref["receipt"]).read_text())
    chain = completed(PHASE / ref["supervisor_directory"], ref["outer_start"])
    if (Path(chain["argv"][1]).resolve() != HERE / "labels.py" or chain["gpu_uuid"] != admitted["gpu_uuid"]
            or receipt.get("schema") != "gnnm-shared-message-validation-target-admission-v2"
            or receipt.get("task") != task or receipt.get("split") != "valid"
            or receipt.get("binding") != binding_ref or receipt.get("bundle") != ref["bundle"]
            or receipt.get("indexed_target_splits") != ["train", "valid"]
            or receipt.get("unrequested_target_values_indexed_or_summarized") is not False):
        raise ValueError("Separately supervised exact validation exporter receipt required")
    if receipt.get("live") != chain["live"]:
        raise ValueError("Validation exporter source-bound live/outer chain differs")
    audit_runtime_import_receipt(receipt)
    value = torch.load(checked(ref["bundle"]), map_location="cpu", weights_only=True)
    if set(value) != {"schema", "task", "split", "node_ids", "targets"} or value["schema"] != "gnnm-shared-message-official-validation-targets-v2" or value["split"] != "valid" or value["task"] != task:
        raise ValueError("Validation bundle contains unprescribed fields or split")
    ids, targets = export["official_split_node_ids"]["valid"], value["targets"]
    if (not torch.equal(ids, value["node_ids"]) or targets.dtype != torch.long or targets.shape != ids.shape
            or targets.device.type != "cpu" or targets.numel() == 0 or targets.min() < 0
            or targets.max() >= export["declared_official_class_count"]):
        raise ValueError("Aligned complete official validation target values required")
    return targets


def export_validation(admitted, binding_ref, request, context):
    from shared_message_predictive_v2.contract import (PHASE, Budget, checked, hash_report, manifest,
        prescribed, reference, runtime_import_receipt, sha, within, write)
    from shared_message_predictive_v2.core import configure, load_inputs
    import torch
    from propagation_cost_engineering_v1.data_contract import _tensor_record
    task = request["task"]
    budget = Budget(admitted, context)
    configure(admitted)
    export, lineage, metadata, directory = load_inputs(admitted, task)
    storage_path = checked(admitted["tasks"][task]["provider_storage"])
    if request["provider_storage"] != admitted["tasks"][task]["provider_storage"]:
        raise ValueError("Provider storage receipt differs from exact root binding")
    storage = json.loads(storage_path.read_text())
    field = request["provider_storage_inventory_field"]
    if field not in ("processed", "original_provider_files") or field not in storage:
        raise ValueError("Exact provider processed inventory field required; no fallback guess")
    processed = checked(request["processed_public_file"])
    if "/processed/" not in request["processed_public_file"]["path"]:
        raise ValueError("Only declared completed public processed provider bytes allowed")
    matches = [r for r in storage[field] if r["path"] == str(processed.relative_to(PHASE))]
    if len(matches) != 1 or matches[0]["sha256"] != sha(processed) or matches[0]["bytes"] != processed.stat().st_size:
        raise ValueError("Processed provider bytes are absent from audited inventory")
    output = prescribed(admitted, "targets", task=task)
    if within(PHASE / request["output_root"], HERE) != output:
        raise ValueError("Unprescribed validation export destination")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", {"binding": binding_ref, "request": request, "live": context,
          "supervisor_directory": context["supervisor_relative_to_project"],
          "all_public_targets_materialized_by_torch_load": True, "test_target_indexing_authorized": False,
          "public_pickle_trust_scope": "exact hashed completed named provider; not arbitrary checkpoint"})
    # Exact PyG provider pickle is intentionally trusted only under this request.
    payload = torch.load(processed, map_location="cpu", weights_only=False)
    if not isinstance(payload, tuple) or len(payload) != request["expected_processed_tuple_length"] or len(payload) not in (2, 3):
        raise ValueError("Unqualified public provider tuple interface")
    graph = payload[0]
    if len(payload) == 3:
        if not isinstance(graph, dict):
            raise ValueError("Declared PyG dict3 interface differs")
        get = graph.__getitem__
    else:
        get = lambda name: getattr(graph, name)
    for name in ("x", "edge_index"):
        original = get(name)
        if original.device.type != "cpu" or _tensor_record(original.contiguous()) != _tensor_record(export[name]):
            raise ValueError("Public graph identity differs from complete training-only export")
    train = export["official_split_node_ids"]["train"]
    if not torch.equal(get("y").index_select(0, train).reshape(-1), export["training_targets"].reshape(-1)):
        raise ValueError("Public training-label identity differs")
    ids = export["official_split_node_ids"]["valid"]
    targets = get("y").index_select(0, ids).reshape(-1)
    if (not torch.isfinite(targets).all() or not (targets == targets.to(torch.long)).all()
            or targets.numel() == 0 or targets.min() < 0 or targets.max() >= export["declared_official_class_count"]):
        raise ValueError("Requested complete validation target values invalid")
    targets = targets.to(torch.long).contiguous()
    from shared_message_predictive_v2.core import save_tensor
    bundle = save_tensor(output / "validation_targets.pt", {"schema": "gnnm-shared-message-official-validation-targets-v2",
        "task": task, "split": "valid", "node_ids": ids, "targets": targets}, budget)
    if sha(processed) != request["processed_public_file"]["sha256"]:
        raise ValueError("Provider changed during validation admission")
    write(output / "TARGET_ADMISSION.json", {"schema": "gnnm-shared-message-validation-target-admission-v2",
          "binding": binding_ref, "task": task, "split": "valid", "indexed_target_splits": ["train", "valid"],
          "bundle": bundle, "provider_storage": reference(storage_path), "processed_public_file": request["processed_public_file"],
          "provider_storage_inventory_field": field, "processed_tuple_length": len(payload), "input_lineage": lineage,
          "validation_target_fingerprint": _tensor_record(targets), "public_storage_all_targets_materialized": True,
          "unrequested_target_values_indexed_or_summarized": False, "hash_work": hash_report(), "live": context,
          "qualification": "pending successful bounded outer terminal, completed supervisor and root integrity receipt",
          **runtime_import_receipt()})
    manifest(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--supervisor", type=Path, required=True)
    args = p.parse_args()
    bootstrap(args.binding)
    from shared_message_predictive_v2.contract import (TASKS, binding, digest, live, prescribed,
        runtime_before_import, within, write)
    admitted, binding_ref = binding(args.binding)
    request = json.loads(within(args.request, HERE).read_text())
    if (request.get("schema") != "gnnm-shared-message-root-validation-request-v2"
            or request.get("root_authorized") is not True or request.get("binding") != binding_ref
            or request.get("source_manifest_sha256") != admitted["source_manifest_sha256"]
            or request.get("task") not in TASKS or request.get("split") != "valid"
            or request.get("test_target_access_authorized") is not False):
        raise ValueError("Only separately root-approved validation export exists; test always refused")
    if digest({k: v for k, v in request.items() if k != "binding"}) != admitted["tasks"][request["task"]]["validation_request_payload_sha256"]:
        raise ValueError("Validation request differs from root-frozen payload")
    runtime_before_import(admitted)
    context = live(args.supervisor, Path(__file__), args.request, admitted)
    if context["gpu_uuid"] != admitted["gpu_uuid"]:
        raise ValueError("Validation exporter wrong live GPU allocation")
    output = prescribed(admitted, "targets", task=request["task"])
    try:
        export_validation(admitted, binding_ref, request, context)
    except BaseException as error:
        if output.exists() and not (output / "FAILURE.json").exists():
            write(output / "FAILURE.json", {"type": type(error).__name__, "message": str(error), "partial_preserved": True, "resume_retry_or_promotion_authorized": False})
        raise


if __name__ == "__main__":
    main()
