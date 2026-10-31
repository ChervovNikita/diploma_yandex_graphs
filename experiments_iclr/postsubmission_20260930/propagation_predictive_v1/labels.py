"""Separately approved split-target admission from exact public provider bytes.

Processed public storage materializes ALL labels. Only authorized split targets
are indexed here. This is scope discipline, not cryptographic label isolation.
"""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
entry_directory = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or Path.cwd()).resolve() != entry_directory]
sys.path.insert(0, str(entry_directory.parent))
from propagation_predictive_v1.contract import (HERE, PHASE, binding, checked, completed, digest, live, manifest,
                      Budget, hash_cache_report, reference, runtime_before_import, sha, target_directory, within, write)


def load_label_bundle(ref, split, task, export, admitted):
    """Never open a provider here; consume one explicitly exported split."""
    import torch
    p = checked(ref["bundle"])
    receipt_path = checked(ref["receipt"])
    receipt = json.loads(receipt_path.read_text())
    chain = completed(PHASE / ref["supervisor_directory"])
    if Path(chain["argv"][1]).resolve() != HERE / "labels.py":
        raise ValueError("Target exporter supervisor did not execute the sealed split entrypoint")
    if (receipt.get("schema") != "gnnm-propagation-admitted-split-targets-v1"
            or receipt.get("split") != split or receipt.get("task") != task
            or receipt.get("indexed_target_splits") != ["train", split]
            or receipt.get("binding_sha256") != admitted["binding_sha256"]
            or receipt.get("bundle") != ref["bundle"] or chain["gpu_uuid"] != admitted["gpu_uuid"]):
        raise ValueError("Unqualified split-target admission chain")
    value = torch.load(p, map_location="cpu", weights_only=True)
    if set(value) != {"schema", "task", "split", "node_ids", "targets"}:
        raise ValueError("Split target export contains forbidden extra fields")
    if value["schema"] != "gnnm-propagation-single-official-split-targets-v1" or value["split"] != split or value["task"] != task:
        raise ValueError("Wrong requested split-target object")
    ids = export["official_split_node_ids"][split]
    y = value["targets"]
    if (not torch.equal(ids, value["node_ids"]) or y.dtype != torch.long
            or y.shape != ids.shape or not torch.isfinite(y).all()
            or y.min() < 0 or y.max() >= export["declared_official_class_count"]):
        raise ValueError("Official aligned split target values failed admission")
    return y


def export_labels(admitted, binding_ref, request, output, context):
    """Provider pickle loading is authorized only after all stdlib checks."""
    import torch
    from propagation_cost_engineering_v1.data_contract import load_acquired, _tensor_record
    task, split = request["task"], request["split"]
    from propagation_predictive_v1.core import prior_seconds
    budget = Budget(admitted, prior_seconds(admitted), context)
    budget.check()
    phase = admitted["tasks"][task]["engineering_phase"]
    export, data_binding = load_acquired(phase)
    acquisition = json.loads(checked(admitted["tasks"][task]["acquisition"]).read_text())
    storage_path = within(checked(admitted["tasks"][task]["acquisition"]).parent / "PROVIDER_STORAGE.json")
    if sha(storage_path) != request["provider_storage_sha256"]:
        raise ValueError("Root did not bind exact raw/processed provider evidence")
    storage = json.loads(storage_path.read_text())
    processed = checked(request["processed_public_file"])
    inventory_field = request["provider_storage_inventory_field"]
    if inventory_field not in ("processed", "original_provider_files") or "/processed/" not in request["processed_public_file"]["path"]:
        raise ValueError("Root must freeze the public processed inventory interface explicitly")
    matching = [r for r in storage[inventory_field] if r["path"] == str(processed.relative_to(PHASE))]
    if len(matching) != 1 or matching[0]["sha256"] != sha(processed) or matching[0]["bytes"] != processed.stat().st_size:
        raise ValueError("Requested processed public bytes are not in the audited provider receipt")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", {"binding": binding_ref, "live": context, "request": request,
          "supervisor_directory": context["supervisor_relative_to_project"],
          "all_public_targets_materialized_by_torch_load": True,
          "public_pickle_trust_scope": "exact hashed completed named public provider storage; not an arbitrary checkpoint",
          "test_target_indexing_authorized": split == "test"})
    # Public PyG processed storage is an object pickle, explicitly bound above.
    # This does not import a dataset or invoke a provider download/process path.
    payload = torch.load(processed, map_location="cpu", weights_only=False)
    if not isinstance(payload, tuple) or len(payload) != request["expected_processed_tuple_length"] or len(payload) not in (2, 3):
        raise ValueError("Unsupported exact provider processed format; no target indexing")
    graph = payload[0]
    if len(payload) == 3:
        if not isinstance(graph, dict):
            raise ValueError("Declared PyG dict3 processed interface differs")
        get = graph.__getitem__
    else:
        get = lambda name: getattr(graph, name)
    # Feature/edge identity and training-label equality tie held-out labels to
    # the exact provider graph, without inspecting their unrequested values.
    for name in ("x", "edge_index"):
        original = get(name)
        if original.device.type != "cpu" or _tensor_record(original.contiguous()) != _tensor_record(export[name]):
            raise ValueError("Public provider graph differs from acquired complete export")
    train_ids = export["official_split_node_ids"]["train"]
    if not torch.equal(get("y").index_select(0, train_ids), export["training_targets"]):
        raise ValueError("Public provider training values differ from acquired export")
    ids = export["official_split_node_ids"][split]
    values = get("y").index_select(0, ids).reshape(-1)
    if (not torch.isfinite(values).all() or not (values == values.to(torch.long)).all()
            or values.min() < 0 or values.max() >= export["declared_official_class_count"]):
        raise ValueError("Requested official split class targets invalid")
    targets = values.to(torch.long).contiguous()
    bundle = output / "split_targets.pt"
    budget.check(ids.numel() * 8 + targets.numel() * 8 + 1024 * 1024)
    with bundle.open("xb") as stream:
        torch.save({"schema": "gnnm-propagation-single-official-split-targets-v1", "task": task,
                    "split": split, "node_ids": ids, "targets": targets}, stream)
    budget.committed(bundle)
    if sha(processed) != request["processed_public_file"]["sha256"]:
        raise ValueError("Provider bytes changed during split admission")
    budget.check()
    write(output / "TARGET_ADMISSION.json", {"schema": "gnnm-propagation-admitted-split-targets-v1",
          "task": task, "split": split, "indexed_target_splits": ["train", split],
          "binding_sha256": binding_ref["sha256"], "bundle": reference(bundle),
          "processed_public_file": request["processed_public_file"], "provider_storage": reference(storage_path),
          "provider_storage_inventory_field": inventory_field, "processed_tuple_length": len(payload),
          "data_binding": data_binding, "public_storage_all_targets_materialized": True,
          "requested_split_target_fingerprint": _tensor_record(targets),
          "unrequested_target_values_indexed_or_summarized": False, "live": context,
          "reference_hash_cache": hash_cache_report(),
          "family_lock": request.get("family_lock"), "qualification": "pending completed supervisor audit"})
    manifest(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--supervisor", type=Path, required=True)
    args = p.parse_args()
    admitted, binding_ref = binding(args.binding)
    request_path = within(args.request, HERE)
    request = json.loads(request_path.read_text())
    if (request.get("schema") != "gnnm-propagation-root-split-admission-request-v1"
            or request.get("root_authorized") is not True or request.get("binding") != binding_ref
            or request.get("task") not in admitted["tasks"] or request.get("split") not in ("valid", "test")):
        raise ValueError("Exact root-approved split admission request required before provider bytes")
    # Requests are root-owned separately hashed entries in the binding for
    # validation, or in the complete post-fit family lock for test.
    if request["split"] == "valid":
        payload = {k: v for k, v in request.items() if k != "binding"}
        if digest(payload) != admitted["tasks"][request["task"]]["validation_label_request_payload_sha256"]:
            raise ValueError("Validation request is not pre-frozen in the method admission")
    else:
        from propagation_predictive_v1.core import audit_family_lock
        lock = audit_family_lock(checked(request["family_lock"]), admitted, binding_ref)
        if request.get("family_lock_payload_sha256") != digest(lock["locked_choices"]):
            raise ValueError("All30 checkpoint/member/execution choices not locked before test")
        payload = {k: v for k, v in request.items() if k != "family_lock"}
        if digest(payload) != lock["test_label_request_payload_sha256"][request["task"]]:
            raise ValueError("Test request differs from root-frozen complete-family lock")
    context = live(args.supervisor, Path(__file__))
    if context["gpu_uuid"] != admitted["gpu_uuid"]:
        raise ValueError("Split exporter wrong authorized allocation")
    output = within(PHASE / request["output_relative_to_phase"], HERE)
    if output != target_directory(admitted, request["task"], request["split"]):
        raise ValueError("Target admission differs from prescribed complete-family budgeted location")
    if output.exists():
        raise ValueError("Split admission output must be new; no overwrite/resume")
    runtime_before_import(admitted)
    try:
        export_labels(admitted, binding_ref, request, output, context)
    except BaseException as error:
        if output.exists() and not (output / "FAILURE.json").exists():
            write(output / "FAILURE.json", {"schema": "gnnm-propagation-target-admission-failure-v1",
                "type": type(error).__name__, "message": str(error), "partial_outputs_preserved": True,
                "retry_resume_or_family_promotion_authorized": False, "live": context})
        raise


if __name__ == "__main__":
    main()
