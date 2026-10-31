"""Read only acquired train-target export, fingerprints and completed receipts."""

from __future__ import annotations

import hashlib
import json
import re
import sys

from .contract import PROJECT, TASKS, sha256, within

EXPORT_SCHEMA = "gnnm-complete-node-graph-training-target-export-v1"
ALLOWED_KEYS = {"schema", "public_dataset", "x", "edge_index", "declared_official_class_count",
                "official_split_node_ids", "official_split_masks", "training_target_node_ids",
                "training_targets", "edge_attr", "edge_weight"}


def _tensor_record(tensor):
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Acquired tensors must remain exact contiguous CPU tensors")
    array = tensor.detach().numpy()
    metadata = {"dtype": str(tensor.dtype), "numpy_dtype": array.dtype.str,
                "shape": list(tensor.shape), "layout": "contiguous_C_order", "byteorder": sys.byteorder}
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    canonical, raw = hashlib.sha256(), hashlib.sha256()
    canonical.update(len(header).to_bytes(8, "big"))
    canonical.update(header)
    view = memoryview(array).cast("B")
    for start in range(0, view.nbytes, 8 * 1024 * 1024):
        chunk = view[start:start + 8 * 1024 * 1024]
        canonical.update(chunk)
        raw.update(chunk)
    return {**metadata, "numel": tensor.numel(), "bytes": view.nbytes,
            "canonical_sha256": canonical.hexdigest(), "raw_tensor_bytes_sha256": raw.hexdigest()}


def load_acquired(phase):
    """No provider import and no provider raw/processed/archive file is opened."""
    import torch

    repo = phase["repo_root"]
    from pathlib import Path
    packet = phase["acquisition_packet"]
    if not re.fullmatch(r"propagation_data_v[1-9][0-9]*", packet):
        raise ValueError("Explicit versioned propagation acquisition packet required")
    acquisition_root = PROJECT / packet
    acquisition_path = within(Path(repo) / phase["acquisition_relative_to_repo"],
                              acquisition_root, existing=True)
    acquisition = json.loads(acquisition_path.read_text())
    if (acquisition.get("schema") != "gnnm-propagation-data-acquisition-v1"
            or acquisition.get("status") != "completed_data_only_acquisition"
            or acquisition.get("heldout_target_values_read_by_exporter") is not False
            or acquisition.get("scores_computed") is not False):
        raise ValueError("Exact acquired training-only export receipt required")
    task_name = acquisition["public_dataset"]
    if task_name != phase["task"] or task_name not in TASKS:
        raise ValueError("Acquisition task differs from frozen phase")
    audit_path = within(Path(repo) / phase["acquisition_completed_audit_relative_to_repo"], PROJECT, existing=True)
    audit = json.loads(audit_path.read_text())
    # This is an explicit root-owned audit receipt, never synthesized here.
    if (audit.get("schema") != "gnnm-propagation-acquisition-completed-admission-v1"
            or audit.get("qualified") is not True
            or audit.get("acquisition_sha256") != sha256(acquisition_path)
            or audit.get("export_sha256") != acquisition["export"]["sha256"]
            or audit.get("authorized_ssh_account") != phase["authorized_ssh_account"]):
        raise ValueError("Completed acquisition chain/root audit is not admitted")
    bindings_path = within(acquisition_root / "SOURCE_BINDINGS.json", acquisition_root, existing=True)
    if sha256(bindings_path) != acquisition.get("source_bindings_sha256"):
        raise ValueError("Acquisition source-binding receipt changed")
    export_path = within(PROJECT / acquisition["export"]["path"], acquisition_root, existing=True)
    if export_path.parent != acquisition_path.parent:
        raise ValueError("Acquisition/export directories differ")
    if export_path.stat().st_size != acquisition["export"]["bytes"] or sha256(export_path) != acquisition["export"]["sha256"]:
        raise ValueError("Acquired export bytes changed")
    fingerprints_path = within(acquisition_path.parent / "TENSOR_FINGERPRINTS.json", PROJECT, existing=True)
    fingerprints = json.loads(fingerprints_path.read_text())
    export = torch.load(export_path, map_location="cpu", weights_only=True)
    if (not isinstance(export, dict) or set(export) - ALLOWED_KEYS
            or export.get("schema") != EXPORT_SCHEMA or export.get("public_dataset") != task_name):
        raise ValueError("Training-only export structure changed or contains forbidden keys")
    spec = TASKS[task_name]
    if export["x"].shape != (spec["nodes"], spec["features"]) or export["x"].dtype != torch.float32:
        raise ValueError("Complete provider feature tensor/precision differs")
    edge = export["edge_index"]
    if edge.device.type != "cpu" or edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("Exact complete source/receiver edge tensor required")
    if edge.numel() and (edge.min() < 0 or edge.max() >= spec["nodes"]):
        raise ValueError("Original edge IDs outside complete node order")
    if export["declared_official_class_count"] != spec["classes"]:
        raise ValueError("Official class declaration differs")
    ids, masks = export["official_split_node_ids"], export["official_split_masks"]
    if set(ids) != {"train", "valid", "test"} or set(masks) != set(ids):
        raise ValueError("Official split names differ")
    coverage = torch.zeros(spec["nodes"], dtype=torch.int8)
    tensors = {"x": export["x"], "edge_index": edge,
               "training_target_node_ids": export["training_target_node_ids"],
               "training_targets": export["training_targets"]}
    for split in ("train", "valid", "test"):
        if ids[split].dtype != torch.long or ids[split].ndim != 1 or masks[split].dtype != torch.bool or masks[split].shape != (spec["nodes"],):
            raise ValueError("Official split tensor type/shape differs")
        if ids[split].numel() == 0 or ids[split].min() < 0 or ids[split].max() >= spec["nodes"]:
            raise ValueError("Official split node order outside graph")
        replay_mask = torch.zeros(spec["nodes"], dtype=torch.bool)
        replay_mask[ids[split]] = True
        if replay_mask.sum() != ids[split].numel() or not torch.equal(replay_mask, masks[split]):
            raise ValueError("Official split IDs/masks differ or contain duplicates")
        coverage += masks[split].to(torch.int8)
        tensors["official_split_node_ids/" + split] = ids[split]
        tensors["official_split_masks/" + split] = masks[split]
    if not (coverage == 1).all() or not torch.equal(export["training_target_node_ids"], ids["train"]):
        raise ValueError("Complete official split coverage/training target order differs")
    for name in ("edge_attr", "edge_weight"):
        if name in export:
            tensors[name] = export[name]
    if set(tensors) != set(fingerprints):
        raise ValueError("Acquired tensor fingerprint coverage differs")
    for name, tensor in tensors.items():
        record = _tensor_record(tensor)
        if any(record[key] != fingerprints[name].get(key) for key in record):
            raise ValueError("Acquired exact tensor changed: " + name)
    labels = export["training_targets"].reshape(-1)
    if labels.numel() != ids["train"].numel() or not torch.isfinite(labels).all() or not (labels == labels.to(torch.long)).all():
        raise ValueError("Training targets must be exact aligned integer class IDs")
    if labels.min() < 0 or labels.max() >= spec["classes"]:
        raise ValueError("Training targets outside fixed official class range")
    # The only target field in this object contains training values. IDs/masks
    # for validation/test are public structure and contain no target values.
    return export, {
        "task": task_name, "acquisition_relative_to_project": str(acquisition_path.relative_to(PROJECT)),
        "acquisition_sha256": sha256(acquisition_path), "export_sha256": acquisition["export"]["sha256"],
        "completed_audit_sha256": sha256(audit_path),
        "acquisition_packet": packet, "acquisition_source_bindings_sha256": sha256(bindings_path),
        "tensor_fingerprints_sha256": sha256(fingerprints_path),
        "source_export_contains_only_training_targets": True,
        "optional_edge_fields": [name for name in ("edge_attr", "edge_weight") if name in export],
        "optional_edge_fields_ignored_by_source_SAGE": True,
        "provider_storage_opened": False, "heldout_target_values_loaded": False,
    }
