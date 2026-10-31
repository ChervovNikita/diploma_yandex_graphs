"""Bounded train-only engineering cost estimator. Retains no weights or logits.

Validation node IDs, topology and features are visible; validation targets are
never loaded. CPU scoring costs use label-free fixed-class reductions and
pairwise predictions, discard their values and report only elapsed work.
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
import math
from pathlib import Path
import statistics
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


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


def request_binding(path):
    from shared_message_predictive_v2.contract import (ARMS, TASKS, checked, digest, reference,
        source_check, within)
    value = json.loads(within(path, HERE).read_text())
    if (value.get("schema") != "gnnm-shared-message-cost-preflight-request-v2"
            or value.get("root_authorized") is not True or value.get("scientific_training_authorized") is not False
            or value.get("validation_target_export_or_access_authorized") is not False
            or value.get("test_target_access_authorized") is not False
            or value.get("warmup_updates") != 1 or value.get("measured_updates") != 3
            or value.get("retained_weights_or_scores") is not False
            or set(value.get("tasks", {})) != set(TASKS) or set(value.get("execution_policy", {})) != set(TASKS)):
        raise ValueError("Separate bounded training-only engineering request required")
    source_check()
    if (value.get("source_manifest_sha256") != reference(HERE / "SOURCE_MANIFEST_v1.json")["sha256"]
            or value["source_bindings_sha256"] != reference(HERE / "SOURCE_BINDINGS.json")["sha256"]):
        raise ValueError("Preflight source closure is not frozen")
    approval = json.loads(checked(value["root_approval"]).read_text())
    if approval.get("schema") != "gnnm-shared-message-cost-preflight-root-approval-v2" or approval.get("qualified") is not True or approval.get("approved_payload_sha256") != digest({k: v for k, v in value.items() if k != "root_approval"}):
        raise ValueError("Actual preflight input/runtime/variant/resource payload lacks root approval")
    for task in TASKS:
        phase = json.loads(checked(value["tasks"][task]["engineering_phase"]).read_text())
        if phase["task"] != task or set(value["execution_policy"][task]) != set(ARMS):
            raise ValueError("Every complete task and model variant must be frozen")
        for key in ("acquisition", "prepared_manifest", "parameter_control_report", "input_completed_audit"):
            checked(value["tasks"][task][key])
        for arm, variant in value["execution_policy"][task].items():
            if set(variant) != {"training", "validation", "deployment", "checkpoint_blocks"} or type(variant["checkpoint_blocks"]) is not bool:
                raise ValueError("Complete preflight variant required")
            if any(variant[k] not in ("serial", "packed") for k in ("training", "validation", "deployment")):
                raise ValueError("Unknown preflight path")
            if arm in ("shared_mean_message", "detached_mean_message") and any(variant[k] != "packed" for k in ("training", "validation", "deployment")):
                raise ValueError("Shared preflight must be packed")
            if arm.startswith("single_") and any(variant[k] != "packed" for k in ("training", "validation", "deployment")):
                raise ValueError("Single interface has one declared packed trajectory")
    within(PHASE / value["output_root"], HERE)
    caps = {"per_child_wall_cap_seconds": 900, "per_child_disk_cap_bytes": 2**30,
            "peak_reserved_bytes_cap": 60 * 2**30}
    if (value.get("bounded_source_sha256") != "b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23"
            or any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k])
                   or not 0 < value[k] <= cap for k, cap in caps.items())):
        raise ValueError("Preflight bounds exceed source scope")
    return value, reference(path)


def label_free_reductions(member, pooled):
    """Actual CPU reduction kernels; fixed class0 gather is not a quality metric."""
    import torch
    m, n, c = member.shape
    for start in range(0, n, 65536):
        raw = member[:, start:start + 65536]
        values = pooled[start:start + 65536].double()
        _ = values.log_softmax(-1)[:, 0].sum()
        _ = values.argmax(-1).sum()
        predictions = []
        for k in range(m):
            values = raw[k].double()
            _ = values.log_softmax(-1)[:, 0].sum()
            predictions.append(values.argmax(-1))
            _ = (predictions[-1] == 0).sum()
        for a in range(m):
            for b in range(a):
                _ = (predictions[a] != predictions[b]).sum()
                _ = ((predictions[a] != 0) & (predictions[b] != 0)).sum()
                _ = ((predictions[a] != 0) | (predictions[b] != 0)).sum()


def measure(request, request_ref, task, arm, context):
    from shared_message_predictive_v2.contract import (hash_report, manifest, runtime_import_receipt, within, write)
    import torch
    from shared_message_predictive_v2.core import configure, load_inputs, setup, cpu_tree, tensor_bytes
    from shared_message_quality_v3 import counts, models
    from propagation_cost_impl_v1.dropout import DropoutKey
    output = within(PHASE / request["output_root"] / ("reddit2" if task == "Reddit2" else "products") / arm, HERE)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "START.json", {"request": request_ref, "task": task, "arm": arm,
          "supervisor_directory": context["supervisor_relative_to_project"], "live": context})
    start_all = time.perf_counter()
    def cap():
        if time.perf_counter() - start_all > request["per_child_wall_cap_seconds"]:
            raise RuntimeError("Bounded preflight child elapsed cap exceeded")
    configure(request)
    begin = time.perf_counter()
    export, lineage, metadata, directory = load_inputs(request, task)
    input_seconds = time.perf_counter() - begin
    torch.cuda.reset_peak_memory_stats(0)
    begin = time.perf_counter()
    cell = {"task": task, "arm": arm, "seed": 17}
    model, family, x, variant, info = setup(request, cell, export, metadata, directory)
    optimizer = models.make_optimizer(model)
    ids = export["training_target_node_ids"].to("cuda:0")
    targets = export["training_targets"].reshape(-1).to(dtype=torch.long, device="cuda:0")
    valid_ids = export["official_split_node_ids"]["valid"].to("cuda:0")
    torch.cuda.synchronize(0)
    setup_seconds = time.perf_counter() - begin
    updates = []
    for epoch in range(1, 5):
        cap()
        model.train()
        torch.cuda.synchronize(0)
        begin = time.perf_counter()
        models.train_step(model, family, x, ids, targets, optimizer, execution=variant["training"],
            dropout_key=DropoutKey(counts.TASKS[task]["index"], 17, epoch), return_loss=False)
        torch.cuda.synchronize(0)
        if epoch > 1:
            updates.append(time.perf_counter() - begin)
    model.eval()
    validations, CPU, writes, hashes, loads, transfers = [], [], [], [], [], []
    payload_bytes = None
    for _ in range(3):
        cap()
        begin = time.perf_counter()
        member, pooled = models.split_logits(model, family, x, valid_ids, execution=variant["validation"])
        torch.cuda.synchronize(0)
        validations.append(time.perf_counter() - begin)
        begin = time.perf_counter()
        label_free_reductions(member, pooled)
        CPU.append(time.perf_counter() - begin)
        begin = time.perf_counter()
        state = {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                 "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        transfers.append(time.perf_counter() - begin)
        for kind, value in (("checkpoint", state), ("validation_logits", {"member_logits": member, "pooled_logits": pooled, "node_ids": export["official_split_node_ids"]["valid"]})):
            path = output / ("EPHEMERAL_" + kind + ".pt")
            estimate = tensor_bytes(value) + 2**20
            if estimate > request["per_child_disk_cap_bytes"]:
                raise RuntimeError("Preflight temporary artifact exceeds cap")
            # Unlink the empty file BEFORE any weights/logits are written. The
            # open descriptor measures project-filesystem I/O but cannot leave
            # a named scientific state behind, even on process termination.
            begin = time.perf_counter()
            with path.open("xb+") as stream:
                path.unlink()
                torch.save(value, stream)
                stream.flush()
                size = os.fstat(stream.fileno()).st_size
                writes.append({"kind": kind, "seconds": time.perf_counter() - begin, "bytes": size})
                begin = time.perf_counter()
                stream.seek(0)
                import hashlib
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 2**20), b""):
                    h.update(block)
                hashes.append({"kind": kind, "seconds": time.perf_counter() - begin, "bytes": size})
                begin = time.perf_counter()
                stream.seek(0)
                replay_payload = torch.load(stream, map_location="cpu", weights_only=True)
                loads.append({"kind": kind, "seconds": time.perf_counter() - begin})
                del replay_payload
        payload_bytes = {"model_tensor_bytes": tensor_bytes(state["model"]), "optimizer_tensor_bytes": tensor_bytes(state["optimizer"]),
                         "RNG_tensor_bytes": tensor_bytes(state["cpu_rng"]) + tensor_bytes(state["cuda_rng"])}
        del state, member, pooled
    inference = []
    with torch.no_grad():
        for _ in range(5):
            cap()
            torch.cuda.synchronize(0)
            begin = time.perf_counter()
            values = models.pooled_inference(model, family, x, execution=variant["deployment"])
            torch.cuda.synchronize(0)
            inference.append(time.perf_counter() - begin)
            del values
    memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
    cap()
    if memory["peak_reserved_bytes"] > request["peak_reserved_bytes_cap"]:
        raise RuntimeError("Complete train-only preflight memory cap exceeded")
    write(output / "COST_PREFLIGHT_END.json", {"schema": "gnnm-shared-message-bounded-cost-preflight-v2", "status": "completed",
          "request": request_ref, "task": task, "arm": arm, "seed": 17, "execution": variant, "model_info": info,
          "input_load_seconds": input_seconds, "model_operator_setup_transfer_seconds": setup_seconds,
          "three_complete_update_seconds": updates, "three_validation_forward_CPU_slice_seconds": validations,
          "three_CPU_label_free_reduction_seconds": CPU, "three_checkpoint_CPU_transfer_seconds": transfers,
          "temporary_artifact_writes": writes, "temporary_artifact_full_hashes": hashes, "temporary_artifact_loads": loads,
          "pooled_inference_seconds": inference, "checkpoint_payload_bytes": payload_bytes,
          "validation_nodes": int(export["official_split_node_ids"]["valid"].numel()), "classes": export["declared_official_class_count"],
          "memory": memory, "retained_weights_or_scores": False, "labels_indexed": ["train"],
          "validation_target_export_or_access": False, "test_target_access": False,
          "CPU_cost_proxy": "label-free fixed-class log-softmax gathers and complete pairwise reduction kernels; no quality values retained",
          "artifact_IO_measurement": "open project file unlinked before payload; actual filesystem write/read/hash cost; no fsync or persistent-directory durability claim",
          "quality_novelty_convergence_or_scientific_admission_established": False,
          "input_lineage": lineage, "live": context, "hash_work": hash_report(),
          **runtime_import_receipt(require_generated=True)})
    manifest(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--supervisor", type=Path, required=True)
    p.add_argument("--task", choices=("Reddit2", "ogbn-products"), required=True)
    p.add_argument("--arm", choices=("exact_member", "shared_mean_message", "detached_mean_message", "single_base64", "single_parameters"), required=True)
    args = p.parse_args()
    bootstrap(args.request)
    from shared_message_predictive_v2.contract import live, runtime_before_import, within
    request, ref = request_binding(args.request)
    runtime_before_import(request)
    context = live(args.supervisor, Path(__file__), args.request, request)
    if context["gpu_uuid"] != request["gpu_uuid"]:
        raise ValueError("Wrong frozen complete engineering runtime")
    output = within(PHASE / request["output_root"] / ("reddit2" if args.task == "Reddit2" else "products") / args.arm, HERE)
    try:
        measure(request, ref, args.task, args.arm, context)
    finally:
        # Temporary state never survives success or a handled Python failure.
        if output.exists():
            for path in output.glob("EPHEMERAL_*.pt"):
                path.unlink()


if __name__ == "__main__":
    main()
