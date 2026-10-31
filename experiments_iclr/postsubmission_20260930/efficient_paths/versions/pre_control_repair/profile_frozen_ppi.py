"""Explicit CPU/CUDA cost CLI on whole frozen PPI TRAIN graphs only.

Defaults to CPU. CUDA requires an explicit physical GPU, logical cuda:0,
matching CUDA_VISIBLE_DEVICES, and an empty-device guard before allocation.
No optimizer steps, checkpoint selection, validation/test loading, or scores.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI

PHASE = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE / "protocols"))
from baseline_adapters import (IndependentEnsemble, ProjectorEnsemble,
                               SharedTrunkMultihead, SingleModel, model_options)
from centered_paths import (CenteredPathEnsemble, IndependentNarrowPaths, MeanOperator,
                            PackedIndependentSAGEEnsemble, PackedSharedSAGEEnsemble,
                            PlainSAGE, mean_member_loss, nearest_parameter_width)

ARMS = ("centered", "independent_narrow", "no_recycling", "no_private_edges",
        "gnnm", "packed_gnnm", "gnnm_serial_backward", "ens", "packed_full",
        "packed_channels", "packed_parameters", "base", "base_parameters", "multihead")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside_phase(path):
    value = Path(path).resolve()
    if PHASE not in value.parents:
        raise ValueError("Data/source paths must remain inside this research phase")
    return value


def tensor_fingerprint(tensor):
    tensor = tensor.detach().cpu().contiguous()
    metadata = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return {**metadata, "sha256": digest.hexdigest()}


def explicit_device_guard(args):
    if args.device == "cpu":
        if args.physical_gpu is not None:
            raise ValueError("--physical-gpu is meaningful only with explicit --device cuda:0")
        return {"device": "cpu", "gpu_requested": False}
    if args.device != "cuda:0" or args.physical_gpu is None:
        raise ValueError("CUDA requires explicit --device cuda:0 and --physical-gpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain exactly the chosen physical GPU")
    result = subprocess.run([
        "nvidia-smi", "-i", str(args.physical_gpu),
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True)
    memory, utilization = map(int, result.stdout.strip().split(","))
    if memory > 100 or utilization != 0:
        raise RuntimeError("Chosen GPU is occupied; no CUDA model/data allocation was performed")
    return {"device": args.device, "gpu_requested": True,
            "physical_gpu": args.physical_gpu, "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "guard_memory_used_mib": memory, "guard_utilization_percent": utilization,
            "guard_rule": "memory.used <= 100 MiB and utilization.gpu == 0 before allocation"}


def make_model(arm, seed, layers):
    torch.manual_seed(seed)
    options = model_options(50, 121, hidden_dim=128, layers=layers, dropout=0.2)
    kw = dict(input_dim=50, output_dim=121, dim=128, members=4,
              layers=layers, dropout=0.2)
    metadata = {}
    if arm in ("gnnm", "gnnm_serial_backward"):
        return ProjectorEnsemble(options), "source", metadata
    if arm == "packed_gnnm":
        original = ProjectorEnsemble(options)
        model = PackedSharedSAGEEnsemble(**kw)
        model.copy_from_source_tabm(original.model)
        metadata["initialization"] = "Exact copied source TABMModel factors and shared weights"
        return model, "sparse", metadata
    if arm == "ens":
        return IndependentEnsemble(options), "source", metadata
    if arm == "packed_full":
        original = IndependentEnsemble(options)
        model = PackedIndependentSAGEEnsemble(**kw)
        model.copy_from_source_members(original.models)
        metadata["initialization"] = "Exact copied independent source Model weights"
        return model, "sparse", metadata
    if arm == "base":
        return SingleModel(options), "source", metadata
    if arm == "multihead":
        return SharedTrunkMultihead(options), "source", metadata
    if arm == "centered":
        return CenteredPathEnsemble(**kw, rank=16), "sparse", metadata
    if arm == "independent_narrow":
        return IndependentNarrowPaths(**kw, rank=16), "sparse", metadata
    if arm == "no_recycling":
        return CenteredPathEnsemble(**kw, rank=16, recycle_mean=False), "sparse", metadata
    if arm == "no_private_edges":
        return CenteredPathEnsemble(**kw, rank=16, private_edges=False), "sparse", metadata
    if arm == "packed_channels":
        return PackedIndependentSAGEEnsemble(**{**kw, "dim": 44}), "sparse", {"width": 44}
    target = sum(p.numel() for p in CenteredPathEnsemble(**kw, rank=16).parameters())
    members = 4 if arm == "packed_parameters" else 1
    width = nearest_parameter_width(target, 50, 121, layers=layers, members=members)
    model = (PackedIndependentSAGEEnsemble(**{**kw, "dim": width}) if members == 4
             else PlainSAGE(50, 121, width, layers=layers, dropout=0.2))
    actual = sum(p.numel() for p in model.parameters())
    return model, "sparse", {"width": width, "target_parameters": target,
        "actual_parameters": actual, "parameter_difference": actual - target,
        "relative_parameter_difference": (actual - target) / target,
        "width_rule": "Closest actual source-style count on widths8..512 in steps4; ties choose smaller; no labels"}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_cost_operation(model, mode, graph, operator, observed, operation, arm):
    model.zero_grad(set_to_none=True)
    if operation == "inference":
        model.eval()
        with torch.no_grad():
            logits = model(graph, graph.x) if mode == "source" else model(operator, graph.x)
            logits.mean(dim=0)
        return
    model.train()
    if arm == "gnnm_serial_backward":
        for member in range(model.members):
            logits = model.model(graph, graph.x, member)
            loss = F.binary_cross_entropy_with_logits(logits[observed], graph.y[observed])
            (loss / model.members).backward()
    else:
        logits = model(graph, graph.x) if mode == "source" else model(operator, graph.x)
        mean_member_loss(logits, graph.y, observed).backward()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PHASE / "new_tasks/data/ppi/frozen_manifest.json")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--graph-indices", nargs="+", type=int, default=[0])
    parser.add_argument("--label-fraction", type=float, choices=(0.1, 1.0), default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--check-plan", action="store_true",
                        help="Validate frozen train data and model counts only; never access CUDA")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if OUTPUT_ROOT not in output.parents:
        raise ValueError("Output must remain inside efficient_paths")
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing profile record")
    if args.layers != 3 or args.seed not in (0, 1, 2):
        raise ValueError("First PPI preflight fixes layers3 and seeds0/1/2")
    if args.warmups < 0 or args.repetitions < 3:
        raise ValueError("Require nonnegative warm-ups and at least three repetitions")
    if len(set(args.graph_indices)) != len(args.graph_indices) or not args.graph_indices:
        raise ValueError("Choose distinct complete training graph indices")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("Choose each arm once")
    manifest = inside_phase(args.manifest)
    frozen = json.loads(manifest.read_text())
    if (frozen["dataset"] != "ppi" or frozen["scoring_performed"]
            or frozen["confirmation_data_accessed"]):
        raise ValueError("Require an unscored development-only frozen PPI manifest")
    cache = manifest.parent
    # Existing processed TRAIN bytes are mandatory: this CLI never downloads.
    if not (cache / "provider/processed/train.pt").exists():
        raise ValueError("Frozen processed train.pt is absent; freeze data before profiling")
    for row in frozen["raw_and_processed_files"]:
        file = inside_phase(cache / row["relative_path"])
        if sha(file) != row["sha256"]:
            raise ValueError("Raw/processed bytes changed after data freeze")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    random.seed(args.seed)
    dataset = PPI(root=str(cache / "provider"), split="train")
    train_fingerprints = frozen["split_summaries"]["train"]["graph_tensor_fingerprints"]
    graphs, masks, selected_fingerprints = [], [], []
    for index in args.graph_indices:
        if index < 0 or index >= len(dataset):
            raise ValueError("Training graph index out of range")
        graph = dataset[index]
        expected = train_fingerprints[index]
        for name in ("x", "edge_index", "y"):
            if tensor_fingerprint(getattr(graph, name)) != expected[name]:
                raise ValueError("Loaded training tensor differs from canonical fingerprint")
        if graph.x.shape[1] != 50 or graph.y.shape[1] != 121:
            raise ValueError("Unexpected PPI feature/target width")
        generator = torch.Generator().manual_seed(9701 + index)
        chosen = torch.randperm(graph.num_nodes, generator=generator)[:max(1, int(graph.num_nodes * args.label_fraction))]
        mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask[chosen] = True
        graphs.append(graph)
        masks.append(mask)
        selected_fingerprints.append({"train_index": index, "raw_graph_id": index + 1,
            "x": expected["x"], "edge_index": expected["edge_index"],
            "observed_mask": tensor_fingerprint(mask)})
    graph = graphs[0] if len(graphs) == 1 else Batch.from_data_list(graphs)
    observed = torch.cat(masks)
    source_paths = [Path(__file__), OUTPUT_ROOT / "centered_paths.py",
                    PHASE / "protocols/baseline_adapters.py", PHASE / "source_snapshot/models.py"]
    report = {"scope": "Whole frozen PPI TRAIN graph cost preflight; no optimization or prediction scores",
        "status": "plan_checked" if args.check_plan else "running", "argv": sys.argv,
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda, "pyg": importlib.metadata.version("torch_geometric"),
        "dtype": "float32", "amp": False, "threads": 1,
        "manifest_sha256": sha(manifest), "source_sha256": {str(p.relative_to(PHASE)): sha(p) for p in source_paths},
        "train_graphs": selected_fingerprints, "batching": "one whole graph" if len(graphs) == 1 else "disconnected whole-graph batch",
        "nodes": graph.num_nodes, "edge_entries": graph.edge_index.shape[1],
        "features": 50, "targets": 121, "observed_nodes": int(observed.sum()),
        "label_fraction": args.label_fraction, "layers": 3, "members": 4,
        "dropout_training": 0.2, "pooling_inference": "mean raw logits",
        "loss": "mean member BCE on the fixed observed training nodes and121 targets",
        "warmups": args.warmups, "repetitions": args.repetitions,
        "optimizer_steps": 0, "validation_loader_accessed": False, "test_loader_accessed": False,
        "memory_scope": "CUDA allocated/reserved peaks; baseline resident graph/model/mask recorded. Working increment includes gradient tensors; optimizer state absent.",
        "arms": {}}
    if args.check_plan:
        for arm in args.arms:
            model, mode, metadata = make_model(arm, args.seed, args.layers)
            report["arms"][arm] = {"mode": mode, "parameters": sum(p.numel() for p in model.parameters()), "budget": metadata}
        report["device_guard"] = {"device": "cpu", "gpu_requested": False,
                                  "check_plan_ignores_requested_cuda": True}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    guard = explicit_device_guard(args)
    report["device_guard"] = guard
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        properties = torch.cuda.get_device_properties(device)
        report["hardware"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                              "compute_capability": [properties.major, properties.minor]}
    graph = graph.to(device)
    observed = observed.to(device)
    synchronize(device)
    start = time.perf_counter()
    operator = MeanOperator.from_edges(graph.edge_index, graph.num_nodes,
                                       dtype=graph.x.dtype, device=device)
    synchronize(device)
    report["adjacency_preparation_seconds"] = time.perf_counter() - start
    arm_order = list(args.arms)
    random.Random(20260930 + args.seed).shuffle(arm_order)
    report["arm_order"] = arm_order
    try:
        for arm in arm_order:
            model, mode, metadata = make_model(arm, args.seed, args.layers)
            model.to(device)
            row = {"mode": mode, "parameters": sum(p.numel() for p in model.parameters()),
                   "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                   "budget": metadata, "operations": {}}
            report["arms"][arm] = row
            for operation in ("inference", "forward_backward"):
                for _ in range(args.warmups):
                    evaluate_cost_operation(model, mode, graph, operator, observed, operation, arm)
                samples = []
                for _ in range(args.repetitions):
                    model.zero_grad(set_to_none=True)
                    synchronize(device)
                    operator.reset_counts()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        baseline_allocated = torch.cuda.memory_allocated(device)
                        baseline_reserved = torch.cuda.memory_reserved(device)
                        first = torch.cuda.Event(enable_timing=True)
                        last = torch.cuda.Event(enable_timing=True)
                        first.record()
                    wall = time.perf_counter()
                    evaluate_cost_operation(model, mode, graph, operator, observed, operation, arm)
                    if device.type == "cuda":
                        last.record()
                    synchronize(device)
                    sample = {"wall_ms": (time.perf_counter() - wall) * 1000}
                    if device.type == "cuda":
                        peak = torch.cuda.max_memory_allocated(device)
                        sample.update({"cuda_event_ms": first.elapsed_time(last),
                            "baseline_allocated_bytes": baseline_allocated,
                            "baseline_reserved_bytes": baseline_reserved,
                            "peak_allocated_bytes": peak,
                            "working_allocated_increment_bytes": peak - baseline_allocated,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
                    samples.append(sample)
                times = sorted(sample["wall_ms"] for sample in samples)
                row["operations"][operation] = {"samples": samples,
                    "median_wall_ms": statistics.median(times),
                    "p90_wall_ms": times[min(len(times) - 1, int(0.9 * len(times)))],
                    "sparse_operator_widths_last_repetition": list(operator.widths),
                    "source_graph_calls_if_not_instrumented": (3 if arm in ("base", "multihead") else 12) if mode == "source" else None}
                if device.type == "cuda":
                    row["operations"][operation].update({
                        "median_cuda_event_ms": statistics.median(sample["cuda_event_ms"] for sample in samples),
                        "max_peak_allocated_bytes": max(sample["peak_allocated_bytes"] for sample in samples),
                        "max_working_allocated_increment_bytes": max(sample["working_allocated_increment_bytes"] for sample in samples),
                        "max_peak_reserved_bytes": max(sample["peak_reserved_bytes"] for sample in samples)})
            model.cpu()
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "arms": {arm: {op: data["median_wall_ms"] for op, data in row["operations"].items()}
                               for arm, row in report["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
