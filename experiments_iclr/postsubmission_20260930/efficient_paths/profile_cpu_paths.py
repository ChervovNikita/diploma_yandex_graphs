"""CPU-only full forward/backward cost preflight; never scores predictions.

Use --npz for feature/edge tensors from a permitted graph artifact. Labels
are generated synthetically; held-out labels are never read. No optimizer
update or model checkpoint is made. CUDA timing is intentionally absent.
"""

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from pathlib import Path

import torch

from centered_paths import (CenteredPathEnsemble, IndependentNarrowPaths,
                            MeanOperator, PackedIndependentSAGEEnsemble, PackedSharedSAGEEnsemble,
                            PlainSAGE, RepeatedFullWidthEnsemble, mean_member_loss,
                            nearest_parameter_width, source_sage_parameter_count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path)
    p.add_argument("--feature-key", default="node_features")
    p.add_argument("--edge-key", default="edges")
    p.add_argument("--classes", type=int, default=18)
    p.add_argument("--nodes", type=int, default=4096)
    p.add_argument("--edges", type=int, default=32768)
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("CPU_PREFLIGHT.json"))
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a previous cost record")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20260930)
    random.seed(20260930)
    data_hash = None
    if args.npz:
        import numpy as np
        data_hash = hashlib.sha256(args.npz.read_bytes()).hexdigest()
        with np.load(args.npz, allow_pickle=False) as archive:
            features = torch.from_numpy(archive[args.feature_key].copy()).float()
            edge_index = torch.from_numpy(archive[args.edge_key].copy()).long()
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T.contiguous()
        if features.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("Expected feature [N,F] and edge [2,E] tensors")
    else:
        features = torch.randn(args.nodes, args.features)
        edge_index = torch.randint(args.nodes, (2, args.edges))
    nodes, inputs = features.shape
    adjacency_start = time.perf_counter()
    operator = MeanOperator.from_edges(edge_index, nodes, dtype=torch.float32)
    adjacency_seconds = time.perf_counter() - adjacency_start
    labels = torch.randint(args.classes, (nodes,))
    train_mask = torch.arange(nodes) < max(1, int(nodes * 0.6))
    kw = dict(input_dim=inputs, output_dim=args.classes, dim=128, members=4,
              layers=2, dropout=0.2)
    models = {
        "centered": CenteredPathEnsemble(**kw, rank=16).cpu(),
        "independent_narrow": IndependentNarrowPaths(**kw, rank=16).cpu(),
        "centered_no_recycling": CenteredPathEnsemble(**kw, rank=16, recycle_mean=False).cpu(),
        "full_width_gnnm_reference": RepeatedFullWidthEnsemble(**kw).cpu(),
        "packed_full_width_ens": PackedIndependentSAGEEnsemble(**kw).cpu(),
        "packed_channel_budget_ens": PackedIndependentSAGEEnsemble(
            **{**kw, "dim": 44}).cpu(),
        "packed_shared_full_width_gnnm": PackedSharedSAGEEnsemble(**kw).cpu(),
        "base_128": PlainSAGE(inputs, args.classes, dim=128, layers=2).cpu(),
    }
    target_parameters = sum(x.numel() for x in models["centered"].parameters())
    packed_budget_width = nearest_parameter_width(target_parameters, inputs, args.classes,
                                                  members=4, maximum=256)
    base_budget_width = nearest_parameter_width(target_parameters, inputs, args.classes)
    models["packed_parameter_budget_ens"] = PackedIndependentSAGEEnsemble(
        **{**kw, "dim": packed_budget_width}).cpu()
    models["base_parameter_budget"] = PlainSAGE(inputs, args.classes,
                                                dim=base_budget_width, layers=2).cpu()
    assert sum(x.numel() for x in models["packed_parameter_budget_ens"].parameters()) == source_sage_parameter_count(
        inputs, args.classes, packed_budget_width, members=4)
    assert sum(x.numel() for x in models["base_parameter_budget"].parameters()) == source_sage_parameter_count(
        inputs, args.classes, base_budget_width)
    report = {"scope": "CPU cost preflight, synthetic labels; no optimization or dataset scores",
              "python": platform.python_version(), "torch": torch.__version__,
              "device": "cpu", "threads": torch.get_num_threads(), "dtype": "float32",
              "graph_path": str(args.npz) if args.npz else None, "graph_sha256": data_hash,
              "nodes": nodes, "edges": edge_index.shape[1], "features": inputs,
              "classes": args.classes, "adjacency_preparation_seconds": adjacency_seconds,
              "matched_budget": {"target_parameters": target_parameters,
                  "packed_width": packed_budget_width, "base_width": base_budget_width,
                  "selection": "Nearest exact parameter count on widths8..maximum in steps4; ties choose smaller width; no label access"},
              "arms": {}}
    excluded = {features.untyped_storage().data_ptr(), operator.adjacency.values().untyped_storage().data_ptr(),
                operator.adjacency.indices().untyped_storage().data_ptr()}
    for name, model in models.items():
        for mode in ("inference", "forward_backward"):
            model.train(mode == "forward_backward")
            for _ in range(2):
                if mode == "inference":
                    with torch.no_grad():
                        model(operator, features).mean(dim=0)
                else:
                    model.zero_grad(set_to_none=True)
                    mean_member_loss(model(operator, features), labels, train_mask).backward()
            report["arms"].setdefault(name, {})[mode + "_milliseconds"] = []
    jobs = [(name, mode) for _ in range(args.repetitions) for name in models
            for mode in ("inference", "forward_backward")]
    random.shuffle(jobs)
    for name, mode in jobs:
        model = models[name]
        model.train(mode == "forward_backward")
        model.zero_grad(set_to_none=True)
        operator.reset_counts()
        start = time.perf_counter()
        if mode == "inference":
            with torch.no_grad():
                model(operator, features).mean(dim=0)
        else:
            mean_member_loss(model(operator, features), labels, train_mask).backward()
        elapsed_ms = (time.perf_counter() - start) * 1000
        report["arms"][name][mode + "_milliseconds"].append(elapsed_ms)
        report["arms"][name]["adjacency_widths"] = list(operator.widths)
    for name, model in models.items():
        row = report["arms"][name]
        row["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        for mode in ("inference", "forward_backward"):
            row[mode + "_median_ms"] = statistics.median(row[mode + "_milliseconds"])
        parameter_storage = {x.untyped_storage().data_ptr() for x in model.parameters()}
        parameter_storage.update(x.untyped_storage().data_ptr() for x in model.buffers())
        saved_storage = {}
        def pack(tensor):
            if tensor.layout == torch.strided:
                storage = tensor.untyped_storage()
                pointer = storage.data_ptr()
                if pointer not in excluded and pointer not in parameter_storage:
                    saved_storage[pointer] = storage.nbytes()
            return tensor
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = mean_member_loss(model(operator, features), labels, train_mask)
        row["retained_autograd_storage_bytes_excluding_parameters_and_cached_inputs"] = sum(saved_storage.values())
        loss.backward()
        row["memory_scope"] = "Retained saved tensor storage before backward, deduplicated by storage pointer. This is not peak RSS or inference peak memory."
    reference = report["arms"]["full_width_gnnm_reference"]
    for row in report["arms"].values():
        row["inference_ratio_to_full_width_reference"] = row["inference_median_ms"] / reference["inference_median_ms"]
        row["forward_backward_ratio_to_full_width_reference"] = row["forward_backward_median_ms"] / reference["forward_backward_median_ms"]
        row["retained_autograd_storage_ratio_to_full_width_reference"] = row["retained_autograd_storage_bytes_excluding_parameters_and_cached_inputs"] / reference["retained_autograd_storage_bytes_excluding_parameters_and_cached_inputs"]
    report["source_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ["centered_paths.py", "profile_cpu_paths.py"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
