"""New-study PPI pilot: train/validation only, immutable run directories.

Default action is an engineering preflight on one complete training graph.
Full training requires --train and a frozen protocol matching this source.
There is deliberately no test-loader or test-scoring path.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import torch
from torch.nn import functional as F
from torch_geometric.datasets import PPI

from baseline_adapters import (SingleModel, ProjectorEnsemble,
    SelectiveProjectorEnsemble, SharedTrunkMultihead, model_options)

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "efficient_paths"))
from centered_paths import (CenteredPathEnsemble, IndependentNarrowPaths,
    PackedIndependentSAGEEnsemble, PackedSharedSAGEEnsemble, PlainSAGE,
    MeanOperator, nearest_parameter_width, mean_member_loss)


ARMS = ("base", "gnnm", "untied", "first_private", "last_private", "multihead",
        "centered", "independent_narrow", "no_recycling", "packed_full",
        "packed_channels", "packed_parameters", "base_parameters")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(path):
    value = Path(path).resolve()
    if PHASE not in value.parents:
        raise ValueError("Input/output must stay inside this research phase")
    return value


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def guard_gpu(args):
    if args.device == "cpu":
        return
    if not args.device.startswith("cuda") or args.physical_gpu is None:
        raise ValueError("CUDA requires an explicit --physical-gpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to the one authorized physical GPU")
    if args.device != "cuda:0":
        raise ValueError("Use logical cuda:0 with the single physical GPU visible")
    query = subprocess.run(["nvidia-smi", "-i", str(args.physical_gpu),
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True)
    memory, utilization = map(int, query.stdout.strip().split(","))
    if memory > 100 or utilization != 0:
        raise RuntimeError("Selected GPU is occupied; no model was moved to CUDA")


def make_model(arm, layers):
    kw = dict(input_dim=50, output_dim=121, dim=128, members=4,
              layers=layers, dropout=0.2)
    options = model_options(50, 121, hidden_dim=128, layers=layers, dropout=0.2)
    if arm == "base":
        return SingleModel(options), "source", {}
    if arm == "gnnm":
        return ProjectorEnsemble(options), "source", {}
    if arm in ("untied", "first_private", "last_private"):
        original = ProjectorEnsemble(options)
        indices = tuple(range(layers)) if arm == "untied" else (0 if arm == "first_private" else layers - 1,)
        return SelectiveProjectorEnsemble(original, indices), "source", {"private_blocks": list(indices)}
    if arm == "multihead":
        return SharedTrunkMultihead(options), "source", {}
    if arm == "centered":
        return CenteredPathEnsemble(**kw, rank=16), "sparse", {}
    if arm == "independent_narrow":
        return IndependentNarrowPaths(**kw, rank=16), "sparse", {}
    if arm == "no_recycling":
        return CenteredPathEnsemble(**kw, rank=16, recycle_mean=False), "sparse", {}
    if arm == "packed_full":
        return PackedIndependentSAGEEnsemble(**kw), "sparse", {}
    if arm == "packed_channels":
        # (128 + (4-1)*16) / 4 = 44 channels per member.
        return PackedIndependentSAGEEnsemble(**{**kw, "dim": 44}), "sparse", {"width": 44}
    target = sum(p.numel() for p in CenteredPathEnsemble(**kw, rank=16).parameters())
    members = 4 if arm == "packed_parameters" else 1
    width = nearest_parameter_width(target, 50, 121, layers=layers, members=members)
    if members == 4:
        model = PackedIndependentSAGEEnsemble(**{**kw, "dim": width})
    else:
        model = PlainSAGE(50, 121, dim=width, layers=layers, dropout=0.2)
    return model, "sparse", {"width": width, "target_parameters": target,
        "parameter_difference": sum(p.numel() for p in model.parameters()) - target,
        "width_rule": "nearest parameter count on prespecified step4 width grid; no label access"}


def forward(model, mode, graph, operator):
    return model(graph, graph.x) if mode == "source" else model(operator, graph.x)


def evaluate(model, mode, graphs, operators):
    model.eval()
    tp = fp = fn = total_loss = entries = 0
    evidence = []
    with torch.no_grad():
        for graph, operator in zip(graphs, operators):
            logits = forward(model, mode, graph, operator)
            pooled = logits.mean(0)
            target = graph.y.bool()
            pred = pooled >= 0
            tp += int((pred & target).sum())
            fp += int((pred & ~target).sum())
            fn += int((~pred & target).sum())
            total_loss += float(F.binary_cross_entropy_with_logits(pooled, graph.y, reduction="sum"))
            entries += graph.y.numel()
            evidence.append(logits.cpu())
    return {"micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "pooled_bce": total_loss / entries}, evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-fraction", type=float, choices=(0.1, 1.0), default=0.1)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--physical-gpu", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train", action="store_true")
    p.add_argument("--protocol", type=Path)
    args = p.parse_args()
    output = inside(args.output)
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing run directory")
    if args.layers != 3 or args.epochs != 300 or args.seed not in (0, 1, 2):
        raise ValueError("First pilot fixes layers3, cap300, seeds0/1/2")
    manifest = PHASE / "new_tasks" / "data" / "ppi" / "frozen_manifest.json"
    data = json.loads(manifest.read_text())
    if data["dataset"] != "ppi" or data["scoring_performed"]:
        raise ValueError("Require the unscored frozen PPI source")
    # Validate the exact frozen provider bytes, not only the manifest name.
    cache = manifest.parent
    for row in data["raw_and_processed_files"]:
        if sha(inside(cache / row["relative_path"])) != row["sha256"]:
            raise ValueError("PPI raw/processed file changed after data freeze")
    source_files = [Path(__file__), PHASE / "protocols" / "baseline_adapters.py",
                    PHASE / "efficient_paths" / "centered_paths.py",
                    PHASE / "source_snapshot" / "models.py"]
    sources = {str(f.relative_to(PHASE)): sha(f) for f in source_files}
    if args.train:
        if args.protocol is None:
            raise ValueError("Training requires a frozen matching protocol")
        protocol = json.loads(inside(args.protocol).read_text())
        if protocol["source_sha256"] != sources or protocol["data_manifest_sha256"] != sha(manifest):
            raise ValueError("Source/data differs from prospective protocol; refreeze before training")
        if args.arm not in protocol["arms"] or args.label_fraction not in protocol["label_fractions"]:
            raise ValueError("Arm/budget is not in the prospective protocol")
        if protocol["epochs"] != args.epochs or protocol["layers"] != args.layers or args.seed not in protocol["seeds"]:
            raise ValueError("Training arguments differ from prospective protocol")
    guard_gpu(args)
    torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_data = PPI(root=str(cache / "provider"), split="train")
    graphs = [g.to(device) for g in train_data]
    operators = [MeanOperator.from_edges(g.edge_index, g.num_nodes,
                   dtype=g.x.dtype, device=device) for g in graphs]
    masks = []
    for index, graph in enumerate(graphs):
        generator = torch.Generator().manual_seed(9701 + index)
        chosen = torch.randperm(graph.num_nodes, generator=generator)[:max(1, int(graph.num_nodes * args.label_fraction))]
        mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask[chosen] = True
        masks.append(mask.to(device))
    model, mode, budget = make_model(args.arm, args.layers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0, eps=1e-8)
    output.mkdir(parents=True)
    for file in source_files:
        dest = output / "source" / file.relative_to(PHASE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, dest)
    torch.save({"state_dict": model.state_dict(), "label_masks": [m.cpu() for m in masks]}, output / "initial.pt")
    contract = {"scope": "new PPI development pilot" if args.train else "one complete training graph engineering preflight",
        "arm": args.arm, "seed": args.seed, "device": str(device), "torch": torch.__version__,
        "label_fraction": args.label_fraction, "members": 1 if args.arm.startswith("base") else 4,
        "layers": args.layers, "hidden_dim_multiplier": 1, "cap_epochs": args.epochs,
        "dropout": 0.2, "optimizer": "AdamW(lr0.001,weight_decay0,eps1e-8)",
        "loss": "mean member BCE on the fixed observed nodes, one complete graph per step",
        "pooling": "mean raw logits, threshold0", "checkpoint": "pooled validation micro-F1; BCE tie-break; earliest epoch",
        "data_manifest_sha256": sha(manifest), "source_sha256": sources,
        "parameters": sum(p.numel() for p in model.parameters()), "budget": budget,
        "initial_checkpoint_sha256": sha(output / "initial.pt"), "test_loader_accessed": False}
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    if not args.train:
        # A full graph, not a truncated data subset. No validation prediction.
        graph, operator, mask = graphs[0], operators[0], masks[0]
        synchronize(device)
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = forward(model, mode, graph, operator)
        loss = mean_member_loss(logits, graph.y, mask)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        synchronize(device)
        report = {"status": "passed", "not_research_evidence": True,
            "complete_training_graph_index": 0, "nodes": graph.num_nodes,
            "edge_entries": graph.edge_index.shape[1], "labeled_nodes": int(mask.sum()),
            "logit_shape": list(logits.shape), "finite_initial_training_loss": float(loss.detach()),
            "one_step_seconds": time.perf_counter() - start,
            "optimizer_state_created": len(optimizer.state),
            "validation_or_test_scored": False}
        (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    val_graphs = [g.to(device) for g in PPI(root=str(cache / "provider"), split="val")]
    val_operators = [MeanOperator.from_edges(g.edge_index, g.num_nodes, dtype=g.x.dtype, device=device)
                     for g in val_graphs]
    best = None
    start = time.perf_counter()
    try:
        with (output / "validation_trace.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "micro_f1", "pooled_bce", "elapsed_seconds"))
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                order = list(range(len(graphs)))
                random.Random(19709 + args.seed * 1000 + epoch).shuffle(order)
                model.train()
                for index in order:
                    optimizer.zero_grad(set_to_none=True)
                    logits = forward(model, mode, graphs[index], operators[index])
                    loss = mean_member_loss(logits, graphs[index].y, masks[index])
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                metrics, evidence = evaluate(model, mode, val_graphs, val_operators)
                synchronize(device)
                row = {"epoch": epoch, **metrics, "elapsed_seconds": time.perf_counter() - start}
                writer.writerow(row)
                stream.flush()
                key = (metrics["micro_f1"], -metrics["pooled_bce"], -epoch)
                if best is None or key > best:
                    best = key
                    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "validation": metrics,
                        "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
                        "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
                        output / "selected.pt")
                    torch.save(evidence, output / "selected_validation_member_logits.pt")
                    (output / "selected.json").write_text(json.dumps(row, indent=2) + "\n")
        selected = json.loads((output / "selected.json").read_text())
        result = {"status": "completed_development_fit", "selected": selected,
            "selected_checkpoint_sha256": sha(output / "selected.pt"),
            "validation_evidence_sha256": sha(output / "selected_validation_member_logits.pt"),
            "total_seconds": time.perf_counter() - start,
            "test_scored": False, "source_sha256": sources}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc),
            "elapsed_seconds": time.perf_counter() - start}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
