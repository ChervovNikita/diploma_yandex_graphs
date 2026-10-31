"""PPI train-only P10/Q10 sharing probe; no validation or test loader path.

Use `freeze` before `probe`. A probe fits P10 only for 25 epochs, records
epochs 0/10/25, then replays the fixed recurrence rule. This script is not a
final-model runner and never selects a checkpoint using predictive scores.
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

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
sys.path.insert(0, str(PHASE / "protocols"))

import torch
from torch.nn import functional as F
from torch_geometric.datasets import PPI

from baseline_adapters import ProjectorEnsemble, SelectiveProjectorEnsemble, model_options
from recurrence_collector import TrainingUnit, collect_record, save_record, tensor_hash
from replay_recurrence import replay_and_decide


RECIPE = {
    "layers": 3, "hidden_dim": 128, "members": 4, "dropout": 0.2,
    "hidden_dim_multiplier": 1, "optimizer": "AdamW",
    "learning_rate": 0.001, "weight_decay": 0.0, "epsilon": 1e-8,
    "probe_epochs": [0, 10, 25], "optimizer_seeds": [0, 1, 2],
    "label_fractions": [0.1, 1.0], "fit_graph_count": 10, "score_graph_count": 10,
    "mask_seed_rule": "9701 + official training split graph index",
    "epoch_order_rule": "shuffle all20 with Random(19709+1000*optimizer_seed+epoch), then retain P10",
    "fit_loss": "native mean of 4 member BCEs, one complete P graph per step, observed nodes only",
    "probe_loss": "unscaled own member equal-graph mean BCE; each graph mean uses finite observed entries only",
    "selection": "recurrence-budget-v2; q>0.5 and positive median recurrence in all3 partitions; rank (S-N)/added_parameters",
    "budget": "at most one private source residual block, 3*p0 additional parameters",
    "label_fraction_budget_rule": "separate prospectively fixed budget and selector lock for each 0.1/1.0 intervention; no pooling",
    "fixed_sharing_family": ["tied", "private0", "private1", "private2", "fully_private"],
    "secondary_diagnostic": "symmetric virtual native Adam first-step linearized loss benefit; zero moments, lr0.001, epsilon1e-8; uses existing fold vectors; not primary selection",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def research_path(path):
    value = Path(path).resolve()
    if PHASE not in value.parents:
        raise ValueError("PPI input must stay inside the permitted research phase")
    return value


def output_path(path):
    value = Path(path).resolve()
    if HERE not in value.parents:
        raise ValueError("PPI probe outputs must stay inside sharing_policy")
    return value


def sources():
    files = [Path(__file__), HERE / "recurrence_collector.py", HERE / "replay_recurrence.py",
             HERE / "replay_ppi_probe_gradients.py",
             PHASE / "protocols" / "baseline_adapters.py", PHASE / "source_snapshot" / "models.py"]
    return {str(path.relative_to(PHASE)): sha(path) for path in files}


def synchronized(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_gate(args):
    if args.device == "cpu":
        return
    if args.device != "cuda:0" or args.physical_gpu is None:
        raise ValueError("Use CPU or explicit logical cuda:0 / physical GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise ValueError("Expose only the explicitly authorized physical GPU")
    result = subprocess.run(["nvidia-smi", "-i", str(args.physical_gpu),
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    memory, utilization = map(int, result.stdout.strip().split(","))
    if memory > 100 or utilization != 0:
        raise RuntimeError("Selected GPU is occupied; no graph/model moved to CUDA")


def load_train_only(manifest):
    manifest = research_path(manifest)
    metadata = json.loads(manifest.read_text())
    if metadata["dataset"] != "ppi" or metadata["scoring_performed"]:
        raise ValueError("Require the new-study unscored frozen PPI manifest")
    cache = manifest.parent
    # Byte fingerprints include official provider files, but only the train
    # dataset loader is constructed; no held-out labels become model inputs.
    for item in metadata["raw_and_processed_files"]:
        if sha(research_path(cache / item["relative_path"])) != item["sha256"]:
            raise ValueError("PPI provider bytes changed after freeze")
    dataset = PPI(root=str(cache / "provider"), split="train")
    if len(dataset) != 20:
        raise ValueError("Expected official PPI20 training graphs")
    graphs = [graph for graph in dataset]
    for graph in graphs:
        if graph.x.ndim != 2 or graph.x.shape[1] != 50 or graph.y.shape != (graph.num_nodes, 121):
            raise ValueError("Unexpected PPI feature/target shape")
        if not bool(torch.isfinite(graph.y).all()) or not bool(((graph.y == 0) | (graph.y == 1)).all()):
            raise ValueError("Unexpected PPI binary targets")
    return graphs


def unit_id(index):
    return f"ppi:train:{index:02d}"


def fixed_folds():
    order = list(range(20))
    random.Random(700).shuffle(order)
    fit, score = sorted(order[:10]), sorted(order[10:])
    partitions = []
    for seed in (701, 702, 703):
        pair = list(score)
        random.Random(seed).shuffle(pair)
        partitions.append({"partition_id": str(seed),
                           "A": [unit_id(i) for i in sorted(pair[:5])],
                           "B": [unit_id(i) for i in sorted(pair[5:])]})
    return {"probe_fit_unit_ids": [unit_id(i) for i in fit],
            "score_unit_ids": [unit_id(i) for i in score], "partitions": partitions,
            "fit_split_indices": fit, "raw_graph_id_rule": "training split index + 1"}


def node_masks(graphs, fraction):
    # Exact nested mask convention used by root protocols/ppi_pilot.py.
    masks = []
    for index, graph in enumerate(graphs):
        generator = torch.Generator().manual_seed(9701 + index)
        selected = torch.randperm(graph.num_nodes, generator=generator)[:max(1, int(graph.num_nodes * fraction))]
        mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask[selected] = True
        masks.append(mask)
    return masks


def mask_manifest(graphs, masks):
    if len(graphs) != len(masks):
        raise ValueError("Graph/mask counts differ")
    return [{"unit_id": unit_id(i), "total_nodes": int(graph.num_nodes),
             "observed_nodes": int(mask.sum()), "mask": tensor_hash(mask)}
            for i, (graph, mask) in enumerate(zip(graphs, masks))]


def frozen_protocol(manifest, graphs):
    return {"schema": "gnnm-ppi-sharing-probe-v2", "recipe": RECIPE,
            "source_sha256": sources(), "data_manifest_sha256": sha(manifest),
            "folds": fixed_folds(),
            "label_masks": {str(fraction): mask_manifest(graphs, node_masks(graphs, fraction))
                            for fraction in RECIPE["label_fractions"]},
            "torch_version": torch.__version__,
            "validation_loader_accessed": False, "test_loader_accessed": False,
            "hidden_node_labels": "Replaced by NaN in all supplied training-unit targets before model/probe use"}


def train_member_loss(logits, safe_targets, mask):
    selected = logits[:, mask, :]
    targets = safe_targets[mask].to(selected.dtype)
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("An observed training label is missing")
    return F.binary_cross_entropy_with_logits(selected, targets[None].expand_as(selected))


def masked_graphs(graphs, masks, device):
    if len(graphs) != len(masks):
        raise ValueError("Graph/mask counts differ")
    safe_graphs = []
    for graph, mask in zip(graphs, masks):
        safe = graph.clone()
        targets = graph.y.clone()
        targets[~mask] = float("nan")
        safe.y = targets
        safe_graphs.append(safe.to(device))
    return safe_graphs


def probe_model(device):
    original = ProjectorEnsemble(model_options(50, 121, hidden_dim=128, layers=3, dropout=0.2))
    model = SelectiveProjectorEnsemble(original)
    del original
    return model.to(device)


def run_probe(args, manifest, graphs, protocol):
    output = output_path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite a probe run")
    actual = frozen_protocol(manifest, graphs)
    if protocol != actual:
        raise ValueError("Source/data/folds/masks/runtime differ from prospective probe freeze")
    gpu_gate(args)
    torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    masks = node_masks(graphs, args.label_fraction)
    # Keep full graphs/features but physically hide all unobserved train
    # labels in every object passed to training or the recurrence collector.
    safe_graphs = masked_graphs(graphs, masks, device)
    masks_on_device = [mask.to(device) for mask in masks]
    model = probe_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0, eps=1e-8)
    block_counts = {key: sum(p.numel() for p in block) for key, block in model.probe_blocks().items()}
    if len(block_counts) != 3 or len(set(block_counts.values())) != 1:
        raise ValueError("First PPI policy expects three equal-cost source blocks")
    private_budget = 3 * block_counts["0"]
    units = [TrainingUnit(unit_id(i), graph, graph.x, graph.y) for i, graph in enumerate(safe_graphs)]
    folds = protocol["folds"]
    output.mkdir(parents=True)
    for relative in protocol["source_sha256"]:
        source, destination = PHASE / relative, output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    torch.save({"masks": masks, "label_fraction": args.label_fraction,
                "unit_ids": [unit_id(i) for i in range(20)]}, output / "label_masks.pt")
    contract = {"scope": "PPI training-only sharing selection probe",
                "seed": args.seed, "label_fraction": args.label_fraction, "recipe": RECIPE,
                "protocol_sha256": sha(args.protocol), "data_manifest_sha256": sha(manifest),
                "source_sha256": protocol["source_sha256"], "folds": folds,
                "mask_manifest": protocol["label_masks"][str(args.label_fraction)],
                "label_masks_sha256": sha(output / "label_masks.pt"),
                "private_parameter_budget": private_budget, "device": str(device),
                "torch_version": torch.__version__, "validation_or_test_scored": False}
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    synchronized(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    overall = time.perf_counter()
    records, events = [], []

    def checkpoint_and_collect(epoch):
        synchronized(device)
        begin = time.perf_counter()
        checkpoint = output / f"probe_epoch{epoch}.pt"
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "seed": args.seed, "label_fraction": args.label_fraction,
                    "protocol_sha256": contract["protocol_sha256"],
                    "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
                    "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, checkpoint)
        checkpoint_seconds = time.perf_counter() - begin
        synchronized(device)
        begin = time.perf_counter()
        record = collect_record(model, units, folds["probe_fit_unit_ids"], folds["partitions"],
                                epoch, args.seed,
                                {"dataset": "PPI", "data_manifest_sha256": sha(manifest),
                                 "label_fraction": args.label_fraction,
                                 "protocol_sha256": contract["protocol_sha256"]})
        record["probe_checkpoint_sha256"] = sha(checkpoint)
        record["probe_checkpoint_filename"] = checkpoint.name
        synchronized(device)
        collect_seconds = time.perf_counter() - begin
        save_record(record, output / f"probe_epoch{epoch}.json")
        records.append(record)
        events.append({"epoch": epoch, "checkpoint_seconds": checkpoint_seconds,
                       "collection_seconds": collect_seconds,
                       "elapsed_seconds": time.perf_counter() - overall})

    try:
        checkpoint_and_collect(0)
        with (output / "fit_trace.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "mean_graph_step_bce", "fit_steps", "fit_epoch_seconds"))
            writer.writeheader()
            for epoch in range(1, 26):
                order = list(range(20))
                random.Random(19709 + args.seed * 1000 + epoch).shuffle(order)
                order = [index for index in order if index in folds["fit_split_indices"]]
                model.train()
                synchronized(device)
                begin = time.perf_counter()
                losses = []
                for index in order:
                    optimizer.zero_grad(set_to_none=True)
                    graph = safe_graphs[index]
                    logits = model(graph, graph.x)
                    loss = train_member_loss(logits, graph.y, masks_on_device[index])
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite P-only fitting loss")
                    loss.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise ValueError("Nonfinite P-only fitting gradient")
                    optimizer.step()
                    losses.append(float(loss.detach()))
                synchronized(device)
                writer.writerow({"epoch": epoch, "mean_graph_step_bce": sum(losses) / len(losses),
                                 "fit_steps": len(losses), "fit_epoch_seconds": time.perf_counter() - begin})
                stream.flush()
                if epoch in (10, 25):
                    checkpoint_and_collect(epoch)
        synchronized(device)
        decision = replay_and_decide(records, private_budget)
        decision["input_record_sha256"] = {f"probe_epoch{epoch}.json": sha(output / f"probe_epoch{epoch}.json")
                                           for epoch in (0, 10, 25)}
        decision["replay_source_sha256"] = sha(HERE / "replay_recurrence.py")
        with (output / "selector_lock.json").open("x") as stream:
            json.dump(decision, stream, indent=2, allow_nan=False)
            stream.write("\n")
        result = {"status": "completed_train_only_probe", "seed": args.seed,
                  "label_fraction": args.label_fraction, "private_indices": decision["private_indices"],
                  "selector_lock_sha256": sha(output / "selector_lock.json"),
                  "fit_trace_sha256": sha(output / "fit_trace.csv"), "events": events,
                  "total_seconds": time.perf_counter() - overall,
                  "wall_seconds_including_setup": time.perf_counter() - args.command_start,
                  "fit_graph_steps": 250, "member_fold_gradient_evaluations": 72,
                  "member_graph_gradient_evaluations": 360,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  "validation_loader_accessed": False, "test_loader_accessed": False,
                  "source_sha256": protocol["source_sha256"]}
        with (output / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, indent=2))
    except Exception as error:
        with (output / "failure.json").open("x") as stream:
            json.dump({"type": type(error).__name__, "message": str(error),
                       "elapsed_seconds": time.perf_counter() - overall, "events": events,
                       "validation_or_test_scored": False}, stream, indent=2)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "probe"))
    parser.add_argument("--data-manifest", type=Path, default=PHASE / "new_tasks" / "data" / "ppi" / "frozen_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--label-fraction", type=float, choices=(0.1, 1.0), default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    args.command_start = time.perf_counter()
    graphs = load_train_only(args.data_manifest)
    if args.command == "freeze":
        output = output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(frozen_protocol(args.data_manifest, graphs), stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"protocol": str(output), "sha256": sha(output), "training_performed": False}, indent=2))
    else:
        if args.protocol is None:
            raise ValueError("The probe command requires a prospective protocol")
        protocol = json.loads(research_path(args.protocol).read_text())
        run_probe(args, args.data_manifest, graphs, protocol)


if __name__ == "__main__":
    main()
