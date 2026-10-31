"""Prospective MNIST-superpixel development pilot: freeze/preflight/train/replay.

Default command is a CPU engineering preflight on one full training batch.
No test-subset loader or test-scoring path exists. Training requires a
matching prospective JSON freeze and an explicit idle physical GPU.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.nn import SAGEConv
from shareable_guard import require_development_manifest, require_shareable_manifest

HERE = Path(__file__).resolve().parent
PHASE = next(p for p in HERE.parents if (p / "CONTEXT.md").is_file())
ALLOWED = PHASE / "new_tasks"
SOURCE_ROOT = HERE.parent
SCHEMA = "gnnm-mnist-pilot-v1"
ARMS = ("centered_feedback", "centered_no_feedback", "uncentered_feedback", "uncentered_no_feedback",
        "base", "gnnm", "untied", "first_private", "middle_private", "last_private",
        "multihead", "ens", "packed_gnnm", "packed_full", "packed_channels",
        "packed_parameters", "base_parameters")
RECIPE = {
    "input_dim": 3, "output_dim": 10, "hidden_dim": 128, "num_layers": 3,
    "SAGE_MLP_multiplier": 1, "rank": 16, "members": 4, "dropout": 0.2,
    "normalization": "LayerNorm_per_node",
    "node_features": "released intensity+coordinates3 unchanged, no rescaling",
    "edges": "released directed entries unchanged, no added self-loops; edge_attr retained but unused",
    "readout": "mean_node_logits_after_node_LayerNorm", "pooling": "mean_member_raw_logits",
    "loss": "mean_member_equal_graph_cross_entropy", "optimizer": "AdamW",
    "lr": 0.001, "weight_decay": 0.0, "adam_eps": 1e-8,
    "batch_size": 128, "epochs": 100, "precision": "float32", "AMP": False, "num_workers": 0,
    "checkpoint": "max_pooled_validation_accuracy_then_min_pooled_CE_then_earliest_epoch",
    "label_fractions": [0.1, 1.0], "train_subset_seed": 101,
    "graph_order_seed_rule": "19709 + optimizer_seed*1000 + epoch",
    "parameter_match_target": "centered_feedback registered parameters",
    "channel_match_target": "centered_feedback adjacency width128+(4-1)*16=176; packed member width44",
    "matched_width_rule": "closest exact source count on integer widths8..512; ties smaller; no labels/scores",
    "initialization": "four cells use same parameter draws; TIED/copied-private same initial source; packed full/source parameters copied",
    "purpose": "distinct-corpus transfer of common/narrow paths, not a SOTA MNIST claim",
}


def inside(path):
    value = Path(path).resolve()
    if value != ALLOWED and ALLOWED not in value.parents:
        raise ValueError("Pilot input/output must stay inside this phase's new_tasks directory")
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def source_paths():
    return {"new_tasks/mnist_pilot.py": Path(__file__).resolve(),
            "new_tasks/mnist_adapters.py": HERE / "mnist_adapters.py",
            "new_tasks/shareable_guard.py": HERE / "shareable_guard.py",
            "protocols/baseline_adapters.py": SOURCE_ROOT / "protocols" / "baseline_adapters.py",
            "efficient_paths/centered_paths.py": SOURCE_ROOT / "efficient_paths" / "centered_paths.py",
            "source_snapshot/models.py": SOURCE_ROOT / "source_snapshot" / "models.py"}

def source_hashes():
    return {key: sha(path) for key, path in source_paths().items()}


def runtime_classes():
    return [GNNBenchmarkDataset, SAGEConv, torch.nn.LayerNorm]


def environment():
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "compiled_cuda": torch.version.cuda,
        "default_dtype": str(torch.get_default_dtype()),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                                 if d.metadata["Name"])),
        "runtime_class_source_sha256": {cls.__name__: hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
                                        for cls in runtime_classes()},
        "cuda_context_created_for_capture": False,
    }

def verify_data(path, shareable_required=False):
    path = inside(path)
    data = json.loads(path.read_text())
    require_development_manifest(data, "mnist_superpixels")
    if shareable_required:
        require_shareable_manifest(data, "mnist_superpixels")
    counts = {key: data["split_summaries"][key]["graphs"] for key in ["train", "val", "test"]}
    if counts != {"train": 55000, "val": 5000, "test": 10000}:
        raise ValueError("Require the frozen released55k/5k/10k MNIST split metadata")
    for row in data["archives"]:
        archive = Path(row["path"])
        # Frozen paths are local to the environment that made the data freeze.
        if sha(inside(archive)) != row["sha256"]:
            raise ValueError("Frozen raw archive changed")
    for row in data["raw_and_processed_files"]:
        if sha(inside(path.parent / row["relative_path"])) != row["sha256"]:
            raise ValueError("Frozen raw/processed file changed: " + row["relative_path"])
    for row in data["provider_sources"]:
        if sha(inside(row["path"])) != row["sha256"]:
            raise ValueError("Frozen provider source snapshot changed")
        if row["class"] == "GNNBenchmarkDataset":
            actual = hashlib.sha256(inspect.getsource(GNNBenchmarkDataset).encode()).hexdigest()
            if actual != row["sha256"]:
                raise ValueError("Installed PyG provider source differs from data freeze")
    return data


def add_positions(graph):
    if graph.x.ndim != 2 or graph.x.shape[1] != 1 or graph.pos.ndim != 2 or graph.pos.shape[1] != 2:
        raise ValueError("Frozen MNIST cache must contain released intensity and coordinates")
    graph.x = torch.cat((graph.x, graph.pos), dim=-1)
    return graph


def load_development(manifest):
    # Only existing frozen train/validation files are loaded. Reprocessing,
    # acquisition, and constructing a test split are not supported.
    root = manifest.parent / "provider"
    processed = root / "MNIST" / "processed"
    if not all((processed / name).is_file() for name in ["train_data.pt", "val_data.pt", "test_data.pt"]):
        raise ValueError("Require the complete already processed frozen MNIST cache")
    train = GNNBenchmarkDataset(root=str(root), name="MNIST", split="train", transform=add_positions)
    val = GNNBenchmarkDataset(root=str(root), name="MNIST", split="val", transform=add_positions)
    if len(train) != 55000 or len(val) != 5000:
        raise ValueError("Require released official MNIST55k/5k development splits")
    return None, train, val, torch.arange(len(train)), torch.arange(len(val))


def train_budgets(train, official_indices):
    labels = torch.cat([graph.y.reshape(-1) for graph in train]).cpu()
    if labels.dtype != torch.long or labels.shape != (len(train),):
        raise ValueError("MNIST budgets require exactly one long digit class per training graph")
    if not bool(((labels >= 0) & (labels < 10)).all()):
        raise ValueError("MNIST labels must be digits0..9")
    groups = [(labels == label).nonzero().view(-1) for label in range(10)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Need all ten classes in official training labels")
    total = int(len(train) * 0.1)
    exact = [len(group) * 0.1 for group in groups]
    quotas = [int(value) for value in exact]
    for label in sorted(range(10), key=lambda k: (-(exact[k] - quotas[k]), k))[:total - sum(quotas)]:
        quotas[label] += 1
    selected = []
    for label, group in enumerate(groups):
        generator = torch.Generator().manual_seed(RECIPE["train_subset_seed"] + label)
        selected.append(group[torch.randperm(len(group), generator=generator)[:quotas[label]]])
    ten = torch.cat(selected).sort().values
    full = torch.arange(len(train))
    result = {"0.1": ten, "1.0": full}
    summary = {
        "rule": "floor(total*.1); ten-class quotas via largest fractional remainder; per-class perm seeds101..110; sorted split-local official training positions",
        "graph_index_scope": "split-local indices in released train/validation files; neither mask nor independent graph replicate",
        "train_subset_seed": RECIPE["train_subset_seed"], "nested": True,
        "conditions_are_not_independent_tasks": True, "label_budget_is_not_a_selection_axis": True,
        "full_training_class_counts": [len(group) for group in groups],
        "budgets": {key: {"graphs": len(indices),
                          "class_counts": torch.bincount(labels[indices], minlength=10).tolist(),
                          "training_positions_sha256": tensor_hash(indices),
                          "official_graph_indices_sha256": tensor_hash(official_indices[indices])}
                    for key, indices in result.items()},
    }
    return result, summary


def load_adapter(path=None):
    path = HERE / "mnist_adapters.py" if path is None else path
    spec = importlib.util.spec_from_file_location("mnist_pilot_graph_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(arm, adapter=None):
    adapter = load_adapter() if adapter is None else adapter
    return adapter.make_graph_model(arm)


def comparison_plan(arms):
    return load_adapter().arm_plan(arms)


def graph_member_ce(logits, targets):
    if logits.ndim != 3 or targets.ndim != 1 or targets.shape[0] != logits.shape[1]:
        raise ValueError("Expect member graph logits[M,G,C] and one class target[G]")
    if targets.dtype != torch.long or not bool(((targets >= 0) & (targets < logits.shape[-1])).all()):
        raise ValueError("Graph class targets must be long and within the output class range")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           targets[None].expand(logits.shape[0], -1).reshape(-1))


def score_evidence(member_logits, targets):
    pooled = member_logits.mean(0)
    return {"accuracy": int((pooled.argmax(-1) == targets).sum()) / targets.numel(),
            "pooled_ce": float(graph_member_ce(pooled[None], targets)),
            "mean_member_ce": float(graph_member_ce(member_logits, targets)),
            "mean_member_accuracy": int((member_logits.argmax(-1) == targets[None]).sum()) / (member_logits.shape[0] * targets.numel())}


def evaluate(model, val, device):
    model.eval()
    values, targets = [], []
    with torch.no_grad():
        for graph in DataLoader(val, batch_size=RECIPE["batch_size"], shuffle=False,
                                num_workers=0):
            graph = graph.to(device)
            values.append(model(graph, graph.x).cpu())
            targets.append(graph.y.reshape(-1).long().cpu())
    logits, target = torch.cat(values, dim=1), torch.cat(targets, dim=0)
    return score_evidence(logits, target), logits, target


def selection_key(metrics, epoch):
    if not all(np.isfinite(metrics[key]) for key in ["accuracy", "pooled_ce", "mean_member_ce"]):
        raise ValueError("Nonfinite validation metric")
    return (metrics["accuracy"], -metrics["pooled_ce"], -epoch)

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_guard(args):
    if args.device == "cpu":
        return {"device": "cpu", "gpu_context_created": False}
    if args.device != "cuda:0" or args.physical_gpu is None:
        raise ValueError("CUDA requires logical cuda:0 and explicit --physical-gpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise ValueError("Expose exactly the authorized physical GPU using CUDA_VISIBLE_DEVICES")
    query = subprocess.run(["nvidia-smi", "-i", str(args.physical_gpu),
        "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    fields = [value.strip() for value in query.stdout.strip().split(",")]
    if len(fields) != 5 or int(fields[2]) > 100 or int(fields[4]) != 0:
        raise RuntimeError("Selected GPU is occupied; model has not moved to CUDA")
    apps = subprocess.run(["nvidia-smi", "-i", str(args.physical_gpu),
        "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True)
    if apps.stdout.strip():
        raise RuntimeError("Selected GPU has an active compute context; refusing launch")
    return {"physical_gpu": args.physical_gpu, "uuid": fields[0], "name": fields[1],
            "memory_used_MiB_before_launch": int(fields[2]), "memory_total_MiB": int(fields[3]),
            "gpu_utilization_percent_before_launch": int(fields[4]), "active_compute_apps_before_launch": 0}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def write_json(path, value):
    path = inside(path)
    if path.exists():
        raise ValueError("Refusing to overwrite artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def new_directory(path):
    path = inside(path)
    if path.exists():
        raise ValueError("Refusing to overwrite run directory")
    path.mkdir(parents=True)
    return path


def write_artifact_manifest(output):
    rows = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "artifact_manifest.json"}
    write_json(output / "artifact_manifest.json", rows)


def archive_source(output, sources, data_manifest, env):
    for key, file in source_paths().items():
        target = output / "source" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
        if sha(target) != sources[key]:
            raise ValueError("Source changed while snapshotting")
    shutil.copyfile(data_manifest, output / "data_manifest.json")
    write_json(output / "environment.json", env)
    runtime = output / "source" / "runtime_classes"
    runtime.mkdir(parents=True)
    for cls in runtime_classes():
        (runtime / (cls.__name__ + ".py.txt")).write_text(inspect.getsource(cls))

def freeze(args):
    manifest = inside(args.data_manifest)
    verify_data(manifest, shareable_required=True)
    _, train, _, official_train, official_val = load_development(manifest)
    _, summary = train_budgets(train, official_train)
    env = environment()
    value = {"schema": SCHEMA, "status": "prospective_protocol_not_fit", "dataset": "MNIST_superpixels",
             "frozen_at_UTC": datetime.now(timezone.utc).isoformat(),
             "recipe": RECIPE, "arms": list(args.arms), "seeds": [0, 1, 2],
             "comparison_plan": comparison_plan(args.arms),
             "source_sha256": source_hashes(), "data_manifest_sha256": sha(manifest),
             "environment_sha256": json_hash(env), "environment": env,
             "train_budget_manifest": summary,
             "official_train_indices_sha256": tensor_hash(official_train),
             "official_validation_indices_sha256": tensor_hash(official_val),
             "official_validation_graphs": len(official_val),
             "official_training_graphs": len(official_train), "released_test_graphs_metadata_only": 10000,
             "test_subset_loaded": False, "test_scored": False,
             "confirmation_dataset_accessed": False, "HPO_candidates": 1,
             "full_training_launch_requires_idle_explicit_GPU": True}
    write_json(args.output, value)
    print(json.dumps({"protocol": str(inside(args.output)), "sha256": sha(args.output),
                      "train_budgets": summary["budgets"], "test_scored": False}, indent=2))


def check_protocol(path, args, sources, env, manifest, summary, official_train, official_val):
    protocol = json.loads(inside(path).read_text())
    required = {"schema": SCHEMA, "status": "prospective_protocol_not_fit", "recipe": RECIPE, "source_sha256": sources,
                "data_manifest_sha256": sha(manifest), "environment_sha256": json_hash(env),
                "train_budget_manifest": summary,
                "official_train_indices_sha256": tensor_hash(official_train),
                "official_validation_indices_sha256": tensor_hash(official_val)}
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError("Prospective protocol mismatch: " + key)
    if protocol.get("comparison_plan") != comparison_plan(protocol["arms"]):
        raise ValueError("Prospective parameter/channel plan differs")
    if args.arm not in protocol["arms"] or args.seed not in protocol["seeds"]:
        raise ValueError("Arm/seed not in prospective protocol")
    if args.label_fraction not in protocol["recipe"]["label_fractions"]:
        raise ValueError("Training budget not in prospective protocol")
    return protocol


def run(args, training):
    run_started = time.perf_counter()
    torch.set_num_threads(1)
    manifest = inside(args.data_manifest)
    verify_data(manifest, shareable_required=training)
    dataset, train, val, official_train, official_val = load_development(manifest)
    budgets, summary = train_budgets(train, official_train)
    indices = budgets[str(args.label_fraction)]
    selected_train = train[indices]
    sources, env = source_hashes(), environment()
    protocol = None
    if training:
        if args.protocol is None:
            raise ValueError("Full training requires a prospective matching protocol")
        if args.device != "cuda:0":
            raise ValueError("Full training requires an explicitly selected idle GPU; CPU is for preflight/replay")
        protocol = check_protocol(args.protocol, args, sources, env, manifest, summary, official_train, official_val)
    elif args.device != "cpu":
        raise ValueError("Engineering preflight is CPU-only")
    guard = gpu_guard(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    model, definition = make_model(args.arm)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0, eps=1e-8)
    output = new_directory(args.output)
    archive_source(output, sources, manifest, env)
    if protocol is not None:
        shutil.copyfile(inside(args.protocol), output / "protocol.json")
    torch.save({"training_positions": indices, "official_graph_indices": official_train[indices],
                "full_official_training_indices": official_train, "budgets": budgets,
                "summary": summary}, output / "train_budget.pt")
    initial = {"state_dict": model.state_dict(), "rng": rng_state(device), "arm": args.arm,
               "recipe": RECIPE, "definition": definition}
    torch.save(initial, output / "initial.pt")
    contract = {"schema": SCHEMA, "scope": "development_fit" if training else "engineering_preflight",
                "dataset": "MNIST_superpixels", "arm": args.arm, "seed": args.seed,
                "label_fraction": args.label_fraction, "selected_training_graphs": len(indices),
                "recipe": RECIPE, "arm_definition": definition, "GPU_guard": guard,
                "device": str(device), "parameter_count": sum(p.numel() for p in model.parameters()),
                "initial_sha256": sha(output / "initial.pt"), "budget_sha256": sha(output / "train_budget.pt"),
                "source_sha256": sources, "data_manifest_sha256": sha(manifest),
                "environment_sha256": json_hash(env), "train_budget_manifest": summary,
                "protocol_sha256": sha(args.protocol) if protocol else None,
                "started_at_UTC": datetime.now(timezone.utc).isoformat(),
                "preparation_seconds_at_initial_artifacts": time.perf_counter() - run_started,
                "test_subset_loaded": False, "test_scored": False,
                "conditions_are_not_independent_tasks": True,
                "label_budget_is_not_a_selection_axis": True}
    write_json(output / "contract.json", contract)
    if not training:
        start = time.perf_counter()
        try:
            batch = next(iter(DataLoader(selected_train, batch_size=128, shuffle=False, num_workers=0)))
            if batch.num_graphs != RECIPE["batch_size"]:
                raise ValueError("Preflight requires one complete128-graph minibatch")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch, batch.x)
            loss = graph_member_ce(logits, batch.y.reshape(-1).long())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite preflight loss")
            loss.backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise RuntimeError("Missing/nonfinite registered-parameter preflight gradient")
            optimizer.step()
            report = {"status": "passed", "not_research_evidence": True,
                      "full_training_batch_graphs": batch.num_graphs, "nodes": batch.num_nodes,
                      "directed_edges": batch.num_edges, "logit_shape": list(logits.shape),
                      "finite_training_loss": float(loss.detach()),
                      "architecture": definition, "sparse_operator_widths": model.last_operator_widths,
                      "sparse_operator_build_host_seconds": model.last_operator_build_seconds,
                      "one_step_seconds": time.perf_counter() - start,
                      "optimizer_states_created": len(optimizer.state),
                      "all_registered_parameters_received_finite_gradients": True,
                      "validation_or_test_scored": False}
            write_json(output / "preflight.json", report)
            write_artifact_manifest(output)
            print(json.dumps(report, indent=2))
        except Exception as exc:
            write_json(output / "failure.json", {"scope": "engineering_preflight", "type": type(exc).__name__,
                                               "message": str(exc), "elapsed_seconds": time.perf_counter() - start,
                                               "validation_or_test_scored": False})
            if not (output / "artifact_manifest.json").exists():
                write_artifact_manifest(output)
            raise
        return
    # Only full official validation graphs are scored after each epoch.
    selected_key = None
    validation_targets = None
    evidence_dir = output / "validation_evidence"
    evidence_dir.mkdir()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    try:
        fields = ["epoch", "train_mean_member_ce", "accuracy", "pooled_ce", "mean_member_ce", "mean_member_accuracy",
                  "train_seconds", "validation_seconds", "elapsed_seconds"]
        with (output / "validation_trace.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, 101):
                generator = torch.Generator().manual_seed(19709 + args.seed * 1000 + epoch)
                loader = DataLoader(selected_train, batch_size=128, shuffle=True,
                                    generator=generator, num_workers=0)
                model.train()
                train_loss_sum, train_count = 0.0, 0
                synchronize(device)
                train_started = time.perf_counter()
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = graph_member_ce(model(batch, batch.x), batch.y.reshape(-1).long())
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += float(loss.detach()) * batch.num_graphs
                    train_count += batch.num_graphs
                synchronize(device)
                train_seconds = time.perf_counter() - train_started
                validation_started = time.perf_counter()
                metrics, evidence, targets = evaluate(model, val, device)
                synchronize(device)
                validation_seconds = time.perf_counter() - validation_started
                if epoch == 1:
                    torch.save({"targets": targets, "official_graph_indices": official_val},
                               output / "validation_targets.pt")
                elif not torch.equal(targets, validation_targets):
                    raise RuntimeError("Validation targets/order changed between epochs")
                torch.save({"epoch": epoch, "member_logits": evidence},
                           evidence_dir / f"epoch_{epoch:04d}.pt")
                synchronize(device)
                row = {"epoch": epoch, "train_mean_member_ce": train_loss_sum / train_count,
                       **metrics, "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                       "elapsed_seconds": time.perf_counter() - started}
                writer.writerow(row)
                stream.flush()
                validation_targets = targets
                key = selection_key(metrics, epoch)
                if selected_key is None or key > selected_key:
                    selected_key = key
                    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "epoch": epoch, "validation": metrics, "rng": rng_state(device),
                                "contract_sha256": sha(output / "contract.json")}, output / "selected.pt")
                    torch.save({"member_logits": evidence, "targets": targets,
                                "official_graph_indices": official_val}, output / "selected_validation_member_logits.pt")
                    (output / "selected.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": 100, "rng": rng_state(device)}, output / "last.pt")
        result = {"status": "completed_development_fit", "selected": json.loads((output / "selected.json").read_text()),
                  "selected_sha256": sha(output / "selected.pt"),
                  "member_logits_sha256": sha(output / "selected_validation_member_logits.pt"),
                  "last_sha256": sha(output / "last.pt"), "trace_sha256": sha(output / "validation_trace.csv"),
                  "validation_targets_sha256": tensor_hash(validation_targets),
                  "architecture": definition,
                  "total_seconds": time.perf_counter() - started,
                  "end_to_end_seconds_from_runner_entry": time.perf_counter() - run_started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "test_subset_loaded": False, "test_scored": False}
        write_json(output / "result.json", result)
        write_artifact_manifest(output)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                           "elapsed_seconds": time.perf_counter() - started,
                                           "test_scored": False})
        if not (output / "artifact_manifest.json").exists():
            write_artifact_manifest(output)
        raise


def replay(args):
    torch.set_num_threads(1)
    run_dir = inside(args.run)
    output = inside(args.output)
    if output == run_dir or run_dir in output.parents:
        raise ValueError("Replay output must be outside the immutable fitted run")
    artifact = json.loads((run_dir / "artifact_manifest.json").read_text())
    actual_files = {str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
                    if p.is_file() and p != run_dir / "artifact_manifest.json"}
    if set(artifact) != actual_files:
        raise ValueError("Run artifact manifest does not cover exactly the retained files")
    for relative, expected in artifact.items():
        if sha(inside(run_dir / relative)) != expected:
            raise ValueError("Run artifact changed: " + relative)
    contract = json.loads((run_dir / "contract.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    if contract["scope"] != "development_fit" or result["status"] != "completed_development_fit":
        raise ValueError("Replay requires a complete development fit")
    if contract["recipe"] != RECIPE or sha(Path(__file__)) != contract["source_sha256"]["new_tasks/mnist_pilot.py"]:
        raise ValueError("Use the archived runner to replay a changed source/recipe")
    for key, expected in contract["source_sha256"].items():
        if sha(inside(run_dir / "source" / key)) != expected:
            raise ValueError("Archived source mismatch")
    manifest = inside(args.data_manifest)
    verify_data(manifest)
    if sha(manifest) != contract["data_manifest_sha256"]:
        raise ValueError("Replay data manifest differs from fitted data")
    _, _, val, _, official_val = load_development(manifest)
    checkpoint = torch.load(run_dir / "selected.pt", map_location="cpu")
    if checkpoint["contract_sha256"] != sha(run_dir / "contract.json"):
        raise ValueError("Selected checkpoint contract binding differs")
    retained = torch.load(run_dir / "selected_validation_member_logits.pt", map_location="cpu")
    archived_adapter = load_adapter(run_dir / "source" / "new_tasks" / "mnist_adapters.py")
    model, _ = make_model(contract["arm"], archived_adapter)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics, replayed, targets = evaluate(model, val, torch.device("cpu"))
    if not torch.equal(official_val, retained["official_graph_indices"]) or not torch.equal(targets, retained["targets"]):
        raise ValueError("Validation order/targets differ")
    retained_metrics = score_evidence(retained["member_logits"], retained["targets"])
    selected = json.loads((run_dir / "selected.json").read_text())
    trace = list(csv.DictReader((run_dir / "validation_trace.csv").open()))
    all_targets = torch.load(run_dir / "validation_targets.pt", map_location="cpu")
    if not torch.equal(all_targets["targets"], targets) or not torch.equal(all_targets["official_graph_indices"], official_val):
        raise ValueError("Full validation evidence target/order binding differs")
    if [int(row["epoch"]) for row in trace] != list(range(1, RECIPE["epochs"] + 1)):
        raise ValueError("Validation trace is incomplete or unordered")
    for row in trace:
        epoch = int(row["epoch"])
        evidence = torch.load(run_dir / "validation_evidence" / f"epoch_{epoch:04d}.pt", map_location="cpu")
        if evidence["epoch"] != epoch:
            raise ValueError("Epoch evidence binding differs")
        recomputed = score_evidence(evidence["member_logits"], targets)
        for key, value in recomputed.items():
            if not np.isclose(value, float(row[key]), atol=1e-9, rtol=1e-8):
                raise ValueError("Validation trace metric does not replay: " + key)
    best_row = max(trace, key=lambda row: selection_key({key: float(row[key])
                        for key in ["accuracy", "pooled_ce", "mean_member_ce"]}, int(row["epoch"])))
    if len(trace) != RECIPE["epochs"] or int(best_row["epoch"]) != checkpoint["epoch"] or checkpoint["epoch"] != selected["epoch"]:
        raise ValueError("Trace selection/epoch does not replay")
    if result["selected"] != selected:
        raise ValueError("Run result and selected report differ")
    selected_evidence = torch.load(run_dir / "validation_evidence" / f"epoch_{checkpoint['epoch']:04d}.pt", map_location="cpu")
    if not torch.equal(selected_evidence["member_logits"], retained["member_logits"]):
        raise ValueError("Selected logits differ from the selected epoch evidence")
    for key in retained_metrics:
        if not np.isclose(retained_metrics[key], float(best_row[key]), atol=1e-9, rtol=1e-8):
            raise ValueError("Selected evidence differs from the selected trace row: " + key)
        if not np.isclose(retained_metrics[key], selected[key], atol=1e-9, rtol=1e-8):
            raise ValueError("Selected report metric differs from retained evidence: " + key)
        if not np.isclose(retained_metrics[key], checkpoint["validation"][key], atol=1e-7, rtol=1e-6):
            raise ValueError("Retained evidence metric mismatch: " + key)
    # CPU replay of a CUDA float32 fit is numerically checked, not bitwise.
    if not torch.allclose(replayed, retained["member_logits"], atol=2e-4, rtol=2e-4):
        raise ValueError("Checkpoint member logits do not replay within declared cross-device tolerance")
    report = {"status": "passed", "scope": "selected validation checkpoint/evidence/trace replay",
              "retained_metrics": retained_metrics, "CPU_checkpoint_metrics": metrics,
              "selected_epoch": checkpoint["epoch"],
              "member_logit_max_abs_error": float((replayed - retained["member_logits"]).abs().max()),
              "cross_device_atol_rtol": [2e-4, 2e-4],
              "environment": environment(), "test_scored": False,
              "run_artifact_manifest_sha256": sha(run_dir / "artifact_manifest.json")}
    output = new_directory(output)
    write_json(output / "replay.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "environment"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "preflight", "train", "replay"], nargs="?", default="preflight")
    parser.add_argument("--data-manifest", type=Path, default=ALLOWED / "data" / "mnist_superpixels" / "frozen_shareable_manifest_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="centered_feedback")
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--label-fraction", type=float, choices=[0.1, 1.0], default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "replay":
        if args.run is None:
            parser.error("Replay requires --run")
        replay(args)
    else:
        run(args, training=args.command == "train")


if __name__ == "__main__":
    main()
