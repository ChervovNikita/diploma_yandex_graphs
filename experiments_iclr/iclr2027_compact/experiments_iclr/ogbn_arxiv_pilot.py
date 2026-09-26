"""A fixed-protocol, full-batch ogbn-arxiv pilot for BASE, ENS, and GNNM.

This script uses the models in the repository root. It is deliberately separate
from the five HeterophilousGraphDataset runs reconstructed elsewhere. The three
variants use one graph, one official OGB split, one training configuration, and
validation-only checkpoint selection. ENS and GNNM each select the epoch of
their *pooled* validation logits. Test labels are held on CPU until that epoch
has been chosen and its checkpoint restored.

Run from the repository root. All data, temporary files, and results stay
inside this repository. The default is three optimization seeds. See
ogbn_arxiv_protocol.md before interpreting any result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import Model, TABMModel  # noqa: E402

MEMBERS = 4
VARIANTS = ("base", "ens", "gnnm")
EPOCH_COLUMNS = (
    "epoch", "train_mean_member_ce", "valid_pooled_accuracy",
    "valid_pooled_ce", "train_step_seconds", "validation_seconds",
    "elapsed_seconds", "is_selected_so_far",
)
SUMMARY_COLUMNS = (
    "seed", "variant", "selected_epoch", "valid_accuracy", "valid_ce",
    "test_accuracy", "test_ce", "parameter_count", "trainable_parameter_count",
    "epochs_run", "train_step_seconds_total", "validation_seconds_total",
    "total_seconds", "mean_train_step_ms", "median_train_step_ms",
    "peak_allocated_mib", "checkpoint_sha256",
)


def repo_path(raw: str | Path) -> Path:
    """Reject CLI paths outside the repository, including symlink escapes."""
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path must stay inside the repository: {raw}")
    return path


def set_repo_temp() -> None:
    temp_root = repo_path("experiments_iclr/.tmp/ogbn_arxiv")
    temp_root.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME"):
        os.environ[name] = str(temp_root)
    tempfile.tempdir = str(temp_root)


def seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    """Fingerprint values, shape, and dtype while they still reside on CPU."""
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_tmp = path.with_suffix(path.suffix + ".tmp")
    target_tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    target_tmp.replace(path)


@dataclass
class Configuration:
    backbone: str = "SAGE"
    layers: int = 2
    hidden_dim: int = 128
    dropout: float = 0.2
    lr: float = 0.001
    weight_decay: float = 0.0
    max_epochs: int = 100
    min_epochs: int = 20
    patience: int = 20
    eval_every: int = 1


@dataclass
class GraphBundle:
    graph: Any
    x: torch.Tensor
    train_idx: torch.Tensor
    valid_idx: torch.Tensor
    test_idx_cpu: torch.Tensor
    train_y: torch.Tensor
    valid_y: torch.Tensor
    test_y_cpu: torch.Tensor
    num_classes: int


def load_official_ogbn_arxiv(data_root: Path, device: torch.device) -> tuple[GraphBundle, dict[str, Any]]:
    try:
        import ogb
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise RuntimeError(
            "The repo-local .venv needs ogb. From the repository root, use "
            "TMPDIR=$PWD/experiments_iclr/.tmp PIP_NO_CACHE_DIR=1 "
            ".venv/bin/python -m pip install ogb"
        ) from exc
    from torch_geometric.utils import to_undirected

    data_root = repo_path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(data_root))
    data = dataset[0]
    split = dataset.get_idx_split()
    expected = {"train", "valid", "test"}
    if set(split) != expected:
        raise RuntimeError(f"Unexpected OGB split keys: {set(split)}")

    n = int(data.num_nodes)
    idx_cpu = {key: split[key].view(-1).long().cpu() for key in expected}
    for key, idx in idx_cpu.items():
        if idx.numel() == 0 or idx.min() < 0 or idx.max() >= n:
            raise RuntimeError(f"Invalid {key} indices in official OGB split")
        if idx.unique().numel() != idx.numel():
            raise RuntimeError(f"Duplicate {key} indices in official OGB split")
    all_indices = torch.cat(list(idx_cpu.values()))
    if all_indices.numel() != n or all_indices.unique().numel() != n:
        raise RuntimeError("Official split is not a disjoint partition of nodes")

    labels_cpu = data.y.view(-1).long().cpu()
    if labels_cpu.numel() != n:
        raise RuntimeError("Node and label counts differ")
    raw_edges = data.edge_index.cpu()
    raw_edge_count = int(raw_edges.size(1))
    undirected_edges = to_undirected(raw_edges, num_nodes=n)
    graph = SimpleNamespace(edge_index=undirected_edges.to(device))
    bundle = GraphBundle(
        graph=graph,
        x=data.x.float().to(device),
        train_idx=idx_cpu["train"].to(device),
        valid_idx=idx_cpu["valid"].to(device),
        test_idx_cpu=idx_cpu["test"],
        train_y=labels_cpu[idx_cpu["train"]].to(device),
        valid_y=labels_cpu[idx_cpu["valid"]].to(device),
        test_y_cpu=labels_cpu[idx_cpu["test"]],
        num_classes=int(dataset.num_classes),
    )
    manifest = {
        "dataset": "ogbn-arxiv",
        "dataset_package": "ogb",
        "ogb_version": getattr(ogb, "__version__", "unknown"),
        "split_source": "PygNodePropPredDataset.get_idx_split()",
        "graph_processing": "torch_geometric.utils.to_undirected on the OGB edge_index; no explicit self loops",
        "num_nodes": n,
        "num_features": int(data.x.size(1)),
        "num_classes": int(dataset.num_classes),
        "raw_directed_edges": raw_edge_count,
        "training_undirected_edges": int(undirected_edges.size(1)),
        "split_sizes": {key: int(idx_cpu[key].numel()) for key in ("train", "valid", "test")},
        "fingerprints_sha256": {
            "x": sha256_tensor(data.x),
            "raw_edge_index": sha256_tensor(raw_edges),
            "y": sha256_tensor(data.y),
            **{f"{key}_index": sha256_tensor(idx_cpu[key]) for key in expected},
        },
    }
    return bundle, manifest


def model_factory(variant: str, bundle: GraphBundle, cfg: Configuration, device: torch.device) -> nn.Module:
    common = dict(
        model_name=cfg.backbone,
        num_layers=cfg.layers,
        input_dim=int(bundle.x.size(1)),
        hidden_dim=cfg.hidden_dim,
        output_dim=bundle.num_classes,
        hidden_dim_multiplier=1,
        num_heads=8,
        normalization="LayerNorm",
        dropout=cfg.dropout,
    )
    if variant == "base":
        return Model(**common).to(device)
    if variant == "ens":
        return nn.ModuleList([Model(**common) for _ in range(MEMBERS)]).to(device)
    if variant == "gnnm":
        return TABMModel(**common, tabm_inits=MEMBERS, device=device).to(device)
    raise ValueError(variant)


def forward_members(model: nn.Module, variant: str, bundle: GraphBundle):
    """Yield one full-node [N, C] logit matrix per member."""
    if variant == "base":
        yield model(bundle.graph, bundle.x)
    elif variant == "ens":
        for member in model:
            yield member(bundle.graph, bundle.x)
    elif variant == "gnnm":
        for member_idx in range(MEMBERS):
            yield model(bundle.graph, bundle.x, tabm_seed=member_idx)
    else:
        raise ValueError(variant)


def pooled_metrics(member_logits: Sequence[torch.Tensor], labels: torch.Tensor) -> tuple[float, float]:
    """Compute accuracy and CE after averaging logits, never probabilities."""
    if not member_logits:
        raise ValueError("At least one member is required")
    logits = torch.stack(tuple(member_logits), dim=0).mean(dim=0)
    if logits.ndim != 2 or labels.ndim != 1 or logits.size(0) != labels.numel():
        raise ValueError("Expected member logits [nodes, classes] and labels [nodes]")
    accuracy = float((logits.argmax(dim=-1) == labels).float().mean().item())
    ce = float(F.cross_entropy(logits, labels).item())
    return accuracy, ce


def better_validation(acc: float, ce: float, best_acc: float, best_ce: float) -> bool:
    """Validation accuracy first, validation CE on ties, earliest epoch last."""
    tol = 1e-12
    return acc > best_acc + tol or (abs(acc - best_acc) <= tol and ce < best_ce - tol)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model: nn.Module, variant: str, bundle: GraphBundle, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for full_logits in forward_members(model, variant, bundle):
        loss = F.cross_entropy(full_logits.index_select(0, bundle.train_idx), bundle.train_y)
        # Four-member variants optimize the same mean-member objective.
        (loss / (1 if variant == "base" else MEMBERS)).backward()
        losses.append(float(loss.detach().item()))
    optimizer.step()
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_pool(model: nn.Module, variant: str, bundle: GraphBundle,
                  idx: torch.Tensor, labels: torch.Tensor,
                  retain_member_logits: bool = False) -> tuple[float, float, torch.Tensor | None]:
    model.eval()
    member_logits = [full.index_select(0, idx) for full in forward_members(model, variant, bundle)]
    acc, ce = pooled_metrics(member_logits, labels)
    if retain_member_logits:
        return acc, ce, torch.stack(member_logits).detach().cpu()
    return acc, ce, None


def run_one(bundle: GraphBundle, cfg: Configuration, variant: str, seed: int,
            device: torch.device, out_dir: Path, save_logits: bool = True) -> dict[str, Any]:
    """Train, select using validation, restore, then score test once."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    out_dir = repo_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed, device)
    model = model_factory(variant, bundle, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    bad_evaluations = 0
    rows: list[dict[str, Any]] = []
    checkpoint = out_dir / "selected_checkpoint.pt"
    run_start = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        synchronize(device)
        train_start = time.perf_counter()
        train_ce = train_epoch(model, variant, bundle, optimizer)
        synchronize(device)
        train_seconds = time.perf_counter() - train_start
        valid_seconds = 0.0
        val_acc = val_ce = None
        selected_now = False
        if epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs:
            synchronize(device)
            valid_start = time.perf_counter()
            val_acc, val_ce, _ = evaluate_pool(
                model, variant, bundle, bundle.valid_idx, bundle.valid_y)
            synchronize(device)
            valid_seconds = time.perf_counter() - valid_start
            if better_validation(val_acc, val_ce, best_acc, best_ce):
                best_acc, best_ce, best_epoch = val_acc, val_ce, epoch
                bad_evaluations = 0
                selected_now = True
                # One state for the whole ENS/GNNM at this pooled-validation epoch.
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "variant": variant, "seed": seed}, checkpoint)
            else:
                bad_evaluations += 1
        row = {
            "epoch": epoch,
            "train_mean_member_ce": train_ce,
            "valid_pooled_accuracy": val_acc,
            "valid_pooled_ce": val_ce,
            "train_step_seconds": train_seconds,
            "validation_seconds": valid_seconds,
            "elapsed_seconds": time.perf_counter() - run_start,
            "is_selected_so_far": int(selected_now),
        }
        rows.append(row)
        print(json.dumps({"seed": seed, "variant": variant, **row}), flush=True)
        if epoch >= cfg.min_epochs and bad_evaluations >= cfg.patience:
            break

    if best_epoch == 0:
        raise RuntimeError("No validation checkpoint was selected")
    with (out_dir / "epochs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["epoch"] != best_epoch or payload["variant"] != variant or payload["seed"] != seed:
        raise RuntimeError("Checkpoint metadata does not match validation selection")
    model.load_state_dict(payload["state_dict"])
    valid_acc, valid_ce, valid_member_logits = evaluate_pool(
        model, variant, bundle, bundle.valid_idx, bundle.valid_y,
        retain_member_logits=save_logits)
    if abs(valid_acc - best_acc) > 1e-7 or abs(valid_ce - best_ce) > 1e-6:
        raise RuntimeError("Restored checkpoint does not reproduce selected validation result")
    # This is the first point at which test indices and test labels enter scoring.
    test_idx = bundle.test_idx_cpu.to(device)
    test_y = bundle.test_y_cpu.to(device)
    test_acc, test_ce, test_member_logits = evaluate_pool(
        model, variant, bundle, test_idx, test_y,
        retain_member_logits=save_logits)
    synchronize(device)
    total_seconds = time.perf_counter() - run_start
    peak_mib = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda" else None)
    if save_logits:
        assert valid_member_logits is not None and test_member_logits is not None
        np.savez_compressed(
            out_dir / "selected_predictions.npz",
            valid_member_logits=valid_member_logits.numpy(),
            valid_labels=bundle.valid_y.detach().cpu().numpy(),
            valid_indices=bundle.valid_idx.detach().cpu().numpy(),
            test_member_logits=test_member_logits.numpy(),
            test_labels=bundle.test_y_cpu.numpy(),
            test_indices=bundle.test_idx_cpu.numpy(),
        )
    step_times = [float(row["train_step_seconds"]) for row in rows]
    result = {
        "seed": seed,
        "variant": variant,
        "selected_epoch": best_epoch,
        "validation_selection": "max pooled accuracy, then min pooled CE, then earliest epoch",
        "valid_accuracy": valid_acc,
        "valid_ce": valid_ce,
        "test_accuracy": test_acc,
        "test_ce": test_ce,
        "parameter_count": params,
        "trainable_parameter_count": trainable,
        "epochs_run": len(rows),
        "train_step_seconds_total": float(sum(step_times)),
        "validation_seconds_total": float(sum(float(row["validation_seconds"]) for row in rows)),
        "total_seconds": total_seconds,
        "mean_train_step_ms": 1000 * float(np.mean(step_times)),
        "median_train_step_ms": 1000 * float(np.median(step_times)),
        "peak_allocated_mib": peak_mib,
        "checkpoint_sha256": sha256_file(checkpoint),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    }
    write_json(out_dir / "selected.json", result)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_summary(output_root: Path) -> None:
    results = [json.loads(path.read_text()) for path in sorted(output_root.glob("seed_*/**/selected.json"))]
    if not results:
        return
    with (output_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [row for row in results if row["variant"] == variant]
        if rows:
            values = np.array([row["test_accuracy"] for row in rows], dtype=float)
            summaries[variant] = {
                "num_seeds": len(values),
                "seeds": [int(row["seed"]) for row in rows],
                "test_accuracy_mean": float(values.mean()),
                "test_accuracy_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            }
    keyed = {(row["seed"], row["variant"]): row for row in results}
    paired: dict[str, Any] = {}
    for left, right in (("gnnm", "ens"), ("gnnm", "base"), ("ens", "base")):
        seeds = sorted({seed for seed, variant in keyed if variant == left}
                       & {seed for seed, variant in keyed if variant == right})
        if seeds:
            diffs = np.array([keyed[(seed, left)]["test_accuracy"] -
                              keyed[(seed, right)]["test_accuracy"] for seed in seeds])
            paired[f"{left}_minus_{right}"] = {
                "seeds": seeds,
                "differences": diffs.tolist(),
                "mean": float(diffs.mean()),
                "sample_sd": float(diffs.std(ddof=1)) if len(diffs) > 1 else None,
            }
    write_json(output_root / "summary.json", {
        "unit_of_replication": "optimization seed on one official temporal graph split",
        "accuracy_units": "fraction correct; multiply by 100 for percentage points",
        "variants": summaries,
        "paired_differences": paired,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="experiments_iclr/data")
    parser.add_argument("--output-root", default="experiments_iclr/ogbn_arxiv_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--backbone", choices=("SAGE", "GCN"), default="SAGE")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--prepare-data-only", action="store_true")
    parser.add_argument("--no-save-logits", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        parser.error("Seeds and variants must each be unique")
    if args.max_epochs < 1 or args.min_epochs < 1 or args.min_epochs > args.max_epochs:
        parser.error("Require 1 <= min_epochs <= max_epochs")
    if args.patience < 1 or args.eval_every < 1 or args.layers < 1 or args.hidden_dim < 1:
        parser.error("Patience, eval_every, layers, and hidden_dim must be positive")
    if not (0 <= args.dropout < 1) or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Invalid dropout, learning rate, or weight decay")
    if args.torch_threads < 0:
        parser.error("torch_threads must be nonnegative")
    return args


def main() -> None:
    args = parse_args()
    set_repo_temp()
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    data_root = repo_path(args.data_root)
    output_root = repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.prepare_data_only else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cfg = Configuration(
        backbone=args.backbone, layers=args.layers, hidden_dim=args.hidden_dim,
        dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, min_epochs=args.min_epochs,
        patience=args.patience, eval_every=args.eval_every,
    )
    fingerprint = {
        "configuration": asdict(cfg),
        "seeds": args.seeds,
        "variants": args.variants,
        "members_for_ens_and_gnnm": MEMBERS,
        "selection_rule": "joint pooled validation accuracy, then pooled CE, then earliest epoch",
        "checkpoint_frequency": f"every {cfg.eval_every} epoch(s) and final epoch",
        "train_objective": "mean member CE for GNNM and ENS; ordinary CE for BASE",
        "torch_version": torch.__version__,
        "torch_geometric_version": __import__("torch_geometric").__version__,
        "source_fingerprints_sha256": {
            "ogbn_arxiv_pilot.py": sha256_file(Path(__file__).resolve()),
            "models.py": sha256_file(ROOT / "models.py"),
        },
    }
    config_path = output_root / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        legacy_config = "source_fingerprints_sha256" not in old
        if legacy_config:
            if any(output_root.glob("seed_*/**/selected.json")):
                raise RuntimeError(
                    "Existing results lack source fingerprints; choose a new output root")
            # The data-preparation-only config predates source fingerprints.
            # It can be adopted safely before any training result exists.
            old["source_fingerprints_sha256"] = fingerprint["source_fingerprints_sha256"]
        # Changing the evaluated seed subset is harmless for continuation.
        if {k: v for k, v in old.items() if k not in ("seeds", "variants")} != {
                k: v for k, v in fingerprint.items() if k not in ("seeds", "variants")}: 
            raise RuntimeError("Output root has a different protocol; choose a new output root")
        if legacy_config:
            write_json(config_path, old)
    else:
        write_json(config_path, fingerprint)

    bundle, manifest = load_official_ogbn_arxiv(data_root, device)
    manifest_path = output_root / "dataset_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Dataset fingerprint changed in an existing output root")
    write_json(manifest_path, manifest)
    print("DATASET", json.dumps(manifest, sort_keys=True), flush=True)
    if args.prepare_data_only:
        return

    for seed in args.seeds:
        for variant in args.variants:
            out_dir = output_root / f"seed_{seed}" / variant
            selected_path = out_dir / "selected.json"
            if selected_path.exists():
                print(f"SKIP completed seed={seed}, variant={variant}", flush=True)
                continue
            result = run_one(bundle, cfg, variant, seed, device, out_dir,
                             save_logits=not args.no_save_logits)
            print("SELECTED", json.dumps(result, sort_keys=True), flush=True)
            write_summary(output_root)
    # Also repair a missing summary when a resumed invocation skips every run.
    write_summary(output_root)


if __name__ == "__main__":
    main()
