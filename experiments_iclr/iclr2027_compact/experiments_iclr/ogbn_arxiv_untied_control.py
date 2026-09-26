"""Matched-initialization ogbn-arxiv GNNM versus untied propagation control.

See ogbn_arxiv_untied_protocol.md. This runner is separate from the pinned
ogbn_arxiv_pilot.py and models.py. The default invocation requires a complete
source lock and a new empty result root. --cpu-smoke does no OGB or GPU work.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import TABMModel  # noqa: E402
from experiments_iclr import ogbn_arxiv_pilot as pilot  # noqa: E402

SEEDS = (0, 1, 2)
ARMS = ("tied", "untied_propagation")
MEMBERS = 4
RESULT_ROOT = Path("experiments_iclr/ogbn_arxiv_untied_results")
LOCK_PATH = Path("experiments_iclr/ogbn_arxiv_untied_source_lock.json")
REFERENCE_CONFIG = Path("experiments_iclr/ogbn_arxiv_300_results/run_config.json")
REFERENCE_DATASET = Path("experiments_iclr/ogbn_arxiv_300_results/dataset_manifest.json")
LOCKED_SOURCES = (
    "models.py",
    "experiments_iclr/ogbn_arxiv_pilot.py",
    "experiments_iclr/ogbn_arxiv_300_results/run_config.json",
    "experiments_iclr/ogbn_arxiv_300_results/dataset_manifest.json",
    "experiments_iclr/ogbn_arxiv_untied_protocol.md",
    "experiments_iclr/ogbn_arxiv_untied_control.py",
    "experiments_iclr/verify_ogbn_arxiv_untied.py",
)
EPOCH_COLUMNS = pilot.EPOCH_COLUMNS
SUMMARY_COLUMNS = (
    "seed", "tied_test_accuracy", "untied_test_accuracy",
    "tied_minus_untied", "tied_selected_epoch", "untied_selected_epoch",
)


def fixed_configuration() -> pilot.Configuration:
    return pilot.Configuration(
        backbone="SAGE", layers=2, hidden_dim=128, dropout=0.2,
        lr=0.001, weight_decay=0.0, max_epochs=300, min_epochs=300,
        patience=20, eval_every=1,
    )


class UntiedPropagationTABM(nn.Module):
    """Only the four residual propagation stacks are independent per member."""

    def __init__(self, tied: TABMModel):
        super().__init__()
        self.model_name = tied.model_name
        self.input_be_block = copy.deepcopy(tied.input_be_block)
        self.dropout = copy.deepcopy(tied.dropout)
        self.act = copy.deepcopy(tied.act)
        self.propagation_stacks = nn.ModuleList(
            [copy.deepcopy(tied.residual_modules) for _ in range(MEMBERS)]
        )
        self.output_normalization = copy.deepcopy(tied.output_normalization)
        self.output_be_block = copy.deepcopy(tied.output_be_block)

    def forward(self, graph: Any, x: torch.Tensor, tabm_seed: int) -> torch.Tensor:
        if not 0 <= tabm_seed < MEMBERS:
            raise ValueError(f"Invalid member index: {tabm_seed}")
        x = self.input_be_block(x, tabm_seed=tabm_seed)
        x = self.dropout(x)
        x = self.act(x)
        for residual_module in self.propagation_stacks[tabm_seed]:
            x = residual_module(graph, x)
        x = self.output_normalization(x)
        return self.output_be_block(x, tabm_seed=tabm_seed).squeeze(1)


def _new_tied(cfg: pilot.Configuration, num_features: int, num_classes: int,
              device: torch.device) -> TABMModel:
    if cfg.backbone != "SAGE" or cfg.layers != 2 or cfg.hidden_dim != 128:
        # The synthetic smoke uses a smaller width and feature space. The
        # production config itself is checked in main() and by the verifier.
        if cfg.max_epochs == 300:
            raise ValueError("Production architecture differs from the fixed protocol")
    return TABMModel(
        model_name=cfg.backbone, num_layers=cfg.layers,
        input_dim=num_features, hidden_dim=cfg.hidden_dim,
        output_dim=num_classes, hidden_dim_multiplier=1, num_heads=8,
        normalization="LayerNorm", dropout=cfg.dropout,
        tabm_inits=MEMBERS, device=device,
    ).to(device)


def build_arm_model(arm: str, cfg: pilot.Configuration, num_features: int,
                    num_classes: int, device: torch.device) -> nn.Module:
    """Stable factory used by the independent artifact verifier."""
    if arm not in ARMS:
        raise ValueError(arm)
    tied = _new_tied(cfg, num_features, num_classes, device)
    return tied if arm == "tied" else UntiedPropagationTABM(tied).to(device)


def build_pair(cfg: pilot.Configuration, num_features: int, num_classes: int,
               device: torch.device) -> tuple[TABMModel, UntiedPropagationTABM]:
    tied = _new_tied(cfg, num_features, num_classes, device)
    untied = UntiedPropagationTABM(tied).to(device)
    return tied, untied


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes()


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def stack_sha256(stack: nn.Module) -> str:
    return state_sha256(stack)


def nonprop_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        if key.startswith("residual_modules.") or key.startswith("propagation_stacks."):
            continue
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def capture_rng(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": tuple(s.clone() for s in torch.cuda.get_rng_state_all())
        if device.type == "cuda" else (),
    }


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(list(state["cuda"]))


def rng_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(pickle.dumps(state["python"], protocol=4))
    name, keys, pos, has_gauss, cached = state["numpy"]
    digest.update(name.encode())
    digest.update(keys.tobytes())
    digest.update(pickle.dumps((pos, has_gauss, cached), protocol=4))
    digest.update(_tensor_bytes(state["torch"]))
    for cuda_state in state["cuda"]:
        digest.update(_tensor_bytes(cuda_state))
    return digest.hexdigest()


def _fixed_synthetic_graph(num_features: int, device: torch.device) -> tuple[Any, torch.Tensor]:
    source = torch.arange(8, dtype=torch.long, device=device)
    target = source + 1
    edges = torch.stack((
        torch.cat((source, target)), torch.cat((target, source))
    ), dim=0)
    x = torch.linspace(-1.0, 1.0, 9 * num_features, device=device).reshape(9, num_features)
    return SimpleNamespace(edge_index=edges), x


def _equal_initial_parameters(tied: TABMModel, untied: UntiedPropagationTABM) -> bool:
    tied_state = tied.state_dict()
    untied_state = untied.state_dict()
    expected: set[str] = set()
    for key, value in tied_state.items():
        if key.startswith("residual_modules."):
            suffix = key[len("residual_modules."):]
            for member in range(MEMBERS):
                other_key = f"propagation_stacks.{member}.{suffix}"
                expected.add(other_key)
                if other_key not in untied_state or not torch.equal(value, untied_state[other_key]):
                    return False
        else:
            expected.add(key)
            if key not in untied_state or not torch.equal(value, untied_state[key]):
                return False
    return expected == set(untied_state)


def _all_distinct_storage(modules: list[nn.Module]) -> bool:
    stores = [[p.untyped_storage().data_ptr() for p in module.parameters()] for module in modules]
    if any(len(ids) != len(set(ids)) for ids in stores):
        return False
    flattened = [pointer for ids in stores for pointer in ids]
    return len(flattened) == len(set(flattened))


def audit_initialization(seed: int, tied: TABMModel, untied: UntiedPropagationTABM,
                         num_features: int, device: torch.device,
                         initial_rng: dict[str, Any]) -> dict[str, Any]:
    same_params = _equal_initial_parameters(tied, untied)
    untied_disjoint = _all_distinct_storage(list(untied.propagation_stacks))
    cross_disjoint = _all_distinct_storage([tied, untied])
    tied_hash = stack_sha256(tied.residual_modules)
    untied_hashes = [stack_sha256(stack) for stack in untied.propagation_stacks]
    nonprop_hashes = {"tied": nonprop_sha256(tied), "untied_propagation": nonprop_sha256(untied)}
    graph, x = _fixed_synthetic_graph(num_features, device)
    tied.eval()
    untied.eval()
    with torch.no_grad():
        deltas = [
            float((tied(graph, x, tabm_seed=m) -
                   untied(graph, x, tabm_seed=m)).abs().max().item())
            for m in range(MEMBERS)
        ]
    restore_rng(initial_rng, device)
    initial_rng_hash = rng_sha256(initial_rng)
    audit = {
        "seed": seed,
        "tied_initial_state_sha256": state_sha256(tied),
        "untied_initial_state_sha256": state_sha256(untied),
        "tied_stack_sha256": tied_hash,
        "untied_stack_sha256": untied_hashes,
        "shared_nonprop_sha256": nonprop_hashes,
        "synthetic_member_max_abs_diff": deltas,
        "all_initial_functions_equal": all(delta <= 1e-7 for delta in deltas),
        "all_parameters_equal_to_tied": same_params,
        "untied_stacks_disjoint": untied_disjoint,
        "cross_arm_storage_disjoint": cross_disjoint,
        "parameter_count": {
            "tied": sum(p.numel() for p in tied.parameters()),
            "untied_propagation": sum(p.numel() for p in untied.parameters()),
        },
        "rng_after_construction_sha256": initial_rng_hash,
        "tied_rng_start_sha256": initial_rng_hash,
        "untied_rng_start_sha256": initial_rng_hash,
    }
    if (not same_params or not untied_disjoint or not cross_disjoint
            or tied_hash not in untied_hashes or any(x != tied_hash for x in untied_hashes)
            or nonprop_hashes["tied"] != nonprop_hashes["untied_propagation"]
            or not audit["all_initial_functions_equal"]):
        raise RuntimeError("Matched-initialization audit failed")
    return audit


def member_forwards(model: nn.Module, graph: Any, x: torch.Tensor):
    for member in range(MEMBERS):
        yield model(graph, x, tabm_seed=member)


def train_epoch(model: nn.Module, bundle: pilot.GraphBundle,
                optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for logits in member_forwards(model, bundle.graph, bundle.x):
        loss = F.cross_entropy(logits.index_select(0, bundle.train_idx), bundle.train_y)
        (loss / MEMBERS).backward()
        losses.append(float(loss.detach().item()))
    optimizer.step()
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_pool(model: nn.Module, bundle: pilot.GraphBundle, idx: torch.Tensor,
                  labels: torch.Tensor, retain_member_logits: bool = False
                  ) -> tuple[float, float, torch.Tensor | None]:
    model.eval()
    members = [
        logits.index_select(0, idx)
        for logits in member_forwards(model, bundle.graph, bundle.x)
    ]
    acc, ce = pilot.pooled_metrics(members, labels)
    return acc, ce, torch.stack(members).detach().cpu() if retain_member_logits else None


def run_one(seed: int, arm: str, model: nn.Module, bundle: pilot.GraphBundle,
            cfg: pilot.Configuration, device: torch.device, out_dir: Path,
            initial_rng: dict[str, Any], initialization_hash: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(arm)
    if out_dir.exists():
        raise FileExistsError(f"Output already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    restore_rng(initial_rng, device)
    start_rng_hash = rng_sha256(capture_rng(device))
    if start_rng_hash != rng_sha256(initial_rng):
        raise RuntimeError("Training RNG was not restored exactly")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    selected_checkpoint = out_dir / "selected_checkpoint.pt"
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        pilot.synchronize(device)
        train_started = time.perf_counter()
        train_ce = train_epoch(model, bundle, optimizer)
        pilot.synchronize(device)
        train_seconds = time.perf_counter() - train_started
        pilot.synchronize(device)
        validation_started = time.perf_counter()
        val_acc, val_ce, _ = evaluate_pool(
            model, bundle, bundle.valid_idx, bundle.valid_y
        )
        pilot.synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
        selected_now = pilot.better_validation(
            val_acc, val_ce, best_acc, best_ce
        )
        if selected_now:
            best_acc, best_ce, best_epoch = val_acc, val_ce, epoch
            torch.save({
                "state_dict": model.state_dict(), "epoch": epoch,
                "arm": arm, "seed": seed,
            }, selected_checkpoint)
        rows.append({
            "epoch": epoch,
            "train_mean_member_ce": train_ce,
            "valid_pooled_accuracy": val_acc,
            "valid_pooled_ce": val_ce,
            "train_step_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "is_selected_so_far": int(selected_now),
        })
        if epoch % 25 == 0 or epoch == cfg.max_epochs:
            print(json.dumps({
                "seed": seed, "arm": arm, "epoch": epoch,
                "valid_accuracy": val_acc, "valid_ce": val_ce,
                "best_epoch": best_epoch,
            }), flush=True)
    if len(rows) != 300 or best_epoch < 1:
        raise RuntimeError("The fixed 300-epoch protocol was not completed")
    with (out_dir / "epochs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    payload = torch.load(selected_checkpoint, map_location=device, weights_only=True)
    if (payload["seed"], payload["arm"], payload["epoch"]) != (seed, arm, best_epoch):
        raise RuntimeError("Checkpoint metadata differs from selected validation epoch")
    model.load_state_dict(payload["state_dict"], strict=True)
    valid_acc, valid_ce, valid_logits = evaluate_pool(
        model, bundle, bundle.valid_idx, bundle.valid_y,
        retain_member_logits=True,
    )
    if abs(valid_acc - best_acc) > 1e-7 or abs(valid_ce - best_ce) > 1e-6:
        raise RuntimeError("Restored checkpoint failed selected validation replay")
    # Official test indices and labels enter scoring only after checkpoint
    # selection, restoration and validation replay have succeeded.
    test_idx = bundle.test_idx_cpu.to(device)
    test_y = bundle.test_y_cpu.to(device)
    test_acc, test_ce, test_logits = evaluate_pool(
        model, bundle, test_idx, test_y, retain_member_logits=True,
    )
    assert valid_logits is not None and test_logits is not None
    pilot.synchronize(device)
    total_seconds = time.perf_counter() - started
    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    np.savez_compressed(
        out_dir / "selected_predictions.npz",
        valid_member_logits=valid_logits.numpy().astype(np.float32, copy=False),
        valid_labels=bundle.valid_y.detach().cpu().numpy().astype(np.int64, copy=False),
        valid_indices=bundle.valid_idx.detach().cpu().numpy().astype(np.int64, copy=False),
        test_member_logits=test_logits.numpy().astype(np.float32, copy=False),
        test_labels=bundle.test_y_cpu.numpy().astype(np.int64, copy=False),
        test_indices=bundle.test_idx_cpu.numpy().astype(np.int64, copy=False),
    )
    step_times = [float(row["train_step_seconds"]) for row in rows]
    result = {
        "seed": seed, "variant": arm, "arm": arm,
        "selected_epoch": best_epoch,
        "validation_selection": "max pooled accuracy, then min pooled CE, then earliest epoch",
        "valid_accuracy": valid_acc, "valid_ce": valid_ce,
        "test_accuracy": test_acc, "test_ce": test_ce,
        "parameter_count": parameters,
        "trainable_parameter_count": trainable,
        "epochs_run": len(rows),
        "train_step_seconds_total": float(sum(step_times)),
        "validation_seconds_total": float(sum(float(row["validation_seconds"]) for row in rows)),
        "total_seconds": total_seconds,
        "mean_train_step_ms": 1000 * float(np.mean(step_times)),
        "median_train_step_ms": 1000 * float(np.median(step_times)),
        "peak_allocated_mib": peak_mib,
        "checkpoint_sha256": pilot.sha256_file(selected_checkpoint),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "rng_start_sha256": start_rng_hash,
        "initialization_sha256": initialization_hash,
    }
    pilot.write_json(out_dir / "selected.json", result)
    return result


def _check_source_lock() -> tuple[dict[str, Any], str]:
    lock_file = pilot.repo_path(LOCK_PATH)
    if not lock_file.is_file() or lock_file.is_symlink():
        raise RuntimeError("Missing or unsafe source lock; complete CPU preflight first")
    lock = json.loads(lock_file.read_text())
    if set(lock) != {"sha256"} or set(lock["sha256"]) != set(LOCKED_SOURCES):
        raise RuntimeError("Source lock key set differs from the protocol")
    for name in LOCKED_SOURCES:
        path = pilot.repo_path(name)
        expected = lock["sha256"][name]
        if (not path.is_file() or path.is_symlink()
                or not isinstance(expected, str) or len(expected) != 64
                or pilot.sha256_file(path) != expected):
            raise RuntimeError(f"Source lock mismatch: {name}")
    pinned = {
        "models.py": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303",
        "experiments_iclr/ogbn_arxiv_pilot.py": "7abf975fcf900f52a01a51652605b2d2519aaee02918948732ee5401952dd89b",
        "experiments_iclr/ogbn_arxiv_300_results/run_config.json": "ee26cd93647d77c964a554960ab4893fdc95e5bd7d73fd9479e0a8c32a1bde97",
        "experiments_iclr/ogbn_arxiv_untied_protocol.md": "8ab706043839c936e74de6919cc33efd17c943248d0a2dee61669fa6140b98f2",
    }
    if any(lock["sha256"][name] != digest for name, digest in pinned.items()):
        raise RuntimeError("Pinned predeclared source hash changed")
    return lock, pilot.sha256_file(lock_file)


def _check_reference_config(cfg: pilot.Configuration) -> None:
    reference = json.loads(pilot.repo_path(REFERENCE_CONFIG).read_text())
    if (reference.get("configuration") != asdict(cfg)
            or reference.get("seeds") != list(SEEDS)
            or reference.get("variants") != ["base", "ens", "gnnm"]
            or reference.get("members_for_ens_and_gnnm") != MEMBERS):
        raise RuntimeError("The existing 300-epoch configuration differs from the frozen control")
    sources = reference.get("source_fingerprints_sha256", {})
    if sources != {
        "ogbn_arxiv_pilot.py": "7abf975fcf900f52a01a51652605b2d2519aaee02918948732ee5401952dd89b",
        "models.py": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303",
    }:
        raise RuntimeError("The existing 300-epoch reference source changed")


def _run_config(cfg: pilot.Configuration, lock_hash: str,
                lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": "ogbn_arxiv_untied_v1",
        "seeds": list(SEEDS), "arms": list(ARMS), "members": MEMBERS,
        "configuration": asdict(cfg),
        "optimizer": {"name": "AdamW", "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        "train_objective": "mean_member_ce",
        "selection_rule": "pooled_validation_accuracy_then_ce_then_earliest",
        "source_lock_sha256": lock_hash,
        "reference_300_config_sha256": lock["sha256"][str(REFERENCE_CONFIG)],
        "reference_300_dataset_manifest_sha256": lock["sha256"][str(REFERENCE_DATASET)],
        "torch_version": torch.__version__,
        "torch_geometric_version": __import__("torch_geometric").__version__,
    }


def _write_summary(root: Path, selected: dict[tuple[int, str], dict[str, Any]]) -> None:
    if set(selected) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise RuntimeError("Cannot summarize an incomplete three-seed pair")
    rows = []
    for seed in SEEDS:
        tied = selected[(seed, "tied")]
        untied = selected[(seed, "untied_propagation")]
        rows.append({
            "seed": seed,
            "tied_test_accuracy": tied["test_accuracy"],
            "untied_test_accuracy": untied["test_accuracy"],
            "tied_minus_untied": tied["test_accuracy"] - untied["test_accuracy"],
            "tied_selected_epoch": tied["selected_epoch"],
            "untied_selected_epoch": untied["selected_epoch"],
        })
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    diffs = np.array([row["tied_minus_untied"] for row in rows], dtype=float)
    pilot.write_json(root / "summary.json", {
        "contrast": "tied_minus_untied_propagation",
        "seeds": list(SEEDS),
        "differences": diffs.tolist(),
        "mean": float(diffs.mean()),
        "sample_sd": float(diffs.std(ddof=1)),
    })


def _write_artifact_manifest(root: Path) -> None:
    manifest = {
        "sha256": {
            str(path.relative_to(root)): pilot.sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        }
    }
    pilot.write_json(root / "artifact_manifest.json", manifest)


def cpu_smoke() -> dict[str, Any]:
    """Check exact pairing and the first mean-member gradient step on CPU."""
    device = torch.device("cpu")
    torch.set_num_threads(1)
    cfg = pilot.Configuration(
        backbone="SAGE", layers=2, hidden_dim=16, dropout=0.2,
        lr=0.001, weight_decay=0.0, max_epochs=1, min_epochs=1,
        patience=1, eval_every=1,
    )
    seed = 37
    pilot.seed_all(seed, device)
    tied, untied = build_pair(cfg, num_features=5, num_classes=3, device=device)
    initial_rng = capture_rng(device)
    audit = audit_initialization(seed, tied, untied, 5, device, initial_rng)
    graph, x = _fixed_synthetic_graph(5, device)
    train_idx = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    def backward_step(model: nn.Module) -> tuple[list[torch.Tensor], float]:
        model.train()
        model.zero_grad(set_to_none=True)
        member_outputs = []
        member_losses = []
        for logits in member_forwards(model, graph, x):
            member_outputs.append(logits.detach().clone())
            loss = F.cross_entropy(logits.index_select(0, train_idx), labels)
            (loss / MEMBERS).backward()
            member_losses.append(float(loss.detach().item()))
        return member_outputs, float(np.mean(member_losses))

    restore_rng(initial_rng, device)
    tied_start_hash = rng_sha256(capture_rng(device))
    tied_logits, tied_loss = backward_step(tied)
    restore_rng(initial_rng, device)
    untied_start_hash = rng_sha256(capture_rng(device))
    untied_logits, untied_loss = backward_step(untied)
    max_train_logit_diff = max(
        float((a - b).abs().max().item()) for a, b in zip(tied_logits, untied_logits)
    )
    max_shared_grad_diff = 0.0
    for name, parameter in tied.named_parameters():
        if name.startswith("residual_modules."):
            continue
        other = untied.get_parameter(name)
        if parameter.grad is None or other.grad is None:
            raise RuntimeError(f"Missing shared-block gradient: {name}")
        max_shared_grad_diff = max(
            max_shared_grad_diff,
            float((parameter.grad - other.grad).abs().max().item()),
        )
    max_stack_grad_diff = 0.0
    for name, parameter in tied.residual_modules.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing tied stack gradient: {name}")
        member_grads = [
            stack.get_parameter(name).grad for stack in untied.propagation_stacks
        ]
        if any(grad is None for grad in member_grads):
            raise RuntimeError(f"Missing untied stack gradient: {name}")
        summed = torch.stack(member_grads).sum(dim=0)
        max_stack_grad_diff = max(
            max_stack_grad_diff,
            float((parameter.grad - summed).abs().max().item()),
        )
    tied_optimizer = torch.optim.AdamW(tied.parameters(), lr=cfg.lr, weight_decay=0)
    untied_optimizer = torch.optim.AdamW(untied.parameters(), lr=cfg.lr, weight_decay=0)
    tied_optimizer.step()
    untied_optimizer.step()
    max_shared_post_step_diff = max(
        float((parameter - untied.get_parameter(name)).abs().max().item())
        for name, parameter in tied.named_parameters()
        if not name.startswith("residual_modules.")
    )
    stacks_differ_after_step = len({
        stack_sha256(stack) for stack in untied.propagation_stacks
    }) > 1
    if (tied_start_hash != untied_start_hash
            or max_train_logit_diff > 1e-6
            or abs(tied_loss - untied_loss) > 1e-7
            or max_shared_grad_diff > 1e-5
            or max_stack_grad_diff > 1e-5
            or max_shared_post_step_diff > 1e-6
            or not stacks_differ_after_step):
        raise RuntimeError("CPU one-step pairing/gradient check failed")
    return {
        "seed": seed, "device": "cpu", "initialization": audit,
        "one_step": {
            "tied_rng_start_sha256": tied_start_hash,
            "untied_rng_start_sha256": untied_start_hash,
            "tied_mean_member_ce": tied_loss,
            "untied_mean_member_ce": untied_loss,
            "max_train_member_logit_abs_diff": max_train_logit_diff,
            "max_shared_gradient_abs_diff": max_shared_grad_diff,
            "max_tied_vs_sum_untied_stack_gradient_abs_diff": max_stack_grad_diff,
            "max_shared_post_step_parameter_abs_diff": max_shared_post_step_diff,
            "untied_stacks_differ_after_step": stacks_differ_after_step,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()
    if args.torch_threads < 0:
        parser.error("torch-threads must be nonnegative")
    pilot.set_repo_temp()
    if args.cpu_smoke:
        report = cpu_smoke()
        path = pilot.repo_path("experiments_iclr/ogbn_arxiv_untied_cpu_preflight.json")
        pilot.write_json(path, report)
        print(json.dumps({
            "cpu_smoke": "passed", "report": str(path),
            "one_step": report["one_step"],
        }, sort_keys=True), flush=True)
        return
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The fixed production run requires an available CUDA device")
    lock, lock_hash = _check_source_lock()
    cfg = fixed_configuration()
    _check_reference_config(cfg)
    result_root = pilot.repo_path(RESULT_ROOT)
    if result_root.exists() or result_root.is_symlink():
        raise FileExistsError("A new empty result root is required; existing artifacts are immutable")
    result_root.mkdir(parents=True)
    pilot.write_json(result_root / "run_config.json", _run_config(cfg, lock_hash, lock))
    bundle, dataset_manifest = pilot.load_official_ogbn_arxiv(
        pilot.repo_path("experiments_iclr/data"), device
    )
    reference_manifest = json.loads(pilot.repo_path(REFERENCE_DATASET).read_text())
    if dataset_manifest != reference_manifest:
        raise RuntimeError("Official graph or split differs from the 300-epoch reference")
    pilot.write_json(result_root / "dataset_manifest.json", dataset_manifest)
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in SEEDS:
        pilot.seed_all(seed, device)
        tied, untied = build_pair(
            cfg, int(bundle.x.size(1)), bundle.num_classes, device
        )
        initial_rng = capture_rng(device)
        initialization = audit_initialization(
            seed, tied, untied, int(bundle.x.size(1)), device, initial_rng
        )
        seed_root = result_root / f"seed_{seed}"
        seed_root.mkdir()
        init_path = seed_root / "initialization.json"
        pilot.write_json(init_path, initialization)
        init_hash = pilot.sha256_file(init_path)
        for arm, model in (("tied", tied), ("untied_propagation", untied)):
            selected[(seed, arm)] = run_one(
                seed, arm, model, bundle, cfg, device, seed_root / arm,
                initial_rng, init_hash
            )
        del tied, untied
        torch.cuda.empty_cache()
    _write_summary(result_root, selected)
    _write_artifact_manifest(result_root)
    print(json.dumps({
        "status": "complete", "result_root": str(result_root),
        "seeds": list(SEEDS), "arms": list(ARMS),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
