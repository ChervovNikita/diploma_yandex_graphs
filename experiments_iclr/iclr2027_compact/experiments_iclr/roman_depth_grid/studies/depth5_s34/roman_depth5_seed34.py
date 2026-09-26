"""Post hoc Roman Empire depth-5 seed-3/4 extension under the OGB-300 schedule.

Runs official mask 0 with optimizer seeds 3/4 and the same architecture,
optimization, and validation selection as the WikiCS/Actor external study.
This seed extension was frozen for depths 2, 3, 4, and 5 before
any depth-3/4 test scores were inspected. It uses 5 residual SAGE blocks. Test labels
are scored only after a selected checkpoint is restored.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import add_remaining_self_loops, coalesce, to_undirected

from models import SAGEModule, TABMModel

ROOT = Path(__file__).resolve().parent
SEEDS = (3, 4)
ARMS = ("tied", "untied_propagation")
MEMBERS = 4
EPOCHS = 300
LR = 0.001
WIDTH = 128
DEPTH = 5
DROP = 0.2
INITIAL_LOGIT_TOL = 1e-5
DATA_FILES = {"roman": ("data/roman_empire.npz",)}
SOURCE_FILES = ("roman_depth5_seed34.py", "roman_depth_seed34_protocol.md", "models.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_state(state) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().contiguous().cpu().numpy()
        h.update(key.encode() + b"\0")
        h.update(str(tensor.dtype).encode() + b"\0")
        h.update(str(tensor.shape).encode() + b"\0")
        h.update(tensor.tobytes())
    return h.hexdigest()


def hash_tensor(tensor: torch.Tensor) -> str:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UntiedPropagation(nn.Module):
    """Exact deep copies of the initially shared residual stack per member."""

    def __init__(self, tied: TABMModel):
        super().__init__()
        self.input_be_block = copy.deepcopy(tied.input_be_block)
        self.dropout = copy.deepcopy(tied.dropout)
        self.act = copy.deepcopy(tied.act)
        self.propagation_stacks = nn.ModuleList(
            [copy.deepcopy(tied.residual_modules) for _ in range(MEMBERS)]
        )
        self.output_normalization = copy.deepcopy(tied.output_normalization)
        self.output_be_block = copy.deepcopy(tied.output_be_block)

    def forward(self, graph, x, tabm_seed):
        x = self.input_be_block(x, tabm_seed=tabm_seed)
        x = self.dropout(x)
        x = self.act(x)
        for block in self.propagation_stacks[tabm_seed]:
            x = block(graph, x)
        x = self.output_normalization(x)
        return self.output_be_block(x, tabm_seed=tabm_seed).squeeze(1)


class MemberFactorLinear(nn.Module):
    """Identity-initialized member factors around one existing linear map."""

    def __init__(self, shared: nn.Module):
        super().__init__()
        weight = getattr(shared, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError("Expected a two-dimensional linear map")
        outputs, inputs = weight.shape
        self.shared = shared
        self.R = nn.Parameter(weight.new_ones((MEMBERS, inputs)))
        self.S = nn.Parameter(weight.new_ones((MEMBERS, outputs)))
        self.B = nn.Parameter(weight.new_zeros((MEMBERS, outputs)))
        self.active_member = None

    def forward(self, x):
        m = self.active_member
        if m is None or not 0 <= m < MEMBERS:
            raise RuntimeError("Select a member before forwarding")
        return self.shared(x * self.R[m]) * self.S[m] + self.B[m]


class AllLayerBE(nn.Module):
    """Boundary projectors plus in-layer SAGE and feed-forward factors."""

    def __init__(self, tied: TABMModel):
        super().__init__()
        self.base = tied
        self.factor_layers = []
        if len(tied.residual_modules) != DEPTH:
            raise RuntimeError("Unexpected SAGE residual stack")
        for residual in tied.residual_modules:
            sage = residual.module
            if not isinstance(sage, SAGEModule):
                raise RuntimeError("Unexpected residual module")
            for owner, name in (
                (sage.conv, "lin_l"), (sage.conv, "lin_r"),
                (sage.feed_forward_module, "linear_1"),
                (sage.feed_forward_module, "linear_2"),
            ):
                wrapped = MemberFactorLinear(getattr(owner, name))
                setattr(owner, name, wrapped)
                self.factor_layers.append(wrapped)
        if len(self.factor_layers) != 4 * DEPTH:
            raise RuntimeError("Missing hidden linear factor")

    def forward(self, graph, x, tabm_seed):
        for layer in self.factor_layers:
            layer.active_member = tabm_seed
        return self.base(graph, x, tabm_seed=tabm_seed)


def make_model(arm: str, input_dim: int, classes: int, device: torch.device):
    tied = TABMModel(
        model_name="SAGE", num_layers=DEPTH, input_dim=input_dim,
        hidden_dim=WIDTH, output_dim=classes, hidden_dim_multiplier=1,
        num_heads=8, normalization="LayerNorm", dropout=DROP,
        tabm_inits=MEMBERS, device=device,
    ).to(device)
    canonical = hash_state(tied.state_dict())
    # Constructing a copied/wrapped arm may advance a library RNG even though
    # its tensors initially equal the tied arm. Restore every RNG to the state
    # immediately after constructing the common tied model. This is the same
    # post-construction training start for all arms.
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (torch.cuda.get_rng_state(device).clone()
                if device.type == "cuda" else None)

    def restore_common_rng():
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)

    if arm == "tied":
        return tied, canonical
    if arm == "untied_propagation":
        model = UntiedPropagation(tied).to(device)
        ptrs = [p.untyped_storage().data_ptr()
                for stack in model.propagation_stacks for p in stack.parameters()]
        if len(ptrs) != len(set(ptrs)):
            raise RuntimeError("Untied stacks share parameter storage")
        restore_common_rng()
        return model, canonical
    if arm == "all_layer_be":
        base_params = sum(p.numel() for p in tied.parameters())
        model = AllLayerBE(tied).to(device)
        expected_extra = sum(p.numel() for layer in model.factor_layers
                             for p in (layer.R, layer.S, layer.B))
        if sum(p.numel() for p in model.parameters()) != base_params + expected_extra:
            raise RuntimeError("All-layer parameter accounting changed")
        if expected_extra <= 0:
            raise RuntimeError("No hidden factors were added")
        restore_common_rng()
        return model, canonical
    raise ValueError(arm)


def load_wikics():
    path = ROOT / DATA_FILES["wikics"][0]
    raw = json.loads(path.read_text())
    x = torch.tensor(raw["features"], dtype=torch.float32)
    y = torch.tensor(raw["labels"], dtype=torch.long)
    edges = [(i, j) for i, js in enumerate(raw["links"]) for j in js]
    raw_edge = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge = to_undirected(raw_edge, num_nodes=len(y))
    train = torch.tensor(raw["train_masks"][0], dtype=torch.bool)
    valid = torch.tensor(raw["val_masks"][0], dtype=torch.bool)
    test = torch.tensor(raw["test_mask"], dtype=torch.bool)
    if len(raw["train_masks"]) != 20 or len(raw["val_masks"]) != 20:
        raise RuntimeError("WikiCS official split count changed")
    return x, y, raw_edge, edge, train, valid, test


def load_actor():
    base = ROOT / "data/actor"
    node_lines = (base / "out1_node_feature_label.txt").read_text().splitlines()[1:]
    node_lines = [line for line in node_lines if line.strip()]
    entries = [line.split("\t") for line in node_lines]
    if any(len(row) != 3 for row in entries):
        raise RuntimeError("Actor node row schema changed")
    num_nodes = len(entries)
    width = max(int(j) for _, cols, _ in entries for j in cols.split(",")) + 1
    x = torch.zeros((num_nodes, width), dtype=torch.float32)
    y = torch.empty(num_nodes, dtype=torch.long)
    for node, cols, label in entries:
        idx = int(node)
        x[idx, [int(j) for j in cols.split(",")]] = 1.0
        y[idx] = int(label)
    edge_lines = (base / "out1_graph_edges.txt").read_text().splitlines()[1:]
    pairs = [[int(v) for v in line.split("\t")] for line in edge_lines if line.strip()]
    raw_edge = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge = to_undirected(coalesce(raw_edge, num_nodes=num_nodes), num_nodes=num_nodes)
    with np.load(base / "film_split_0.6_0.2_0.npz", allow_pickle=False) as raw:
        train = torch.from_numpy(raw["train_mask"].copy()).bool()
        valid = torch.from_numpy(raw["val_mask"].copy()).bool()
        test = torch.from_numpy(raw["test_mask"].copy()).bool()
    return x, y, raw_edge, edge, train, valid, test


def load_roman():
    path = ROOT / DATA_FILES["roman"][0]
    with np.load(path, allow_pickle=False) as data:
        expected = {"node_features", "node_labels", "edges",
                    "train_masks", "val_masks", "test_masks"}
        if set(data.files) != expected:
            raise RuntimeError("Roman NPZ fields changed")
        if (data["node_features"].shape != (22662, 300) or
                data["node_labels"].shape != (22662,) or
                data["edges"].shape != (32927, 2) or
                any(data[key].shape != (10, 22662) for key in
                    ("train_masks", "val_masks", "test_masks"))):
            raise RuntimeError("Roman NPZ shape changed")
        x = torch.from_numpy(data["node_features"].copy()).float()
        y = torch.from_numpy(data["node_labels"].copy()).long()
        raw_edge = torch.from_numpy(data["edges"].T.copy()).long()
        train = torch.from_numpy(data["train_masks"][0].copy()).bool()
        valid = torch.from_numpy(data["val_masks"][0].copy()).bool()
        test = torch.from_numpy(data["test_masks"][0].copy()).bool()
    edge = to_undirected(coalesce(raw_edge, num_nodes=len(y)), num_nodes=len(y))
    edge, _ = add_remaining_self_loops(edge, num_nodes=len(y))
    if int((edge[0] == edge[1]).sum()) != len(y):
        raise RuntimeError("Expected exactly one explicit self-loop per node")
    return x, y, raw_edge, edge, train, valid, test


def load_graph(dataset: str, device: torch.device):
    if dataset != "roman":
        raise ValueError(dataset)
    x, y, raw_edge, edge, train, valid, test = load_roman()
    n = len(y)
    if x.size(0) != n or any(len(mask) != n for mask in (train, valid, test)):
        raise RuntimeError("Node/mask length mismatch")
    if bool(train.any() and valid.any() and test.any()) is False:
        raise RuntimeError("Empty published partition")
    if torch.any(train & valid) or torch.any(train & test) or torch.any(valid & test):
        raise RuntimeError("Published masks overlap")
    if not torch.isfinite(x).all() or int(y.min()) != 0:
        raise RuntimeError("Invalid features or labels")
    classes = int(y.max()) + 1
    if not torch.equal(torch.unique(y), torch.arange(classes)):
        raise RuntimeError("Non-contiguous class IDs")
    if edge.min() < 0 or edge.max() >= n:
        raise RuntimeError("Edge endpoint outside graph")
    indices = {k: mask.nonzero().flatten().long()
               for k, mask in (("train", train), ("valid", valid), ("test", test))}
    descriptor = {
        "dataset": dataset,
        "split": 0,
        "num_nodes": n,
        "num_features": int(x.size(1)),
        "num_classes": classes,
        "raw_edges": int(raw_edge.size(1)),
        "undirected_edges": int(edge.size(1)),
        "explicit_self_loops_added": True,
        "split_sizes": {k: int(v.numel()) for k, v in indices.items()},
        "fingerprints_sha256": {
            "features": hash_tensor(x), "labels": hash_tensor(y),
            "raw_edges": hash_tensor(raw_edge), "training_edges": hash_tensor(edge),
            **{f"{k}_indices": hash_tensor(v) for k, v in indices.items()},
        },
    }
    graph = SimpleNamespace(edge_index=edge.to(device))
    bundle = SimpleNamespace(
        graph=graph, x=x.to(device),
        train_idx=indices["train"].to(device),
        valid_idx=indices["valid"].to(device),
        test_idx_cpu=indices["test"],
        train_y=y[indices["train"]].to(device),
        valid_y=y[indices["valid"]].to(device),
        test_y_cpu=y[indices["test"]],
        classes=classes,
    )
    return bundle, descriptor


def expected_manifest(dataset: str, descriptor: dict) -> dict:
    paths = SOURCE_FILES + DATA_FILES[dataset]
    return {
        "protocol": "roman_posthoc_ogb300_selfloop_depth5_seed34_v1",
        "dataset": dataset,
        "split": 0,
        "optimization_seeds": list(SEEDS),
        "arms": list(ARMS),
        "architecture": "original models.TABMModel SAGE with OGB-300 configuration",
        "source_sha256": {p: sha256_file(ROOT / p) for p in paths},
        "data_descriptor": descriptor,
        "configuration": {
            "layers": DEPTH, "width": WIDTH, "dropout": DROP,
            "optimizer": "AdamW", "learning_rate": LR,
            "weight_decay": 0, "epochs": EPOCHS,
            "members": MEMBERS,
            "objective": "mean of member cross-entropies",
            "validation": "every epoch pooled logit accuracy; CE then earliest epoch ties",
            "test_scoring": "after restoring selected checkpoint only",
        },
        "versions": {
            "torch": torch.__version__,
            "torch_geometric": __import__("torch_geometric").__version__,
        },
    }


def verify_manifest(dataset: str, descriptor: dict) -> str:
    expected = expected_manifest(dataset, descriptor)
    path = ROOT / "results" / dataset / "source_manifest.json"
    if path.exists():
        actual = json.loads(path.read_text())
        if actual != expected:
            raise RuntimeError("Source/data/protocol manifest changed")
    else:
        write_json(path, expected)
    return sha256_file(path)


def member_logits(model, bundle):
    return torch.stack([model(bundle.graph, bundle.x, tabm_seed=m)
                        for m in range(MEMBERS)], dim=0)


@torch.no_grad()
def evaluate(model, bundle, idx, labels, save=False):
    model.eval()
    logits = member_logits(model, bundle)[:, idx]
    pooled = logits.mean(0)
    acc = float((pooled.argmax(-1) == labels).float().mean().item())
    ce = float(F.cross_entropy(pooled, labels).item())
    if save:
        return acc, ce, logits.detach().cpu().numpy()
    return acc, ce, None


def initial_audit(model, canonical_sha, bundle, seed: int, arm: str,
                  work: Path, device: torch.device):
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone()
    model.eval()
    with torch.no_grad():
        logits = member_logits(model, bundle).detach().cpu().numpy()
    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise RuntimeError("Initial forward consumed CPU RNG")
    if not torch.equal(gpu_rng, torch.cuda.get_rng_state(device)):
        raise RuntimeError("Initial forward consumed CUDA RNG")
    np.save(work / "initial_logits.npy", logits)
    audit = {
        "canonical_tied_state_sha256": canonical_sha,
        "cpu_rng_sha256": hash_tensor(cpu_rng),
        "cuda_rng_sha256": hash_tensor(gpu_rng),
        "initial_logits_sha256": sha256_file(work / "initial_logits.npy"),
        "initial_logits_shape": list(logits.shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if arm != "tied":
        paired = ROOT / "results" / dataset_arg / f"seed{seed}" / "tied"
        previous = json.loads((paired / "initialization.json").read_text())
        for key in ("canonical_tied_state_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
            if audit[key] != previous[key]:
                raise RuntimeError(f"Initial {key} differs from tied arm")
        reference = np.load(paired / "initial_logits.npy", allow_pickle=False)
        max_diff = float(np.max(np.abs(logits - reference)))
        if max_diff > INITIAL_LOGIT_TOL:
            raise RuntimeError(f"Initial member logits differ: {max_diff}")
        audit["paired_initial_logits_max_abs_diff"] = max_diff
    else:
        audit["paired_initial_logits_max_abs_diff"] = 0.0
    write_json(work / "initialization.json", audit)
    return audit


def train_one(dataset: str, bundle, source_sha: str, seed: int,
              arm: str, device: torch.device) -> None:
    final = ROOT / "results" / dataset / f"seed{seed}" / arm
    if final.exists():
        raise RuntimeError(f"Run directory already exists: {final}")
    work = final.with_name(arm + ".inprogress")
    if work.exists():
        raise RuntimeError(f"Interrupted run requires review: {work}")
    work.mkdir(parents=True)
    seed_all(seed)
    model, canonical_sha = make_model(arm, bundle.x.size(1), bundle.classes, device)
    initial = initial_audit(model, canonical_sha, bundle, seed, arm, work, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
    best_acc, best_ce, best_epoch = -1.0, float("inf"), 0
    trace = []
    start = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for m in range(MEMBERS):
            logits = model(bundle.graph, bundle.x, tabm_seed=m)
            ce = F.cross_entropy(logits[bundle.train_idx], bundle.train_y)
            (ce / MEMBERS).backward()
        optimizer.step()
        val_acc, val_ce, _ = evaluate(
            model, bundle, bundle.valid_idx, bundle.valid_y)
        improved = (val_acc > best_acc + 1e-12 or
                    (abs(val_acc - best_acc) <= 1e-12 and val_ce < best_ce - 1e-12))
        if improved:
            best_acc, best_ce, best_epoch = val_acc, val_ce, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "seed": seed, "arm": arm}, work / "checkpoint.pt")
        trace.append((epoch, val_acc, val_ce, int(improved)))
        if epoch == 1 or epoch % 25 == 0:
            print(json.dumps({"dataset": dataset, "seed": seed, "arm": arm,
                              "epoch": epoch, "valid_accuracy": val_acc,
                              "selected_epoch_so_far": best_epoch}), flush=True)
    with (work / "validation_trace.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "valid_accuracy", "valid_ce", "improved"))
        writer.writerows(trace)
    selected = torch.load(work / "checkpoint.pt", map_location=device, weights_only=True)
    if selected["epoch"] != best_epoch or selected["seed"] != seed or selected["arm"] != arm:
        raise RuntimeError("Selected checkpoint metadata mismatch")
    model.load_state_dict(selected["state_dict"], strict=True)
    valid_acc, valid_ce, valid_logits = evaluate(
        model, bundle, bundle.valid_idx, bundle.valid_y, save=True)
    if abs(valid_acc - best_acc) > 1e-7 or abs(valid_ce - best_ce) > 1e-6:
        raise RuntimeError("Selected validation checkpoint did not reproduce")
    test_idx = bundle.test_idx_cpu.to(device)
    test_y = bundle.test_y_cpu.to(device)
    test_acc, test_ce, test_logits = evaluate(model, bundle, test_idx, test_y, save=True)
    np.savez_compressed(
        work / "selected_predictions.npz",
        valid_member_logits=valid_logits,
        valid_indices=bundle.valid_idx.cpu().numpy(),
        valid_labels=bundle.valid_y.cpu().numpy(),
        test_member_logits=test_logits,
        test_indices=bundle.test_idx_cpu.numpy(),
        test_labels=bundle.test_y_cpu.numpy(),
    )
    files = ("checkpoint.pt", "validation_trace.csv", "initialization.json",
             "initial_logits.npy", "selected_predictions.npz")
    row = {
        "protocol": "roman_posthoc_ogb300_selfloop_depth5_seed34_v1",
        "dataset": dataset, "published_split": 0,
        "optimization_seed": seed, "arm": arm,
        "source_manifest_sha256": source_sha,
        "epochs_run": EPOCHS, "selected_epoch": best_epoch,
        "valid_accuracy": valid_acc, "valid_ce": valid_ce,
        "test_accuracy": test_acc, "test_ce": test_ce,
        "train_seconds": time.monotonic() - start,
        "parameter_count": initial["parameter_count"],
        "initial_logits_max_abs_diff_from_tied":
            initial["paired_initial_logits_max_abs_diff"],
        "artifact_sha256": {name: sha256_file(work / name) for name in files},
    }
    write_json(work / "result.json", row)
    work.rename(final)
    print("ARM_COMPLETE", dataset, seed, arm, flush=True)
    del model, optimizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATA_FILES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Invalid optimization seeds")
    if len(set(args.arms)) != len(args.arms):
        parser.error("Duplicate arm")
    global dataset_arg
    dataset_arg = args.dataset
    torch.set_num_threads(2)
    device = torch.device("cpu" if args.preflight else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    bundle, descriptor = load_graph(args.dataset, device)
    source_sha = verify_manifest(args.dataset, descriptor)
    print("PREFLIGHT", json.dumps({"dataset": args.dataset,
                                  "source_manifest_sha256": source_sha,
                                  "descriptor": descriptor}, sort_keys=True), flush=True)
    if args.preflight:
        return
    if args.smoke:
        reference_logits = None
        reference_cpu_rng = None
        reference_cuda_rng = None
        reference_state = None
        for arm in ARMS:
            seed_all(0)
            model, canonical_state = make_model(
                arm, bundle.x.size(1), bundle.classes, device)
            cpu_rng = hash_tensor(torch.get_rng_state())
            cuda_rng = hash_tensor(torch.cuda.get_rng_state(device))
            model.eval()
            with torch.no_grad():
                out = member_logits(model, bundle)
            if out.shape != (MEMBERS, descriptor["num_nodes"], bundle.classes):
                raise RuntimeError("Smoke output shape mismatch")
            if reference_logits is None:
                reference_logits = out
                reference_cpu_rng = cpu_rng
                reference_cuda_rng = cuda_rng
                reference_state = canonical_state
            else:
                delta = float((out - reference_logits).abs().max().item())
                if delta > INITIAL_LOGIT_TOL:
                    raise RuntimeError(f"Smoke initial logits differ: {arm} {delta}")
                if cpu_rng != reference_cpu_rng or cuda_rng != reference_cuda_rng:
                    raise RuntimeError(f"Smoke post-construction RNG differs: {arm}")
                if canonical_state != reference_state:
                    raise RuntimeError(f"Smoke canonical initial state differs: {arm}")
            print("SMOKE", args.dataset, arm, tuple(out.shape), flush=True)
        return
    if device.type != "cuda":
        raise RuntimeError("Production study requires CUDA")
    for seed in args.seeds:
        for arm in args.arms:
            train_one(args.dataset, bundle, source_sha, seed, arm, device)


if __name__ == "__main__":
    main()
