"""Prospective matched Cora feature-preprocessing sensitivity grid.

Commands: freeze, preflight, run, score.  The independent verifier owns the
validation-only selection lock.  The training bundle excludes test labels
and indices, and `run` never computes a test score.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, to_undirected

from models import Model, TABMModel


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora_raw", "cora_normalized")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
PROJECTOR_ARMS = ARMS[2:]
SEEDS = (0, 1, 2)
LRS = (0.0003, 0.001, 0.003)
WEIGHT_DECAYS = (0.0, 0.01)
CANDIDATES = tuple((lr, wd) for lr in LRS for wd in WEIGHT_DECAYS)
DEFAULT = (0.001, 0.0)
EPOCHS = 1000
MEMBERS = 4
WIDTH = 128
DEPTH = 2
DROP = 0.2
INITIAL_LOGIT_TOL = 1e-5
TRAIN_ONE_REFERENCE_SHA = "4fbbfed4226f9b9e2edc10d7f91df2b4086f530f3e64ad2401d722f7c635c4f4"
SOURCE_FILES = ("tuning.py", "verify_tuning.py", "models.py", "STUDY_PROTOCOL.md",
                "PRETRAIN_REPAIR.md")
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(str(array.dtype).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def state_sha(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().contiguous().cpu().numpy()
        h.update(key.encode() + b"\0")
        h.update(str(value.shape).encode() + b"\0")
        h.update(str(value.dtype).encode() + b"\0")
        h.update(value.tobytes())
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
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


def candidate_name(lr: float, wd: float) -> str:
    if (lr, wd) not in CANDIDATES:
        raise ValueError((lr, wd))
    return f"lr{lr:g}_wd{wd:g}"


def cell_path(dataset: str, arm: str, lr: float, wd: float, seed: int) -> Path:
    return ROOT / "results" / dataset / arm / candidate_name(lr, wd) / f"seed{seed}"


class UntiedPropagation(nn.Module):
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
        x = self.act(self.dropout(self.input_be_block(x, tabm_seed=tabm_seed)))
        for block in self.propagation_stacks[tabm_seed]:
            x = block(graph, x)
        return self.output_be_block(self.output_normalization(x), tabm_seed=tabm_seed).squeeze(1)


class OnePrivateBlock(nn.Module):
    def __init__(self, tied: TABMModel, private_index: int):
        super().__init__()
        if len(tied.residual_modules) != 2 or private_index not in (0, 1):
            raise ValueError("This study has exactly two residual blocks")
        self.private_index = private_index
        self.input_be_block = copy.deepcopy(tied.input_be_block)
        self.dropout = copy.deepcopy(tied.dropout)
        self.act = copy.deepcopy(tied.act)
        self.shared_block = copy.deepcopy(tied.residual_modules[1 - private_index])
        self.private_blocks = nn.ModuleList(
            [copy.deepcopy(tied.residual_modules[private_index]) for _ in range(MEMBERS)]
        )
        self.output_normalization = copy.deepcopy(tied.output_normalization)
        self.output_be_block = copy.deepcopy(tied.output_be_block)

    def forward(self, graph, x, tabm_seed):
        x = self.act(self.dropout(self.input_be_block(x, tabm_seed=tabm_seed)))
        for index in range(2):
            block = self.private_blocks[tabm_seed] if index == self.private_index else self.shared_block
            x = block(graph, x)
        return self.output_be_block(self.output_normalization(x), tabm_seed=tabm_seed).squeeze(1)


def common_kwargs(bundle) -> dict:
    return dict(model_name="SAGE", num_layers=DEPTH, input_dim=bundle.x.size(1),
                hidden_dim=WIDTH, output_dim=bundle.classes, hidden_dim_multiplier=1,
                num_heads=8, normalization="LayerNorm", dropout=DROP)


def make_model(arm: str, bundle, device: torch.device):
    common = common_kwargs(bundle)
    if arm == "base":
        return Model(**common).to(device), None
    if arm == "ens":
        model = nn.ModuleList([Model(**common) for _ in range(MEMBERS)]).to(device)
        params = list(model.parameters())
        if len({p.untyped_storage().data_ptr() for p in params}) != len(params):
            raise RuntimeError("ENS paths share parameter storage")
        return model, None
    tied = TABMModel(**common, tabm_inits=MEMBERS, device=device).to(device)
    canonical_sha = state_sha(tied.state_dict())
    if arm == "tied":
        return tied, canonical_sha
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    if arm == "untied":
        model = UntiedPropagation(tied).to(device)
    elif arm in ("private_first", "private_last"):
        model = OnePrivateBlock(tied, 0 if arm == "private_first" else 1).to(device)
    else:
        raise ValueError(arm)
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    if len({p.untyped_storage().data_ptr() for p in model.parameters()}) != len(list(model.parameters())):
        raise RuntimeError("Distinct stored parameters share storage")
    return model, canonical_sha


def member_logits(model, arm: str, bundle):
    if arm == "base":
        return (model(bundle.graph, bundle.x),)
    if arm == "ens":
        return tuple(m(bundle.graph, bundle.x) for m in model)
    return tuple(model(bundle.graph, bundle.x, tabm_seed=m) for m in range(MEMBERS))


@torch.no_grad()
def evaluate(model, arm: str, bundle, idx, labels):
    model.eval()
    logits = torch.stack([v.index_select(0, idx) for v in member_logits(model, arm, bundle)])
    pooled = logits.mean(0)
    accuracy = float((pooled.argmax(-1) == labels).float().mean().item())
    ce = float(F.cross_entropy(pooled, labels).item())
    return accuracy, ce, pooled.detach().cpu(), logits.detach().cpu()


def load_planetoid(name: str):
    assert name in DATASETS
    data = Planetoid(root=str(ROOT / "data" / "cora"), name="Cora", split="public")[0]
    x, y = data.x.float().contiguous(), data.y.long().flatten().contiguous()
    assert bool((x >= 0).all())
    if name == "cora_normalized":
        row_sum = x.sum(dim=1, keepdim=True)
        x = (x / row_sum.clamp_min(1.0)).contiguous()
    raw = data.edge_index.long().contiguous()
    edges = to_undirected(coalesce(raw, num_nodes=len(y)), num_nodes=len(y))
    return x, y, raw, edges, data.train_mask.bool(), data.val_mask.bool(), data.test_mask.bool()



def load_graph(dataset: str, device: torch.device, include_test: bool = False):
    loaders = {"cora_raw": lambda: load_planetoid("cora_raw"),
               "cora_normalized": lambda: load_planetoid("cora_normalized")}
    if dataset not in loaders:
        raise ValueError(dataset)
    x, y, raw, edge, train, valid, test = loaders[dataset]()
    n = len(y)
    if x.size(0) != n or any(len(mask) != n for mask in (train, valid, test)):
        raise RuntimeError("Node/mask shape mismatch")
    if not (train.any() and valid.any() and test.any()):
        raise RuntimeError("Empty split")
    if (train & valid).any() or (train & test).any() or (valid & test).any():
        raise RuntimeError("Overlapping masks")
    if not torch.isfinite(x).all() or int(y.min()) != 0:
        raise RuntimeError("Invalid features or labels")
    classes = int(y.max()) + 1
    if not torch.equal(torch.unique(y), torch.arange(classes)):
        raise RuntimeError("Noncontiguous classes")
    if edge.min() < 0 or edge.max() >= n:
        raise RuntimeError("Edge endpoint out of range")
    idx = {k: m.nonzero().flatten().long() for k, m in
           (("train", train), ("valid", valid), ("test", test))}
    split = "Planetoid public"
    descriptor = {
        "dataset": dataset, "split": split, "nodes": n, "features": x.size(1),
        "classes": classes, "raw_edges": raw.size(1), "training_edges": edge.size(1),
        "explicit_self_loops_added": False,
        "feature_transform": ("nonnegative float32 x divided by max(row sum, 1.0); zero rows remain zero"
                              if dataset == "cora_normalized" else
                              "source float32 x unchanged"),
        "split_sizes": {k: v.numel() for k, v in idx.items()},
        "tensor_sha256": {"features": tensor_sha(x), "labels": tensor_sha(y),
                          "raw_edges": tensor_sha(raw), "training_edges": tensor_sha(edge),
                          **{f"{k}_indices": tensor_sha(v) for k, v in idx.items()}},
    }
    bundle = SimpleNamespace(graph=SimpleNamespace(edge_index=edge.to(device)), x=x.to(device),
                             train_idx=idx["train"].to(device), valid_idx=idx["valid"].to(device),
                             train_y=y[idx["train"]].to(device), valid_y=y[idx["valid"]].to(device),
                             classes=classes)
    if include_test:
        bundle.test_idx = idx["test"].to(device)
        bundle.test_y = y[idx["test"]].to(device)
    return bundle, descriptor


def raw_files(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    base = ROOT / "data" / "cora" / "Cora" / "raw"
    files = sorted(p.relative_to(ROOT).as_posix() for p in base.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"{dataset} raw files not downloaded")
    return tuple(files)


def expected_freeze() -> dict:
    train_one_sha = hashlib.sha256(inspect.getsource(train_one).encode()).hexdigest()
    if train_one_sha != TRAIN_ONE_REFERENCE_SHA:
        raise RuntimeError("Frozen train_one function differs from the original six-arm grid")
    source = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    graphs = {}
    for dataset in DATASETS:
        _, descriptor = load_graph(dataset, torch.device("cpu"))
        graphs[dataset] = {"descriptor": descriptor,
                           "raw_sha256": {name: sha256_file(ROOT / name) for name in raw_files(dataset)}}
    raw_graph, normalized_graph = (graphs[name] for name in DATASETS)
    if raw_graph["raw_sha256"] != normalized_graph["raw_sha256"]:
        raise RuntimeError("Conditions use different source data")
    for field in ("labels", "raw_edges", "training_edges", "train_indices",
                  "valid_indices", "test_indices"):
        if raw_graph["descriptor"]["tensor_sha256"][field] != normalized_graph["descriptor"]["tensor_sha256"][field]:
            raise RuntimeError(f"Conditions differ outside features: {field}")
    return {
        "protocol": "cora_feature_normalization_sensitivity_v1", "source_sha256": source,
        "reused_train_one_sha256": train_one_sha,
        "graphs": graphs, "matrix": {
            "datasets": DATASETS, "arms": ARMS, "seeds": SEEDS,
            "candidates": [{"lr": lr, "weight_decay": wd} for lr, wd in CANDIDATES],
            "default_candidate": {"lr": DEFAULT[0], "weight_decay": DEFAULT[1]},
            "epochs": EPOCHS, "width": WIDTH, "depth": DEPTH, "dropout": DROP,
            "members": MEMBERS, "initial_logit_tolerance": INITIAL_LOGIT_TOL,
            "optimizer": "AdamW", "objective": "mean member CE for four-member arms; ordinary CE for BASE",
            "checkpoint_rule": "maximum pooled validation accuracy; lowest pooled CE; earliest epoch",
            "candidate_rule": "maximum mean selected validation accuracy; lowest mean CE; lower LR; lower decay",
            "partial_family_rule": "tune both positions; higher selected mean validation accuracy; lower selected mean CE; private_last exact tie",
            "partial_family_search_cost": "12 configurations per condition versus 6 per individual comparator arm",
            "failure_rule": "nonfinite seed invalidates its entire arm/condition candidate",
            "test_rule": "after global validation-only lock: selected and (0.001,0) default only",
            "feature_transform": {"cora_raw": "source float32 x unchanged",
                                  "cora_normalized": "nonnegative float32 x divided by max(row sum, 1.0); zero rows remain zero"},
            "study_status": "post hoc sensitivity chosen after earlier Cora and other graph outcomes; both matched conditions prospectively frozen before new training",
            "complete_study_audit_cutoff_utc": "2026-09-26T07:50:00Z",
        },
    }


def check_freeze() -> str:
    path = ROOT / "FROZEN_STUDY.json"
    if not path.exists():
        raise RuntimeError("Missing prospective FROZEN_STUDY.json")
    stored = json.loads(path.read_text())
    # JSON encodes the fixed Python tuples in the matrix as arrays.
    expected = json.loads(json.dumps(expected_freeze(), sort_keys=True, allow_nan=False))
    if stored != expected:
        raise RuntimeError("Source, data, tensor, or matrix differs from study freeze")
    return sha256_file(path)


def initial_audit(model, arm: str, bundle, seed: int, device: torch.device, canonical_sha):
    result = {"parameter_count": sum(p.numel() for p in model.parameters()),
              "parameter_storage_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
              "canonical_projector_state_sha256": canonical_sha}
    if arm not in PROJECTOR_ARMS:
        result["paired_initial_logits_max_abs_diff"] = None
        return result
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    model.eval()
    with torch.no_grad():
        actual = torch.stack(member_logits(model, arm, bundle))
    if not torch.equal(cpu_rng, torch.get_rng_state()) or (cuda_rng is not None and
            not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))):
        raise RuntimeError("Evaluation altered RNG state")
    if random.getstate() != python_rng or not np.array_equal(np.random.get_state()[1], numpy_rng[1]):
        raise RuntimeError("Evaluation altered Python/NumPy RNG state")
    # Reconstruct the tied reference from the same seed; then restore this
    # arm's post-construction RNG so training starts identically across arms.
    seed_all(seed)
    reference, reference_sha = make_model("tied", bundle, device)
    reference.eval()
    with torch.no_grad():
        expected = torch.stack(member_logits(reference, "tied", bundle))
    if reference_sha != canonical_sha:
        raise RuntimeError("Canonical projector initialization differs")
    diff = float((actual - expected).abs().max().item())
    if diff > INITIAL_LOGIT_TOL:
        raise RuntimeError(f"Initial projector logits differ by {diff:g}")
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    result.update({"paired_initial_logits_max_abs_diff": diff,
                   "python_rng_sha256": hashlib.sha256(repr(python_rng).encode()).hexdigest(),
                   "numpy_rng_sha256": hashlib.sha256(
                       repr((numpy_rng[0], numpy_rng[2:])).encode() + numpy_rng[1].tobytes()).hexdigest(),
                   "cpu_rng_sha256": tensor_sha(cpu_rng),
                   "cuda_rng_sha256": tensor_sha(cuda_rng) if cuda_rng is not None else None,
                   "initial_member_logits_sha256": tensor_sha(actual)})
    return result


def train_one(dataset: str, bundle, freeze_sha: str, arm: str, lr: float,
              wd: float, seed: int, device: torch.device):
    final = cell_path(dataset, arm, lr, wd, seed)
    if final.exists():
        result_path, trace_path = final / "result.json", final / "validation_trace.csv"
        if not result_path.is_file() or not trace_path.is_file():
            raise RuntimeError(f"Incomplete existing cell: {final}")
        old = json.loads(result_path.read_text())
        if (old.get("freeze_sha256") != freeze_sha or old.get("dataset") != dataset or
                old.get("arm") != arm or old.get("lr") != lr or
                old.get("weight_decay") != wd or old.get("seed") != seed or
                old.get("epochs_required") != EPOCHS or
                old.get("validation_trace_sha256") != sha256_file(trace_path)):
            raise RuntimeError(f"Existing cell fails frozen identity/hash gate: {final}")
        if old.get("failure") is None:
            checkpoint = final / "checkpoint.pt"
            if (old.get("epochs_completed") != EPOCHS or not checkpoint.is_file() or
                    old.get("checkpoint_sha256") != sha256_file(checkpoint)):
                raise RuntimeError(f"Existing cell fails complete-checkpoint gate: {final}")
        return "skipped"
    work = final.with_name(final.name + ".inprogress")
    if work.exists():
        raise RuntimeError(f"Interrupted cell needs inspection: {work}")
    work.mkdir(parents=True)
    seed_all(seed)
    model, canonical_sha = make_model(arm, bundle, device)
    initial = initial_audit(model, arm, bundle, seed, device, canonical_sha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    best_state = None
    trace = []
    failure = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for logits in member_logits(model, arm, bundle):
            loss = F.cross_entropy(logits.index_select(0, bundle.train_idx), bundle.train_y)
            if not torch.isfinite(loss):
                failure = f"nonfinite training CE at epoch {epoch}"
                break
            (loss / (1 if arm == "base" else MEMBERS)).backward()
        if failure:
            break
        optimizer.step()
        val_acc, val_ce, _, _ = evaluate(model, arm, bundle, bundle.valid_idx, bundle.valid_y)
        if not (np.isfinite(val_acc) and np.isfinite(val_ce)):
            failure = f"nonfinite validation at epoch {epoch}"
            break
        improved = val_acc > best_acc or (val_acc == best_acc and val_ce < best_ce)
        if improved:
            best_acc, best_ce, best_epoch = val_acc, val_ce, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        trace.append((epoch, val_acc, val_ce, int(improved)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    with (work / "validation_trace.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRACE_FIELDS)
        writer.writerows(trace)
    result = {
        "protocol": "validation_tuning_sensitivity_v1", "freeze_sha256": freeze_sha,
        "dataset": dataset, "arm": arm, "lr": lr, "weight_decay": wd, "seed": seed,
        "epochs_required": EPOCHS, "epochs_completed": len(trace),
        "training_seconds": elapsed, "initialization": initial,
        "validation_trace_sha256": sha256_file(work / "validation_trace.csv"),
        "failure": failure,
    }
    if failure is None:
        if len(trace) != EPOCHS or best_state is None:
            raise RuntimeError("Incomplete finite training trace")
        checkpoint = {"state_dict": best_state, "epoch": best_epoch,
                      "dataset": dataset, "arm": arm, "lr": lr,
                      "weight_decay": wd, "seed": seed, "freeze_sha256": freeze_sha}
        torch.save(checkpoint, work / "checkpoint.pt")
        model.load_state_dict(best_state, strict=True)
        val_acc, val_ce, pooled, _ = evaluate(model, arm, bundle, bundle.valid_idx, bundle.valid_y)
        if abs(val_acc - best_acc) > 1e-7 or abs(val_ce - best_ce) > 1e-6:
            raise RuntimeError("Selected checkpoint validation replay failed")
        result.update({"selected_epoch": best_epoch, "selected_valid_accuracy": best_acc,
                       "selected_valid_ce": best_ce,
                       "selected_valid_pooled_logits_sha256": tensor_sha(pooled),
                       "selected_state_sha256": state_sha(best_state),
                       "checkpoint_sha256": sha256_file(work / "checkpoint.pt")})
    write_json(work / "result.json", result)
    work.rename(final)
    print(json.dumps({"completed": str(final.relative_to(ROOT)), "seconds": elapsed,
                      "failure": failure}), flush=True)
    return failure or "complete"


def preflight(device: torch.device):
    freeze_sha = check_freeze()
    checks = {"freeze_sha256": freeze_sha, "device": str(device), "graphs": {}}
    for dataset in DATASETS:
        bundle, _ = load_graph(dataset, torch.device("cpu"))
        counts = {}
        for arm in ARMS:
            seed_all(0)
            model, canonical = make_model(arm, bundle, torch.device("cpu"))
            row = initial_audit(model, arm, bundle, 0, torch.device("cpu"), canonical)
            counts[arm] = row["parameter_count"]
        if counts["private_first"] != counts["private_last"]:
            raise RuntimeError("Partial sharing positions have unequal parameter counts")
        if counts["ens"] != MEMBERS * counts["base"]:
            raise RuntimeError("ENS parameter count is not four ordinary paths")
        if not counts["base"] < counts["tied"] < counts["private_first"] < counts["untied"]:
            raise RuntimeError("Unexpected stored parameter ordering")
        checks["graphs"][dataset] = {"parameter_counts": counts}
    # A real first update of all arms on the selected CUDA device.
    bundle, _ = load_graph("cora_raw", device)
    gpu = {}
    for arm in ARMS:
        seed_all(0)
        model, canonical = make_model(arm, bundle, device)
        initial = initial_audit(model, arm, bundle, 0, device, canonical)
        optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT[0], weight_decay=DEFAULT[1])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for logits in member_logits(model, arm, bundle):
            loss = F.cross_entropy(logits.index_select(0, bundle.train_idx), bundle.train_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite first-step loss: {arm}")
            (loss / (1 if arm == "base" else MEMBERS)).backward()
            losses.append(float(loss.detach().item()))
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f"Nonfinite parameters after first update: {arm}")
        gpu[arm] = {"mean_member_ce": float(np.mean(losses)), "initialization": initial}
    checks["first_step"] = gpu
    write_json(ROOT / "preflight.json", checks)
    print(json.dumps({"preflight": "passed", "device": str(device),
                      "freeze_sha256": freeze_sha}), flush=True)


def run_dataset(dataset: str, device: torch.device):
    freeze_sha = check_freeze()
    bundle, _ = load_graph(dataset, device, include_test=False)
    for arm in ARMS:
        for lr, wd in CANDIDATES:
            for seed in SEEDS:
                train_one(dataset, bundle, freeze_sha, arm, lr, wd, seed, device)


def score(device: torch.device, dataset_subset: str | None = None):
    freeze_sha = check_freeze()
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("No independent validation-only selection lock")
    lock = json.loads(lock_path.read_text())
    if lock["freeze_sha256"] != freeze_sha or lock["protocol"] != "cora_feature_normalization_sensitivity_v1":
        raise RuntimeError("Selection lock does not match frozen study")
    # Fresh verifier is run before any test access. Its audit digest is part of
    # the lock and is checked again by the independent final score audit.
    if len(lock["cells"]) != len(DATASETS) * len(ARMS) * len(CANDIDATES) * len(SEEDS):
        raise RuntimeError("Selection lock lacks complete 216-cell audit")
    for dataset in ((dataset_subset,) if dataset_subset else DATASETS):
        bundle, _ = load_graph(dataset, device, include_test=True)
        for arm in ARMS:
            group = lock["selections"][dataset][arm]
            configurations = [tuple(group["selected_candidate"]), DEFAULT]
            seen = set()
            for lr, wd in configurations:
                if (lr, wd) in seen:
                    continue
                seen.add((lr, wd))
                for seed in SEEDS:
                    cell = cell_path(dataset, arm, lr, wd, seed)
                    key = f"{dataset}/{arm}/{candidate_name(lr, wd)}/seed{seed}"
                    frozen_cell = lock["cells"][key]
                    if sha256_file(cell / "result.json") != frozen_cell["result_sha256"] or \
                            sha256_file(cell / "checkpoint.pt") != frozen_cell["checkpoint_sha256"]:
                        raise RuntimeError(f"Cell changed after validation lock: {key}")
                    result = json.loads((cell / "result.json").read_text())
                    seed_all(seed)
                    model, _ = make_model(arm, bundle, device)
                    payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
                    model.load_state_dict(payload["state_dict"], strict=True)
                    valid_acc, valid_ce, valid_logits, _ = evaluate(
                        model, arm, bundle, bundle.valid_idx, bundle.valid_y)
                    if (abs(valid_acc - result["selected_valid_accuracy"]) > 1e-7 or
                            abs(valid_ce - result["selected_valid_ce"]) > 1e-6):
                        raise RuntimeError(f"Selected validation replay failed: {key}")
                    test_acc, test_ce, test_logits, member_test = evaluate(
                        model, arm, bundle, bundle.test_idx, bundle.test_y)
                    out = ROOT / "scores" / dataset / arm / candidate_name(lr, wd) / f"seed{seed}"
                    out.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out / "predictions.npz",
                                        valid_pooled_logits=valid_logits.numpy(),
                                        test_pooled_logits=test_logits.numpy(),
                                        test_member_logits=member_test.numpy())
                    write_json(out / "score.json", {
                        "freeze_sha256": freeze_sha,
                        "validation_selection_lock_sha256": sha256_file(lock_path),
                        "dataset": dataset, "arm": arm, "lr": lr, "weight_decay": wd,
                        "seed": seed, "selected_candidate": (lr, wd) == tuple(group["selected_candidate"]),
                        "predeclared_default": (lr, wd) == DEFAULT,
                        "checkpoint_sha256": frozen_cell["checkpoint_sha256"],
                        "valid_accuracy": valid_acc, "valid_ce": valid_ce,
                        "test_accuracy": test_acc, "test_ce": test_ce,
                        "test_predictions_sha256": sha256_file(out / "predictions.npz"),
                    })
                    print(json.dumps({"scored": key, "test_accuracy": test_acc}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("check-freeze")
    sub.add_parser("preflight").add_argument("--device", default="cuda")
    run = sub.add_parser("run")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--device", default="cuda")
    scoring = sub.add_parser("score")
    scoring.add_argument("--device", default="cuda")
    scoring.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    if args.command == "freeze":
        if (ROOT / "FROZEN_STUDY.json").exists():
            raise RuntimeError("Prospective freeze already exists; refusing overwrite")
        write_json(ROOT / "FROZEN_STUDY.json", expected_freeze())
        print(json.dumps({"freeze_sha256": sha256_file(ROOT / "FROZEN_STUDY.json")}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"freeze_sha256": check_freeze()}), flush=True)
    elif args.command == "preflight":
        preflight(torch.device(args.device))
    elif args.command == "run":
        run_dataset(args.dataset, torch.device(args.device))
    elif args.command == "score":
        score(torch.device(args.device), args.dataset)


if __name__ == "__main__":
    main()
