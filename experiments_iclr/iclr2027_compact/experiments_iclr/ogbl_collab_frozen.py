"""Frozen OGBL-Collab link-prediction extension; run only after preflight approval.

Install/copy this file to experiments_iclr/ogbl_collab_frozen.py in the repository.
All data, cache, temporary files and outputs are required to remain in that repo.
Protocol: TASK_BREADTH_SCOUT.md, frozen before any task score was seen.
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
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import Model, TABMModel  # noqa: E402

PROTOCOL = "ogbl-collab-link-v2-frozen-20260925"
MEMBERS = 4
SEEDS = (0, 1, 2)
VARIANTS = ("tied", "untied", "ens", "base")
STEPS = 400
EVAL_EVERY = 20
PAIRS_PER_STEP = 65_536
WIDTH = 128
DEPTH = 2
DROPOUT = 0.2
LR = 0.001
WEIGHT_DECAY = 0.0
HISTORY_COLUMNS = (
    "step", "train_mean_member_bce", "valid_pooled_hits50", "valid_pooled_bce",
    "train_seconds", "validation_seconds", "elapsed_seconds", "selected_so_far",
)


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path must remain in repository: {raw}")
    return path


def set_repo_temp() -> None:
    path = repo_path("experiments_iclr/.tmp/ogbl_collab")
    path.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME"):
        os.environ[key] = str(path)
    tempfile.tempdir = str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor]]:
    return (random.getstate(), copy.deepcopy(np.random.get_state()),
            torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else [])


def restore_rng(state: tuple[Any, Any, torch.Tensor, list[torch.Tensor]]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


def same_rng(left: tuple[Any, Any, torch.Tensor, list[torch.Tensor]],
             right: tuple[Any, Any, torch.Tensor, list[torch.Tensor]]) -> bool:
    numpy_equal = (left[1][0] == right[1][0] and
                   np.array_equal(left[1][1], right[1][1]) and
                   left[1][2:] == right[1][2:])
    return (left[0] == right[0] and numpy_equal and torch.equal(left[2], right[2])
            and len(left[3]) == len(right[3]) and
            all(torch.equal(a, b) for a, b in zip(left[3], right[3])))


def model_kwargs(device: torch.device) -> dict[str, Any]:
    return dict(model_name="SAGE", num_layers=DEPTH, input_dim=128,
                hidden_dim=WIDTH, output_dim=WIDTH, hidden_dim_multiplier=1,
                num_heads=8, normalization="LayerNorm", dropout=DROPOUT)


class EdgeDecoder(nn.Module):
    """Symmetric score: a shared MLP on node-embedding elementwise products."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(WIDTH, WIDTH),
                                     nn.Linear(WIDTH, WIDTH), nn.Linear(WIDTH, 1)])
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = u * v
        for layer in self.layers[:-1]:
            out = self.dropout(F.relu(layer(out)))
        return self.layers[-1](out).squeeze(-1)


class UntiedTABMEncoder(nn.Module):
    """Identical step-zero member functions; only SAGE parameters are untied."""

    def __init__(self, reference: TABMModel) -> None:
        super().__init__()
        self.input_be_block = reference.input_be_block
        self.dropout = reference.dropout
        self.act = reference.act
        self.residual_modules = nn.ModuleList(
            [copy.deepcopy(reference.residual_modules) for _ in range(MEMBERS)]
        )
        self.output_normalization = reference.output_normalization
        self.output_be_block = reference.output_be_block

    def forward(self, graph: SimpleNamespace, x: torch.Tensor, member: int) -> torch.Tensor:
        h = self.input_be_block(x, tabm_seed=member)
        h = self.act(self.dropout(h))
        for layer in self.residual_modules[member]:
            h = layer(graph, h)
        h = self.output_normalization(h)
        return self.output_be_block(h, tabm_seed=member)


class LinkSystem(nn.Module):
    def __init__(self, variant: str, device: torch.device) -> None:
        super().__init__()
        self.variant = variant
        kw = model_kwargs(device)
        if variant == "tied":
            self.encoder = TABMModel(**kw, tabm_inits=MEMBERS, device=device)
            self.decoder = EdgeDecoder()
        elif variant == "untied":
            reference = TABMModel(**kw, tabm_inits=MEMBERS, device=device)
            # Keep decoder initialization and the post-construction random streams
            # identical to TIED. Extra module copies must not change either.
            self.decoder = EdgeDecoder()
            state_after_common = capture_rng()
            self.encoder = UntiedTABMEncoder(reference)
            restore_rng(state_after_common)
        elif variant == "ens":
            self.encoders = nn.ModuleList([Model(**kw) for _ in range(MEMBERS)])
            self.decoders = nn.ModuleList([EdgeDecoder() for _ in range(MEMBERS)])
        elif variant == "base":
            self.encoder = Model(**kw)
            self.decoder = EdgeDecoder()
        else:
            raise ValueError(variant)

    @property
    def n_members(self) -> int:
        return 1 if self.variant == "base" else MEMBERS

    def node_embeddings(self, graph: SimpleNamespace, x: torch.Tensor,
                        member: int) -> torch.Tensor:
        if self.variant in ("tied", "untied"):
            return self.encoder(graph, x, member)
        if self.variant == "ens":
            return self.encoders[member](graph, x)
        return self.encoder(graph, x)

    def scores(self, h: torch.Tensor, pairs: torch.Tensor, member: int) -> torch.Tensor:
        decoder = self.decoders[member] if self.variant == "ens" else self.decoder
        return decoder(h[pairs[:, 0]], h[pairs[:, 1]])


def pair_keys(pairs: np.ndarray, n_nodes: int) -> np.ndarray:
    low = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    high = np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    return low * n_nodes + high


class DataBundle:
    def __init__(self, data_root: Path, device: torch.device) -> None:
        from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

        dataset = PygLinkPropPredDataset(name="ogbl-collab", root=str(data_root))
        data = dataset[0]
        split = dataset.get_edge_split()
        if data.x is None or data.x.ndim != 2 or data.x.shape[1] != 128:
            raise RuntimeError("Expected OGBL-Collab node features [N,128]")
        if not hasattr(data, "edge_year") or data.edge_year is None:
            raise RuntimeError("Expected OGBL-Collab training-graph edge years")
        if data.edge_year.numel() != data.edge_index.shape[1]:
            raise RuntimeError("Graph edge-year vector does not match message edges")
        if int(torch.max(data.edge_year).item()) > 2017:
            raise RuntimeError("Message graph contains an edge after the 2017 train cutoff")
        n = int(data.num_nodes)
        if n != 235_868:
            raise RuntimeError(f"Unexpected node count: {n}")
        for stage in ("train", "valid", "test"):
            if "edge" not in split[stage] or split[stage]["edge"].ndim != 2:
                raise RuntimeError(f"Bad {stage} positive-edge split")
            if stage != "train" and "edge_neg" not in split[stage]:
                raise RuntimeError(f"Missing official {stage} negative edges")
        graph_pairs = data.edge_index.t().cpu().numpy()
        train_pairs = split["train"]["edge"].cpu().numpy()
        if (graph_pairs < 0).any() or (graph_pairs >= n).any():
            raise RuntimeError("Out-of-range message-graph endpoint")
        graph_keys = np.unique(pair_keys(graph_pairs, n))
        train_keys = np.unique(pair_keys(train_pairs, n))
        if not np.array_equal(graph_keys, train_keys):
            raise RuntimeError("Message graph differs from official training-edge pairs")
        directed_keys = graph_pairs[:, 0].astype(np.int64) * n + graph_pairs[:, 1]
        reverse_keys = graph_pairs[:, 1].astype(np.int64) * n + graph_pairs[:, 0]
        if not np.array_equal(np.unique(directed_keys), np.unique(reverse_keys)):
            raise RuntimeError("Expected both directions for each training graph edge")
        self.train_undirected_keys = graph_keys
        self.n_nodes = n
        self.x = data.x.float().to(device)
        self.graph = SimpleNamespace(edge_index=data.edge_index.to(device))
        self.split = split
        self.train_pos = split["train"]["edge"].cpu().numpy().astype(np.int64)
        self.evaluator = Evaluator(name="ogbl-collab")
        self.evaluator.K = 50
        self.fingerprints = {
            "x": sha256_tensor(data.x),
            "message_edges": sha256_tensor(data.edge_index),
            "message_edge_year": sha256_tensor(data.edge_year),
            "train_pos": sha256_tensor(split["train"]["edge"]),
            "valid_pos": sha256_tensor(split["valid"]["edge"]),
            "valid_neg": sha256_tensor(split["valid"]["edge_neg"]),
            "test_pos": sha256_tensor(split["test"]["edge"]),
            "test_neg": sha256_tensor(split["test"]["edge_neg"]),
        }

    def sample_train_pairs(self, seed: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(700_000_003 + seed * 10_000 + step)
        count = min(PAIRS_PER_STEP, len(self.train_pos))
        index = rng.choice(len(self.train_pos), size=count, replace=False)
        positive = self.train_pos[index]
        negative_key_parts: list[np.ndarray] = []
        missing = count
        while missing > 0:
            draw = max(1024, missing * 2)
            u = rng.integers(0, self.n_nodes, size=draw, dtype=np.int64)
            v = rng.integers(0, self.n_nodes, size=draw, dtype=np.int64)
            low = np.minimum(u, v)
            high = np.maximum(u, v)
            valid = low < high
            keys = low[valid] * self.n_nodes + high[valid]
            locations = np.searchsorted(self.train_undirected_keys, keys)
            in_bounds = locations < len(self.train_undirected_keys)
            exists = np.zeros(len(keys), dtype=bool)
            exists[in_bounds] = self.train_undirected_keys[locations[in_bounds]] == keys[in_bounds]
            candidates = keys[~exists]
            _, first_indices = np.unique(candidates, return_index=True)
            candidates = candidates[np.sort(first_indices)]
            if negative_key_parts:
                earlier = np.concatenate(negative_key_parts)
                candidates = candidates[~np.isin(candidates, earlier)]
            chosen = candidates[:missing]
            if len(chosen):
                negative_key_parts.append(chosen)
                missing -= len(chosen)
        negative_keys = np.concatenate(negative_key_parts)
        negative = np.stack((negative_keys // self.n_nodes,
                             negative_keys % self.n_nodes), axis=1)
        if len(negative) != count:
            raise AssertionError("Wrong sampled-negative count")
        return (torch.as_tensor(positive, dtype=torch.long, device=self.x.device),
                torch.as_tensor(negative, dtype=torch.long, device=self.x.device))


def balanced_bce(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return (F.softplus(-pos).mean() + F.softplus(neg).mean()) / 2


def hits50(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.ndim != 1 or neg.ndim != 1 or len(neg) < 50:
        raise ValueError("Expected 1D scores and >=50 official negatives")
    threshold = np.partition(neg, len(neg) - 50)[len(neg) - 50]
    return float(np.mean(pos > threshold))


def bce_numpy(pos: np.ndarray, neg: np.ndarray) -> float:
    return float((np.logaddexp(0, -pos).mean() + np.logaddexp(0, neg).mean()) / 2)


@torch.no_grad()
def score_split(model: LinkSystem, bundle: DataBundle, stage: str,
                chunk_size: int = 65_536) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    positive = bundle.split[stage]["edge"].to(bundle.x.device)
    negative = bundle.split[stage]["edge_neg"].to(bundle.x.device)
    member_pos, member_neg = [], []
    for member in range(model.n_members):
        h = model.node_embeddings(bundle.graph, bundle.x, member)
        for pairs, target in ((positive, member_pos), (negative, member_neg)):
            pieces = []
            for start in range(0, len(pairs), chunk_size):
                logits = model.scores(h, pairs[start:start + chunk_size], member)
                pieces.append(logits.detach().cpu().numpy().astype(np.float32))
            target.append(np.concatenate(pieces))
    return np.stack(member_pos), np.stack(member_neg)


def pooled_metrics(pos: np.ndarray, neg: np.ndarray,
                   evaluator: Any | None = None) -> tuple[float, float]:
    p = pos.mean(axis=0)
    n = neg.mean(axis=0)
    hit = hits50(p, n)
    if evaluator is not None:
        official = evaluator.eval({"y_pred_pos": p, "y_pred_neg": n})["hits@50"]
        if abs(hit - official) > 1e-12:
            raise RuntimeError("Local Hits@50 disagrees with official OGB evaluator")
    return hit, bce_numpy(p, n)


def better(hits: float, bce: float, old_hits: float, old_bce: float) -> bool:
    tol = 1e-12
    return hits > old_hits + tol or (abs(hits - old_hits) <= tol and bce < old_bce - tol)


def selected_paths(root: Path, variant: str, seed: int) -> Path:
    return root / variant / f"seed_{seed}"


def run(variant: str, seed: int, bundle: DataBundle, result_root: Path,
        device: torch.device) -> dict[str, Any]:
    run_dir = selected_paths(result_root, variant, seed)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Result directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(12_000 + seed)
    model = LinkSystem(variant, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    source = {"runner_sha256": sha256_file(Path(__file__)),
              "models_sha256": sha256_file(ROOT / "models.py")}
    config = {
        "protocol": PROTOCOL, "variant": variant, "seed": seed,
        "steps": STEPS, "eval_every": EVAL_EVERY, "pairs_per_step": PAIRS_PER_STEP,
        "width": WIDTH, "depth": DEPTH, "dropout": DROPOUT,
        "lr": LR, "weight_decay": WEIGHT_DECAY,
        "decoder": "three-layer-ReLU-elementwise-product-raw-logit",
        "message_graph": "official train edges only for all stages",
        "sampled_unknowns": "unordered nonself pairs absent from train edges only",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source": source, "data_fingerprints": bundle.fingerprints,
        "torch_version": torch.__version__,
    }
    write_json(run_dir / "run_config.json", config)
    checkpoint = run_dir / "selected_checkpoint.pt"
    best_hits, best_bce, best_step = -1.0, float("inf"), 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, STEPS + 1):
        pos_pairs, neg_pairs = bundle.sample_train_pairs(seed, step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_start = time.perf_counter()
        member_loss_values = []
        for member in range(model.n_members):
            h = model.node_embeddings(bundle.graph, bundle.x, member)
            pos = model.scores(h, pos_pairs, member)
            neg = model.scores(h, neg_pairs, member)
            loss = balanced_bce(pos, neg)
            (loss / model.n_members).backward()
            member_loss_values.append(float(loss.detach().item()))
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_start
        if step % EVAL_EVERY:
            continue
        eval_start = time.perf_counter()
        pos_valid, neg_valid = score_split(model, bundle, "valid")
        valid_hits, valid_bce = pooled_metrics(pos_valid, neg_valid, bundle.evaluator)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - eval_start
        chosen_now = better(valid_hits, valid_bce, best_hits, best_bce)
        if chosen_now:
            best_hits, best_bce, best_step = valid_hits, valid_bce, step
            temp = checkpoint.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "variant": variant, "seed": seed, "protocol": PROTOCOL}, temp)
            temp.replace(checkpoint)
        row = {"step": step, "train_mean_member_bce": float(np.mean(member_loss_values)),
               "valid_pooled_hits50": valid_hits, "valid_pooled_bce": valid_bce,
               "train_seconds": train_seconds, "validation_seconds": validation_seconds,
               "elapsed_seconds": time.perf_counter() - started,
               "selected_so_far": int(chosen_now)}
        history.append(row)
        with (run_dir / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, HISTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
    if len(history) != STEPS // EVAL_EVERY or best_step == 0:
        raise RuntimeError("Incomplete validation history")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved["step"] != best_step or saved["variant"] != variant or saved["seed"] != seed:
        raise RuntimeError("Selected checkpoint metadata mismatch")
    model.load_state_dict(saved["state_dict"], strict=True)
    pos_valid, neg_valid = score_split(model, bundle, "valid")
    replay_hits, replay_bce = pooled_metrics(pos_valid, neg_valid, bundle.evaluator)
    if abs(replay_hits - best_hits) > 1e-12 or abs(replay_bce - best_bce) > 1e-6:
        raise RuntimeError("Selected checkpoint validation replay disagrees")
    # Only after selection and validation replay is the test split scored.
    pos_test, neg_test = score_split(model, bundle, "test")
    test_hits, test_bce = pooled_metrics(pos_test, neg_test, bundle.evaluator)
    np.savez_compressed(
        run_dir / "selected_predictions.npz", valid_pos_edges=bundle.split["valid"]["edge"].numpy(),
        valid_neg_edges=bundle.split["valid"]["edge_neg"].numpy(),
        test_pos_edges=bundle.split["test"]["edge"].numpy(),
        test_neg_edges=bundle.split["test"]["edge_neg"].numpy(),
        valid_pos_member_logits=pos_valid, valid_neg_member_logits=neg_valid,
        test_pos_member_logits=pos_test, test_neg_member_logits=neg_test,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "protocol": PROTOCOL, "variant": variant, "seed": seed,
        "selected_step": best_step, "steps_run": STEPS,
        "valid_hits50": replay_hits, "valid_bce": replay_bce,
        "test_hits50": test_hits, "test_bce": test_bce,
        "valid_member_hits50": [hits50(p, n) for p, n in zip(pos_valid, neg_valid)],
        "test_member_hits50": [hits50(p, n) for p, n in zip(pos_test, neg_test)],
        "parameter_count": config["parameter_count"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "predictions_sha256": sha256_file(run_dir / "selected_predictions.npz"),
        "history_sha256": sha256_file(run_dir / "history.csv"),
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_mib": (torch.cuda.max_memory_allocated(device) / 2**20
                         if device.type == "cuda" else None),
    }
    write_json(run_dir / "selected.json", result)
    print(json.dumps({"complete": True, **result}), flush=True)
    return result


def initial_match_check(device: torch.device) -> None:
    """No OGB labels or splits are used by this fixed synthetic check."""
    graph = SimpleNamespace(edge_index=torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 3], [1, 2, 3, 4, 5, 0, 4, 0]], device=device))
    x = torch.linspace(-1, 1, 6 * 128, device=device).reshape(6, 128)
    pairs = torch.tensor([[0, 1], [2, 5], [4, 3]], device=device)
    seed_all(12_000)
    tied = LinkSystem("tied", device).to(device).eval()
    tied_rng = capture_rng()
    seed_all(12_000)
    untied = LinkSystem("untied", device).to(device).eval()
    untied_rng = capture_rng()
    if not same_rng(tied_rng, untied_rng):
        raise RuntimeError("Tied/untied post-construction random states differ")

    def require_same_state(left: nn.Module, right: nn.Module, label: str) -> None:
        lhs, rhs = left.state_dict(), right.state_dict()
        if lhs.keys() != rhs.keys():
            raise RuntimeError(f"{label} state key mismatch")
        for key in lhs:
            if not torch.equal(lhs[key], rhs[key]):
                raise RuntimeError(f"{label}.{key} differs at initialization")

    require_same_state(tied.encoder.input_be_block,
                       untied.encoder.input_be_block, "input_projector")
    require_same_state(tied.encoder.output_normalization,
                       untied.encoder.output_normalization, "output_norm")
    require_same_state(tied.encoder.output_be_block,
                       untied.encoder.output_be_block, "output_projector")
    require_same_state(tied.decoder, untied.decoder, "edge_decoder")
    for member in range(MEMBERS):
        require_same_state(tied.encoder.residual_modules,
                           untied.encoder.residual_modules[member],
                           f"propagation_stack_member_{member}")
    with torch.no_grad():
        for member in range(MEMBERS):
            t = tied.scores(tied.node_embeddings(graph, x, member), pairs, member)
            u = untied.scores(untied.node_embeddings(graph, x, member), pairs, member)
            if not torch.allclose(t, u, rtol=0, atol=1e-6):
                raise RuntimeError(f"Tied/untied initial logits differ for member {member}")
    positive = torch.tensor([[0, 1], [2, 3]], device=device)
    negative = torch.tensor([[0, 3], [1, 5]], device=device)
    for variant in VARIANTS:
        seed_all(12_000)
        model = LinkSystem(variant, device).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer.zero_grad(set_to_none=True)
        for member in range(model.n_members):
            h = model.node_embeddings(graph, x, member)
            loss = balanced_bce(model.scores(h, positive, member),
                                model.scores(h, negative, member))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite synthetic loss in {variant}")
            (loss / model.n_members).backward()
        optimizer.step()
    print(json.dumps({"synthetic_initial_match": True,
                      "synthetic_backward_arms": list(VARIANTS), "members": MEMBERS}))


def timing_probe(variant: str, bundle: DataBundle, device: torch.device) -> None:
    """Five real training steps and one validation forward; no metric or test scoring."""
    seed_all(12_000)
    model = LinkSystem(variant, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_times = []
    for step in range(1, 6):
        positive, negative = bundle.sample_train_pairs(0, step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        for member in range(model.n_members):
            h = model.node_embeddings(bundle.graph, bundle.x, member)
            loss = balanced_bce(model.scores(h, positive, member),
                                model.scores(h, negative, member))
            (loss / model.n_members).backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_times.append(time.perf_counter() - start)
    start = time.perf_counter()
    _ = score_split(model, bundle, "valid")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    valid_seconds = time.perf_counter() - start
    estimate_per_seed = 400 * float(np.median(train_times)) + 20 * valid_seconds
    print(json.dumps({
        "timing_probe": True, "variant": variant,
        "source_sha256": sha256_file(Path(__file__)),
        "data_fingerprints": bundle.fingerprints,
        "training_seconds_each": train_times,
        "validation_forward_seconds": valid_seconds,
        "estimated_per_seed_seconds": estimate_per_seed,
        "three_seed_estimate_seconds": 3 * estimate_per_seed,
        "no_test_scores_computed": True,
    }, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen OGBL-Collab breadth study")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data/ogb")
    parser.add_argument("--result-root", default="experiments_iclr/ogbl_collab_results")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timing-probe", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and args.timing_probe:
        parser.error("Choose either --preflight-only or --timing-probe")
    set_repo_temp()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    initial_match_check(device)
    if args.preflight_only:
        print(json.dumps({"preflight_only": True, "protocol": PROTOCOL}))
        return
    if args.variant is None or (args.seed is None and not args.timing_probe):
        parser.error("--variant required; --seed required for production")
    data_root = repo_path(args.data_root)
    result_root = repo_path(args.result_root)
    data_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    bundle = DataBundle(data_root, device)
    print(json.dumps({"dataset_loaded": True, "fingerprints": bundle.fingerprints,
                      "train_rows": len(bundle.train_pos), "device": str(device)}), flush=True)
    if args.timing_probe:
        timing_probe(args.variant, bundle, device)
        return
    run(args.variant, args.seed, bundle, result_root, device)


if __name__ == "__main__":
    main()
