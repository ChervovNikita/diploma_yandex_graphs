"""Roman mask-0 adapter for the byte-pinned external mechanism primitives.

This module deliberately provides the narrow API used by mechanism.py and
norm_sync.py. It does not import their training, tuning, or scoring workflows.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import roman_multimask as roman

ROOT = Path(__file__).resolve().parent
CANDIDATES = ((0.001, 0.0),)
INITIAL_LOGIT_TOL = 1e-5
MEMBERS = 4
WIDTH = 128


def configure(depth: int) -> None:
    if depth not in (2, 5):
        raise ValueError(depth)
    roman.MASK = 0
    roman.DEPTH = depth


def sha256_file(path: Path) -> str:
    return roman.sha256_file(path)


def write_json(path: Path, value) -> None:
    roman.write_json(path, value)


def seed_all(seed: int) -> None:
    roman.seed_all(seed)


def state_sha(state: dict) -> str:
    return roman.hash_state(state)


def tensor_sha(tensor: torch.Tensor) -> str:
    return roman.hash_tensor(tensor)


def make_model(arm: str, bundle, device: torch.device):
    if arm not in ("tied", "untied"):
        raise ValueError(arm)
    roman_arm = "tied" if arm == "tied" else "untied_propagation"
    return roman.make_model(roman_arm, bundle.x.size(1), bundle.classes, device)


def member_logits(model, arm: str, bundle):
    if arm not in ("tied", "untied"):
        raise ValueError(arm)
    return [model(bundle.graph, bundle.x, tabm_seed=m) for m in range(MEMBERS)]


@torch.no_grad()
def evaluate(model, arm: str, bundle, idx, labels):
    model.eval()
    logits = torch.stack(member_logits(model, arm, bundle), dim=0)[:, idx]
    pooled = logits.mean(0)
    acc = float((pooled.argmax(-1) == labels).float().mean().item())
    ce = float(F.cross_entropy(pooled, labels).item())
    return acc, ce, logits, pooled


def load_graph(dataset: str, device: torch.device, include_test: bool = False):
    if dataset != "roman":
        raise ValueError(dataset)
    x, y, raw, edge, train, valid, test = roman.load_roman()
    if edge.shape != (2, 65854) or bool((edge[0] == edge[1]).any()):
        raise RuntimeError("Wrong no-added-loop Roman graph")
    indices = {name: mask.nonzero().flatten().long()
               for name, mask in (("train", train), ("valid", valid), ("test", test))}
    if tuple(int(indices[k].numel()) for k in ("train", "valid", "test")) != (11331, 5665, 5666):
        raise RuntimeError("Wrong official mask-0 partition")
    if torch.any(train & valid) or torch.any(train & test) or torch.any(valid & test):
        raise RuntimeError("Overlapping Roman masks")
    bundle = SimpleNamespace(graph=SimpleNamespace(edge_index=edge.to(device)),
                             x=x.to(device),
                             train_idx=indices["train"].to(device),
                             train_y=y[indices["train"]].to(device),
                             valid_idx=indices["valid"].to(device),
                             valid_y=y[indices["valid"]].to(device),
                             classes=int(y.max().item()) + 1)
    if include_test:
        bundle.test_idx_cpu = indices["test"]
        bundle.test_y_cpu = y[indices["test"]]
    descriptor = {"mask": 0, "depth": roman.DEPTH,
                  "nodes": int(y.numel()), "classes": bundle.classes,
                  "train_size": int(train.sum()), "valid_size": int(valid.sum()),
                  "test_size": int(test.sum()), "raw_edges": raw.size(1),
                  "training_edges": edge.size(1), "added_self_loops": False,
                  "npz_sha256": sha256_file(ROOT / "data/roman_empire.npz"),
                  "train_indices_sha256": tensor_sha(indices["train"]),
                  "valid_indices_sha256": tensor_sha(indices["valid"]),
                  "test_indices_sha256": tensor_sha(indices["test"]),
                  "edge_sha256": tensor_sha(edge),
                  "features_sha256": tensor_sha(x),
                  "labels_sha256": tensor_sha(y)}
    return bundle, descriptor


@torch.no_grad()
def initial_audit(model, arm: str, bundle, seed: int,
                  device: torch.device, canonical_sha: str) -> dict:
    model.eval()
    python_sha = roman.hash_rng_state(random.getstate())
    numpy_sha = roman.hash_rng_state(np.random.get_state())
    cpu_sha = tensor_sha(torch.get_rng_state())
    cuda_sha = tensor_sha(torch.cuda.get_rng_state(device))
    logits = torch.stack(member_logits(model, arm, bundle), dim=0)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Nonfinite initial member logit")
    if (python_sha != roman.hash_rng_state(random.getstate()) or
            numpy_sha != roman.hash_rng_state(np.random.get_state()) or
            cpu_sha != tensor_sha(torch.get_rng_state()) or
            cuda_sha != tensor_sha(torch.cuda.get_rng_state(device))):
        raise RuntimeError("Initial evaluation changed RNG state")
    return {"seed": seed, "canonical_projector_state_sha256": canonical_sha,
            "python_rng_sha256": python_sha, "numpy_rng_sha256": numpy_sha,
            "cpu_rng_sha256": cpu_sha, "cuda_rng_sha256": cuda_sha,
            "initial_member_logits_shape": list(logits.shape),
            "initial_member_logits_sha256": tensor_sha(logits),
            "parameter_count": sum(p.numel() for p in model.parameters())}
