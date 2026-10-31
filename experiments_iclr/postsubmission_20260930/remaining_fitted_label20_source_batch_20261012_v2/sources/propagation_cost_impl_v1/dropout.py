"""Paired member noise independent of color sampling and execution packing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch


@dataclass(frozen=True)
class DropoutKey:
    task_index: int
    optimizer_seed: int
    epoch: int

    def seed(self, member, layer, site):
        payload = (f"gnnm-propagation-dropout-v1|{self.task_index}|"
                   f"{self.optimizer_seed}|{self.epoch}|{member}|{layer}|{site}")
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)


def dropout_member(features, p, training, key, member, layer, site):
    """An ND mask. It is reproducible on checkpoint recomputation.

    This preserves Bernoulli dropout's distribution and pairs the three
    ensemble arms, but is NOT the vanilla source RNG order. Source equivalence
    is qualified only with dropout disabled. Finite-precision kernel order
    and CUDA determinism remain runtime questions.
    """
    if not training or p == 0:
        return features
    if not 0 <= p < 1 or key is None:
        raise ValueError("Training dropout needs p<1 and an explicit immutable key")
    if features.ndim != 2:
        raise ValueError("Dropout member input must be [N,D]")
    generator = torch.Generator(device=features.device)
    generator.manual_seed(key.seed(member, layer, site))
    keep = torch.rand(features.shape, device=features.device, generator=generator) >= p
    # Autograd retains the fixed mask, never an edge-feature expansion.
    return features * keep.to(features.dtype) / (1.0 - p)


def dropout_packed(features, p, training, key, layer, site):
    if features.ndim != 3:
        raise ValueError("Packed dropout input must be [N,M,D]")
    if not training or p == 0:
        return features
    return torch.stack([
        dropout_member(features[:, member, :], p, training, key, member, layer, site)
        for member in range(features.shape[1])
    ], dim=1)


def training_colors(task_index, optimizer_seed, epochs=100, layers=2):
    """Prospective NumPy PCG64 stream, separate from Torch initialization/noise."""
    import numpy as np

    if task_index not in (0, 1) or optimizer_seed not in (0, 1, 2):
        raise ValueError("Frozen pilot task/seed required")
    generator = np.random.Generator(np.random.PCG64(20260930 + 1000 * task_index + optimizer_seed))
    return generator.integers(0, 4, size=(epochs, layers)).tolist()
