"""Exact validated SAGE hidden-affine wrapper; no initialization RNG draws."""
from __future__ import annotations
import torch
from torch import nn


class MemberFactorLinear(nn.Module):
    """Member-specific rank-one factors around an existing shared affine map.

    The constructor and forward are copied from the existing Roman SAGE port.
    R indexes input features; S and B index output features.
    """

    def __init__(self, shared: nn.Module, members: int):
        super().__init__()
        weight = getattr(shared, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError("Expected a linear map with a two-dimensional weight")
        outputs, inputs = weight.shape
        self.shared = shared
        self.R = nn.Parameter(weight.new_ones((members, inputs)))
        self.S = nn.Parameter(weight.new_ones((members, outputs)))
        self.B = nn.Parameter(weight.new_zeros((members, outputs)))
        self.members = members
        self.active_member: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        member = self.active_member
        if member is None or not 0 <= member < self.members:
            raise RuntimeError("Select a BatchEnsemble member before forwarding")
        return self.shared(x * self.R[member]) * self.S[member] + self.B[member]

    @property
    def added_parameters(self) -> int:
        return self.R.numel() + self.S.numel() + self.B.numel()
