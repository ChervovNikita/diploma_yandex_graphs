"""Wrap original SAGE neighbor maps; retain source propagation and bias."""

import copy

import torch
from torch import nn
from torch.nn import functional as F


def helmert(members, *, dtype=torch.float64, device="cpu"):
    if type(members) is not int or members < 2:
        raise ValueError("Need at least two members")
    value = torch.zeros(members, members - 1, dtype=dtype, device=device)
    for q in range(members - 1):
        denominator = ((q + 1) * (q + 2)) ** 0.5
        value[:q + 1, q] = 1.0 / denominator
        value[q + 1, q] = -(q + 1) / denominator
    return value


def source_member_count(source):
    """Read the original TABMModel ABI from its two BEBlock boundaries."""
    boundaries = [getattr(source, name, None)
                  for name in ("input_be_block", "output_be_block")]
    if any(boundary is None for boundary in boundaries):
        raise ValueError("Require original TABMModel member boundaries")
    first_r = getattr(boundaries[0], "R", None)
    if not isinstance(first_r, torch.Tensor) or first_r.ndim != 2:
        raise ValueError("Source input boundary R must be a matrix")
    members = first_r.shape[0]
    if members < 2:
        raise ValueError("Need at least two source members")
    for boundary in boundaries:
        count = getattr(boundary, "tabm_inits", None)
        if type(count) is not int or count != members:
            raise ValueError("Inconsistent source boundary member count")
        for name in ("R", "S", "B"):
            value = getattr(boundary, name, None)
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != members:
                raise ValueError("Inconsistent source boundary R/S/B member dimensions")
    return members


def source_neighbor_maps(source):
    if getattr(source, "model_name", None) != "SAGE":
        raise ValueError("Prototype only supports the original source SAGE")
    source_member_count(source)
    result = {}
    for layer, residual in enumerate(source.residual_modules):
        conv = residual.module.conv
        if type(conv).__name__ != "SAGEConv" or not hasattr(conv, "lin_l"):
            raise ValueError("Target must be the original SAGEConv.lin_l map")
        base = conv.lin_l.base if isinstance(conv.lin_l, FrozenOutputNeighbor) else conv.lin_l
        if base.weight.ndim != 2 or base.weight.shape[0] != base.weight.shape[1]:
            raise ValueError("Neighbor transform must be square [out,in]")
        if not isinstance(residual.normalization, nn.LayerNorm):
            raise ValueError("First prototype requires source LayerNorm")
        result[f"residual_modules.{layer}.module.conv.lin_l.weight"] = (conv, base)
    if not result:
        raise ValueError("No source neighbor targets")
    return result


class FrozenOutputNeighbor(nn.Module):
    """One source map with U sum_q C_mq A_q; U frozen, A initially zero.

    C is stored once by the owning ensemble. Forward context is temporary
    and does not enter state_dict. Serial member calls are supported;
    concurrent calls to the same object are not.
    """

    def __init__(self, base, basis, members):
        super().__init__()
        self.base = base
        h, inputs = base.weight.shape
        basis = basis.detach().to(device=base.weight.device, dtype=base.weight.dtype).clone()
        if basis.ndim != 2 or basis.shape[0] != h or basis.shape[1] > h:
            raise ValueError("Output basis must have shape [out,rank<=out]")
        if not bool(torch.isfinite(basis).all()):
            raise ValueError("Nonfinite output basis")
        r = basis.shape[1]
        tolerance = 5e-6 if basis.dtype == torch.float32 else 1e-10
        if not torch.allclose(basis.T @ basis, torch.eye(r, dtype=basis.dtype, device=basis.device),
                              atol=tolerance, rtol=tolerance):
            raise ValueError("Output basis columns must be orthonormal")
        self.register_buffer("U", basis)
        self.A = nn.Parameter(torch.zeros(members - 1, r, inputs,
                                          dtype=base.weight.dtype, device=base.weight.device))
        self.members = members
        self.current_member = None
        self.current_contrasts = None

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def rank(self):
        return self.U.shape[1]

    def delta_weight(self, member, contrasts):
        if not 0 <= member < self.members:
            raise ValueError("Invalid member")
        coefficients = torch.einsum("q,qri->ri", contrasts[member], self.A)
        return self.U @ coefficients

    def forward(self, features):
        output = self.base(features)  # Includes the original lin_l bias exactly once.
        if not self.rank:
            return output
        if self.current_member is None or self.current_contrasts is None:
            raise RuntimeError("Use the owning ensemble's forward_member")
        coefficients = torch.einsum("q,qri->ri", self.current_contrasts[self.current_member], self.A)
        return output + F.linear(F.linear(features, coefficients), self.U)


class SourceSAGEEnsemble(nn.Module):
    """Source TABMModel plus fixed-target frozen-output corrections.

    The default recipe gate is M4/h128/three blocks. Disabling it is only
    for explicit synthetic or separately versioned compatibility checks.
    No source reset_parameters() is called and no RNG is consumed.
    """

    def __init__(self, source, bases=None, *, clone_source=True, enforce_recipe=True):
        super().__init__()
        maps = source_neighbor_maps(source)
        if any(isinstance(conv.lin_l, FrozenOutputNeighbor) for conv, _ in maps.values()):
            raise ValueError("Do not wrap an already adapted source")
        members = source_member_count(source)
        if enforce_recipe and (members != 4 or len(maps) != 3
                               or any(base.weight.shape != (128, 128) for _, base in maps.values())):
            raise ValueError("Frozen first recipe is M4/h128/three SAGE blocks")
        if enforce_recipe:
            for residual in source.residual_modules:
                ff = residual.module.feed_forward_module
                if ff.linear_1.weight.shape != (128, 256) or ff.linear_2.weight.shape != (128, 128):
                    raise ValueError("Frozen first recipe requires hidden multiplier one")
        self.source = copy.deepcopy(source) if clone_source else source
        self.members = members
        first_weight = next(iter(maps.values()))[1].weight
        self.register_buffer("C", helmert(members, dtype=first_weight.dtype, device=first_weight.device))
        maps = source_neighbor_maps(self.source)
        if bases is not None and set(bases) != set(maps):
            raise ValueError("Basis keys must exactly match all fixed neighbor targets")
        # A plain mapping avoids duplicate module registration/state_dict entries.
        self._neighbors = {}
        for key, (conv, base) in maps.items():
            basis = (torch.empty(base.weight.shape[0], 0, dtype=base.weight.dtype,
                                 device=base.weight.device) if bases is None else bases[key])
            adapter = FrozenOutputNeighbor(base, basis, members)
            conv.lin_l = adapter
            self._neighbors[key] = adapter

    def named_neighbor_matrices(self):
        return {key: adapter.base.weight for key, adapter in self._neighbors.items()}

    def named_private_coefficients(self):
        return {key: adapter.A for key, adapter in self._neighbors.items()}

    def frozen_bases(self):
        return {key: adapter.U for key, adapter in self._neighbors.items()}

    def private_corrections(self):
        return {key: torch.stack([adapter.delta_weight(m, self.C) for m in range(self.members)])
                for key, adapter in self._neighbors.items()}

    def is_tied_at_state(self):
        return all(not bool(torch.count_nonzero(adapter.A)) for adapter in self._neighbors.values())

    def tied_source_copy(self):
        """Export a plain source checkpoint after a tied P-only probe fit."""
        if not self.is_tied_at_state():
            raise ValueError("Cannot discard nonzero private coefficients")
        result = copy.deepcopy(self.source)
        for conv, _ in source_neighbor_maps(result).values():
            conv.lin_l = conv.lin_l.base
        return result

    def forward_member(self, graph, features, member):
        if type(member) is not int or not 0 <= member < self.members:
            raise ValueError("Invalid member index")
        previous = [(adapter, adapter.current_member, adapter.current_contrasts)
                    for adapter in self._neighbors.values()]
        try:
            for adapter in self._neighbors.values():
                adapter.current_member, adapter.current_contrasts = member, self.C
            return self.source(graph, features, tabm_seed=member)
        finally:
            for adapter, old_member, old_contrasts in previous:
                adapter.current_member, adapter.current_contrasts = old_member, old_contrasts

    def forward(self, graph, features):
        return torch.stack([self.forward_member(graph, features, m) for m in range(self.members)])

    def added_storage(self):
        return {
            "private_trainable_scalars": sum(adapter.A.numel() for adapter in self._neighbors.values()),
            "basis_buffer_scalars": sum(adapter.U.numel() for adapter in self._neighbors.values()),
            "contrast_buffer_scalars": self.C.numel(),
            "ranks": {key: adapter.rank for key, adapter in self._neighbors.items()},
        }
