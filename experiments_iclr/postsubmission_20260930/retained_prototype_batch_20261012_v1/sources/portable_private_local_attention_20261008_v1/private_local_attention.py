"""Initially copied local GAT scorers; native propagation/value maps are unchanged.

Install after native reset and BE installation, BEFORE constructing an optimizer.
Retrofitting any existing optimizer is unsupported, including an empty Session's
optimizer. A new prospective constructor must install before its fresh Adam.
Every forward must select a member explicitly. Do not call native reset after
installation, cache parametrizations across members, deepcopy an installed bank,
or serialize the complete Python model. Rebuild and load state_dict instead.
No training driver, scientific release, target generation or score access here.
"""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch_geometric.nn import GATConv


class _SelectScorer(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = members
        self.member = 0  # Registration checks the initially copied row.

    def right_inverse(self, original):
        return original.detach().unsqueeze(0).expand(
            self.members, *original.shape).clone()

    def forward(self, bank):
        if self.member is None:
            raise RuntimeError("Explicit private-attention member context required")
        return bank[self.member]


class PrivateLocalAttention:
    """Controller for existing native attention getters, not another nn.Module."""
    def __init__(self, body, sites, members, before_parameters, original_scalars):
        self.body, self.sites, self.members = body, tuple(sites), members
        self.before_parameters = before_parameters
        self.original_scalars = original_scalars

    @contextmanager
    def member_context(self, member):
        """Select scorer rows only; the original BE context still selects factors.

        Mutable route selection follows the existing sequential BE execution.
        This controller does not support concurrent member forwards.
        """
        if isinstance(member, bool) or not isinstance(member, int):
            raise TypeError("Member must be an integer")
        if not 0 <= member < self.members:
            raise IndexError(member)
        selectors = [selector for _, _, selector in self.sites]
        previous = [selector.member for selector in selectors]
        try:
            for selector in selectors:
                selector.member = member
            yield
        finally:
            for selector, old in zip(selectors, previous):
                selector.member = old

    def forward_member(self, original_member_forward, batch, member):
        """Wrap a captured ORIGINAL ensemble.member_forward callable.

        The original callable retains BE factor selection and representation
        capture. This method supplies only the matching attention-row context.
        Never pass a wrapper that calls this method back recursively.
        """
        with self.member_context(member):
            return original_member_forward(batch, member)

    def metadata(self):
        banks = []
        for layer, name, _ in self.sites:
            bank = getattr(self.body.local_convs[layer].parametrizations, name).original
            banks.append({"layer": layer, "name": name, "shape": list(bank.shape)})
        after = sum(p.numel() for p in self.body.parameters())
        return {"members": self.members, "local_layers": len(self.body.local_convs),
                "banks": banks, "original_shared_scorer_scalars": self.original_scalars,
                "private_scorer_scalars": self.members * self.original_scalars,
                "total_original_prediction_parameters": self.before_parameters,
                "total_candidate_prediction_parameters": after,
                "additional_prediction_parameters": after - self.before_parameters,
                "expected_additional_prediction_parameters":
                    (self.members - 1) * self.original_scalars,
                "native_GAT_forward_replaced": False,
                "dense_value_maps_or_BE_factors_replaced": False,
                "explicit_member_scope_required": True,
                "scientific_training_enabled": False}


def install_private_local_attention(body, members=4):
    """Untie ONLY native local att_src/att_dst, with no new random draw.

    Torch parametrizations retain each original scorer Parameter object's id,
    expanding its storage to [M, 1, heads, channels]. Native GAT's attribute
    getter returns one [1, heads, channels] view. No unused common scorer remains.
    State keys become local_convs.i.parametrizations.att_*.original.

    Call only on a freshly reset native Polynormer body with BE already installed,
    before Adam construction. Old optimizer retrofit is unsupported.
    Shape/alias checks precede mutation; scientific-source and anchor matching
    remain the driver's responsibility.
    """
    if isinstance(members, bool) or not isinstance(members, int) or members < 2:
        raise ValueError("At least two integer members required")
    if not isinstance(getattr(body, "local_convs", None), nn.ModuleList):
        raise TypeError("Native local_convs ModuleList required")
    if len(body.local_convs) == 0:
        raise ValueError("At least one native local attention layer required")
    pending, identities = [], set()
    for i, conv in enumerate(body.local_convs):
        if not isinstance(conv, GATConv):
            raise TypeError("Native GATConv required at every local site")
        if conv.add_self_loops or conv.bias is not None or conv.edge_dim is not None:
            raise ValueError("Pinned no-self-loop/no-bias/no-edge native GAT contract required")
        if not conv.concat or getattr(conv, "res", None) is not None:
            raise ValueError("Pinned concat/no-internal-residual native GAT contract required")
        for name in ("att_src", "att_dst"):
            if parametrize.is_parametrized(conv, name):
                raise ValueError("Scorer already parametrized; installation is not repeatable")
            scorer = getattr(conv, name)
            if not isinstance(scorer, nn.Parameter) or not scorer.requires_grad:
                raise TypeError("Native live scorer Parameter required")
            if tuple(scorer.shape) != (1, conv.heads, conv.out_channels):
                raise ValueError("Unexpected native scorer shape")
            if id(scorer) in identities:
                raise ValueError("Aliased native scorer sites require a separate contract")
            identities.add(id(scorer))
            pending.append((i, name, scorer.numel()))
    before = sum(p.numel() for p in body.parameters())
    sites = []
    for i, name, _ in pending:
        selector = _SelectScorer(members)
        # Default safety checks verify the selected output's native shape/dtype.
        parametrize.register_parametrization(body.local_convs[i], name, selector)
        selector.member = None
        sites.append((i, name, selector))
    return PrivateLocalAttention(body, sites, members, before,
                                 sum(n for _, _, n in pending))
