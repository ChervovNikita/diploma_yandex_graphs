"""Unexecuted source draft for a NEW prospective strong-backbone amendment.

No data acquisition, labels, optimizer, score method, fitting or launch is included.
Construct a fresh pinned author model per fit, then wrap once. Do not invoke its
native reset_parameters after wrapping. Numerical/runtime qualification is pending.
Ancestry: GNNM boundary projectors and BatchEnsemble input/output factor algebra.
"""

import copy
import torch
from torch import nn
from torch.nn import functional as F


class BoundaryProjector(nn.Module):
    """Shared W; private R/S/B. Author bias is copied into each private B row."""

    def __init__(self, native_linear, members, private_input_signs=False):
        super().__init__()
        if not isinstance(native_linear, nn.Linear):
            raise TypeError("This draft expects the pinned author's nn.Linear")
        self.members = int(members)
        self.weight = native_linear.weight
        self.R = nn.Parameter(self.weight.new_ones(members, self.weight.shape[1]))
        self.S = nn.Parameter(self.weight.new_ones(members, self.weight.shape[0]))
        bias = native_linear.bias
        if bias is None:
            bias = self.weight.new_zeros(self.weight.shape[0])
        self.B = nn.Parameter(bias.detach().expand(members, -1).clone())
        if private_input_signs:
            with torch.no_grad():
                self.R.bernoulli_(0.5).mul_(2).sub_(1)

    def forward(self, x, member):
        if not 0 <= member < self.members:
            raise IndexError("member outside admitted range")
        return F.linear(x * self.R[member], self.weight) * self.S[member] + self.B[member]


class PolyFormerBoundaryFamily(nn.Module):
    """Pinned native blocks retain [node, token, hidden] axes per member."""

    def __init__(self, fresh_native_model, members=4):
        super().__init__()
        self.core = fresh_native_model
        self.members = members
        self.stem = BoundaryProjector(self.core.lin1, members, private_input_signs=True)
        self.head = BoundaryProjector(self.core.lin3, members)
        # Transfer ownership: dormant author boundaries must not remain registered.
        self.core.lin1 = nn.Identity()
        self.core.lin3 = nn.Identity()

    def forward_member(self, tokens, member):
        if tokens.ndim != 3 or tokens.shape[1] != self.core.K:
            raise ValueError("tokens must be [N, author K+1, F]")
        x = self.stem(tokens, member)
        for block in self.core.attn:
            x = block(x)
        x = x.sum(dim=1)
        x = F.dropout(x, self.core.dropout, training=self.training)
        x = F.relu(self.core.lin2(x))
        x = F.dropout(x, self.core.dropout, training=self.training)
        return self.head(x, member)

    def forward(self, tokens):
        return torch.stack([self.forward_member(tokens, m) for m in range(self.members)], dim=0)


class PolynormerBoundaryFamily(nn.Module):
    """Pinned native local/global bodies run complete [node, hidden] trajectories."""

    def __init__(self, fresh_native_model, members=4):
        super().__init__()
        self.core = fresh_native_model
        self.members = members
        self.stem = BoundaryProjector(self.core.lin_in, members, private_input_signs=True)
        self.local_head = BoundaryProjector(self.core.pred_local, members)
        self.global_head = BoundaryProjector(self.core.pred_global, members)
        self.core.lin_in = nn.Identity()
        self.core.pred_local = nn.Identity()
        self.core.pred_global = nn.Identity()

    def set_global_stage(self, enabled):
        # Driver must bind this flag explicitly in checkpoint metadata.
        self.core._global = bool(enabled)

    def forward_member(self, features, edge_index, member):
        c = self.core
        x = F.dropout(features, p=c.in_drop, training=self.training)
        x = self.stem(x, member)
        x = F.dropout(x, p=c.dropout, training=self.training)
        x_local = 0
        for i, local_conv in enumerate(c.local_convs):
            if c.pre_ln:
                x = c.pre_lns[i](x)
            h = F.relu(c.h_lins[i](x))
            x = F.relu(local_conv(x, edge_index) + c.lins[i](x))
            x = F.dropout(x, p=c.dropout, training=self.training)
            beta = (torch.sigmoid(c.betas[i]) if c.beta < 0 else c.betas[i]).unsqueeze(0)
            x = (1 - beta) * c.lns[i](h * x) + beta * x
            x_local = x_local + x
        if c._global:
            return self.global_head(c.global_attn(c.ln(x_local)), member)
        return self.local_head(x_local, member)

    def forward(self, features, edge_index):
        return torch.stack(
            [self.forward_member(features, edge_index, m) for m in range(self.members)], dim=0
        )


def set_boundary_identity_(family):
    """Prospective graph-init helper: preserve the copied native W/bias function.

    Call directly after wrapping a warmed native single predictor. All R/S become
    one and B remains the copied native bias. This is NOT the baseline GNNM
    initialization, whose stem R is independently Rademacher. No reset follows.
    """
    with torch.no_grad():
        for module in family.modules():
            if isinstance(module, BoundaryProjector):
                module.R.fill_(1)
                module.S.fill_(1)
    return family


def copy_member_function_(family, source_member=0):
    """Clone one warmed boundary member into all K rows without changing W/body."""
    with torch.no_grad():
        for module in family.modules():
            if isinstance(module, BoundaryProjector):
                if not 0 <= source_member < module.members:
                    raise IndexError("member outside admitted range")
                for value in (module.R, module.S, module.B):
                    value.copy_(value[source_member].clone().expand_as(value))
    return family


def clone_warm_native_boundary(warm_native_model, backbone, members=4):
    """Copy a warm native predictor without consuming/resetting the donor.

    Copying must precede wrapping because boundary transfer replaces native
    owners with Identity. Charge both donor and clone at their transient peak.
    The returned raw wrapper has identity R/S and copied warm native B; its
    train/eval mode and Photo stage follow the donor. Baseline fits do not call it.
    """
    if backbone not in ('polyformer_mono', 'polynormer_r') or members != 4:
        raise ValueError("Only the registered native pair and four copies are admitted")
    cloned_native = copy.deepcopy(warm_native_model)
    wrapper = PolyFormerBoundaryFamily if backbone == 'polyformer_mono' else PolynormerBoundaryFamily
    family = set_boundary_identity_(wrapper(cloned_native, members=members))
    family.train(warm_native_model.training)
    return family
