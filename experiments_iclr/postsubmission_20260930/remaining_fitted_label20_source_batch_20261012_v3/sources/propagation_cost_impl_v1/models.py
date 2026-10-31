"""Same source parameters; intervene only in linear neighbor aggregation.

Fixture-only execution helpers. No data acquisition or scientific fit runner.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contract import check_pins, fixture_shape
from .dropout import dropout_member, dropout_packed

check_pins()
from ._pinned.frozen_centered_paths import PackedSharedSAGEEnsemble, PlainSAGE


def _validate_source(model, layers, p):
    if model.model_name != "SAGE" or len(model.residual_modules) != layers:
        raise ValueError("Matched source Model/TABMModel SAGE required")
    norms = [r.normalization for r in model.residual_modules] + [model.output_normalization]
    if any(not isinstance(norm, nn.LayerNorm) or norm.eps != 1e-5 for norm in norms):
        raise ValueError("All source normalizations must be default LayerNorm")
    if model.dropout.p != p:
        raise ValueError("Input dropout does not match the fixed source")
    for residual in model.residual_modules:
        module = residual.module
        if module.conv.aggr != "mean":
            raise ValueError("Incoming mean SAGE only")
        if module.feed_forward_module.dropout_1.p != p or module.feed_forward_module.dropout_2.p != p:
            raise ValueError("Source FFN dropout does not match")


def _paired_block(block, normalized, propagated, *, p, training, key,
                  layer, member=None):
    message = block.neighbor(propagated) + block.root(normalized)
    hidden = block.first(torch.cat((normalized, message), dim=-1))
    if member is None:
        hidden = dropout_packed(hidden, p, training, key, layer, "ffn_first")
    else:
        hidden = dropout_member(hidden, p, training, key, member, layer, "ffn_first")
    hidden = block.second(F.gelu(hidden))
    if member is None:
        return dropout_packed(hidden, p, training, key, layer, "ffn_second")
    return dropout_member(hidden, p, training, key, member, layer, "ffn_second")


def centered_aggregation(family, normalized, color, *, trace=False):
    """Compute differentiable post-LayerNorm mean and a common residual mask.

    Only mean of these LINEAR aggregates equals the full aggregate of the
    member mean. Do not transfer the identity through member nonlinearities.
    """
    mean = normalized.mean(dim=1)
    deviation = normalized - mean[:, None, :]
    family.ledger.record_dense("differentiable_post_norm_mean_and_center", normalized)
    common = family.full(mean)
    residual = family.color(color)(deviation.flatten(1)).reshape_as(normalized) * 4.0
    propagated = common[:, None, :] + residual
    family.ledger.record_dense("common_plus_residual", propagated)
    if not trace:
        return propagated
    with torch.no_grad():
        scale = normalized.detach().norm(dim=(0, 2)).sum().clamp_min(1.0)
        diagnostics = {
            "centered_sum_normalized_norm": deviation.detach().sum(dim=1).norm() / scale,
            "aggregation_mean_normalized_error": (
                propagated.detach().mean(dim=1) - common.detach()).norm() / scale,
            "contrast_energy_normalized": deviation.detach().square().sum()
                / normalized.detach().square().sum().clamp_min(1e-30),
            "dtype": str(normalized.dtype), "color": int(color),
        }
    return propagated, diagnostics


class PropagationSAGE(PackedSharedSAGEEnsemble):
    """Exact, centered-quarter and naive-half arms with identical tensors.

    Candidate serial member-at-a-time full trajectories are invalid because
    the current member mean couples layers. Its declared path is packed.
    Exact and naive arms support packed or serial member trajectories.
    """

    def __init__(self, input_dim, output_dim, dim=64, members=4, layers=2,
                 dropout=0.2, *, arm="candidate", naive_path="lazy_two_colors",
                 checkpoint_blocks=False):
        if members != 4 or layers != 2 or dim > 64 or dim < 1:
            raise ValueError("Bounded source fixture uses M4, L2, D<=64")
        if arm not in ("candidate", "exact_member", "naive_shared_sample"):
            raise ValueError("Unknown ensemble arm")
        if naive_path not in ("lazy_two_colors", "cached_half"):
            raise ValueError("Unknown naive execution path")
        super().__init__(input_dim, output_dim, dim, members, layers, dropout)
        self.input_dim, self.output_dim = input_dim, output_dim
        self.p, self.arm, self.naive_path = dropout, arm, naive_path
        self.checkpoint_blocks = bool(checkpoint_blocks)

    @torch.no_grad()
    def copy_from_source_tabm(self, model):
        _validate_source(model, self.layers, self.p)
        return super().copy_from_source_tabm(model)

    def _propagate(self, family, normalized, color):
        if self.arm == "candidate":
            return centered_aggregation(family, normalized, color)
        flat = normalized.flatten(1)
        if self.arm == "exact_member":
            return family.full(flat).reshape_as(normalized)
        return family.naive(flat, color, self.naive_path).reshape_as(normalized)

    def _check(self, family, features, colors):
        fixture_shape(features.shape[0], family.full.nnz, features.shape[1])
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Input features must match the transferred source")
        if len(colors) != self.layers or any(color not in range(4) for color in colors):
            raise ValueError("Two immutable layer colors required, including exact controls")

    def forward(self, family, features, *, colors=(0, 2), dropout_key=None,
                execution="packed", return_trace=False):
        self._check(family, features, colors)
        if execution == "serial":
            if self.arm == "candidate" or return_trace:
                raise ValueError("Candidate needs coupled packed states; serial trace is unsupported")
            return torch.stack([
                self.forward_member(family, features, member, colors=colors,
                                    dropout_key=dropout_key)
                for member in range(self.members)
            ], dim=0)
        if execution != "packed":
            raise ValueError("Execution must be declared before outcomes")
        if return_trace and self.checkpoint_blocks and self.training:
            raise ValueError("Measure diagnostics separately from checkpoint timings")
        weights = self.input_weight[None, :, :] * self.input_r[:, :, None]
        hidden = torch.einsum("nf,mfd->nmd", features, weights)
        hidden = hidden * self.input_s + self.input_b
        hidden = F.gelu(dropout_packed(hidden, self.p, self.training, dropout_key, -1, "input"))
        traces = []
        for layer, (norm, block, color) in enumerate(zip(self.norm, self.blocks, colors)):
            # Freeze closure values explicitly: checkpoint backward occurs
            # after the loop has advanced to later layers.
            def step(state, layer=layer, norm=norm, block=block, color=color):
                normalized = norm(state)
                propagated = self._propagate(family, normalized, color)
                family.ledger.record_dense("source_sage_ffn_concat", normalized,
                                           concatenated_channels=2 * self.dim)
                return state + _paired_block(
                    block, normalized, propagated, p=self.p, training=self.training,
                    key=dropout_key, layer=layer,
                )
            if return_trace and self.arm == "candidate":
                normalized = norm(hidden)
                propagated, diagnostic = centered_aggregation(family, normalized, color, trace=True)
                hidden = hidden + _paired_block(
                    block, normalized, propagated, p=self.p, training=self.training,
                    key=dropout_key, layer=layer,
                )
                traces.append(diagnostic)
            elif self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    step, hidden, use_reentrant=False, preserve_rng_state=False,
                    context_fn=lambda: (family.ledger.scope("checkpoint_forward"),
                                        family.ledger.scope("checkpoint_recompute")),
                )
            else:
                hidden = step(hidden)
        output_weights = self.output_weight[None, :, :] * self.output_r[:, :, None]
        result = torch.einsum("nmd,mdc->nmc", self.output_norm(hidden), output_weights)
        result = (result * self.output_s + self.output_b).permute(1, 0, 2)
        return (result, traces) if return_trace else result

    def forward_member(self, family, features, member, *, colors=(0, 2), dropout_key=None):
        self._check(family, features, colors)
        if self.arm == "candidate":
            raise ValueError("Member-at-a-time candidate trajectory changes the method")
        if member not in range(self.members):
            raise ValueError("Member outside frozen ensemble")
        hidden = features @ (self.input_weight * self.input_r[member, :, None])
        hidden = hidden * self.input_s[member] + self.input_b[member]
        hidden = F.gelu(dropout_member(
            hidden, self.p, self.training, dropout_key, member, -1, "input"))
        for layer, (norm, block, color) in enumerate(zip(self.norm, self.blocks, colors)):
            def step(state, layer=layer, norm=norm, block=block, color=color):
                normalized = norm(state)
                if self.arm == "exact_member":
                    propagated = family.full(normalized)
                else:
                    propagated = family.naive(normalized, color, self.naive_path)
                family.ledger.record_dense("source_sage_ffn_concat", normalized,
                                           concatenated_channels=2 * self.dim)
                return state + _paired_block(
                    block, normalized, propagated, p=self.p, training=self.training,
                    key=dropout_key, layer=layer, member=member,
                )
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    step, hidden, use_reentrant=False, preserve_rng_state=False,
                    context_fn=lambda: (family.ledger.scope("checkpoint_forward"),
                                        family.ledger.scope("checkpoint_recompute")),
                )
            else:
                hidden = step(hidden)
        output_weight = self.output_weight * self.output_r[member, :, None]
        return (self.output_norm(hidden) @ output_weight) * self.output_s[member] + self.output_b[member]


class SourcePlainSAGE(PlainSAGE):
    def __init__(self, input_dim, output_dim, dim=64, layers=2, dropout=0.2,
                 *, checkpoint_blocks=False):
        if layers != 2 or dim > 64 or dim < 1:
            raise ValueError("Bounded source fixture uses L2, D<=64")
        super().__init__(input_dim, output_dim, dim, layers, dropout)
        self.input_dim, self.output_dim, self.dim = input_dim, output_dim, dim
        self.layers, self.p = layers, dropout
        self.checkpoint_blocks = bool(checkpoint_blocks)

    @torch.no_grad()
    def copy_from_source_model(self, model):
        _validate_source(model, self.layers, self.p)
        self.input.load_state_dict(model.input_linear.state_dict())
        self.output.load_state_dict(model.output_linear.state_dict())
        self.output_norm.load_state_dict(model.output_normalization.state_dict())
        for layer, residual in enumerate(model.residual_modules):
            self.norm[layer].load_state_dict(residual.normalization.state_dict())
            self.blocks[layer].neighbor.load_state_dict(residual.module.conv.lin_l.state_dict())
            self.blocks[layer].root.load_state_dict(residual.module.conv.lin_r.state_dict())
            ff = residual.module.feed_forward_module
            self.blocks[layer].first.load_state_dict(ff.linear_1.state_dict())
            self.blocks[layer].second.load_state_dict(ff.linear_2.state_dict())

    def forward(self, family, features, *, colors=(0, 2), dropout_key=None):
        fixture_shape(features.shape[0], family.full.nnz, features.shape[1])
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Source input shape mismatch")
        hidden = F.gelu(dropout_member(self.input(features), self.p, self.training,
                                       dropout_key, 0, -1, "input"))
        for layer, (norm, block) in enumerate(zip(self.norm, self.blocks)):
            def step(state, layer=layer, norm=norm, block=block):
                normalized = norm(state)
                propagated = family.full(normalized)
                return state + _paired_block(
                    block, normalized, propagated, p=self.p, training=self.training,
                    key=dropout_key, layer=layer, member=0,
                )
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    step, hidden, use_reentrant=False, preserve_rng_state=False,
                    context_fn=lambda: (family.ledger.scope("checkpoint_forward"),
                                        family.ledger.scope("checkpoint_recompute")),
                )
            else:
                hidden = step(hidden)
        return self.output(self.output_norm(hidden))[None, :, :]


def source_parameter_counts(input_dim, output_dim, dim=64, layers=2, members=4):
    ordinary = 5 * layers * dim**2 + dim * (input_dim + output_dim + 3 + 5 * layers) + output_dim
    ensemble = (input_dim * dim + dim * output_dim + members * input_dim
                + 3 * members * dim + 2 * members * output_dim
                + layers * (5 * dim**2 + 5 * dim) + 2 * dim)
    return {"single_base": ordinary, "candidate": ensemble,
            "exact_member": ensemble, "naive_shared_sample": ensemble}
