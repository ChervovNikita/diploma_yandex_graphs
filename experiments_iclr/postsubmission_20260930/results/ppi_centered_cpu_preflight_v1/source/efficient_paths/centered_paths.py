"""Research prototype: common graph state and centered narrow relational paths.

No dataset access, training loop, checkpoint reuse, or CUDA default is included.
This is a new constrained architecture, not an exact acceleration of TABMModel.
The sparse mean operator matches PyG SAGEConv's incoming-neighbor mean.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def helmert_contrasts(members: int, dtype=None) -> torch.Tensor:
    """C^T C = I and 1^T C = 0; C has M rows and M-1 columns."""
    if members < 2:
        raise ValueError("At least two members are required")
    result = torch.zeros(members, members - 1, dtype=dtype)
    for column in range(members - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        result[:column + 1, column] = 1.0 / denominator
        result[column + 1, column] = -(column + 1) / denominator
    return result


class MeanOperator:
    """A fixed row-normalized sparse adjacency; no implicit self-loops."""

    def __init__(self, adjacency: torch.Tensor):
        if adjacency.layout != torch.sparse_coo:
            raise ValueError("Provide a sparse COO adjacency")
        self.adjacency = adjacency.coalesce()
        self.widths = []

    @classmethod
    def from_edges(cls, edge_index, nodes, dtype=None, device="cpu"):
        edge_index = edge_index.to(device=device, dtype=torch.long)
        source, target = edge_index
        degree = torch.bincount(target, minlength=nodes)
        values = torch.ones(source.numel(), dtype=dtype, device=device)
        values = values / degree[target].clamp_min(1).to(values.dtype)
        adjacency = torch.sparse_coo_tensor(
            torch.stack((target, source)), values, (nodes, nodes),
            device=device, dtype=values.dtype,
        ).coalesce()
        return cls(adjacency)

    def __call__(self, features):
        self.widths.append(features.shape[-1])
        return torch.sparse.mm(self.adjacency, features)

    def reset_counts(self):
        self.widths.clear()


class GroupedLinear(nn.Module):
    """Independent member weights operating only on [N, M, r] tensors."""

    def __init__(self, members, input_dim, output_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(members, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(members, output_dim)) if bias else None
        for weight in self.weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, features):
        output = torch.einsum("nmi,mio->nmo", features, self.weight)
        return output if self.bias is None else output + self.bias


class CommonBlock(nn.Module):
    """Same SAGE-plus-concatenation FFN structure as the source snapshot."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.neighbor = nn.Linear(dim, dim)
        self.root = nn.Linear(dim, dim, bias=False)
        self.first = nn.Linear(2 * dim, dim)
        self.second = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, normalized, propagated):
        message = self.neighbor(propagated) + self.root(normalized)
        hidden = self.dropout(self.first(torch.cat((normalized, message), dim=-1)))
        return self.dropout(self.second(F.gelu(hidden)))


class PrivateBlock(nn.Module):
    """Narrow member-specific SAGE/FFN updates, conditioned on common state."""

    def __init__(self, dim, rank, members, dropout):
        super().__init__()
        self.neighbor = GroupedLinear(members, rank, rank)
        self.root = GroupedLinear(members, rank, rank, bias=False)
        self.condition = nn.Linear(2 * dim, rank, bias=False)
        self.gates = nn.Parameter(torch.empty(members, rank))
        nn.init.normal_(self.gates, mean=1.0, std=0.1)
        self.first = GroupedLinear(members, 2 * rank, rank)
        self.second = GroupedLinear(members, rank, rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, private, propagated_private, common, propagated_common):
        condition = self.condition(torch.cat((common, propagated_common), dim=-1))
        message = (self.neighbor(propagated_private) + self.root(private)
                   + condition[:, None, :] * self.gates)
        hidden = self.dropout(self.first(torch.cat((private, message), dim=-1)))
        return self.dropout(self.second(F.gelu(hidden)))


class CenteredPathEnsemble(nn.Module):
    """Member logits [M,N,C], mean-logit inference, M >= 2 and rank >= 1.

    H [N,D] is the common state. U [N,M-1,r] stores contrast coordinates;
    Z = C U [N,M,r] has zero mean across members. All adjacency applications
    operate on [LN(H), normalized(U)] together, at width D+(M-1)r.
    Member updates are nonlinear at width r, never at width D.

    Conceptually member hidden states are H + Z B with learned B [r,D].
    When V = Z + private_update has mean mu, folding mu B into H and
    storing C^T V preserves the conceptual hidden state exactly. This exact
    centering identity is local to this architecture, not to an arbitrary GNN.
    """

    def __init__(self, input_dim, output_dim, dim=128, rank=16, members=4,
                 layers=2, dropout=0.2, private_edges=True, recycle_mean=True):
        super().__init__()
        if rank < 1 or layers < 1:
            raise ValueError("rank and layers must be positive")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.dim, self.rank, self.members, self.layers = dim, rank, members, layers
        self.private_edges, self.recycle_mean = private_edges, recycle_mean
        self.register_buffer("contrast", helmert_contrasts(members))
        self.input_common = nn.Linear(input_dim, dim)
        # BatchEnsemble-style input factors are evaluated by modulating the
        # small weight tensor, avoiding a materialized [M,N,F] feature tensor.
        self.input_private = nn.Parameter(torch.empty(input_dim, rank))
        nn.init.xavier_uniform_(self.input_private)
        self.input_factors = nn.Parameter(torch.empty(members, input_dim))
        with torch.no_grad():
            self.input_factors.bernoulli_(0.5).mul_(2).sub_(1)
        self.input_bias = nn.Parameter(torch.zeros(members, rank))
        self.expansion = nn.Parameter(torch.empty(rank, dim))
        nn.init.xavier_uniform_(self.expansion)
        self.dropout = nn.Dropout(dropout)
        self.common_norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.common_blocks = nn.ModuleList([CommonBlock(dim, dropout) for _ in range(layers)])
        self.private_blocks = nn.ModuleList([
            PrivateBlock(dim, rank, members, dropout) for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(dim)
        self.common_head = nn.Linear(dim, output_dim)
        # Member-specific narrow heads are essential: a shared linear head
        # on zero-mean Z would cancel exactly under mean-logit pooling.
        self.private_heads = GroupedLinear(members, rank, output_dim)

    def expand(self, coordinates):
        return torch.einsum("nqr,mq->nmr", coordinates, self.contrast)

    def compress(self, member_features):
        return torch.einsum("nmr,mq->nqr", member_features, self.contrast)

    @staticmethod
    def scale(private):
        # A shared per-node scale preserves centering. Separate per-member
        # LayerNorm would generally destroy zero mean and the M-1 reuse.
        return private.square().mean(dim=(1, 2), keepdim=True).add(1e-6).sqrt()

    def initial(self, features):
        common = F.gelu(self.dropout(self.input_common(features)))
        weights = self.input_private[None, :, :] * self.input_factors[:, :, None]
        private = F.gelu(self.dropout(
            torch.einsum("nf,mfr->nmr", features, weights) + self.input_bias
        ))
        if self.recycle_mean:
            common = common + private.mean(dim=1) @ self.expansion
        return common, self.compress(private)

    def forward(self, operator: MeanOperator, features, explicit=False,
                return_trace=False):
        common, coordinates = self.initial(features)
        trace = []
        for norm, common_block, private_block in zip(
                self.common_norm, self.common_blocks, self.private_blocks):
            private = self.expand(coordinates)
            normalized_common = norm(common)
            denominator = self.scale(private)
            normalized_coordinates = coordinates / denominator
            normalized_private = private / denominator
            if explicit:
                propagated_common = operator(normalized_common)
                if self.private_edges:
                    propagated_private = operator(
                        normalized_private.flatten(1)
                    ).reshape_as(private)
                else:
                    propagated_private = torch.zeros_like(private)
            else:
                if self.private_edges:
                    packed = torch.cat((normalized_common,
                                        normalized_coordinates.flatten(1)), dim=1)
                    propagated = operator(packed)
                    propagated_common = propagated[:, :self.dim]
                    propagated_coordinates = propagated[:, self.dim:].reshape_as(coordinates)
                    propagated_private = self.expand(propagated_coordinates)
                else:
                    propagated_common = operator(normalized_common)
                    propagated_private = torch.zeros_like(private)
            common_delta = common_block(normalized_common, propagated_common)
            raw_private = private + private_block(
                normalized_private, propagated_private,
                normalized_common, propagated_common,
            )
            mean_private = raw_private.mean(dim=1)
            updated_common = common + common_delta
            if self.recycle_mean:
                updated_common = updated_common + mean_private @ self.expansion
            updated_coordinates = self.compress(raw_private)
            if return_trace:
                trace.append({
                    "before_common": common, "common_delta": common_delta,
                    "raw_private": raw_private, "after_common": updated_common,
                    "after_private": self.expand(updated_coordinates),
                })
            common, coordinates = updated_common, updated_coordinates
        private = self.expand(coordinates)
        private = private / self.scale(private)
        logits = self.common_head(self.output_norm(common))[:, None, :]
        logits = logits + self.private_heads(private)
        logits = logits.permute(1, 0, 2)
        return (logits, trace) if return_trace else logits

    def mean_logits(self, operator, features):
        return self(operator, features).mean(dim=0)

    def structural_cost(self):
        """Dominant per-node linear MACs; not a wall-clock or byte estimate."""
        d, r, m, f, c, l = (self.dim, self.rank, self.members,
                            self.input_dim, self.output_dim, self.layers)
        return {
            "stored_state_channels": d + (m - 1) * r,
            "adjacency_channels_per_layer": d + (m - 1) * r if self.private_edges else d,
            "largest_private_state_channels": m * r,
            "block_linear_macs_per_node": 5 * d * d + 5 * m * r * r + 2 * d * r
                + (d * r if self.recycle_mean else 0),
            "boundary_linear_macs_per_node": f * d + m * f * r + d * c + m * r * c
                + (d * r if self.recycle_mean else 0),
            "baseline_m_full_width_adjacency_channels": m * d,
            "baseline_m_full_width_block_macs_per_node": 5 * m * d * d,
            "layers": l,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


class IndependentNarrowPaths(nn.Module):
    """Critical prior-art-style control: common trunk plus narrow graph branches.

    Each branch has nonlinear relational updates at every layer, conditioned
    on the common state. It stores M uncentered r-wide states and applies one
    packed adjacency at D+M r channels. There is no recycling into the trunk.
    This control prevents attributing an ordinary small-branch ensemble's
    benefit to the centering representation or to bidirectional coupling.
    """

    def __init__(self, input_dim, output_dim, dim=128, rank=16, members=4,
                 layers=2, dropout=0.2):
        super().__init__()
        anchor = CenteredPathEnsemble(input_dim, output_dim, dim, rank,
                                      members, layers, dropout)
        self.dim, self.rank, self.members, self.layers = dim, rank, members, layers
        for name in ("input_common", "input_private", "input_factors", "input_bias",
                     "dropout", "common_norm", "common_blocks", "private_blocks",
                     "output_norm", "common_head", "private_heads"):
            setattr(self, name, getattr(anchor, name))

    def forward(self, operator, features):
        common = F.gelu(self.dropout(self.input_common(features)))
        weights = self.input_private[None, :, :] * self.input_factors[:, :, None]
        private = F.gelu(self.dropout(
            torch.einsum("nf,mfr->nmr", features, weights) + self.input_bias
        ))
        for norm, common_block, private_block in zip(
                self.common_norm, self.common_blocks, self.private_blocks):
            normalized_common = norm(common)
            normalized_private = private / CenteredPathEnsemble.scale(private)
            packed = torch.cat((normalized_common, normalized_private.flatten(1)), dim=1)
            propagated = operator(packed)
            propagated_common = propagated[:, :self.dim]
            propagated_private = propagated[:, self.dim:].reshape_as(private)
            common = common + common_block(normalized_common, propagated_common)
            private = private + private_block(normalized_private, propagated_private,
                                              normalized_common, propagated_common)
        private = private / CenteredPathEnsemble.scale(private)
        logits = self.common_head(self.output_norm(common))[:, None, :]
        return (logits + self.private_heads(private)).permute(1, 0, 2)


class RepeatedFullWidthEnsemble(nn.Module):
    """Compute-optimized pure-Torch reference for source SAGE GNNM semantics.

    Boundary BatchEnsemble factors, shared full-width blocks, separate member
    hidden states, mean member loss. It uses sparse mean rather than PyG;
    dataset experiments must include the unmodified frozen source as well.
    """

    def __init__(self, input_dim, output_dim, dim=128, members=4, layers=2,
                 dropout=0.2):
        super().__init__()
        self.dim, self.members, self.layers = dim, members, layers
        self.input_weight = nn.Parameter(torch.empty(input_dim, dim))
        self.output_weight = nn.Parameter(torch.empty(dim, output_dim))
        nn.init.xavier_uniform_(self.input_weight)
        nn.init.xavier_uniform_(self.output_weight)
        self.input_r = nn.Parameter(torch.empty(members, input_dim))
        with torch.no_grad():
            self.input_r.bernoulli_(0.5).mul_(2).sub_(1)
        self.input_s = nn.Parameter(torch.ones(members, dim))
        self.input_b = nn.Parameter(torch.zeros(members, dim))
        self.output_r = nn.Parameter(torch.ones(members, dim))
        self.output_s = nn.Parameter(torch.ones(members, output_dim))
        self.output_b = nn.Parameter(torch.zeros(members, output_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.blocks = nn.ModuleList([CommonBlock(dim, dropout) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)

    def forward_member(self, operator, features, member):
        weight = self.input_weight * self.input_r[member, :, None]
        hidden = features @ weight
        hidden = F.gelu(self.dropout(hidden * self.input_s[member] + self.input_b[member]))
        for norm, block in zip(self.norm, self.blocks):
            normalized = norm(hidden)
            hidden = hidden + block(normalized, operator(normalized))
        output_weight = self.output_weight * self.output_r[member, :, None]
        return ((self.output_norm(hidden) @ output_weight)
                * self.output_s[member] + self.output_b[member])

    def forward(self, operator, features):
        return torch.stack([self.forward_member(operator, features, member)
                            for member in range(self.members)], dim=0)


class GroupedLayerNorm(nn.Module):
    """Independent LayerNorm parameters, with the same per-member semantics."""

    def __init__(self, members, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(members, dim))
        self.bias = nn.Parameter(torch.zeros(members, dim))
        self.dim = dim

    def forward(self, features):
        return F.layer_norm(features, (self.dim,)) * self.weight + self.bias


class PackedSharedSAGEEnsemble(RepeatedFullWidthEnsemble):
    """Exact SAGE TABM semantics in one channel-packed aggregation per layer.

    This is an efficient execution comparator. It still retains M full-width
    member states and performs M full-width dense nonlinear computations.
    """

    def forward(self, operator, features, explicit=False):
        if explicit:
            return super().forward(operator, features)
        weights = self.input_weight[None, :, :] * self.input_r[:, :, None]
        hidden = torch.einsum("nf,mfd->nmd", features, weights)
        hidden = F.gelu(self.dropout(hidden * self.input_s + self.input_b))
        for norm, block in zip(self.norm, self.blocks):
            normalized = norm(hidden)
            propagated = operator(normalized.flatten(1)).reshape_as(normalized)
            hidden = hidden + block(normalized, propagated)
        output_weights = self.output_weight[None, :, :] * self.output_r[:, :, None]
        result = torch.einsum("nmd,mdc->nmc", self.output_norm(hidden), output_weights)
        return (result * self.output_s + self.output_b).permute(1, 0, 2)

    @torch.no_grad()
    def copy_from_source_tabm(self, model):
        if model.model_name != "SAGE" or len(model.residual_modules) != self.layers:
            raise ValueError("Source must be matched TABMModel/SAGE")
        for source, r, s, b, weight in (
                (model.input_be_block, self.input_r, self.input_s, self.input_b, self.input_weight),
                (model.output_be_block, self.output_r, self.output_s, self.output_b, self.output_weight)):
            r.copy_(source.R)
            s.copy_(source.S)
            b.copy_(source.B)
            weight.copy_(source.W.weight.T)
        for layer, residual in enumerate(model.residual_modules):
            if not isinstance(residual.normalization, nn.LayerNorm) or residual.normalization.eps != 1e-5:
                raise ValueError("Source must use the default LayerNorm")
            self.norm[layer].load_state_dict(residual.normalization.state_dict())
            self.blocks[layer].neighbor.load_state_dict(residual.module.conv.lin_l.state_dict())
            self.blocks[layer].root.load_state_dict(residual.module.conv.lin_r.state_dict())
            ff = residual.module.feed_forward_module
            self.blocks[layer].first.load_state_dict(ff.linear_1.state_dict())
            self.blocks[layer].second.load_state_dict(ff.linear_2.state_dict())
        self.output_norm.load_state_dict(model.output_normalization.state_dict())


class PlainSAGE(nn.Module):
    """Single ordinary source-structure SAGE model, without ensemble factors."""

    def __init__(self, input_dim, output_dim, dim=128, layers=2, dropout=0.2):
        super().__init__()
        self.input = nn.Linear(input_dim, dim)
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.blocks = nn.ModuleList([CommonBlock(dim, dropout) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, operator, features):
        hidden = F.gelu(self.dropout(self.input(features)))
        for norm, block in zip(self.norm, self.blocks):
            normalized = norm(hidden)
            hidden = hidden + block(normalized, operator(normalized))
        return self.output(self.output_norm(hidden))[None, :, :]


def source_sage_parameter_count(input_dim, output_dim, dim, layers=2, members=1):
    """Exact ordinary Model/SAGE/LayerNorm/multiplier1 count; no score access."""
    return members * (5 * layers * dim * dim + dim * (input_dim + output_dim + 3 + 5 * layers)
                      + output_dim)


def nearest_parameter_width(target_parameters, input_dim, output_dim, layers=2,
                            members=1, maximum=512, step=4):
    choices = range(8, maximum + 1, step)
    return min(choices, key=lambda width: (
        abs(source_sage_parameter_count(input_dim, output_dim, width, layers, members)
            - target_parameters), width,
    ))


class PackedIndependentSAGEEnsemble(nn.Module):
    """Independent SAGE models with channel-packed linear neighbor means.

    This graph adaptation of Packed-Ensembles is a required comparator, not
    new technical content. The default per-member width44 gives 176 graph
    channels, matching D128+(M-1)*r16. Width128 packs four full-size models.
    """

    def __init__(self, input_dim, output_dim, dim=44, members=4, layers=2,
                 dropout=0.2):
        super().__init__()
        self.dim, self.members, self.layers = dim, members, layers
        self.input = GroupedLinear(members, input_dim, dim)
        self.norm = nn.ModuleList([GroupedLayerNorm(members, dim) for _ in range(layers)])
        self.neighbor = nn.ModuleList([GroupedLinear(members, dim, dim) for _ in range(layers)])
        self.root = nn.ModuleList([GroupedLinear(members, dim, dim, bias=False) for _ in range(layers)])
        self.first = nn.ModuleList([GroupedLinear(members, 2 * dim, dim) for _ in range(layers)])
        self.second = nn.ModuleList([GroupedLinear(members, dim, dim) for _ in range(layers)])
        self.output_norm = GroupedLayerNorm(members, dim)
        self.output = GroupedLinear(members, dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, operator, features, explicit=False):
        hidden = F.gelu(self.dropout(self.input(features[:, None, :].expand(-1, self.members, -1))))
        for norm, neighbor, root, first, second in zip(
                self.norm, self.neighbor, self.root, self.first, self.second):
            normalized = norm(hidden)
            if explicit:
                propagated = torch.stack([operator(normalized[:, member, :])
                                          for member in range(self.members)], dim=1)
            else:
                propagated = operator(normalized.flatten(1)).reshape_as(normalized)
            message = neighbor(propagated) + root(normalized)
            intermediate = self.dropout(first(torch.cat((normalized, message), dim=-1)))
            hidden = hidden + self.dropout(second(F.gelu(intermediate)))
        return self.output(self.output_norm(hidden)).permute(1, 0, 2)

    @torch.no_grad()
    def copy_from_source_members(self, models):
        """Transfer frozen source Model/SAGE/LayerNorm weights, eval equivalence.

        Requires source Model(... model_name='SAGE', hidden_dim_multiplier=1,
        normalization='LayerNorm'), with identical dimension/depth and exactly
        self.members models. Keep graph preprocessing identical. This method
        does not select checkpoints or access any dataset.
        """
        if len(models) != self.members:
            raise ValueError("Member count mismatch")
        def linear(destination, source, member):
            destination.weight[member].copy_(source.weight.T)
            if destination.bias is not None:
                destination.bias[member].copy_(source.bias)
        def normalization(destination, source, member):
            if not isinstance(source, nn.LayerNorm) or source.eps != 1e-5:
                raise ValueError("Source must use the default LayerNorm")
            destination.weight[member].copy_(source.weight)
            destination.bias[member].copy_(source.bias)
        for member, model in enumerate(models):
            if model.model_name != "SAGE" or len(model.residual_modules) != self.layers:
                raise ValueError("Source must be the matched SAGE Model")
            linear(self.input, model.input_linear, member)
            for layer, residual in enumerate(model.residual_modules):
                normalization(self.norm[layer], residual.normalization, member)
                linear(self.neighbor[layer], residual.module.conv.lin_l, member)
                linear(self.root[layer], residual.module.conv.lin_r, member)
                linear(self.first[layer], residual.module.feed_forward_module.linear_1, member)
                linear(self.second[layer], residual.module.feed_forward_module.linear_2, member)
            normalization(self.output_norm, model.output_normalization, member)
            linear(self.output, model.output_linear, member)


def graph_mean_logits(member_node_logits, batch_index, num_graphs: Optional[int] = None):
    """[M,N,C] -> [M,G,C] by mean of normalized node readouts per graph."""
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
    output = member_node_logits.new_zeros(member_node_logits.shape[0], num_graphs,
                                        member_node_logits.shape[-1])
    output.index_add_(1, batch_index, member_node_logits)
    count = torch.bincount(batch_index, minlength=num_graphs).clamp_min(1)
    return output / count[None, :, None]


def mean_member_loss(logits, labels, train_mask):
    """Identical mean-member objective for candidate and all ensemble controls."""
    selected = logits[:, train_mask, :]
    if labels.ndim == 2:
        target = labels[train_mask].to(selected.dtype)
        if target.shape != selected.shape[1:]:
            raise ValueError("Multi-label target width mismatch")
        return F.binary_cross_entropy_with_logits(
            selected, target[None, :, :].expand_as(selected)
        )
    if selected.shape[-1] == 1:
        target = labels[train_mask].to(selected.dtype)
        return F.binary_cross_entropy_with_logits(
            selected[..., 0], target[None, :].expand(selected.shape[0], -1)
        )
    target = labels[train_mask].long()
    return F.cross_entropy(
        selected.reshape(-1, selected.shape[-1]),
        target[None, :].expand(selected.shape[0], -1).reshape(-1),
    )
