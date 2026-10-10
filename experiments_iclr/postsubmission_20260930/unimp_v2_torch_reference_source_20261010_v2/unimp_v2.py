# Copyright 2019 PaddlePaddle Authors. Licensed under Apache-2.0; see LICENSE.
"""Torch adaptation of PGL UniMPv2 at commit 6dbb47c4559352ea1b1e327ee0039c47095583af.

Derived from the Apache-2.0 PGL author files bound in SOURCE_BINDINGS.json.
Original pinned files/license remain unmodified in the predecessor packet.
Adaptation: current Torch operations; float .1 finite pre-softmax edge masking
chosen prospectively. Original int64 Paddle feed acceptance/coercion unresolved.
Importing this file imports stdlib only. The caller supplies an admitted Torch
provider. No numerical qualification or legacy-runtime equivalence is claimed.
"""
from dataclasses import dataclass
import math


ATTENTION_MASK_POLICY = "float_0p1_pre_softmax_explicit_adaptation"


@dataclass(frozen=True)
class Topology:
    x: object
    original_nodes: int
    edge_index: object
    receiver_permutation: object
    source: object
    destination: object
    indegree: object
    normalized_adjacency: object


def author_topology(torch, x, edge_index):
    """Exact declared vnode -> undirected -> unique self-loop construction.

    Canonical edges are lexicographic (source, destination), like np.unique.
    A single explicit permutation groups messages by destination/source. All
    attention, value aggregation and subsequent attention APPNP use that order.
    No labels are accepted. Virtual-node means use ORIGINAL raw feature rows.
    """
    if x.ndim != 2 or x.dtype != torch.float32:
        raise ValueError("Factual X must be a float32 matrix")
    n = x.shape[0]
    if n < 1 or edge_index.dtype != torch.int64 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected int64 [2,E] original edge_index")
    if edge_index.device != x.device:
        raise ValueError("Edges/X must have compatible shape and device")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n):
        raise ValueError("Edge endpoints must be original factual node rows")
    total = n + 3
    # Author add_vnode includes virtual sources, hence a complete vnode clique.
    source = torch.arange(total, device=x.device).repeat(3)
    destination = torch.arange(n, total, device=x.device).repeat_interleave(total)
    added = torch.stack((source, destination))
    edges = torch.cat((edge_index, added), dim=1)
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    ids = torch.arange(total, device=x.device)
    edges = torch.cat((edges, torch.stack((ids, ids))), dim=1)
    keys = torch.unique(edges[0] * total + edges[1], sorted=True)
    canonical = torch.stack((keys // total, keys % total))
    permutation = torch.argsort(canonical[1] * total + canonical[0])
    ordered = canonical[:, permutation]
    source, destination = ordered[0], ordered[1]
    degree = torch.bincount(destination, minlength=total).to(torch.float32)
    norm = degree.clamp_min(1).pow(-0.5)
    adjacency = torch.sparse_coo_tensor(
        torch.stack((destination, source)), norm[destination] * norm[source],
        (total, total), device=x.device, dtype=torch.float32,
    ).coalesce()
    augmented_x = torch.cat((x, x.mean(0, keepdim=True).expand(3, -1)), dim=0)
    return Topology(augmented_x, n, canonical, permutation, source,
                    destination, degree, adjacency)


def build_model(torch, feature_width, classes, *, attention_mask_policy):
    """Construct one source-prescribed label-aware single, with explicit repair.

    The sole supported mask is a float .1 pre-softmax score mask. The author
    declares an int64 feed while supplying .1; its legacy acceptance/coercion
    is unresolved. This policy MUST NOT be called identical to that runtime.
    """
    if attention_mask_policy != ATTENTION_MASK_POLICY:
        raise ValueError("Explicit float-mask adaptation acknowledgement required")
    if feature_width < 1 or classes < 2:
        raise ValueError("Invalid feature/class dimensions")
    nn = torch.nn
    functional = torch.nn.functional

    class Dense(nn.Module):
        def __init__(self, incoming, outgoing, kind):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(outgoing, incoming, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(outgoing, dtype=torch.float32))
            if kind == "gcn":
                nn.init.xavier_uniform_(self.weight)
                nn.init.zeros_(self.bias)
            else:
                bound = 1 / math.sqrt(incoming)
                nn.init.uniform_(self.weight, -bound, bound)
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, value):
            return functional.linear(value, self.weight, self.bias)

    def aggregate(topology, value, weight=None):
        message = value[topology.source]
        if weight is not None:
            message = message * weight.reshape(weight.shape + (1,) * (message.ndim - weight.ndim))
        out = value.new_zeros((topology.x.shape[0],) + value.shape[1:])
        return out.index_add(0, topology.destination, message)

    def incoming_softmax(topology, score):
        index = topology.destination[:, None].expand_as(score)
        maxima = score.new_full((topology.x.shape[0], score.shape[1]), -float("inf"))
        # Detaching the stabilizing max preserves the softmax derivative.
        maxima.scatter_reduce_(0, index, score.detach(), reduce="amax", include_self=True)
        numerator = (score - maxima[topology.destination]).exp()
        denominator = numerator.new_zeros(maxima.shape).index_add(0, topology.destination, numerator)
        return numerator / denominator[topology.destination]

    class Transformer(nn.Module):
        def __init__(self, incoming, width, heads, concat, hidden_mask):
            super().__init__()
            self.width, self.heads, self.concat, self.hidden_mask = width, heads, concat, hidden_mask
            outgoing = width * heads if concat else width
            self.q = Dense(incoming, width * heads, "gcn")
            self.k = Dense(incoming, width * heads, "gcn")
            self.v = Dense(incoming, width * heads, "gcn")
            # L.fc([p,h0,p-h0],1) equals one concatenated 3D -> 1 affine.
            # The author initializes each input block with the 3D fan-in bound.
            self.propagation_gates = nn.ModuleList(
                [Dense(3 * outgoing, 1, "lin") for _ in range(3)] if concat else []
            )
            self.skip = Dense(incoming, outgoing, "lin")
            self.residual_gate = Dense(3 * outgoing, 1, "lin")

        def forward(self, topology, feature):
            n = feature.shape[0]
            q = self.q(feature).reshape(n, self.heads, self.width)
            k = self.k(feature).reshape(n, self.heads, self.width)
            v = self.v(feature).reshape(n, self.heads, self.width)
            score = (q[topology.destination] * k[topology.source]).sum(-1) / math.sqrt(self.width)
            if self.hidden_mask and self.training:
                # One edge draw shared across heads; finite -10000 BEFORE softmax.
                keep = torch.rand(score.shape[0], device=score.device) > 0.1
                score = score + (keep.to(score.dtype) - 1).unsqueeze(-1) * 10000
            attention = incoming_softmax(topology, score)
            attended = aggregate(topology, v, attention)
            uniform = aggregate(topology, v) / topology.indegree[:, None, None]
            out = 0.8 * attended + 0.2 * uniform
            out = out.reshape(n, -1) if self.concat else out.mean(1)
            if self.concat:
                h0 = out
                for gate in self.propagation_gates:
                    propagated = torch.sparse.mm(topology.normalized_adjacency, out)
                    alpha = gate(torch.cat((propagated, h0, propagated - h0), -1)).sigmoid()
                    out = propagated * (1 - alpha) + h0 * alpha
            skip = self.skip(feature)
            alpha = self.residual_gate(torch.cat((skip, out, out - skip), -1)).sigmoid()
            out = skip * alpha + out * (1 - alpha)
            if self.concat:
                out = functional.layer_norm(out, (out.shape[-1],), eps=1e-5)
                out = functional.relu(out)
            return out, attention

    class UniMPv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(classes, feature_width, dtype=torch.float32)
            # Embedding default is Normal(0,1), matching the bound author source.
            self.layers = nn.ModuleList([
                Transformer(feature_width, 100, 3, True, True),
                Transformer(300, 100, 3, True, True),
                Transformer(300, classes, 4, False, False),
            ])

        def forward(self, topology, context_ids, context_labels):
            if context_ids.dtype != torch.int64 or context_labels.dtype != torch.int64:
                raise ValueError("Only int64 role-restricted context IDs/labels accepted")
            if context_ids.ndim != 1 or context_labels.shape != context_ids.shape:
                raise ValueError("Context values must align one-to-one with context IDs")
            if context_ids.device != topology.x.device or context_labels.device != topology.x.device:
                raise ValueError("Context/topology device mismatch")
            if context_ids.numel():
                if int(context_ids.min()) < 0 or int(context_ids.max()) >= topology.original_nodes:
                    raise ValueError("Virtual/nonfactual context identity")
                if context_ids.unique().numel() != context_ids.numel():
                    raise ValueError("Context IDs must be unique")
                if int(context_labels.min()) < 0 or int(context_labels.max()) >= classes:
                    raise ValueError("Context label outside declared TRAIN schema")
            feature = functional.layer_norm(topology.x, (feature_width,), eps=1e-5)
            embed = self.embedding(context_labels)
            embed = functional.relu(functional.layer_norm(embed, (feature_width,), eps=1e-5))
            feature = feature.index_add(0, context_ids, embed)
            feature = functional.dropout(feature, p=0.1, training=self.training)
            for layer in self.layers[:-1]:
                feature, _ = layer(topology, feature)
                feature = functional.dropout(feature, p=0.3, training=self.training)
            feature, attention = self.layers[-1](topology, feature)
            # Final P uses mean PURE attention, not the .8/.2 value mixture.
            transition = attention.mean(1)
            h0 = feature
            for _ in range(10):
                feature = 0.8 * aggregate(topology, feature, transition) + 0.2 * h0
            return feature[:topology.original_nodes]

    return UniMPv2()
