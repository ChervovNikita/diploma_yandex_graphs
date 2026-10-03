"""One exact native N64 scorer/decoder plus the frozen all-count head.

SOURCE ONLY. Native source is supplied only by cardinality_dependencies.
There is no teacher stored in or passed to this model. The changed final
route is 1.05*detached sampled bits, with four nonlinear native decodes.
"""
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from torch import nn
from cardinality_graph import enumerate_neighbors, feature_sum
from cardinality_counter import separated_seed
from cardinality_density import PredictedLaw, center_by_query, segmented_log_softmax, completion_draws, require


@dataclass(frozen=True)
class OrderedSupport:
    neighbors: object
    rows: torch.Tensor
    nodes: torch.Tensor
    counterpart_pairs: torch.Tensor
    permutation: torch.Tensor
    offsets: torch.Tensor
    left_count: torch.Tensor
    right_count: torch.Tensor


def ordered_support(graph, queries):
    neighbors = enumerate_neighbors(graph, queries)
    lq, ln = neighbors.left
    rq, rn = neighbors.right
    rows = torch.cat((lq, rq))
    nodes = torch.cat((ln, rn))
    pairs = torch.cat((torch.stack((queries[lq, 1], ln), 1), torch.stack((queries[rq, 0], rn), 1)))
    pairs = torch.sort(pairs, dim=1).values
    # Stable lexicographic row, canonical counterpart-low, counterpart-high.
    # No node-count multiplication overflow or teacher/synthetic indicator.
    order = torch.arange(len(rows), device=rows.device)
    for coordinate in (pairs[:, 1], pairs[:, 0], rows):
        order = order[torch.argsort(coordinate[order], stable=True)]
    rows, nodes, pairs = rows[order], nodes[order], pairs[order]
    count = torch.bincount(rows, minlength=len(queries))
    offsets = torch.cat((count.new_zeros(1), count.cumsum(0)))
    if len(rows) > 1:
        same_query = rows[1:] == rows[:-1]
        repeated_pair = (pairs[1:] == pairs[:-1]).all(1)
        require(not bool((same_query & repeated_pair).any()), "Repeated residual counterpart identity")
    return OrderedSupport(neighbors, rows, nodes, pairs, order, offsets,
                          torch.bincount(lq, minlength=len(queries)), torch.bincount(rq, minlength=len(queries)))


class CountHead(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear defaults consume RNG. fork_rng captures/restores the CPU
        # native initializer state; head draws use a separate CPU generator.
        with torch.random.fork_rng(devices=[]):
            self.net = nn.Sequential(nn.Linear(270, 16, device="cpu", dtype=torch.float32), nn.ReLU(),
                                     nn.Linear(16, 16, device="cpu", dtype=torch.float32), nn.ReLU(),
                                     nn.Linear(16, 1, device="cpu", dtype=torch.float32))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(separated_seed("count-init"))
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, generator=generator)
                    nn.init.zeros_(module.bias)
        require(sum(p.numel() for p in self.parameters()) == 4625, "Frozen head tensor count differs")

    def forward(self, usable_context, residual_counts):
        require(usable_context.ndim == 2 and usable_context.shape[1] == 264 and usable_context.dtype == torch.float32,
                "Masked usable context must have264 FP32 entries")
        require(residual_counts.shape == (len(usable_context),) and residual_counts.dtype == torch.long, "Residual size contract differs")
        q = len(usable_context)
        class_counts = residual_counts + 1
        class_rows = torch.repeat_interleave(torch.arange(q, device=residual_counts.device), class_counts)
        class_offsets = torch.cat((class_counts.new_zeros(1), class_counts.cumsum(0)))
        k = torch.arange(int(class_offsets[-1]), device=residual_counts.device) - class_offsets[:-1][class_rows]
        r = residual_counts[class_rows]
        nonempty = r > 0
        logits = usable_context.new_zeros(len(class_rows))
        if bool(nonempty.any()):
            kf, rf = k[nonempty].to(torch.float32), r[nonempty].to(torch.float32)
            ratio = kf / rf
            features = torch.stack((ratio, ratio.square(), torch.log1p(kf), torch.log1p(rf - kf),
                                    (kf == 0).to(torch.float32), (kf == rf).to(torch.float32)), 1)
            inputs = torch.cat((usable_context[class_rows[nonempty]], features), 1)
            logits[nonempty] = self.net(inputs).flatten()
        # All R+1 classes are evaluated. R0 has a constant a0=logpi0=0 and
        # never passes its empty context to the head, as frozen in the plan.
        log_pi = segmented_log_softmax(logits.to(torch.float64), class_rows, q)
        return logits, log_pi, class_offsets


def usable_context(h, h_outer, graph, queries, support, affinity):
    """264 masked-only entries; no target label, split flag or teacher K."""
    q = len(queries)
    r = support.offsets[1:] - support.offsets[:-1]
    common_count = torch.bincount(support.neighbors.common[0], minlength=q)
    common_mean = feature_sum(h_outer, support.neighbors.common, q) / common_count.clamp_min(1).to(h.dtype)[:, None]
    residual_mean = feature_sum(h_outer, (support.rows, support.nodes), q) / r.clamp_min(1).to(h.dtype)[:, None]
    endpoint_i, endpoint_j = h[queries[:, 0]], h[queries[:, 1]]
    degree_i = (graph.rowptr[queries[:, 0] + 1] - graph.rowptr[queries[:, 0]]).to(h.dtype)
    degree_j = (graph.rowptr[queries[:, 1] + 1] - graph.rowptr[queries[:, 1]]).to(h.dtype)
    log_i, log_j = torch.log1p(degree_i), torch.log1p(degree_j)
    affinity_mean = affinity.new_zeros(q).index_add(0, support.rows, affinity) / r.clamp_min(1).to(affinity.dtype)
    affinity_square_mean = affinity.new_zeros(q).index_add(0, support.rows, affinity.square()) / r.clamp_min(1).to(affinity.dtype)
    affinity_max = affinity.new_full((q,), float("-inf"))
    affinity_max = affinity_max.scatter_reduce(0, support.rows, affinity, reduce="amax", include_self=True)
    affinity_max = torch.where(r > 0, affinity_max, torch.zeros_like(affinity_max))
    # scatter_reduce(amax) uses PyTorch's equally shared max-tie subgradient;
    # the prospective exact-runtime fixture must qualify this declared rule.
    scalar = torch.stack((log_i + log_j, (log_i - log_j).abs(), torch.log1p(common_count.to(h.dtype)),
                          torch.log1p(r.to(h.dtype)), (support.left_count - support.right_count).abs().to(h.dtype) / r.clamp_min(1).to(h.dtype),
                          affinity_mean.to(h.dtype), affinity_square_mean.to(h.dtype), affinity_max.to(h.dtype)), 1)
    context = torch.cat((endpoint_i + endpoint_j, endpoint_i * endpoint_j, common_mean, residual_mean, scalar), 1)
    require(context.shape == (q, 264) and bool(torch.isfinite(context).all()), "Nonfinite/incorrect usable context")
    return context


def native_adjacency(graph):
    from torch_sparse import SparseTensor
    # Graph row/col were coalesced AFTER record masking. The encoder owns its
    # normal .25 directed adjacency dropout; support uses this undropped graph.
    return SparseTensor(row=graph.row, col=graph.col, sparse_sizes=(graph.nodes, graph.nodes), is_sorted=True)


class CardinalitySingle(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        device = next(decoder.parameters()).device
        self.count_head = CountHead().to(device)
        require(self.decoder.depth == 1 and self.decoder.splitsize == -1 and not self.decoder.learnablept,
                "Only the frozen native depth1/full-support decoder is admitted")
        require(self.decoder.cndeg == self.decoder.trainresdeg == self.decoder.testresdeg == -1, "Native support sampling is forbidden")
        require(parameter_audit(self)["total"] == 42772 and parameter_audit(self)["active"] == 38547,
                "Actual model parameter count differs from frozen C64")

    def encode(self, x, graph):
        require(x.dtype == torch.float32 and x.shape == (graph.nodes, 128), "Complete FP32 node features required")
        return self.encoder(x, native_adjacency(graph))

    def native_decode(self, endpoint_product, common_features):
        # Source-native xijlin before xcnlin preserves draw dropout ordering.
        endpoint = self.decoder.xijlin(endpoint_product)
        common = self.decoder.xcnlin(common_features)
        return self.decoder.lin(common * self.decoder.beta + endpoint).flatten()

    def predict_law(self, h, graph, queries):
        require(h.dtype == torch.float32 and h.shape == (graph.nodes, 64), "Full-node native N64 representation required")
        support = ordered_support(graph, queries)
        lq, ln = support.neighbors.left
        rq, rn = support.neighbors.right
        endpoint_product = h[queries[:, 0]] * h[queries[:, 1]]
        h_outer = h + self.decoder.xlin(h)
        adjacency = native_adjacency(graph)
        left_queries = torch.stack((queries[lq, 1], ln), 1)
        right_queries = torch.stack((queries[rq, 0], rn), 1)
        # Both complete source-native paths execute even when their query set
        # is empty. depth0 excludes native recursive no_grad while retaining
        # its full second xlin, overlap, decode and native training dropout.
        score_left = self.decoder(h_outer, adjacency, left_queries.T, depth=0).flatten()
        score_right = self.decoder(h_outer, adjacency, right_queries.T, depth=0).flatten()
        score = torch.cat((score_left, score_right))[support.permutation]
        # Native score->affinity algebra, alpha deliberately outside q.
        # Network maps FP32; cast outputs for stable FP64 density arithmetic.
        t = 2.5 * (score.to(torch.float64) - 6.) + math.log(.1)
        affinity = torch.sigmoid(t)
        context = usable_context(h, h_outer, graph, queries, support, affinity)
        count_logits, log_pi, class_offsets = self.count_head(context, support.offsets[1:] - support.offsets[:-1])
        centered = center_by_query(t, support.rows, support.offsets)
        law = PredictedLaw(t, centered, affinity, log_pi, count_logits, support.rows, support.offsets, class_offsets)
        detail = {"support": support, "law": law, "context": context, "h_outer": h_outer,
                  "endpoint_product": endpoint_product, "score_left": score_left, "score_right": score_right,
                  "queries": queries, "native_recursive_scorer_auxiliary_autograd": True,
                  "native_full_node_xlin_calls": 3, "native_depth_zero_calls": 2}
        return detail

    def decode_route(self, detail, keys, *, route):
        require(route in ("D4", "M4"), "Unknown frozen route")
        if route == "M4":
            require(not self.training and keys.mode == "eval", "M4 is fixed-bank serving only")
        require(keys.mode == ("train" if self.training else "eval"), "Dropout/key mode mismatch")
        support = detail["support"]
        draw = completion_draws(detail["law"], detail["queries"], support.counterpart_pairs, keys, route=route)
        q = len(detail["queries"])
        common = feature_sum(detail["h_outer"], support.neighbors.common, q)
        output = []
        for d in range(4):
            weight = 1.05 * draw["bits"][d].to(detail["h_outer"].dtype)
            residual = feature_sum(detail["h_outer"], (support.rows, support.nodes), q, weight.detach())
            output.append(self.native_decode(detail["endpoint_product"], common + residual))
        logits = torch.stack(output, dim=1)
        require(logits.shape == (q, 4) and logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()), "Incomplete/nonfinite four-draw target forward")
        return logits, draw

    def query_forward(self, h, graph, queries, keys, *, route="D4"):
        detail = self.predict_law(h, graph, queries)
        logits, draw = self.decode_route(detail, keys, route=route)
        return logits, detail, draw

    @staticmethod
    def serve(logits):
        require(logits.ndim == 2 and logits.shape[1] == 4, "Frozen four raw logits required")
        return logits.mean(1)


def parameter_audit(model):
    rows = [{"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
             "active": not name.startswith("decoder.ptlin.")}
            for name, parameter in model.named_parameters()]
    return {"total": sum(row["numel"] for row in rows), "active": sum(row["numel"] for row in rows if row["active"]),
            "head": sum(row["numel"] for row in rows if row["name"].startswith("count_head.")),
            "unused_ptlin": sum(row["numel"] for row in rows if not row["active"]), "tensors": rows}


def make_single(native, device):
    """Future caller only: same seed0 constructor order as pinned make_native."""
    require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled(), "Plain native FP32 profile required")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    encoder = native.GCN(128, 64, 64, 1, .1, True, True, -1, "gcn", False, .25, xdropout=.25, taildropout=.05).to(device)
    decoder = native.IncompleteCN1Predictor(64, 64, 1, 3, .3, edrop=0., ln=True, cndeg=-1,
        use_xlin=True, tailact=True, twolayerlin=False, beta=1., alpha=1.05, scale=2.5, offset=6.,
        trainresdeg=-1, testresdeg=-1, pt=.1, learnablept=False, depth=1, splitsize=-1).to(device)
    model = CardinalitySingle(encoder, decoder)
    # Exactly two native Adam groups; count parameters join the decoder group.
    optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": .0082},
                                  {"params": [*model.decoder.parameters(), *model.count_head.parameters()], "lr": .0037}],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)
    return model, optimizer
