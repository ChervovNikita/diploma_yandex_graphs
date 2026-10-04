"""C_mu: exact native N64, full-context two-sided count potential, one decode.

No teacher is stored or accepted here. Exact detached unary completion feeds
live outer native features. This computes D(E[Z]), not E[D(Z)].
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
from torch_sparse import SparseTensor
from graph_ops import enumerate_neighbors, feature_sum
from pilot_model import make_native
from pooled_model import MODEL_SEED, domain_seed
from count_density import require, unary_marginals, query_nll


def adjacency(graph):
    return SparseTensor(row=graph.row, col=graph.col, sparse_sizes=(graph.nodes, graph.nodes), is_sorted=True)


def summed(values, rows, queries):
    return values.new_zeros((queries, values.shape[1])).index_add(0, rows, values)


class CountPotential(nn.Module):
    def __init__(self):
        super().__init__()
        # Restore ambient CPU initialization RNG; extra head uses its own seed.
        with torch.random.fork_rng(devices=[]):
            self.slot = nn.Sequential(nn.Linear(195, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.count = nn.Sequential(nn.Linear(530, 64), nn.ReLU(), nn.Linear(64, 1))
            gen = torch.Generator(device="cpu").manual_seed(domain_seed("count-head-init"))
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, generator=gen)
                    nn.init.zeros_(module.bias)
        require(sum(p.numel() for p in self.parameters()) == 50753, "Specified count head dimensions differ")

    def potential(self, context, swapped_context, nl, nr):
        # Every hypothetical class; never teacher counts. One hidden width64.
        k, l = torch.meshgrid(torch.arange(nl + 1, device=context.device, dtype=context.dtype),
                              torch.arange(nr + 1, device=context.device, dtype=context.dtype), indexing="ij")
        def features(a, b, na, nb):
            return torch.stack((a / max(na, 1), b / max(nb, 1),
                                torch.log1p(a), torch.log1p(b),
                                torch.log1p(na - a), torch.log1p(nb - b),
                                (a == 0).to(a.dtype), (b == 0).to(a.dtype),
                                (a == na).to(a.dtype), (b == nb).to(a.dtype)), -1)
        f, fs = features(k, l, nl, nr), features(l, k, nr, nl)
        c = context.expand(nl + 1, nr + 1, -1)
        cs = swapped_context.expand_as(c)
        # Exact endpoint/side exchange symmetry; both full head evaluations paid.
        return (.5 * (self.count(torch.cat((c, f), -1)).squeeze(-1)
                      + self.count(torch.cat((cs, fs), -1)).squeeze(-1))).to(torch.float64)


class CountSingle(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder, self.count_head = encoder, decoder, CountPotential()

    def encode(self, x, graph):
        require(x.dtype == torch.float32 and x.shape == (graph.nodes, 128), "Complete admitted raw feature table required")
        return self.encoder(x, adjacency(graph))

    def native_decode(self, endpoint, common):
        # Preserve source-native xijlin then xcnlin dropout ordering.
        endpoint_features = self.decoder.xijlin(endpoint)
        common_features = self.decoder.xcnlin(common)
        return self.decoder.lin(common_features * self.decoder.beta + endpoint_features).flatten()

    def contexts(self, h, outer, graph, queries, neighbors, eta):
        q = len(queries)
        lq, ln = neighbors.left
        rq, rn = neighbors.right
        rows, nodes = torch.cat((lq, rq)), torch.cat((ln, rn))
        counterpart = torch.cat((queries[lq, 1], queries[rq, 0]))
        role = torch.cat((eta.new_zeros(len(lq)), eta.new_ones(len(rq)))).to(h.dtype)
        symmetric_endpoint = h[queries[:, 0]] + h[queries[:, 1]]
        base = torch.cat((outer[nodes], h[counterpart], symmetric_endpoint[rows], eta.to(h.dtype)[:, None]), 1)
        phi = self.count_head.slot(torch.cat((base, 1 - role[:, None], role[:, None]), 1))
        phis = self.count_head.slot(torch.cat((base, role[:, None], 1 - role[:, None]), 1))
        nl, nr = torch.bincount(lq, minlength=q), torch.bincount(rq, minlength=q)
        nc = torch.bincount(neighbors.common[0], minlength=q)
        common = feature_sum(outer, neighbors.common, q)
        def pools(mapped, flip):
            a, b = summed(mapped[:len(lq)], lq, q), summed(mapped[len(lq):], rq, q)
            pa = torch.cat((a, a / nl.clamp_min(1)[:, None]), 1)
            pb = torch.cat((b, b / nr.clamp_min(1)[:, None]), 1)
            return torch.cat((pb, pa) if flip else (pa, pb), 1)
        degree = (graph.rowptr[1:] - graph.rowptr[:-1]).to(h.dtype)
        du, dv = torch.log1p(degree[queries[:, 0]]), torch.log1p(degree[queries[:, 1]])
        affinity = torch.sigmoid(eta).to(h.dtype)
        mean = summed(affinity[:, None], rows, q).flatten() / (nl + nr).clamp_min(1)
        mean2 = summed(affinity.square()[:, None], rows, q).flatten() / (nl + nr).clamp_min(1)
        scalars = torch.stack((du + dv, (du - dv).abs(), torch.log1p(nc),
                               torch.log1p(nl), torch.log1p(nr), torch.log1p(nl + nr), mean, mean2), 1).to(h.dtype)
        swapped = scalars[:, [0, 1, 2, 4, 3, 5, 6, 7]]
        shared = torch.cat((symmetric_endpoint, h[queries[:, 0]] * h[queries[:, 1]],
                            common, common / nc.clamp_min(1)[:, None]), 1)
        return torch.cat((shared, pools(phi, False), scalars), 1), torch.cat((shared, pools(phis, True), swapped), 1), nl, nr

    def query_forward(self, h, graph, queries, *, auxiliary_grad=True):
        neighbors = enumerate_neighbors(graph, queries)
        lq, ln = neighbors.left
        rq, rn = neighbors.right
        outer = h + self.decoder.xlin(h)
        native_graph = adjacency(graph)
        def scorer(pair):
            # Full native second xlin and depth0 execute also for empty supports.
            return self.decoder(outer, native_graph, pair.T, depth=0).flatten()
        with torch.enable_grad() if auxiliary_grad else torch.no_grad():
            left = scorer(torch.stack((queries[lq, 1], ln), 1))
            right = scorer(torch.stack((queries[rq, 0], rn), 1))
            eta = 2.5 * (torch.cat((left, right)).to(torch.float64) - 6.) + math.log(.1)
            context, swapped, nl, nr = self.contexts(h, outer, graph, queries, neighbors, eta)
        # Side arrays retain native source order; query slices use offsets only.
        lo = torch.cat((nl.new_zeros(1), nl.cumsum(0))).detach().cpu().tolist()
        ro = torch.cat((nr.new_zeros(1), nr.cumsum(0))).detach().cpu().tolist()
        mu = torch.empty_like(eta)
        with torch.no_grad():
            for i in range(len(queries)):
                ls, rs = slice(lo[i], lo[i + 1]), slice(len(lq) + ro[i], len(lq) + ro[i + 1])
                potential = self.count_head.potential(context[i].detach(), swapped[i].detach(), lo[i+1]-lo[i], ro[i+1]-ro[i])
                current, _ = unary_marginals(eta[ls].detach(), eta[rs].detach(), potential)
                mu[ls], mu[rs] = current[:lo[i+1]-lo[i]], current[lo[i+1]-lo[i]:]
        common = feature_sum(outer, neighbors.common, len(queries))
        weight = (1.05 * mu).to(outer.dtype).detach()
        common = common + feature_sum(outer, neighbors.left, len(queries), weight[:len(lq)])
        common = common + feature_sum(outer, neighbors.right, len(queries), weight[len(lq):])
        logits = self.native_decode(h[queries[:, 0]] * h[queries[:, 1]], common)[:, None]
        return logits, {"neighbors": neighbors, "eta": eta, "context": context, "swapped_context": swapped,
                        "left_offsets": lo, "right_offsets": ro, "left_slots": len(lq),
                        "mu": mu.detach(), "completion_weight": weight, "queries": len(queries)}

    def source_nll(self, detail, labels):
        eta, context, swapped = detail["eta"], detail["context"], detail["swapped_context"]
        require(labels.shape == eta.shape, "Teacher cannot change source support")
        lo, ro, left_slots = detail["left_offsets"], detail["right_offsets"], detail["left_slots"]
        result = []
        for i in range(detail["queries"]):
            ls, rs = slice(lo[i], lo[i+1]), slice(left_slots + ro[i], left_slots + ro[i+1])
            z = torch.cat((labels[ls], labels[rs])).detach()
            def one(left, right, c, cs, z):
                g = self.count_head.potential(c, cs, len(left), len(right))
                return query_nll(left, right, g, z)
            # Recompute all count pairs/DP for backward; no all-query tables.
            with set_checkpoint_early_stop(False):
                result.append(checkpoint(one, eta[ls], eta[rs], context[i], swapped[i], z,
                                         use_reentrant=False, preserve_rng_state=True))
        return torch.stack(result) if result else eta.new_empty(0)

    @staticmethod
    def serve(logits):
        require(logits.ndim == 2 and logits.shape[1] == 1, "One C_mu target logit required")
        return logits[:, 0]


def make_count(mods, device):
    (encoder, decoder), _ = make_native(mods, MODEL_SEED, 64, device)
    model = CountSingle(encoder, decoder).to(device)
    require(sum(p.numel() for p in model.parameters()) == 88900, "Specified C_mu parameter total differs")
    optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": .0082},
                                  {"params": [*model.decoder.parameters(), *model.count_head.parameters()], "lr": .0037}],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)
    return model, optimizer
