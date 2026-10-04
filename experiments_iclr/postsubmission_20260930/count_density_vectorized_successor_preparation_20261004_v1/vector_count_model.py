"""Unexecuted native integration successor; same frozen C_mu architecture.

Requires v1 source assembly/native dependencies under root admission. CPU QA
imports vector density/head only; it does not import this native model adapter.
"""
import math
import torch
from count_model import CountSingle, adjacency
from graph_ops import enumerate_neighbors, feature_sum
from pilot_model import make_native
from pooled_model import MODEL_SEED
from vector_density import require
from ragged_density import unary_marginals, source_nll


class VectorCountSingle(CountSingle):
    def query_forward(self, h, graph, queries, *, auxiliary_grad=True):
        neighbors = enumerate_neighbors(graph, queries)
        lq, ln = neighbors.left
        rq, rn = neighbors.right
        outer = h + self.decoder.xlin(h)
        native_graph = adjacency(graph)
        with torch.enable_grad() if auxiliary_grad else torch.no_grad():
            # Same source-native depth0 left/right calls and empty-call schedule.
            left = self.decoder(outer, native_graph, torch.stack((queries[lq, 1], ln), 1).T, depth=0).flatten()
            right = self.decoder(outer, native_graph, torch.stack((queries[rq, 0], rn), 1).T, depth=0).flatten()
            eta = 2.5 * (torch.cat((left, right)).to(torch.float64) - 6.) + math.log(.1)
            context, swapped, nl, nr = self.contexts(h, outer, graph, queries, neighbors, eta)
        lo = torch.cat((nl.new_zeros(1), nl.cumsum(0)))
        ro = torch.cat((nr.new_zeros(1), nr.cumsum(0)))
        lm, rm, _ = unary_marginals(eta[:len(lq)], eta[len(lq):], lo, ro, context, swapped, self.count_head.count)
        mu = torch.cat((lm, rm)).detach()
        weight = (1.05 * mu).to(outer.dtype).detach()
        common = feature_sum(outer, neighbors.common, len(queries))
        common = common + feature_sum(outer, neighbors.left, len(queries), weight[:len(lq)])
        common = common + feature_sum(outer, neighbors.right, len(queries), weight[len(lq):])
        logits = self.native_decode(h[queries[:, 0]] * h[queries[:, 1]], common)[:, None]
        return logits, {"neighbors": neighbors, "eta": eta, "context": context, "swapped_context": swapped,
                        "left_offsets": lo, "right_offsets": ro, "left_slots": len(lq),
                        "mu": mu, "completion_weight": weight, "queries": len(queries)}

    def source_nll(self, detail, labels):
        n = detail["left_slots"]
        require(labels.shape == detail["eta"].shape, "Teacher cannot change source support")
        return source_nll(detail["eta"][:n], detail["eta"][n:], detail["left_offsets"], detail["right_offsets"],
                          detail["context"], detail["swapped_context"], self.count_head.count, labels[:n], labels[n:])


def make_vector_count(mods, device):
    require(torch.get_default_dtype() == torch.float32 and not torch.is_autocast_enabled()
            and not torch.is_autocast_enabled("cpu"), "Plain native FP32 profile required")
    (encoder, decoder), _ = make_native(mods, MODEL_SEED, 64, device)
    model = VectorCountSingle(encoder, decoder).to(device)
    require(sum(p.numel() for p in model.parameters()) == 88900, "Frozen C_mu architecture changed")
    optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": .0082},
                                  {"params": [*model.decoder.parameters(), *model.count_head.parameters()], "lr": .0037}],
                                 betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False)
    return model, optimizer
