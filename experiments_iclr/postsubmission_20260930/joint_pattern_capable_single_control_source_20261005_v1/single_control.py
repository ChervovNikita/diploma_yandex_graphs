"""One native NCNC target with an internal conditional-pattern density.

SOURCE ONLY: no data loader, fit loop, launch entry point, or state donor.
Callers supply the bound native constructor, graph helpers, adjacency factory,
teacher, and exact existing conditional-loss module. Nothing is loaded here.
"""
import torch
from torch import nn
from torch.nn import functional as F


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SingleEndpointPattern(nn.Module):
    """One N64 encoder/scorer/target decoder; four auxiliary emissions only.

    The emitted density components share each native candidate's complete
    penultimate feature. They are never independently decoded target members.
    The teacher/counts are absent from all model-forward interfaces.
    """
    def __init__(self, encoder, decoder, graph_ops, adjacency_factory):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.graph_ops, self.adjacency_factory = graph_ops, adjacency_factory
        require(decoder.depth == 1 and decoder.splitsize == -1 and not decoder.learnablept,
                "Requires bound native depth1/fixed-pt/full-support recipe")
        require(decoder.cndeg == decoder.trainresdeg == decoder.testresdeg == -1,
                "Residual/common support sampling is not this control")
        require(len(decoder.lin) == 9 and isinstance(decoder.lin[8], nn.Linear)
                and decoder.lin[8].in_features == 64 and decoder.lin[8].out_features == 1,
                "Bound native lin[0:8] penultimate/lin[8] scalar readout required")
        last = decoder.lin[8]
        require(last.weight.dtype == torch.float32, "Native FP32 target required")
        # nn.Linear.reset_parameters initializes four distinct rows using its
        # ordinary rule. CPU fork restores the existing native RNG stream; no
        # new initializer seed, identical-row copy, dropout or search is added.
        with torch.random.fork_rng(devices=[]):
            emission = nn.Linear(last.in_features, 4, device="cpu", dtype=torch.float32)
        self.auxiliary_emission = emission.to(device=last.weight.device)

    def encode(self, features, graph):
        return self.encoder(features, self.adjacency_factory(graph))

    def _decode_features(self, endpoint_product, common_features):
        # Preserve native order: xijlin, xcnlin, then lin. lin[4:8] are the
        # pinned twolayerlin=False identities; phi follows native lin[7].
        endpoint = self.decoder.xijlin(endpoint_product)
        common = self.decoder.xcnlin(common_features)
        phi = common * self.decoder.beta + endpoint
        for index in range(8):
            phi = self.decoder.lin[index](phi)
        scalar = self.decoder.lin[8](phi).flatten()
        return scalar, phi

    def _depth_zero(self, outer, graph, queries, *, emit):
        # This is the native recursive scorer, including its second full-node
        # xlin and complete common-neighbor decode even for empty queries.
        endpoint = outer[queries[:, 0]] * outer[queries[:, 1]]
        second = outer + self.decoder.xlin(outer)
        neighbors = self.graph_ops.enumerate_neighbors(graph, queries)
        common = self.graph_ops.feature_sum(second, neighbors.common, len(queries))
        scalar, phi = self._decode_features(endpoint, common)
        eta = None
        if emit:
            raw = self.auxiliary_emission(phi).T
            eta = self.decoder.scale * (raw - self.decoder.offset) + torch.log(self.decoder.pt)
        return scalar, eta

    def query_forward(self, h, graph, queries, *, emit_auxiliary=False):
        """One original scalar soft-completion target trajectory.

        Auxiliary emissions can train the candidate scorer trunk. Main target
        gradients do not traverse completion scores, as in native NCNC; live
        outer features and outer native scalar decoder retain main gradients.
        """
        require(queries.ndim == 2 and queries.shape[1] == 2 and queries.dtype == torch.long,
                "Native integer endpoint pairs required")
        require(h.dtype == torch.float32 and h.shape == (graph.nodes, 64),
                "Bound native N64 representation required")
        neighbors = self.graph_ops.enumerate_neighbors(graph, queries)
        lq, ln = neighbors.left
        rq, rn = neighbors.right
        left_queries = torch.stack((queries[lq, 1], ln), dim=1)
        right_queries = torch.stack((queries[rq, 0], rn), dim=1)
        endpoint = h[queries[:, 0]] * h[queries[:, 1]]
        outer = h + self.decoder.xlin(h)
        # Preserve outer -> left recursion -> right recursion -> target order.
        # The auxiliary branch enables the entire two scorer trunks, never
        # just their new projection. Serving keeps native recursive no_grad.
        context = torch.enable_grad() if emit_auxiliary else torch.no_grad()
        with context:
            left_score, eta_left = self._depth_zero(outer, graph, left_queries, emit=emit_auxiliary)
            right_score, eta_right = self._depth_zero(outer, graph, right_queries, emit=emit_auxiliary)
        # Delegate alpha/pt/p0/order to the original native scalar clamp.
        left_weight = self.decoder.clampprob(left_score.detach(), self.decoder.pt)
        right_weight = self.decoder.clampprob(right_score.detach(), self.decoder.pt)
        common = self.graph_ops.feature_sum(outer, neighbors.common, len(queries))
        # Native scalar source adds right residuals before left residuals.
        # Keep this floating accumulation order rather than the F4 prototype's
        # equivalent real-arithmetic left-before-right expression.
        common = common + self.graph_ops.feature_sum(outer, neighbors.right, len(queries), right_weight)
        common = common + self.graph_ops.feature_sum(outer, neighbors.left, len(queries), left_weight)
        scalar, _ = self._decode_features(endpoint, common)
        require(scalar.shape == (len(queries),), "Exactly one scalar target logit required")
        return scalar, {"neighbors": neighbors, "eta_left": eta_left, "eta_right": eta_right,
                        "native_left_score": left_score, "native_right_score": right_score,
                        "native_left_weight": left_weight, "native_right_weight": right_weight}

    @staticmethod
    def serve(scalar):
        require(scalar.ndim == 1, "One native raw logit per query required")
        return scalar


def make_control(mods, seed, device, graph_ops, adjacency_factory, make_native):
    """Reuse the pinned fresh native constructor and its unchanged Adam recipe.

    The future caller authenticates make_native/dependencies before this call.
    No trained source state is accepted. The new head joins the existing native
    decoder optimizer group; no optimizer setting is introduced.
    """
    native, optimizer = make_native(mods, seed, 64, device)
    encoder, decoder = native
    require(len(optimizer.param_groups) == 2 and not optimizer.state,
            "Fresh bound two-group native Adam required")
    model = SingleEndpointPattern(encoder, decoder, graph_ops, adjacency_factory)
    existing = {id(p) for group in optimizer.param_groups for p in group["params"]}
    added = list(model.auxiliary_emission.parameters())
    require(not any(id(p) in existing for p in added), "Auxiliary tensors already in optimizer")
    optimizer.param_groups[1]["params"].extend(added)
    require(sum(p.numel() for p in model.parameters()) == 38407,
            "Native38147 plus260 auxiliary parameters required")
    return model, optimizer


def objective(model, h, graph, positive_queries, negative_queries, teacher, loss_core, *, arm):
    """Reusable per-batch objective, without masks, optimizer steps or a loop.

    teacher is the exact existing ObservationTeacher. loss_core is the bound
    conditional_loss module, including TrainPatternLabels. Caller constructs
    the same record-masked graph and native queries before this function.
    """
    require(arm in ("target_only", "joint", "separate"), "Unknown control objective")
    key = {"joint": "J_K", "separate": "J_K_sep"}.get(arm)
    logits, auxiliary = [], []
    for queries in (positive_queries, negative_queries):
        require(len(queries) > 0, "Original nonempty native query batch required")
        scalar, detail = model.query_forward(h, graph, queries, emit_auxiliary=key is not None)
        logits.append(scalar)
        if key is not None:
            rows, bits = teacher.labels(queries, detail["neighbors"])
            left_rows, _ = detail["neighbors"].left
            right_rows, _ = detail["neighbors"].right
            require(torch.equal(rows, torch.cat((left_rows, right_rows))), "Exact teacher/support order required")
            labels = loss_core.TrainPatternLabels(bits.to(detail["eta_left"].dtype),
                                                 "complete_TRAIN_observation_membership")
            values = loss_core.training_pattern_losses(detail["eta_left"], detail["eta_right"],
                                                      left_rows, right_rows, labels, len(queries))
            auxiliary.append(values[key].mean())
    main = -F.logsigmoid(logits[0]).mean() - F.logsigmoid(-logits[1]).mean()
    aux = sum(auxiliary) if auxiliary else main.new_zeros(())
    return {"main": main, "auxiliary": aux, "total": main + aux, "logits": logits}
