"""Native NCN nonlinear incoming-message growth; no numerical import at load.

All numerical objects come from the pinned native runtime. The native base
encoder arithmetic is executed once. Only the new tanh message channels are
packed; no existing member trajectory is replaced by a cached representation.
"""
from math import comb

CONDITIONS = ("no_growth", "graph_growth", "unfiltered_growth", "unfiltered_top8_partition",
              "capable_single_rank8", "independent_graph_growth4")
WIDTH, MEMBERS, RANK = 256, 4, 2
RANK_RTOL, RANK_ATOL, PROJECTOR_MIN = 1e-6, 1e-12, 1e-4


def require_site(torch, encoder):
    """Admit only the actual one-layer Citeseer puregcn donor interface."""
    nn = torch.nn
    if len(encoder.convs) != 1 or len(encoder.lins) != 1:
        raise ValueError("Growth requires exact one-layer native encoder")
    conv = encoder.convs[0]
    if conv.__class__.__name__ != "PureConv" or conv.aggr != "gcn" or not isinstance(conv.lin, nn.Identity):
        raise ValueError("Native normalized PureConv identity map changed")
    if encoder.res or not encoder.jk or tuple(encoder.jkparams.shape) != (1,):
        raise ValueError("Native residual/JK site changed")
    if not isinstance(encoder.lins[0], nn.Dropout) or encoder.lins[0].p != 0.:
        raise ValueError("Only native zero-dropout linear tail can be split")
    if encoder.adjdrop.dp != 0. or len(encoder.xemb) != 3:
        raise ValueError("Native support or input interface changed")
    left, linear, right = encoder.xemb
    if not isinstance(left, nn.Dropout) or left.p != .4 or not isinstance(linear, nn.Linear):
        raise ValueError("Native feature dropout/projection changed")
    if (linear.in_features, linear.out_features) != (3703, WIDTH):
        raise ValueError("Native feature projection geometry changed")
    if not isinstance(right, nn.Dropout) or right.p != .3 or not right.inplace:
        raise ValueError("Native projected-feature dropout changed")
    return conv


def finish(torch, encoder, aggregated):
    """Literal donor one-layer tail and JK arithmetic, including stack/sum."""
    value = encoder.lins[0](aggregated)
    jkx = torch.stack([value], dim=0)
    sftmax = encoder.jkparams.reshape(-1, 1, 1)
    return torch.sum(jkx*sftmax, dim=0)


def score_bank(native, predictor, h, support, edges, member):
    """Actual unchanged member NCN head operators, with member-specific h."""
    adj = predictor.dropadj(support)
    left, right = h[edges[0]], h[edges[1]]
    cn = native.adjoverlap(adj, adj, edges, False, cnsampledeg=-1)
    transformed = h + predictor.xlin.forward_member(h, member)
    common = native.spmm_add(cn, transformed)
    pair = predictor.xijlin.forward_member(left*right, member)
    context = predictor.xcnlin.forward_member(common, member)
    return predictor.lin.forward_member(context*predictor.beta[member]+pair, member)


def calibrate(torch, warm, features, support, queries):
    """One dropout-off TRAIN episode, upstream gradient before native tail/JK.

    Support already removes the endpoint/random union targets. Never accepts
    labels, query roles or support from VALID/TEST. The caller authenticates the
    fixed TRAIN episode and stores its exact draws/support identity.
    """
    require_site(torch, warm.encoder)
    modes = [(m, m.training) for m in warm.modules()]
    warm.eval()
    try:
        h = warm.encoder.xemb(features)
        prepared = warm.encoder.adjdrop(support)
        y = warm.encoder.convs[0](h, prepared)
        latent = finish(torch, warm.encoder, y)
        positive = warm.predictor(latent, support, queries[0])
        negative = warm.predictor(latent, support, queries[1])
        objective = (-torch.nn.functional.logsigmoid(positive).mean()
                     -torch.nn.functional.logsigmoid(-negative).mean())
        g = torch.autograd.grad(objective, y)[0]
        if not bool(torch.isfinite(h).all() and torch.isfinite(g).all()) or not bool(g.norm() > 0):
            raise ValueError("No finite nonzero TRAIN calibration gradient; no fallback")
        return h.detach(), g.detach(), {"objective": float(objective.detach()),
            "H_shape": list(h.shape), "G_shape": list(g.shape),
            "G_norm": float(g.norm()), "dropout_off": True,
            "upstream_site": "PureConv output before native tail/JK"}
    finally:
        for module, mode in modes: module.training = mode


def bases(torch, encoder, h, g, support, required=None):
    """Degree-three Bernstein error bands and explicit Stiefel right bases.

    P is the *same* native PureConv on the same symmetric masked support. The
    packed four-band action pays four wide columns through three steps and
    another P action. No eigenvectors of the graph or scalar strength search.
    SVD is fixed CPU float64 on the four 256x256 matrices and unfiltered K.
    """
    conv = require_site(torch, encoder)
    if not support.is_symmetric() or support.storage.value() is not None:
        raise ValueError("Require native symmetric unweighted target-masked support")
    row, col, _ = support.coo()
    if bool((row == col).any()):
        raise ValueError("Native support must exclude loops; PureConv adds one self term")
    if required not in (None,"graph","unfiltered","single","partition"):
        raise ValueError("Fixed initializer requirement changed")
    with torch.no_grad():
        hd = h.detach().cpu().double()
        ks, ku, widths = [], None, []
        if required in (None,"graph"):
            packed = torch.cat([g]*MEMBERS, dim=1)
            for step in range(3):
                ppacked = conv(packed, support); widths.append(1024)
                old = packed.split(WIDTH, dim=1)
                propagated = ppacked.split(WIDTH, dim=1)
                packed = torch.cat([old[q]+propagated[q] if step < 3-q
                                    else old[q]-propagated[q] for q in range(MEMBERS)], dim=1)
            filtered = torch.cat([part*(comb(3, q)/8.)
                                 for q, part in enumerate(packed.split(WIDTH, dim=1))], dim=1)
            pfg = conv(filtered, support).detach().cpu().double(); widths.append(1024)
            ks = [hd.t()@part for part in pfg.split(WIDTH, dim=1)]
        if required in (None,"unfiltered","single","partition"):
            pg = conv(g, support).detach().cpu().double(); widths.append(256)
            ku = hd.t()@pg
        records, result, target, distances = [], [], {}, []

        def right_basis(k, rank, label):
            if not bool(torch.isfinite(k).all()):
                raise ValueError("Nonfinite fixed calibration matrix: "+label)
            _, singular, right = torch.linalg.svd(k, full_matrices=False)
            cutoff = max(RANK_ATOL, float(singular[0])*RANK_RTOL)
            usable = int((singular > cutoff).sum())
            if usable < rank:
                raise ValueError("Unusable fixed calibration rank: "+label+"; no redraw")
            basis = right[:rank].contiguous()
            records.append({"label": label, "rank_required": rank, "usable_rank": usable,
                "relative_cutoff": RANK_RTOL, "absolute_cutoff": RANK_ATOL,
                "singular_values": singular.tolist(), "K_frobenius": float(k.norm()),
                "orthonormal_max_error_float64": float((basis@basis.t()-torch.eye(rank,dtype=basis.dtype)).abs().max())})
            return basis

        if ks:
            for q, k in enumerate(ks): result.append(right_basis(k, RANK, "graph_band"+str(q)))
            graph = torch.stack(result)
            projectors = graph.transpose(1,2)@graph
            distances = [{"members": [a,b], "normalized_projector_distance":
                          float((projectors[a]-projectors[b]).norm()/(2*RANK)**.5)}
                         for a in range(MEMBERS) for b in range(a+1,MEMBERS)]
            if min(row["normalized_projector_distance"] for row in distances) <= PROJECTOR_MIN:
                raise ValueError("Graph output projectors collapsed; no diversity rescue")
            target["graph"] = graph.to(device=h.device,dtype=h.dtype)
        if required in (None,"unfiltered"):
            unfiltered = right_basis(ku,RANK,"unfiltered_rank2")
            target["unfiltered"] = unfiltered.unsqueeze(0).expand(MEMBERS,-1,-1).clone().to(device=h.device,dtype=h.dtype)
        partition_distances = []
        if required in (None,"single","partition"):
            single = right_basis(ku,8,"single_and_partition_unfiltered_rank8")
            if required in (None,"single"):
                target["single"] = single.unsqueeze(0).to(device=h.device,dtype=h.dtype)
            if required in (None,"partition"):
                partition = single.reshape(MEMBERS,RANK,WIDTH)
                target["partition"] = partition.to(device=h.device,dtype=h.dtype)
                projectors = partition.transpose(1,2)@partition
                partition_distances = [{"members":[a,b],"normalized_projector_distance":
                    float((projectors[a]-projectors[b]).norm()/(2*RANK)**.5)}
                    for a in range(MEMBERS) for b in range(a+1,MEMBERS)]
                if min(row["normalized_projector_distance"] for row in partition_distances) <= PROJECTOR_MIN:
                    raise ValueError("Unfiltered top8 partition projectors are not distinct")
        for key, values in target.items():
            identity = torch.eye(values.shape[1],device=values.device,dtype=values.dtype)
            if not torch.allclose(values@values.transpose(1,2),identity.expand(len(values),-1,-1),rtol=1e-5,atol=1e-5):
                raise ValueError("Stiefel basis lost orthonormality in native FP32: "+key)
        return target, {"records": records, "graph_projector_distances": distances,
            "unfiltered_top8_partition_projector_distances":partition_distances,
            "projector_stop_threshold": PROJECTOR_MIN,
            "P_actions": len(widths), "P_column_widths": widths,
            "filter": "binom(3,q)(I+P)^(3-q)(I-P)^q/8",
            "K": "H^T P^T F_q(P) G", "P_transpose_equals_P_checked": True,
            "larger_gradient_norm_is_not_Adam_benefit_guarantee": True}


def make(torch, native, heads, models, initializer, warm, condition, seed, factor_seed, device, basis):
    """Copy fixed warm states; new V/B remain private, base parameters learn."""
    if condition not in CONDITIONS: raise ValueError("Unknown fixed growth condition")
    initial = ("single" if condition == "capable_single_rank8" else
               "independent_warm4" if condition == "independent_graph_growth4" else "warm_identity")
    base, partition = initializer.make(torch,native,heads,models,warm,initial,seed,factor_seed,device,{})
    if condition == "no_growth":
        partition["growth"] = None
        return base, partition
    nn = torch.nn

    class IncomingMessages(nn.Module):
        def __init__(self, output_basis):
            super().__init__()
            self.B = nn.Parameter(output_basis.detach().clone())
            self.V = nn.Parameter(self.B.new_zeros((len(self.B), WIDTH, self.B.shape[1])))
        def forward(self, h, support, conv, members):
            # torch.cat of d x r matrices makes column groups [member, unit].
            incoming = torch.cat([self.V[m] for m in members],dim=1)
            packed = torch.tanh(h@incoming)
            propagated = conv(packed,support)
            groups = propagated.split(self.B.shape[1],dim=1)
            return [group@self.B[m] for group,m in zip(groups,members)]

    class SharedGrowth(nn.Module):
        def __init__(self, copied, output_basis):
            super().__init__()
            self.encoder, self.predictor = copied.encoder, copied.predictor
            self.growth = IncomingMessages(output_basis)
            self.member_count = 4
            require_site(torch,self.encoder)
        def forward(self, features, support, positive, negative, route=None):
            members = list(range(4)) if route is None else [route]
            if any(type(m) is not int or m not in range(4) for m in members):
                raise ValueError("Growth bank route must be 0..3")
            h = self.encoder.xemb(features)
            prepared = self.encoder.adjdrop(support)
            conv = self.encoder.convs[0]
            native_base = finish(torch,self.encoder,conv(h,prepared))
            corrections = self.growth(h,prepared,conv,members)
            # The zero-tail/one-scalar-JK interface is linear. Preserve the
            # literal native base path and add the same tail/JK correction.
            latents = [native_base+finish(torch,self.encoder,c) for c in corrections]
            return (torch.cat([score_bank(native,self.predictor,z,support,positive,m) for z,m in zip(latents,members)],1),
                    torch.cat([score_bank(native,self.predictor,z,support,negative,m) for z,m in zip(latents,members)],1))

    class SingleGrowth(nn.Module):
        member_count = 1
        def __init__(self, copied, output_basis):
            super().__init__()
            self.encoder, self.predictor = copied.encoder, copied.predictor
            self.growth = IncomingMessages(output_basis)
            require_site(torch,self.encoder)
        def forward(self, features, support, positive, negative, route=None):
            if route not in (None,0): raise ValueError("Native growth single has one route")
            h = self.encoder.xemb(features)
            prepared = self.encoder.adjdrop(support)
            conv = self.encoder.convs[0]
            base_latent = finish(torch,self.encoder,conv(h,prepared))
            correction = self.growth(h,prepared,conv,[0])[0]
            latent = base_latent+finish(torch,self.encoder,correction)
            return self.predictor(latent,support,positive),self.predictor(latent,support,negative)

    if condition == "independent_graph_growth4":
        for m, route in enumerate(list(base.routes)):
            base.routes[m] = SingleGrowth(route,basis["graph"][m:m+1])
        model = base
    elif condition == "capable_single_rank8": model = SingleGrowth(base,basis["single"])
    else:
        key = {"graph_growth":"graph","unfiltered_growth":"unfiltered",
               "unfiltered_top8_partition":"partition"}[condition]
        model = SharedGrowth(base,basis[key])
    for parameter in model.parameters(): parameter.requires_grad_(True)
    names = dict(model.named_parameters())
    added = [name for name in names if name.startswith("growth.") or ".growth." in name]
    total = sum(names[name].numel() for name in added)
    if total != 4096:
        raise ValueError("Every growth arm must add exactly4096 parameters")
    partition["growth"] = {"condition":condition,"private_names":added,"new_parameters":total,
        "V_initial":0.,"B":"fixed TRAIN-gradient right Stiefel basis",
        "filtering_scope":"output subspace initialization only; no persistent graph-frequency specialization constraint",
        "base_trainable":True,"packed_shared_serving_channels":8,
        "exact_real_function_inclusion":True,"FP32_gate_required":True,
        "old_base_arithmetic":"native base tail/stack/JK sum once plus linear tail/JK correction"}
    partition["private"] += added
    partition["parameters"] = [{"name":n,"shape":list(v.shape),"numel":v.numel(),
        "role":"private" if n in partition["private"] else "shared"} for n,v in names.items()]
    if condition == "independent_graph_growth4":
        partition["initialization"] = "four complete native models copied from one common warm; separate postwarm optimizers and query/dropout streams"
        partition["selection"] = "synchronous first maximum pooled VALID MRR; joint selection adaptation, not ordinary independently selected/cold native4"
        partition["optimization"] = "four independent own losses and Adam states; no pooled-gradient coupling"
    return model, partition
