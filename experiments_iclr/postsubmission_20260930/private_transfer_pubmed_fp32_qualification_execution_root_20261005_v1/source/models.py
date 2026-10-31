"""Unframed native NCN architectures and exhaustive episode parameter roles.

Numerical modules are supplied only by the authorized runner/qualification.
The donor head source is unchanged; no Householder or new graph operator.
"""

ARMS = ("shared_f4", "capable_single", "untied4", "ordinary_native4")


def build(native, donor_heads, *, arm, seed, factor_seed, device):
    torch, nn = native.torch, native.nn
    if arm not in ARMS or type(seed) is not int or type(factor_seed) is not int:
        raise ValueError("Explicit supported architecture/initialization required")

    def encoder():
        return native.GCN(500, 256, 256, 1, .3, True, False, -1,
                          "puregcn", True, 0., xdropout=.4, taildropout=0., noinputlin=False)

    def bank():
        return donor_heads.make_predictor(native, arm="unframed_f4", members=4,
                                          width=256, factor_seed=factor_seed)

    def native_single():
        return donor_heads.make_predictor(native, arm="native_single", members=1, width=256)

    def private_bank_ids(predictor):
        result = {id(predictor.beta)}
        for sequence_name in ("xlin", "xcnlin", "xijlin", "lin"):
            for op in getattr(predictor, sequence_name).ops:
                if hasattr(op, "r") and hasattr(op, "s"):
                    result.update((id(op.r), id(op.s)))
                elif hasattr(op, "forward_member") and hasattr(op, "normalized_shape"):
                    result.update((id(op.weight), id(op.bias)))
        return result

    def select_bank_route(predictor, member):
        # Preserve the donor route's exact operations and initialization, but
        # remove unused private rows. The donor four-route forward is not called.
        private = private_bank_ids(predictor)
        for module in predictor.modules():
            for name, value in list(module.named_parameters(recurse=False)):
                if id(value) in private:
                    if value.ndim < 1 or value.shape[0] != 4:
                        raise ValueError("Donor route-private row geometry changed")
                    setattr(module, name, nn.Parameter(value.detach()[member:member+1].clone()))
        predictor.members = 1
        return predictor

    def score_bank(predictor, h, support, edges, member):
        adjacency = predictor.dropadj(support)
        left, right = h[edges[0]], h[edges[1]]
        context_matrix = native.adjoverlap(adjacency, adjacency, edges, False, cnsampledeg=-1)
        transformed = h + predictor.xlin.forward_member(h, member)
        common = native.spmm_add(context_matrix, transformed)
        pair = predictor.xijlin.forward_member(left*right, member)
        context = predictor.xcnlin.forward_member(common, member)
        return predictor.lin.forward_member(context*predictor.beta[member]+pair, member)

    class Route(nn.Module):
        def __init__(self, enc, pred, native_head=False):
            super().__init__(); self.encoder, self.predictor = enc, pred
            self.native_head = native_head
        def forward(self, features, support, positive, negative):
            h = self.encoder(features, support)
            if self.native_head:
                return self.predictor(h, support, positive), self.predictor(h, support, negative)
            return (score_bank(self.predictor, h, support, positive, 0),
                    score_bank(self.predictor, h, support, negative, 0))

    class Shared(nn.Module):
        def __init__(self, enc, pred):
            super().__init__(); self.encoder, self.predictor = enc, pred
            self.member_count = 4
        def forward(self, features, support, positive, negative, route=None):
            h = self.encoder(features, support)
            members = range(4) if route is None else (route,)
            if any(type(member) is not int or member not in range(4) for member in members):
                raise ValueError("Route outside unframed F4")
            return (torch.cat([score_bank(self.predictor, h, support, positive, member) for member in members], 1),
                    torch.cat([score_bank(self.predictor, h, support, negative, member) for member in members], 1))

    class Single(Route):
        member_count = 1
        def forward(self, features, support, positive, negative, route=None):
            if route not in (None, 0):
                raise ValueError("Single predictor has one route")
            return super().forward(features, support, positive, negative)

    class Untied(nn.Module):
        def __init__(self, routes):
            super().__init__(); self.routes = nn.ModuleList(routes); self.member_count = 4
        def forward(self, features, support, positive, negative, route=None):
            if route is not None:
                if type(route) is not int or route not in range(4):
                    raise ValueError("Route outside untied bank")
                return self.routes[route](features, support, positive, negative)
            logits = [member(features, support, positive, negative) for member in self.routes]
            return torch.cat([row[0] for row in logits], 1), torch.cat([row[1] for row in logits], 1)

    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device): torch.cuda.manual_seed(seed)
        if arm == "shared_f4":
            model = Shared(encoder(), bank())
            private_ids = private_bank_ids(model.predictor)
        elif arm == "capable_single":
            model = Single(encoder(), native_single(), native_head=True)
            # Entire nonlinear native predictor adapts, not an artificially
            # restricted F1 factor vector. Encoder is the only outer block.
            private_ids = {id(parameter) for parameter in model.predictor.parameters()}
        else:
            routes = []
            for member in range(4):
                route_seed = seed+5*member if arm == "ordinary_native4" else seed
                # Untying alone preserves every initial shared matrix and the
                # corresponding F4 private row. The ordinary native ensemble
                # separately keeps independent base initializations.
                torch.random.default_generator.manual_seed(route_seed)
                if device.type == "cuda":
                    with torch.cuda.device(device): torch.cuda.manual_seed(route_seed)
                routes.append(Route(encoder(), native_single(), native_head=True) if arm == "ordinary_native4"
                              else Route(encoder(), select_bank_route(bank(), member)))
            model = Untied(routes)
            private_ids = ({id(parameter) for route in model.routes for parameter in route.predictor.parameters()}
                           if arm == "ordinary_native4" else
                           set().union(*(private_bank_ids(route.predictor) for route in model.routes)))
    model = model.to(device)
    # Module.to preserves Parameter identities in the declared native FP32
    # runtime; assert complete role coverage after moving rather than assume it.
    names = {name: value for name, value in model.named_parameters()}
    private = tuple(name for name, value in names.items() if id(value) in private_ids)
    shared = tuple(name for name in names if name not in private)
    if not private or not shared or len(set(shared)|set(private)) != len(names):
        raise ValueError("Exhaustive shared/private parameter partition failed")
    if arm == "shared_f4" and (sum(names[n].numel() for n in shared), sum(names[n].numel() for n in private)) != (589058, 25608):
        raise ValueError("Unchanged unframed F4 parameter partition differs")
    if arm == "untied4" and (sum(names[n].numel() for n in shared), sum(names[n].numel() for n in private)) != (4*589058, 25608):
        raise ValueError("Same-operation untied4 parameter partition differs")
    if arm == "capable_single" and any(name.startswith("predictor.") for name in shared):
        raise ValueError("Capable nonlinear predictor must be fully privately adaptable")
    for module in model.modules():
        if hasattr(module, "inplace") and module.inplace:
            # Preserve native in-place flags. A new direct derivative gate must
            # verify their actual live path; this is not a silent rewrite.
            pass
    partition = {"arm": arm, "shared": list(shared), "private": list(private),
                 "parameters": [{"name": name, "shape": list(value.shape),
                                 "numel": value.numel(), "role": "private" if name in private else "shared"}
                                for name, value in names.items()],
                 "buffers": [{"name": name, "shape": list(value.shape), "role": "constant_native_buffer"}
                             for name, value in model.named_buffers()],
                 "capable_single_partition_difference": "encoder outer / entire nonlinear NCN head inner; richer private adaptation" if arm == "capable_single" else None}
    partition["initialization"] = ("identical_initial_shared_encoder_and_base_matrices_per_route; corresponding_F4_private_rows"
                                   if arm == "untied4" else "independent_seed_plus5member" if arm == "ordinary_native4"
                                   else "native_seed_and_explicit_F4_factor_seed")
    return model, shared, private, partition
