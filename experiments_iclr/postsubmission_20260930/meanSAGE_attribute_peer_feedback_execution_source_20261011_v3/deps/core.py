"""Inactive source-only meanSAGE attribute peer feedback.

No torch/numpy/scientific imports or work at module import. Root injects torch/F,
qualified public X/A and fixed incidence C/D. No trainer, optimizer, selector,
provider, scores, file I/O or scientific execution admission is implemented here.
"""
from contextlib import nullcontext

NODES, FEATURES, HIDDEN, CLASSES = 34493, 8415, 32, 5
RANK, FEEDBACK_SCALE = 16, .25
BLOCKS = (10103, 10427, 10949)

ARM_SPEC = {
    'P1':  {'members': 1, 'factors': False, 'ownership': 'shared',   'feedback': 'none',          'independent': False},
    'F1':  {'members': 1, 'factors': True,  'ownership': 'shared',   'feedback': 'none',          'independent': False},
    'I4':  {'members': 1, 'factors': True,  'ownership': 'shared',   'feedback': 'none',          'independent': True},
    'S4':  {'members': 4, 'factors': True,  'ownership': 'shared',   'feedback': 'none',          'independent': False},
    'PA1': {'members': 1, 'factors': False, 'ownership': 'shared',   'feedback': 'self',          'independent': False},
    'IA4': {'members': 1, 'factors': True,  'ownership': 'shared',   'feedback': 'self',          'independent': True},
    'SA4': {'members': 4, 'factors': True,  'ownership': 'shared',   'feedback': 'self',          'independent': False},
    'SM4': {'members': 4, 'factors': True,  'ownership': 'shared',   'feedback': 'mean_then_phi', 'independent': False},
    'SX4': {'members': 4, 'factors': True,  'ownership': 'shared',   'feedback': 'phi_then_mean', 'independent': False},
    'UX4': {'members': 4, 'factors': True,  'ownership': 'unshared', 'feedback': 'phi_then_mean', 'independent': False},
}
EXPECTED_ARM_PARAMETERS = {
    'P1': 538917, 'F1': 555885, 'I4': 2223540, 'S4': 606900,
    'PA1': 539941, 'IA4': 2227636, 'SA4': 607924, 'SM4': 607924,
    'SX4': 607924, 'UX4': 2227636,
}


def incidence_from_raw_binary_features(torch, raw_features):
    """Runtime helper only: label-free C/D, built once from UNDROPPED full X.

    Root qualifies raw binary/coalesced input and binds this incidence to that X.
    No N-by-N graph, raw-input normalization or label argument exists.
    """
    if not raw_features.is_sparse or not raw_features.is_coalesced():
        raise ValueError('Qualified coalesced sparse raw X required')
    if tuple(raw_features.shape) != (NODES, FEATURES):
        raise ValueError('Frozen complete Physics dimensions required')
    ids = raw_features.indices()[:, raw_features.values() > 0]
    rows, columns = ids[0], ids[1]
    ones = torch.ones(rows.numel(), dtype=raw_features.dtype, device=raw_features.device)
    row_counts = torch.bincount(rows, minlength=NODES).clamp_min(1).to(raw_features.dtype)
    column_counts = torch.bincount(columns, minlength=FEATURES).clamp_min(1).to(raw_features.dtype)
    D = torch.sparse_coo_tensor(ids, ones / row_counts[rows], (NODES, FEATURES),
                               device=raw_features.device).coalesce()
    reverse_ids = torch.stack((columns, rows))
    C = torch.sparse_coo_tensor(reverse_ids, ones / column_counts[columns], (FEATURES, NODES),
                               device=raw_features.device).coalesce()
    return C, D


def build_arm(torch, F, arm, block):
    """Construct CPU source cores after root admission; no numerical imports here.

    Returns a plain dict with one joint model, or FOUR disjoint M1 models for
    genuine I4/IA4. Never train the latter via a joint bank objective/selector.
    Caller supplies persistent route RNG contexts to all TRAIN forwards; eval
    needs no context. Construction forks/restores global CPU RNG. Root seeds and
    owns persistent post-constructor dropout streams; this core never reseeds
    during training or evaluation.
    """
    if arm not in ARM_SPEC or type(block) is not int or block not in BLOCKS:
        raise ValueError('Frozen arm and integer block required')
    spec = dict(ARM_SPEC[arm])

    def clip_rows(z):
        return z / torch.linalg.vector_norm(z, ord=2, dim=1, keepdim=True).clamp_min(1.)

    def rademacher(size, generator):
        return torch.randint(0, 2, (size,), generator=generator).to(torch.float32).mul_(2).sub_(1)

    class RouteLayer(torch.nn.Module):
        def __init__(self, input_dim, output_dim, members, factor_generators, factorized):
            super().__init__()
            self.aggregate_weight = torch.nn.Parameter(torch.empty(input_dim, output_dim))
            self.skip_weight = torch.nn.Parameter(torch.empty(input_dim, output_dim))
            self.bias = torch.nn.Parameter(torch.zeros(members, output_dim))
            torch.nn.init.xavier_uniform_(self.aggregate_weight)
            torch.nn.init.xavier_uniform_(self.skip_weight)
            self.factorized = factorized
            if factorized:
                # Per-route generators maintain F1/shared route0/I4 body0 pairing.
                # Factors do not consume the source slow-weight/global RNG.
                ragg, sagg, rskip, sskip = [], [], [], []
                for generator in factor_generators:
                    ragg.append(rademacher(input_dim, generator))
                    sagg.append(rademacher(output_dim, generator))
                    rskip.append(rademacher(input_dim, generator))
                    sskip.append(rademacher(output_dim, generator))
                self.aggregate_r = torch.nn.Parameter(torch.stack(ragg))
                self.aggregate_s = torch.nn.Parameter(torch.stack(sagg))
                self.skip_r = torch.nn.Parameter(torch.stack(rskip))
                self.skip_s = torch.nn.Parameter(torch.stack(sskip))

        def _transform(self, x, weight, r, s):
            if r is None:
                multiply = torch.sparse.mm if x.is_sparse else torch.matmul
                return multiply(x, weight)
            if x.is_sparse:
                # Multiply input factors at the STORED sparse feature entries.
                # Native dropout has already been applied once, shared by agg/skip.
                scaled_values = x.values() * r[x.indices()[1]]
                factored_x = torch.sparse_coo_tensor(x.indices(), scaled_values, x.shape,
                                                    device=x.device, is_coalesced=True)
                return torch.sparse.mm(factored_x, weight) * s
            return torch.matmul(x * r, weight) * s

        def forward(self, x, adjacency, degrees, route):
            if self.training:
                if x.is_sparse:
                    if not x.is_coalesced():
                        raise ValueError('Coalesced native sparse input required')
                    dropped_values = F.dropout(x.values(), p=.4, training=True)
                    x = torch.sparse_coo_tensor(x.indices(), dropped_values, x.shape,
                                               device=x.device, is_coalesced=True)
                else:
                    x = F.dropout(x, p=.4, training=True)
            if self.factorized:
                ragg, sagg = self.aggregate_r[route], self.aggregate_s[route]
                rskip, sskip = self.skip_r[route], self.skip_s[route]
            else:
                ragg = sagg = rskip = sskip = None
            transformed = self._transform(x, self.aggregate_weight, ragg, sagg)
            aggregated = torch.sparse.mm(adjacency, transformed) / degrees
            z = aggregated + self._transform(x, self.skip_weight, rskip, sskip) + self.bias[route]
            return clip_rows(z)

        def effective_square(self, route):
            if not self.factorized:
                return self.aggregate_weight.square().sum() + self.skip_weight.square().sum()
            # Full effective-weight penalty; no shared-W-only gauge substitution.
            aggregate = (self.aggregate_r[route].square()[:, None]
                         * self.aggregate_weight.square()
                         * self.aggregate_s[route].square()[None, :]).sum()
            skip = (self.skip_r[route].square()[:, None]
                    * self.skip_weight.square()
                    * self.skip_s[route].square()[None, :]).sum()
            return aggregate + skip

    class Body(torch.nn.Module):
        def __init__(self, members, factorized, seed, first_route):
            super().__init__()
            generators = [torch.Generator(device='cpu').manual_seed(
                block + 281000033 + 1000003 * (first_route + m)) for m in range(members)]
            # Source Glorot laws; CPU-only default generator seeding inside fork.
            # No CUDA/global dropout stream is seeded by this constructor.
            # No factor/feedback draws perturb slow draws.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                self.layers = torch.nn.ModuleList((
                    RouteLayer(FEATURES, HIDDEN, members, generators, factorized),
                    RouteLayer(HIDDEN, CLASSES, members, generators, factorized)))
                # Custody seed/state only; caller qualifies per-route RNG ownership.
                self.post_constructor_cpu_rng = torch.get_rng_state().clone()

        def first(self, x, adjacency, degrees, local_route):
            return F.relu(self.layers[0](x, adjacency, degrees, local_route))

        def last(self, hidden, adjacency, degrees, local_route):
            return self.layers[1](hidden, adjacency, degrees, local_route)

        def regularizer(self, local_route):
            return .05 * self.layers[0].effective_square(local_route)

    class Feedback(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            self.Bmap = torch.nn.Parameter(torch.empty(HIDDEN, RANK))
            self.Vmap = torch.nn.Parameter(torch.zeros(RANK, HIDDEN))
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                torch.nn.init.xavier_uniform_(self.Bmap)

        def forward(self, hub):
            return clip_rows(torch.matmul(F.relu(torch.matmul(hub, self.Bmap)), self.Vmap))

    class Model(torch.nn.Module):
        def __init__(self, members, factorized, ownership, feedback, body_index=0):
            super().__init__()
            self.members, self.ownership, self.feedback = members, ownership, feedback
            if ownership == 'unshared':
                self.bodies = torch.nn.ModuleList([
                    Body(1, factorized, block + 1000003 * m, m) for m in range(members)])
            else:
                self.bodies = torch.nn.ModuleList([
                    Body(members, factorized, block + 1000003 * body_index, body_index)])
            if feedback != 'none':
                feedback_count = members if ownership == 'unshared' else 1
                self.feedback_maps = torch.nn.ModuleList([
                    Feedback(block + 281000039 + 1000003 * (body_index + m))
                    for m in range(feedback_count)])
            else:
                self.feedback_maps = torch.nn.ModuleList([])
            self.body_index = body_index

        def _body(self, route):
            return (self.bodies[route], 0) if self.ownership == 'unshared' else (self.bodies[0], route)

        def _phi(self, route, hub):
            return self.feedback_maps[route if self.ownership == 'unshared' else 0](hub)

        def regularizer(self):
            # Caller uses mean own CE; genuine M1 core has one full own penalty.
            penalties = [self._body(m)[0].regularizer(self._body(m)[1]) for m in range(self.members)]
            return torch.stack(penalties).mean()

        def forward(self, x, adjacency, degrees, C=None, D=None, route_rng_context=None):
            if tuple(x.shape) != (NODES, FEATURES) or tuple(degrees.shape) != (NODES, 1):
                raise ValueError('Frozen full graph input and Nx1 positive native degrees required')
            if tuple(adjacency.shape) != (NODES, NODES):
                raise ValueError('Qualified complete native support required')
            if self.training and route_rng_context is None:
                raise ValueError('TRAIN requires persistent independently owned route RNG context')
            context = route_rng_context if self.training else lambda m: nullcontext()
            hidden = []
            for m in range(self.members):
                body, local_route = self._body(m)
                with context(m):
                    hidden.append(body.first(x, adjacency, degrees, local_route))
            if self.feedback != 'none':
                if C is None or D is None or tuple(C.shape) != (FEATURES, NODES) or tuple(D.shape) != (NODES, FEATURES):
                    raise ValueError('Same fixed raw-X incidence C/D required in all equipped arms')
                hubs = [torch.sparse.mm(C, h) for h in hidden]
                if self.feedback == 'self':
                    received = [self._phi(m, hubs[m]) for m in range(self.members)]
                elif self.feedback == 'mean_then_phi':
                    if self.members != 4 or self.ownership != 'shared':
                        raise ValueError('Frozen SM4 ownership required')
                    received = [self._phi(m, torch.stack([hubs[n] for n in range(4) if n != m]).mean(0))
                                for m in range(4)]
                elif self.feedback == 'phi_then_mean':
                    if self.members != 4:
                        raise ValueError('Frozen peer context requires four routes')
                    donors = [self._phi(m, hubs[m]) for m in range(4)]
                    received = [torch.stack([donors[n] for n in range(4) if n != m]).mean(0)
                                for m in range(4)]
                else:
                    raise ValueError('Frozen feedback operator required')
                # No detach/no_grad/cache: native own CE has full donor credit.
                hidden = [clip_rows(hidden[m] + FEEDBACK_SCALE * torch.sparse.mm(D, received[m]))
                          for m in range(self.members)]
            logits = []
            for m in range(self.members):
                body, local_route = self._body(m)
                with context(m):
                    logits.append(body.last(hidden[m], adjacency, degrees, local_route))
            return torch.stack(logits)

    if spec['independent']:
        models = tuple(Model(1, spec['factors'], 'shared', spec['feedback'], body_index=m) for m in range(4))
    else:
        models = (Model(spec['members'], spec['factors'], spec['ownership'], spec['feedback']),)
    return {'arm': arm, 'block': block, 'models': models, 'genuine_independent': spec['independent'],
            'optimizer_trajectories': 4 if spec['independent'] else 1,
            'physical_backbones': 4 if spec['independent'] or spec['ownership'] == 'unshared' else 1,
            'members': 4 if spec['independent'] else spec['members'],
            'expected_parameters': EXPECTED_ARM_PARAMETERS[arm]}


def inspect_parameter_ownership(core):
    """Deferred runtime inspection; root calls after admitted torch construction.

    Four returned genuine I4 models and four UX4 bodies require disjoint Python
    parameters AND disjoint underlying storage. Count is necessary, not runtime
    or scientific qualification. No tensor payload is exported.
    """
    models = core['models']
    parameters = [p for model in models for p in model.parameters()]
    count = sum(p.numel() for p in parameters)
    if count != core['expected_parameters']:
        raise ValueError('Frozen algebraic parameter count differs')
    owners = list(models) if core['genuine_independent'] else (
        list(models[0].bodies) if models[0].ownership == 'unshared' else [])
    seen_objects, seen_storage = set(), set()
    for owner in owners:
        group = list(owner.parameters())
        objects = {id(p) for p in group}
        storage = {(str(p.device), p.untyped_storage().data_ptr()) for p in group}
        if seen_objects.intersection(objects) or seen_storage.intersection(storage):
            raise ValueError('Distinct body ownership violated')
        seen_objects.update(objects)
        seen_storage.update(storage)
    if not core['genuine_independent'] and models[0].ownership == 'unshared':
        phi_objects, phi_storage = set(), set()
        for phi in models[0].feedback_maps:
            group = list(phi.parameters())
            objects = {id(p) for p in group}
            storage = {(str(p.device), p.untyped_storage().data_ptr()) for p in group}
            if objects & (seen_objects | phi_objects) or storage & (seen_storage | phi_storage):
                raise ValueError('Distinct donor-feedback ownership violated')
            phi_objects.update(objects)
            phi_storage.update(storage)
    return {'parameters': count, 'members': core['members'],
            'physical_backbones': core['physical_backbones'],
            'optimizer_trajectories': core['optimizer_trajectories'],
            'genuine_independent': core['genuine_independent']}
