"""Narrow batch-coherent down-weight noise; torch supplied by the native session.

No imports of tensor libraries, model/data construction, or global RNG use.
CPU FP32 generators make the noise streams separate from native CPU/CUDA RNG.
"""
import hashlib
import types

MEMBERS, RANK, WIDTH, NTRAIN = 4, 8, 300, 1210
S0 = 0.1 / (300 ** 0.5)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reparameterize(mu, rho, epsilon):
    return mu + rho.exp() * epsilon


def residual_value(functional, encoded, down, up):
    return encoded + functional.linear(functional.gelu(functional.linear(encoded, down)), up)


def kl_coordinates(standardized_mean_delta, log_scale_ratio):
    # expm1 avoids subtracting near-equal variance terms. This is the exact
    # Gaussian KL algebra, with zero value AND derivative at q=p.
    twice = 2 * log_scale_ratio
    return 0.5 * (standardized_mean_delta ** 2 + twice.expm1() - twice)


class NoiseStreams:
    def __init__(self, torch, seed):
        require(type(seed) is int and seed in (101, 202, 303), 'Frozen cohort seed required')
        self.torch = torch
        self.seeds = [seed + 170000003 + 104729 * m for m in range(MEMBERS)]
        self.generators = [torch.Generator(device='cpu').manual_seed(s) for s in self.seeds]
        self.counts = [0] * MEMBERS
        self.draws = None
        self.used = set()
        self.last_digest = None

    def _draw(self, generator):
        return self.torch.randn(RANK, WIDTH, generator=generator, device='cpu', dtype=self.torch.float32)

    @staticmethod
    def digest(draws):
        return hashlib.sha256(b''.join(x.numpy().tobytes() for x in draws)).hexdigest()

    def begin(self):
        require(self.draws is None, 'One nonnested eligible native TRAIN update required')
        self.draws = [self._draw(g) for g in self.generators]
        self.counts = [n + 1 for n in self.counts]
        self.used = set()
        self.last_digest = self.digest(self.draws)

    def for_route(self, member, reference):
        require(type(member) is int and 0 <= member < MEMBERS and self.draws is not None,
                'Noise is available only inside an eligible native TRAIN update')
        self.used.add(member)
        return self.draws[member].to(device=reference.device, dtype=reference.dtype)

    def finish(self, *, complete):
        try:
            require(self.draws is not None, 'No active noise draw')
            if complete:
                require(self.used == set(range(MEMBERS)), 'All four native routes must consume their draw')
        finally:
            self.draws, self.used = None, set()

    def state(self):
        require(self.draws is None, 'Save noise only between native updates')
        return {'schema': 'four-private-CPU-noise-streams-v1', 'seeds': list(self.seeds),
                'counts': list(self.counts), 'states': [g.get_state().clone() for g in self.generators]}

    def load_state(self, value, updates):
        require(self.draws is None and value.get('schema') == 'four-private-CPU-noise-streams-v1'
                and value.get('seeds') == self.seeds and value.get('counts') == [updates] * MEMBERS
                and len(value.get('states', [])) == MEMBERS, 'Exact independent noise checkpoint required')
        # Validate all states on temporary generators before altering live ones.
        for seed, state in zip(self.seeds, value['states']):
            self.torch.Generator(device='cpu').manual_seed(seed).set_state(state.cpu().clone())
        for generator, state in zip(self.generators, value['states']):
            generator.set_state(state.cpu().clone())
        self.counts, self.last_digest = list(value['counts']), None

    def preview_digest(self):
        """Qualification-only next-draw oracle; no live stream/global RNG advance."""
        require(self.draws is None, 'Preview only between updates')
        clones = [self.torch.Generator(device='cpu').set_state(g.get_state().clone()) for g in self.generators]
        return self.digest([self._draw(g) for g in clones])


def attach(torch, residual, optimizer, *, arm, seed):
    require(arm in ('fixed', 'learned') and tuple(residual.down.shape) == (MEMBERS, RANK, WIDTH)
            and tuple(residual.up.shape) == (MEMBERS, WIDTH, RANK), 'Original rank8 residual required')
    require(not hasattr(residual, 'prior_down'), 'Attach stochastic law once')
    mu, up = residual.down, residual.up  # Preserve paid-control Parameter identities and names.
    residual.register_buffer('prior_down', mu.detach().clone())
    residual.register_buffer('prior_log_sigma', mu.new_tensor(S0).log())
    center, log_sigma = residual.prior_down.detach().cpu().clone(), residual.prior_log_sigma.detach().cpu().clone()
    if arm == 'learned':
        residual.register_parameter('rho', torch.nn.Parameter(torch.full_like(mu, float(log_sigma))))
        owners = [g for g in optimizer.param_groups if any(p is mu for p in g['params'])]
        require(len(owners) == 1 and any(p is up for p in owners[0]['params']), 'Original adapter Adam group required')
        owners[0]['params'].append(residual.rho)
    streams = NoiseStreams(torch, seed)

    def identity():
        require(residual.down is mu and residual.up is up
                and not residual.prior_down.requires_grad and not residual.prior_log_sigma.requires_grad
                and torch.equal(residual.prior_down.detach().cpu(), center)
                and torch.equal(residual.prior_log_sigma.detach().cpu(), log_sigma),
                'Original means/up and immutable data-free prior centers required')

    def forward(self, encoded, member):
        require(type(member) is int and 0 <= member < MEMBERS and encoded.shape[-1] == WIDTH,
                'Original full AtomEncoder width and route required')
        down = self.down[member]
        if self.training:
            epsilon = streams.for_route(member, down)
            rho = self.rho[member] if arm == 'learned' else self.prior_log_sigma
            down = reparameterize(down, rho, epsilon)
        else:
            require(streams.draws is None, 'Mean serving cannot be nested inside a noise update')
        return residual_value(torch.nn.functional, encoded, down, self.up[member])

    def kl_routes():
        delta = (mu - residual.prior_down) / residual.prior_log_sigma.exp()
        coordinates = (0.5 * delta ** 2 if arm == 'fixed' else
                       kl_coordinates(delta, residual.rho - residual.prior_log_sigma))
        return coordinates.reshape(MEMBERS, -1).sum(dim=1)

    residual.forward = types.MethodType(forward, residual)
    identity()
    return streams, kl_routes, identity
