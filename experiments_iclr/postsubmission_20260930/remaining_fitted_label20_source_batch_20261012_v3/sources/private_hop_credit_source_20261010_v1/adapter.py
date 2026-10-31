"""Disabled source delta for pinned native Sessions; no worker or data loader.

The owner must review and explicitly change ENABLED before numerical use.
Only stdlib imports occur at module import. No numerical execution is admitted.
"""
from contextlib import contextmanager
from types import SimpleNamespace

ENABLED = False
LAMBDA = 0.5
M4 = (
    'private_missinghop', 'full_aux', 'common_nonfull', 'allblock_missinghop',
)
M1 = ('factor1_native', 'factor1_allview')
NONFULL = (1, 2, 3)


def _enabled():
    if not ENABLED:
        raise RuntimeError('Disabled source protocol: numerical owner review required')


def prepare_factor1(session):
    """Upgrade one fresh pinned single_native Session before its first update.

    Original native parameters survive; one unit R/S row at all23 existing
    sites; rebuild the unchanged native name-based Adam grouping once.
    This is a competent factorized single, with no ensemble divisor.
    """
    _enabled()
    if (session.name != 'single_native' or len(session.bodies) != 1 or
            session.decoder is not None or session.counters['updates'] != 0 or
            any(optimizer.state for optimizer in session.optimizers)):
        raise ValueError('One fresh exact pinned factual single Session required')
    body = session.bodies[0]
    original = dict(body.named_parameters())
    sites = session.factors.install_factors(body, 1)
    body.to(session.device)
    named = dict(body.named_parameters())
    if sites != 23 or any(named[name] is not p for name, p in original.items()):
        raise ValueError('Preserve all original parameter objects at all23 sites')
    session.optimizers = [session._optimizer(body)]
    return session


class HopCredit:
    """Only token views, first-order gradient routing and work counters change.

    Supply the hash-bound V3 shared4_own Session for M4, or its single_native
    Session upgraded by prepare_factor1 for M1. Existing factual preprocessing,
    native constructor, all TRAIN labels, Adam and serving remain owner-bound.
    """
    def __init__(self, session, condition):
        _enabled()
        import torch
        self.torch = torch
        self.session, self.condition = session, condition
        self.members = 4 if condition in M4 else 1
        if condition not in M4 + M1:
            raise ValueError('One fixed source condition required')
        if (len(session.bodies) != 1 or len(session.optimizers) != 1 or
                session.decoder is not None or session.spec['masked'] or
                session.counters['updates'] != 0 or session.optimizers[0].state):
            raise ValueError('Fresh factual-only pinned Session and one native Adam required')
        expected_name = 'shared4_own' if self.members == 4 else 'single_native'
        if session.name != expected_name:
            raise ValueError('Exact native base condition required')
        self.body = session.bodies[0]
        maps = [m for m in self.body.modules() if isinstance(m, session.factors.FactorLinear)]
        if len(maps) != 23 or any(m.r.shape[0] != self.members or m.s.shape[0] != self.members for m in maps):
            raise ValueError('All23 existing factor sites and exact row count required')
        if any(not torch.equal(p.detach(), torch.ones_like(p)) for m in maps for p in (m.r, m.s)):
            raise ValueError('Existing unit factor initializer required')
        self.private = tuple(p for m in maps for p in (m.r, m.s))
        private_ids = {id(p) for p in self.private}
        self.parameters = tuple(self.body.parameters())
        if len(private_ids) != 46 or sum(p.numel() for p in self.private) != 13943*self.members:
            raise ValueError('Exact native rank-one coordinates required')
        if sum(p.numel() for p in self.parameters) != 2069875 + 13943*self.members:
            raise ValueError('Exact complete native body required')
        self.is_private = tuple(id(p) in private_ids for p in self.parameters)
        optimized = [p for group in session.optimizers[0].param_groups for p in group['params']]
        if len(optimized) != len(self.parameters) or {id(p) for p in optimized} != {id(p) for p in self.parameters}:
            raise ValueError('Native Adam owns each complete body coordinate once')
        for (name, p), group in zip(self.body.named_parameters(), session.optimizers[0].param_groups):
            attention = 'attnmodule' in name
            if (group['params'] != [p] or group['lr'] != (.0005 if attention else .005) or
                    group['weight_decay'] != (1e-8 if attention else .001) or
                    group['eps'] != 1e-8 or group['betas'] != (.9, .999)):
                raise ValueError('Unchanged native Adam grouping and defaults required')
        if list(self.body.named_buffers()):
            raise ValueError('Pinned LayerNorm provider has no registered mutable buffers')
        full = tuple(session.factual.list_mat)
        if len(full) != 3 or self.body.K != 3 or any(t.shape != full[0].shape for t in full):
            raise ValueError('Ordered [X,PX,P²X] K=2 interface required')
        if full[0] is not session.x:
            raise ValueError('Native raw X must remain slot0')
        zero = torch.zeros_like(full[1])
        # Persistent slots are retained. Native biased maps can make them nonzero.
        self.views = (
            session.factual,
            SimpleNamespace(list_mat=(full[0], zero, full[2])),
            SimpleNamespace(list_mat=(full[0], full[1], zero)),
            SimpleNamespace(list_mat=(full[0], zero, zero)),
        )
        self.counters = dict(updates=0, factual_forwards=0, auxiliary_forwards=0,
                             gradient_calls=0, optimizer_steps=0)
        self._pending = None

    @contextmanager
    def _guard_plain_state(self):
        """FactorLinear.member and unregistered PolyAttn.bias survive each call."""
        torch = self.torch
        factors = [m for m in self.body.modules() if isinstance(m, self.session.factors.FactorLinear)]
        members = [(m, m.member) for m in factors]
        biases = [(m, m.bias, m.bias.detach().clone()) for m in self.body.modules()
                  if type(m).__name__ == 'PolyAttn']
        if len(biases) != 2 or list(self.body.named_buffers()):
            raise ValueError('Exact native plain-state inventory required')
        try:
            yield
        finally:
            changed = False
            for module, prior in members:
                changed |= module.member != prior
                module.member = prior
            for module, prior, saved in biases:
                attr_changed = module.bias is not prior
                value_changed = not torch.equal(prior, saved)
                changed |= attr_changed or value_changed
                if attr_changed:
                    module.bias = prior
                if value_changed:
                    with torch.no_grad():
                        prior.copy_(saved)
            if changed:
                raise RuntimeError('Native mutable state changed; restored and update rejected')

    def _forward(self, route, view, auxiliary):
        session = self.session
        stream = 'masked' if auxiliary else 'factual'  # Existing isolated RNG keys.
        with self._guard_plain_state():
            with session.factors.member_context(self.body, route):
                with session.streams.use(route if self.members == 4 else view, stream):
                    logits = self.body(self.views[view])
        if logits.shape != (19717, 3) or not self.torch.isfinite(logits).all():
            raise FloatingPointError('Complete finite native logits required')
        key = 'auxiliary_forwards' if auxiliary else 'factual_forwards'
        self.counters[key] += 1
        session.counters['masked_forwards' if auxiliary else key] += 1
        return logits

    def _gradient(self, logits, parameters):
        torch = self.torch
        ce = torch.nn.functional.cross_entropy(logits[self.session.train_ids], self.session.train_y)
        if not torch.isfinite(ce):
            raise FloatingPointError('Complete TRAIN CE required')
        value = torch.autograd.grad(ce, parameters, create_graph=False,
                                   retain_graph=False, allow_unused=False)
        self.counters['gradient_calls'] += 1
        self.session.counters['backwards'] += 1
        return value, float(ce.detach())

    def gather(self):
        """Gather every cotangent at one parameter state; no optimizer transition."""
        _enabled()
        if self._pending is not None:
            raise RuntimeError('Commit the complete existing gather before another update')
        torch, session = self.torch, self.session
        before = dict(self.counters)
        self.body.train()
        session.optimizers[0].zero_grad(set_to_none=True)
        total = [torch.zeros_like(p) for p in self.parameters]
        position = {id(p): i for i, p in enumerate(self.parameters)}
        factual_ce, auxiliary_ce = [], []
        full_aux = self.condition == 'full_aux'
        allblock = self.condition == 'allblock_missinghop'
        allview = self.condition == 'factor1_allview'
        for route in range(self.members):
            grad, ce = self._gradient(self._forward(route, 0, False), self.parameters)
            factual_ce.append(ce)
            for i, value in enumerate(grad):
                scale = 1/self.members
                if self.members == 4 and route != 0 and (self.is_private[i] or allblock):
                    scale /= 1 + LAMBDA
                if allview and self.is_private[i]:
                    # Full auxiliary view is exactly this factual derivative.
                    scale = (1 + LAMBDA/4)/(1 + LAMBDA)
                total[i].add_(value, alpha=scale)
        # Aux calls use separate RNG states and cannot advance factual streams.
        factual_states = {key: {name: value.clone() if value is not None else None
                          for name, value in state.items()}
                          for key, state in session.streams.states.items() if key[1] == 'factual'}
        routes = NONFULL if self.members == 4 or allview else ()
        for slot in routes:
            route = slot if self.members == 4 else 0
            view = 0 if full_aux else (1 + self.counters['updates'] % 3 if self.condition == 'common_nonfull' else slot)
            targets = self.parameters if allblock else self.private
            grad, ce = self._gradient(self._forward(route, view, True), targets)
            auxiliary_ce.append(ce)
            scale = LAMBDA/(4*(1 + LAMBDA))
            for p, value in zip(targets, grad):
                total[position[id(p)]].add_(value, alpha=scale)
        for key, state in factual_states.items():
            for name, value in state.items():
                current = session.streams.states[key][name]
                if (value is None) != (current is None) or (value is not None and not torch.equal(value, current)):
                    raise RuntimeError('Auxiliary call advanced a factual RNG stream')
        if any(not torch.isfinite(value).all() for value in total):
            raise FloatingPointError('Gathered first-order gradients must be finite')
        expected_aux = 3 if self.members == 4 or allview else 0
        if (self.counters['factual_forwards'] - before['factual_forwards'] != self.members or
                self.counters['auxiliary_forwards'] - before['auxiliary_forwards'] != expected_aux or
                self.counters['gradient_calls'] - before['gradient_calls'] != self.members + expected_aux or
                self.counters['optimizer_steps'] != before['optimizer_steps']):
            raise RuntimeError('Exact path/gradient count before the first optimizer transition required')
        self._pending = (total, dict(factual_ce=factual_ce, auxiliary_ce=auxiliary_ce,
                         route0_full_derivative_reused=self.members == 4,
                         whole_TRAIN_supervision=True))
        return self._pending

    def commit(self, gathered):
        """One existing native Adam after the complete gather; no new clipping."""
        _enabled()
        if gathered is not self._pending or gathered is None:
            raise RuntimeError('One complete outstanding old-state gather required')
        gradients, metrics = gathered
        self._pending = None
        for p, grad in zip(self.parameters, gradients):
            p.grad = grad
        self.session.optimizers[0].step()
        self.counters['optimizer_steps'] += 1
        self.counters['updates'] += 1
        self.session.counters['optimizer_steps'] += 1
        self.session.counters['updates'] += 1
        if any(not self.torch.isfinite(p).all() for p in self.parameters):
            raise FloatingPointError('Native parameter state after one Adam transition')
        if any(isinstance(value, self.torch.Tensor) and not self.torch.isfinite(value).all()
               for state in self.session.optimizers[0].state.values() for value in state.values()):
            raise FloatingPointError('Native Adam state after one transition')
        return metrics

    def step(self):
        return self.commit(self.gather())


def independent_factor1_step(adapters):
    """Genuine I4: four complete disjoint factored bodies and native Adams.

    Each has unscaled M1 credit; never divide its own gradient by four.
    Provide four separately constructed/seeded factor1 adapters and keep four
    own VALID selectors/checkpoints in the external owner. No pooled selector.
    factor1_native costs4 paths; factor1_allview costs16 paths per cohort step.
    """
    _enabled()
    if len(adapters) != 4 or any(a.condition not in M1 or a.members != 1 for a in adapters):
        raise ValueError('Four capable separately factored M1 adapters required')
    if len({a.condition for a in adapters}) != 1:
        raise ValueError('One declared I4 learning rule required')
    if (len({id(a.body) for a in adapters}) != 4 or
            len({id(a.session.streams) for a in adapters}) != 4 or
            len({a.session.seed for a in adapters}) != 4):
        raise ValueError('Four separately constructed/seeded bodies and RNG streams required')
    if len({id(a.session.optimizers[0]) for a in adapters}) != 4:
        raise ValueError('Four independent native optimizers required')
    identities = [{id(p) for p in a.parameters} for a in adapters]
    if any(identities[a] & identities[b] for a in range(4) for b in range(a)):
        raise ValueError('Every complete native/factor parameter must be disjoint')
    gathered = [a.gather() for a in adapters]
    return [a.commit(g) for a, g in zip(adapters, gathered)]
