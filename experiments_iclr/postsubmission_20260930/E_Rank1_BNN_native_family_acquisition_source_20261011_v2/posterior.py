"""Native FactorLinear Gaussian rows; caller supplies admitted Torch/Family.

Only fresh native sites are attached. Registered r/s remain the actual means.
Draws are temporary autograd tensors, never parameter assignments or buffers.
"""
from contextlib import contextmanager
import hashlib
import math
import types

INITIAL_RHO = math.log(.1)


def require(value, message):
    if not value:
        raise ValueError(message)


def _forward(module, x):
    effective = module._rank1_effective
    if effective is None:
        r, s = module.r[module.member], module.s[module.member]
    else:
        require(module.member in effective, 'One complete draw for the active original route')
        r, s = effective[module.member]
    y = module._rank1_torch.nn.functional.linear(x * r, module.weight) * s
    return y if module.bias is None else y + module.bias


class Rows:
    def __init__(self, family, body, members, law, delta):
        require(law in ('fixed_noise', 'learned_q'), 'Only frozen Gaussian laws')
        t = family.torch
        require(all(p.device.type == 'cpu' and p.dtype == t.float32 for p in body.parameters()), 'Attach fresh CPU native sites before AdamW')
        rng = t.get_rng_state().clone()
        self.torch, self.delta, self.members, self.law = t, delta, members, law
        self.sites = {name: module for name, module in body.named_modules()
                      if isinstance(module, family.factors.FactorLinear)}
        require(self.sites and all(m.r.shape[0] == m.s.shape[0] == members for m in self.sites.values()), 'Every actual native/builder factor row')
        self.centres = {}
        for name in sorted(self.sites):
            module = self.sites[name]
            require(not hasattr(module, 'r_log_std'), 'Fresh sites only; no trained donor')
            for vector in ('r', 's'):
                mean = getattr(module, vector)
                self.centres[name + '.' + vector] = mean.detach().clone()
                # Original make enrolls all parameters and immediately checks
                # ownership. Frozen-noise scales become frozen only after that
                # unchanged check, before any model operation or state exists.
                module.register_parameter(vector + '_log_std', t.nn.Parameter(t.full_like(mean, INITIAL_RHO)))
            object.__setattr__(module, '_rank1_torch', t)
            object.__setattr__(module, '_rank1_effective', None)
            module.forward = types.MethodType(_forward, module)
        require(t.equal(t.get_rng_state(), rng), 'Scale registration and prior capture draw no randomness')

    def move_centres(self, device):
        self.centres = {key: value.to(device).detach() for key, value in self.centres.items()}

    def vectors(self, route):
        require(0 <= route < self.members, 'Original complete route identity')
        means, scales, centres = {}, {}, {}
        for name, module in self.sites.items():
            for vector in ('r', 's'):
                key = name + '.' + vector
                means[key] = getattr(module, vector)[route]
                scales[key] = getattr(module, vector + '_log_std')[route]
                centres[key] = self.centres[key][route]
        return means, scales, centres

    def draw(self, route, generator):
        means, scales, _ = self.vectors(route)
        result = self.delta.draw_route(self.torch, means, scales, generator)
        require(all(bool(self.torch.isfinite(value).all()) for value in result.values()), 'Finite unclipped Gaussian factors')
        return result

    def KL(self, route):
        value = self.delta.conditional_factor_KL(self.torch, *self.vectors(route))
        require(bool(self.torch.isfinite(value)), 'Finite exact conditional-component Gaussian KL')
        return value

    @contextmanager
    def effective(self, route_draws):
        require(route_draws and all(0 <= route < self.members for route in route_draws), 'Declared complete route draws')
        versions = {(name, v): getattr(module, v)._version for name, module in self.sites.items() for v in ('r', 's')}
        require(all(module._rank1_effective is None for module in self.sites.values()), 'No nested or leaked sampled-factor scope')
        try:
            for name, module in self.sites.items():
                object.__setattr__(module, '_rank1_effective', {
                    route: (draw[name + '.r'], draw[name + '.s']) for route, draw in route_draws.items()})
            yield
        finally:
            for module in self.sites.values(): object.__setattr__(module, '_rank1_effective', None)
            require(versions == {(name, v): getattr(module, v)._version for name, module in self.sites.items() for v in ('r', 's')},
                    'Sampling never mutates registered means')

    def mean_scope_closed(self):
        require(all(m._rank1_effective is None for m in self.sites.values()), 'Mean selector/restore consumes no draws')
        if self.law == 'fixed_noise':
            require(all(not getattr(m, v + '_log_std').requires_grad for m in self.sites.values() for v in ('r', 's')),
                    'Fixed-noise scales frozen before any draw/forward/update')
            require(all(self.torch.equal(getattr(m, v + '_log_std'), self.torch.full_like(getattr(m, v + '_log_std'), INITIAL_RHO))
                        for m in self.sites.values() for v in ('r', 's')), 'Fixed .1 scales remain frozen')

    def checkpoint(self):
        self.mean_scope_closed()
        return dict(law=self.law, members=self.members, prior_std=.1, initial_q_std=.1,
                    prior_centres={key: value.detach().cpu().clone() for key, value in self.centres.items()},
                    site_order=sorted(self.sites), factor_coordinates_per_route=sum(m.r.shape[1] + m.s.shape[1] for m in self.sites.values()))

    def restore_constants(self, value):
        require(value['law'] == self.law and value['members'] == self.members and value['prior_std'] == value['initial_q_std'] == .1
                and value['site_order'] == sorted(self.sites) and set(value['prior_centres']) == set(self.centres), 'Exact selected law/sites/external prior constants')
        require(all(self.torch.equal(self.centres[key].cpu(), prior) for key, prior in value['prior_centres'].items()),
                'Prior centres equal fresh original data-free native initialization')
        self.mean_scope_closed()

    def summary(self):
        scale = [getattr(m, v + '_log_std').detach().exp().cpu().reshape(-1) for m in self.sites.values() for v in ('r', 's')]
        scale = self.torch.cat(scale)
        require(bool(self.torch.isfinite(scale).all()) and bool((scale > 0).all()), 'Finite positive unclipped scales')
        return dict(law=self.law, sites=sorted(self.sites), std_min=float(scale.min()), std_max=float(scale.max()),
                    std_mean=float(scale.mean()), scale_coordinates=scale.numel(), trainable_scales=self.law == 'learned_q',
                    means_mutated_to_install_draw=False, prior_constants_registered_as_buffers=False)


def bind(family, delta):
    """Reuse original native_body; attach only fresh Gaussian procedure instances."""
    require(not getattr(family, '_rank1_bound', False), 'One fresh family binding')
    original = family.native_body
    family._rank1_law, family._rank1_rows = 'point', None
    def native_body(self, seed, native_member, factor_member, factorized, members):
        body, wrapper = original(seed, native_member, factor_member, factorized, members)
        self._rank1_rows = None if self._rank1_law == 'point' else Rows(self, body, members, self._rank1_law, delta)
        return body, wrapper
    family.native_body = types.MethodType(native_body, family)
    family._rank1_bound = True
    return family


def generators(torch, device, seed, member, members):
    return [torch.Generator(device=device).manual_seed(seed + 5000011 + 1000003 * (member + route)) for route in range(members)]


def rng_hashes(generators):
    return [hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest() for g in generators]
