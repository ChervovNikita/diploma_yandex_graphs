"""Canonical boundary hooks preserving native affine classes/Parameters/BN."""
from contextlib import contextmanager
from functools import wraps
import copy
import random


class TaskCallScope:
    """Track native task/recursive calls without changing their operator body.

    NCNC's original forward calls multidomainforward for both its outer task
    and recursive completion. Wrapping that one dispatch point distinguishes
    those calls in TRAIN and no-grad evaluation alike.
    """
    def __init__(self):
        self.depth = 0
        self.dispatch_counts = {'task': 0, 'recursive': 0}
        self.output_counts_by_route = {str(route): dict.fromkeys(
            ('task_applied', 'task_bypassed', 'recursive_applied', 'recursive_bypassed',
             'outside_task_applied', 'outside_task_bypassed'), 0) for route in range(4)}

    def wrap(self, native_method):
        @wraps(native_method)
        def scoped(*args, **kwargs):
            self.depth += 1
            self.dispatch_counts['task' if self.depth == 1 else 'recursive'] += 1
            try:
                return native_method(*args, **kwargs)
            finally:
                self.depth -= 1
        return scoped

    def task_output_active(self):
        return self.depth == 1

    def observe_output(self, route, applied):
        kind = 'task' if self.depth == 1 else 'recursive' if self.depth > 1 else 'outside_task'
        self.output_counts_by_route[str(route)][kind + ('_applied' if applied else '_bypassed')] += 1

    def capture(self):
        return copy.deepcopy({'depth': self.depth, 'dispatch_counts': self.dispatch_counts,
                              'output_counts_by_route': self.output_counts_by_route})

    def restore(self, state):
        if int(state['depth']) != 0:
            raise ValueError('NCNC checkpoint cannot be captured during a task call')
        self.depth = 0
        self.dispatch_counts = copy.deepcopy(state['dispatch_counts'])
        self.output_counts_by_route = copy.deepcopy(state['output_counts_by_route'])

    def routing_since(self, before):
        dispatch = {key: value - before['dispatch_counts'][key]
                    for key, value in self.dispatch_counts.items()}
        routes = {route: {key: value - before['output_counts_by_route'][route][key]
                          for key, value in counts.items()}
                  for route, counts in self.output_counts_by_route.items()}
        totals = {key: sum(counts[key] for counts in routes.values())
                  for key in self.output_counts_by_route['0']}
        passed = (self.depth == 0 and dispatch['task'] > 0 and dispatch['recursive'] > 0 and
                  totals['task_applied'] == dispatch['task'] and
                  totals['recursive_bypassed'] == dispatch['recursive'] and
                  all(totals[key] == 0 for key in ('task_bypassed', 'recursive_applied',
                                                   'outside_task_applied', 'outside_task_bypassed')))
        if not passed:
            raise ValueError('NCNC output-boundary task scope was violated: ' + str(totals))
        return {'dispatch_counts': dispatch, 'final_affine_calls_by_route': routes,
                'final_affine_call_totals': totals, 'task_only_output_scope_verified': True,
                'grad_enabled_status_used_for_scope': False}


def make_boundary_core(torch, model, sites, seed):
    class Bank(torch.nn.Module):
        def __init__(self, source, first, generator):
            super().__init__()
            out_dim, in_dim = source.weight.shape
            r = torch.ones(4, in_dim, dtype=source.weight.dtype)
            if first:
                r = torch.randint(0, 2, r.shape, generator=generator).to(r.dtype).mul_(2).sub_(1)
            self.R = torch.nn.Parameter(r.to(source.weight.device))
            self.S = torch.nn.Parameter(torch.ones(4, out_dim, dtype=source.weight.dtype, device=source.weight.device))
            self.delta_bias = torch.nn.Parameter(torch.zeros(4, out_dim, dtype=source.weight.dtype, device=source.weight.device))

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.route = 0
            self._source_parameters = tuple(model.named_parameters())
            self.banks = torch.nn.ModuleDict()
            self.binding_metadata = []
            self._handles = []
            generator = torch.Generator(device='cpu').manual_seed(int(seed) + 9000011)
            for i, site in enumerate(sites):
                name, module, first = site[:3]
                active = site[3] if len(site) > 3 else None
                observer = site[4] if len(site) > 4 else None
                if module.weight.ndim != 2 or not hasattr(module, 'bias'):
                    raise ValueError('A native learned affine leaf is required: ' + name)
                bank = Bank(module, first, generator)
                key = 'site_' + str(i)
                self.banks[key] = bank
                self.binding_metadata.append({'site': name, 'native_class': type(module).__module__ + '.' + type(module).__name__,
                    'first_input': first, 'source_weight_id': id(module.weight),
                    'source_bias_id': id(module.bias) if module.bias is not None else None,
                    'native_module_and_parameters_preserved': True,
                    'R_initialization': 'Rademacher' if first else 'ones', 'S_initialization': 'ones',
                    'delta_bias_initialization': 'zero',
                    'affine_law': 'S * native_affine(R * x) + delta_bias',
                    'application_scope': 'top_level_native_task_call' if active else 'native_boundary_call'})

                def pre(source, args, bank=bank, active=active):
                    if active is not None and not active():
                        return None
                    return (args[0] * bank.R[self.route],) + tuple(args[1:])

                def post(source, args, output, bank=bank, active=active, observer=observer):
                    enabled = active is None or active()
                    if observer is not None:
                        observer(self.route, enabled)
                    if not enabled:
                        return output
                    return output * bank.S[self.route] + bank.delta_bias[self.route]

                self._handles.append(module.register_forward_pre_hook(pre))
                self._handles.append(module.register_forward_hook(post))

        def private_parameters(self):
            return tuple(p for bank in self.banks.values() for p in bank.parameters())

        @contextmanager
        def member(self, route):
            old = self.route
            self.route = int(route)
            try:
                yield
            finally:
                self.route = old

    return Core()


class PrivateRNG:
    """Four persistent Python/NumPy/Torch streams, serialized by the session."""
    def __init__(self, torch, seed, members):
        import numpy as np
        self.t, self.np = torch, np
        saved = self.capture()
        self.streams = []
        for member in range(members):
            value = int(seed) + 900003 + 1009 * member
            random.seed(value); np.random.seed(value); torch.manual_seed(value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(value)
            self.streams.append(self.capture())
        self.restore(saved)

    def capture(self):
        t = self.t
        ns = self.np.random.get_state()
        primitive_numpy = (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4])
        return {'python': random.getstate(), 'numpy': primitive_numpy,
                'torch_cpu': t.get_rng_state().clone(),
                'torch_cuda': [x.clone() for x in t.cuda.get_rng_state_all()] if t.cuda.is_available() else []}

    def restore(self, value):
        random.setstate(value['python'])
        ns = value['numpy']
        self.np.random.set_state((ns[0], self.np.asarray(ns[1], dtype=self.np.uint32), ns[2], ns[3], ns[4]))
        self.t.set_rng_state(value['torch_cpu'].cpu())
        if value['torch_cuda']:
            self.t.cuda.set_rng_state_all([x.cpu() for x in value['torch_cuda']])

    @contextmanager
    def member(self, member):
        outside = self.capture()
        self.restore(self.streams[member])
        try:
            yield
        finally:
            self.streams[member] = self.capture()
            self.restore(outside)
