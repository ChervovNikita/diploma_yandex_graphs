"""Canonical boundary hooks preserving native affine classes/Parameters/BN."""
from contextlib import contextmanager
import random


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
            for i, (name, module, first) in enumerate(sites):
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
                    'delta_bias_initialization': 'zero'})

                def pre(source, args, bank=bank):
                    return (args[0] * bank.R[self.route],) + tuple(args[1:])

                def post(source, args, output, bank=bank):
                    s = bank.S[self.route]
                    y = output * s
                    if source.bias is not None:
                        y = y + source.bias * (1 - s)
                    return y + bank.delta_bias[self.route]

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
