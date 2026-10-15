"""Only stdlib, source/JSON reads and scalar/mock checks; never imports torch.

The stand-ins below test production control flow and scalar derivatives. They
do not establish Torch autograd, native model, data, CUDA or fit qualification.
"""
import array
import ast
import copy
import json
import math
from pathlib import Path
import random
import sys
import types

import extension as ex


class State:
    def __init__(self, value): self.value = copy.deepcopy(value)
    def clone(self): return State(self.value)
    def cpu(self): return self


class Draw:
    def __init__(self, values): self.values = array.array('f', values)
    def numpy(self): return self.values
    def to(self, **kwargs): return self


class Generator:
    def __init__(self, *, device):
        assert device == 'cpu'
        self.rng = random.Random(0)
    def manual_seed(self, seed): self.rng.seed(seed); return self
    def get_state(self): return State(self.rng.getstate())
    def set_state(self, state): self.rng.setstate(copy.deepcopy(state.value)); return self


class MockTorch:
    Generator, float32 = Generator, 'float32'
    @staticmethod
    def randn(rank, width, *, generator, device, dtype):
        assert (rank, width, device, dtype) == (8, 300, 'cpu', 'float32')
        return Draw(generator.rng.gauss(0, 1) for _ in range(rank * width))


class Dual:
    """One-direction scalar dual, used only to check exact production algebra."""
    def __init__(self, v, d=0.): self.v, self.d = float(v), float(d)
    @staticmethod
    def of(x): return x if isinstance(x, Dual) else Dual(x)
    def __add__(self, other):
        o = self.of(other); return Dual(self.v + o.v, self.d + o.d)
    __radd__ = __add__
    def __neg__(self): return Dual(-self.v, -self.d)
    def __sub__(self, other): return self + -self.of(other)
    def __rsub__(self, other): return self.of(other) - self
    def __mul__(self, other):
        o = self.of(other); return Dual(self.v * o.v, self.d * o.v + self.v * o.d)
    __rmul__ = __mul__
    def __truediv__(self, other): return self * self.of(other) ** -1
    def __pow__(self, n): return Dual(self.v ** n, n * self.v ** (n - 1) * self.d)
    def exp(self): return Dual(math.exp(self.v), math.exp(self.v) * self.d)
    def expm1(self): return Dual(math.expm1(self.v), math.exp(self.v) * self.d)


class Functional:
    @staticmethod
    def linear(x, weight): return x * weight
    @staticmethod
    def gelu(x):
        cdf = .5 * (1 + math.erf(x.v / math.sqrt(2)))
        return Dual(x.v * cdf, x.d * (cdf + x.v * math.exp(-.5 * x.v * x.v) / math.sqrt(2 * math.pi)))


def rejected(callback):
    try: callback()
    except ValueError: return
    raise AssertionError('Expected a lifecycle/identity rejection')


def main():
    assert 'torch' not in sys.modules
    ex.verify_sources(ex.PHASE.parent, data=False)
    for path in ex.HERE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    noise = ex.load(ex.HERE / 'noise.py')
    global_before = random.getstate()
    streams = noise.NoiseStreams(MockTorch, 101)
    reference = types.SimpleNamespace(device='cpu', dtype='float32')
    rejected(lambda: streams.for_route(0, reference))
    before = streams.state()
    predicted = streams.preview_digest()
    assert all(x.value == y.value for x, y in zip(before['states'], streams.state()['states']))
    streams.begin()
    assert streams.last_digest == predicted and streams.counts == [1] * 4
    rejected(streams.begin)
    rejected(streams.state)
    route_draws = [streams.for_route(m, reference) for m in range(4)]
    assert streams.for_route(0, reference) is route_draws[0]  # Same V draw for all atoms/graphs.
    assert len({noise.NoiseStreams.digest([d]) for d in route_draws}) == 4
    streams.finish(complete=True)
    saved = streams.state()
    preview = streams.preview_digest()
    fresh = noise.NoiseStreams(MockTorch, 101)
    fresh.load_state(saved, 1)
    assert fresh.preview_digest() == preview
    fresh.begin()
    assert fresh.last_digest == preview
    for m in range(4): fresh.for_route(m, reference)
    fresh.finish(complete=True)
    assert saved['counts'] == [1] * 4 and fresh.counts == [2] * 4
    assert all(x.value == y.value for x, y in zip(saved['states'], streams.state()['states']))
    rejected(lambda: noise.NoiseStreams(MockTorch, 202).load_state(saved, 1))
    rejected(lambda: streams.load_state(saved, 2))
    broken = noise.NoiseStreams(MockTorch, 101)
    broken.begin()
    broken.for_route(0, reference)
    rejected(lambda: broken.finish(complete=True))
    assert broken.draws is None and random.getstate() == global_before

    # Derivatives of the actual scalar kernels, not a second formula-only implementation.
    for active in ('mu', 'rho', 'up'):
        def forward(up_value):
            mu, rho, up = Dual(.2, active == 'mu'), Dual(math.log(noise.S0), active == 'rho'), Dual(up_value, active == 'up')
            down = noise.reparameterize(mu, rho, .7)
            return noise.residual_value(Functional, Dual(1.3), down, up)
        at_zero, moved = forward(0), forward(.1)
        assert at_zero.v == 1.3
        assert (at_zero.d != 0) == (active == 'up')
        assert moved.d != 0
    assert noise.kl_coordinates(Dual(0, 1), Dual(0)).v == 0
    assert noise.kl_coordinates(Dual(0, 1), Dual(0)).d == 0
    assert noise.kl_coordinates(Dual(0), Dual(0, 1)).d == 0
    for delta, logratio in ((.5, -.2), (-.3, .4), (0., 1e-7)):
        value = noise.kl_coordinates(Dual(delta, 1), Dual(logratio))
        oracle = .5 * (delta ** 2 + math.exp(2 * logratio) - 1 - 2 * logratio)
        assert math.isclose(value.v, oracle, abs_tol=1e-15) and value.d == delta
        scale = noise.kl_coordinates(Dual(delta), Dual(logratio, 1))
        assert math.isclose(scale.d, math.expm1(2 * logratio), abs_tol=1e-15)
    # Fixed scale reduces to exactly the mean-anchor quadratic; bank denominator
    # occurs once. This scalar count checks 4*2400 coordinate sums, not means.
    assert sum([2400 * .5] * 4) / (4 * 1210) == 1200 / 1210
    tree = ast.parse((ex.HERE / 'extension.py').read_text())
    update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'train_update')
    calls = [n.func.attr for n in ast.walk(update) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.count('backward') == calls.count('step') == 1
    assert 'penalty = per_route_KL.sum() / (4 * 1210)' in ast.unparse(update)
    assert 'loss = own + penalty' in ast.unparse(update)
    forward = next(n for n in ast.walk(ast.parse((ex.HERE / 'noise.py').read_text()))
                   if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    condition = next(n for n in forward.body if isinstance(n, ast.If))
    assert ast.unparse(condition.test) == 'self.training'
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'for_route'
                   for statement in condition.orelse for n in ast.walk(statement))
    for seed in (101, 202, 303):
        parent = json.loads((ex.PARENT / ('JOB_seed' + str(seed) + '.json')).read_text())
        for arm in ex.ARMS:
            job = ex.descriptor(seed, arm)
            assert job == json.loads((ex.HERE / ('JOB_seed' + str(seed) + '_' + arm + '.json')).read_text())
            assert {k: v for k, v in job.items() if k not in ('id', 'procedure', 'method_settings', 'session_transform')} == {
                k: v for k, v in parent.items() if k not in ('id', 'procedure', 'method_settings', 'session_transform')}
    assert 'torch' not in sys.modules
    report = {'status': 'passed', 'scope': 'stdlib source/descriptor and scalar/mock semantics only',
              'checks': ['source hashes and syntax', 'six exact parent-preserving descriptors',
                         'one scalar J/backward/optimizer transition', 'KL coordinate sum and bank denominator',
                         'scalar production-kernel derivatives at U0 and after U moves', 'q=p KL zero value/derivatives',
                         'four separate mock noise streams', 'coherent reuse within route',
                         'save/fresh-restore next-draw replay and snapshot non-aliasing',
                         'wrong seed/count and nested/partial lifecycle rejection',
                         'mean eval source branch has no draws', 'stdlib global RNG untouched'],
              'torch_imported': False, 'model_constructions': 0, 'data_payload_reads': 0, 'native_forwards': 0,
              'optimizer_updates': 0, 'GPU_execution': False, 'actual_runtime_qualification': 'unexecuted'}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__': main()
