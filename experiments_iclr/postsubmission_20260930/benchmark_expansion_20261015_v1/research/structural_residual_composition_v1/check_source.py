"""Stdlib source/contract checks only; no scientific/numerical library import."""
import ast
import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
E, PHASE = HERE.parents[1], HERE.parents[1].parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


extension = load(HERE / 'extension.py', '_structural_residual_source_check')
table = load(E / 'structural_class_responsibility_v1/table.py', '_retained_exact_table_check')
rules = load(PHASE / 'reference_wrapper_redesign_20261013_v1/runtime/training_rules.py', '_retained_old_state_rules_check')


class Parameter:
    def __init__(self, shape, nonzero=0):
        self.shape, self.nonzero, self.requires_grad, self.value, self.grad = shape, nonzero, True, 7, None
    def numel(self):
        result = 1
        for v in self.shape: result *= v
        return result
    def count_nonzero(self): return SimpleNamespace(item=lambda: self.nonzero)


def model_and_optimizer():
    shared = Parameter((10,)); down = Parameter((4, 8, 300), 9600); up = Parameter((4, 300, 8))
    model = SimpleNamespace(native=SimpleNamespace(named_parameters=lambda: (('native_weight', shared),)),
        input_residual=SimpleNamespace(down=down, up=up), private_parameters=lambda: (down, up),
        parameters=lambda: (shared, down, up))
    return model, SimpleNamespace(param_groups=[{'params': [shared, down, up], 'lr': .001}])


class Checks(unittest.TestCase):
    def test_reused_pins_and_no_scientific_descriptors(self):
        self.assertEqual(extension.pin(HERE / 'SOURCES.json')['sha256'], extension.SOURCES_SHA)
        for row in json.loads((HERE / 'SOURCES.json').read_text())['files']:
            actual = extension.pin(PHASE / row['path_relative_to_phase'])
            for key in ('bytes', 'sha256'): self.assertEqual(actual[key], row[key])
        self.assertFalse(list(HERE.glob('JOB*.json')))
        for path in HERE.glob('*.py'): ast.parse(path.read_text())

    def test_in_memory_descriptor_retains_native_parent(self):
        for seed in (101, 202, 303):
            parent = json.loads((extension.PARENT / f'JOB_seed{seed}.json').read_text())
            before = copy.deepcopy(parent)
            for method in extension.METHODS:
                value = extension.descriptor(parent, method, 'a' * 64)
                for key in set(parent) - {'id', 'procedure', 'method_settings', 'session_transform'}:
                    self.assertEqual(value[key], parent[key])
            self.assertEqual(parent, before)
        with self.assertRaises(ValueError): extension.descriptor(parent, 'unlisted', 'a' * 64)

    def test_exact_tied_masses_and_whole_vector_permutation(self):
        ids = list(range(14)); labels = [0] * 7 + [1] * 7
        atoms = [1, 2, 2, 2, 4, 5, 6] * 2
        a = table.make_table(ids, labels, atoms, role='TRAIN')
        b = table.make_table(ids, labels, atoms, role='TRAIN', permutation_seed=101)
        for result in (a, b):
            for i in ids: self.assertEqual(sum(result.weights[m][i] for m in range(4)), 4)
            for label in (0, 1):
                positions = [i for i in ids if labels[i] == label]
                for m in range(4):
                    self.assertEqual(sum(result.q[m][i] for i in positions), Fraction(7, 4))
                    self.assertEqual(sum(result.weights[m][i] for i in positions), 7)
        for i, origin in enumerate(b.permutation):
            self.assertEqual(labels[i], labels[origin])
            self.assertEqual(tuple(b.weights[m][i] for m in range(4)), tuple(a.weights[m][origin] for m in range(4)))
        with self.assertRaises(ValueError): table.make_table(ids, labels, atoms, role='VALID')

    def test_zero_up_and_exhaustive_residual_only_partition(self):
        model, optimizer = model_and_optimizer()
        view = extension.partition(model, optimizer, require_zero_up=True)
        self.assertEqual(tuple(view.private_parameters()), (model.input_residual.down, model.input_residual.up))
        model.input_residual.up.nonzero = 1
        with self.assertRaises(ValueError): extension.partition(model, optimizer, require_zero_up=True)
        extension.partition(model, optimizer)  # trained nonzero state remains restorable
        model, optimizer = model_and_optimizer(); optimizer.param_groups[0]['params'].pop()
        with self.assertRaises(ValueError): extension.partition(model, optimizer)
        model, optimizer = model_and_optimizer(); optimizer.param_groups[0]['params'].append(model.input_residual.up)
        with self.assertRaises(ValueError): extension.partition(model, optimizer)

    def test_actual_updater_wiring_uses_same_old_state_then_one_step(self):
        model, optimizer = model_and_optimizer(); view = extension.partition(model, optimizer)
        events = []
        class Scalar:
            ndim = 0
            def __init__(self, name): self.name = name
        class Gradient:
            is_sparse = False
            def __init__(self, name): self.name = name
        class Autograd:
            def grad(self, loss, parameters, **kw):
                self_test.assertTrue(all(p.value == 7 and p.grad is None for p in model.parameters()))
                events.append((loss.name, tuple(map(id, parameters)), kw))
                return tuple(Gradient(loss.name) for p in parameters)
        self_test = self
        finite = SimpleNamespace(item=lambda: True)
        finite.all = lambda: finite
        fake = SimpleNamespace(Tensor=Scalar, isfinite=lambda x: finite, autograd=Autograd())
        def step():
            self.assertEqual([p.grad.name for p in model.parameters()], ['F', 'W', 'W'])
            self.assertEqual(len(events), 2)
            self.assertTrue(all(p.value == 7 for p in model.parameters()))
            events.append('step')
        optimizer.step = step
        result = rules.step_private_responsibility(fake, view, optimizer, F=Scalar('F'), F_weighted=Scalar('W'))
        self.assertEqual([e[0] for e in events[:2]], ['F', 'W'])
        self.assertEqual(events[0][1], (id(model.native.named_parameters()[0][1]),))
        self.assertEqual(events[1][1], tuple(map(id, model.private_parameters())))
        self.assertTrue(events[0][2]['retain_graph']); self.assertFalse(events[1][2]['retain_graph'])
        self.assertTrue(result['all_gradients_from_old_state']); self.assertEqual(result['optimizer_transitions'], 1)

    def test_forward_and_loss_paths_are_finite_and_native(self):
        tree = ast.parse((HERE / 'extension.py').read_text())
        transform = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'transform')
        update = next(n for n in transform.body if isinstance(n, ast.FunctionDef) and n.name == 'train_update')
        code = ast.get_source_segment((HERE / 'extension.py').read_text(), update)
        self.assertEqual(code.count('run_member('), 1)
        self.assertIn('range(4)', code); self.assertEqual(code.count('step_private_responsibility('), 1)
        self.assertNotIn('.backward(', code); self.assertNotIn('.step(', code)
        terms = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_weighted_terms')
        code = ast.get_source_segment((HERE / 'extension.py').read_text(), terms)
        self.assertIn('session._native_loss(route[0], batch)', code)
        self.assertIn("reduction='none'", code); self.assertIn('native_reduce=lambda row: row.mean()', code)
        self.assertIn('route[0].to(t.float32)[observed]', code)
        self.assertNotIn('softmax', code)


if __name__ == '__main__': unittest.main()
