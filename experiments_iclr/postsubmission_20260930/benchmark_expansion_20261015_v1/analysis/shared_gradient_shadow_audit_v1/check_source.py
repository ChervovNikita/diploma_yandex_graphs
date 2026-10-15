"""Stdlib-only partition, accounting, gate and source checks; no model import."""
import ast
import builtins
import importlib.util
import json
from pathlib import Path
import symtable
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('private_BN_shadow_source_checks', HERE / 'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Checks(unittest.TestCase):
    def test_identity_partition(self):
        a, b, c = object(), object(), object()
        named = [('native.weight', a), ('input.private', b), ('native.bn.weight', c)]
        self.assertEqual(audit.partition(named, [('native.weight', a), ('native.bn.weight', c)], [b], [a, b, c]),
                         ([0, 2], [1]))
        for shared, private, owners in [([('a', a)], [a, b, c], [a, b, c]),
                                        ([('a', a)], [b], [a, b, c]),
                                        ([('a', a), ('c', c)], [b], [a, b, b, c])]:
            with self.assertRaises(ValueError):
                audit.partition(named, shared, private, owners)

    def test_success_and_failure_costs(self):
        ledger = audit.Ledger()
        for key, bound in audit.BOUNDS.items():
            for _ in range(bound):
                ledger.start(key)
                ledger.finish(key)
            with self.assertRaises(RuntimeError):
                ledger.start(key)
        self.assertTrue(ledger.complete())
        failed = audit.Ledger()
        failed.start('optimizer')
        self.assertEqual(failed.counts['optimizer'], {'attempts': 1, 'completed': 0})
        self.assertFalse(failed.complete())
        with self.assertRaises(RuntimeError):
            failed.finish('native_reverse')

    def test_exact_work(self):
        inputs = json.loads((HERE / 'INPUTS.json').read_text())
        self.assertEqual([c['seed'] for c in inputs['cases']], [101, 202, 303])
        self.assertEqual([c['selected_epoch'] for c in inputs['cases']], [92, 54, 116])
        b = inputs['bounds']
        self.assertEqual(3 * audit.BOUNDS['optimizer'], b['temporary_Adam_maps'])
        self.assertEqual(3 * (audit.BOUNDS['native_reverse'] + audit.BOUNDS['route_reverse']), 570)
        self.assertEqual(3 * sum(n for k, n in audit.BOUNDS.items() if k.endswith('_forward')), b['total_route_forwards'])
        self.assertEqual(b['total_route_forwards'], 1380)
        self.assertEqual(b['VALID_calls'] + b['TEST_calls'] + b['retained_trained_models'], 0)

    def test_fixed_gate_and_complete_denominator(self):
        rows = [{'batch': b, 'member': m, 'predicted_joint_change': 0., 'finite_shared_after_private_change': 0.,
                 'finite_joint_change': 0., 'predicted_shared_change': 0., 'tau': .0001}
                for b in range(1, 39) for m in range(4)]
        for row in rows[:16]:
            row.update(predicted_joint_change=.001, finite_shared_after_private_change=.001,
                       finite_joint_change=.001)
        self.assertTrue(audit.gate(rows)['seed_gate_open'])
        rows[15]['finite_joint_change'] = .0001
        self.assertFalse(audit.gate(rows)['seed_gate_open'])
        with self.assertRaises(ValueError):
            audit.gate(rows[:-1])

    def test_native_step_loop_and_no_exports(self):
        source = (HERE / 'audit.py').read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        original_steps = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'original_step']
        self.assertEqual(len(original_steps), 1)
        self.assertEqual(len([n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'train_epoch']), 1)
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr in ('validate', 'save', 'atomic_torch') for n in calls))
        top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(n) or 'numpy' in ast.unparse(n) for n in top_imports))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        preflight = next(i for i, n in enumerate(main.body) if 'preflight(args.root' in ast.unparse(n))
        torch_import = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Import) and n.names[0].name == 'torch')
        self.assertLess(preflight, torch_import)
        assigned = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        self.assertFalse(any('.train_update =' in s or 'checkpoint[' in s.split('=')[0] for s in assigned))

    def test_no_unresolved_global_symbols(self):
        source = (HERE / 'audit.py').read_text()
        table = symtable.symtable(source, 'audit.py', 'exec')
        defined = set(table.get_identifiers()) | set(dir(builtins))
        missing = []
        def visit(t):
            for symbol in t.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in defined:
                    missing.append((t.get_name(), symbol.get_name()))
            for child in t.get_children():
                visit(child)
        visit(table)
        self.assertEqual(missing, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
