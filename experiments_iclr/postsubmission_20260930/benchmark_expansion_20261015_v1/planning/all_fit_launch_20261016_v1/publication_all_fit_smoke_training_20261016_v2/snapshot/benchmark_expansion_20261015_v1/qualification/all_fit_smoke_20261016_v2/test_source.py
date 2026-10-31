"""Scalar control/coverage regressions; no Torch, NumPy, models or data imports."""
import ast
import copy
import math
from pathlib import Path
import unittest
from smoke_control import install_step_cap, train_two_updates
from support import CANONICAL_SHA, COUNTS, PORT, digest
from validate import serialized_pin, validate_results


def observer_class(runtime_folder):
    path = PORT / runtime_folder / 'run_fit.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OptimizerObserver')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['OptimizerObserver']


class Optimizer:
    param_groups = []

    def __init__(self):
        self.calls = 0

    def step(self):
        self.calls += 1


class Session:
    def __init__(self, optimizer, loss=1.0, maps=1):
        self.optimizer, self.loss, self.maps = optimizer, loss, maps
        self.calls = 0

    def train_update(self):
        self.calls += 1
        for _ in range(self.maps):
            self.optimizer.step()
        return {'loss': self.loss}


class Adapter:
    completed_epochs = 0

    def __init__(self, session, epoch_maps):
        self.session, self.epoch_maps = session, epoch_maps

    def train_epoch(self):
        for _ in range(self.epoch_maps):
            self.session.train_update()
        self.completed_epochs += 1


class SourceTests(unittest.TestCase):
    def test_actual_ogb_epoch_cleanup(self):
        path = PORT / 'runtime_node_molecular/adapters.py'
        tree = ast.parse(path.read_text())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OGBAdapter')
        function = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'train_epoch')
        exception = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EpochComplete')
        namespace = {'math': math}
        exec(compile(ast.Module(body=[exception, function], type_ignores=[]), str(path), 'exec'), namespace)
        optimizer = Optimizer()
        adapter = Adapter(Session(optimizer), 38)
        adapter._iterator = None
        adapter._invalidate_validation = lambda: None
        adapter.train_loader = range(38)
        adapter.train_epoch = namespace['train_epoch'].__get__(adapter)
        observer = observer_class('runtime_node_molecular')(None, [optimizer])
        install_step_cap(observer, [optimizer])
        train_two_updates(adapter, observer, {})
        self.assertIsNone(adapter._iterator)
        self.assertEqual(adapter.completed_epochs, 0)
        self.assertEqual(optimizer.calls, 2)
        observer.close()

    def test_real_observer_signatures_and_native_loop_shapes(self):
        for folder in ('runtime_node_molecular', 'runtime_link', 'composed/runtime'):
            for maps in (1, 17, 38, 1585):
                with self.subTest(folder=folder, maps=maps):
                    optimizer = Optimizer()
                    session = Session(optimizer)
                    adapter = Adapter(session, maps)
                    observer = observer_class(folder)(None, [optimizer])
                    install_step_cap(observer, [optimizer])
                    report = {}
                    train_two_updates(adapter, observer, report)
                    self.assertEqual(optimizer.calls, 2)
                    self.assertEqual(session.calls, 2)
                    self.assertEqual(report['TRAIN_losses'], [1.0, 1.0])
                    self.assertTrue(report['TRAIN_finite'])
                    self.assertEqual(report['TRAIN_boundary_probes'], 0 if maps == 1 else 1)
                    with self.assertRaises(ValueError):
                        optimizer.step()
                    self.assertEqual(optimizer.calls, 2)
                    observer.close()

    def test_failure_costs_preserved(self):
        for loss, maps in ((float('nan'), 1), (1.0, 0), (1.0, 2), (1.0, 3)):
            with self.subTest(loss=loss, maps=maps):
                optimizer = Optimizer()
                session = Session(optimizer, loss, maps)
                observer = observer_class('runtime_node_molecular')(None, [optimizer])
                install_step_cap(observer, [optimizer])
                with self.assertRaises(ValueError):
                    train_two_updates(Adapter(session, 38), observer, {})
                self.assertLessEqual(optimizer.calls, 2)
                self.assertEqual(observer.state()['completed'], [optimizer.calls])
                observer.close()

    @staticmethod
    def coverage_fixture():
        universe = [f'scalar-fixture-{i:04d}' for i in range(6460)]
        source_sha = 'a' * 64
        source = {'path': '/fixture/MANIFEST.json', 'bytes': 1, 'sha256': source_sha}
        items = []
        offset = 0
        for worker, count in COUNTS.items():
            ids = universe[offset:offset + count]
            offset += count
            rows, results = [], []
            inventory_pin = {'path': '/fixture/' + worker + '/inventory.json', 'bytes': count, 'sha256': digest(worker)}
            for fid in ids:
                receipt = {'path': '/fixture/' + fid + '.json', 'bytes': 1, 'sha256': digest(fid)}
                row = {'id': fid, 'worker': worker, 'runtime_job_hash': digest(fid), 'canonical_runtime_job_hash': digest(fid),
                       'science_contract_hash': digest(fid), 'descriptor': receipt, 'runtime_entrypoint': source}
                result = {**row, 'schema': 'all-fit-two-update-smoke-result-v1', 'status': 'PASS', 'inventory': inventory_pin,
                    'source_manifest': source, 'optimizer_calls': {'attempts': [2], 'completed': [2]}, 'TRAIN_updates': 2,
                    'TRAIN_finite': True, 'TRAIN_losses': [1.0, 0.9], 'VALID_finite': True, 'VALID_calls': 1,
                    'VALID': {'role': 'VALID', 'complete_source_VALID': True, 'records': 3, 'loss': 0.5, 'metrics': {'scalar': 0.5}},
                    'checkpoint_reload': {'status': 'PASS', 'exact_adapter_state_restored': True, 'parameter_perturbation_restored': True,
                        'optimizer_state_reloaded': True, 'optimizer_moment_perturbation_restored': True, 'TRAIN_updates_after_reload': 0},
                    'TEST_accessed': False, 'scientific_checkpoint_reused': False, 'scientific_fit_policy_changed': False, 'automatic_retries': 0}
                rows.append(row)
                results.append({'id': fid, 'runtime_job_hash': digest(fid), 'attempt': 1, 'result': result,
                                'result_receipt': {'path': '/fixture/' + fid + '/RESULT.json', **serialized_pin(result)}})
            inventory = {'schema': 'all-fit-smoke-inventory-v1', 'source_manifest': source, 'canonical_plan': {'sha256': CANONICAL_SHA},
                'class_substitution': False, 'canonical_fit_ids': universe, 'canonical_fit_ids_sha256': digest(universe), 'fits': rows}
            summary = {'schema': 'all-fit-worker-smoke-summary-v1', 'status': 'PASS', 'worker': worker, 'inventory': inventory_pin,
                'expected_fit_ids': ids, 'results': results, 'counts': {'expected': count, 'passed': count}}
            items.append({'worker': worker, 'inventory': inventory_pin, 'inventory_value': inventory,
                          'summary': {'path': '/fixture/' + worker + '/SUMMARY.json', **serialized_pin(summary)}, 'summary_value': summary})
        return items, source_sha

    def test_all_6460_scalar_coverage_and_missing_or_stale_refusal(self):
        items, source_sha = self.coverage_fixture()
        result = validate_results(items, source_sha)
        self.assertEqual(result['fit_count'], 6460)
        self.assertEqual(result['TRAIN_updates_completed_total'], 12920)
        for item in items:
            summary = item['summary_value']
            summary['schema'] = 'allfit-worker-smoke-summary-v1'
            summary['fit_count'] = summary.pop('counts')['passed']
            summary['TEST_accessed'] = False
        self.assertEqual(validate_results(items, source_sha)['fit_count'], 6460)
        original = items[0]['summary_value']['results'][0]
        for field, bad in [('status', 'running'), ('TRAIN_updates', 1), ('VALID_finite', False)]:
            altered = copy.deepcopy(original)
            altered['result'][field] = bad
            altered['result_receipt'].update(serialized_pin(altered['result']))
            items[0]['summary_value']['results'][0] = altered
            with self.assertRaises(ValueError):
                validate_results(items, source_sha)
        items[0]['summary_value']['results'][0] = original
        items[0]['summary_value']['results'].pop()
        with self.assertRaises(ValueError):
            validate_results(items, source_sha)


if __name__ == '__main__':
    unittest.main()
