"""Offline exact-row/invocation metadata tests; no native or remote work."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import run_diagnostic as d


class DiagnosticTests(unittest.TestCase):
    def test_preserves_exact_row_except_actual_entrypoint(self):
        original = {'fits': [{'id': 'other'}], 'counts': {'old': 2}, 'source_manifest': {'sha256': 'old'}}
        row = {'id': d.FIT_ID, 'worker': d.WORKER, 'descriptor': {'path': '/exact', 'sha256': 'unchanged'},
               'runtime_job_hash': 'same', 'runtime_entrypoint': {'path': '/old'}}
        old = copy.deepcopy(row)
        result = d.diagnostic_inventory(original, row, {'sha256': 'new'}, {'sha256': 'port'}, {'path': '/actual'})
        self.assertEqual(row, old)
        expected = copy.deepcopy(old); expected['runtime_entrypoint'] = {'path': '/actual'}
        self.assertEqual(result['fits'], [expected])
        self.assertEqual(result['counts'], {d.WORKER: 1})
        self.assertFalse(result['global_smoke_coverage_eligible'])
        self.assertFalse(result['full_training_admitted'])
        self.assertEqual(original['fits'], [{'id': 'other'}])

    def test_refuses_any_other_fit_or_worker(self):
        for row in ({'id': 'other', 'worker': d.WORKER}, {'id': d.FIT_ID, 'worker': 'server51/gpu0'}):
            with self.assertRaises(ValueError):
                d.diagnostic_inventory({}, row, {}, {}, {})

    def test_actual_native_cli_and_full_phase_refusal(self):
        worker = d.DiagnosticWorker.__new__(d.DiagnosticWorker); worker.actual_source = {'diagnostic': True}
        request = {'argv': ['python', '-B', '-u', str(d.c.SMOKE / 'run_one.py'), '--project-root', '/root'], 'environment': {}}
        with patch.object(d.Worker, 'invocation', return_value=copy.deepcopy(request)):
            result = worker.invocation({'id': d.FIT_ID}, 'smoke', Path('/attempt'))
        self.assertEqual(result['argv'][3], str(d.SMOKE / 'run_one.py'))
        self.assertEqual(result['argv'][4:], request['argv'][4:])
        self.assertEqual(result['environment']['BENCHMARK_HOST_CONFIG'], str(d.PORT / 'hosts/server51_gpu1.json'))
        self.assertFalse(result['global_smoke_coverage_eligible'])
        with self.assertRaises(ValueError):
            worker.invocation({'id': d.FIT_ID}, 'full', Path('/attempt'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
