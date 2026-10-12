"""Scheduling/startup checks with no model, CUDA or scientific jobs."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_startup


class GPUAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['GNNM_STARTUP_TEST_ROOT'])
        self.root = Path(self.temp.name)
        launcher = Path(os.environ['GNNM_STARTUP_LAUNCHER'])
        tree = ast.parse(launcher.read_text())
        selected = {'enabled_workers', 'request_gpu_admission', 'schedule_ready', 'detach_host'}
        self.ns = {'Path': Path, 'json': json, 'os': os,
                   'canonical': lambda value: json.dumps(value, sort_keys=True),
                   '__file__': str(launcher), 'fcntl': __import__('fcntl')}
        module = ast.Module(body=[node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name in selected], type_ignores=[])
        exec(compile(module, str(launcher), 'exec'), self.ns)
        self.rows = [{'index': 0}, {'index': 1}]

    def tearDown(self):
        self.temp.cleanup()

    def test_legacy_host_and_single_gpu_request(self):
        enabled = self.ns['enabled_workers']
        request = self.ns['request_gpu_admission']
        self.assertEqual(enabled(self.root, 'server77', 'p', self.rows), {'server77/gpu0', 'server77/gpu1'})
        self.assertEqual(request(self.root, 'server77', 'p', self.rows, 0), ['server77/gpu0'])
        self.assertEqual(enabled(self.root, 'server77', 'p', self.rows), {'server77/gpu0'})

    def test_second_gpu_and_repeated_request_are_idempotent(self):
        request = self.ns['request_gpu_admission']
        request(self.root, 'server77', 'p', self.rows, 0)
        request(self.root, 'server77', 'p', self.rows, 1)
        self.assertEqual(request(self.root, 'server77', 'p', self.rows, 1), ['server77/gpu0', 'server77/gpu1'])

    def test_unknown_gpu_or_changed_plan_cannot_change_request(self):
        request = self.ns['request_gpu_admission']
        request(self.root, 'server77', 'p', self.rows, 0)
        path = self.root / 'server77.gpu-admission.json'
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            request(self.root, 'server77', 'p', self.rows, 2)
        with self.assertRaises(ValueError):
            request(self.root, 'server77', 'other', self.rows, 1)
        self.assertEqual(path.read_bytes(), before)

    def test_two_slots_only_on_enabled_gpu(self):
        runs = {key: {'run': {'kind': 'model_fit', 'config': {'reservation_mib': 10}}}
                for key in ('a', 'b', 'c', 'd')}
        placement = {key: {'worker': 'server77/gpu' + ('1' if key == 'd' else '0')} for key in runs}
        ready = self.ns['schedule_ready'](runs, {key: 'registered' for key in runs}, placement,
                {key: [] for key in runs}, [], {'server77/gpu0': 100}, 'server77')
        self.assertEqual(ready, ['a', 'b'])

    def fake_detach(self, owner):
        output = self.root / 'outputs'
        output.mkdir()
        (output / 'server77.startup-owner.json').write_text(json.dumps(owner))
        self.ns.update(load_plan=Mock(return_value=({'output_root': 'outputs'}, 'p', {}, {}, {}, {})),
            host_check=Mock(return_value=self.rows), receipt_authentication=Mock(return_value=(self.root, {})),
            confined=lambda root, relative: root / relative,
            live_owned=lambda identity: True, all_global_gates_closed=lambda *args: False,
            local_terminal_roster=lambda *args: None,
            runtime_preflight=Mock(side_effect=AssertionError('No runtime child should start')),
            subprocess=Mock())
        return output

    def test_second_command_enables_gpu_without_second_supervisor(self):
        owner = {'host': 'server77', 'plan_hash': 'p', 'supervisor': {'pid': 1}, 'rendezvous': {'pid': 2}}
        output = self.fake_detach(owner)
        self.ns['request_gpu_admission'](output, 'server77', 'p', self.rows, 0)
        value = self.ns['detach_host'](self.root, self.root / 'plan', self.root / 'go', 'server77', Path('/python'), gpu_index=1)
        self.assertEqual(value['status'], 'already_running')
        self.assertEqual(self.ns['enabled_workers'](output, 'server77', 'p', self.rows), {'server77/gpu0', 'server77/gpu1'})
        self.ns['subprocess'].Popen.assert_not_called()

    def test_partial_startup_keeps_gpu_requests_unchanged(self):
        output = self.fake_detach({'host': 'server77', 'plan_hash': 'p', 'supervisor': {'pid': 1}})
        self.ns['request_gpu_admission'](output, 'server77', 'p', self.rows, 0)
        before = (output / 'server77.gpu-admission.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Partial startup'):
            self.ns['detach_host'](self.root, self.root / 'plan', self.root / 'go', 'server77', Path('/python'), gpu_index=1)
        self.assertEqual((output / 'server77.gpu-admission.json').read_bytes(), before)
        self.ns['subprocess'].Popen.assert_not_called()

    def test_go_validation_precedes_output_creation(self):
        self.ns['load_plan'] = Mock(side_effect=ValueError('Explicit GO required'))
        with self.assertRaisesRegex(ValueError, 'Explicit GO'):
            self.ns['detach_host'](self.root, self.root / 'plan', self.root / 'go', 'server77', Path('/python'), gpu_index=0)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
