"""Local custody/scheduler fixtures only; no native work, GPU access or remote calls."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import retry_worker as r


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.auth = copy.deepcopy(r.AUTHORIZED)
        self.auth['expected_fit_count'] = 3
        self.auth_patch = patch.object(r, 'AUTHORIZED', self.auth)
        self.auth_patch.start(); self.addCleanup(self.auth_patch.stop)
        self.worker = r.IsolatedRetryWorker.__new__(r.IsolatedRetryWorker)
        w = self.worker
        w.root = self.root / 'runs'
        w.args = SimpleNamespace(worker=self.auth['worker'], retry_failed=True, poll_seconds=1, smoke_wall_seconds=1800)
        w.states = {}; w.processes = {}; w.retry = set(); w.phase = 'smoke'
        w.retry_disposition = 'not_loaded'
        w.scheduler_source = {'explicit_retry_authorization': {'sha256': 'fixture'}}
        w.plan = {'inventory': {'path': 'fixture-inventory', 'sha256': self.auth['inventory_sha256']},
                  'smoke_source_manifest': {'path': 'fixture-smoke', 'sha256': self.auth['smoke_manifest_sha256']},
                  'project_root': str(self.root)}
        self.tasks = [{'id': task_id, 'kind': 'fit', 'descriptor': {'fixture': task_id}, 'smoke_order': number}
                      for number, task_id in enumerate((self.auth['id'], 'successful-a', 'successful-b'))]
        w.tasks = self.tasks; w.index = {task['id']: task for task in self.tasks}
        w.smokes = {task['id']: {'id': task['id'], 'worker': self.auth['worker'], 'runtime_job_hash': 'runtime',
                                'canonical_runtime_job_hash': 'canonical', 'science_contract_hash': 'science'}
                    for task in self.tasks}
        self.attempt(self.tasks[0], 1, 'failed')
        for task in self.tasks[1:]:
            self.attempt(task, 1, 'PASS')
        original = w.folder(self.tasks[0], 'smoke') / 'attempts/0001'
        self.auth['original_attempt_result_sha256'] = r.c.pin(original / 'ATTEMPT_RESULT.json')['sha256']
        self.auth['original_native_result_sha256'] = r.c.pin(original / 'NATIVE_RESULT.json')['sha256']

    def attempt(self, task, number, status, phase='smoke'):
        w = self.worker
        folder = w.folder(task, phase)
        directory = folder / 'attempts' / f'{number:04d}'
        if phase == 'smoke':
            native = {**w.smokes[task['id']], 'schema': 'all-fit-two-update-smoke-result-v1',
                      'status': status, 'TEST_accessed': False, 'descriptor': task['descriptor'],
                      'inventory': w.plan['inventory'], 'source_manifest': w.plan['smoke_source_manifest'],
                      'producer_manifest': {'sha256': self.auth['native_port_manifest_sha256']},
                      'TRAIN_updates': 2 if status == 'PASS' else 0, 'TRAIN_finite': status == 'PASS',
                      'VALID_finite': status == 'PASS',
                      'optimizer_calls': {'attempts': [2 if status == 'PASS' else 0], 'completed': [2 if status == 'PASS' else 0]},
                      'checkpoint_reload': {'status': 'PASS'},
                      'failure': {'type': 'OutOfMemoryError', 'message': 'CUDA out of memory'}}
        else:
            native = {'schema': 'native-full-epoch-fit-result-v1', 'job_id': task['id'],
                      'job_hash': task['runtime_job_hash'], 'status': status, 'TEST_accessed': False}
        r.c.write(directory / 'NATIVE_RESULT.json', native)
        r.c.write(directory / 'TERMINAL.json', {'exit_code': 1 if status == 'failed' else 0})
        result = {'schema': 'all-fit-attempt-result-v1', 'id': task['id'], 'worker': self.auth['worker'],
                  'phase': phase, 'attempt': number, 'status': status,
                  'native_result': r.c.pin(directory / 'NATIVE_RESULT.json'),
                  'terminal': r.c.pin(directory / 'TERMINAL.json'),
                  'output': str(directory / 'output'), 'exit_code': 1 if status == 'failed' else 0}
        r.c.write(directory / 'ATTEMPT_RESULT.json', result)
        r.c.write(folder / 'STATE.json', {**result, 'attempt_result': r.c.pin(directory / 'ATTEMPT_RESULT.json')})
        return directory

    def retained_bytes(self):
        return {str(path): path.read_bytes() for path in self.root.rglob('*.json')}

    def test_only_authorized_id_pending_in_memory_and_all_evidence_bytes_preserved(self):
        before = self.retained_bytes()
        self.worker.load_states(self.tasks, 'smoke')
        state = self.worker.states[self.auth['id']]
        self.assertEqual((state['status'], state['attempt']), ('pending', 1))
        self.assertEqual(self.retained_bytes(), before)
        disk = r.c.read(self.worker.folder(self.tasks[0], 'smoke') / 'STATE.json')
        self.assertEqual(disk['status'], 'failed')
        self.assertEqual([self.worker.states[t['id']]['status'] for t in self.tasks[1:]], ['PASS', 'PASS'])

    def test_latest_pass_stays_pass_without_mutating_successful_evidence(self):
        self.attempt(self.tasks[0], 2, 'PASS')
        before = self.retained_bytes()
        self.worker.load_states(self.tasks, 'smoke')
        self.assertEqual(self.worker.states[self.auth['id']]['status'], 'PASS')
        self.assertEqual(self.worker.retry_disposition, 'already_PASS')
        self.assertEqual(self.retained_bytes(), before)

    def test_attempt_two_failure_never_retries_even_with_retry_failed(self):
        self.attempt(self.tasks[0], 2, 'failed')
        before = self.retained_bytes()
        with patch.object(self.worker, 'start') as start, patch.object(self.worker, 'report'):
            self.assertFalse(self.worker.run_phase('smoke'))
            start.assert_not_called()
        self.assertEqual(self.worker.states[self.auth['id']]['status'], 'failed')
        self.assertEqual(self.worker.retry, set())
        self.assertEqual(self.retained_bytes(), before)

    def test_parent_start_creates_only_attempt_two_and_preserves_original_receipts(self):
        w = self.worker
        w.load_states(self.tasks, 'smoke')
        before = self.retained_bytes()
        plan_path = self.root / 'plan.json'; r.c.write(plan_path, w.plan)
        w.args.plan = str(plan_path)
        w.config = {'physical_gpu_uuid': 'fixture-gpu'}
        w.blocked_profiles = {}; w.leases = []; w.ledger = self.root / 'ledger'
        w.resources = SimpleNamespace(can_pair=lambda left, right: True)
        task = self.tasks[0]; task['reservation'] = {'profile': 'fixture'}
        environment = {name: 'fixture' for name in ('BENCHMARK_HOST_CONFIG', 'BENCHMARK_HOST_CONFIG_SHA256',
                       'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                       'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')}
        with patch.dict(r.os.environ, environment), patch.object(r.parent, 'probe', return_value={}), \
             patch.object(r.parent, 'capacity_decision', return_value={'admitted': True}), \
             patch.object(r.parent.subprocess, 'Popen', return_value=SimpleNamespace()) as launch:
            self.assertTrue(w.start(task, 'smoke'))
            launch.assert_called_once()
        directory = w.folder(task, 'smoke') / 'attempts/0002'
        self.assertEqual(w.states[task['id']]['attempt'], 2)
        self.assertEqual(w.states[task['id']]['status'], 'running')
        invocation = r.c.read(directory / 'INVOCATION.json')
        self.assertEqual(invocation['argv'][3], str(r.parent.SMOKE / 'run_one.py'))
        self.assertIn('--updates', invocation['argv'])
        self.assertEqual(invocation['argv'][invocation['argv'].index('--updates') + 1], '2')
        self.assertEqual(invocation['explicit_smoke_retry']['maximum_smoke_attempt'], 2)
        self.assertEqual(invocation['actual_scheduler_producer'], w.scheduler_source)
        state_path = str(w.folder(task, 'smoke') / 'STATE.json')
        for path, raw in before.items():
            if path != state_path:
                self.assertEqual(Path(path).read_bytes(), raw)

    def test_wrong_original_receipt_hash_refused(self):
        self.auth['original_attempt_result_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'original attempt receipt changed'):
            self.worker.load_states(self.tasks, 'smoke')

    def test_wrong_native_receipt_hash_refused(self):
        self.auth['original_native_result_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'original native receipt changed'):
            self.worker.load_states(self.tasks, 'smoke')

    def test_wrong_id_and_extra_failed_smoke_refused(self):
        with self.assertRaisesRegex(ValueError, 'exact assigned fit coverage'):
            self.worker.load_states(self.tasks[1:], 'smoke')
        self.attempt(self.tasks[1], 2, 'failed')
        with self.assertRaisesRegex(ValueError, 'every other assigned smoke PASS'):
            self.worker.load_states(self.tasks, 'smoke')

    def test_third_attempt_refused(self):
        self.attempt(self.tasks[0], 3, 'PASS')
        with self.assertRaisesRegex(ValueError, 'identity/attempt differs'):
            self.worker.load_states(self.tasks, 'smoke')

    def test_authorization_hash_and_expanded_scope_refused(self):
        path = self.root / 'authorization.json'
        r.c.write(path, self.auth)
        with self.assertRaisesRegex(ValueError, 'authorization hash differs'):
            r.authorization(path, '0' * 64)
        bad = {**self.auth, 'id': 'different-id'}
        r.c.write(path, bad)
        with self.assertRaisesRegex(ValueError, 'authorization scope differs'):
            r.authorization(path, r.c.pin(path)['sha256'])

    def test_full_state_loading_and_retry_behavior_match_parent(self):
        for task in self.tasks:
            task['runtime_job_hash'] = 'runtime'
            self.attempt(task, 1, 'no_selected_state' if task != self.tasks[0] else 'failed', phase='full')
        self.worker.load_states(self.tasks, 'full')
        wrapper_states = copy.deepcopy(self.worker.states)
        wrapper_retry = self.worker.retry.copy()
        r.parent.Worker.load_states(self.worker, self.tasks, 'full')
        self.assertEqual(self.worker.states, wrapper_states)
        self.assertEqual(self.worker.retry, wrapper_retry)
        self.assertEqual(wrapper_retry, {self.auth['id']})

    def test_full_invocation_only_adds_scheduler_provenance(self):
        task = self.tasks[0]
        dummy = self.root / 'source.txt'; dummy.write_text('fixture\n')
        receipt = r.c.pin(dummy)
        task.update(runtime_entrypoint=receipt, descriptor=receipt, preset=receipt)
        self.worker.plan.update(reporting_entrypoint=receipt)
        self.worker.args.plan = str(dummy)
        self.worker.states = {task['id']: {'status': 'pending', 'attempt': 0}}
        environment = {name: 'fixture' for name in ('BENCHMARK_HOST_CONFIG', 'BENCHMARK_HOST_CONFIG_SHA256',
                       'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                       'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')}
        directory = self.worker.folder(task, 'full') / 'attempts/0001'
        with patch.dict(r.os.environ, environment), patch.object(r.parent.time, 'time', return_value=42):
            original = r.parent.Worker.invocation(self.worker, task, 'full', directory)
            wrapped = self.worker.invocation(task, 'full', directory)
        self.assertEqual(wrapped.pop('actual_scheduler_producer'), self.worker.scheduler_source)
        self.assertEqual(wrapped, original)

    def test_lifecycle_and_native_validation_are_inherited_unchanged(self):
        for name in ('run', 'run_phase', 'start', 'acquire', 'prerequisites', 'collect', 'reap',
                     'save_state', 'validate_result', 'smoke_summary', 'active_ledger'):
            self.assertIs(getattr(r.IsolatedRetryWorker, name), getattr(r.parent.Worker, name))

    def test_exact_owner_bytes_archived_under_supervisor_lease(self):
        owner_path = self.root / 'OWNER.json'
        raw = b'{ "historical": true, "spacing": "must stay" }\n'
        owner_path.write_bytes(raw)
        plan_path = self.root / 'plan.json'
        r.c.write(plan_path, {'state_root': str(self.root)})
        self.auth['plan_sha256'] = r.c.pin(plan_path)['sha256']
        auth_path = self.root / 'auth.json'; r.c.write(auth_path, self.auth)
        args = SimpleNamespace(worker=self.auth['worker'], plan=str(plan_path),
                               plan_sha256=self.auth['plan_sha256'], authorization=str(auth_path),
                               authorization_sha256=r.c.pin(auth_path)['sha256'])
        held = []

        def parent_init(instance, args):
            instance.supervisor_lease = r.c.lock(self.root / 'SUPERVISOR.lock')
            held.append(instance.supervisor_lease)
            self.assertTrue(r.c.locked(self.root / 'SUPERVISOR.lock'))
            r.parent.write(owner_path, {'new_owner': True})

        try:
            with patch.object(r.parent.Worker, '__init__', parent_init):
                instance = r.IsolatedRetryWorker(args, {'sha256': 'wrapper-fixture'})
            owner = r.c.read(owner_path)
            self.assertEqual(Path(owner['previous_owner']['path']).read_bytes(), raw)
            self.assertEqual(owner['actual_scheduler_producer'], instance.scheduler_source)
        finally:
            for handle in held:
                handle.close()


class FrozenProductionTests(unittest.TestCase):
    def test_frozen_source_verifies_and_wrong_hash_refused(self):
        receipt = r.c.pin(r.HERE / 'MANIFEST.json')
        self.assertEqual(r.own_source(receipt['sha256']), receipt)
        with self.assertRaisesRegex(ValueError, 'Exact retry wrapper source required'):
            r.own_source('0' * 64)

    def test_authorization_matches_fixed_production_scope(self):
        path = r.HERE / 'AUTHORIZATION.json'
        self.assertEqual(r.c.read(path), r.AUTHORIZED)
        self.assertEqual(r.authorization(path, r.c.pin(path)['sha256']), r.c.pin(path))

    def test_local_original_evidence_reconstitutes_exact_production_hashes(self):
        path = r.HERE.parent / 'planning/all_fit_launch_20261016_v1/SERVER51_ISOLATED_FAILURE_20261019.json'
        evidence = r.c.read(path)
        for basename, field in (('ATTEMPT_RESULT.json', 'original_attempt_result_sha256'),
                                ('NATIVE_RESULT.json', 'original_native_result_sha256')):
            value = next(value for key, value in evidence.items() if key.endswith('/' + basename))
            raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
            self.assertEqual(r.hashlib.sha256(raw).hexdigest(), r.AUTHORIZED[field])


if __name__ == '__main__':
    unittest.main(verbosity=2)
