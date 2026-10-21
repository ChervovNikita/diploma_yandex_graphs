"""Focused local receipt/admission fixtures; no native fits, GPUs or network."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import queue_worker as q

c = q.c
GiB = c.GiB


class SafeguardTests(unittest.TestCase):
    def i4_fixture(self, root):
        instance = q.LaunchSafeguardsWorker.__new__(q.LaunchSafeguardsWorker)
        instance.root = root
        instance.args = SimpleNamespace(retry_failed=False, worker='allocation/gpu0')
        instance.plan = {'project_root': str(root)}
        instance.states = {}
        task = {'id': 'I4', 'kind': 'aggregate_I4_exports', 'dependencies': ['body0', 'body1', 'body2', 'body3']}
        instance.index = {key: {'runtime_job_hash': 'hash-' + key} for key in task['dependencies']}
        directory = instance.folder(task, 'full') / 'attempts/0001'
        output = directory / 'output'
        output.mkdir(parents=True)
        bank = output / 'I4_selected_VALID.pt'
        bank.write_bytes(b'fixture-bank')
        value = {'schema': 'independent-own-selected-VALID-bank-result-v1', 'status': 'complete',
                 'members': [{'job_id': key, 'job_hash': instance.index[key]['runtime_job_hash']}
                             for key in task['dependencies']],
                 'bank_receipt': c.pin(bank), 'TEST_accessed': False}
        return instance, task, directory, output, value

    def test_i4_receipt_requires_actual_file_bytes_hash_and_own_output(self):
        with tempfile.TemporaryDirectory(dir=q.HERE) as temp:
            instance, task, directory, output, value = self.i4_fixture(Path(temp))
            self.assertEqual(instance.validate_result(task, 'full', value, output), ('complete', None))
            for key in ('path', 'bytes', 'sha256'):
                altered = copy.deepcopy(value)
                del altered['bank_receipt'][key]
                with self.subTest(missing_field=key), self.assertRaisesRegex(ValueError, 'requires path, bytes and SHA256'):
                    instance.validate_result(task, 'full', altered, output)
            altered = copy.deepcopy(value)
            del altered['bank_receipt']
            with self.assertRaisesRegex(ValueError, 'requires path, bytes and SHA256'):
                instance.validate_result(task, 'full', altered, output)
            for key, replacement in (('bytes', value['bank_receipt']['bytes'] + 1), ('sha256', '0' * 64)):
                altered = copy.deepcopy(value)
                altered['bank_receipt'][key] = replacement
                with self.subTest(changed_field=key), self.assertRaisesRegex(ValueError, 'Changed receipt'):
                    instance.validate_result(task, 'full', altered, output)
            other = directory / 'other.pt'
            other.write_bytes(b'fixture-bank')
            altered = dict(value, bank_receipt=c.pin(other))
            with self.assertRaisesRegex(ValueError, 'escapes own output'):
                instance.validate_result(task, 'full', altered, output)
            bank = output / 'I4_selected_VALID.pt'
            bank.unlink()
            with self.assertRaises(FileNotFoundError):
                instance.validate_result(task, 'full', value, output)
            bank.symlink_to(other)
            with self.assertRaisesRegex(ValueError, 'escapes own output'):
                instance.validate_result(task, 'full', value, output)

    def test_missing_i4_bank_cannot_be_collected_as_complete(self):
        with tempfile.TemporaryDirectory(dir=q.HERE) as temp:
            instance, task, directory, output, value = self.i4_fixture(Path(temp))
            c.write(directory / 'INVOCATION.json', {'output': str(output)})
            c.write(directory / 'TERMINAL.json', {'exit_code': 0})
            c.write(output / 'RESULT.json', value)
            (output / 'I4_selected_VALID.pt').unlink()
            instance.collect(task, 'full', directory)
            self.assertEqual(instance.states['I4']['status'], 'failed')
            self.assertIn('FileNotFoundError', instance.states['I4']['failure'])
            self.assertEqual(c.read(directory / 'NATIVE_RESULT.json'), value)
            self.assertEqual(c.read(output / 'RESULT.json'), value)

    def test_i4_restart_rechecks_bank_and_preserves_prior_attempt(self):
        with tempfile.TemporaryDirectory(dir=q.HERE) as temp:
            instance, task, directory, output, value = self.i4_fixture(Path(temp))
            native = directory / 'NATIVE_RESULT.json'
            c.write(native, value)
            result = directory / 'ATTEMPT_RESULT.json'
            c.write(result, {'status': 'complete', 'native_result': c.pin(native), 'attempt': 1, 'output': str(output)})
            original = {path: path.read_bytes() for path in (native, result)}
            instance.load_states([task], 'full')
            self.assertEqual(instance.states['I4']['status'], 'complete')
            bank = output / 'I4_selected_VALID.pt'
            bank.write_bytes(b'changed-bank')
            with self.assertRaisesRegex(ValueError, 'Changed receipt'):
                instance.load_states([task], 'full')
            bank.unlink()
            with self.assertRaises(FileNotFoundError):
                instance.load_states([task], 'full')
            self.assertEqual(original, {path: path.read_bytes() for path in original})

    def capacity_fixture(self):
        budget = {'vram_bytes': GiB, 'ram_bytes': GiB, 'disk_bytes': 4 * GiB, 'cpu_slots': 1, 'profile': 'fixture'}
        candidate = {'gpu_uuid': 'gpu', 'reservation': budget}
        active = [{'directory': 'peer', 'gpu_uuid': 'other', 'reservation': dict(budget, disk_bytes=2 * GiB)},
                  {'directory': 'same', 'gpu_uuid': 'gpu', 'reservation': budget}]
        live = {'free_vram': {'gpu': 80 * GiB}, 'available_ram': 100 * GiB,
                'free_disk': 100 * GiB, 'available_cpu_before_owned_reservations': 8, 'owned_usage': {}}
        return candidate, active, live

    def test_disk_floor_includes_all_active_and_candidate_reservations(self):
        candidate, active, live = self.capacity_fixture()
        required = 30 * GiB
        for free_disk, admitted in ((20 * GiB, False), (required - 1, False), (required, True)):
            current = dict(live, free_disk=free_disk)
            decision = q.capacity_decision(candidate, active, current, lambda *_: True)
            self.assertEqual(decision['admitted'], admitted)
            self.assertEqual(decision['disk_required_free_bytes'], required)
            self.assertEqual(decision['disk_reserved'], 6 * GiB)
            self.assertEqual(decision['disk_floor_bytes'], 20 * GiB)
            self.assertFalse(decision['whole_campaign_storage_reserved'])
            self.assertFalse(decision['checkpoint_boundary_checks'])
        for changes in ({'available_ram': 0}, {'free_vram': {'gpu': 0}}, {'available_cpu_before_owned_reservations': 0}):
            current = dict(live, **changes)
            original = q.memory.capacity.capacity_decision(candidate, active, current, lambda *_: True)
            decision = q.capacity_decision(candidate, active, current, lambda *_: True)
            self.assertEqual(decision['reasons'], original['reasons'])
            self.assertFalse(decision['admitted'])

    def test_low_disk_blocks_new_attempt_under_inherited_start(self):
        with tempfile.TemporaryDirectory(dir=q.HERE) as temp:
            candidate, active, live = self.capacity_fixture()
            instance = q.LaunchSafeguardsWorker.__new__(q.LaunchSafeguardsWorker)
            instance.root = Path(temp)
            instance.ledger = instance.root / 'ledger'
            instance.config = {'physical_gpu_uuid': 'gpu'}
            instance.args = SimpleNamespace(poll_seconds=1)
            instance.blocked_profiles = {}
            instance.active_ledger = lambda: active
            instance.resources = SimpleNamespace(can_pair=lambda *_: True)
            instance.invocation = Mock(side_effect=AssertionError('Low disk must not construct an attempt'))
            task = {'id': 'candidate', 'reservation': candidate['reservation']}
            with patch.object(q.parent, 'probe', return_value=dict(live, free_disk=30 * GiB - 1)), \
                 patch.object(q.parent, 'capacity_decision', q.capacity_decision), \
                 patch.object(q.parent.subprocess, 'Popen') as child:
                self.assertFalse(instance.start(task, 'full'))
                child.assert_not_called()
            instance.invocation.assert_not_called()
            self.assertEqual(c.read(instance.root / 'CAPACITY_WAIT.json')['decision']['reasons'], ['disk capacity'])
            self.assertFalse((instance.ledger / 'reservations').exists())

    def test_existing_smoke_PASS_is_verified_and_reused_with_original_evidence(self):
        with tempfile.TemporaryDirectory(dir=q.HERE) as temp:
            instance = q.LaunchSafeguardsWorker.__new__(q.LaunchSafeguardsWorker)
            instance.root = Path(temp)
            instance.args = SimpleNamespace(retry_failed=False, poll_seconds=1)
            instance.plan = {'inventory': {'sha256': 'inventory'}}
            task = {'id': 'passed', 'kind': 'fit', 'descriptor': {'sha256': 'descriptor'}, 'smoke_order': 0}
            instance.smokes = {'passed': dict.fromkeys(('worker', 'runtime_job_hash', 'canonical_runtime_job_hash', 'science_contract_hash'), 'bound')}
            directory = instance.folder(task, 'smoke') / 'attempts/0001'
            directory.mkdir(parents=True)
            native = directory / 'NATIVE_RESULT.json'
            c.write(native, {**instance.smokes['passed'], 'schema': 'all-fit-two-update-smoke-result-v1',
                'id': 'passed', 'TEST_accessed': False, 'inventory': instance.plan['inventory'],
                'descriptor': task['descriptor'], 'status': 'PASS', 'TRAIN_updates': 2,
                'TRAIN_finite': True, 'VALID_finite': True,
                'optimizer_calls': {'attempts': [2], 'completed': [2]}, 'checkpoint_reload': {'status': 'PASS'}})
            result = directory / 'ATTEMPT_RESULT.json'
            c.write(result, {'status': 'PASS', 'native_result': c.pin(native), 'attempt': 1, 'output': str(directory / 'output')})
            original = {path: path.read_bytes() for path in (native, result)}
            instance.tasks = [task]
            instance.processes = {}
            instance.reap = lambda _: None
            instance.report = lambda **_: None
            instance.start = Mock(side_effect=AssertionError('Existing PASS must be reused'))
            self.assertTrue(instance.run_phase('smoke'))
            instance.start.assert_not_called()
            self.assertEqual(instance.states['passed']['status'], 'PASS')
            self.assertFalse(instance.retry)
            self.assertEqual(original, {path: path.read_bytes() for path in original})
            native.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Changed receipt'):
                instance.load_states([task], 'smoke')

    def test_full_control_gate_and_native_commands_are_inherited_without_retry(self):
        for name in ('run', 'run_phase', 'start', 'load_states', 'smoke_summary', 'collect', 'reap', 'acquire', 'prerequisites'):
            self.assertIs(getattr(q.LaunchSafeguardsWorker, name), getattr(q.parent.Worker, name))
        self.assertIs(q.parent.global_barrier, c.global_barrier)
        with self.assertRaisesRegex(ValueError, 'separate explicit retry'):
            q.LaunchSafeguardsWorker(SimpleNamespace(retry_failed=True), {})
        instance = q.LaunchSafeguardsWorker.__new__(q.LaunchSafeguardsWorker)
        instance.scheduler_source = {'source': 'current'}
        for phase in ('smoke', 'full'):
            request = {'argv': ['python', '-B', 'native.py', '--output', 'existing-path'], 'resume_uses_smoke': False}
            with patch.object(q.parent.Worker, 'invocation', return_value=request):
                actual = instance.invocation({}, phase, Path('.'))
            self.assertEqual(actual, dict(request, actual_scheduler_producer=instance.scheduler_source))

    def test_owner_bytes_preserved_and_current_producer_in_owner_status_and_invocation(self):
        with tempfile.TemporaryDirectory(dir=q.HERE) as temp:
            root = Path(temp)
            plan = root / 'PLAN.json'
            owner = root / 'OWNER.json'
            c.write(plan, {'state_root': temp})
            previous = b'{ "historical_owner": true }\n'
            owner.write_bytes(previous)
            args = SimpleNamespace(plan=str(plan), retry_failed=False)
            source = {'sha256': 'current-wrapper'}
            original_write = c.write
            def checked_write(path, value):
                if Path(path) == owner:
                    self.assertEqual(c.verify(value['previous_owner']).read_bytes(), previous)
                    self.assertEqual(owner.read_bytes(), previous)
                return original_write(path, value)
            def constructor(instance, args):
                instance.root = root
                q.parent.write(owner, {'identity': 'new-owner'})
            with patch.object(q.parent, 'probe'), patch.object(q.parent, 'capacity_decision'), \
                 patch.object(q.parent.Worker, '__init__', constructor), \
                 patch.object(q.parent, 'write', side_effect=checked_write), \
                 patch.object(c, 'identity', return_value={'pid': 1}):
                instance = q.LaunchSafeguardsWorker(args, source)
                self.assertIs(q.parent.probe, q.memory.probe)
                self.assertIs(q.parent.capacity_decision, q.capacity_decision)
            value = c.read(owner)
            self.assertEqual(value['actual_scheduler_producer'], instance.scheduler_source)
            self.assertEqual(value['actual_scheduler_producer']['actual_outer_source_manifest'], source)
            self.assertEqual(c.verify(value['previous_owner']).read_bytes(), previous)
            retained = list(root.glob('LAUNCH_SAFEGUARDS_OWNER-*.json'))
            self.assertEqual(len(retained), 1)
            self.assertEqual(c.read(retained[0])['producer'], instance.scheduler_source)
            instance.last_report = 0
            with patch.object(q.parent.Worker, 'report') as report:
                instance.report(event='fixture')
                self.assertEqual(report.call_args.kwargs['actual_scheduler_producer'], instance.scheduler_source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
