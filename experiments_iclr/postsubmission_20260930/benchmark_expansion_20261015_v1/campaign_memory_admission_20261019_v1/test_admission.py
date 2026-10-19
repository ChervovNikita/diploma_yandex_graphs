"""Focused scalar/state fixtures; no GPU, network, native fit or dataset access."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import admission_worker as a

c = a.c
GiB = c.GiB


def stats(**extra):
    return dict(dict.fromkeys(('file', 'active_file', 'inactive_file', 'shmem',
                              'file_dirty', 'file_writeback', 'unevictable'), 0), **extra)


class AdmissionTests(unittest.TestCase):
    def test_observed_clean_active_cache_is_reclaimable(self):
        row = a.memory_headroom('260919263232', 255045541888,
            stats(file=250763968512, active_file=235813666816,
                  inactive_file=14950277120, shmem=24576))
        self.assertEqual(row['reclaimable'], 250763943936)
        self.assertEqual(row['headroom'], 256637665280)
        self.assertGreater(row['headroom'], 32 * GiB + 8 * GiB)

    def test_dirty_shmem_writeback_unevictable_excluded(self):
        row = a.memory_headroom('100', 96,
            stats(file=70, active_file=50, inactive_file=20, shmem=10,
                  file_dirty=6, file_writeback=4, unevictable=3))
        self.assertEqual(row['reclaimable'], 47)
        self.assertEqual(row['headroom'], 51)
        self.assertEqual(a.memory_headroom('100', 96,
            stats(file=70, active_file=50, inactive_file=20, shmem=70))['headroom'], 4)
        with self.assertRaises(KeyError):
            a.memory_headroom('100', 96, {'file': 70})

    def test_no_cache_double_count_or_overlimit_headroom(self):
        row = a.memory_headroom('100', 80, stats(file=100, active_file=100, inactive_file=100))
        self.assertEqual(row['reclaimable'], 80)
        self.assertEqual(row['headroom'], 100)
        self.assertEqual(a.memory_headroom('100', 130,
            stats(file=40, active_file=40))['headroom'], 10)
        self.assertEqual(a.memory_headroom('100', 130,
            stats(file=20, active_file=20))['headroom'], 0)
        self.assertEqual(a.memory_headroom('100', 100,
            stats(file=20, active_file=20, file_dirty=30))['reclaimable'], 0)
        with self.assertRaises(ValueError):
            a.memory_headroom('100', -1, stats())

    def group(self, root, name, maximum, current, values):
        path = root / name; path.mkdir()
        (path / 'memory.max').write_text(str(maximum))
        (path / 'memory.current').write_text(str(current))
        (path / 'memory.stat').write_text(''.join(f'{key} {value}\n' for key, value in values.items()))
        return path

    def test_ancestor_and_host_bounds_are_minima_not_sums(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as temp:
            root = Path(temp)
            child = self.group(root, 'child', 100, 90, stats(file=80, active_file=80))
            ancestor = self.group(root, 'ancestor', 200, 190, stats(file=20, inactive_file=20))
            unlimited = self.group(root, 'unlimited', 'max', 500, stats())
            self.assertEqual(a.memory_available(1000, [child, ancestor, unlimited])[0], 30)
            self.assertEqual(a.memory_available(25, [child, ancestor, unlimited])[0], 25)
            self.assertEqual(a.memory_available(1000, [ancestor, child, ancestor])[0], 30)
            self.assertEqual(a.memory_available(25, [unlimited])[0], 25)

    def test_probe_retains_original_nonmemory_measurements(self):
        live = {'available_ram': 1, 'memory_accounting': [], 'free_vram': {'gpu': 70},
                'available_cpu_before_owned_reservations': 4, 'free_disk': 123,
                'owned_usage': {'own': {'rss': 7}}, 'sampled_at': 456,
                'external_processes_signalled': False}
        with tempfile.TemporaryDirectory(dir=a.HERE) as temp:
            group = self.group(Path(temp), 'group', 100000, 99000, stats(file=90000, active_file=90000))
            with patch.object(a.capacity, 'probe', return_value=live) as original, \
                 patch.object(a.capacity, 'effective_cgroups', return_value=[group]), \
                 patch.object(a, 'Path', return_value=SimpleNamespace(read_text=lambda: 'MemAvailable: 50 kB\n')):
                actual = a.probe({'config': 'same'}, [{'active': 'same'}])
        original.assert_called_once_with({'config': 'same'}, [{'active': 'same'}])
        self.assertEqual(actual['available_ram'], 50 * 1024)
        self.assertEqual(actual['host_mem_available'], 50 * 1024)
        for key in live.keys() - {'available_ram', 'memory_accounting'}:
            self.assertEqual(actual[key], live[key])

    def test_genuine_shortage_reservations_and_other_guards_still_deny(self):
        budget = {'vram_bytes': 20 * GiB, 'ram_bytes': 12 * GiB, 'disk_bytes': GiB, 'cpu_slots': 2}
        candidate = {'gpu_uuid': 'gpu', 'reservation': budget}
        active = [{'gpu_uuid': 'other', 'directory': 'own', 'reservation': budget}]
        live = {'free_vram': {'gpu': 80 * GiB}, 'available_ram': 30 * GiB,
                'free_disk': 50 * GiB, 'available_cpu_before_owned_reservations': 8,
                'owned_usage': {'own': {'rss': 0, 'vram': 0}}}
        decision = a.capacity.capacity_decision(candidate, active, live, lambda *_: True)
        self.assertEqual(decision['reasons'], ['host RAM free capacity/growth'])
        self.assertEqual(decision['remaining_ram'], 12 * GiB)
        live['available_ram'] = a.memory_headroom(str(40 * GiB), 39 * GiB, stats())['headroom']
        self.assertFalse(a.capacity.capacity_decision(candidate, [], live, lambda *_: True)['admitted'])
        live['available_ram'] = 100 * GiB
        self.assertTrue(a.capacity.capacity_decision(candidate, active, live, lambda *_: True)['admitted'])
        for field, value, reason in [('free_vram', {'gpu': 20 * GiB}, 'GPU free capacity/growth'),
                                     ('free_disk', 8 * GiB, 'disk capacity'),
                                     ('available_cpu_before_owned_reservations', 3, 'CPU capacity')]:
            altered = dict(live, **{field: value})
            self.assertIn(reason, a.capacity.capacity_decision(candidate, active, altered, lambda *_: True)['reasons'])

    def test_campaign_control_and_receipt_rules_are_inherited(self):
        for name in ('run', 'run_phase', 'load_states', 'start', 'validate_result',
                     'smoke_summary', 'collect', 'reap', 'acquire', 'prerequisites'):
            self.assertIs(getattr(a.AdmissionWorker, name), getattr(a.parent.Worker, name))
        self.assertIs(a.parent.capacity_decision, a.capacity.capacity_decision)

    def test_existing_PASS_is_verified_reused_and_never_relabelled(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as temp:
            root = Path(temp)
            instance = a.AdmissionWorker.__new__(a.AdmissionWorker)
            instance.root = root; instance.args = SimpleNamespace(retry_failed=False)
            instance.plan = {'inventory': {'sha256': 'inventory'}}
            task = {'id': 'passed', 'kind': 'fit', 'descriptor': {'sha256': 'descriptor'}}
            instance.smokes = {'passed': dict.fromkeys(('worker', 'runtime_job_hash',
                'canonical_runtime_job_hash', 'science_contract_hash'), 'bound')}
            directory = instance.folder(task, 'smoke') / 'attempts/0001'; directory.mkdir(parents=True)
            native = directory / 'NATIVE_RESULT.json'
            c.write(native, {**instance.smokes['passed'], 'schema': 'all-fit-two-update-smoke-result-v1',
                'id': 'passed', 'TEST_accessed': False, 'inventory': instance.plan['inventory'],
                'descriptor': task['descriptor'], 'status': 'PASS', 'TRAIN_updates': 2,
                'TRAIN_finite': True, 'VALID_finite': True,
                'optimizer_calls': {'attempts': [2], 'completed': [2]}, 'checkpoint_reload': {'status': 'PASS'}})
            result = directory / 'ATTEMPT_RESULT.json'
            c.write(result, {'status': 'PASS', 'native_result': c.pin(native), 'attempt': 1, 'output': str(directory / 'output')})
            original = {path: path.read_bytes() for path in (native, result)}
            instance.load_states([task, {'id': 'pending'}], 'smoke')
            self.assertEqual(instance.states['passed']['status'], 'PASS')
            self.assertEqual(instance.states['pending'], {'status': 'pending', 'attempt': 0})
            self.assertFalse(instance.retry)
            self.assertEqual(original, {path: path.read_bytes() for path in original})
            native.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Changed receipt'):
                instance.load_states([task], 'smoke')

    def phase_fixture(self, root, states):
        instance = a.AdmissionWorker.__new__(a.AdmissionWorker)
        instance.root = root; instance.args = SimpleNamespace(poll_seconds=1, retry_failed=False)
        instance.tasks = [{'id': key, 'kind': 'fit', 'smoke_order': i} for i, key in enumerate(states)]
        instance.processes = {}; instance.started_ids = []; instance.retry = set()
        instance.prerequisites = lambda: None; instance.acquire = lambda: None
        instance.report = lambda **kw: None
        def load(tasks, phase):
            instance.states = copy.deepcopy(states)
        def start(task, phase):
            instance.started_ids.append((phase, task['id']))
            instance.states[task['id']] = {'status': 'PASS', 'attempt': 1}
            return True
        instance.load_states = load; instance.start = start; instance.reap = lambda phase: None
        return instance

    def test_failed_smoke_not_retried_and_blocks_full_training(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as temp:
            failed = {'status': 'failed', 'attempt': 1, 'receipt': 'retained'}
            instance = self.phase_fixture(Path(temp), {'passed': {'status': 'PASS'},
                'failed': failed, 'pending': {'status': 'pending'}})
            instance.smoke_summary = Mock(side_effect=AssertionError('Failed smoke cannot authorize full'))
            with patch.object(a.parent.time, 'sleep'), patch.object(a.parent, 'global_barrier') as barrier:
                self.assertEqual(instance.run(), 2)
            self.assertEqual(instance.started_ids, [('smoke', 'pending')])
            self.assertEqual(instance.states['failed'], failed)
            instance.smoke_summary.assert_not_called(); barrier.assert_not_called()

    def test_full_phase_still_waits_for_global_barrier(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as temp:
            root = Path(temp); marker = root / 'GLOBAL.json'; marker.write_text('{}')
            instance = self.phase_fixture(root, {'passed': {'status': 'PASS'}})
            instance.plan = {'global_smoke_pass': str(marker), 'inventory': {'sha256': 'inventory'}}
            events = []; barriers = iter((False, True))
            instance.run_phase = lambda phase: events.append(phase) or True
            instance.smoke_summary = lambda: {'sha256': 'summary'}
            def barrier(*args):
                self.assertNotIn('full', events)
                result = next(barriers); events.append('barrier-' + str(result)); return result
            with patch.object(a.parent.time, 'sleep'), patch.object(a.parent, 'global_barrier', side_effect=barrier):
                self.assertEqual(instance.run(), 0)
            self.assertEqual(events, ['smoke', 'barrier-False', 'barrier-True', 'full'])
            self.assertTrue((root / 'TRAINING_BARRIER_ACCEPTED.json').exists())

    def test_original_global_barrier_still_requires_all_exact_6460_fits(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as temp:
            root = Path(temp); inventory = root / 'inventory.json'; source = root / 'source'
            source.write_text('frozen smoke source'); ids = [f'fit-{i:04d}' for i in range(6460)]
            c.write(inventory, {'canonical_fit_ids': ids})
            plan = {'worker': 'allocation/gpu0', 'inventory': c.pin(inventory),
                    'canonical_fit_ids_sha256': 'ids', 'smoke_source_manifest': c.pin(source), 'project_root': temp}
            value = {'schema': 'all-fit-global-smoke-pass-v1', 'status': 'PASS', 'fit_count': 6460,
                     'canonical_fit_ids': ids, 'coverage': [{'id': key} for key in ids],
                     'TEST_accessed': False, 'class_substitution': False,
                     'inventory_sha256s': dict.fromkeys(c.WORKERS, c.pin(inventory)['sha256']),
                     'worker_summary_sha256s': dict.fromkeys(c.WORKERS, 'summary'),
                     'canonical_fit_ids_sha256': 'ids', 'source_manifest': c.pin(source)}
            target = root / 'GLOBAL.json'; summary = {'sha256': 'summary'}
            c.write(target, value); self.assertTrue(c.global_barrier(target, plan, summary))
            value['coverage'].pop(); c.write(target, value)
            with self.assertRaisesRegex(ValueError, 'coverage differs'):
                c.global_barrier(target, plan, summary)

    def test_owner_bytes_preserved_before_replacement_and_provenance_recorded(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as temp:
            root = Path(temp); plan = root / 'PLAN.json'; owner = root / 'OWNER.json'
            c.write(plan, {'state_root': temp}); previous = b'{ "previous": true }\n'; owner.write_bytes(previous)
            args = SimpleNamespace(plan=str(plan), retry_failed=False)
            source = {'sha256': 'wrapper'}
            original_write = c.write
            def checked_write(path, value):
                if Path(path) == owner:
                    self.assertEqual(c.verify(value['previous_owner']).read_bytes(), previous)
                    self.assertEqual(owner.read_bytes(), previous)
                return original_write(path, value)
            def constructor(instance, args):
                instance.root = root
                a.parent.write(owner, {'identity': 'new'})
            old_probe = a.parent.probe
            try:
                with patch.object(a.parent.Worker, '__init__', constructor), \
                     patch.object(a.parent, 'write', side_effect=checked_write), \
                     patch.object(c, 'identity', return_value={'pid': 1}):
                    instance = a.AdmissionWorker(args, source)
                self.assertIs(a.parent.probe, a.probe)
                self.assertEqual(c.read(owner)['actual_scheduler_producer'], instance.scheduler_source)
                instance.last_report = 0
                with patch.object(a.parent.Worker, 'report') as report:
                    instance.report(event='check')
                    self.assertEqual(report.call_args.kwargs['actual_scheduler_producer'], instance.scheduler_source)
                with patch.object(a.parent.Worker, 'invocation', return_value={'argv': ['original']}):
                    for phase in ('smoke', 'full'):
                        request = instance.invocation({}, phase, root)
                        self.assertEqual(request['argv'], ['original'])
                        self.assertEqual(request['actual_scheduler_producer'], instance.scheduler_source)
            finally:
                a.parent.probe = old_probe


if __name__ == '__main__':
    unittest.main(verbosity=2)
