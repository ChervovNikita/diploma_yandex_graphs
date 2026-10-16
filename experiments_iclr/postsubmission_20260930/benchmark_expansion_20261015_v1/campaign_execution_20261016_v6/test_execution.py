"""Scalar/process-stub tests only: no torch import, dataset read, or native work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from common import *
from capacity import capacity_decision, cgroup_paths
from worker import Worker


class ExecutionTests(unittest.TestCase):
    def test_duplicate_cgroup_alias_selects_complete_canonical_view(self):
        membership = '0::/user.slice/user-569201780.slice/session-14949.scope\n'
        mounts = ('35 25 0:30 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n'
                  '927 29 0:30 / /run/cgroups/cpu rw - cgroup2 none rw\n')
        expected = [Path('/sys/fs/cgroup/user.slice/user-569201780.slice/session-14949.scope'),
                    Path('/sys/fs/cgroup/user.slice/user-569201780.slice'),
                    Path('/sys/fs/cgroup/user.slice'), Path('/sys/fs/cgroup')]
        self.assertEqual(cgroup_paths(membership, mounts, lambda path: True), expected)
        self.assertEqual(cgroup_paths(membership, mounts, lambda path: str(path).startswith('/sys/fs/cgroup')), expected)

    def test_cgroup_broader_alias_keeps_limits_above_subtree_view(self):
        membership = '0::/user.slice/user-1.slice/session.scope\n'
        mounts = ('35 25 0:30 /user.slice /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n'
                  '927 29 0:30 / /run/cgroups/all rw - cgroup2 none rw\n')
        actual = cgroup_paths(membership, mounts, lambda path: True)
        self.assertEqual(actual[-1], Path('/run/cgroups/all'))
        self.assertIn(Path('/run/cgroups/all/user.slice'), actual)
        self.assertEqual(len(actual), 4)
        with self.assertRaises(ValueError):
            cgroup_paths(membership, mounts, lambda path: False)

    def test_cgroup_distinct_filesystem_views_are_not_discarded(self):
        mounts = ('35 25 0:30 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n'
                  '927 29 0:31 / /run/cgroup\\040view rw - cgroup2 none rw\n')
        actual = cgroup_paths('0::/group\n', mounts, lambda path: True)
        self.assertIn(Path('/sys/fs/cgroup/group'), actual)
        self.assertIn(Path('/run/cgroup view/group'), actual)
        self.assertEqual(len(actual), 4)

    def test_barrier_checks_all_worker_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.write_text('source')
            ids = [f'fit-{i:04d}' for i in range(6460)]
            inventory = root / 'inventory.json'; write(inventory, {'canonical_fit_ids': ids})
            plan = {'worker': 'server51/gpu0', 'inventory': pin(inventory),
                'canonical_fit_ids_sha256': 'ids', 'smoke_source_manifest': pin(source), 'project_root': temp}
            summary = {'sha256': 'summary'}; target = root / 'PASS.json'
            self.assertFalse(global_barrier(target, plan, summary))
            value = {'schema': 'all-fit-global-smoke-pass-v1', 'status': 'PASS', 'fit_count': 6460,
                'canonical_fit_ids': ids, 'coverage': [{'id': i} for i in ids], 'TEST_accessed': False, 'class_substitution': False,
                'inventory_sha256s': dict.fromkeys(WORKERS, pin(inventory)['sha256']),
                'worker_summary_sha256s': dict.fromkeys(WORKERS, 'summary'), 'canonical_fit_ids_sha256': 'ids',
                'source_manifest': dict(pin(source), path='/different/host/path')}
            write(target, value); self.assertTrue(global_barrier(target, plan, summary))
            value['worker_summary_sha256s']['server51/gpu0'] = 'different'; write(target, value)
            with self.assertRaises(ValueError): global_barrier(target, plan, summary)
            value['worker_summary_sha256s']['server51/gpu0'] = 'summary'; value['canonical_fit_ids_sha256'] = 'wrong'; write(target, value)
            with self.assertRaises(ValueError): global_barrier(target, plan, summary)

    def test_child_inherits_lease_after_supervisor_closes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lease'; held = lock(path)
            child = subprocess.Popen([sys.executable, '-c', 'import os,time; os.write(1,b"ready\\n"); time.sleep(30)'],
                stdout=subprocess.PIPE, pass_fds=(held.fileno(),))
            try:
                self.assertEqual(child.stdout.readline(), b'ready\n'); held.close()
                self.assertTrue(locked(path))
            finally:
                child.terminate(); child.wait(); child.stdout.close()
            self.assertFalse(locked(path))

    def test_prerequisite_process_can_take_slot_before_queue_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); slot = root / 'slot.lock'; receipt = root / 'PREREQUISITE.json'
            worker = Worker.__new__(Worker)
            worker.root = root; worker.plan = {'prerequisites': str(receipt)}
            worker.args = type('Args', (), {'poll_seconds': 0.01})()
            worker.report = lambda **kwargs: None
            held = []
            def acquire():
                value = lock(slot)
                self.assertIsNotNone(value)
                held.append(value)
            def prerequisite():
                # Represents a distinct prerequisite owner using the same GPU slot.
                command = ('import fcntl,json,sys; '
                    'f=open(sys.argv[1],"a+"); '
                    'fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); '
                    'open(sys.argv[2],"w").write(json.dumps({"status":"PASS"}))')
                completed = subprocess.run([sys.executable, '-c', command, str(slot), str(receipt)], capture_output=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                Worker.prerequisites(worker)
            worker.acquire = acquire; worker.prerequisites = prerequisite
            worker.run_phase = lambda phase: False
            try:
                self.assertEqual(worker.run(), 2)
                self.assertTrue(locked(slot))
            finally:
                for handle in held:
                    handle.close()

    def test_resource_host_growth_and_preserved_peer(self):
        budget = {'vram_bytes': 20 * GiB, 'ram_bytes': 12 * GiB, 'disk_bytes': GiB, 'cpu_slots': 2}
        candidate = {'gpu_uuid': 'a', 'reservation': budget}
        active = [{'gpu_uuid': 'b', 'directory': 'otherGPU', 'reservation': budget}]
        live = {'free_vram': {'a': 80 * GiB}, 'available_ram': 30 * GiB, 'free_disk': 50 * GiB,
            'available_cpu_before_owned_reservations': 8, 'owned_usage': {'otherGPU': {'vram': 0, 'rss': 0}}}
        self.assertFalse(capacity_decision(candidate, active, live, lambda a, b: True)['admitted'])
        live['available_ram'] = 40 * GiB
        self.assertTrue(capacity_decision(candidate, active, live, lambda a, b: True)['admitted'])
        peer = {'gpu_uuid': 'a', 'directory': 'peer', 'reservation': dict(budget, vram_bytes=32 * GiB, ram_bytes=16 * GiB), 'external_identity': {'pid': 1}}
        live['available_ram'] = 100 * GiB; live['owned_usage']['peer'] = {'vram': GiB, 'rss': GiB}
        heavy = {'gpu_uuid': 'a', 'reservation': dict(budget, vram_bytes=69 * GiB)}
        self.assertFalse(capacity_decision(heavy, [peer], live, lambda a, b: True)['admitted'])
        self.assertTrue(capacity_decision(candidate, [peer], live, lambda a, b: False)['admitted'])
        self.assertFalse(capacity_decision(candidate, [peer, dict(peer, directory='peer2')], live, lambda a, b: True)['admitted'])

    def test_smoke_cannot_adopt_unobserved_exit_status(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / '0001'; output = directory / 'output'; output.mkdir(parents=True)
            write(directory / 'INVOCATION.json', {'output': str(output)})
            write(output / 'RESULT.json', {'status': 'PASS', 'TEST_accessed': False})
            worker = Worker.__new__(Worker); worker.args = type('Args', (), {'worker': 'server51/gpu0'})()
            saved = []; worker.validate_result = lambda *args: ('PASS', None)
            worker.save_state = lambda task, phase, result: saved.append(result)
            worker.collect({'id': 'fit1'}, 'smoke', directory)
            self.assertEqual(saved[0]['status'], 'failed')
            self.assertIsNone(saved[0]['exit_code'])
            self.assertTrue(saved[0]['unknown_partial_work'])
            self.assertEqual(read(verify(saved[0]['native_result']))['status'], 'PASS')

    def test_failed_attempt_keeps_terminal_and_actual_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); directory = root / 'attempts/0001'; directory.mkdir(parents=True)
            output = directory / 'output'; output.mkdir()
            write(directory / 'INVOCATION.json', {'output': str(output)})
            write(directory / 'TERMINAL.json', {'exit_code': -9, 'forced_kill': True, 'unknown_partial_work': True})
            write(output / 'RESULT.json', {'status': 'failed', 'optimizer_calls': {'attempts': [1], 'completed': [0]}, 'TEST_accessed': False})
            worker = Worker.__new__(Worker); worker.args = type('Args', (), {'worker': 'server51/gpu0'})()
            saved = []
            worker.validate_result = lambda *args: ('PASS', None)
            worker.save_state = lambda task, phase, result: saved.append(result)
            worker.collect({'id': 'fit1'}, 'smoke', directory)
            self.assertEqual(saved[0]['status'], 'failed')
            self.assertEqual(saved[0]['exit_code'], -9)
            self.assertTrue(saved[0]['unknown_partial_work'])
            archived = read(verify(saved[0]['native_result']))
            self.assertEqual(archived['optimizer_calls']['attempts'], [1])
            self.assertTrue((output / 'RESULT.json').exists())


if __name__ == '__main__':
    unittest.main()
