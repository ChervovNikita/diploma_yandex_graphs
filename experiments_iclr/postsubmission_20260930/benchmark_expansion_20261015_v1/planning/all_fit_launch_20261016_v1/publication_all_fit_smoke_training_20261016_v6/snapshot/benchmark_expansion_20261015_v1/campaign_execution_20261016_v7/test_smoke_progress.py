"""Actual scheduler control flow with deterministic child/process fixtures."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from common import HERE
from worker import Worker


class SmokeProgressTests(unittest.TestCase):
    def fixture(self, root, states, outcomes=None, *, retry_failed=False, drain_ticks=1):
        worker = Worker.__new__(Worker)
        worker.root = Path(root)
        worker.args = SimpleNamespace(poll_seconds=1, retry_failed=retry_failed)
        worker.tasks = [{'id': key, 'kind': 'fit', 'smoke_order': i} for i, key in enumerate(states)]
        worker.processes = {}; worker.states = {}; worker.retry = set()
        worker.started_ids = []; worker.events = []; worker.peak_children = 0
        worker.prerequisites = lambda: None
        worker.acquire = lambda: None
        initial = copy.deepcopy(states)
        outcomes = outcomes or {}

        def load_states(tasks, phase):
            worker.states = copy.deepcopy(initial)
            worker.retry = {key for key, state in initial.items() if state['status'] == 'failed'} if retry_failed else set()
        worker.load_states = load_states

        def start(task, phase):
            key = task['id']
            worker.started_ids.append((phase, key))
            worker.states[key] = {'status': 'running', 'attempt': worker.states[key].get('attempt', 0) + 1}
            worker.processes[key] = {'ticks': drain_ticks, 'phase': phase}
            worker.retry.discard(key)
            worker.peak_children = max(worker.peak_children, len(worker.processes))
            self.assertLessEqual(len(worker.processes), 2)
            return True
        worker.start = start

        def reap(phase):
            for key, child in list(worker.processes.items()):
                child['ticks'] -= 1
                if child['ticks']:
                    continue
                worker.states[key]['status'] = outcomes.get(key, 'PASS' if phase == 'smoke' else 'complete')
                worker.states[key]['receipt'] = 'retained-outcome-' + key
                del worker.processes[key]
                worker.events.append(('reaped', phase, key))
        worker.reap = reap
        worker.report = lambda **kwargs: worker.events.append(('report', worker.phase, kwargs))
        return worker

    def test_new_failure_continues_pending_once_with_two_child_limit(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            worker = self.fixture(root, {key: {'status': 'pending'} for key in ('bad', 'good1', 'good2', 'good3')},
                                  {'bad': 'failed'})
            with patch('worker.time.sleep'):
                self.assertFalse(worker.run_phase('smoke'))
            self.assertEqual(worker.started_ids, [('smoke', key) for key in ('bad', 'good1', 'good2', 'good3')])
            self.assertEqual(worker.peak_children, 2)
            self.assertEqual(worker.states['bad']['status'], 'failed')
            self.assertTrue(all(worker.states[key]['status'] == 'PASS' for key in ('good1', 'good2', 'good3')))
            self.assertEqual(worker.phase, 'smoke_failed')
            self.assertFalse(worker.processes)

    def test_retained_failed_ID_never_retries_and_blocks_summary_and_training(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            failed = {'status': 'failed', 'attempt': 1, 'receipt': 'original-failure'}
            worker = self.fixture(root, {'failed': failed, 'pending': {'status': 'pending'}}, retry_failed=True)
            worker.smoke_summary = Mock(side_effect=AssertionError('A failed smoke cannot authorize a summary'))
            with patch('worker.time.sleep'), patch('worker.global_barrier') as barrier:
                self.assertEqual(worker.run(), 2)
                barrier.assert_not_called()
            worker.smoke_summary.assert_not_called()
            self.assertEqual(worker.started_ids, [('smoke', 'pending')])
            self.assertEqual(worker.states['failed'], failed)
            self.assertFalse(worker.retry)
            with self.assertRaisesRegex(ValueError, 'every assigned smoke PASS'):
                Worker.smoke_summary(worker)
            self.assertFalse((Path(root) / 'SMOKE_SUMMARY.json').exists())
            self.assertFalse((Path(root) / 'TRAINING_BARRIER_ACCEPTED.json').exists())

    def test_STOP_prevents_new_starts_and_drains_both_active_children(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            worker = self.fixture(root, {key: {'status': 'pending'} for key in ('first', 'second', 'pending')}, drain_ticks=2)
            sleeps = []
            def stop_after_launch(_):
                sleeps.append(tuple(worker.processes))
                (Path(root) / 'STOP').write_text('graceful stop')
            with patch('worker.time.sleep', side_effect=stop_after_launch):
                self.assertFalse(worker.run_phase('smoke'))
            self.assertEqual(worker.started_ids, [('smoke', 'first'), ('smoke', 'second')])
            self.assertEqual(len(sleeps), 2)
            self.assertEqual(worker.states['first']['status'], 'PASS')
            self.assertEqual(worker.states['second']['status'], 'PASS')
            self.assertEqual(worker.states['pending']['status'], 'pending')
            self.assertEqual(worker.phase, 'stopped')
            self.assertFalse(worker.processes)

    def test_all_success_still_waits_for_global_barrier_before_full_phase(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            worker = self.fixture(root, {'first': {'status': 'pending'}, 'second': {'status': 'pending'}})
            marker = Path(root) / 'GLOBAL.json'; marker.write_text('{}')
            worker.plan = {'global_smoke_pass': str(marker), 'inventory': {'sha256': 'inventory'}}
            calls = []
            def phase(name):
                calls.append(name)
                return Worker.run_phase(worker, name) if name == 'smoke' else True
            worker.run_phase = phase
            worker.smoke_summary = lambda: calls.append('summary') or {'sha256': 'summary'}
            accepted = iter((False, True))
            def barrier(*args):
                self.assertNotIn('full', calls)
                value = next(accepted); calls.append('barrier-' + str(value)); return value
            with patch('worker.time.sleep'), patch('worker.global_barrier', side_effect=barrier):
                self.assertEqual(worker.run(), 0)
            self.assertEqual(calls, ['smoke', 'summary', 'barrier-False', 'barrier-True', 'full'])
            self.assertTrue((Path(root) / 'TRAINING_BARRIER_ACCEPTED.json').exists())
            self.assertEqual(worker.phase, 'complete')

    def test_full_phase_keeps_failure_stop_and_explicit_retry_behavior(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            states = {'failed': {'status': 'failed', 'attempt': 1}, 'pending': {'status': 'pending'}}
            worker = self.fixture(root, states)
            with patch('worker.time.sleep'):
                self.assertFalse(worker.run_phase('full'))
            self.assertEqual(worker.started_ids, [])
            self.assertEqual(worker.phase, 'full_failed')
            retry = self.fixture(root, states, retry_failed=True)
            with patch('worker.time.sleep'):
                self.assertTrue(retry.run_phase('full'))
            self.assertEqual(retry.started_ids, [('full', 'failed'), ('full', 'pending')])
            self.assertEqual(retry.states['failed']['attempt'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
