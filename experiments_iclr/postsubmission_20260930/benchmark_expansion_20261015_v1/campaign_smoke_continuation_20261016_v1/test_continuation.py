"""Scheduler metadata tests only; no GPU, remote route or native model."""
import copy
from pathlib import Path
import tempfile
import unittest
from continue_smokes import ContinuationWorker, HERE


class ContinuationTests(unittest.TestCase):
    def worker(self, root, states, failures=()):
        worker = ContinuationWorker.__new__(ContinuationWorker)
        worker.root = Path(root)
        worker.tasks = [{'id': key, 'kind': 'fit', 'smoke_order': i} for i, key in enumerate(states)]
        worker.states = copy.deepcopy(states); worker.processes = {}; worker.retry = set()
        worker.args = type('Args', (), {'poll_seconds': 1})()
        worker.load_states = lambda *args: None
        worker.reap = lambda *args: None
        worker.report = lambda **kwargs: None
        worker.started_ids = []
        def start(task, phase):
            self.assertEqual(phase, 'smoke')
            self.assertEqual(worker.states[task['id']]['status'], 'pending')
            worker.started_ids.append(task['id'])
            worker.states[task['id']] = {'status': 'failed' if task['id'] in failures else 'PASS'}
            return True
        worker.start = start
        return worker

    def test_reuses_PASS_preserves_failure_continues_pending(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            states = {'passed': {'status': 'PASS', 'receipt': 'original'},
                      'failed': {'status': 'failed', 'receipt': 'retained'}, 'next': {'status': 'pending'}}
            worker = self.worker(root, states)
            self.assertFalse(worker.continue_pending())
            self.assertEqual(worker.started_ids, ['next'])
            self.assertEqual(worker.states['passed'], states['passed'])
            self.assertEqual(worker.states['failed'], states['failed'])
            self.assertFalse((Path(root) / 'SMOKE_SUMMARY.json').exists())

    def test_new_failure_does_not_stop_other_pending_or_retry(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            worker = self.worker(root, {'bad': {'status': 'pending'}, 'good': {'status': 'pending'}}, failures=('bad',))
            self.assertFalse(worker.continue_pending())
            self.assertEqual(worker.started_ids, ['bad', 'good'])
            self.assertEqual(worker.states['bad']['status'], 'failed')

    def test_all_PASS_and_STOP_scope(self):
        with tempfile.TemporaryDirectory(dir=HERE) as root:
            worker = self.worker(root, {'done': {'status': 'PASS'}})
            self.assertTrue(worker.continue_pending())
            self.assertEqual(worker.started_ids, [])
            (Path(root) / 'STOP').write_text('stop')
            worker = self.worker(root, {'next': {'status': 'pending'}})
            self.assertFalse(worker.continue_pending())
            self.assertEqual(worker.started_ids, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
