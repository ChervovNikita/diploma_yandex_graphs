"""CPU-only numerical failure receipts, retry boundaries and artifact-free scoring."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ablation.common.resumable import JobStore
from ablation.common.worker import Job, run_worker
from ablation.campaign.analyses import score_transfer
from ablation.campaign.tasks import output, read_json, transfer_fit, write_json


class TransferNumericalFailureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stop = SimpleNamespace(requested=False)
        self.job = Job('job', dict(method='gnnm', seed=0))

    def run_failure(self, error, campaign):
        def execute(job, store, stop):
            store.save_state(dict(progress=7))
            raise error
        invoked = Mock(side_effect=execute)
        with patch.dict(os.environ, GNNM_CAMPAIGN_TASK='fixture/task' if campaign else ''), \
                patch('ablation.common.worker.torch.cuda.is_available', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()):
            code = run_worker(self.root, [self.job], invoked, max_failures=3,
                              wait_for_others=False, stop=self.stop)
        return code, invoked, JobStore(self.root / self.job.key)

    def test_campaign_numerical_failure_is_terminal_and_preserves_state(self):
        code, invoked, store = self.run_failure(FloatingPointError('nonfinite loss'), campaign=True)
        self.assertEqual(code, 0)
        self.assertEqual(invoked.call_count, 1)
        receipt = store.read_done()
        self.assertEqual(receipt['scientific_status'], 'numerical_failure')
        self.assertEqual(receipt['config'], self.job.config)
        self.assertEqual(receipt['error'], 'nonfinite loss')
        self.assertTrue(receipt['state_preserved'])
        self.assertEqual(store.load_state(), dict(progress=7))
        self.assertEqual(store.failure_count(), 1)
        self.assertFalse(store.claim_path.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_worker(self.root, [self.job], invoked, stop=self.stop), 0)
        self.assertEqual(invoked.call_count, 1)

    def test_infrastructure_failure_keeps_bounded_retries(self):
        code, invoked, store = self.run_failure(OSError('temporary disk error'), campaign=True)
        self.assertEqual(code, 1)
        self.assertEqual(invoked.call_count, 3)
        self.assertFalse(store.is_done())
        self.assertTrue(store.has_state())
        self.assertEqual(store.failure_count(), 3)

    def test_legacy_numerical_failure_keeps_previous_retry_behavior(self):
        code, invoked, store = self.run_failure(FloatingPointError('nonfinite loss'), campaign=False)
        self.assertEqual(code, 1)
        self.assertEqual(invoked.call_count, 3)
        self.assertFalse(store.is_done())
        self.assertEqual(store.failure_count(), 3)

    def test_transfer_fit_surfaces_failure_and_reuses_terminal_receipt(self):
        out = self.root / 'transfer'
        meta = dict(module='p09_molecular', pair='molbace_gcn', method='gnnm', cohort=0)
        inner = out / 'training' / 'cohort0' / 'gnnm' / 'done.json'
        def finish(argv, **kwargs):
            write_json(dict(scientific_status='numerical_failure', error='nonfinite fit',
                            config=dict(seed=0)), inner)
        with patch('ablation.campaign.tasks.inherited_task_locks', return_value=()), \
                patch('ablation.campaign.tasks.subprocess.run', side_effect=finish) as invoked:
            first = transfer_fit(meta, out, 'cpu')
            second = transfer_fit(meta, out, 'cpu')
        self.assertEqual(invoked.call_count, 1)
        self.assertEqual(first['scientific_status'], 'numerical_failure')
        self.assertEqual(first['error'], 'nonfinite fit')
        self.assertEqual(first['fit']['config'], dict(seed=0))
        self.assertEqual(first, second)

    def test_transfer_fit_infrastructure_error_propagates(self):
        meta = dict(module='p09_molecular', pair='molbace_gcn', method='gnnm', cohort=0)
        with patch('ablation.campaign.tasks.inherited_task_locks', return_value=()), \
                patch('ablation.campaign.tasks.subprocess.run',
                      side_effect=subprocess.CalledProcessError(1, ['fixture'])):
            with self.assertRaises(subprocess.CalledProcessError):
                transfer_fit(meta, self.root / 'transfer', 'cpu')

    def method_fixture(self, module, numerical_fit=False):
        fit = 'fit/gnnm'
        base = output(self.root, fit) / 'training'
        trained = dict(method='gnnm', fit_dir=str(base / 'cohort0' / 'gnnm'),
                       fit=dict(seed=0), argv=['python', '-m', 'fixture', '--root', str(base)])
        if numerical_fit:
            trained.update(scientific_status='numerical_failure', error='nonfinite training')
        write_json(trained, output(self.root, fit) / 'done.json')
        meta = dict(module=module, pair='fixture', cohort=0, fits=[fit], methods=['gnnm'])
        return meta, base

    def test_training_failure_scoring_requires_no_artifacts_or_subprocess(self):
        meta, _ = self.method_fixture('p09_molecular', numerical_fit=True)
        with patch('ablation.campaign.analyses.subprocess.run') as invoked, \
                patch('torch.load', side_effect=AssertionError('must not load predictions')):
            receipt = score_transfer(meta, self.root, self.root / 'score', 'cpu')
        invoked.assert_not_called()
        self.assertEqual(receipt['missing_methods'], ['gnnm'])
        self.assertEqual(receipt['method_statuses']['gnnm']['failure_kind'], 'numerical')
        self.assertFalse(receipt['complete'])

    def test_score_only_worker_failure_requires_no_prediction_files(self):
        for module in ('p09_molecular', 'p11_collab_buddy'):
            with self.subTest(module=module):
                meta, base = self.method_fixture(module)
                scored = base / 'scoring' / 'cohort0' / 'gnnm' / 'done.json'
                write_json(dict(scientific_status='numerical_failure', error='nonfinite scoring',
                                config=dict(seed=0)), scored)
                with patch('ablation.campaign.analyses.subprocess.run'), \
                        patch('torch.load', side_effect=AssertionError('must not load predictions')):
                    receipt = score_transfer(meta, self.root, self.root / 'score', 'cpu')
                status = receipt['method_statuses']['gnnm']
                self.assertEqual(status['scientific_status'], 'numerical_failure')
                self.assertEqual(status['phase'], 'scoring')
                self.assertEqual(status['error'], 'nonfinite scoring')
                self.assertFalse(receipt['complete'])
                self.assertEqual(receipt['directories'], {})

    def test_ncn_numerical_scoring_failure_is_caught_and_reusable(self):
        meta, _ = self.method_fixture('p12_collab_ncn')
        out = self.root / 'ncn_score'
        with patch('ablation.p12_collab_ncn.score.score_fit',
                   side_effect=FloatingPointError('nonfinite NCN score')) as invoked, \
                patch('torch.load', side_effect=AssertionError('must not load predictions')):
            first = score_transfer(meta, self.root, out, 'cpu')
            second = score_transfer(meta, self.root, out, 'cpu')
        self.assertEqual(invoked.call_count, 1)
        self.assertEqual(first['method_statuses']['gnnm']['scientific_status'], 'numerical_failure')
        self.assertEqual(first, second)
        self.assertEqual(read_json(out / 'gnnm' / 'done.json')['error'], 'nonfinite NCN score')

    def test_ncn_infrastructure_scoring_failure_propagates(self):
        meta, _ = self.method_fixture('p12_collab_ncn')
        with patch('ablation.p12_collab_ncn.score.score_fit', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                score_transfer(meta, self.root, self.root / 'ncn_score', 'cpu')


if __name__ == '__main__':
    unittest.main()
