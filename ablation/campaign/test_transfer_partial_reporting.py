"""Bounded CPU fixtures for independently retried transfer scoring and partial reports."""
import copy
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ablation.common.resumable import atomic_write_json
from ablation.campaign.analyses import execute_analysis, score_transfer, transfer_report
from ablation.campaign.tasks import output, read_json
from ablation.campaign import test_transfer_analyses


class PartialTransferReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = test_transfer_analyses.TransferAnalysisTests(
            'test_three_runners_and_reference100_ensembles')
        self.fixture.root = self.root
        self.meta = self.fixture.fixture('p09_molecular')
        self.methods = [read_json(output(self.root, fit) / 'done.json')['method']
                        for fit in self.meta['fits']]

    def score_methods(self, failed=(), numerical=()):
        entries = []
        for fit, method in zip(self.meta['fits'], self.methods):
            task = 'method_score/' + method
            entries.append(dict(method=method, task=task, fit=fit))
            directory = output(self.root, task)
            directory.mkdir(parents=True, exist_ok=True)
            single = dict(self.meta, fits=[fit], methods=[method])
            if method in failed:
                with patch('ablation.campaign.analyses.subprocess.run',
                           side_effect=subprocess.CalledProcessError(1, ['fixture'])):
                    with self.assertRaises(subprocess.CalledProcessError):
                        score_transfer(single, self.root, directory, 'cpu')
                self.assertFalse((directory / 'done.json').exists())
                atomic_write_json(dict(status='failed', reason='retries_exhausted'),
                                  directory.parent / 'state.json')
                continue
            if method in numerical:
                trained = read_json(output(self.root, fit) / 'done.json')
                trained['fit'] = dict(scientific_status='numerical_failure')
                atomic_write_json(trained, output(self.root, fit) / 'done.json')
            with patch('ablation.campaign.analyses.subprocess.run'):
                receipt = score_transfer(single, self.root, directory, 'cpu')
            atomic_write_json(receipt, directory / 'done.json')
        meta = dict(module=self.meta['module'], pair=self.meta['pair'], cohort=0,
                    methods=self.methods, method_scores=entries)
        with patch('ablation.campaign.analyses.subprocess.run') as invoked:
            receipt = score_transfer(meta, self.root, self.root / 'aggregate', 'cpu')
        invoked.assert_not_called()
        return receipt

    def test_failed_extension_preserves_primary_comparisons_and_both_horizons(self):
        receipt = self.score_methods(failed={'private_bn'})
        self.assertEqual(receipt['missing_methods'], ['private_bn'])
        self.assertEqual(receipt['method_statuses']['private_bn']['failure_kind'],
                         'infrastructure_or_scoring')
        report = transfer_report([receipt])
        self.assertFalse(report['complete'])
        for horizon in report['horizons'].values():
            primary = horizon['paired_comparisons']['fixture/gnnm-minus-ENS']
            self.assertEqual(primary['n'], 1)
            self.assertEqual(primary['cohorts'], [0])
            self.assertTrue(primary['complete'])
            extension = horizon['paired_comparisons']['fixture/private_bn-minus-gnnm']
            self.assertEqual(extension['n'], 0)
            self.assertEqual(extension['missing_cohorts'],
                             [dict(cohort=0, missing_settings=['private_bn'])])

    def test_missing_native_member_removes_only_ensemble_and_numerical_status_is_distinct(self):
        receipt = self.score_methods(failed={'native2'}, numerical={'private_bn'})
        self.assertIn('BASE', receipt['settings'])
        self.assertIn('gnnm', receipt['settings'])
        self.assertNotIn('ENS', receipt['settings'])
        self.assertEqual(receipt['method_statuses']['private_bn']['failure_kind'], 'numerical')
        self.assertEqual(receipt['method_statuses']['private_bn']['scientific_status'], 'numerical_failure')
        self.assertEqual(receipt['method_statuses']['native2']['task_status'], 'failed')
        report = transfer_report([receipt])
        self.assertEqual(report['paired_comparisons']['fixture/gnnm-minus-BASE']['n'], 1)
        self.assertEqual(report['paired_comparisons']['fixture/gnnm-minus-ENS']['n'], 0)
        self.assertFalse(report['complete'])

    def test_all_adapters_preserve_base_without_all_native_members(self):
        for module in ('p09_molecular', 'p11_collab_buddy', 'p12_collab_ncn'):
            with self.subTest(module=module):
                metadata = self.fixture.fixture(module)
                metadata['methods'] = [read_json(output(self.root, fit) / 'done.json')['method']
                                       for fit in metadata['fits']]
                metadata['fits'].remove('fit_native2')
                receipt = self.fixture.run_case(metadata)
                self.assertIn('BASE', receipt['settings'])
                self.assertIn('gnnm', receipt['settings'])
                self.assertNotIn('ENS', receipt['settings'])
                self.assertEqual(receipt['missing_methods'], ['native2'])
                self.assertFalse(receipt['complete'])

    def test_missing_native_zero_preserves_original_method(self):
        receipt = self.score_methods(failed={'native0'})
        self.assertIn('gnnm', receipt['settings'])
        self.assertNotIn('BASE', receipt['settings'])
        self.assertNotIn('ENS', receipt['settings'])
        self.assertEqual(receipt['missing_methods'], ['native0'])

    def test_missing_published_horizon_does_not_remove_extended_scores(self):
        path = (output(self.root, 'fit_gnnm') / 'training' / 'scoring' /
                'cohort0' / 'gnnm' / 'reference100_predictions.pt')
        path.unlink()
        receipt = self.score_methods()
        self.assertIn('gnnm', receipt['settings'])
        self.assertNotIn('gnnm', receipt['reference100_settings'])
        self.assertEqual(receipt['missing_reference100_methods'], ['gnnm'])
        report = transfer_report([receipt])
        self.assertEqual(report['horizons']['extended']['paired_comparisons']['fixture/gnnm-minus-ENS']['n'], 1)
        self.assertEqual(report['horizons']['reference100']['paired_comparisons']['fixture/gnnm-minus-ENS']['n'], 0)
        self.assertTrue(report['complete'])

    def test_score_subprocess_uses_current_device_and_inherited_locks(self):
        fit = self.meta['fits'][0]
        path = output(self.root, fit) / 'done.json'
        trained = read_json(path)
        trained['argv'].extend(['--device', 'cuda:5'])
        atomic_write_json(trained, path)
        with patch.dict(os.environ, GNNM_CAMPAIGN_LOCK_FDS='3,4'), \
                patch('ablation.campaign.analyses.subprocess.run') as invoked:
            score_transfer(dict(self.meta, fits=[fit]), self.root, self.root / 'score', 'cuda:2')
        argv = invoked.call_args.args[0]
        self.assertEqual(argv[argv.index('--device') + 1], 'cuda:2')
        self.assertEqual(invoked.call_args.kwargs['pass_fds'], (3, 4))
        self.assertEqual(argv[-1], '--score-only')

    def test_missing_cohort_task_is_explicit_for_each_primary_contrast(self):
        receipt = self.score_methods()
        existing, missing = 'cohort_score/zero', 'cohort_score/one'
        output(self.root, existing).mkdir(parents=True)
        output(self.root, missing).parent.mkdir(parents=True)
        atomic_write_json(receipt, output(self.root, existing) / 'done.json')
        atomic_write_json(dict(status='failed', reason='retries_exhausted'),
                          output(self.root, missing).parent / 'state.json')
        plan = dict(tasks=[dict(id=name, metadata=dict(pair='fixture', cohort=cohort,
                                                        methods=self.methods))
                           for name, cohort in ((existing, 0), (missing, 1))])
        report = execute_analysis('transfer_report', dict(scores=[existing, missing]),
                                  self.root, self.root / 'report', 'cpu', plan)
        contrast = report['paired_comparisons']['fixture/gnnm-minus-ENS']
        self.assertEqual(contrast['cohorts'], [0])
        self.assertEqual(contrast['missing_cohorts'],
                         [dict(cohort=1, missing_settings=['gnnm', 'ENS'])])
        self.assertFalse(contrast['complete'])
        self.assertFalse(report['complete'])
        self.assertEqual(report['missing_cohort_task_states'][missing]['status'], 'failed')

    def test_per_contrast_convergence_uses_constituent_methods(self):
        receipt = self.score_methods()
        receipt['unresolved_convergence'] = ['private_bn']
        receipt['complete'] = False
        report = transfer_report([receipt])
        self.assertTrue(report['paired_comparisons']['fixture/gnnm-minus-ENS']['complete'])
        extension = report['paired_comparisons']['fixture/private_bn-minus-gnnm']
        self.assertEqual(extension['unresolved_convergence'],
                         [dict(cohort=0, methods=['private_bn'])])
        self.assertFalse(extension['complete'])
        self.assertFalse(report['complete'])

    def test_different_partial_cohorts_pair_only_available_methods(self):
        first = self.score_methods(failed={'private_bn'})
        second = copy.deepcopy(first)
        second['cohort'] = 1
        for bank in ('settings', 'reference100_settings'):
            second[bank]['private_bn'] = dict(val_metric=.8, test_metric=.8)
            del second[bank]['ENS']
        second['missing_methods'] = ['native2']
        report = transfer_report([first, second])
        primary = report['paired_comparisons']['fixture/gnnm-minus-BASE']
        self.assertEqual(primary['cohorts'], [0, 1])
        self.assertEqual(report['paired_comparisons']['fixture/gnnm-minus-ENS']['cohorts'], [0])
        self.assertEqual(report['paired_comparisons']['fixture/private_bn-minus-BASE']['cohorts'], [1])
        self.assertFalse(report['complete'])


if __name__ == '__main__':
    unittest.main()
