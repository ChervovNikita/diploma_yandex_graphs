"""Regression coverage for paired initialization effects and selected cost reports."""

import copy
import hashlib
from itertools import product
from pathlib import Path
import tempfile
import unittest

from ablation.common.resumable import atomic_write_json
from ablation.campaign.analyses import comparison
from ablation.campaign.architecture_search import (
    ALTERNATIVES, BACKBONES, STAGE, add_architecture_search,
    bind_selected_profile, execute_comparison,
)
from ablation.campaign.plan import Builder, node_spec
from ablation.campaign.schema import SCHEMA, validate_plan
from ablation.campaign.tasks import output, selected_fit_dirs


class InitializationReportingTests(unittest.TestCase):
    def rows(self, models=('GAT',), depths=(2,), splits=range(10)):
        rows = []
        for model, depth, split in product(models, depths, splits):
            for setting, gain in [('xavier_same_weights', 0), ('xavier_activation_rms', .01 + split * .001)]:
                rows.append(dict(dataset='roman-empire', model=model, num_layers=depth,
                                 split=split, setting=setting, stage='roman_ablation',
                                 val_metric=.5 + gain, test_metric=.6 + gain,
                                 unresolved_convergence=[]))
        return rows

    def test_direct_effect_is_paired_by_backbone_depth_and_split_in_holm_family(self):
        report = comparison(self.rows(models=('GAT', 'SAGE'), depths=(1, 2)))
        self.assertTrue(report['complete'])
        self.assertEqual(len(report['paired_comparisons']), 8)
        for model, depth, role in product(('GAT', 'SAGE'), (1, 2), ('val', 'test')):
            key = f'roman-empire/{model}/{depth}/xavier_activation_rms-minus-xavier_same_weights/{role}'
            contrast = report['paired_comparisons'][key]
            self.assertEqual(contrast['n'], 10)
            self.assertEqual(contrast['paired_splits'], list(range(10)))
            self.assertEqual(contrast['planned_splits'], list(range(10)))
            self.assertEqual(contrast['missing_paired_splits'], [])
            self.assertEqual(contrast['unresolved_paired_splits'], [])
            for split, delta in enumerate(contrast['deltas']):
                self.assertAlmostEqual(delta, .01 + split * .001)
            self.assertAlmostEqual(contrast['exact_sign_flip_p'], 2 / 1024)
            self.assertAlmostEqual(contrast['holm_adjusted_p'], 8 * 2 / 1024)
            self.assertIsNotNone(contrast['descriptive_paired_t_interval'])

    def test_pair_coverage_preserves_failed_absent_and_unresolved_splits(self):
        rows = self.rows(splits=range(3))
        rows[0]['unresolved_convergence'] = ['fit/raw0']
        rows[2].update(scientific_status='numerical_failure', val_metric=None, test_metric=None)
        rows.pop()  # The matched-scale score for split 2 is absent.
        report = comparison(rows)
        contrast = report['paired_comparisons'][
            'roman-empire/GAT/2/xavier_activation_rms-minus-xavier_same_weights/test']
        self.assertFalse(report['complete'])
        self.assertEqual(contrast['n'], 1)
        self.assertEqual(contrast['planned_splits'], [0, 1, 2])
        self.assertEqual(contrast['available_paired_splits'], [0])
        self.assertEqual(contrast['missing_paired_splits'], [1, 2])
        self.assertEqual(contrast['missing_splits_by_arm']['xavier_activation_rms'], [2])
        self.assertEqual(contrast['numerical_failure_splits']['xavier_same_weights'], [1])
        self.assertEqual(contrast['unresolved_paired_splits'], [0])
        self.assertEqual(contrast['unresolved_convergence'][0]['methods']['xavier_same_weights'], ['fit/raw0'])
        self.assertAlmostEqual(contrast['mean'], .01)

    def test_all_failed_pairs_remain_explicit_and_duplicates_are_rejected(self):
        rows = self.rows(splits=range(2))
        for row in rows:
            row.update(scientific_status='numerical_failure', val_metric=None, test_metric=None)
        report = comparison(rows)
        contrast = report['paired_comparisons'][
            'roman-empire/GAT/2/xavier_activation_rms-minus-xavier_same_weights/test']
        self.assertFalse(report['complete'])
        self.assertEqual(contrast['n'], 0)
        self.assertEqual(contrast['planned_splits'], [0, 1])
        self.assertEqual(contrast['missing_paired_splits'], [0, 1])
        with self.assertRaises(ValueError):
            comparison([*rows, rows[0]])


class SelectedArchitectureProfileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def save(self, identifier, value):
        folder = output(self.root, identifier)
        folder.mkdir(parents=True, exist_ok=True)
        atomic_write_json(value, folder / 'done.json')

    def fixture(self, setting='heads_single_pass'):
        binding = dict(task=f'profile/architecture/GAT/{setting}', score=f'score/selected/{setting}',
                       selection='decision/architecture/GAT/selected', setting=setting)
        methods = [f'native{member}' for member in range(4)] if setting == 'ENS' else [setting]
        fits = []
        for member, method in enumerate(methods):
            identifier = f'fit/selected/{member}'
            fit = node_spec('roman-empire', 'GAT', 4, 3, method)
            fit.update(selected_checkpoint_sha256=hashlib.sha256(identifier.encode()).hexdigest(),
                       needs_convergence_extension=False)
            self.save(identifier, fit)
            fits.append(identifier)
        selected = dict(config_name='selected_lr', depth=4, spec=fit)
        decision = dict(role='val', ready=True, selections={setting: selected}, selected_unresolved=[])
        if setting == 'ENS':
            decision.update(ready=False, ready_ENS=True)
        self.save(binding['selection'], decision)
        score = dict(dataset='roman-empire', model='GAT', num_layers=4, split=3,
                     setting=setting, stage=STAGE, fit_ids=fits, val_metric=.7,
                     test_metric=.6, unresolved_convergence=[])
        self.save(binding['score'], score)
        profile = dict(dataset='roman-empire', model='GAT', num_layers=4, split=3,
                       job_dirs=[str(output(self.root, identifier).resolve()) for identifier in fits],
                       full_step_ms=1.2, inference_ms=.2, initial_state='validation_selected_checkpoint')
        self.save(binding['task'], profile)
        metadata = dict(scores=[binding['score']], decisions=[binding['selection']], profiles=[binding],
                        dataset='roman-empire', model='GAT', learning_rates=[3e-5, 1e-4, 3e-4])
        return binding, profile, metadata

    def test_all_sixteen_profiles_follow_exact_selections_on_exclusive_gpus(self):
        builder = Builder()
        reports = add_architecture_search(builder)
        validate_plan(dict(schema=SCHEMA, campaign_id='architecture-profile-test',
                           tasks=list(builder.tasks.values())))
        profiles = [task for task in builder.tasks.values() if task['metadata']['action'] == 'profile']
        self.assertEqual(len(profiles), 16)
        for model in BACKBONES:
            report = builder.tasks[f'report/architecture/{model}']
            bindings = report['metadata']['profiles']
            self.assertEqual({item['setting'] for item in bindings}, {'BASE', 'ENS', 'GNNM', *ALTERNATIVES})
            for binding in bindings:
                task = builder.tasks[binding['task']]
                self.assertEqual(task['kind'], 'gpu')
                self.assertTrue(task['resources']['exclusive'])
                self.assertEqual(set(task['dependencies']), {binding['score'], binding['selection']})
                self.assertEqual(task['metadata']['score'], binding['score'])
                self.assertIn('/split3/', binding['score'])
                self.assertIn(binding['task'], report['optional_dependencies'])
                expected = 'ready_' + binding['setting'] if binding['setting'] in ('BASE', 'ENS', 'GNNM') else 'ready'
                self.assertEqual(task['when']['field'], expected)
        self.assertEqual(len(reports), 2)

    def test_profile_report_binds_selection_score_and_all_ensemble_checkpoint_hashes(self):
        binding, profile, metadata = self.fixture('ENS')
        bound = bind_selected_profile(binding, profile, self.root)
        self.assertEqual(bound['fit_ids'], [f'fit/selected/{member}' for member in range(4)])
        self.assertEqual(len(bound['selected_checkpoint_sha256']), 4)
        for task_key, hash_key in [('selection', 'selection_receipt_sha256'),
                                   ('score', 'score_receipt_sha256'), ('task', 'profile_receipt_sha256')]:
            expected = hashlib.sha256((output(self.root, binding[task_key]) / 'done.json').read_bytes()).hexdigest()
            self.assertEqual(bound[hash_key], expected)
        self.assertEqual(bound['selected_depth'], 4)
        self.assertEqual(bound['selected_config_name'], 'selected_lr')
        directories = selected_fit_dirs({'score': binding['score']}, self.root)
        self.assertEqual([str(path.resolve()) for path in directories], profile['job_dirs'])

    def test_changed_profile_directory_or_fit_selection_is_rejected(self):
        binding, profile, metadata = self.fixture()
        changed = copy.deepcopy(profile)
        changed['job_dirs'] = [str(self.root / 'different')]
        with self.assertRaises(ValueError):
            bind_selected_profile(binding, changed, self.root)
        changed = copy.deepcopy(profile)
        changed['num_layers'] = 1
        with self.assertRaises(ValueError):
            bind_selected_profile(binding, changed, self.root)
        fit = node_spec('roman-empire', 'GAT', 1, 3, 'heads_single_pass')
        fit['selected_checkpoint_sha256'] = 'a' * 64
        self.save('fit/selected/0', fit)
        with self.assertRaises(ValueError):
            bind_selected_profile(binding, profile, self.root)

    def test_missing_checkpoint_hash_and_unready_selection_are_rejected(self):
        binding, profile, metadata = self.fixture()
        fit = node_spec('roman-empire', 'GAT', 4, 3, 'heads_single_pass')
        self.save('fit/selected/0', fit)
        with self.assertRaises(ValueError):
            bind_selected_profile(binding, profile, self.root)
        binding, profile, metadata = self.fixture()
        self.save(binding['selection'], {'role': 'val', 'ready': False, 'selections': {}})
        with self.assertRaises(ValueError):
            bind_selected_profile(binding, profile, self.root)

    def test_profile_keeps_selected_score_convergence_flags(self):
        binding, profile, metadata = self.fixture()
        score = dict(dataset='roman-empire', model='GAT', num_layers=4, split=3,
                     setting=binding['setting'], stage=STAGE, fit_ids=['fit/selected/0'],
                     val_metric=.7, test_metric=.6, unresolved_convergence=['fit/selected/0'])
        self.save(binding['score'], score)
        report = execute_comparison(metadata, self.root)
        self.assertFalse(report['complete'])
        self.assertEqual(report['cost_profiles'][binding['task']]['unresolved_convergence'], ['fit/selected/0'])

    def test_report_retains_missing_and_infeasible_profiles(self):
        binding, profile, metadata = self.fixture()
        report = execute_comparison(metadata, self.root)
        self.assertTrue(report['complete'])
        self.assertEqual(report['cost_profiles'][binding['task']]['score_task'], binding['score'])
        (output(self.root, binding['task']) / 'done.json').unlink()
        report = execute_comparison(metadata, self.root)
        self.assertFalse(report['complete'])
        self.assertFalse(report['evaluation_complete'])
        self.assertEqual(report['missing_profile_tasks'], [binding['task']])
        self.save(binding['task'], {'scientific_status': 'infeasible'})
        report = execute_comparison(metadata, self.root)
        self.assertFalse(report['complete'])
        self.assertTrue(report['evaluation_complete'])
        self.assertEqual(report['infeasible_profile_tasks'], [binding['task']])


if __name__ == '__main__':
    unittest.main()
