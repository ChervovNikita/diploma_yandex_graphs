"""Regression checks for prospective node training and dependent analysis."""

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.common.resumable import JobStore, atomic_torch_save, atomic_write_json
from ablation.common.tabm_trainer import TrainSettings, train_resumable
from ablation.p01_placement.test_p01 import toy_bundle
from ablation.p05_hparam_grid import collect, grid
from ablation.p05_hparam_grid.models import build_model
from ablation.p05_hparam_grid.selected import score_selected, read_config, load_checkpoint
from ablation.p06_member_diversity.selected import analyse_selected
from ablation.p07_cost_small_setup.selected import profile_selected
from ablation.p08_training_steps.history import read_results_csv
from ablation.p10_depth_metrics.depth_table import pair_current_scores, cell_means


class CampaignTrainingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bundle = toy_bundle()
        self.device = torch.device('cpu')

    def model(self, method='base'):
        return build_model(method, 'SAGE', 2, 12, 64, 3, grid.members_for(method),
                           self.device, .2, .2, 8)

    def settings(self, **changes):
        values = dict(dataset='roman-empire', split=0, seed=0, members=1,
                      lr=1e-3, num_steps=20, min_steps=8, patience=2,
                      selection_interval=1, defer_test=True, save_selected=True)
        return TrainSettings(**{**values, **changes})

    def test_floor_prevents_early_stop_and_defer_never_scores_test(self):
        store = JobStore(self.root / 'floor')

        def constant(logits, data, mask, binary, dataset):
            self.assertFalse(torch.equal(mask, self.bundle[3][:, 0]))
            return {'metric': .5, 'metric_name': 'acc', 'acc': .5}

        with patch('ablation.common.tabm_trainer.evaluate', side_effect=constant):
            result = train_resumable(self.settings(), store, self.bundle,
                                     self.device, self.model)
        self.assertEqual(result['stopped_step'], 8)
        self.assertEqual(result['best_step'], 1)
        self.assertEqual(result['stop_reason'], 'patience')
        self.assertFalse(result['needs_convergence_extension'])
        self.assertNotIn('test_metric', result)
        payload = torch.load(store.dir / 'predictions.pt', weights_only=True)
        self.assertEqual(set(payload['roles']), {'train', 'val'})
        self.assertEqual(len(json.loads((store.dir / 'curve.json').read_text())), 8)

    def test_capped_fit_is_flagged_and_can_continue_exact_state(self):
        store = JobStore(self.root / 'cap')
        first = train_resumable(self.settings(num_steps=2, min_steps=2, patience=100),
                                store, self.bundle, self.device, self.model)
        self.assertTrue(first['needs_convergence_extension'])
        second = train_resumable(self.settings(num_steps=4, min_steps=2, patience=100),
                                 store, self.bundle, self.device, self.model)
        self.assertEqual(second['stopped_step'], 4)
        self.assertEqual(second['resumes'], 1)
        self.assertEqual(len(json.loads((store.dir / 'curve.json').read_text())), 4)

    def test_numerical_failure_retains_attempted_update_counter(self):
        store = JobStore(self.root / 'failure')
        with patch('ablation.common.tabm_trainer.train_step',
                   side_effect=[.5, .5, FloatingPointError('nonfinite loss')]):
            with self.assertRaises(FloatingPointError) as caught:
                train_resumable(self.settings(), store, self.bundle,
                                self.device, self.model)
        self.assertEqual(caught.exception.attempted_steps, 3)
        self.assertEqual(caught.exception.completed_steps, 2)

    def test_named_grids_preserve_legacy_and_have_expected_union(self):
        self.assertEqual(len(grid.CONFIGS), 17)
        self.assertEqual(len(grid.configs_for('lrwd')), 9)
        self.assertEqual(len(grid.configs_for('combined')), 24)

    def test_role_only_composition_and_duplicate_member_guard(self):
        rows = []
        for member in range(4):
            directory = self.root / str(member)
            directory.mkdir()
            role = {'indices': torch.arange(4), 'labels': torch.tensor([0, 1, 0, 1]),
                    'logits': torch.tensor([[[2., 0.], [0., 2.], [2., 0.], [0., 2.]]])}
            atomic_torch_save({'roles': {'val': role}, 'is_binary': False,
                               'dataset': 'roman-empire'}, directory / 'predictions.pt')
            rows.append(dict(dataset='roman-empire', model='SAGE', config_name='c',
                             num_layers=2, split=0, method=f'ens_m{member}', seed=member,
                             job_dir=str(directory)))
        frame = pd.DataFrame(rows)
        result = collect.compose_ensembles(frame, None, reveal_test=False)
        self.assertEqual(result.val_metric.iloc[0], 1)
        self.assertNotIn('test_metric', result)
        with self.assertRaises(ValueError):
            collect.compose_ensembles(pd.concat([frame, frame.iloc[:1]]), None, False)

    def test_selected_scoring_diversity_and_profiling_use_saved_models(self):
        configs = []
        for method in ['base', 'gnnm', 'ens_m1', 'ens_m2', 'ens_m3']:
            directory = self.root / method
            store = JobStore(directory)
            config = dict(dataset='roman-empire', model='SAGE', config_name='c',
                          model_module='ablation.p05_hparam_grid.models', method=method,
                          num_layers=2, split=0, seed=grid.seed_for(method, 0),
                          hidden_dim=64, members=grid.members_for(method),
                          dropout=.2, attention_dropout=.2, num_heads=8, lr=1e-3,
                          weight_decay=0., min_steps=2, num_steps=2, patience=100,
                          selection_interval=1)
            store.ensure_config(config)
            result = train_resumable(
                self.settings(seed=config['seed'], members=config['members'],
                              num_steps=2, min_steps=2, patience=100),
                store, self.bundle, self.device, lambda: self.model(method))
            atomic_write_json({**config, **result}, directory / 'done.json')
            configs.append(directory)
        data, train, val, test, binary = self.bundle
        loaded = (data, train, val, test, data.x.shape[1], 3, binary)
        with patch('ablation.p05_hparam_grid.selected.load_dataset', return_value=loaded):
            single = score_selected([configs[1]], self.root / 'score', ['val', 'test'])
            self.assertIn('test_metric', single)
            ensemble = score_selected([configs[0], *configs[2:]], self.root / 'ens', ['val'])
            self.assertEqual(ensemble['members'], 4)
            diversity = analyse_selected([configs[1]], self.root / 'diversity')
            self.assertEqual(len(diversity['member_metrics']), 4)
            profile = profile_selected([configs[1]], self.root / 'profile',
                                       device='cpu', warmup=0, measured=2)
            self.assertGreater(profile['full_step_ms'], 0)
            self.assertGreater(profile['parameter_bytes'], 0)

    def test_selected_checkpoint_hash_and_config_integrity(self):
        directory = self.root / 'identity'
        directory.mkdir()
        atomic_write_json({'seed': 0}, directory / 'config.json')
        atomic_torch_save({'weight': torch.ones(2)}, directory / 'model.pt')
        digest = hashlib.sha256((directory / 'model.pt').read_bytes()).hexdigest()
        atomic_write_json({'seed': 0, 'selected_checkpoint_sha256': digest}, directory / 'done.json')
        config = read_config(directory)
        self.assertTrue(torch.equal(load_checkpoint(directory, config)['weight'], torch.ones(2)))
        self.assertTrue(config['checkpoint_sha256_verified'])
        atomic_torch_save({'weight': torch.zeros(2)}, directory / 'model.pt')
        with self.assertRaises(ValueError):
            load_checkpoint(directory, read_config(directory))
        atomic_write_json({'seed': 1}, directory / 'done.json')
        with self.assertRaises(ValueError):
            read_config(directory)

    def test_historical_repeated_headers_and_current_depth_pairs(self):
        path = self.root / 'scores.csv'
        path.write_text('a,SAGE,1,0,10,.5,.4\n'
                        'dataset,model,num_layers,split,best_step,val_metric,test_metric\n'
                        'a,SAGE,2,0,20,.6,.5\n'
                        'dataset,model,num_layers,split,best_step,val_metric,test_metric\n')
        historical = read_results_csv(path)
        self.assertEqual(historical.best_step.tolist(), [10, 20])
        scores = pd.DataFrame([dict(dataset='a', model='SAGE', num_layers=1, split=0,
                                    method=method, val_metric=score)
                               for method, score in [('base', .5), ('ens', .6), ('gnnm', .7)]])
        paired = pair_current_scores(scores)
        summary = cell_means(paired, role='val')
        self.assertAlmostEqual(summary.gnnm_minus_ens.iloc[0], 10)


if __name__ == '__main__':
    unittest.main()
