"""CPU checks for the GAT hyperparameter pilot.

    python -m unittest ablation.p05_hparam_grid.test_p05 -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from ablation.common.test_support import import_datasets

import_datasets()

import run_base  # noqa: E402
import run_base_ensemble  # noqa: E402
from ablation.common.resumable import JobStore  # noqa: E402
from ablation.common.tabm_trainer import TrainSettings, train_resumable  # noqa: E402
from ablation.p05_hparam_grid import collect, grid  # noqa: E402
from ablation.p05_hparam_grid.models import build_model  # noqa: E402


DEVICE = torch.device('cpu')
HIDDEN = 64
CLASSES = 3
NODES = 80
STEPS = 40
LR = 1e-3


def toy_bundle():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(NODES, 12, generator=generator)
    y = x[:, :3].argmax(dim=1)
    edge_index = torch.randint(0, NODES, (2, 500), generator=generator)
    data = Data(x=x, y=y, edge_index=edge_index)
    train = torch.zeros(NODES, 10, dtype=torch.bool)
    val = train.clone()
    test = train.clone()
    for split in range(10):
        order = torch.randperm(NODES, generator=generator)
        train[order[:40], split] = True
        val[order[40:60], split] = True
        test[order[60:], split] = True
    return data, train, val, test, False


def run_method(store, method):
    settings = TrainSettings(dataset='roman-empire', split=0, seed=grid.seed_for(method, 0),
                             members=grid.members_for(method), lr=LR, num_steps=STEPS, patience=300)
    return train_resumable(
        settings, store, toy_bundle(), DEVICE,
        lambda: build_model(method, 'GAT', 2, 12, HIDDEN, CLASSES, grid.members_for(method),
                            DEVICE, 0.2, 0.2, 8),
        logits_path=store.dir / 'logits.pt',
    )


class GridTests(unittest.TestCase):
    def test_grid_is_within_the_agreed_budget_and_contains_the_reference(self):
        self.assertEqual(len(grid.CONFIGS), 17)
        self.assertLessEqual(len(grid.CONFIGS), 25)
        self.assertIn(grid.REFERENCE_NAME, grid.CONFIGS)

    def test_seeds_and_step_limits_follow_the_main_table(self):
        self.assertEqual(grid.seed_for('gnnm', 3), 3)
        self.assertEqual(grid.seed_for('ens_m2', 3), 3002)
        self.assertEqual(grid.steps_for('tolokers'), 20000)
        self.assertEqual(grid.steps_for('roman-empire'), 5000)


class AttentionDropoutTests(unittest.TestCase):
    def output(self, attention):
        data = toy_bundle()[0]
        model = build_model('base', 'GAT', 2, 12, HIDDEN, CLASSES, 1, DEVICE, 0.0, attention, 8)
        model.train()
        return [model(data, data.x, tabm_seed=0).detach() for _ in range(2)]

    def test_attention_dropout_is_independent_of_feature_dropout(self):
        first, second = self.output(0.0)
        self.assertTrue(torch.equal(first, second))
        first, second = self.output(0.5)
        self.assertFalse(torch.equal(first, second))


class MatchesMainTableTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_base_matches_run_base(self):
        data, train, val, test, _ = toy_bundle()
        original = run_base.train_single('GAT', 2, 0, data, train, val, test, CLASSES, False,
                                         DEVICE, str(self.root), 'roman-empire', HIDDEN, LR, STEPS)
        ours = run_method(JobStore(self.root / 'base'), 'base')
        for key in ('best_step', 'val_metric', 'test_acc', 'test_loss'):
            self.assertEqual(original[key], ours[key], key)

    def test_composed_ensemble_matches_run_base_ensemble(self):
        data, train, val, test, _ = toy_bundle()
        original = run_base_ensemble.train_ensemble(
            'GAT', 2, 0, data, train, val, test, CLASSES, False, DEVICE, str(self.root),
            'roman-empire', HIDDEN, LR, STEPS)
        rows = []
        for k in range(grid.ENSEMBLE_MEMBERS):
            store = JobStore(self.root / f'm{k}')
            result = run_method(store, f'ens_m{k}')
            rows.append({'method': f'ens_m{k}', 'dataset': 'roman-empire', 'model': 'GAT',
                         'config_name': 'c', 'num_layers': 2, 'split': 0,
                         'job_dir': str(store.dir), **{'best_step': result['best_step']}})
        fresh = toy_bundle()
        composed = collect.compose_ensembles(
            pd.DataFrame(rows),
            lambda dataset, split: (fresh[0], fresh[2][:, split], fresh[3][:, split], False))
        self.assertAlmostEqual(composed.val_metric[0], original['val_metric'], places=6)
        self.assertAlmostEqual(composed.test_metric[0], original['test_metric'], places=6)


class DecisionTests(unittest.TestCase):
    def table(self, values):
        return pd.DataFrame([{'dataset': d, 'method': m, 'val_metric': v / 100}
                             for (d, m), v in values.items()])

    def test_three_of_five_datasets_is_enough(self):
        datasets = ['a', 'b', 'c', 'd', 'e']
        best = self.table({**{(d, 'gnnm'): 80 for d in datasets},
                           **{(d, 'ens'): 80 if d in 'abc' else 81 for d in datasets}})
        reference = self.table({**{(d, 'gnnm'): 79 for d in datasets},
                                **{(d, 'ens'): 80 for d in datasets}})
        decision = collect.decide(best, reference)
        self.assertEqual(decision['gnnm_not_worse_on'], 3)
        self.assertTrue(decision['continue'])

    def test_stops_without_wins_or_a_half_point_improvement(self):
        best = self.table({('a', 'gnnm'): 79, ('a', 'ens'): 80})
        reference = self.table({('a', 'gnnm'): 79.2, ('a', 'ens'): 80})
        self.assertFalse(collect.decide(best, reference)['continue'])

    def test_a_half_point_improvement_continues(self):
        best = self.table({('a', 'gnnm'): 79.8, ('a', 'ens'): 80})
        reference = self.table({('a', 'gnnm'): 79, ('a', 'ens'): 80})
        self.assertTrue(collect.decide(best, reference)['continue'])

    def test_incomplete_configurations_are_not_selected(self):
        selected = pd.DataFrame([
            {'dataset': 'a', 'method': 'gnnm', 'model': 'GAT', 'config_name': name,
             'split': split, 'val_metric': score, 'test_metric': 0.0}
            for name, score, splits in (('good', 0.9, [0]), ('complete', 0.5, [0, 1]))
            for split in splits
        ])
        _, best = collect.choose_configuration(selected, 2)
        self.assertEqual(best.config_name.tolist(), ['complete'])


if __name__ == '__main__':
    unittest.main()
