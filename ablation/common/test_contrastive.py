"""Tests of objective semantics, gradient routing, continuation and owner stopping policy."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from ablation.common.contrastive import unit_contrastive_loss, private_gradient_step
from ablation.p09_molecular.test_p09 import toy_data, config_for
from ablation.p09_molecular.models import build_model
from ablation.p09_molecular.training import train
from ablation.common.resumable import JobStore
from ogb.graphproppred import Evaluator


class ContrastiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_loss_matches_explicit_ordered_pair_formula_and_preserves_global_rng(self):
        torch.manual_seed(12)
        features = torch.randn(4, 4, 6, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        state = torch.get_rng_state().clone()
        actual, info = unit_contrastive_loss(features, labels, seed=7, anchor_limit=4, negative_limit=4)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        values = F.normalize(features, dim=-1)
        expected = []
        for member in range(4):
            for view in range(4):
                if member == view:
                    continue
                for anchor in range(4):
                    negative = (labels != labels[anchor]).nonzero().flatten()
                    candidates = torch.cat((values[view, anchor:anchor+1], values[view, negative]))
                    scores = candidates @ values[member, anchor] / 0.2
                    expected.append(torch.logsumexp(scores, 0) - scores[0])
        torch.testing.assert_close(actual, torch.stack(expected).mean())
        self.assertEqual(info['ordered_pairs'], 12)
        self.assertEqual(info['anchors'], 4)

    def test_multiclass_and_empty_negative_support(self):
        features = torch.randn(4, 5, 6, requires_grad=True)
        value, info = unit_contrastive_loss(features, torch.tensor([0, 1, 2, 3, 2]), 19)
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(info['active'])
        zero, info = unit_contrastive_loss(features, torch.ones(5), 19)
        self.assertEqual(float(zero), 0.0)
        self.assertFalse(info['active'])

    def test_auxiliary_gradient_changes_only_declared_private_parameters(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        initial_weight = model.weight.detach().clone()
        initial_bias = model.bias.detach().clone()
        task = model(torch.ones(1, 2)).sum()
        auxiliary = model.weight.square().sum() + model.bias.square().sum()
        expected_weight = initial_weight - 0.1 * torch.ones_like(initial_weight)
        expected_bias = initial_bias - 0.1 * (torch.ones_like(initial_bias) + 0.05 * 2 * initial_bias)
        private_gradient_step(model, optimizer, task, auxiliary, [model.bias])
        torch.testing.assert_close(model.weight, expected_weight)
        torch.testing.assert_close(model.bias, expected_bias)

    def test_molecular_contrastive_arms_train_and_resume_exactly(self):
        from ablation.common.worker import Preempted
        data, splits = toy_data()
        evaluator = Evaluator(name='ogbg-molbace')
        with tempfile.TemporaryDirectory(dir=Path('ablation/p09_molecular/test_runs')) as folder:
            for method in ('gnnm_contrast', 'gnnm_private_bn_contrast'):
                config = config_for(method, members=4, smoke_updates=2, defer_test=True)
                def run(name, stop, hook=None):
                    store = JobStore(Path(folder)/method/name)
                    store.ensure_config(config)
                    result = train(config, store, stop, data, splits, evaluator,
                                   {'unit_fixture': True}, torch.device('cpu'), on_update=hook)
                    return result, store.load_state()
                expected, expected_state = run('direct', SimpleNamespace(requested=False))
                stop = SimpleNamespace(requested=False)
                with self.assertRaises(Preempted):
                    run('resume', stop, lambda updates: setattr(stop, 'requested', updates == 1))
                actual, actual_state = run('resume', SimpleNamespace(requested=False))
                self.assertEqual(actual['validation_curve'], expected['validation_curve'])
                self.assertTrue(actual['last_contrast_diagnostic']['active'])
                self.assertFalse(actual['test_evaluated'])
                for name in expected_state['model']:
                    torch.testing.assert_close(actual_state['model'][name], expected_state['model'][name],
                                               rtol=0, atol=0)

    def test_molecular_patience_starts_before_epoch_one_hundred(self):
        data, splits = toy_data()
        with tempfile.TemporaryDirectory(dir=Path('ablation/p09_molecular/test_runs')) as folder:
            config = config_for('gnnm', members=4, smoke=False, smoke_epochs=None,
                                epochs=1000, cap_schedule=[1000, 2000], minimum_epochs=0,
                                patience=50, defer_test=True)
            store = JobStore(folder)
            store.ensure_config(config)
            with patch('ablation.p09_molecular.training.score', return_value=0.5):
                result = train(config, store, SimpleNamespace(requested=False), data, splits,
                               Evaluator(name='ogbg-molbace'), {'unit_fixture': True},
                               torch.device('cpu'))
            self.assertEqual(result['stopped_epoch'], 51)
            self.assertEqual(result['stale_validation_checks'], 50)
            self.assertEqual(result['stop_reason'], 'patience')
            self.assertFalse(result['reference100_available'])

    def test_buddy_contrast_only_keeps_shared_batchnorm(self):
        from ablation.p11_collab_buddy.models import build_model as buddy_model
        from ablation.p09_molecular.normalization import RouteBatchNorm
        model = buddy_model('gnnm_contrast', 7)
        self.assertFalse(any(isinstance(m, RouteBatchNorm) for m in model.modules()))


if __name__ == '__main__':
    unittest.main()
