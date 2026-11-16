"""Engineering regressions for transfer extensions and heldout scoring stages."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ogb.graphproppred import Evaluator

from ablation.common.resumable import JobStore
from ablation.common.worker import Job, Preempted
from ablation.p09_molecular.models import build_model
from ablation.p09_molecular.normalization import RouteBatchNorm
from ablation.p09_molecular.test_p09 import config_for, toy_data
from ablation.p09_molecular.training import score_selected, train
from ablation.p11_collab_buddy.models import build_model as buddy_model, contrastive_step
from ablation.p11_collab_buddy.run import (
    make_jobs, parse_args, train_job, score_selected as score_buddy,
)
from ablation.p11_collab_buddy.test_p11 import Stop, InterruptStore
from ablation.p11_collab_buddy import test_p11 as buddy_tests


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent / 'test_runs')
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_batchnorm_buffers_are_private_affines_shared(self):
        native = torch.nn.BatchNorm1d(3)
        route = RouteBatchNorm(native)
        self.assertIs(route.weight, native.weight)
        self.assertIs(route.bias, native.bias)
        route.route = 2
        route(torch.ones(4, 3) * 8).sum().backward()
        self.assertEqual(route.num_batches_tracked.tolist(), [0, 0, 1, 0])
        self.assertTrue(torch.equal(route.running_mean[0], torch.zeros(3)))
        self.assertTrue(torch.all(route.running_mean[2] > 0))
        before = copy.deepcopy(route.state_dict())
        route.eval()
        route(torch.randn(4, 3))
        for key, value in before.items():
            torch.testing.assert_close(route.state_dict()[key], value, rtol=0, atol=0)
        original = build_model(config_for('gnnm'), 1)
        private = build_model(config_for('gnnm_private_bn', members=4), 1)
        self.assertEqual(sum(p.numel() for p in original.parameters()),
                         sum(p.numel() for p in private.parameters()))

    def test_molecular_private_resume_and_deferred_scoring(self):
        dataset, splits = toy_data()
        evaluator = Evaluator(name='ogbg-molbace')
        config = config_for('gnnm_private_bn', members=4, smoke=False, smoke_epochs=None,
                            epochs=2, cap_schedule=[2, 8], minimum_epochs=0, patience=2, defer_test=True)
        direct = JobStore(self.root / 'direct')
        direct.ensure_config(config)
        with patch('ablation.p09_molecular.training.score', return_value=0.5):
            expected = train(config, direct, Stop(), dataset, splits, evaluator,
                             {'fixture': True}, torch.device('cpu'))
        direct.write_done(expected)
        self.assertEqual(expected['stopped_epoch'], 3)
        self.assertEqual(len(expected['continuations']), 1)
        self.assertEqual(expected['stop_reason'], 'patience')
        self.assertFalse(expected['test_evaluated'])
        interrupted = JobStore(self.root / 'interrupted')
        interrupted.ensure_config(config)
        stop = SimpleNamespace(requested=False)
        def interrupt(updates):
            stop.requested = updates == 2
        with self.assertRaises(Preempted):
            train(config, interrupted, stop, dataset, splits, evaluator,
                  {'fixture': True}, torch.device('cpu'), on_update=interrupt)
        with patch('ablation.p09_molecular.training.score', return_value=0.5):
            actual = train(config, interrupted, Stop(), dataset, splits, evaluator,
                           {'fixture': True}, torch.device('cpu'))
        self.assertEqual(expected['validation_curve'], actual['validation_curve'])
        for key, value in direct.load_state()['model'].items():
            torch.testing.assert_close(value, interrupted.load_state()['model'][key], rtol=0, atol=0)
        scoring = JobStore(self.root / 'scoring')
        scoring.ensure_config(config)
        result = score_selected(config, direct.dir, scoring, Stop(), dataset, splits,
                                evaluator, {'fixture': True}, torch.device('cpu'))
        self.assertTrue(result['test_evaluated'])
        prediction = torch.load(scoring.dir / 'predictions.pt', weights_only=False)
        self.assertEqual(prediction['splits']['test']['member_logits'].shape, (4, 4, 1))
        with (direct.dir / 'model.pt').open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'differs from the completed'):
            score_selected(config, direct.dir, scoring, Stop(), dataset, splits,
                           evaluator, {'fixture': True}, torch.device('cpu'))

    def test_buddy_contrast_is_active_and_private_gradient_only(self):
        torch.manual_seed(14)
        model = buddy_model('gnnm_private_bn_contrast', 3)
        # Shared endpoint/opposite labels ensure genuine active contrast anchors.
        endpoints = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [1, 2], [0, 3]])
        labels = torch.tensor([1., 0., 1., 0., 1., 0.])
        inputs = (torch.rand(6, 8), torch.rand(6, 2, 3), torch.ones(6), torch.ones(6))
        baseline = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
        rng = torch.get_rng_state()
        loss, diagnostic = contrastive_step(model, inputs, labels, endpoints, 11, optimizer)
        self.assertGreater(diagnostic['anchors'], 0)
        self.assertGreater(diagnostic['auxiliary_loss'], 0)
        torch.set_rng_state(rng)
        logits = torch.stack([baseline(*inputs, route=i).reshape(-1) for i in range(4)])
        reference_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.expand_as(logits))
        reference_loss.backward()
        baseline_optimizer.step()
        torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(model.lin_out.weight, baseline.lin_out.weight, rtol=1e-6, atol=1e-7)
        self.assertGreater(float((model.lin_feat.R - baseline.lin_feat.R).abs().max()), 0)
        self.assertEqual(model.bn_feats.num_batches_tracked.tolist(), [1, 1, 1, 1])

    def test_buddy_contrast_resume_convergence_and_score(self):
        args = parse_args(['--root', str(self.root / 'buddy'), '--cohorts', '0', '--device', 'cpu',
                           '--only', 'gnnm_private_bn_contrast', '--convergence', '--defer-test'])
        job = make_jobs(args, 'test', 'data')[-1]
        config = copy.deepcopy(job.config)
        config['recipe'].update(epochs=2, cap_schedule=[2, 8], minimum_epochs=0, patience=2)
        job = Job(job.key, config)
        bundle = buddy_tests.BuddyTests.fixture(self)
        direct = JobStore(args.root / 'direct')
        direct.ensure_config(config)
        with patch('ablation.p11_collab_buddy.run.evaluate_scores', return_value=0.5):
            expected = train_job(job, direct, Stop(), args, bundle, self.root)
        direct.write_done(expected)
        self.assertEqual(expected['stopped_epoch'], 3)
        self.assertEqual(len(expected['continuations']), 1)
        self.assertFalse(expected['test_evaluated'])
        stop = Stop()
        interrupted = InterruptStore(args.root / 'interrupted', stop)
        interrupted.ensure_config(config)
        with self.assertRaises(Preempted):
            train_job(job, interrupted, stop, args, bundle, self.root)
        resumed = JobStore(interrupted.dir)
        with patch('ablation.p11_collab_buddy.run.evaluate_scores', return_value=0.5):
            actual = train_job(job, resumed, Stop(), args, bundle, self.root)
        self.assertEqual(expected['global_step'], actual['global_step'])
        for key, value in direct.load_state()['model'].items():
            torch.testing.assert_close(value, resumed.load_state()['model'][key], rtol=0, atol=0)
        scoring = JobStore(args.root / 'scoring')
        scoring.ensure_config(config)
        with patch('ablation.p11_collab_buddy.data.prepare_test', return_value=bundle['valid']):
            result = score_buddy(job, direct.dir, scoring, Stop(), args, bundle, self.root)
        self.assertTrue(result['test_evaluated'])
        saved = torch.load(scoring.dir / 'predictions.pt', weights_only=False)
        self.assertEqual(saved['test']['member_scores'].shape, (4, 80))


if __name__ == '__main__':
    unittest.main()
