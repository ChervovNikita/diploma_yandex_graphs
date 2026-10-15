"""Native block-gradient helper on finite tensors, not a GNN experiment."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from table import make_table

RULES_PATH = Path(__file__).resolve().parents[2]/"reference_wrapper_redesign_20261013_v1/runtime/training_rules.py"
spec = importlib.util.spec_from_file_location("_structural_fixed_source_rules", RULES_PATH)
rules = importlib.util.module_from_spec(spec); spec.loader.exec_module(rules)


class GradientChecks(unittest.TestCase):
    def test_shared_F_private_W_old_state_and_one_native_Adam_transition(self):
        x = torch.tensor([-2., -.8, .3, 2.4, -1.3, -.2, 1.7, 3.1])
        targets = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.])
        shared = torch.nn.Parameter(torch.tensor(.3))
        private = torch.nn.Parameter(torch.tensor([-.4, .2, .5, -.1]))
        optimizer = torch.optim.Adam([{"params": [shared], "lr": .001}, {"params": [private], "lr": .001}])
        group_ids = [id(group) for group in optimizer.param_groups]
        core = SimpleNamespace(_source_parameters=(("shared", shared),), private_parameters=lambda: (private,))
        logits = (shared+private[:, None])*x
        records = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.expand(4, -1), reduction="none")
        F = records.mean(1).mean()
        table = make_table(range(8), [0]*4+[1]*4, [0, 1, 2, 3]*2, role="TRAIN")
        weights = table.tensor_weights(torch)
        W = rules.weighted_native_own_loss(torch, records, weights, native_reduce=lambda row: row.mean(), native_regularizer=F.new_zeros(()))
        expected_shared = torch.autograd.grad(F, shared, retain_graph=True)[0]
        expected_private = torch.autograd.grad(W, private, retain_graph=True)[0]
        wrong_shared = torch.autograd.grad(W, shared, retain_graph=True)[0]
        wrong_private = torch.autograd.grad(F, private, retain_graph=True)[0]
        self.assertGreater(float((expected_shared-wrong_shared).abs()), 1e-4)
        self.assertGreater(float((expected_private-wrong_private).abs().max()), 1e-4)
        old = shared.detach().clone(), private.detach().clone()
        calls = []
        original_step = optimizer.step
        def step(*args, **kwargs):
            calls.append(1)
            return original_step(*args, **kwargs)
        optimizer.step = step
        def before_step():
            torch.testing.assert_close(shared.grad, expected_shared)
            torch.testing.assert_close(private.grad, expected_private)
            self.assertTrue(torch.equal(shared, old[0]) and torch.equal(private, old[1]))
        counts = rules.step_private_responsibility(torch, core, optimizer, F=F, F_weighted=W, before_step=before_step)
        self.assertEqual(len(calls), 1)
        self.assertEqual(counts["reverse_collections"], 2)
        self.assertEqual([id(group) for group in optimizer.param_groups], group_ids)

    def test_uniform_table_retains_native_loss_gradient(self):
        logits = torch.tensor([[-.3, .8], [.1, -.2], [.5, .4], [-.9, .6]], requires_grad=True)
        targets = torch.tensor([0., 1.]).expand_as(logits)
        records = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        F = records.mean(1).mean()
        W = rules.weighted_native_own_loss(torch, records, torch.ones_like(records, dtype=torch.float64),
            native_reduce=lambda row: row.mean(), native_regularizer=F.new_zeros(()))
        torch.testing.assert_close(torch.autograd.grad(F, logits, retain_graph=True)[0], torch.autograd.grad(W, logits)[0])


if __name__ == "__main__":
    unittest.main()
