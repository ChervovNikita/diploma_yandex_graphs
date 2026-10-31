"""Deterministic tensor/gradient engineering tests, not scientific evidence."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters import node_losses as losses
from adapters.classifier import ClassifierAdapter

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
    from torch.nn import functional as F


@unittest.skipIf(torch is None, "Torch unavailable; do not install a local runtime")
class NodeLossTests(unittest.TestCase):
    def fixture(self):
        labels = torch.tensor([0, 0, 1, 1])
        logits = torch.tensor([[[2., 0., -1.], [.7, -.1, .3], [0., 1., .5], [.3, .9, -.2]],
                               [[.5, .2, -.3], [1., -.5, .3], [1., 1.5, .2], [1., .7, -.2]]],
                              dtype=torch.float64, requires_grad=True)
        hidden = torch.tensor([[[1., 0., .1], [.8, .2, .2], [.1, 1., .2], [.2, .7, .1]],
                               [[.3, .2, .8], [.5, .1, .7], [.8, .2, .1], [.7, .3, .2]]],
                              dtype=torch.float64, requires_grad=True)
        return logits, hidden, labels

    def assert_tensor_close(self, left, right):
        self.assertTrue(torch.allclose(left, right, rtol=1e-10, atol=1e-10), (left, right))

    def test_gncl_matches_probability_mean_and_live_gradients(self):
        logits, _, labels = self.fixture()
        expected_own = F.cross_entropy(logits.flatten(0, 1), labels.repeat(2))
        expected_pool = -logits.softmax(-1).mean(0).gather(1, labels[:, None]).log().mean()
        expected = .5 * expected_own + .5 * expected_pool
        actual = losses.gncl(torch, F, logits, labels, pool_weight=.5)
        self.assert_tensor_close(actual, expected)
        self.assert_tensor_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                 torch.autograd.grad(expected, logits)[0])
        # Raw-logit pooling is a different loss and must not replace this law.
        self.assertFalse(torch.allclose(expected_pool, F.cross_entropy(logits.mean(0), labels)))
        self.assert_tensor_close(losses.gncl(torch, F, logits, labels, pool_weight=0), expected_own)
        self.assert_tensor_close(losses.gncl(torch, F, logits, labels, pool_weight=1), expected_pool)

    def test_within_route_supcon_against_anchorwise_oracle(self):
        _, hidden, labels = self.fixture()
        z = F.normalize(hidden, p=2, dim=-1, eps=1e-12)
        anchors = []
        for member in range(2):
            for row in range(4):
                candidates = [j for j in range(4) if j != row]
                positive = [j for j in candidates if labels[j] == labels[row]]
                denominator = torch.logsumexp(torch.stack([torch.dot(z[member, row], z[member, j]) / .2
                                                           for j in candidates]), dim=0)
                score = torch.stack([torch.dot(z[member, row], z[member, j]) / .2 for j in positive]).mean()
                anchors.append(denominator - score)
        expected = torch.stack(anchors).mean()
        actual, route_losses = losses.within_route_supcon(torch, F, hidden, labels, temperature=.2)
        self.assert_tensor_close(actual, expected)
        self.assert_tensor_close(route_losses.mean(), actual)
        self.assert_tensor_close(torch.autograd.grad(actual, hidden, retain_graph=True)[0],
                                 torch.autograd.grad(expected, hidden)[0])

    def test_paired_supcon_outside_log_positives_and_all_anchor_support(self):
        _, a, labels = self.fixture()
        b = (a.detach().flip(-1) + .03).requires_grad_(True)
        z = torch.cat((F.normalize(a, dim=-1), F.normalize(b, dim=-1)), dim=1)
        truth = labels.repeat(2)
        anchors = []
        for member in range(2):
            for row in range(8):
                eligible = [j for j in range(8) if j != row]
                positive = [j for j in eligible if truth[j] == truth[row]]
                log_z = torch.logsumexp(torch.stack([torch.dot(z[member, row], z[member, j]) / .2 for j in eligible]), 0)
                positive_mean = torch.stack([torch.dot(z[member, row], z[member, j]) / .2 for j in positive]).mean()
                anchors.append(log_z - positive_mean)
        expected = torch.stack(anchors).mean()
        actual = losses.canonical_supcon_eq2(torch, F, a, b, labels, temperature=.2)
        self.assert_tensor_close(actual, expected)
        expected_grads = torch.autograd.grad(expected, (a, b), retain_graph=True)
        actual_grads = torch.autograd.grad(actual, (a, b))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            self.assert_tensor_close(actual_grad, expected_grad)

    def test_cross_view_alignment_is_symmetric_and_has_live_both_views(self):
        _, a, labels = self.fixture()
        b = (a.detach().flip(-1) + .03).requires_grad_(True)
        actual = losses.class_full_cross_view(torch, F, a, b, labels, temperature=.2)
        reverse = losses.class_full_cross_view(torch, F, b, a, labels, temperature=.2)
        self.assert_tensor_close(actual, reverse)
        grads = torch.autograd.grad(actual, (a, b))
        self.assertTrue(all(torch.isfinite(g).all().item() and g.abs().sum().item() > 0 for g in grads))
        eq2 = losses.canonical_supcon_eq2(torch, F, a, b, labels, temperature=.2)
        self.assertFalse(torch.allclose(actual, eq2))

    def test_class_centering_is_live_translation_invariant_and_skips_singletons(self):
        _, a, labels = self.fixture()
        b = (a.detach().flip(-1) + .07).requires_grad_(True)
        actual = losses.residual_member_contrast(torch, F, a, b, labels, temperature=.2)
        offset = torch.tensor([[[1., -2., 3.], [1., -2., 3.], [-3., 1., 2.], [-3., 1., 2.]]], dtype=a.dtype)
        shifted = losses.residual_member_contrast(torch, F, a + offset, b - offset, labels, temperature=.2)
        self.assert_tensor_close(actual, shifted)
        for gradient in torch.autograd.grad(actual, (a, b)):
            self.assert_tensor_close(gradient[:, :2].sum(1), torch.zeros_like(gradient[:, 0]))
            self.assert_tensor_close(gradient[:, 2:].sum(1), torch.zeros_like(gradient[:, 0]))
        sparse_labels = torch.tensor([0, 0, 1, 2])
        singletons = losses.residual_member_contrast(torch, F, a, b, sparse_labels, temperature=.2)
        for gradient in torch.autograd.grad(singletons, (a, b)):
            self.assert_tensor_close(gradient[:, 2:], torch.zeros_like(gradient[:, 2:]))
        self.assertEqual(losses.residual_member_contrast(torch, F, a[:1], b[:1], labels, temperature=.2).item(), 0.)

    def test_combined_and_zero_coefficient_losses_preserve_own_scope(self):
        logits, hidden, labels = self.fixture()
        terms = losses.node_objective(torch, F, logits, hidden, labels,
                                      variant="gncl_within_route_supcon",
                                      coefficients={"pool_weight": .5, "supcon_weight": .05, "temperature": .2})
        expected = losses.gncl(torch, F, logits, labels, pool_weight=.5) + .05 * losses.within_route_supcon(
            torch, F, hidden, labels, temperature=.2)[0]
        self.assert_tensor_close(terms["objective"], expected)
        zero = losses.node_objective(torch, F, logits, None, labels,
                                     variant="gncl_within_route_supcon",
                                     coefficients={"pool_weight": 0., "supcon_weight": 0., "temperature": .2})
        self.assertTrue(torch.equal(zero["objective"], losses.own_ce(torch, F, logits, labels)))
        self.assertEqual(zero["regularization"].item(), 0.)
        self.assertIsNone(torch.autograd.grad(zero["objective"], hidden, allow_unused=True)[0])

    def test_two_view_reduction_and_explicit_auxiliary_panel(self):
        logits, hidden, labels = self.fixture()
        paired_logits = logits + .1
        kwargs = dict(variant="two_view_internal_be", coefficients={"alignment_weight": .05, "residual_weight": .05, "temperature": .2},
                      paired_logits=paired_logits, paired_hidden=hidden[:, :2].flip(-1), auxiliary_labels=labels[:2])
        shared = losses.node_objective(torch, F, logits, hidden[:, :2], labels, **kwargs)
        independent = losses.node_objective(torch, F, logits, hidden[:, :2], labels, independent=True, **kwargs)
        self.assert_tensor_close(independent["objective"], 2 * shared["objective"])
        expected_own = .5 * (torch.stack([F.cross_entropy(z, labels) for z in logits])
                             + torch.stack([F.cross_entropy(z, labels) for z in paired_logits]))
        self.assert_tensor_close(shared["components"]["own_per_route"], expected_own)

    def test_shape_finite_coefficient_and_positive_guards(self):
        logits, hidden, labels = self.fixture()
        for invalid in (-.1, 1.1, float("nan"), True):
            with self.assertRaises(ValueError):
                losses.gncl(torch, F, logits, labels, pool_weight=invalid)
        with self.assertRaises(ValueError):
            losses.within_route_supcon(torch, F, hidden, torch.arange(4), temperature=.2)
        with self.assertRaises(ValueError):
            losses.canonical_supcon_eq2(torch, F, hidden, hidden[:, :3], labels, temperature=.2)
        with self.assertRaises(FloatingPointError):
            losses.probability_pool_ce(torch, logits * float("nan"), labels)
        with self.assertRaises(ValueError):
            losses.own_ce(torch, F, logits, labels.to(torch.int32))
        with self.assertRaises(ValueError):
            losses.node_objective(torch, F, logits, hidden, labels, variant="own", coefficients={"supcon_weight": 0.})

    def test_context_mixture_keeps_complete_denominators_and_literal_row_mass(self):
        rows = 580
        hidden = torch.zeros(1, rows, 3, requires_grad=True)
        labels = torch.zeros(rows, dtype=torch.long)
        weights = torch.zeros(4, rows, rows)
        support = torch.zeros_like(weights, dtype=torch.bool)
        row = torch.arange(rows)
        for context in range(4):
            weights[context, row, (row + 1) % rows] = .25
            for neighbor in range(1, context + 2):
                support[context, row, (row + neighbor) % rows] = True
        targets = {"weights": weights, "retained": support, "full_allowed": ~torch.eye(rows, dtype=torch.bool)}
        result, route, pairs = losses.context_auxiliary(torch, F, hidden, labels, source="positive_exclusion",
                                                       mode="mixture", targets=targets, temperature=.2)
        self.assertAlmostEqual(result.item(), .25 * sum(math.log(n) for n in (1, 2, 3, 4)) / 4, places=6)
        self.assertFalse(math.isclose(result.item(), .25 * math.log(4), abs_tol=1e-4))
        self.assertEqual(tuple(pairs.shape), (1, 4))
        self.assertTrue(torch.isfinite(torch.autograd.grad(result, hidden)[0]).all().item())
        with self.assertRaises(ValueError):
            losses.context_auxiliary(torch, F, hidden, labels, source="positive_exclusion", mode="private",
                                     targets=targets, temperature=.2)
        targets["weights"] = weights.requires_grad_(True)
        with self.assertRaises(ValueError):
            losses.context_auxiliary(torch, F, hidden, labels, source="positive_exclusion", mode="mixture",
                                     targets=targets, temperature=.2)

    def test_classifier_optional_callback_updates_and_restores_owned_state(self):
        model = torch.nn.Linear(2, 2)
        x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
        y = torch.tensor([0, 1, 0, 1])

        class Callback:
            calls = 0

            def __call__(self, adapter, step, full_logits, hidden):
                self.calls += 1
                logits = full_logits[:, adapter.train_ids]
                result = losses.node_objective(torch, F, logits, None, adapter.train_y,
                                               variant="gncl", coefficients={"pool_weight": .5})
                return {**result, "operation_counts": {"model_forwards": 1, "route_predictions": 1,
                                                       "supervised_examples": len(adapter.train_y)}}

            def capture_state(self):
                return {"calls": self.calls}

            def restore_state(self, state):
                self.calls = state["calls"]

        callback = Callback()
        arguments = dict(logits_fn=lambda model, training, step: model(x).unsqueeze(0),
                         train_ids=torch.tensor([0, 1]), train_y=y[:2], select_ids=torch.tensor([2, 3]), select_y=y[2:],
                         regularizer=lambda model: model.weight.sum() * 0, fusion="mean_probabilities",
                         selector={"metric": "accuracy", "direction": "max", "ties": "first", "patience_steps": 10, "save_every": 1},
                         provenance={"engineering": True})
        adapter = ClassifierAdapter(torch, F, model, torch.optim.SGD(model.parameters(), lr=.1),
                                    objective_callback=callback, objective_provenance={"variant": "gncl", "pool_weight": .5}, **arguments)
        before = model.weight.detach().clone()
        result = adapter.train_step(1)
        self.assertFalse(torch.equal(before, model.weight))
        self.assertEqual(result["operation_counts"]["optimizer_updates"], 1)
        self.assertIn("own_mean_ce", result["objective_components"])
        state = adapter.capture_state()
        callback.calls = 99
        adapter.restore_state(state)
        self.assertEqual(callback.calls, 1)
        state["objective_provenance"] = {"different": True}
        with self.assertRaises(ValueError):
            adapter.restore_state(state)
        with self.assertRaises(ValueError):
            ClassifierAdapter(torch, F, model, adapter.optimizer, objective_callback=callback, **arguments)


if __name__ == "__main__":
    unittest.main()
