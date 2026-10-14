"""CPU tensor semantics only: no molecular data, native model, fit or server."""
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from objectives import MolecularObjective, ObjectiveConfig, embedding_diagnostics, prediction_diagnostics


def reference(h, labels, config):
    """Independent scalar-loop definition, not the vectorized Torch code."""
    keep = ~np.isnan(labels)
    h, labels = h[:, keep], labels[keep]
    z = h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), config.normalization_epsilon)
    m, n, d = z.shape
    classes = set(labels.tolist())
    per_class = {0: [], 1: []}
    if len(classes) == 2:
        for route in range(m):
            for i in range(n):
                positives, by_class = [], {0: [], 1: []}
                for peer in range(m):
                    for j in range(n):
                        if (route, i) == (peer, j):
                            continue
                        score = float(z[route, i] @ z[peer, j] / config.temperature)
                        by_class[int(labels[j])].append(math.exp(score))
                        if labels[i] == labels[j]:
                            positives.append(score)
                if positives:
                    denominator = sum(sum(v) / len(v) for v in by_class.values() if v)
                    per_class[int(labels[i])].append(math.log(denominator) - np.mean(positives))
    alignment = np.mean([np.mean(v) for v in per_class.values() if v]) if any(per_class.values()) else 0.0
    penalties, gamma = [], config.std_fraction / math.sqrt(d)
    for cls in (0, 1):
        rows = labels == cls
        if sum(rows) >= 2:
            std = np.sqrt(np.var(z[:, rows], axis=1) + config.variance_epsilon)
            penalties.append(np.maximum(gamma - std, 0).mean() / gamma)
    variance = np.mean(penalties) if penalties else 0.0
    return alignment, variance


class ObjectiveSemantics(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(15102026)

    def run_hook(self, h, labels, mode="alignment_variance", **kwargs):
        config = ObjectiveConfig(mode=mode, **kwargs)
        labels = torch.as_tensor(labels, dtype=h.dtype)
        result = MolecularObjective(config)(prehead=h, labels=labels[:, None],
                  graph_ids=torch.arange(h.shape[1]), role="TRAIN")
        return result

    def test_scalar_reference_and_finite_difference_gradient(self):
        h = torch.randn(2, 4, 3, generator=self.generator, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([0., 0., 1., 1.], dtype=torch.float64)
        cfg = ObjectiveConfig(temperature=.7, std_fraction=.8)
        hook = MolecularObjective(cfg)
        result = hook(prehead=h, labels=y, graph_ids=torch.arange(4), role="TRAIN")
        a, v = reference(h.detach().numpy(), y.numpy(), cfg)
        self.assertAlmostEqual(float(result.terms["alignment"].detach()), a, places=12)
        self.assertAlmostEqual(float(result.terms["variance"].detach()), v, places=12)
        self.assertTrue(torch.autograd.gradcheck(lambda x: hook(prehead=x, labels=y,
            graph_ids=torch.arange(4), role="TRAIN").loss, (h,), eps=1e-6,
            atol=5e-5, rtol=1e-4, fast_mode=True))

    def test_single_rare_graph_has_other_route_positives(self):
        h = torch.randn(4, 4, 7, generator=self.generator, requires_grad=True)
        r = self.run_hook(h, [0, 0, 0, 1])
        self.assertEqual(r.diagnostics["alignment"]["eligible_anchors"], 16)
        self.assertEqual(r.diagnostics["alignment"]["positive_count_min"], 3)
        self.assertFalse(r.diagnostics["conditional_variance"]["by_class"]["1"]["supported"])
        r.loss.backward()
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertGreater(float(h.grad[:, 3].norm()), 0)

    def test_single_member_singleton_anchor_has_finite_backward(self):
        h = torch.randn(1, 4, 7, generator=self.generator, requires_grad=True)
        r = self.run_hook(h, [0, 0, 0, 1])
        self.assertEqual(r.diagnostics["alignment"]["eligible_anchors"], 3)
        self.assertEqual(r.diagnostics["alignment"]["eligible_anchor_classes"], 1)
        r.loss.backward()
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_one_class_is_not_contrasted_against_other_graphs(self):
        h = torch.randn(4, 8, 5, generator=self.generator, requires_grad=True)
        r = self.run_hook(h, [0] * 8, mode="alignment")
        self.assertEqual(float(r.loss.detach()), 0)
        self.assertFalse(r.diagnostics["alignment"]["two_class_gate"])
        r.loss.backward()
        self.assertEqual(float(h.grad.norm()), 0)

    def test_missing_labels_have_exactly_zero_auxiliary_gradient(self):
        h = torch.randn(4, 5, 7, generator=self.generator, dtype=torch.float64, requires_grad=True)
        y = [0, 0, float("nan"), 1, 1]
        r = self.run_hook(h, y)
        small = self.run_hook(h[:, [0, 1, 3, 4]], [0, 0, 1, 1])
        torch.testing.assert_close(r.loss, small.loss, rtol=0, atol=0)
        r.loss.backward()
        self.assertEqual(float(h.grad[:, 2].norm()), 0)

    def test_member_offsets_do_not_satisfy_conditional_variance(self):
        offsets = torch.randn(4, 1, 7, generator=self.generator)
        h = offsets.expand(-1, 8, -1).clone().requires_grad_()
        r = self.run_hook(h, [0] * 4 + [1] * 4)
        self.assertGreater(float(r.terms["variance"].detach()), .99)
        for cls in ("0", "1"):
            self.assertEqual(r.diagnostics["conditional_variance"]["by_class"][cls]
                             ["fraction_dimensions_below_target"], 1)

    def test_class_averaging_is_not_majority_count_weighting(self):
        losses = []
        for n0 in (16, 31):
            h = torch.zeros(4, 32, 3, dtype=torch.float64)
            h[:, :n0, 0] = 1
            h[:, n0:, 1] = 1
            r = self.run_hook(h, [0] * n0 + [1] * (32 - n0), mode="alignment")
            losses.append(float(r.loss))
        self.assertAlmostEqual(losses[0], losses[1], places=14)
        self.assertAlmostEqual(losses[0], .05 * math.log1p(math.exp(-5)), places=14)

    def test_graph_and_member_permutation_invariance(self):
        h = torch.randn(4, 6, 9, generator=self.generator, dtype=torch.float64)
        y = torch.tensor([0., 1., 0., 1., 0., 1.], dtype=torch.float64)
        a = self.run_hook(h, y)
        p = torch.tensor([5, 0, 2, 1, 4, 3])
        b = self.run_hook(h[[3, 1, 0, 2]][:, p], y[p])
        torch.testing.assert_close(a.loss, b.loss, rtol=1e-12, atol=1e-12)

    def test_gradient_recipients_and_native_loss_preservation(self):
        x = torch.randn(6, 4, generator=self.generator)
        shared = torch.randn(4, 7, generator=self.generator, requires_grad=True)
        private = torch.randn(4, 1, 7, generator=self.generator, requires_grad=True)
        head = torch.randn(7, 1, generator=self.generator, requires_grad=True)
        h = (x @ shared)[None, :, :] * private
        labels = torch.tensor([0., 0., 0., 1., 1., 1.])
        logits = h @ head
        own = torch.stack([F.binary_cross_entropy_with_logits(z[:, 0], labels) for z in logits]).mean()
        r = self.run_hook(h, labels)
        gradients = torch.autograd.grad(r.loss, (shared, private, head), retain_graph=True, allow_unused=True)
        self.assertGreater(float(gradients[0].norm()), 0)
        self.assertGreater(float(gradients[1].norm()), 0)
        self.assertIsNone(gradients[2])
        torch.testing.assert_close(r.scalar_total(own) - own, r.loss)
        restricted = self.run_hook(h, labels, gradient_scope="private")
        with self.assertRaises(ValueError):
            restricted.scalar_total(own)

    def test_roles_identity_and_bad_targets_fail_explicitly(self):
        h = torch.randn(4, 4, 7, generator=self.generator)
        hook = MolecularObjective()
        for y, ids, role in [(torch.tensor([0., 0., 1., 1.]), torch.arange(4), "VALID"),
                             (torch.tensor([0., 0., 1., 1.]), torch.tensor([0, 0, 2, 3]), "TRAIN"),
                             (torch.tensor([0., float("inf"), 1., 1.]), torch.arange(4), "TRAIN"),
                             (torch.tensor([0., 2., 1., 1.]), torch.arange(4), "TRAIN")]:
            with self.assertRaises(ValueError):
                hook(prehead=h, labels=y, graph_ids=ids, role=role)

    def test_empty_observation_support_and_stateless_resume(self):
        h = torch.randn(4, 4, 7, generator=self.generator, requires_grad=True)
        hook = MolecularObjective()
        state = hook.state_dict()
        r = self.run_hook(h, [float("nan")] * 4)
        self.assertEqual(float(r.loss.detach()), 0)
        self.assertEqual(r.diagnostics["observed_graphs"], 0)
        r.loss.backward()
        self.assertEqual(float(h.grad.norm()), 0)
        hook.load_state_dict(state)
        self.assertEqual(hook.state_dict(), state)
        state["schema"] = "different"
        with self.assertRaises(ValueError):
            hook.load_state_dict(state)

    def test_native_evaluator_receives_actual_member_and_mean_raw_logits(self):
        class RecordingEvaluator:
            def __init__(self): self.calls = []
            def eval(self, values):
                self.calls.append(values)
                return {"rocauc": .5}
        evaluator = RecordingEvaluator()
        logits = torch.tensor([[[1.], [-1.], [2.]], [[-2.], [-3.], [-4.]]])
        y, ids = torch.tensor([1., 0., float("nan")]), torch.arange(3)
        report = prediction_diagnostics(raw_logits=logits, labels=y, graph_ids=ids,
                                         role="VALID", evaluator=evaluator)
        self.assertEqual(len(evaluator.calls), 3)
        np.testing.assert_array_equal(evaluator.calls[-1]["y_pred"], logits.mean(0).numpy())
        self.assertEqual(report["pool_repairs_relative_member0"], 0)
        self.assertEqual(report["pool_harms_relative_member0"], 1)
        self.assertEqual(report["member_metrics"][0]["rocauc"], .5)
        self.assertFalse(report["TEST_accessed"])
        h = torch.randn(2, 3, 7, generator=self.generator)
        d = embedding_diagnostics(prehead=h, labels=y, graph_ids=ids, role="VALID")
        self.assertEqual(d["conditional_variance"]["eligible_classes"], 0)


if __name__ == "__main__":
    began = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObjectiveSemantics)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    folder = Path(__file__).resolve().parent
    receipt = {"schema": "molecular-objective-local-semantic-checks-v1", "passed": result.wasSuccessful(),
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "seconds": time.perf_counter() - began, "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(), "device": "CPU",
        "native_models_constructed": 0, "dataset_records_read": 0, "fits": 0, "server_operations": 0,
        "source_sha256": hashlib.sha256((folder / "objectives.py").read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "qualification": "synthetic tensor value/gradient semantics only; no OGB runtime or quality claim"}
    (folder / "SEMANTIC_CHECKS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
