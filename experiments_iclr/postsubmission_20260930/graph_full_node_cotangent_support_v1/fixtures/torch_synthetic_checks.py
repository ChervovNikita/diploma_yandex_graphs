"""Authored, NOT RUN by the Mac author. Synthetic CPU AD tests only.

Future command: python -B fixtures/torch_synthetic_checks.py
This file opens no scientific data or native model, writes no result artifacts,
and does not qualify the actual native private-factor route or a training job.
"""
import importlib.util
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

PACKET = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'full_node_support_subject', PACKET / 'prototype/graph_band_route_initializer.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class SyntheticChecks(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device='cpu').manual_seed(941)
        self.n, self.c, self.d = 7, 3, 8
        self.a = torch.randn(self.n, self.c, self.d, generator=generator,
                             dtype=torch.float64)/3
        self.b = torch.randn(self.n, self.c, self.d, generator=generator,
                             dtype=torch.float64)/5
        self.theta = torch.ones(self.d, dtype=torch.float64)
        self.calls = 0

        def logits(theta):
            self.calls += 1
            return torch.einsum('ncd,d->nc', self.a, theta) + .03*torch.sin(
                torch.einsum('ncd,d->nc', self.b, theta))

        self.logits = logits
        edges = torch.tensor([[i for i in range(self.n-1)],
                              [i+1 for i in range(self.n-1)]], dtype=torch.int64)
        self.s = subject.symmetric_normalized_adjacency(
            self.n, edges, self.theta.dtype, self.theta.device)
        self.nodes = torch.arange(self.n, dtype=torch.int64)
        self.rows = torch.tensor([0, self.n-1], dtype=torch.int64)
        self.labels = torch.tensor([0, 2], dtype=torch.int64)

    def invoke(self, support, mode='graph', nodes=None):
        before = self.calls
        slices, stats = subject.initialize_four_routes(
            self.logits, self.theta, self.s,
            self.nodes if nodes is None else nodes, self.rows, self.labels,
            tangent_mode=mode, control_seed=942 if mode == 'random_tangent' else None,
            cotangent_support=support, homogeneous_full_node_outputs=True)
        self.assertEqual(self.calls-before,
                         stats['vjp_forwards'] + stats['jvp_calls']
                         + stats['line_search_forward_calls'])
        self.assertEqual(stats['line_search_forward_calls'],
                         stats['candidate_trial_forward_calls']
                         + stats['same_alpha_common_forward_calls'])
        self.assertEqual(tuple(slices.shape), (4, self.d))
        return slices, stats

    def test_supports_share_TRAIN_gradient_full_diagnostics_and_counts(self):
        reports = []
        for support in ['train_remasked', 'full_node']:
            slices, stats = self.invoke(support)
            reports.append(stats)
            self.assertLessEqual(stats['band_gradient_sum_relative_error'],
                                 subject.ALGEBRA_TOLERANCE)
            self.assertTrue(stats['full_output_tangent_finite'])
            self.assertEqual(len(stats['full_output_tangent_gram']), 4)
            self.assertEqual(len(stats['full_output_tangent_pair_rms']), 6)
            self.assertEqual(stats['acceptance_scope'],
                             'unchanged_TRAIN_CE_and_TRAIN_functional_separation')
            self.assertEqual(stats['signed_graph_contrast_functional'], 'shared_full_node_q')
            self.assertEqual(stats['signed_graph_contrast_support_matched'],
                             support == 'full_node')
            self.assertFalse(stats['support_matched_remask_contrast_computed'])
            for attempt in stats['attempts']:
                diagnostic = attempt['signed_graph_contrast']
                self.assertEqual(diagnostic['cotangent_functional'], 'shared_full_node_q')
                self.assertEqual(diagnostic['initializer_cotangent_support'], support)
                self.assertEqual(diagnostic['support_matched'], support == 'full_node')
                if attempt['branch'] != 'common_only' and attempt['finite']:
                    self.assertEqual(attempt['signed_graph_contrast']['status'], 'available')
                    self.assertTrue(attempt['signed_graph_contrast']['diagnostic_only'])
                    self.assertEqual(attempt['signed_graph_contrast']['alpha'], attempt['alpha'])
                    self.assertEqual(len(attempt['full_output_actual_centered_pair_rms']), 6)
            if stats['status'] == 'graph_band':
                alpha = stats['accepted_alpha']
                z = self.logits(self.theta)
                residual = torch.zeros_like(z)
                residual[self.rows] = (z[self.rows].softmax(-1)-F.one_hot(
                    self.labels, self.c).to(z.dtype))/len(self.rows)
                _, pullback = torch.func.vjp(self.logits, self.theta)
                g = pullback(residual.detach())[0]
                q = subject.bernstein_cubic_bands(self.s, residual.detach())-residual.detach()[None]/4
                zs = torch.stack([self.logits(theta) for theta in slices])
                common = self.logits(self.theta-alpha*g)
                direct = subject.signed_graph_contrast(q, zs, common, alpha)
                recorded = next(a for a in reversed(stats['attempts'])
                                if a['branch'] == 'graph_band' and a['alpha'] == alpha)
                self.assertAlmostEqual(direct['signed_value'],
                    recorded['signed_graph_contrast']['signed_value'], places=11)
        self.assertAlmostEqual(reports[0]['common_gradient_squared_norm'],
                               reports[1]['common_gradient_squared_norm'], places=12)

    def test_common_and_random_do_not_acquire_predictive_gate(self):
        _, common = self.invoke('full_node', mode='common_only')
        self.assertEqual(common['same_alpha_common_forward_calls'], 0)
        self.assertEqual(common['graph_sparse_products'], 0)
        _, random = self.invoke('full_node', mode='random_tangent')
        self.assertEqual(random['cotangent_support'], 'full_node')
        for attempt in random['attempts']:
            self.assertFalse(attempt['signed_graph_contrast'].get(
                'prediction_is_predictive_success', False))

    def test_exact_universe_is_required(self):
        for support in ['train_remasked', 'full_node']:
            with self.assertRaisesRegex(ValueError, 'full-node target coverage'):
                self.invoke(support, nodes=self.nodes[:-1])
        with self.assertRaisesRegex(ValueError, 'Explicit homogeneous'):
            subject.initialize_four_routes(self.logits, self.theta, self.s,
                self.nodes, self.rows, self.labels, cotangent_support='full_node')

    def test_support_semantics_against_explicit_full_output_jacobian(self):
        # Deterministic complete graph with unequal TRAIN residuals. Identity
        # margin coordinates make unlabeled output-root gradients explicit.
        # Both supports also have four distinct TRAIN-visible tangents, so the
        # preserved TRAIN gate does not hide a wrong/ignored support selection.
        n, c, d = 4, 2, 4
        theta0 = torch.ones(d, dtype=torch.float64)
        warm_margins = torch.tensor([0., 1., .2, -.7], dtype=torch.float64)

        def logits(theta):
            margins = warm_margins+theta-theta0
            return torch.stack((margins, -margins), dim=-1)/2

        edges = torch.tensor([[i for i in range(n) for j in range(i+1, n)],
                              [j for i in range(n) for j in range(i+1, n)]],
                             dtype=torch.int64)
        s = subject.symmetric_normalized_adjacency(n, edges, theta0.dtype, theta0.device)
        nodes = torch.arange(n, dtype=torch.int64)
        rows = torch.tensor([0, 1], dtype=torch.int64)
        labels = torch.tensor([0, 0], dtype=torch.int64)
        z0 = logits(theta0)
        residual = torch.zeros_like(z0)
        residual[rows] = (z0[rows].softmax(-1)-F.one_hot(labels, c).to(z0.dtype))/len(rows)
        jacobian = torch.func.jacrev(logits)(theta0)
        self.assertEqual(tuple(jacobian.shape), (n, c, d))
        g = torch.einsum('ncd,nc->d', jacobian, residual)

        # Independent dense operator construction, not the subject band helper.
        sd, ident = s.to_dense(), torch.eye(n, dtype=torch.float64)
        sd2, sd3 = sd @ sd, sd @ sd @ sd
        operators = torch.stack(((ident+3*sd+3*sd2+sd3)/8,
                                 3*(ident+sd-sd2-sd3)/8,
                                 3*(ident-sd-sd2+sd3)/8,
                                 (ident-3*sd+3*sd2-sd3)/8))
        full_band_cotangents = torch.einsum('mnk,kc->mnc', operators, residual)
        remask_band_cotangents = torch.zeros_like(full_band_cotangents)
        remask_band_cotangents[:, rows] = full_band_cotangents[:, rows]
        h_full = torch.einsum('ncd,mnc->md', jacobian, full_band_cotangents)
        h_remask = torch.einsum('ncd,mnc->md', jacobian, remask_band_cotangents)
        self.assertGreater(float((h_full-h_remask).norm()), 1e-4)
        projection = torch.eye(d, dtype=torch.float64)-torch.outer(g, g)/g.square().sum()
        expected_tangents = {}

        for support, h in [('train_remasked', h_remask), ('full_node', h_full)]:
            torch.testing.assert_close(h.sum(0), g, rtol=1e-12, atol=1e-12)
            # Center then apply one explicit matrix projection, independently
            # of the subject's repeated projection/centering implementation.
            u = (h-h.mean(0, keepdim=True)) @ projection
            self.assertGreater(float(u.norm()), 1e-4)
            tangent = subject.CAP*g.norm()*u/u.norm()
            expected_tangents[support] = tangent
            directions = -g[None]-tangent
            fields = torch.einsum('ncd,md->mnc', jacobian, tangent)
            fields = fields-fields.mean(-1, keepdim=True)
            flat = fields.reshape(4, -1)
            expected_gram = flat @ flat.T/(n*c)
            expected_train_pairs = torch.stack([
                (fields[i, rows]-fields[j, rows]).square().mean().sqrt()
                for i in range(4) for j in range(i)])
            self.assertGreater(float(expected_train_pairs.min()), 1e-6)
            slices, stats = subject.initialize_four_routes(
                logits, theta0, s, nodes, rows, labels,
                cotangent_support=support, homogeneous_full_node_outputs=True)
            self.assertEqual(stats['status'], 'graph_band')
            torch.testing.assert_close(slices, theta0[None]+stats['accepted_alpha']*directions,
                                       rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(torch.tensor(stats['full_output_tangent_gram'],
                                                     dtype=torch.float64), expected_gram,
                                       rtol=1e-10, atol=1e-12)
            torch.testing.assert_close(torch.tensor(stats['source_tangent_pair_rms'],
                                                     dtype=torch.float64), expected_train_pairs,
                                       rtol=1e-10, atol=1e-12)
        self.assertGreater(float((expected_tangents['full_node']
                                  - expected_tangents['train_remasked']).norm()), 1e-4)

    def test_signed_check_uses_capped_slope_and_is_logit_gauge_invariant(self):
        generator = torch.Generator(device='cpu').manual_seed(943)
        q = torch.randn(4, 5, 3, generator=generator, dtype=torch.float64)
        q = q-q.mean(-1, keepdim=True)
        q = q-q.mean(0, keepdim=True)
        common = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        alpha, cap = .03, .2
        zs = common[None]-alpha*cap*q
        slope = float(cap*q.square().sum())
        first = subject.signed_graph_contrast(q, zs, common, alpha, slope)
        self.assertGreater(first['signed_value'], 0)
        self.assertAlmostEqual(first['signed_value'], alpha*slope, places=12)
        shifted = subject.signed_graph_contrast(q,
            zs+torch.arange(4, dtype=torch.float64)[:, None, None], common+7,
            alpha, slope)
        self.assertAlmostEqual(first['signed_value'], shifted['signed_value'], places=12)
        self.assertTrue(math.isfinite(first['finite_to_first_order_ratio']))


if __name__ == '__main__':
    unittest.main()
