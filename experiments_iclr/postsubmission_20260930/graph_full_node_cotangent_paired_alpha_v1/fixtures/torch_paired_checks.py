"""Authored, NOT RUN by the Mac author. Synthetic CPU qualification only.

No scientific data/native model, optimizer, continuation or GPU is used.
Future command: python -B fixtures/torch_paired_checks.py
"""
import importlib.util
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

PACKET = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'paired_subject', PACKET / 'prototype/paired_shared_alpha_initializer.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class PairedChecks(unittest.TestCase):
    def setUp(self):
        self.n, self.c, self.d = 5, 2, 5
        self.theta = torch.ones(self.d, dtype=torch.float64)
        self.warm = torch.tensor([0., 1., .2, -.7, .4], dtype=torch.float64)
        self.nodes = torch.arange(self.n, dtype=torch.int64)
        self.rows = torch.tensor([0, 1, 3], dtype=torch.int64)
        self.labels = torch.tensor([0, 0, 1], dtype=torch.int64)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int64)
        self.s = subject.base.symmetric_normalized_adjacency(
            self.n, edges, self.theta.dtype, self.theta.device)
        permutation = torch.tensor([2, 0, 4, 1, 3], dtype=torch.int64)
        self.sp = torch.sparse_coo_tensor(permutation[self.s.indices()], self.s.values(),
                                          self.s.shape).coalesce()
        self.calls = 0

        def logits(theta):
            self.calls += 1
            margin = self.warm+theta-self.theta
            return torch.stack((margin, -margin), dim=-1)/2

        self.logits = logits

    def invoke(self, logits=None, theta=None):
        return subject.initialize_paired_four_arms(
            self.logits if logits is None else logits,
            self.theta if theta is None else theta, self.s, self.sp, self.nodes,
            self.rows, self.labels, homogeneous_full_node_outputs=True)

    def test_independent_shared_step_reference(self):
        # Analytic full-output Jacobian of the margin closure. No subject AD,
        # band helper or repeated projection code is used for the reference.
        jacobian = torch.stack((torch.eye(self.n, dtype=torch.float64)/2,
                                -torch.eye(self.n, dtype=torch.float64)/2), dim=1)
        z0 = self.logits(self.theta)
        residual = torch.zeros_like(z0)
        residual[self.rows] = (z0[self.rows].softmax(-1)-F.one_hot(
            self.labels, self.c).to(z0.dtype))/len(self.rows)
        g = torch.einsum('ncd,nc->d', jacobian, residual)
        project = torch.eye(self.d, dtype=torch.float64)-torch.outer(g, g)/g.square().sum()
        base_loss = float(F.cross_entropy(z0[self.rows], self.labels))
        threshold = subject.base.FUNCTION_RMS_TOLERANCE*float(
            (z0[self.rows]-z0[self.rows].mean(-1, keepdim=True)).square().mean().sqrt().clamp_min(1.))
        alpha0 = subject.base.RELATIVE_FACTOR_RADIUS*math.sqrt(self.d)/(
            math.sqrt(1+subject.base.CAP**2)*float(g.norm()))
        directions, tangents, fields, qfields = {}, {}, {}, {}
        for name, graph in [('common_only', None), ('train_remasked', self.s),
                             ('full_node', self.s), ('full_node_permuted', self.sp)]:
            if graph is None:
                tangent = torch.zeros((4, self.d), dtype=torch.float64)
            else:
                sd, ident = graph.to_dense(), torch.eye(self.n, dtype=torch.float64)
                plus, minus = (ident+sd)/2, (ident-sd)/2
                bank = torch.stack((plus @ plus @ plus, 3*plus @ plus @ minus,
                                    3*plus @ minus @ minus, minus @ minus @ minus))
                cotangents = torch.einsum('mnk,kc->mnc', bank, residual)
                qfields[name] = cotangents-residual[None]/4
                if name == 'train_remasked':
                    remask = torch.zeros_like(cotangents)
                    remask[:, self.rows] = cotangents[:, self.rows]
                    cotangents = remask
                h = torch.einsum('ncd,mnc->md', jacobian, cotangents)
                u = (h-h.mean(0, keepdim=True)) @ project
                self.assertGreater(float(u.norm()), 1e-4)
                tangent = subject.base.CAP*g.norm()*u/u.norm()
            tangents[name] = tangent
            directions[name] = -g[None]-tangent
            fields[name] = torch.einsum('ncd,md->mnc', jacobian, tangent)
            if name != 'common_only':
                pair_min = min(float((fields[name][i, self.rows]-fields[name][j, self.rows])
                                     .square().mean().sqrt())
                               for i in range(4) for j in range(i))
                self.assertGreater(pair_min, threshold)
            self.assertLessEqual(float(directions[name].norm(dim=1).max()),
                                 math.sqrt(1+subject.base.CAP**2)*float(g.norm())+1e-12)
        self.assertGreater(float((tangents['full_node']-tangents['train_remasked']).norm()), 1e-4)
        self.assertGreater(float((tangents['full_node_permuted']-tangents['full_node']).norm()), 1e-4)

        reference_trials, expected_alpha, expected_slices = [], None, None
        for k in range(6):
            alpha = alpha0/(2**k)
            arm_accepted, arm_slices = {}, {}
            for name in subject.ARM_NAMES:
                trial = self.theta[None]+alpha*directions[name]
                arm_slices[name] = trial
                zs = torch.stack([self.logits(row) for row in trial])
                losses = torch.stack([F.cross_entropy(z[self.rows], self.labels) for z in zs])
                pooled = float(F.cross_entropy(zs.mean(0)[self.rows], self.labels))
                bound = base_loss-subject.base.ARMIJO_C*alpha*float(g.square().sum())
                pair_min = min(float((zs[i, self.rows]-zs[j, self.rows]).square().mean().sqrt())
                               for i in range(4) for j in range(i))
                arm_accepted[name] = bool((losses <= bound).all()) and pooled <= bound and (
                    name == 'common_only' or pair_min/alpha > threshold)
            reference_trials.append(arm_accepted)
            if all(arm_accepted.values()):
                expected_alpha, expected_slices = alpha, arm_slices
                break
        self.assertIsNotNone(expected_alpha)
        before = self.calls
        slices, report = self.invoke()
        self.assertEqual(report['status'], 'joint_accepted')
        self.assertFalse(report['fallback_enabled'])
        self.assertAlmostEqual(report['alpha0'], alpha0, places=14)
        self.assertAlmostEqual(report['accepted_alpha'], expected_alpha, places=14)
        self.assertEqual(len(report['attempts']), len(reference_trials))
        self.assertEqual(self.calls-before, report['vjp_forwards']+report['jvp_calls']
                         +report['line_search_forward_calls'])
        self.assertEqual(report['vjp_forwards'], 1)
        self.assertEqual(report['vjp_calls'], 13)
        self.assertEqual(report['jvp_calls'], 12)
        self.assertEqual(report['graph_sparse_products'], 6)
        self.assertEqual(report['candidate_trial_forward_calls'], 16*len(reference_trials))
        self.assertEqual(report['same_alpha_common_forward_calls'], 0)
        for k, reference in enumerate(reference_trials):
            self.assertEqual({name: row['accepted'] for name, row in report['attempts'][k]['arms'].items()},
                             reference)
        for name in subject.ARM_NAMES:
            torch.testing.assert_close(slices[name], expected_slices[name], rtol=1e-10, atol=1e-12)
            flat = fields[name].reshape(4, -1)
            expected_gram = flat @ flat.T/(self.n*self.c)
            torch.testing.assert_close(torch.tensor(report['arms'][name]['full_output_tangent_gram'],
                                                     dtype=torch.float64), expected_gram,
                                       rtol=1e-10, atol=1e-12)
            if name != 'common_only':
                expected_slope = float((qfields[name]*fields[name]).sum())
                self.assertAlmostEqual(report['arms'][name]['signed_graph_contrast_first_order_slope'],
                                       expected_slope, places=12)

    def test_forced_joint_failure_retains_all_arms_and_trials(self):
        # One scalar private coordinate makes every h collinear with g, so
        # projection removes graph diversity. Common descent remains valid.
        theta0 = torch.ones(1, dtype=torch.float64)
        calls = 0

        def scalar_logits(theta):
            nonlocal calls
            calls += 1
            margin = torch.ones(self.n, dtype=torch.float64)+(theta[0]-1)
            return torch.stack((margin, -margin), dim=-1)/2

        slices, report = self.invoke(logits=scalar_logits, theta=theta0)
        self.assertIsNone(slices)
        self.assertEqual(report['status'], 'joint_failure')
        self.assertEqual(report['accepted_alpha'], None)
        self.assertEqual(len(report['attempts']), 6)
        self.assertEqual(report['candidate_trial_forward_calls'], 96)
        self.assertEqual(report['trial_forward_calls_by_arm'], {name: 24 for name in subject.ARM_NAMES})
        self.assertEqual(calls, report['vjp_forwards']+report['jvp_calls']+96)
        for trial in report['attempts']:
            self.assertFalse(trial['joint_accepted'])
            self.assertTrue(trial['arms']['common_only']['accepted'])
            for name in subject.ARM_NAMES[1:]:
                self.assertFalse(trial['arms'][name]['accepted'])
                self.assertIn('TRAIN_functional_separation_failed', trial['arms'][name]['failure_reasons'])

    def test_rejected_forward_exceptions_are_counted(self):
        # The reference closure has twelve geometry JVPs. Fail every subsequent
        # candidate call and require all 96 attempted forwards to be charged.
        original = self.logits

        def failing_trials(theta):
            if self.calls >= 13:
                self.calls += 1
                raise RuntimeError('forced trial failure')
            return original(theta)

        slices, report = self.invoke(logits=failing_trials)
        self.assertIsNone(slices)
        self.assertEqual(report['status'], 'joint_failure')
        self.assertEqual(report['jvp_calls'], 12)
        self.assertEqual(report['candidate_trial_forward_calls'], 96)
        self.assertEqual(self.calls, 109)
        self.assertEqual(sum(len(row['closure_errors']) for trial in report['attempts']
                             for row in trial['arms'].values()), 96)

    def test_geometry_exception_retains_attempted_call_receipt(self):
        def unavailable(theta):
            raise RuntimeError('forced geometry failure')

        with self.assertRaises(subject.PairedGeometryError) as caught:
            self.invoke(logits=unavailable)
        report = caught.exception.report
        self.assertEqual(report['status'], 'joint_failure')
        self.assertEqual(report['geometry_error']['stage'], 'common_VJP_primal')
        self.assertEqual(report['vjp_forwards'], 1)
        self.assertEqual(report['vjp_calls'], 0)
        self.assertEqual(report['line_search_forward_calls'], 0)


if __name__ == '__main__':
    unittest.main()
