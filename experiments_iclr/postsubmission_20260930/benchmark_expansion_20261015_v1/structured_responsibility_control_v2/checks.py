"""Bounded CPU semantic checks, with synthetic tensors only. No native fits."""
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from structured_responsibility_control_v2.sampler import (
    COVARIANCE_QUERY_LIMIT, MEMBER_LIMIT, QUERY_LIMIT, QueryBatch, authenticate,
    half_incidence_rows, ideal_raw_covariance, make_query_batch, raw_responsibilities,
)
from structured_responsibility_control_v2.objectives import (
    build_objectives, native_record_losses, reduce_native,
)
import torch

NUMERICS = {}
DTYPE = torch.float64


def batch(pairs, relations=None, directed=False, scope="synthetic-audit/node-and-relation-ids-v1"):
    return make_query_batch(pairs, relations, scope=scope, directed=directed)


class SemanticChecks(unittest.TestCase):
    def test_persistent_identity_duplicates_orientation_and_subsets(self):
        original = batch([(7, 2), (2, 7), (8, 9), (2, 2)])
        reordered = batch([(2, 2), (9, 8), (2, 7)])
        for mode in ("endpoint", "query_iid"):
            left = raw_responsibilities(original, 31, mode=mode)
            right = raw_responsibilities(reordered, 31, mode=mode)
            for member in range(4):
                self.assertEqual(left[member][0], left[member][1])
                self.assertEqual(right[member], (left[member][3], left[member][2], left[member][0]))
            self.assertNotEqual(left[0], left[1])
            self.assertNotEqual(left, raw_responsibilities(original, 32, mode=mode))
        directed = batch([(7, 2), (2, 7)], directed=True)
        endpoint = raw_responsibilities(directed, 31, mode="endpoint")[0]
        query = raw_responsibilities(directed, 31, mode="query_iid")[0]
        self.assertEqual(endpoint[0], endpoint[1])
        self.assertNotEqual(query[0], query[1])

    def test_scope_relation_and_label_separation(self):
        original = batch([(2, 7), (2, 7)], [0, 1])
        other_scope = batch([(2, 7), (2, 7)], [0, 1], scope="other-fixed-node-namespace")
        for mode in ("endpoint", "query_iid"):
            values = raw_responsibilities(original, 12, mode=mode)
            self.assertNotEqual(values[0][0], values[0][1])
            self.assertNotEqual(values, raw_responsibilities(other_scope, 12, mode=mode))
        self.assertNotEqual(authenticate(original, [0, 1]), authenticate(original, [1, 0]))
        with self.assertRaises(ValueError):
            authenticate(original, [0, 1], role="VALID")

    def test_half_incidence_covariance_duplicates_loops_relations(self):
        queries = batch([(0, 1), (0, 2), (3, 4), (1, 0), (0, 0), (0, 1), (0, 0)], [0, 0, 0, 0, 0, 1, 0])
        B = half_incidence_rows(queries)
        self.assertEqual(B[4], {(0, 0): 1.0})
        K = ideal_raw_covariance(queries, mode="endpoint")
        self.assertEqual(K[0], (0.5, 0.25, 0.0, 0.5, 0.5, 0.0, 0.5))
        self.assertEqual(K[4], (0.5, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0))
        columns = sorted(set().union(*(row.keys() for row in B)))
        matrix = torch.tensor([[row.get(col, 0.0) for col in columns] for row in B], dtype=DTYPE)
        torch.testing.assert_close(torch.tensor(K, dtype=DTYPE), matrix @ matrix.T, rtol=0, atol=0)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(matrix @ matrix.T).min()), -1e-12)
        iid = ideal_raw_covariance(queries, mode="query_iid")
        self.assertEqual(iid[0], (0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0))
        self.assertEqual(iid[4][6], 1.0)
        directed = batch([(0, 1), (1, 0)], directed=True)
        self.assertEqual(ideal_raw_covariance(directed, mode="endpoint")[0][1], 0.5)
        self.assertEqual(ideal_raw_covariance(directed, mode="query_iid")[0][1], 0.0)

    def test_rng_isolation_and_finite_positive_hash_values(self):
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state().clone()
        queries = batch([(0, 1), (0, 0), (1, 2)])
        for mode in ("endpoint", "query_iid"):
            values = raw_responsibilities(queries, -12, mode=mode, floor=0)
            self.assertEqual(values, raw_responsibilities(queries, -12, mode=mode, floor=0))
            tensor = torch.tensor(values, dtype=DTYPE)
            self.assertTrue(bool((tensor > 0).all()))
            self.assertTrue(bool((tensor <= 53 * torch.log(torch.tensor(2.0, dtype=DTYPE))).all()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_validation_and_resource_bounds(self):
        for pairs in ([], [(0.1, 1)], [(-1, 2)], [(True, 2)]):
            with self.assertRaises(ValueError):
                batch(pairs)
        with self.assertRaises(ValueError):
            batch([(0, 1)], scope="")
        with self.assertRaises(ValueError):
            batch([(0, 1)], [0, 1])
        with self.assertRaises(ValueError):
            QueryBatch([(0, 1, 0)], "scope")
        queries = batch([(0, 1)])
        for kwargs in ({"members": MEMBER_LIMIT + 1}, {"floor": -0.1}, {"floor": float("nan")}):
            with self.assertRaises(ValueError):
                raw_responsibilities(queries, 1, mode="endpoint", **kwargs)
        with self.assertRaises(ValueError):
            batch([(0, 1)] * (QUERY_LIMIT + 1))
        with self.assertRaises(ValueError):
            ideal_raw_covariance(batch([(0, 1)] * (COVARIANCE_QUERY_LIMIT + 1)), mode="endpoint")

    def test_unit_weight_parity_and_native_group_laws(self):
        groups = [1, 1, 0, 0, 0]
        logits = torch.tensor([[0.2, -0.5, 1.2, 0.4, -0.8], [-0.3, 0.9, 0.2, -0.5, 0.7]], dtype=DTYPE)
        target = torch.tensor(groups, dtype=DTYPE).expand_as(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        for family in ("NCN", "NCNC", "BUDDY"):
            records = native_record_losses(torch, logits, groups, family)
            expected = bce.mean(-1) if family == "BUDDY" else bce[:, :2].mean(-1) + bce[:, 2:].mean(-1)
            torch.testing.assert_close(reduce_native(torch, records, groups, family), expected)
            torch.testing.assert_close(reduce_native(torch, records, groups, family, torch.ones_like(records)), expected)
        weights = torch.tensor([[1., 3., 1., 2., 6.], [3., 1., 4., 2., 1.]], dtype=DTYPE)
        expected_ncn = (bce[:, :2] * weights[:, :2]).sum(-1) / weights[:, :2].sum(-1)
        expected_ncn += (bce[:, 2:] * weights[:, 2:]).sum(-1) / weights[:, 2:].sum(-1)
        expected_buddy = (bce * weights).sum(-1) / weights.sum(-1)
        torch.testing.assert_close(reduce_native(torch, bce, groups, "NCN", weights), expected_ncn)
        torch.testing.assert_close(reduce_native(torch, bce, groups, "BUDDY", weights), expected_buddy)
        self.assertFalse(torch.allclose(expected_ncn, expected_buddy))

    def test_empty_native_groups_and_weight_validation(self):
        records = torch.ones((4, 2), dtype=DTYPE)
        torch.testing.assert_close(reduce_native(torch, records, [1, 1], "BUDDY"), torch.ones(4, dtype=DTYPE))
        with self.assertRaises(ValueError):
            reduce_native(torch, records, [1, 1], "NCN")
        for weights in (torch.zeros_like(records), torch.ones(2, dtype=DTYPE),
                        torch.ones_like(records, requires_grad=True), torch.full_like(records, float("inf"))):
            with self.assertRaises(ValueError):
                reduce_native(torch, records, [1, 0], "BUDDY", weights)
        with self.assertRaises(ValueError):
            reduce_native(torch, records, [1, 0.1], "BUDDY")

    def test_native_131072_rows_kept_and_full_groups_normalized(self):
        # One complete NCN source batch: 65,536 positives + 65,536 negatives.
        quarter = 32_768
        queries = batch([(0, 1)] * (2*quarter) + [(2, 3)] * (2*quarter))
        groups = [1] * (2*quarter) + [0] * (2*quarter)
        sampled = raw_responsibilities(queries, 91, mode="query_iid", members=1)
        self.assertEqual(len(sampled[0]), 131_072)
        self.assertEqual(sampled[0][0], sampled[0][65_535])
        self.assertEqual(sampled[0][65_536], sampled[0][-1])
        records = torch.tensor([1.]*quarter + [3.]*quarter + [2.]*quarter + [4.]*quarter, dtype=DTYPE)
        weights = torch.tensor([1.]*quarter + [3.]*quarter + [2.]*quarter + [6.]*quarter, dtype=DTYPE)
        # Normalizing four arbitrary quarters independently would produce 5,
        # whereas the two complete native groups give 2.5 + 3.5 = 6.
        torch.testing.assert_close(reduce_native(torch, records, groups, "NCN", weights), torch.tensor(6., dtype=DTYPE))
        logits = torch.zeros((4, len(groups)), dtype=DTYPE)
        source = tuple((logits.sum(-1)*0 + 2*torch.log(torch.tensor(2., dtype=DTYPE))).unbind())
        constructed = build_objectives(torch, batch=queries, groups=groups, family="NCN", logits=logits,
            native_member_losses=source, route_query_signatures=[authenticate(queries, groups)]*4, seed=91, mode="unit")
        self.assertEqual(constructed["diagnostics"]["query_count"], 131_072)
        self.assertEqual(tuple(constructed["raw_weights"].shape), (4, 131_072))
        NUMERICS["native_batch_acceptance"] = {"positives": 65_536, "negatives": 65_536,
            "total_rows": 131_072, "rows_dropped": 0, "normalization_groups": 2,
            "complete_group_synthetic_weighted_sum": 6.0, "incorrect_quarterwise_sum": 5.0}

    def test_objective_block_gradients_and_source_parity_gate(self):
        queries = batch([(0, 1), (0, 2), (3, 4), (1, 3), (4, 4)])
        groups = [1, 1, 0, 0, 0]
        shared = torch.tensor(0.4, dtype=DTYPE, requires_grad=True)
        private = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=DTYPE, requires_grad=True)
        x = torch.tensor([1.0, 2.0, -1.0, 0.5, 3.0], dtype=DTYPE)
        y = torch.tensor([2.0, -1.0, 0.5, 1.5, -2.0], dtype=DTYPE)
        logits = shared * x + private[:, None] * y
        for family in ("NCN", "NCNC", "BUDDY"):
            records = native_record_losses(torch, logits, groups, family)
            native = tuple(reduce_native(torch, records, groups, family).unbind())
            kwargs = dict(batch=queries, groups=groups, family=family, logits=logits,
                          native_member_losses=native, route_query_signatures=[authenticate(queries, groups)] * 4, seed=103)
            unit = build_objectives(torch, mode="unit", **kwargs)
            self.assertIs(unit["shared_objective"], unit["private_objective"])
            expected_shared = torch.autograd.grad(torch.stack(native).mean(), shared, retain_graph=True)[0]
            private_gradients = []
            for mode in ("endpoint", "query_iid"):
                objectives = build_objectives(torch, mode=mode, **kwargs)
                actual_shared = torch.autograd.grad(objectives["shared_objective"], shared, retain_graph=True)[0]
                actual_private = torch.autograd.grad(objectives["private_objective"], private, retain_graph=True)[0]
                expected_private = torch.autograd.grad(
                    reduce_native(torch, records, groups, family, objectives["raw_weights"]).mean(), private, retain_graph=True)[0]
                torch.testing.assert_close(actual_shared, expected_shared)
                torch.testing.assert_close(actual_private, expected_private)
                private_gradients.append(actual_private)
                self.assertFalse(objectives["raw_weights"].requires_grad)
                self.assertEqual(objectives["diagnostics"]["optimizer_transitions"], 0)
            self.assertFalse(torch.allclose(*private_gradients))
            with self.assertRaises(ValueError):
                build_objectives(torch, mode="endpoint", **{**kwargs, "route_query_signatures": ["wrong"] * 4})
            with self.assertRaises(ValueError):
                build_objectives(torch, mode="endpoint", **{**kwargs, "native_member_losses": tuple(loss + 0.1 for loss in native)})

    def test_local_quadratic_sensitivity_and_covariance_algebra(self):
        # Fixed theta; strongly convex 2D private quadratic, uneven native groups.
        X = torch.tensor([[1., 0.], [0., 2.], [1., 1.], [-1., 2.], [2., -1.]], dtype=DTYPE)
        y = torch.tensor([1., -0.3, 0.4, -1.2, 0.8], dtype=DTYPE)
        a = torch.tensor([0.5, 0.5, 1/3, 1/3, 1/3], dtype=DTYPE)
        P = torch.eye(5, dtype=DTYPE)
        P[:2, :2] -= 0.5
        P[2:, 2:] -= 1/3
        ridge = 0.7 * torch.eye(2, dtype=DTYPE)
        H = X.T @ (a[:, None] * X) + ridge
        psi0 = torch.linalg.solve(H, X.T @ (a * y))
        G = (a * (X @ psi0 - y)) * X.T
        delta = torch.tensor([0.2, -0.4, 0.7, -0.3, 0.5], dtype=DTYPE)
        mu = 1.25
        expected = -torch.linalg.solve(H, G @ P @ delta / mu)
        errors = []
        for t in (1e-3, 1e-4, 1e-5):
            w = mu + t * delta
            r = torch.cat((w[:2] / w[:2].mean(), w[2:] / w[2:].mean()))
            psi = torch.linalg.solve(X.T @ ((a*r)[:, None] * X) + ridge, X.T @ (a*r*y))
            errors.append(float(torch.linalg.vector_norm((psi-psi0)/t-expected)))
        self.assertLess(errors[-1], 2e-6)
        self.assertLess(errors[-1], errors[0] / 50)
        queries = batch([(0, 1), (0, 2), (3, 4), (1, 3), (4, 4)])
        rows = half_incidence_rows(queries)
        columns = sorted(set().union(*(row.keys() for row in rows)))
        B = torch.tensor([[row.get(col, 0.0) for col in columns] for row in rows], dtype=DTYPE)
        A = -torch.linalg.solve(H, G @ P) / mu
        K = torch.tensor(ideal_raw_covariance(queries, mode="endpoint"), dtype=DTYPE)
        covariance = A @ K @ A.T
        torch.testing.assert_close(covariance, (A @ B) @ (A @ B).T)
        NUMERICS["local_quadratic_sensitivity"] = {"finite_difference_t": [1e-3, 1e-4, 1e-5],
            "derivative_errors": errors, "linear_proxy_covariance": covariance.tolist(),
            "scope": "synthetic fixed-theta strongly convex quadratic only"}

    def test_bounded_hash_moments_and_irregular_normalization(self):
        # 30k independent seed labels are a numerical hash check, not iid proof.
        queries = batch([(0, 1), (0, 2), (3, 4), (0, 0), (0, 1)], [0, 0, 0, 0, 1])
        count = 30_000
        for mode in ("endpoint", "query_iid"):
            sums = torch.zeros(5, dtype=DTYPE)
            outer = torch.zeros((5, 5), dtype=DTYPE)
            normalized_sum = torch.zeros(3, dtype=DTYPE)
            gap_sum = gap_square_sum = 0.0
            for seed in range(count):
                values = torch.tensor(raw_responsibilities(queries, seed, mode=mode, members=1)[0], dtype=DTYPE)
                sums += values
                outer += values[:, None] * values[None, :]
                normalized = values[:3] / values[:3].mean()
                normalized_sum += normalized
                gap = float(normalized[2] - (normalized[0] + normalized[1]) / 2)
                gap_sum += gap
                gap_square_sum += gap * gap
            means = sums / count
            covariance = (outer - count * means[:, None] * means[None, :]) / (count - 1)
            expected = torch.tensor(ideal_raw_covariance(queries, mode=mode), dtype=DTYPE)
            self.assertLess(float((means - 1.25).abs().max()), 0.025)
            self.assertLess(float((covariance - expected).abs().max()), 0.04)
            normalized_means = normalized_sum / count
            gap_mean = gap_sum / count
            gap_se = ((gap_square_sum - count * gap_mean**2) / (count-1) / count)**0.5
            if mode == "endpoint":
                self.assertGreater(gap_mean, 0.02)
                self.assertGreater(gap_mean / gap_se, 5)
            else:
                self.assertLess(abs(gap_mean), 0.02)
            NUMERICS[mode] = {"seed_labels": count, "floor": 0.25, "raw_means": means.tolist(),
                "raw_covariance": covariance.tolist(), "analytic_raw_covariance": expected.tolist(),
                "normalized_means_first_three_simple_graph_edges": normalized_means.tolist(),
                "normalized_disjoint_minus_incident_pair_mean": gap_mean,
                "monte_carlo_standard_error_of_gap": gap_se,
                "interpretation": "synthetic deterministic hash moment check; no dataset or predictive evidence"}


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SemanticChecks))
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "python": platform.python_version(),
        "torch": torch.__version__, "device": "CPU", "numerics": NUMERICS,
        "no_native_model_or_dataset_import": True, "optimizer_transitions": 0,
        "numerical_tolerances": {"maximum_absolute_raw_mean_error": 0.025,
            "maximum_absolute_raw_covariance_error": 0.04,
            "maximum_local_derivative_error_at_smallest_t": 2e-6,
            "minimum_irregular_endpoint_gap_over_descriptive_standard_error": 5},
        "files": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                  for name in ("__init__.py", "sampler.py", "objectives.py", "checks.py")}}
    (Path(__file__).parent / "CHECKS.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
