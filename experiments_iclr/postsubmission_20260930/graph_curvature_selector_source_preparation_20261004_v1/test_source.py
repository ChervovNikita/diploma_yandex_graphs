"""Stdlib-only source/decision checks; no tensor, data or trial execution."""
import ast
from pathlib import Path
import sys
import unittest

from selector import ARMS, PAIRS, FrozenConstants, choose_pair


def constants(**changes):
    fields = dict(radius=0.01, radius_cap=0.02, rank_atol=1e-10, rank_rtol=1e-8,
                  sign_atol=1e-10, mean_logit_atol=1e-6, mean_logit_rtol=1e-5,
                  d_positive_min=1e-10, d_match_atol=1e-9, d_match_rtol=1e-5,
                  bisection_steps=32, tie_tolerance=0.01, abstention_tolerance=0.02)
    fields.update(changes)
    return FrozenConstants(**fields)


def row(index, value, status="eligible"):
    return dict(pair_index=index, post_trial_pooled_train_ce=value, status=status)


class SourceChecks(unittest.TestCase):
    def test_fixed_pair_order_and_complete_controls(self):
        self.assertEqual(PAIRS, ((0, 1), (0, 2), (1, 2)))
        self.assertEqual(len(ARMS), 5)
        self.assertIn("fixed_first_graph_pair", ARMS)

    def test_earliest_tie_against_true_minimum(self):
        winner, reason = choose_pair([row(0, 0.808), row(1, 0.800), row(2, 0.799)], 1.0, constants())
        self.assertEqual((winner, reason), (0, "selected"))

    def test_strict_common_trial_abstention(self):
        self.assertEqual(choose_pair([row(0, 0.981)], 1.0, constants()),
                         (None, "common_trial_abstention"))

    def test_failed_and_nonfinite_trials_are_retained_but_not_selected(self):
        self.assertEqual(choose_pair([row(0, 0.1, "trial_failure"), row(1, float("nan")),
                                      row(2, 0.7)], 1.0, constants())[0], 2)

    def test_no_eligible_pair_abstains(self):
        self.assertEqual(choose_pair([row(0, None, "rank_or_D0_failure")], 1.0, constants()),
                         (None, "no_eligible_pair"))

    def test_unbounded_or_invalid_constants_rejected(self):
        for change in (dict(radius=0), dict(radius_cap=0.001), dict(bisection_steps=0),
                       dict(d_match_rtol=float("inf")), dict(sign_atol=0)):
            with self.assertRaises(ValueError):
                constants(**change).validate()

    def test_compile_and_import_without_numerical_libraries(self):
        import driver
        for name in ("selector.py", "driver.py", "test_source.py"):
            path = Path(__file__).parent/name
            tree = ast.parse(path.read_text(), filename=str(path))
            compile(tree, str(path), "exec")
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
                    self.assertFalse(any(name.startswith(("torch", "numpy")) for name in names))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertEqual(driver.SEEDS, (17, 29, 43))


if __name__ == "__main__":
    unittest.main()
