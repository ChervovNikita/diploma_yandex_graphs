"""Checks for the training-step cost estimate.

    python -m unittest ablation.p08_training_steps.test_p08 -v
"""

import unittest
from pathlib import Path

import pandas as pd

from ablation.common.test_support import import_datasets

import_datasets()

from ablation.common.study import PAPER_MODELS  # noqa: E402
from ablation.p08_training_steps import estimate  # noqa: E402


ARCHIVE = Path('results_merged')

SINGLE_LOG = """Loading data...
Training TABM-roman-empire SAGE 2L h=512 lr=3e-05 split=3
Early stopping at step 900
Training TABM-roman-empire SAGE 3L h=512 lr=3e-05 split=3
Step   50 | loss 0.5
Training TABM-roman-empire SAGE 2L h=512 lr=3e-05 split=3
Early stopping at step 1200
"""

ENSEMBLE_LOG = """ENS-roman-empire GCN 1L h=512 lr=3e-05 split=1
  Training model seed 0 (seed=1000)
  [mseed=1000] Early stopping at step 700
  Training model seed 1 (seed=1001)
  [mseed=1001] Early stopping at step 800
  Training model seed 2 (seed=1002)
  Training model seed 3 (seed=1003)
  [mseed=1003] Early stopping at step 900
"""


class ParserTests(unittest.TestCase):
    def test_later_section_replaces_a_restarted_one_and_no_stop_means_none(self):
        sections = estimate.parse_log(SINGLE_LOG)
        self.assertEqual(sections[(2, 3)], {0: 1200})
        self.assertEqual(sections[(3, 3)], {})

    def test_ensemble_members_are_tracked_separately(self):
        sections = estimate.parse_log(ENSEMBLE_LOG)
        self.assertEqual(sections[(1, 1)], {0: 700, 1: 800, 2: None, 3: 900})

    def test_paper_models_match_the_studies(self):
        self.assertEqual(estimate.PAPER_MODELS, PAPER_MODELS)


class CostTests(unittest.TestCase):
    def runs(self):
        return pd.DataFrame([
            {'dataset': 'a', 'model': 'SAGE', 'method': 'gnnm', 'num_layers': 1, 'split': 0,
             'member_steps_total': 1000},
            {'dataset': 'a', 'model': 'SAGE', 'method': 'ens', 'num_layers': 1, 'split': 0,
             'member_steps_total': 4000},
        ])

    def test_ensemble_step_time_is_divided_between_members(self):
        times = pd.DataFrame([
            {'dataset': 'a', 'model': 'SAGE', 'method': 'gnnm', 'num_layers': 1, 'full_step_ms': 40.0},
            {'dataset': 'a', 'model': 'SAGE', 'method': 'ens', 'num_layers': 1, 'full_step_ms': 40.0},
        ])
        frame = estimate.attach_cost(self.runs(), times)
        self.assertAlmostEqual(frame.cost_hours[0], 40.0 * 1000 / 3.6e6)
        self.assertAlmostEqual(frame.cost_hours[1], 10.0 * 4000 / 3.6e6)

    def test_selection_uses_validation_and_prefers_the_shallower_depth_on_ties(self):
        frame = pd.DataFrame([
            {'dataset': 'a', 'model': 'SAGE', 'method': 'gnnm', 'split': 0, 'num_layers': depth,
             'val_metric': score, 'test_metric': 1 - score}
            for depth, score in ((1, 0.5), (2, 0.7), (3, 0.7))
        ])
        self.assertEqual(estimate.select_depths(frame).num_layers.tolist(), [2])


@unittest.skipUnless(ARCHIVE.exists(), 'archived results are not available')
class ArchiveTests(unittest.TestCase):
    def test_every_archived_tolokers_run_has_a_logged_stop(self):
        for method in estimate.METHODS:
            rows = pd.DataFrame(estimate.archived_runs(ARCHIVE, 'tolokers', method))
            self.assertTrue((rows.stop_source == 'log').all(), method)
            self.assertEqual(rows.members_at_limit.sum(), 0, method)

    def test_logged_stop_is_at_least_patience_after_the_best_step(self):
        rows = pd.DataFrame(estimate.archived_runs(ARCHIVE, 'tolokers', 'gnnm'))
        self.assertTrue((rows.steps_run >= rows.steps_to_best + 300).all())

    def test_runs_without_early_stopping_are_counted_at_the_limit(self):
        rows = pd.DataFrame(estimate.archived_runs(ARCHIVE, 'roman_empire', 'gnnm'))
        self.assertEqual(int(rows.members_at_limit.sum()), 400 - 331)
        self.assertLessEqual(rows.steps_run.max(), 5000)


if __name__ == '__main__':
    unittest.main()
