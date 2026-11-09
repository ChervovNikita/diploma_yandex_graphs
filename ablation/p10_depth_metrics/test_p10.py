"""Checks for the depth table.

    python -m unittest ablation.p10_depth_metrics.test_p10 -v
"""

import unittest

import pandas as pd

from ablation.p10_depth_metrics import depth_table


class DepthTableTests(unittest.TestCase):
    def paired(self):
        return pd.DataFrame([
            {'dataset': 'a', 'model': 'SAGE', 'num_layers': 2, 'split': split,
             'val_base': 0.5, 'test_base': 0.5, 'val_ens': 0.6, 'test_ens': 0.6,
             'val_gnnm': 0.7, 'test_gnnm': gnnm}
            for split, gnnm in ((0, 0.7), (1, 0.5))
        ])

    def test_gaps_are_in_points_and_averaged_over_splits(self):
        table = depth_table.cell_means(self.paired())
        self.assertEqual(table.splits[0], 2)
        self.assertAlmostEqual(table.gnnm_minus_ens[0], 0.0, places=6)
        self.assertAlmostEqual(table.gnnm_minus_base[0], 10.0, places=6)

    def test_win_counts_use_cell_means(self):
        summary = depth_table.summarise(depth_table.cell_means(self.paired()), ['num_layers'])
        self.assertEqual(summary.gnnm_beats_ens[0], 0)
        self.assertEqual(summary.gnnm_beats_base[0], 1)


if __name__ == '__main__':
    unittest.main()
