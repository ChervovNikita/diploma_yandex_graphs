"""Focused metadata checks for the new author-family controller dependency."""
import unittest
from compile_master_plan import CONTROLLER_FILES, typed_counts
from test_ogb_author_cli import fixture


class OGBMasterExtensionTests(unittest.TestCase):
    def test_author_costs_do_not_become_standardized_model_fits(self):
        value = typed_counts([fixture()])
        counts = value['totals']
        self.assertEqual(counts['run_records'], 80)
        self.assertEqual(counts['author_cli_groups'], 80)
        self.assertEqual(counts['author_independent_seed_trajectories'], 80)
        self.assertEqual(counts['author_sequential_repeat_trajectories'], 0)
        self.assertEqual(counts['author_registered_maximum_optimizer_updates'], 4268000)
        self.assertEqual(counts['model_fit_groups'], 0)
        self.assertEqual(counts['actual_author_fits'], 0)

    def test_all_new_author_runtime_dependencies_are_frozen(self):
        self.assertEqual(len(CONTROLLER_FILES), 20)
        self.assertEqual(len(set(CONTROLLER_FILES)), 20)
        for name in ('ogb_author_cli_spec.py', 'run_ogb_author_cli.py', 'ogb_declared_seed_entry.py'):
            self.assertIn(name, CONTROLLER_FILES)


if __name__ == '__main__':unittest.main()
