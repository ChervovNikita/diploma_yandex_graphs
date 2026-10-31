"""Pure molecular data-custody checks; no OGB data, providers or scoring."""
import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from prepare_molecular_data import (forbid_provider_prompts, fraction, prepare,
                                   stratified_select_cal, validate_roles)


class MolecularDataContractTests(unittest.TestCase):
    def test_select_cal_are_stable_disjoint_stratified_official_valid_subsets(self):
        ids = list(range(2, 10))
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        select, cal = stratified_select_cal(ids, labels, seed=123)
        self.assertEqual(len(select), 4)
        self.assertEqual(len(cal), 4)
        self.assertFalse(set(select) & set(cal))
        self.assertEqual(set(select) | set(cal), set(ids))
        self.assertEqual(sum(i < 6 for i in select), 2)
        self.assertEqual(sum(i >= 6 for i in select), 2)
        # Provider ordering changes do not change the public-ID ranking.
        self.assertEqual((select, cal), stratified_select_cal(ids[::-1], labels[::-1], seed=123))
        roles = {"train": [0, 1], "select": select, "cal": cal, "evaluate": [10, 11]}
        validate_roles(roles, 12)

    def test_four_role_overlap_or_uncovered_graph_is_rejected(self):
        roles = {"train": [0], "select": [1], "cal": [2], "evaluate": [3]}
        validate_roles(roles, 4)
        bad = copy.deepcopy(roles)
        bad["cal"] = [1]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_roles(bad, 4)
        with self.assertRaisesRegex(ValueError, "every provider graph"):
            validate_roles(roles, 5)

    def test_stratification_requires_complete_binary_labels_and_both_roles(self):
        with self.assertRaisesRegex(ValueError, "complete binary"):
            stratified_select_cal([0, 1], [0, float("nan")], seed=1)
        with self.assertRaisesRegex(ValueError, "both SELECT and CAL"):
            stratified_select_cal([0, 1, 2], [0, 1, 1], seed=1)
        self.assertEqual(fraction("0.5"), Fraction(1, 2))
        for value in ("nan", "0", "1", "1/0"):
            with self.assertRaises(ValueError):
                fraction(value)

    def test_preparation_rejects_other_datasets_before_importing_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Only official"):
                prepare(project_root=directory, dataset_name="fixture-not-ogb", dataset_root=directory,
                        output=Path(directory)/"out", split_seed=1)
            self.assertFalse((Path(directory)/"out").exists())

    def test_provider_prompt_is_never_implicit_cache_upgrade_consent(self):
        import builtins
        previous = builtins.input
        with forbid_provider_prompts():
            with self.assertRaisesRegex(RuntimeError, "cache/version"):
                input("Will you update the dataset?")
        self.assertIs(builtins.input, previous)


if __name__ == "__main__":
    unittest.main()
