"""The new study excludes old dataset cells while retaining GNNM on new data."""
import unittest

from spec import validate
from test_master_plan import study


class OriginalPaperScopeTests(unittest.TestCase):
    def test_original_five_are_excluded_even_under_other_method_names(self):
        for dataset in ("Roman Empire", "Amazon_Ratings", "Minesweeper", "Questions", "Tolokers"):
            with self.subTest(dataset=dataset):
                proposal = study("scope")
                proposal["dataset"] = dataset
                with self.assertRaisesRegex(ValueError, "excluded by the user"):
                    validate(proposal)

    def test_original_gnnm_on_a_new_dataset_is_retained(self):
        proposal = study("scope")
        proposal["dataset"] = "AmazonPhoto"
        candidate = proposal["runs"][3]
        candidate["method"] = "original_boundary_GNNM"
        candidate["config"]["architecture"] = "original_gnnm"
        candidate["config"]["is_original_gnnm"] = True
        self.assertIs(validate(proposal), proposal)


if __name__ == "__main__":
    unittest.main()
