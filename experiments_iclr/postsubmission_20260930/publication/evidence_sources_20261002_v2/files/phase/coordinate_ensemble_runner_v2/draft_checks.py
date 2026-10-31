"""Unexecuted, standard-library-only check draft; root must authorize running it.

These checks import only the coordinator's standard-library definitions. They do
not download data, import a model, or fit anything. Real qualification is still
required separately before scientific fit admission.
"""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("draft_coordinate_runner", Path(__file__).with_name("runner.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PartitionDraftChecks(unittest.TestCase):
    def test_shared_test_holdout_and_full_disjoint_cover(self):
        labels = [c for c in range(3) for _ in range(20)]
        dataset = {"test_holdout": {"seed": 4100, "fraction": 0.2},
                   "splits": [{"id": "a", "seed": 4101, "train_fraction_of_development": 0.75},
                              {"id": "b", "seed": 4102, "train_fraction_of_development": 0.75}]}
        held_out, splits = runner.split_indices(labels, dataset)
        self.assertEqual(len(held_out), 12)
        for split in splits.values():
            train, val, test = map(set, (split["train"], split["validation"], held_out))
            self.assertFalse(train & val or train & test or val & test)
            self.assertEqual(train | val | test, set(range(60)))
            self.assertEqual((len(train), len(val), len(test)), (36, 12, 12))
        self.assertEqual((held_out, splits), runner.split_indices(labels, dataset))

    def test_class_too_small_rejected(self):
        dataset = {"test_holdout": {"seed": 1, "fraction": 0.2},
                   "splits": [{"id": "a", "seed": 2, "train_fraction_of_development": 0.75}]}
        with self.assertRaises(ValueError):
            runner.split_indices([0, 0, 1, 1], dataset)

    def test_validation_floor_training_remainder(self):
        dataset = {"test_holdout": {"seed": 4100, "fraction": 0.2},
                   "splits": [{"id": "a", "seed": 4101, "train_fraction_of_development": 0.75}]}
        held_out, splits = runner.split_indices([0] * 8 + [1] * 8, dataset)
        self.assertEqual((len(splits["a"]["train"]), len(splits["a"]["validation"]), len(held_out)), (12, 2, 2))

    def test_no_test_scoring_cli_mode(self):
        mode = next(action for action in runner.parser()._actions if action.dest == "mode")
        self.assertEqual(set(mode.choices), {"acquire", "qualify", "preflight", "fit"})


if __name__ == "__main__":
    unittest.main()
