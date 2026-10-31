"""Pure molecular data-custody checks; no OGB data, providers or scoring."""
import copy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch runtime required")
class MolecularEndpointPreparationTests(unittest.TestCase):
    def test_mock_provider_preparation_separates_graphs_from_targets_with_bound_receipts(self):
        import torch
        from adapters.molecular import bound_file, validate_prediction_pack
        from compile_molecular_matrix import bind_custody
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root/"cache"
            dataset_dir = cache/"synthetic_provider"
            dataset_dir.mkdir(parents=True)
            (dataset_dir/"provider_fixture.txt").write_text("engineering synthetic cache fixture\n")
            labels = [0., 1., 0., 0., 1., 1., 0., 1.]
            class Dataset:
                name, root = "ogbg-molhiv", str(dataset_dir)
                def __len__(self):
                    return len(labels)
                def __getitem__(self, identifier):
                    return SimpleNamespace(x=torch.zeros((1, 9), dtype=torch.long),
                        edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 3), dtype=torch.long),
                        y=torch.tensor([[labels[identifier]]]))
                def get_idx_split(self):
                    return {"train": torch.tensor([0, 1]), "valid": torch.tensor([2, 3, 4, 5]), "test": torch.tensor([6, 7])}
            provider = {"dataset_name": "ogbg-molhiv", "split_type": "scaffold", "num_tasks": 1,
                        "versions": {"torch": torch.__version__}, "programs": []}
            # prepare binds its own program inside project_root; copy it into
            # this synthetic root and patch only that custody path.
            program = root/"prepare_molecular_data.py"
            program.write_bytes(Path(__file__).resolve().parents[1].joinpath("prepare_molecular_data.py").read_bytes())
            with patch("prepare_molecular_data.provider_dataset", return_value=(Dataset(), object, [])), \
                 patch("prepare_molecular_data.provider_custody", return_value=provider), \
                 patch("prepare_molecular_data.__file__", str(program)):
                manifest = prepare(project_root=root, dataset_name="ogbg-molhiv", dataset_root=cache,
                                   output=root/"roles", split_seed=123)
            self.assertFalse(manifest["model_constructed"])
            self.assertFalse(manifest["scoring_performed"])
            self.assertEqual(set(manifest["endpoint_inputs"]), {"cal_graphs", "evaluate_graphs", "cal_targets", "evaluate_targets"})
            for role in ("cal", "evaluate"):
                graph = torch.load(bound_file(root, manifest["endpoint_inputs"][role+"_graphs"]), weights_only=True)
                target = torch.load(bound_file(root, manifest["endpoint_inputs"][role+"_targets"]), weights_only=True)
                validate_prediction_pack(torch, graph)
                self.assertNotIn("y", graph)
                self.assertEqual(set(target), {"ids", "y"})
                self.assertTrue(torch.equal(graph["ids"], target["ids"]))
            admitted, files = bind_custody(root, root/"roles/MANIFEST.json")
            self.assertEqual(admitted, manifest)
            self.assertIn("roles/EVALUATE_GRAPHS.pt", {row["path"] for row in files})
            self.assertEqual(json.loads((root/"roles/STATUS.json").read_text())["status"], "complete_dataprep")


if __name__ == "__main__":
    unittest.main()
