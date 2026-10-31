"""Molecular schema, role custody and engine-interface checks; no fitting."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from engine import _counts, _metrics

from adapters.molecular import (MolecularAdapter, MolecularEpochMeans, bound_file,
                                validate_config, validate_independent_roster, validate_pack,
                                validate_role_authorities, make_adapter)
from engine import restore_checkpoint, run_trajectory

try:
    import torch
except ImportError:
    torch = None


def receipt(path):
    return {"path": path, "bytes": 0, "sha256": "0" * 64}


def config(arm="single"):
    value = {
        "molecular_inputs": {"train": receipt("roles/train.pt"), "select": receipt("roles/select.pt")},
        "molecular_source": {key: receipt("archived/"+key+".py") for key in ("models", "factors", "objectives")},
        "model": {"hidden": 256, "layers": 5, "dropout": .5, "outputs": 1},
        "optimizer": {"name": "Adam", "learning_rate": .001, "weight_decay": 0., "eps": 1e-8, "betas": [.9, .999]},
        "training": {"batch_size": 128, "eval_batch_size": 128, "views_per_update": 2},
        "arm": arm, "fusion": "mean_logits", "initialization_seed": 7101,
        "dropout_seed": 307102, "shuffle_seed": 7101, "head_policy": "archived_scalar",
        "target_policy": "complete_binary",
        "contrastive": {"max_objects": 512, "alignment_weight": .05, "residual_weight": .05, "temperature": .2},
    }
    if "independent4_body" in arm:
        value["independence_group"] = "fixture-i4"
    return value


def run(arm="single"):
    return {"run_id": "molecular-fixture", "body": 0, "kind": "model_fit", "config": config(arm),
            "budget": {"unit": "epochs", "maximum": 100, "policy": "full_budget"},
            "selection": {"metric": "rocauc", "direction": "max", "ties": "first",
                          "evaluation_interval": 1, "checkpoint0": False}}


class MolecularContractTests(unittest.TestCase):
    def test_archived_full_budget_recipe_admitted_without_numerical_imports(self):
        row = run("be_init_contrastive")
        validated = validate_config(row["config"], row)
        self.assertEqual(validated["training"]["views_per_update"], 2)
        validated["model"]["hidden"] = 17
        self.assertEqual(row["config"]["model"]["hidden"], 256)

    def test_fit_rejects_heldout_roles_and_extra_config(self):
        row = run()
        for role in ("cal", "dev", "test", "evaluate"):
            changed = copy.deepcopy(row)
            changed["config"]["molecular_inputs"][role] = receipt("roles/"+role+".pt")
            with self.assertRaisesRegex(ValueError, "only separate TRAIN"):
                validate_config(changed["config"], changed)
        row["config"]["test_targets"] = receipt("test.pt")
        with self.assertRaisesRegex(ValueError, "heldout"):
            validate_config(row["config"], row)

    def test_full_epoch_and_fusion_laws_are_explicit(self):
        for key, value in (("unit", "optimizer_updates"), ("policy", "minimum_then_patience")):
            row = run()
            row["budget"][key] = value
            with self.assertRaisesRegex(ValueError, "full-epoch"):
                validate_config(row["config"], row)
        row = run()
        row["config"]["fusion"] = "mean_probabilities"
        with self.assertRaisesRegex(ValueError, "serving law"):
            validate_config(row["config"], row)

    def test_multitask_head_and_missing_label_adaptation_are_declared(self):
        row = run()
        row["config"]["model"]["outputs"] = 12
        with self.assertRaisesRegex(ValueError, "adapted head"):
            validate_config(row["config"], row)
        row["config"]["head_policy"] = "declared_multitask_linear"
        row["config"]["target_policy"] = "observed_binary_mask"
        validate_config(row["config"], row)
        row["config"]["arm"] = "be_unit_contrastive"
        with self.assertRaisesRegex(ValueError, "one complete binary"):
            validate_config(row["config"], row)

    def test_receipts_reject_unbound_and_outside_inputs(self):
        row = run()
        row["config"]["molecular_inputs"]["train"]["path"] = "../train.pt"
        with self.assertRaisesRegex(ValueError, "project-relative"):
            validate_config(row["config"], row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("archived source\n")
            item = {"path": "source.py", "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(bound_file(directory, item), path.resolve())
            path.write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "receipt changed"):
                bound_file(directory, item)

    def test_pack_receipts_match_prospective_role_authority(self):
        cfg = config()
        study = {"data": {"task": "graph_binary_classification",
                           "roles": copy.deepcopy(cfg["molecular_inputs"])}}
        validate_role_authorities(study, cfg)
        study["data"]["roles"]["train"] = receipt("a_different_split.pt")
        with self.assertRaisesRegex(ValueError, "declared role authority"):
            validate_role_authorities(study, cfg)

    def test_epoch_mean_includes_tail_and_does_not_report_last_batch(self):
        means = MolecularEpochMeans()
        means.add({"objective": 1.2, "ce": 1., "regularization": .2, "alignment": 2., "residual": 2.},
                  graphs=128, observed_targets=128)
        means.add({"objective": 3.6, "ce": 3., "regularization": .6, "alignment": 6., "residual": 6.},
                  graphs=1, observed_targets=1)
        result = means.finish()
        self.assertAlmostEqual(result["ce"], 131/129)
        self.assertNotEqual(result["ce"], 3.)
        self.assertEqual(result["sample_count"], 129)
        self.assertEqual(result["operation_counts"]["optimizer_updates"], 2)

    def test_missing_targets_use_a_separate_observed_target_denominator(self):
        means = MolecularEpochMeans()
        means.add({"objective": 1., "ce": 1., "regularization": 0., "alignment": 0., "residual": 0.},
                  graphs=2, observed_targets=2)
        means.add({"objective": 3., "ce": 3., "regularization": 0., "alignment": 0., "residual": 0.},
                  graphs=1, observed_targets=4)
        result = means.finish()
        self.assertAlmostEqual(result["objective"], 5/3)
        self.assertAlmostEqual(result["ce"], 14/6)
        self.assertEqual((result["ce_denominator"], result["objective_denominator"]), (6, 3))

    def test_empty_epoch_or_nonfinite_batch_cannot_succeed(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            MolecularEpochMeans().finish()
        with self.assertRaises(ValueError):
            MolecularEpochMeans().add({"objective": float("nan"), "ce": 1., "regularization": 0.,
                                       "alignment": 0., "residual": 0.}, graphs=1, observed_targets=1)

    def test_whole_epoch_result_satisfies_engine_metric_and_count_interface(self):
        class PlainValuesOnly:
            @staticmethod
            def is_tensor(value):
                return False
        means = MolecularEpochMeans()
        means.add({"objective": 1., "ce": .9, "regularization": .1, "alignment": 1., "residual": 1.},
                  graphs=7, observed_targets=7)
        result = means.finish()
        accepted = _metrics(PlainValuesOnly(), result)
        counts = _counts(result["operation_counts"], positive_updates=True)
        self.assertEqual(accepted["ce_denominator"], 7)
        self.assertEqual(counts["optimizer_updates"], 1)
        self.assertEqual(counts["observed_targets"], 7)

    def test_independent_i4_requires_four_distinct_paid_seeded_trajectories(self):
        members = []
        for body in range(4):
            row = run("factorized_independent4_body")
            row["run_id"] += str(body)
            row["body"] = body
            for key in ("initialization_seed", "dropout_seed", "shuffle_seed"):
                row["config"][key] += 1009*body
            members.append(row)
        study = {"runs": members}
        validate_independent_roster(study, members[0])
        with self.assertRaisesRegex(ValueError, "four distinct paid"):
            validate_independent_roster({"runs": members[:3]}, members[0])
        members[3]["config"]["initialization_seed"] = members[0]["config"]["initialization_seed"]
        with self.assertRaisesRegex(ValueError, "distinct initialization_seed"):
            validate_independent_roster(study, members[0])

    def test_selector_is_own_trajectory_first_tie_and_never_shortens_budget(self):
        adapter = object.__new__(MolecularAdapter)
        adapter.selector = {"metric": "rocauc", "direction": "max", "ties": "first"}
        adapter._best = adapter._best_step = None
        first = adapter.observe_validation(1, {"selection_metric": .7})
        tied = adapter.observe_validation(2, {"selection_metric": .7})
        worse = adapter.observe_validation(3, {"selection_metric": .6})
        self.assertTrue(first["save_selected"])
        self.assertFalse(tied["save_selected"])
        self.assertFalse(worse["stop_requested"])
        self.assertEqual(worse["best_saved_step"], 1)


@unittest.skipIf(torch is None, "Torch unavailable; no runtime installation for these tests")
class MolecularPackTests(unittest.TestCase):
    def pack(self):
        return {"x": torch.zeros((3, 9), dtype=torch.long),
                "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "edge_attr": torch.zeros((2, 3), dtype=torch.long),
                "node_ptr": torch.tensor([0, 2, 3]), "edge_ptr": torch.tensor([0, 2, 2]),
                "ids": torch.tensor([10, 20]), "y": torch.tensor([0., 1.])}

    def test_variable_graphs_and_edgeless_tail_graph_are_preserved(self):
        value = validate_pack(torch, self.pack(), 1, "complete_binary")
        self.assertEqual(tuple(value["y"].shape), (2, 1))
        self.assertEqual(value["edge_ptr"].tolist(), [0, 2, 2])

    def test_cross_graph_edges_and_duplicate_ids_are_rejected(self):
        pack = self.pack()
        pack["edge_index"][0, 0] = 2
        with self.assertRaisesRegex(ValueError, "graph-local"):
            validate_pack(torch, pack, 1, "complete_binary")
        pack = self.pack()
        pack["ids"][1] = 10
        with self.assertRaisesRegex(ValueError, "identities"):
            validate_pack(torch, pack, 1, "complete_binary")

    def test_missing_binary_targets_are_opt_in_and_each_graph_needs_a_label(self):
        pack = self.pack()
        pack["y"] = torch.tensor([[0., float("nan")], [float("nan"), 1.]])
        validate_pack(torch, pack, 2, "observed_binary_mask")
        with self.assertRaisesRegex(ValueError, "complete binary"):
            validate_pack(torch, pack, 2, "complete_binary")
        pack["y"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "observed binary"):
            validate_pack(torch, pack, 2, "observed_binary_mask")


MOLECULAR_RUNTIME = torch is not None and all(importlib.util.find_spec(name) is not None
    for name in ("numpy", "torch_geometric", "ogb", "sklearn"))


@unittest.skipUnless(MOLECULAR_RUNTIME, "Molecular providers unavailable; no dependency installation or scientific run")
class MolecularCPUFixtureTests(unittest.TestCase):
    def pack(self, first_id, molecules, labels):
        # OGB feature indices for real neutral sp3 carbon/oxygen categories:
        # atomic number, chirality, total degree, formal charge, total H,
        # radical electrons, hybridization, aromaticity, ring membership.
        carbon = [5, 0, 4, 5, 3, 0, 2, 0, 0]
        internal_carbon = [5, 0, 4, 5, 2, 0, 2, 0, 0]
        methane = [5, 0, 4, 5, 4, 0, 2, 0, 0]
        oxygen = [7, 0, 2, 5, 1, 0, 2, 0, 0]
        atom_rows, edge_rows, bond_rows, node_ptr, edge_ptr = [], [], [], [0], [0]
        for name in molecules:
            atoms = ([methane] if name == "methane" else [carbon, carbon] if name == "ethane"
                     else [carbon, oxygen] if name == "methanol" else [carbon, internal_carbon, oxygen])
            edges = ([] if name == "methane" else [(0, 1), (1, 0)] if name in ("ethane", "methanol")
                     else [(0, 1), (1, 0), (1, 2), (2, 1)])
            atom_rows.extend(atoms)
            edge_rows.extend(edges)
            # OGB single bond, no stereo, not conjugated.
            bond_rows.extend([[0, 0, 0]] * len(edges))
            node_ptr.append(len(atom_rows))
            edge_ptr.append(len(edge_rows))
        return {"x": torch.tensor(atom_rows, dtype=torch.long),
                "edge_index": torch.tensor(edge_rows, dtype=torch.long).t().contiguous(),
                "edge_attr": torch.tensor(bond_rows, dtype=torch.long),
                "node_ptr": torch.tensor(node_ptr), "edge_ptr": torch.tensor(edge_ptr),
                "ids": torch.arange(first_id, first_id+len(molecules)),
                "y": torch.tensor(labels, dtype=torch.float32)}

    def test_real_source_full_epoch_select_and_exact_engine_restore(self):
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_dir = root / "source"
                source_dir.mkdir()
                research = Path(os.environ.get("GNNM_REFERENCE_PROJECT", str(Path(__file__).resolve().parents[2]))).resolve()
                archived = research / "portable_internal_be_public_interface_20261007_v2/core"
                row = run("be_init")
                row["config"]["model"].update(hidden=8, layers=2, dropout=.2)
                row["config"]["training"].update(batch_size=2, eval_batch_size=2)
                row["budget"]["maximum"] = 1
                for name in ("models", "factors", "objectives"):
                    path = source_dir / (name+".py")
                    shutil.copyfile(archived / path.name, path)
                    self.assertEqual(path.read_bytes(), (archived / path.name).read_bytes())
                    row["config"]["molecular_source"][name] = {"path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                roles = {"train": self.pack(10, ("methane", "ethane"), [0., 1.]),
                         "select": self.pack(20, ("ethanol", "methanol"), [0., 1.])}
                for role, value in roles.items():
                    path = root / (role+".pt")
                    torch.save(value, path)
                    row["config"]["molecular_inputs"][role] = {"path": path.name,
                        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                study = {"dataset": "engineering-four-molecule-fixture", "runs": [row],
                         "data": {"task": "graph_binary_classification",
                                  "roles": copy.deepcopy(row["config"]["molecular_inputs"])},
                         "source": {"files": list(row["config"]["molecular_source"].values())}}
                adapter = make_adapter(torch=torch, project_root=root, study=study, run=row, device="cpu")
                self.assertEqual(type(adapter.model).__name__, "MoleculeBackbone")
                engine_spec = {"fit_id": row["run_id"], "budget_steps": 1, "budget_unit": "epochs",
                    "validation_interval": 1, "validate_initial": False, "prefixes": [],
                    "inherited_fit_ids": [], "adapter_spec": {"config": row["config"]}, "recovery_interval": 1}
                output = root / "engineering-only-output"
                result = run_trajectory(torch, adapter, engine_spec, output)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["completed_steps"], 1)
                self.assertEqual(result["operation_counts"]["optimizer_updates"], 1)
                self.assertEqual(result["operation_counts"]["supervised_examples"], 2)
                self.assertEqual(result["selected"]["step"], 1)
                predictions = adapter.predict_pack(adapter.select).clone()
                state = adapter.capture_state()
                with torch.no_grad():
                    for parameter in adapter.model.parameters():
                        parameter.add_(17.)
                adapter._best = -1.
                restore_checkpoint(torch, output / "selected.pt", adapter, expected_fit_id=row["run_id"])
                self.assertTrue(torch.equal(adapter.predict_pack(adapter.select), predictions))
                self.assertEqual(adapter._best, state["selector"]["best"])
                self.assertEqual(adapter.validate(1)["selection_metric"], result["selected"]["validation"]["selection_metric"])
                self.assertTrue(result["convergence"]["diagnostic_only"])
        finally:
            torch.set_num_threads(original_threads)


if __name__ == "__main__":
    unittest.main()
