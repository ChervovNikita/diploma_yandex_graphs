"""Synthetic source-specific assignment, state, order and controller checks."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adapters.continuous_affinity import (ARMS, SOURCE_ARMS, SOURCE_FILES, SOURCE_CONTRACT,
    make_adapter, objective_module, receipt, target_payload, validate_config)
from compile_continuous_affinity_matrix import DOMAIN, I4, METHODS, make_study

PROJECT = Path(os.environ.get("GNNM_REFERENCE_PROJECT", Path(__file__).resolve().parents[2]))
NUMERIC = all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch", "torch_geometric"))


def source_bindings(root, copy_sources=False):
    result = {}
    for key, (relative, digest) in SOURCE_FILES.items():
        source = PROJECT / relative
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise AssertionError("Pinned affinity dependency changed: " + relative)
        if copy_sources:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        result[key] = {"path": relative, "bytes": len(raw), "sha256": digest}
    return result


def metadata_fixture(root, dataset="wikics"):
    files, splits = [], []
    def record(name):
        files.append({"path": name, "bytes": 1, "sha256": hashlib.sha256(name.encode()).hexdigest()})
        return name
    graph = record("graph.pt")
    for split in range(20 if dataset == "wikics" else 10):
        splits.append({"split_id": split, "node_inputs": {"graph": graph,
            **{r: record(str(split) + "/" + r + ".pt") for r in ("train", "select")}},
            "endpoint_inputs": {r: record(str(split) + "/" + r + ".pt") for r in ("cal_ids", "cal_targets", "evaluate_ids", "evaluate_targets")},
            "evaluate_purpose": "development", "role_class_counts": {"train": {str(i): 4 for i in range(3)}}})
    custody = {"dataset_id": dataset, "status": "prepared", "benchmark_fits": 0, "predictive_scores": 0,
        "request_hash": "a" * 64, "model_graph": {"classes": 3}, "files": files, "splits": splits,
        "request": {"graph_policy": {"synthetic": True}, "evaluation_exposure": "development"}}
    bindings = source_bindings(PROJECT)
    targets = {"schema": "continuous-affinity-target-custody-v1", "dataset_id": dataset,
        "source_contract": SOURCE_CONTRACT, "request_hash": custody["request_hash"], "source_bindings": bindings,
        "SELECT_CAL_TEST_tensors_opened": False, "model_constructed": False, "optimizer_created": False,
        "splits": [{"split_id": row["split_id"], "target": {"path": "targets/" + str(row["split_id"]) + ".pt", "bytes": 1, "sha256": "b" * 64}} for row in splits]}
    profile = {"reservation_mib": 1024, "wall_seconds": 1000, "estimated_seconds_per_update": .01,
               "evidence": "synthetic controller declaration"}
    sources = {"git_commit": "synthetic", "files": [{"path": "fixture.py", "sha256": "c" * 64}]}
    return make_study(root, custody, root / "data", targets, sources, bindings, {"*": profile}, "runtime")


class AffinityMetadataTests(unittest.TestCase):
    def test_pinned_source_and_complete_two_provider_roster(self):
        self.assertEqual(len(source_bindings(PROJECT)), 7)
        helper = objective_module(PROJECT / SOURCE_FILES["objective"][0])
        self.assertEqual(helper.TEMPERATURE, .2)
        self.assertEqual(helper.TEACHER_TEMPERATURE, .2)
        self.assertEqual(helper.COEFFICIENT, .05)
        self.assertEqual(len(DOMAIN), 9)
        self.assertIn({"learning_rate": .001, "dropout": .2}, DOMAIN)
        totals = {}
        with tempfile.TemporaryDirectory() as directory:
            for dataset, comparative in (("wikics", 1320), ("amazon-photo", 660)):
                study, counts, banks = metadata_fixture(Path(directory), dataset)
                self.assertEqual(counts["optimizer_starts_by_stage"], {"search": 234, "comparative": comparative})
                self.assertEqual(counts["comparative_banks"], (20 if dataset == "wikics" else 10) * 3 * 13)
                self.assertEqual({b["arm"] for b in banks}, set(ARMS) | {"ordinary_m1", "factorized_m1", "shared4_unchanged", "original_boundary", "ordinary_i4", "factorized_i4"})
                rows = {r["run_id"]: r for r in study["runs"]}
                self.assertTrue({METHODS[a] for a in ARMS} <= {r["method"] for r in study["runs"]})
                for bank in banks:
                    if bank["genuine_I4"]:
                        members = [rows[i] for i in bank["fit_parents"]]
                        self.assertEqual([m["body"] for m in members], [0, 1, 2, 3])
                        self.assertTrue(all(m["config"] == members[0]["config"] for m in members))
                        self.assertEqual(len({m["seed"] + 1000003 * m["body"] for m in members}), 4)
                    if bank["source_bound_context_assignment"]:
                        self.assertFalse(bank["body0_reused"])
                        self.assertEqual(rows[bank["fit_parents"][0]]["config"]["context_assignment_law"], SOURCE_CONTRACT["independent_assignment"])
                    if bank["arm"] == "original_boundary":
                        self.assertEqual(rows[bank["export"]]["config"]["fusion"], "mean_logits")
                for row in study["runs"]:
                    if row["kind"] == "model_fit":
                        validate_config(study, row)
                for key in ("atomic_fit_groups", "full_budget_optimizer_updates", "all_run_records", "comparative_banks"):
                    totals[key] = totals.get(key, 0) + counts[key]
        self.assertEqual(totals, {"atomic_fit_groups": 2448, "full_budget_optimizer_updates": 12240000,
            "all_run_records": 5978, "comparative_banks": 1170})

    def test_guard_precedes_project_reads_in_both_cli_modes(self):
        import compile_continuous_affinity_matrix as compiler
        argv = ["program", "--project-root", "missing", "--custody", "missing", "--source-bindings", "missing", "--output", "missing"]
        for extra in ([], ["--prepare-targets"]):
            with patch("sys.argv", argv + extra), patch.object(compiler, "verify_allocation", side_effect=RuntimeError("guard")), patch.object(Path, "read_text") as opened:
                with self.assertRaisesRegex(RuntimeError, "guard"):
                    compiler.main()
                opened.assert_not_called()


@unittest.skipUnless(NUMERIC, "Allocation scientific interpreter required for synthetic CPU checks")
class AffinitySyntheticTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        self.t, self.np = torch, np
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = source_bindings(self.root, True)
        self.helper = objective_module(self.root / SOURCE_FILES["objective"][0])
        generator = torch.Generator().manual_seed(913)
        n = 18
        self.graph = {"x": torch.randn(n, 5, generator=generator), "edge_index": torch.stack(
            [torch.arange(n).repeat(3), torch.cat([torch.arange(n), (torch.arange(n) + 1) % n, (torch.arange(n) + 7) % n])])}
        ids = torch.randperm(12, generator=generator)
        self.train = {"ids": ids, "y": (torch.arange(12) // 4)[ids]}
        self.select = {"ids": torch.arange(12, 18), "y": torch.arange(6) % 3}
        self.inputs = {}
        for name, value in (("graph", self.graph), ("train", self.train), ("select", self.select)):
            torch.save(value, self.root / (name + ".pt"))
            self.inputs[name] = receipt(self.root, name + ".pt")
        self.targets = target_payload(torch, np, self.helper, self.graph, self.train,
            {"graph": self.inputs["graph"], "train": self.inputs["train"], "objective_sha256": SOURCE_FILES["objective"][1]})
        torch.save(self.targets, self.root / "targets.pt")
        self.target_binding = receipt(self.root, "targets.pt")

    def tearDown(self):
        self.temporary.cleanup()

    def run_row(self, arm=ARMS[4], body=0):
        architecture, members, mode = SOURCE_ARMS[arm]
        return {"run_id": "synthetic-b" + str(body), "kind": "model_fit", "seed": 75101, "body": body,
            "config": {"affinity_arm": arm, "architecture": architecture, "members": members, "affinity_mode": mode,
                "backbone": "SAGE", "width": 128, "depth": 2, "weight_decay": 0., "learning_rate": .001, "dropout": .2,
                "classes": 3, "fusion": "mean_logits" if architecture == "original_gnnm" else "mean_probabilities",
                "node_inputs": self.inputs, "affinity_source": self.sources,
                "affinity_targets": self.target_binding if mode != "none" else None,
                "context_assignment_law": SOURCE_CONTRACT["independent_assignment"] if mode == "weighted_assigned" else None},
            "budget": {"unit": "optimizer_updates", "policy": "full_budget", "maximum": 5000},
            "selection": {"metric": "accuracy", "direction": "max", "ties": "first", "evaluation_interval": 1, "checkpoint0": False}}

    def make(self, arm=ARMS[4], body=0):
        return make_adapter(torch=self.t, project_root=self.root,
            study={"dataset": "synthetic_fixture", "data": {"task": "node_classification"}},
            run=self.run_row(arm, body), device="cpu")

    def test_train_only_targets_full_support_joint_order_and_rng(self):
        t, np = self.t, self.np
        np_before, torch_before = np.random.get_state(), t.get_rng_state().clone()
        arrays, metadata = self.helper.prepare_targets(np, self.graph["x"].numpy(), self.graph["edge_index"].numpy(), self.train["ids"].numpy(), self.train["y"].numpy())
        self.assertTrue(np.array_equal(np_before[1], np.random.get_state()[1]))
        self.assertTrue(t.equal(torch_before, t.get_rng_state()))
        self.assertTrue(np.array_equal(arrays["full_allowed"], ~np.eye(12, dtype=bool)))
        expected = (self.train["y"][:, None] == self.train["y"][None]).numpy() & ~np.eye(12, dtype=bool)
        self.assertTrue(np.array_equal(arrays["positive"], expected))
        self.assertTrue((arrays["weights"][:, expected] > 0).all())
        self.assertFalse(arrays["weights"][:, ~expected].any())
        self.assertFalse(metadata["targets_use_VALID_or_predictions"])
        order = self.targets["single_order"]
        bank = self.targets["banks"]["bank"]["weights"]
        self.assertTrue(t.equal(self.targets["banks"]["single"]["weights"], bank[:, order[:, None], order[None, :]]))
        bad_labels = self.train["y"].numpy().copy()
        bad_labels[0] = 9
        with self.assertRaisesRegex(ValueError, "nonself classmate"):
            self.helper.prepare_targets(np, self.graph["x"].numpy(), self.graph["edge_index"].numpy(), self.train["ids"].numpy(), bad_labels)

    def test_assigned_Qm_consumption_and_only_own_hidden_gradient(self):
        t = self.t
        targets = self.targets["banks"]["bank"]
        hidden = t.randn(4, 12, 128, requires_grad=True)
        private = self.helper.context_auxiliary(t, t.nn.functional, hidden, targets, "weighted_private")
        for body in range(4):
            assigned = self.helper.context_auxiliary(t, t.nn.functional, hidden[body:body + 1], targets, "weighted_assigned", context_index=body)
            self.assertTrue(t.equal(assigned[0], private[1][body]))
            gradient = t.autograd.grad(assigned[0], hidden, retain_graph=True)[0]
            self.assertGreater(float(gradient[body].abs().sum()), 0)
            self.assertEqual(float(gradient[[i for i in range(4) if i != body]].abs().sum()), 0.)
            self.assertEqual(assigned[3]["context_target_support_pair_reductions"], 1)
        self.assertTrue(all(not value.requires_grad for value in targets.values()))
        for index in (None, True, -1, 4):
            with self.assertRaises(ValueError):
                self.helper.context_auxiliary(t, t.nn.functional, hidden[:1], targets, "weighted_assigned", context_index=index)

    def test_all_source_arms_controls_live_gradients_no_extra_parameters_or_forward(self):
        t = self.t
        parameter_counts = {}
        for arm in SOURCE_ARMS:
            adapter = self.make(arm)
            before = t.get_rng_state().clone()
            result = adapter.train_step(1)
            self.assertTrue(t.equal(before, t.get_rng_state()))
            counts = result["operation_counts"]
            self.assertEqual(counts["optimizer_updates"], 1)
            self.assertEqual(counts["model_forwards"], SOURCE_ARMS[arm][1])
            self.assertEqual(counts["backward_calls"], 1)
            parameter_counts[arm] = sum(p.numel() for p in adapter.model.parameters())
            if SOURCE_ARMS[arm][2] != "none":
                self.assertEqual(counts["context_extra_native_forwards"], 0)
                self.assertEqual(counts["context_serving_calls"], 0)
                body = adapter.model.source_model.body if SOURCE_ARMS[arm][1] == 4 else adapter.model.source_model
                self.assertIsNotNone(body.input_linear.weight.grad)
                if SOURCE_ARMS[arm][0] != "single":
                    self.assertIsNotNone(body.input_linear.r.grad)
        self.assertEqual(parameter_counts[ARMS[0]], parameter_counts["ordinary_m1"])
        self.assertEqual(parameter_counts[ARMS[1]], parameter_counts["factorized_m1"])
        self.assertEqual(parameter_counts[ARMS[4]], parameter_counts["shared4_unchanged"])

    def test_real_engine_restore_binds_assigned_index_targets_and_persistent_streams(self):
        from engine import inspect_independence, restore_checkpoint, run_trajectory
        t = self.t
        independent = [self.make(ARMS[3], body) for body in range(4)]
        inspect_independence(independent)
        self.assertEqual([a.context_index for a in independent], [0, 1, 2, 3])
        adapter = independent[2]
        contract = {"fit_id": "synthetic-affinity-assigned2", "budget_steps": 2, "budget_unit": "optimizer_updates",
            "validation_interval": 1, "validate_initial": False, "prefixes": [], "inherited_fit_ids": [], "adapter_spec": {}}
        result = run_trajectory(t, adapter, contract, self.root / "trajectory")
        self.assertEqual(result["operation_counts"]["optimizer_updates"], 2)
        replacement = self.make(ARMS[3], 2)
        restore_checkpoint(t, self.root / "trajectory/selected.pt", replacement, expected_fit_id=contract["fit_id"])
        first = replacement.train_step(3)
        restore_checkpoint(t, self.root / "trajectory/selected.pt", replacement, expected_fit_id=contract["fit_id"])
        self.assertEqual(first, replacement.train_step(3))
        state = copy.deepcopy(replacement.capture_state())
        wrong = copy.deepcopy(state)
        wrong["external"]["context_index"] = 1
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            replacement.restore_state(wrong)
        with t.no_grad():
            replacement.affinity_targets["weights"].add_(.01)
        with self.assertRaises(ValueError):
            replacement.restore_state(state)

    def test_actual_bank_export_accepts_only_complete_source_assigned_I4(self):
        from engine import run_trajectory
        from adapters import node_endpoints
        t = self.t
        study = {"study_id": "synthetic-affinity-export", "dataset": "synthetic_fixture", "data": {"task": "node_classification"}}
        output = self.root / "output"
        records, parents = {}, []
        for body in range(4):
            row = self.run_row(ARMS[3], body)
            row["adapter"] = "synthetic-factory:make_adapter"
            identifier = row["run_id"]
            parents.append(identifier)
            adapter = self.make(ARMS[3], body)
            contract = {"fit_id": identifier, "budget_steps": 1, "budget_unit": "optimizer_updates",
                "validation_interval": 1, "validate_initial": False, "prefixes": [], "inherited_fit_ids": [], "adapter_spec": {}}
            result = run_trajectory(t, adapter, contract, output / study["study_id"] / identifier)
            records[identifier] = {"state": "complete", "contract": row, "result": result}
        class Registry:
            def run(self, study_id, identifier):
                self.assert_study = study_id
                return records[identifier]
        paths = {}
        for role, ids in (("cal", t.tensor([12, 13])), ("evaluate", t.tensor([14, 15]))):
            path = self.root / (role + "_ids.pt")
            t.save(ids, path)
            paths[role + "_ids"] = path.name
        export = {"parents": parents, "config": {"fit_parents": parents, "endpoint_inputs": paths, "fusion": "mean_probabilities"}}
        directory = self.root / "export"
        directory.mkdir()
        with patch.object(node_endpoints, "load_factory", return_value=make_adapter):
            result = node_endpoints.export_bank(project_root=self.root, study=study, run=export,
                output_root=output, output_dir=directory, registry=Registry(), device="cpu")
        self.assertTrue(result["independent_complete_learners"])
        self.assertFalse(result["CAL_targets_read"])
        self.assertFalse(result["evaluation_targets_read"])
        self.assertEqual(len({row["actual_model_seed"] for row in result["ancestry"]}), 4)
        bank = t.load(directory / "cal_logits.pt", weights_only=True)
        self.assertEqual(tuple(bank["logits"].shape), (4, 2, 3))


if __name__ == "__main__":
    unittest.main()
