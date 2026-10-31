"""Masked binary serving/CAL and real tiny own-selected four-body custody."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from adapters.molecular import validate_prediction_pack
from adapters.molecular_endpoints import (binary_metrics, evaluate_bank, export_bank,
    pooled_logits, receipt, scalar_calibration, _targets)
import test_molecular as molecular_fixture
from test_molecular import MOLECULAR_RUNTIME, run


class FakeRegistry:
    def __init__(self, rows):
        self.rows = rows
    def run(self, study, identifier):
        return self.rows[identifier]


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("sklearn"), "CPU Torch/sklearn runtime required")
class MolecularBinaryEndpointTests(unittest.TestCase):
    def test_mean_raw_logits_precedes_sigmoid_and_mask_eligible_auc(self):
        import torch
        logits = torch.tensor([[[8., 1., 3.], [-2., 2., 4.], [1., 3., 5.]],
                               [[-2., 3., 5.], [-4., 4., 6.], [-1., 5., 7.]]], dtype=torch.float64)
        scores = pooled_logits(torch, logits, "mean_logits")
        self.assertTrue(torch.equal(scores, logits.mean(0)))
        self.assertFalse(torch.allclose(scores.sigmoid(), logits.sigmoid().mean(0)))
        targets = {"ids": torch.arange(3), "y": torch.tensor([[1., 1., float("nan")], [0., 1., float("nan")], [float("nan"), 1., float("nan")]])}
        y, mask = _targets(torch, targets, targets["ids"], 3, "observed_binary_mask")
        metrics = binary_metrics(torch, scores, y, mask)
        self.assertEqual(metrics["rocauc"], 1.)
        self.assertEqual(metrics["rocauc_eligible_tasks"], 1)
        self.assertEqual(metrics["rocauc_ineligible_tasks"], 2)
        self.assertEqual(metrics["observed_targets"], 5)
        self.assertEqual(metrics["missing_targets"], 4)
        self.assertEqual(metrics["proper_loss_denominator"], 5)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(scores[mask], y[mask])
        self.assertAlmostEqual(metrics["nll_observed_targets"], float(expected))
        with self.assertRaisesRegex(ValueError, "Complete binary"):
            _targets(torch, targets, targets["ids"], 3, "complete_binary")
        with self.assertRaisesRegex(ValueError, "mean_logits"):
            pooled_logits(torch, logits, "mean_probabilities")

    def test_all_missing_or_single_class_tasks_remain_explicit_ineligible(self):
        import torch
        scores = torch.zeros((2, 2), dtype=torch.float64)
        y = torch.tensor([[float("nan"), 1.], [float("nan"), 1.]], dtype=torch.float64)
        metrics = binary_metrics(torch, scores, y, torch.isfinite(y))
        self.assertIsNone(metrics["rocauc"])
        self.assertEqual(metrics["rocauc_eligible_tasks"], 0)
        self.assertEqual(metrics["rocauc_task_accounting"][0]["observed"], 0)
        y[:] = float("nan")
        metrics = binary_metrics(torch, scores, y, torch.isfinite(y))
        self.assertIsNone(metrics["nll_observed_targets"])
        self.assertIsNone(metrics["brier_observed_targets"])
        self.assertEqual(metrics["proper_loss_denominator"], 0)

    def test_graph_only_pack_cannot_carry_truth_and_preserves_local_edges(self):
        import torch
        pack = {"x": torch.zeros((3, 9), dtype=torch.long),
                "edge_index": torch.tensor([[0, 1], [1, 0]]), "edge_attr": torch.zeros((2, 3), dtype=torch.long),
                "node_ptr": torch.tensor([0, 2, 3]), "edge_ptr": torch.tensor([0, 2, 2]), "ids": torch.tensor([10, 11])}
        self.assertNotIn("y", validate_prediction_pack(torch, pack))
        with self.assertRaisesRegex(ValueError, "exact plain-tensor"):
            validate_prediction_pack(torch, {**pack, "y": torch.zeros(2)})
        pack["edge_index"][0, 0] = 2
        with self.assertRaisesRegex(ValueError, "graph-local"):
            validate_prediction_pack(torch, pack)

    def test_CAL_failure_preserves_completed_scalar_updates(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank_path = root / "cal_logits.pt"
            torch.save({"ids": torch.tensor([4, 5]), "logits": torch.tensor([[[2.], [-1.]]])}, bank_path)
            target_path = root / "cal_targets.pt"
            torch.save({"ids": torch.tensor([4, 5]), "y": torch.tensor([[0.], [1.]])}, target_path)
            inputs = {key: {"path": key+".pt", "bytes": 0, "sha256": "0"*64}
                      for key in ("cal_graphs", "evaluate_graphs", "cal_targets", "evaluate_targets")}
            inputs["cal_targets"] = receipt(target_path, root)
            study = {"study_id": "fixture", "data": {"target_policy": "complete_binary", "evidence": {"endpoint_inputs": inputs}}}
            exported = {"fusion": "mean_logits", "target_policy": "complete_binary", "files": {"cal": receipt(bank_path, root)}}
            registry = FakeRegistry({"export": {"state": "complete", "contract": {"kind": "ensemble_assembly"}, "result": exported}})
            cal_run = {"parents": ["export"], "config": {"endpoint_inputs": inputs, "target_policy": "complete_binary",
                       "fusion": "mean_logits", "export_parent": "export", "placement": "after_serving_pool", "steps": 500, "learning_rate": .01}}
            original_step, calls = torch.optim.Adam.step, []
            def failing_step(optimizer, *args, **kwargs):
                calls.append(1)
                if len(calls) == 3:
                    raise RuntimeError("injected third CAL update failure")
                return original_step(optimizer, *args, **kwargs)
            with patch.object(torch.optim.Adam, "step", failing_step):
                with self.assertRaisesRegex(RuntimeError, "third CAL"):
                    scalar_calibration(project_root=root, study=study, run=cal_run, output_root=root,
                                       output_dir=root/"out", registry=registry, device="cpu")
            costs = json.loads((root/"out/failure_costs.json").read_text())["operation_counts"]
            self.assertEqual(costs["scalar_optimizer_updates"], 2)
            self.assertEqual(costs["scalar_optimizer_update_attempts"], 3)
            self.assertEqual(costs["scalar_backward_calls"], 3)
            self.assertEqual(costs["scientific_backbone_fits"], 0)


@unittest.skipUnless(MOLECULAR_RUNTIME, "Real archived molecular CPU runtime required")
class MolecularSelectedBankFixtureTests(unittest.TestCase):
    def test_four_paid_own_selected_bodies_export_without_targets_then_CAL_and_evaluate(self):
        import torch
        from adapters.molecular import make_adapter
        from engine import run_trajectory
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = Path(__file__).resolve().parents[1]
                reference = Path(os.environ.get("GNNM_REFERENCE_PROJECT", str(package.parent)))
                source_dir = root/"source"
                source_dir.mkdir()
                molecular_source = {}
                for name in ("models", "factors", "objectives"):
                    path = source_dir/(name+".py")
                    shutil.copyfile(reference/"portable_internal_be_public_interface_20261007_v2/core"/path.name, path)
                    molecular_source[name] = receipt(path, root)
                (root/"adapters").mkdir()
                shutil.copyfile(package/"adapters/molecular.py", root/"adapters/molecular.py")
                fixture = molecular_fixture.MolecularCPUFixtureTests()
                fit_inputs = {}
                for role, ids, molecules in (("train", 10, ("methane", "ethane")), ("select", 20, ("ethanol", "methanol"))):
                    path = root/(role+".pt")
                    torch.save(fixture.pack(ids, molecules, [0., 1.]), path)
                    fit_inputs[role] = receipt(path, root)
                endpoint_inputs, withheld = {}, {}
                for role, ids in (("cal", 30), ("evaluate", 40)):
                    pack = fixture.pack(ids, ("ethanol", "methanol"), [0., 1.])
                    graph_path, target_path = root/(role+"_graphs.pt"), root/(role+"_targets.pt")
                    torch.save({k:v for k,v in pack.items() if k != "y"}, graph_path)
                    torch.save({"ids": pack["ids"], "y": pack["y"]}, target_path)
                    endpoint_inputs[role+"_graphs"] = receipt(graph_path, root)
                    endpoint_inputs[role+"_targets"] = receipt(target_path, root)
                    withheld[role] = target_path.read_bytes()
                    target_path.unlink()
                study = {"study_id": "engineering-fixture", "dataset": "engineering-molecules", "runs": [],
                    "source": {"files": list(molecular_source.values())},
                    "data": {"task": "graph_binary_classification", "roles": fit_inputs, "target_policy": "complete_binary",
                        "prior_exposure": "synthetic_engineering_fixture", "evidence": {"endpoint_inputs": endpoint_inputs}}}
                for body in range(4):
                    row = run("factorized_independent4_body_contrastive")
                    row.update(run_id="body"+str(body), body=body, seed=101, split=0,
                        method="molecular_factorized_independent4_body_contrastive", adapter="adapters/molecular.py:make_adapter", parents=[])
                    row["config"].update(molecular_inputs=fit_inputs, molecular_source=molecular_source,
                        initialization_seed=101+body, dropout_seed=1001+body, shuffle_seed=2001+body)
                    row["config"]["model"].update(hidden=8, layers=2, dropout=.2)
                    row["config"]["training"].update(batch_size=2, eval_batch_size=2)
                    row["budget"]["maximum"] = 1
                    study["runs"].append(row)
                output = root/"runs"
                rows = {}
                for row in study["runs"]:
                    adapter = make_adapter(torch=torch, project_root=root, study=study, run=row, device="cpu")
                    result = run_trajectory(torch, adapter, {"fit_id": row["run_id"], "budget_steps": 1, "budget_unit": "epochs",
                        "validation_interval": 1, "validate_initial": False, "prefixes": [], "inherited_fit_ids": [],
                        "adapter_spec": {"fixture": True}}, output/study["study_id"]/row["run_id"])
                    self.assertEqual(result["operation_counts"]["optimizer_updates"], 1)
                    self.assertEqual(result["selected"]["step"], 1)
                    self.assertEqual(result["selected"]["validation"]["regularization"], 0.)
                    trace = [json.loads(line) for line in (output/study["study_id"]/row["run_id"]/"trace.jsonl").read_text().splitlines()]
                    training = next(record["metrics"] for record in trace if record["kind"] == "training")
                    self.assertEqual(training["residual"], 0.)
                    if row["body"] == 0:
                        # HPO transfer is justified by identical live M1
                        # construction/objective/update, not by arm naming.
                        equivalent = copy.deepcopy(row)
                        equivalent["config"]["arm"] = "factorized_single_contrastive"
                        equivalent["config"].pop("independence_group")
                        standalone = make_adapter(torch=torch, project_root=root, study=study, run=equivalent, device="cpu")
                        one_step = standalone.train_step(1)
                        for key in ("objective", "ce", "alignment", "residual"):
                            self.assertEqual(one_step[key], training[key])
                        for key, value in adapter.model.state_dict().items():
                            self.assertTrue(torch.equal(value, standalone.model.state_dict()[key]))
                    rows[row["run_id"]] = {"state": "complete", "contract": row, "result": result}
                registry = FakeRegistry(rows)
                endpoint = {"endpoint_inputs": endpoint_inputs, "fusion": "mean_logits", "target_policy": "complete_binary"}
                export_run = {"parents": list(rows), "config": {**endpoint, "fit_parents": list(rows)}}
                export_dir = output/study["study_id"]/"export"
                export_dir.mkdir()
                exported = export_bank(project_root=root, study=study, run=export_run, output_root=output,
                    output_dir=export_dir, registry=registry, device="cpu")
                self.assertTrue(exported["independent_complete_learners"])
                self.assertEqual(len(exported["ancestry"]), 4)
                self.assertEqual(exported["operation_counts"]["model_forwards"], 8)
                self.assertFalse((root/"cal_targets.pt").exists())
                self.assertFalse((root/"evaluate_targets.pt").exists())
                bank = torch.load(root/exported["files"]["cal"]["path"], map_location="cpu", weights_only=True)
                self.assertEqual(tuple(bank["logits"].shape), (4, 2, 1))
                # The predeclared singleton is exactly member0 of that bank.
                singleton_run = {"parents": ["body0"], "config": {**endpoint, "fit_parents": ["body0"],
                                 "single_reference_arm": "factorized_single_contrastive"}}
                singleton_dir = output/study["study_id"]/"singleton"
                singleton_dir.mkdir()
                singleton = export_bank(project_root=root, study=study, run=singleton_run, output_root=output,
                    output_dir=singleton_dir, registry=registry, device="cpu")
                own = torch.load(root/singleton["files"]["cal"]["path"], map_location="cpu", weights_only=True)
                self.assertTrue(torch.equal(own["logits"], bank["logits"][:1]))
                self.assertTrue(singleton["singleton_reference_is_fixed_body0"])
                rows["export"] = {"state": "complete", "contract": {"kind": "ensemble_assembly"}, "result": exported}
                (root/"cal_targets.pt").write_bytes(withheld["cal"])
                cal_run = {"parents": ["export"], "config": {**endpoint, "export_parent": "export",
                           "placement": "after_serving_pool", "steps": 500, "learning_rate": .01}}
                calibrated = scalar_calibration(project_root=root, study=study, run=cal_run, output_root=output,
                    output_dir=export_dir, registry=registry, device="cpu")
                self.assertEqual(calibrated["scalar_optimizer_updates"], 500)
                self.assertFalse((root/"evaluate_targets.pt").exists())
                rows["cal"] = {"state": "complete", "contract": {"kind": "calibration"}, "result": calibrated}
                (root/"evaluate_targets.pt").write_bytes(withheld["evaluate"])
                evaluation_run = {"parents": ["export", "cal"], "config": {**endpoint,
                                  "export_parent": "export", "calibration_parent": "cal"}}
                evaluated = evaluate_bank(project_root=root, study=study, run=evaluation_run, output_root=output,
                    output_dir=export_dir, registry=registry, device="cpu")
                self.assertEqual(evaluated["metrics"]["raw"]["rocauc"], evaluated["metrics"]["calibrated"]["rocauc"])
                self.assertEqual(evaluated["metrics"]["raw"]["observed_targets"], 2)
                # Calibration cannot be replayed against changed target bytes.
                (root/"cal_targets.pt").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "receipt changed"):
                    scalar_calibration(project_root=root, study=study, run=cal_run, output_root=output,
                        output_dir=export_dir, registry=registry, device="cpu")
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
