"""Finite metadata and arithmetic checks; no tensors/model/data are read."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parents[1]
spec = importlib.util.spec_from_file_location("_residual_selected_addition_checks", HERE / "run.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class Checks(unittest.TestCase):
    def test_source_and_plan_pins(self):
        plan = json.loads((HERE / "PLAN.json").read_text())
        self.assertEqual(source.sha(HERE / "PLAN.json"), source.PLAN_SHA)
        for pin in plan["source_pins"]:
            path = EXPANSION / pin["path"]
            self.assertEqual(path.stat().st_size, pin["bytes"])
            self.assertEqual(source.sha(path), pin["sha256"])
        self.assertEqual((plan["reused_banks"], plan["new_banks"], plan["result_banks"]), (30, 3, 33))
        self.assertEqual(plan["unique_historical_plus_new_training_fits"], 39)
        for job in plan["jobs"]:
            self.assertEqual(source.digest(json.loads((EXPANSION / job["source_JOB"]).read_text())), job["job_hash"])

    def test_rejects_partial_failed_reordered_or_unfinished_trajectories(self):
        jobs = [{"seed": seed, "id": "synthetic-" + str(seed)} for seed in source.SEEDS]
        outcomes = [dict(j, status="complete", returncode=0) for j in jobs]
        source.terminal_gate({"status": "complete", "outcomes": outcomes}, outcomes, jobs)
        for bad in (outcomes[:2], list(reversed(outcomes)), [dict(outcomes[0], status="running")] + outcomes[1:],
                    outcomes[:2] + [dict(outcomes[2], returncode=1)]):
            with self.assertRaises(ValueError):
                source.terminal_gate({"status": "complete", "outcomes": bad}, bad, jobs)
        with self.assertRaises(ValueError):
            source.terminal_gate({"status": "running", "outcomes": outcomes}, outcomes, jobs)
        with self.assertRaises(ValueError):
            source.terminal_gate({"status": "complete", "outcomes": []}, outcomes, jobs)

    def test_actual_pins_required_even_after_review(self):
        for bad in (None, "", "deferred", "0" * 63, "g" * 64, 17):
            with self.assertRaises(ValueError):
                source.required_hash(bad)
        self.assertEqual(source.required_hash("a" * 64), "a" * 64)

    def test_collection_requires_exact_selection_and_zero_training(self):
        pin = {"path": "synthetic-result", "bytes": 7, "sha256": "synthetic"}
        fit = {"job_id": "synthetic", "job_hash": "synthetic-job-hash", "selected_epoch": 17,
               "selected_checkpoint": {"sha256": "synthetic-selected"},
               "selected_VALID_export": {"sha256": "synthetic-export"}}
        value = {"status": "complete_with_sanity_warning", "role": "VALID", "TEST_accessed": False,
                 "job_id": fit["job_id"], "job_hash": fit["job_hash"], "selected_epoch": 17,
                 "fit_receipt": pin, "members": 4, "evaluated_VALID_graphs": 151,
                 "selected_checkpoint": fit["selected_checkpoint"], "saved_VALID_export": fit["selected_VALID_export"],
                 "actual_work": {"TRAIN_callbacks": 0, "optimizer_steps": 0}}
        source.collection_gate(value, fit, pin)
        for key, bad in (("selected_epoch", 16), ("TEST_accessed", True), ("role", "TEST"),
                         ("members", 3), ("evaluated_VALID_graphs", 150), ("close_error", "restore failed")):
            with self.assertRaises(ValueError):
                source.collection_gate(dict(value, **{key: bad}), fit, pin)
        bad = copy.deepcopy(value); bad["actual_work"]["optimizer_steps"] = 1
        with self.assertRaises(ValueError):
            source.collection_gate(bad, fit, pin)
        bad = copy.deepcopy(value); bad["selected_checkpoint"] = {"sha256": "other-state"}
        with self.assertRaises(ValueError):
            source.collection_gate(bad, fit, pin)

    def test_signed_quality_and_bce_contrasts_keep_reductions_separate(self):
        ref = {"bank": "I4", "pool_AUC": .75, "mean_member_AUC": .72,
               "uniform_graph_BCE": .7, "mean_member_BCE": .8, "saved_native_VALID_BCE": None}
        new = {"pool_AUC": .73, "mean_member_AUC": .70, "uniform_graph_BCE": .6,
               "mean_member_BCE": .9, "saved_native_VALID_BCE": .61}
        result = source.contrast(101, "genuine_native_I4", new, ref)
        self.assertAlmostEqual(result["delta_pool_AUC_pp"], -2)
        self.assertAlmostEqual(result["delta_uniform_graph_BCE"], -.1)
        self.assertAlmostEqual(result["delta_mean_member_BCE"], .1)
        self.assertIsNone(result["delta_saved_native_VALID_BCE"])

    def test_import_is_stdlib_only_and_analyzer_opens_predictions_only(self):
        tree = ast.parse((HERE / "run.py").read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertNotIn("torch", [a.name for n in top if isinstance(n, ast.Import) for a in n.names])
        analyzer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
        text = ast.get_source_segment((HERE / "run.py").read_text(), analyzer)
        self.assertNotIn("selected.pt", text)
        self.assertEqual(text.count("torch.load("), 2)
        self.assertNotIn("create_adapter", text)


if __name__ == "__main__":
    unittest.main()
