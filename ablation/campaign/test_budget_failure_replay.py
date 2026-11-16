"""Numerical failures retain priced search prefixes without inventing compute."""
import random
from pathlib import Path
import unittest
from unittest.mock import patch

from ablation.campaign.analyses import budget_report


class BudgetFailureReplayTests(unittest.TestCase):
    def setUp(self):
        self.records = {}
        self.meta = {"cells": [], "profiles": []}
        for candidate in range(8):
            name = f"c{candidate}"
            profile_id = f"profile/{name}"
            self.records[profile_id] = dict(config_name=name, depth=1, profiles={
                method: {"full_step_ms": value}
                for method, value in [("BASE", 100), ("ENS", 400), ("GNNM", 200)]})
            self.meta["profiles"].append(profile_id)
            for split in range(3):
                fits = [f"fit/{name}/{split}/{member}" for member in range(5)]
                for identifier in fits:
                    self.records[identifier] = {"stopped_step": 10}
                cell_id = f"cell/{name}/{split}"
                self.records[cell_id] = dict(config_name=name, depth=1, split=split,
                    specs={method: {"lr": .001, "num_layers": 1}
                           for method in ["BASE", "ENS", "GNNM"]},
                    fit_ids=fits, scores={method: .5 + candidate * .01
                                         for method in ["BASE", "ENS", "GNNM"]})
                self.meta["cells"].append(cell_id)
        self.order = [f"c{candidate}" for candidate in range(8)]
        random.Random(20261111).shuffle(self.order)

    def fail(self, name, member=4, split=1, attempted_steps=5):
        identifier = f"fit/{name}/{split}/{member}"
        receipt = dict(scientific_status="numerical_failure", attempted_steps=attempted_steps)
        self.records[identifier] = receipt
        scores = self.records[f"cell/{name}/{split}"]["scores"]
        if member == 4:
            scores["GNNM"] = None
        else:
            scores["ENS"] = None
            if member == 0:
                scores["BASE"] = None
        return identifier

    def replay(self):
        def read_result(root, identifier):
            if identifier not in self.records:
                raise FileNotFoundError(identifier)
            return self.records[identifier]

        with patch("ablation.campaign.analyses.result", side_effect=read_result):
            return budget_report(self.meta, Path("unused_mock_root"))

    def test_successful_replay_preserves_original_budget_and_order(self):
        report = self.replay()
        self.assertEqual(report["common_gpu_seconds"], 24)
        self.assertTrue(report["accounting_complete"])
        self.assertTrue(report["complete"])
        for method, count in [("BASE", 8), ("ENS", 2), ("GNNM", 4)]:
            self.assertEqual(report["matched"][method]["completed_candidates"], count)
            self.assertEqual([row["config_name"] for row in report["full_trace"][method]], self.order)

    def test_measured_failure_consumes_compute_and_later_candidates_remain_selectable(self):
        failed = self.fail(self.order[0])
        report = self.replay()
        row = report["full_trace"]["GNNM"][0]
        self.assertEqual(row["candidate_gpu_seconds"], 5)
        self.assertEqual(row["failed_fit_ids"], [failed])
        self.assertIsNone(row["mean_valid"])
        self.assertEqual(report["matched"]["GNNM"]["completed_candidates"], 3)
        self.assertEqual(report["matched"]["GNNM"]["attempted_candidates"], 4)
        self.assertEqual(report["matched"]["GNNM"]["failed_candidates"], 1)
        self.assertTrue(report["ready_GNNM"])
        self.assertTrue(report["accounting_complete"])
        self.assertEqual(len(report["numerical_failures"]), 1)
        self.assertNotEqual(report["selections"]["GNNM"]["config_name"], self.order[0])

    def test_native_member_failure_uses_separate_native_profile(self):
        self.fail(self.order[0], member=2)
        self.records[f"profile/{self.order[0]}"]["profiles"]["ENS"] = {
            "scientific_status": "infeasible"}
        report = self.replay()
        self.assertEqual(report["full_trace"]["ENS"][0]["candidate_gpu_seconds"], 11.5)
        self.assertTrue(report["ready_BASE"])
        self.assertTrue(report["ready_ENS"])
        self.assertTrue(report["accounting_complete"])

    def test_unknown_early_failure_keeps_other_method_results_and_exact_budget(self):
        self.fail(self.order[0], attempted_steps=None)
        report = self.replay()
        self.assertFalse(report["accounting_complete"])
        self.assertFalse(report["complete"])
        self.assertTrue(report["common_budget_identified"])
        self.assertEqual(report["common_gpu_seconds"], 24)
        self.assertTrue(report["ready_BASE"])
        self.assertTrue(report["ready_ENS"])
        self.assertFalse(report["ready_GNNM"])
        self.assertIsNone(report["full_search_gpu_seconds"]["GNNM"])
        self.assertEqual(report["full_search_gpu_seconds_lower_bounds"]["GNNM"], 46)
        self.assertTrue(all(row["cumulative_gpu_seconds"] is None
                            for row in report["full_trace"]["GNNM"]))

    def test_late_unpriced_failure_keeps_verified_prefix_selection(self):
        self.fail(self.order[6], attempted_steps=None)
        report = self.replay()
        self.assertTrue(report["ready_GNNM"])
        self.assertEqual(report["matched"]["GNNM"]["completed_candidates"], 4)
        self.assertFalse(report["accounting_complete"])
        self.assertEqual(len(report["full_trace"]["GNNM"]), 8)

    def test_missing_profile_is_reported_without_borrowing_a_rate(self):
        self.fail(self.order[0])
        self.records[f"profile/{self.order[0]}"]["profiles"]["GNNM"] = {
            "scientific_status": "infeasible"}
        report = self.replay()
        self.assertFalse(report["accounting_complete"])
        self.assertTrue(report["ready_BASE"])
        self.assertTrue(report["ready_ENS"])
        self.assertFalse(report["ready_GNNM"])
        self.assertEqual(report["full_trace"]["GNNM"][0]["candidate_gpu_seconds_lower_bound"], 0)
        self.assertEqual(len(report["accounting_gaps"]["GNNM"]), 3)

    def test_unidentified_common_minimum_is_explicit_lower_bound_sensitivity(self):
        self.fail(self.order[0], member=0, attempted_steps=None)
        report = self.replay()
        self.assertFalse(report["common_budget_identified"])
        self.assertEqual(report["common_budget_mode"], "conservative_lower_bound_sensitivity")
        self.assertEqual(report["common_gpu_seconds"], 23)
        self.assertFalse(report["complete"])
        self.assertTrue(report["ready_GNNM"])
        self.assertEqual(report["matched"]["GNNM"]["completed_candidates"], 3)

    def test_method_without_feasible_candidates_does_not_disable_others(self):
        for name in self.order:
            self.fail(name)
        report = self.replay()
        self.assertTrue(report["accounting_complete"])
        self.assertFalse(report["ready_GNNM"])
        self.assertTrue(report["ready_BASE"])
        self.assertTrue(report["ready_ENS"])
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["numerical_failures"]), 8)

    def test_infrastructure_missingness_blocks_replay(self):
        del self.records[f"fit/{self.order[0]}/0/4"]
        with self.assertRaises(FileNotFoundError):
            self.replay()

    def test_null_score_without_numerical_failure_is_rejected(self):
        self.records[f"cell/{self.order[0]}/0"]["scores"]["GNNM"] = None
        with self.assertRaisesRegex(ValueError, "numerical-failure receipt"):
            self.replay()


if __name__ == "__main__":
    unittest.main()
