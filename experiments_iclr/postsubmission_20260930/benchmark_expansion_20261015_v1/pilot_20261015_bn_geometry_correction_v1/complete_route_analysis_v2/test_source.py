"""Non-scientific synthetic checks; no torch, model, data, or prediction access."""
import copy
import importlib.util
from pathlib import Path
import unittest

definition = importlib.util.spec_from_file_location("route_analysis_source", Path(__file__).with_name("analyze.py"))
source = importlib.util.module_from_spec(definition)
definition.loader.exec_module(source)


class SourceChecks(unittest.TestCase):
    def test_repository_to_expansion_receipt_resolution(self):
        root = Path("/synthetic-repository")
        expansion = root / "experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
        for relative in ("runtime/run_fit.py", "analysis/complete_route_diagnostic_v1/INPUTS.json",
                         "pilot_20261015_bn_geometry_correction_v1/SPEC.json", "results/fresh-output"):
            self.assertEqual(source.resolve_project_path(root, relative), expansion / relative)
        absolute = expansion / "analysis_outputs/bn_route_diagnostic_v1/fit_0/RESULT.json"
        self.assertEqual(source.resolve_project_path(root, absolute), absolute)
        with self.assertRaises(ValueError):
            source.resolve_project_path(root, "../../../../../outside-repository")

    def test_terminal_gate_requires_all_nine(self):
        ids = [f"synthetic-job-{i}" for i in range(9)]
        outcomes = [{"job_id": j, "status": "complete", "exit_code": 0} for j in ids]
        end = {"status": "complete", "TEST_accessed": False, "outcomes": outcomes}
        source.terminal_gate(end, outcomes, ids)
        for field, value in (("status", "running"), ("exit_code", 1)):
            bad = copy.deepcopy(outcomes)
            bad[8][field] = value
            with self.assertRaises(ValueError):
                source.terminal_gate(dict(end, outcomes=bad), bad, ids)
        with self.assertRaises(ValueError):
            source.terminal_gate(dict(end, status="stopped_no_retry"), outcomes, ids)
        with self.assertRaises(ValueError):
            source.terminal_gate(dict(end, outcomes=outcomes[:-1]), outcomes[:-1], ids)
        with self.assertRaises(ValueError):
            source.terminal_gate(end, list(reversed(outcomes)), ids)
        failed = {"status": "failed", "actual_work": {"native_route_forward_attempts": 5,
                  "native_route_forwards_completed": 4, "optimizer_steps": 0},
                  "wall_seconds": 2.5, "failure": {"message": "synthetic failure"}}
        retained = source.collection_outcome(ids[0], failed, {"path": "synthetic-only"}, "new")
        self.assertEqual(retained["actual_work"]["native_route_forward_attempts"], 5)
        self.assertEqual(retained["wall_seconds"], 2.5)
        self.assertEqual(retained["status"], "failed")

    def test_report_gate_rejects_missing_duplicate_and_relabelled_cells(self):
        controls, methods, jobs, rows = [], [], [], []
        new_cells = [(1, 1, 0), (1, 0, 1), (1, 1, 1)]
        methods = [{"id": f"method-{i}", "cell": list(cell)} for i, cell in enumerate(new_cells)]
        for seed in source.SEEDS:
            for n in (0, 1):
                for g in (0, 1):
                    for c in (0, 1):
                        cell = (n, g, c)
                        new = cell in new_cells
                        job_id = f"synthetic-{seed}-{n}{g}{c}"
                        rows.append({"seed": seed, "cell": list(cell), "job_id": job_id,
                                     "origin": "new" if new else "reused", "status": "complete"})
                        if new:
                            jobs.append({"id": job_id, "seed": seed, "procedure": f"method-{new_cells.index(cell)}"})
                        else:
                            controls.append({"seed": seed, "cell": list(cell), "job_id": job_id})
        spec = {"methods": methods, "reused_controls": controls}
        report = {"all_24_cells": rows, "complete_new_fits": 9, "TEST_accessed": False}
        source.report_gate(report, spec, jobs)
        for bad_rows in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                source.report_gate(dict(report, all_24_cells=bad_rows), spec, jobs)
        wrong = copy.deepcopy(report)
        wrong["all_24_cells"][0]["origin"] = "new"
        with self.assertRaises(ValueError):
            source.report_gate(wrong, spec, jobs)

    def test_hand_calculated_tie_credit_and_repairs(self):
        labels = [1, 1, 0, 0]
        first, second = [0, 2, 1, 2], [3, 1, 0, 2]
        reference = source.margins(first, labels)
        candidate = source.margins(second, labels)
        self.assertEqual(reference, [-1, -2, 1, 0])
        self.assertEqual(candidate, [3, 1, 1, -1])
        comparison = source.compare_pairs(reference, candidate)
        self.assertEqual(comparison["strict_repairs"], 2)
        self.assertEqual(comparison["strict_harms"], 0)
        self.assertEqual(comparison["changed_credit_pairs"], 3)
        self.assertEqual(comparison["resolved_ties"], 1)
        self.assertEqual(comparison["delta_AUC_pp"], 37.5)
        pooled = [(a + b) / 2 for a, b in zip(first, second)]
        result, margin, member_auc = source.pair_diagnostics([first, second], pooled, labels)
        self.assertEqual(member_auc, [0.375, 0.75])
        self.assertEqual(margin, [1, -0.5, 1, -0.5])
        self.assertEqual(result["pool_reversed_despite_some_member_correct"], 1)
        self.assertEqual(result["pool_reversed_despite_strict_majority_correct"], 0)
        self.assertEqual(result["all_members_reversed"], 0)
        self.assertEqual(result["all_members_correct"], 1)
        self.assertEqual(result["member_sign_disagreement_pairs"], 2)
        self.assertEqual(result["pool_vs_members"][0]["strict_repairs"], 1)
        self.assertEqual(result["pool_vs_members"][1]["strict_harms"], 1)

    def test_summary_never_uses_two_seed_subset(self):
        rows = [{"seed": seed, "bank": "synthetic", "metric": float(i)}
                for i, seed in enumerate(source.SEEDS)]
        summary = source.summarize(rows, "bank", ["metric"])[0]["summary"]["metric"]
        self.assertEqual(summary["mean"], 1)
        self.assertEqual(summary["sample_SD"], 1)
        with self.assertRaises(ValueError):
            source.summarize(rows[:-1], "bank", ["metric"])


if __name__ == "__main__":
    unittest.main()
