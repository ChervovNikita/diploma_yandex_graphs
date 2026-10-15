"""Integer/selector/manifest tests only; no tensors, models or quality fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from bookkeeping import Policy, Selector, batches_in_epoch, epoch_boundary_record
from run_pilot import build_jobs


class BookkeepingChecks(unittest.TestCase):
    def test_first_tie_and_minimum_horizon(self):
        policy = Policy("matched_pilot", 8, 4, 2, "rocauc")
        selector = Selector(policy)
        rows = [selector.observe(i, {"rocauc": value}) for i, value in
                enumerate([.7, .7, .6, .6, .6], 1)]
        self.assertEqual(selector.selected_epoch, 1)
        self.assertEqual(rows[2]["stale_validation_checks"], 0)
        self.assertFalse(rows[3]["stop"])
        self.assertTrue(rows[4]["stop"])
        self.assertEqual(rows[4]["stop_reason"], "patience")

    def test_resume_selector_and_cap_are_explicit(self):
        policy = Policy("matched_pilot", 3, 2, 10, "accuracy")
        first = Selector(policy)
        first.observe(1, {"accuracy": .3})
        second = Selector(policy)
        second.load_state(first.state())
        self.assertTrue(second.observe(2, {"accuracy": .4})["save_selected"])
        self.assertEqual(second.observe(3, {"accuracy": .5})["stop_reason"], "cap_without_patience_plateau")
        with self.assertRaises(ValueError):
            second.observe(3, {"accuracy": .6})

    def test_native_skipped_singleton_tail_is_still_traversed(self):
        self.assertEqual(batches_in_epoch(33, 32), 2)
        row = epoch_boundary_record(33, list(range(33)), 2, 1, 1)
        self.assertEqual(row["TRAIN_records_traversed"], 33)
        self.assertEqual(row["optimizer_maps"], 1)
        with self.assertRaises(ValueError):
            epoch_boundary_record(33, list(range(32)) + [0], 2, 1, 1)
        with self.assertRaises(ValueError):
            epoch_boundary_record(33, list(range(33)), 2, 2, 1)

    def test_stopped_resume_requires_zero_additional_observations(self):
        for policy, values, reason in [
                (Policy("matched_pilot", 8, 2, 2, "rocauc"), [.7, .6, .6], "patience"),
                (Policy("source_reference", 2, 2, None, "rocauc"), [.7, .6], "fixed_source_horizon"),
                (Policy("matched_pilot", 2, 1, 20, "rocauc"), [.7, .6], "cap_without_patience_plateau")]:
            selector = Selector(policy)
            for epoch, value in enumerate(values, 1):
                last = selector.observe(epoch, {"rocauc": value})
            before = copy.deepcopy(selector.state())
            restored = Selector(policy)
            restored.load_state(before)
            self.assertEqual(restored.stopping_status(), {k: v for k, v in last.items() if k != "save_selected"})
            self.assertEqual(restored.stopping_status()["stop_reason"], reason)
            self.assertEqual(restored.state(), before)
            with self.assertRaises(ValueError):
                restored.observe(restored.completed_epoch+1, {"rocauc": .99})
            self.assertEqual(restored.state(), before)

    def test_source_horizon_cannot_be_relabelled(self):
        policy = {"track": "source_reference", "maximum_epochs": 100, "minimum_epochs": 100,
                  "patience": None, "metric": "rocauc", "evaluate_TRAIN_each_epoch": True}
        self.assertEqual(Policy.from_dict(policy, "ogb").maximum_epochs, 100)
        policy["maximum_epochs"] = 120
        with self.assertRaises(ValueError):
            Policy.from_dict(policy, "ogb")

    def test_three_trials_four_bodies_four_arms_is_twenty_four(self):
        spec = {"schema": "frozen-native-comparison-pilot-v1", "id": "bookkeeping_only",
                "family": "ogb", "TEST_access": False,
                "recipe": {"id": "metadata_only", "native_seed_offsets": [0, 100003, 200003, 300007]},
                "callback_module": "not_executed.py", "data_root": "not_opened",
                "trials": [{"seed": 101, "split": 0}, {"seed": 202, "split": 0}, {"seed": 303, "split": 0}],
                "policy": {"track": "matched_pilot", "maximum_epochs": 1000, "minimum_epochs": 100,
                           "patience": 50, "metric": "rocauc"},
                "methods": [{"id": "gnnm_boundary_native"}, {"id": "aux"}, {"id": "adapter"}, {"id": "combined"}]}
        jobs, cohorts = build_jobs(spec)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len(cohorts), 3)
        self.assertEqual(sum(j["procedure"] == "native_single" for j in jobs), 12)
        for cohort in cohorts:
            self.assertEqual(cohort["M1_fit_id"], cohort["native_fit_ids"][0])
            self.assertEqual(len(cohort["native_fit_ids"]), 4)
        changed = copy.deepcopy(spec)
        changed["trials"] = changed["trials"][:2]
        with self.assertRaises(ValueError):
            build_jobs(changed)

    def test_link_planning_uses_explicit_native_policy_without_model_imports(self):
        root = Path(__file__).resolve().parents[3]
        phase = root/"postsubmission_research_20260930"
        expansion = phase/"benchmark_expansion_20261015_v1"
        roster = json.loads((phase/"reference_wrapper_redesign_20261013_v1/ROSTER.json").read_text())
        recipe = next(row for row in roster["recipes"] if row["id"] == "collab-buddy")
        policy_path = expansion/"link_policy_v1/policy.py"
        spec = {"schema": "frozen-native-comparison-pilot-v1", "id": "link_metadata_only",
            "family": "link", "TEST_access": False, "recipe": recipe,
            "callback_module": str(expansion/"link_native_integration_v1/session.py"),
            "data_root": "not_opened", "native_seed_offsets": [0, 100003, 200003, 300007],
            "trials": [{"seed": 0, "split": 0}, {"seed": 1, "split": 0}, {"seed": 2, "split": 0}],
            "link_policy_module": str(policy_path), "link_policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "policy": {"family": "link", "track": "source_reference", "maximum_epochs": 100,
                "minimum_epochs": 100, "patience": None, "metric": "hits@50", "direction": "max",
                "ties": "first", "evaluate_TRAIN_each_epoch": False,
                "evaluation_mode": "VALID_only_disclosed_adaptation"}, "methods": []}
        jobs, cohorts = build_jobs(spec, root)
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(not row["I4_required"] for row in cohorts))
        changed = copy.deepcopy(spec)
        changed.pop("link_policy_module")
        with self.assertRaises(ValueError):
            build_jobs(changed, root)
        changed = copy.deepcopy(spec)
        changed["policy"].pop("evaluation_mode")
        with self.assertRaises(ValueError):
            build_jobs(changed, root)


if __name__ == "__main__":
    unittest.main()
