"""Pure finite practical-policy checks; no numerical models or benchmark data."""
import copy
from pathlib import Path
import unittest
from test_polyformer_matrix import fixture_inputs
from compile_polyformer_matrix import make_study, templates
from compile_polyformer_boundary_matrix import append_boundary
from compile_polyformer_practical_matrix import POLICY, practical_domain, practical_study
from adapters.polyformer_boundary import M4, M1


class PracticalPolicyTests(unittest.TestCase):
    def test_explicit_nine_points_all36_source_anchors_native_coefficients_and_counts(self):
        totals = []
        for dataset in ("citeseer", "pubmed"):
            custody, recipe, source, profile = fixture_inputs(dataset)
            root = Path.cwd()
            old, counts, banks, domains = make_study(root, custody, root / "tiny", recipe, source, profile,
                                                   "native_fixed", "v1/runtime")
            full, fc, fb, fd = append_boundary(old, {"counts": counts, "served_banks": banks, "search_domains": domains}, "v2/runtime", [])
            immutable = copy.deepcopy(full)
            value, total, served, declared = practical_study(full, {"counts": fc, "served_banks": fb, "search_domains": fd},
                                                           recipe, POLICY, "practical/runtime", [])
            self.assertEqual(full, immutable)
            self.assertEqual(total["optimizer_starts_by_stage"], {"search": 648, "comparative": 279})
            self.assertEqual(total["initial_training_segments"], 927)
            self.assertEqual(total["all_run_records"], 1458)
            self.assertEqual(total["search_selection_jobs"], 36)
            self.assertEqual(total["bank_exports"], 165)
            self.assertEqual(total["genuine_I4_banks"], 57)
            self.assertEqual(total["comparative_served_prediction_views"], 489)
            self.assertEqual(len(declared), 36)
            native_loss = {t["method"]: t["loss"] for t in templates()}
            for method, domain in declared.items():
                points = domain["parameter_points"]
                self.assertEqual(len(points), 9)
                self.assertEqual(sum(p["source_anchor"] for p in points), 1)
                for point in points:
                    parameter = point["parameters"]
                    self.assertEqual(parameter["loss"], native_loss.get(method, {"variant": "own", "coefficients": {}}))
                    self.assertAlmostEqual(parameter["attn_lr"] / parameter["learning_rate"],
                        POLICY["datasets"][dataset]["attn_lr"] / POLICY["datasets"][dataset]["learning_rate"])
                    self.assertEqual(parameter["native"]["hidden"], 256)
                    self.assertEqual(parameter["native"]["nlayer"], POLICY["datasets"][dataset]["nlayer"])
                    if method in (M4, M1):
                        self.assertEqual(parameter["weight_decay"], 0)
                        self.assertEqual(parameter["attn_weight_decay"], 0)
            ids = {r["run_id"] for r in value["runs"]}
            self.assertTrue(all(set(r["parents"]) <= ids for r in value["runs"]))
            self.assertTrue(all(r["config"]["selected_config_from"] in ids for r in value["runs"] if r["kind"] == "model_fit"
                               and r["config"]["stage"] == "comparative"))
            totals.append(total)
        self.assertEqual(sum(t["initial_training_segments"] for t in totals), 1854)
        self.assertEqual(sum(t["full_budget_optimizer_updates"] for t in totals), 3708000)
        self.assertEqual(sum(t["planned_TRAIN_route_forwards"] for t in totals), 12840000)
        self.assertEqual(sum(t["planned_validation_route_forwards"] for t in totals), 7992000)

    def test_policy_changes_require_new_explicit_version(self):
        _, recipe, _, _ = fixture_inputs()
        wrong = copy.deepcopy(POLICY); wrong["horizon"] = 100
        with self.assertRaisesRegex(ValueError, "frozen"):
            practical_domain(recipe, M4, wrong)


if __name__ == "__main__": unittest.main()
