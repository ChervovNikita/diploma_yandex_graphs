"""Finite domains, own-selected I4 typing/reuse and disabled root plan checks."""
import copy
import json
from pathlib import Path
import unittest

from adapters.molecular import validate_config, validate_independent_roster
from compile_molecular_matrix import (BASE_ARMS, I4_SEARCH_MEMBER, make_plan, make_study,
                                      parameter_domain)
from test_molecular import config, receipt, run


def inputs():
    cfg = config()
    custody = {"dataset": "ogbg-molhiv", "task": "graph_binary_classification", "target_policy": "complete_binary",
        "roles": {r: receipt("roles/"+r+".pt") for r in ("train", "select", "cal", "evaluate")},
        "endpoint_inputs": {r+s: receipt("roles/"+r+s+".pt") for r in ("cal", "evaluate") for s in ("_graphs", "_targets")},
        "split_receipt": receipt("SPLITS.json"), "provider_custody_receipt": receipt("PROVIDER_CUSTODY.json"),
        "pack_descriptives": {}}
    matrix = json.loads((Path(__file__).resolve().parents[1]/"docs/MATRIX_AXES_DRAFT.json").read_text())
    recipe = next(r for r in matrix["baseline_recipe_catalogue"]["rows"] if r["recipe_id"] == "gine_residual_virtualnode_molhiv")
    source = {"git_commit": "engineering-fixture", "files": list(cfg["molecular_source"].values())}
    profile = {"reservation_mib": 1024, "wall_seconds": 100, "estimated_seconds_per_epoch": 1.,
               "estimated_export_seconds": 1., "estimated_CAL_seconds": 1., "estimated_evaluation_seconds": 1.,
               "evidence": "synthetic engineering-only scheduling assumptions"}
    exposure = {"evaluate_purpose": "development", "prior_exposure": "engineering_fixture", "protected_confirmation": False}
    return custody, recipe, source, cfg["molecular_source"], profile, exposure


class MolecularMatrixTests(unittest.TestCase):
    def test_effective_domains_keep_native_anchor_and_do_not_tune_zero_M1_residual(self):
        _, recipe, *_ = inputs()
        for arm in BASE_ARMS:
            self.assertEqual(len(parameter_domain(recipe, arm, "native_fixed")), 40)
        self.assertEqual(len(parameter_domain(recipe, "single_contrastive", "neighbors")), 360)
        self.assertEqual(len(parameter_domain(recipe, "factorized_single_contrastive", "neighbors")), 360)
        self.assertEqual(len(parameter_domain(recipe, "be_init_contrastive", "neighbors")), 1080)
        values = parameter_domain(recipe, "factorized_single_contrastive", "neighbors")
        self.assertEqual({v["contrastive"]["residual_weight"] for v in values}, {.05})

    def test_full_epoch_domains_pay_four_I4_fits_and_reuse_fixed_body0(self):
        custody, recipe, source, molecular_source, profile, exposure = inputs()
        prepared = []
        for epochs in (100, 300):
            packet = make_study(Path.cwd(), custody, recipe, source, molecular_source,
                                profile, epochs, exposure, "native_fixed")
            prepared.append(packet)
            spec, counts, banks, domains = packet
            fits = {r["run_id"]: r for r in spec["runs"] if r["kind"] == "model_fit"}
            self.assertEqual(counts["optimizer_starts_by_stage"], {"search": 640, "comparative": 60})
            self.assertEqual(counts["full_budget_epochs"], 700*epochs)
            self.assertEqual(counts["scalar_CAL_fits"], 36)
            self.assertEqual(counts["genuine_I4_banks"], 12)
            self.assertEqual(counts["reused_I4_body0_artifacts"], 12)
            self.assertEqual(counts["continuation_segments"], 0)
            self.assertEqual(len(domains), 8)
            for bank in banks:
                if bank["arm"] in I4_SEARCH_MEMBER:
                    group = [fits[identifier] for identifier in bank["fit_parents"]]
                    self.assertEqual(len(group), 4)
                    self.assertEqual({r["body"] for r in group}, {0, 1, 2, 3})
                    validate_independent_roster(spec, group[0])
                    single = next(b for b in banks if b["seed"] == bank["seed"] and b["arm"] == I4_SEARCH_MEMBER[bank["arm"]])
                    self.assertEqual(single["fit_parents"], bank["fit_parents"][:1])
                    self.assertTrue(single["singleton_reference_is_fixed_body0"])
            for fit in fits.values():
                self.assertEqual(fit["budget"], {"unit": "epochs", "policy": "full_budget", "maximum": epochs})
                self.assertEqual(set(fit["config"]["molecular_inputs"]), {"train", "select"})
        plan = make_plan(prepared, "engineering-packet", {"allocation": {"status": "fixture"}})
        self.assertFalse(plan["scientific_launch_enabled"])
        self.assertTrue(plan["human_GO_required"])
        self.assertEqual({g["gpu_assignment"] for g in plan["groups"]}, {"allocation/gpu0"})
        self.assertEqual(plan["global_gates"]["whole_search"]["required_for_stages"], ["comparative"])
        self.assertIn("evaluation", plan["global_gates"]["whole_comparative"]["required_for_stages"])
        self.assertEqual(plan["global_gates"]["whole_calibration"]["required_for_stages"], ["evaluation"])

    def test_resolved_I4_parameters_keep_declared_roster_and_reject_role_change(self):
        custody, recipe, source, molecular_source, profile, exposure = inputs()
        spec, _, _, _ = make_study(Path.cwd(), custody, recipe, source, molecular_source,
                                   profile, 100, exposure, "native_fixed")
        row = copy.deepcopy(next(r for r in spec["runs"] if r["config"].get("independence_group")))
        row["config"]["model"]["hidden"] = 512
        row["config"]["resolved_selection"] = {"selection_run_id": row["config"]["selected_config_from"],
            "config_id": "selected", "ranking_source": "whole_search", "choice_hash": "abc"}
        validate_config(row["config"], row)
        validate_independent_roster(spec, row)
        row["config"]["molecular_inputs"]["train"] = receipt("different_train.pt")
        with self.assertRaisesRegex(ValueError, "non-HPO"):
            validate_independent_roster(spec, row)

    def test_protected_or_undeclared_exposure_cannot_release_development(self):
        custody, recipe, source, molecular_source, profile, exposure = inputs()
        for bad in ({}, {**exposure, "protected_confirmation": True}, {**exposure, "evaluate_purpose": "confirmation"}):
            with self.assertRaises(ValueError):
                make_study(Path.cwd(), custody, recipe, source, molecular_source, profile, 100, bad, "native_fixed")

    def test_orchestration_is_explicit_and_unknown_scientific_fields_stay_closed(self):
        row = run("factorized_single_contrastive")
        row["config"].update(stage="search", reservation_mib=256, estimated_seconds=1.)
        validate_config(row["config"], row)
        for key, value in (("stage", "evaluation"), ("reservation_mib", 0), ("estimated_seconds", float("nan")), ("unsupported_objective", "silently_new")):
            bad = copy.deepcopy(row)
            bad["config"][key] = value
            with self.assertRaises(ValueError):
                validate_config(bad["config"], bad)


if __name__ == "__main__":
    unittest.main()
