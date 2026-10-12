"""Prospective link domains/counts and disabled study contracts; no science."""
import copy
import unittest
from adapters.link_search import select_configuration
from adapters.link import validate_roster
from compile_link_matrix import fit_formula, make_study, native_parameters, parameter_domain
from spec import fingerprint
from test_link import config


class LinkCompilerTests(unittest.TestCase):
    def test_default_and_legacy_exact_fit_formulas(self):
        legacy = fit_formula("legacy_widths")
        self.assertEqual(legacy["configuration_counts"], {"NCN": 2, "NCNC": 2})
        self.assertEqual((legacy["model_fits"], legacy["full_budget_epochs"], legacy["prospective_optimizer_updates"]), (38, 3800, 64600))
        broad = fit_formula()
        self.assertEqual(broad["configuration_counts"], {"NCN": 302, "NCNC": 311})
        self.assertEqual((broad["search_model_fits"], broad["comparative_model_fits"], broad["model_fits"]), (1226, 30, 1256))
        self.assertEqual((broad["full_budget_epochs"], broad["prospective_optimizer_updates"]), (125600, 2135200))
        self.assertFalse(broad["single_reuses_I4_body0"])
        self.assertEqual((broad["calibration_jobs"], broad["evaluation_jobs"]), (0, 0))

    def test_native_anchor_effective_controls_and_ignored_depth_are_declared(self):
        for predictor in ("NCN", "NCNC"):
            points = parameter_domain(predictor)
            hashes = [fingerprint(point) for point in points]
            self.assertEqual(len(hashes), len(set(hashes)))
            self.assertIn(fingerprint(native_parameters(predictor)), hashes)
            self.assertEqual({p["predictor"]["layers"] for p in points}, {3})
            self.assertEqual({p["encoder"]["taildropout"] for p in points}, {.05})
            self.assertEqual(len({p["optimizer"]["encoder_lr"] for p in points}), 3)
            self.assertEqual(len({p["optimizer"]["predictor_lr"] for p in points}), 3)
        with self.assertRaisesRegex(ValueError, "full 100"):
            fit_formula(epochs=200)

    def test_materialized_native_singles_and_i4_have_own_selected_full_paid_fits(self):
        cfg = config()
        cfg["support"].update(select_overlap="official_temporal_records", select_negative_self_policy="preserve_official_records")
        custody = {key: cfg[key] for key in ("link_inputs", "link_source", "sampler", "metric", "support")}
        source = {"git_commit": "synthetic-source-contract", "files": list(cfg["link_source"].values())}
        profile = {"reservation_mib": 1024, "wall_seconds": 1000, "estimated_seconds_per_epoch": 1,
                   "estimated_selection_seconds": 1, "evidence": "synthetic engineering fixture, not a resource measurement"}
        exposure = {"protected_confirmation": False, "evaluate_purpose": "development", "prior_exposure": "synthetic fixture"}
        spec, banks = make_study(custody, source, profile, exposure, "legacy_widths")
        fits = [row for row in spec["runs"] if row["kind"] == "model_fit"]
        self.assertEqual((len(fits), len(banks)), (38, 12))
        self.assertTrue(all(row["budget"] == {"unit": "epochs", "policy": "full_budget", "maximum": 100} for row in fits))
        self.assertTrue(all(row["selection"]["ties"] == "first" for row in fits))
        comparative = [row for row in fits if row["config"]["stage"] == "comparative"]
        self.assertEqual(len(comparative), 30)
        i4 = [bank for bank in banks if bank["arm"] == "I4"]
        self.assertTrue(all(len(bank["fit_parents"]) == 4 and bank["own_selected_checkpoints"] == 4 for bank in i4))
        self.assertFalse(spec["release"]["scientific_launch_enabled"])
        self.assertTrue(spec["release"]["human_GO_required"])
        self.assertEqual(spec["data"]["roles"]["cal"]["status"], "raw_only_absent")
        self.assertEqual(spec["data"]["roles"]["evaluate"]["status"], "sealed_authority_missing")
        self.assertFalse(spec["data"]["evidence"]["SELECT_labels_reused_as_CAL"])

    def test_whole_native_search_metric_and_loss_ties_use_synthetic_records(self):
        class Registry:
            def run(self, study_id, identifier):
                return {"state": "complete", "contract": {"config": {"stage": "search"}, "budget": {"maximum": 100}},
                    "result": {"completed_steps": 100, "selected": {"validation": {"hits@50": .5, "ce": 1. if identifier == "a" else 2.}}}}
        row = {"config": {"ranking_metric": "hits@50", "ranking_direction": "max", "configurations": [
            {"config_id": "b", "parameters": {"point": 2}, "run_ids": ["b"]},
            {"config_id": "a", "parameters": {"point": 1}, "run_ids": ["a"]}]}}
        value = select_configuration(project_root=None, study={"study_id": "synthetic"}, run=row,
            output_root=None, output_dir=None, registry=Registry(), device=None)
        self.assertEqual(value["selected_configuration"]["config_id"], "a")
        self.assertFalse(value["CAL_read"])
        self.assertFalse(value["evaluate_read"])
        self.assertEqual(value["scientific_backbone_fits"], 0)

    def test_resolved_i4_hpo_uses_frozen_declarations_and_preserves_protocol(self):
        rows=[]
        for body in range(4):
            cfg=config("native_independent4_body", "NCN")
            cfg["initialization_seed"] += body*1009
            cfg["selected_config_from"]="select-NCN"
            rows.append({"run_id":"paid-"+str(body),"body":body,"kind":"model_fit","config":cfg,
                "method":"original_NCN_I4","split":0,"adapter":"adapters/link.py:make_adapter",
                "budget":{"unit":"epochs","policy":"full_budget","maximum":100},
                "selection":{"metric":"hits@50","direction":"max","ties":"first"}})
        spec={"dataset":"ogbl-collab","data":{"task":"link_prediction","roles":{
            name:rows[0]["config"]["link_inputs"][name] for name in ("train","select")}},"runs":rows}
        actual=copy.deepcopy(rows[0])
        actual["config"]["encoder"]["hidden"]=128
        actual["config"]["optimizer"]["encoder_lr"]*=.5
        actual["config"]["resolved_selection"]={"selection_run_id":"select-NCN","config_id":"synthetic-frozen-choice"}
        validate_roster(spec,actual)
        actual["config"]["support"]["select_overlap"]="official_temporal_records"
        with self.assertRaisesRegex(ValueError,"non-HPO"):
            validate_roster(spec,actual)


if __name__ == "__main__":unittest.main()
