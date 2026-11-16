"""Regression checks for the frozen primary reporting and selection contract."""
import unittest
from unittest.mock import patch
from .analyses import transfer_report, comparison
from .decisions import select

class ReportingContracts(unittest.TestCase):
    def test_transfer_horizons_and_extension_against_original(self):
        scores = []
        for cohort in range(3):
            values = {"BASE": .5, "ENS": .6, "gnnm": .65, "private_bn": .7}
            scores.append(dict(pair="molecular", cohort=cohort,
                settings={name: dict(val_metric=value, test_metric=value) for name, value in values.items()},
                reference100_settings={name: dict(val_metric=value-.1, test_metric=value-.1) for name,value in values.items()},
                unresolved_convergence=["gnnm"] if cohort==2 else []))
        report = transfer_report(scores)
        self.assertAlmostEqual(report["paired_comparisons"]["molecular/private_bn-minus-gnnm"]["mean"], .05)
        self.assertIn("reference100", report["horizons"])
        self.assertAlmostEqual(report["horizons"]["reference100"]["settings"]["molecular/gnnm/test"]["mean"], .55)
        self.assertEqual(report["unresolved_convergence"][0]["cohort"], 2)
        self.assertFalse(report["complete"])

    def test_feasible_methods_remain_ready_when_ensemble_infeasible(self):
        cells = [dict(task_id=f"cell{split}", role="val", config_name="a", depth=1, split=split,
                      scores={"BASE": .5, "ENS": None, "GNNM": .6},
                      specs={"BASE": {}, "ENS": {}, "GNNM": {}},
                      fit_ids=[f"fit{split}_{member}" for member in range(5)],
                      unresolved_convergence=[]) for split in range(3)]
        selected = select(cells)
        self.assertTrue(selected["ready_BASE"])
        self.assertTrue(selected["ready_GNNM"])
        self.assertFalse(selected["ready_ENS"])
        self.assertFalse(selected["ready"])

    def test_resolver_uses_method_readiness_when_another_method_fails(self):
        from .tasks import resolve_spec
        spec = dict(num_layers=2, lr=.001, weight_decay=0,
                    dropout=.2, attention_dropout=.2, num_heads=8)
        decision = dict(ready=False, ready_BASE=True, selections={"BASE": {"spec": spec}})
        with patch("ablation.campaign.tasks.result", return_value=decision):
            resolved = resolve_spec(dict(spec={}, resolve={"selection": "x", "setting": "BASE"}), None)
        self.assertEqual(resolved, spec)

    def test_architecture_primary_difference_points_from_original_to_alternative(self):
        rows = []
        for setting,value in (("GNNM",.7),("heads_single_pass",.6),("BASE",.5),("ENS",.65)):
            for split in range(3,6):
                rows.append(dict(dataset="roman-empire",model="GAT",stage="architecture_confirmation",
                    setting=setting,split=split,val_metric=value,test_metric=value,
                    unresolved_convergence=[]))
        contrasts=comparison(rows)["paired_comparisons"]
        key="roman-empire/GAT/selected/GNNM-minus-heads_single_pass/test"
        self.assertAlmostEqual(contrasts[key]["mean"],.1)

if __name__=="__main__":
    unittest.main()
