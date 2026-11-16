"""Regression checks for the bounded owner-approved additions."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from ablation.common.tabm_trainer import train_step, member_logits
from ablation.p01_placement.test_p01 import toy_bundle
from ablation.p05_hparam_grid.selected import build_selected
from .plan import historical_spec, MODELS
from .analyses import transfer_report
from .owner_reports import build_report
from .tasks import output, write_json


class OwnerUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_node_contrast_all_backbones_binary_and_multiclass_and_sampler_resume(self):
        data, train, val, _, _ = toy_bundle(80)
        data.train_mask = train[:, 0]
        data.val_mask = val[:, 0]
        for binary in (False, True):
            if binary:
                data.y = (data.y % 2).float()
            for backbone in MODELS:
                with self.subTest(binary=binary, model=backbone):
                    spec = historical_spec("roman-empire", backbone, 2, 0, "gnnm", contrast=True)
                    spec["hidden_dim"] = 64
                    model, members = build_selected(spec, data, 1 if binary else 3, torch.device("cpu"))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
                    self.assertTrue(torch.isfinite(torch.tensor(train_step(model, data, optimizer, binary, members))))
                    saved = copy.deepcopy(model.state_dict())
                    sampler = int(model.auxiliary_step)
                    model.auxiliary_step.add_(10)
                    model.load_state_dict(saved)
                    self.assertEqual(int(model.auxiliary_step), sampler)
                    self.assertEqual(member_logits(model, data, members).shape[0], 4)

    def test_node_contrast_exact_training_resume(self):
        from ablation.common.resumable import JobStore
        from ablation.common.tabm_trainer import TrainSettings, train_resumable
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            spec = historical_spec("roman-empire", "SAGE", 2, 0, "gnnm", contrast=True)
            spec["hidden_dim"] = 64
            bundle = toy_bundle(80)
            settings = TrainSettings(dataset="roman-empire", split=0, seed=19, members=4,
                                     lr=3e-5, num_steps=4, patience=50, selection_interval=1,
                                     save_every_steps=1, defer_test=True, save_selected=True)
            def build():
                return build_selected(spec, bundle[0], 3, torch.device("cpu"))[0]
            direct = JobStore(root/"direct")
            expected = train_resumable(settings, direct, bundle, torch.device("cpu"), build)
            interrupted = JobStore(root/"resume")
            def interrupt(step):
                if step == 2:
                    raise InterruptedError("fixture preemption")
            with self.assertRaises(InterruptedError):
                train_resumable(settings, interrupted, bundle, torch.device("cpu"), build, on_step=interrupt)
            actual = train_resumable(settings, interrupted, bundle, torch.device("cpu"), build)
            self.assertEqual(expected["val_metric"], actual["val_metric"])
            self.assertEqual(expected["best_step"], actual["best_step"])
            for name, tensor in direct.load_state()["model"].items():
                torch.testing.assert_close(tensor, interrupted.load_state()["model"][name], rtol=0, atol=0)

    def test_optional_reference100_does_not_invalidate_complete_early_stop(self):
        scores = [dict(pair="fixture", cohort=0, settings={
            method: dict(val_metric=.5, test_metric=.5) for method in ("BASE", "ENS", "gnnm")},
            expected_settings=["BASE", "ENS", "gnnm"], reference100_settings={},
            unresolved_convergence=[], missing_methods=[], complete=True)]
        report = transfer_report(scores)
        self.assertTrue(report["complete"])
        self.assertFalse(report["horizons"]["reference100"]["complete"])

    def test_reports_write_real_figures_from_fixture_receipts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            rows = [
                ("node", dict(dataset="amazon-ratings", model="GAT", split=0, num_layers=1,
                              val_metric=.6, test_metric=.6, unresolved_convergence=[])),
                ("div/gnnm", dict(dataset="roman-empire", model="SAGE", split=0, num_layers=2,
                                 role="val", ensemble_metric=.8, mean_member_metric=.7,
                                 ensemble_gain=.1, pair_disagreement=.2, error_correlation=.3,
                                 probability_correlation=.4, js_divergence=.02))]
            for name, row in rows:
                write_json(row, output(root, name)/"done.json")
            for action, task in (("historical_contrast_report","node"),
                                 ("diversity_report","div/gnnm")):
                out = root/action
                out.mkdir()
                report = build_report(action, dict(results=[task]), root, out)
                self.assertTrue(report["complete"])
                self.assertEqual(len(report["plots"]), 1)
                self.assertGreater((out/report["plots"][0]).stat().st_size, 1000)

    def test_optional_horizon_is_not_a_final_scientific_failure(self):
        from .plan import Builder
        from .schema import SCHEMA
        from .store import CampaignStore
        from .analyses import final_report
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            b = Builder()
            b.add("transfer", "transfer_report", {})
            b.add("final", "final_report", dict(reports=["transfer"]), optional=["transfer"])
            plan = dict(schema=SCHEMA, campaign_id="fixture", tasks=list(b.tasks.values()))
            store = CampaignStore(root,plan)
            store.prepare()
            store.write_state("transfer",status="complete")
            write_json(dict(complete=True,horizons=dict(reference100=dict(required=False,complete=False))),
                       output(root,"transfer")/"done.json")
            report = final_report(dict(reports=["transfer"]),root,output(root,"final"),plan)
            self.assertTrue(report["scientific_complete"])
            self.assertTrue((output(root,"final")/"training_steps.csv").exists())

    def test_buddy_patience_starts_immediately(self):
        from ablation.p11_collab_buddy.test_p11 import BuddyTests, Stop
        from ablation.p11_collab_buddy.run import parse_args, make_jobs, train_job
        from ablation.common.resumable import JobStore
        from ablation.common.worker import Job
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            args = parse_args(["--root", str(root), "--cohorts","0","--device","cpu","--convergence","--defer-test"])
            config = make_jobs(args, "fixture", "fixture")[4].config
            fixture = BuddyTests().fixture()
            store = JobStore(root/"fit")
            store.ensure_config(config)
            with patch("ablation.p11_collab_buddy.run.evaluate_scores", return_value=.5):
                result = train_job(Job("gnnm",config), store, Stop(), args, fixture, root)
            self.assertEqual(result["stopped_epoch"],51)
            self.assertEqual(result["stale_validation_checks"],50)
            self.assertFalse(result["reference100_available"])
            self.assertEqual(result["stop_reason"],"patience")


if __name__ == "__main__":
    unittest.main()
