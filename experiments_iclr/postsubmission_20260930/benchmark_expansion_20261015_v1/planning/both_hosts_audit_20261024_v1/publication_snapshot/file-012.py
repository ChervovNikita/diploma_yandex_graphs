"""Check the recovery limit and frozen command preservation without training."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import recovery_worker as r


class RecoveryTests(unittest.TestCase):
    def worker(self):
        worker = r.RecoveryWorker.__new__(r.RecoveryWorker)
        worker.recovery_config = {"repairs": {"fit": {
            "completed_epochs": 149, "selected_epoch": 32,
            "optimizer_calls": {"attempts": [236165], "completed": [236165]},
            "failed_attempt_result": {"path": "failed-attempt"},
            "failed_native_result": {"path": "failed-native"},
        }}}
        worker.plan = {"project_root": "/project"}
        worker.named_source = {"sha256": "repair-sha"}
        worker.args = SimpleNamespace(recovery_config="/config.json", recovery_config_sha256="config-sha")
        return worker

    def result(self):
        return {"finalization_only_resume": True, "training_epochs_this_invocation": 0,
                "completed_epochs": 149, "selected_epoch": 32,
                "optimizer_calls": {"attempts": [236165], "completed": [236165]},
                "optimizer_calls_this_invocation": {"attempts": [0], "completed": [0]}}

    def validate(self, result, fit_id="fit"):
        with patch.object(r.named.NamedWorker, "validate_result", return_value=("complete", {})):
            return self.worker().validate_result({"id": fit_id}, "full", result, Path("/output"))

    def test_finalization_with_no_new_updates_passes(self):
        self.assertEqual(self.validate(self.result())[0], "complete")

    def test_extra_epoch_or_update_is_rejected(self):
        for key, value in (("training_epochs_this_invocation", 1), ("completed_epochs", 150),
                           ("finalization_only_resume", False), ("selected_epoch", 33)):
            result = self.result()
            result[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(result)
        result = self.result()
        result["optimizer_calls_this_invocation"]["completed"] = [1]
        with self.assertRaises(ValueError):
            self.validate(result)

    def test_unrelated_new_fit_uses_normal_validation(self):
        self.assertEqual(self.validate({}, "new-fit")[0], "complete")

    def test_only_exact_audited_failed_attempt_is_eligible(self):
        worker = self.worker()
        state = {"status": "failed", "attempt_result": {"path": "failed-attempt"},
                 "native_result": {"path": "failed-native"}}
        def load(tasks, phase):
            worker.states = {"fit": copy.deepcopy(state)}
            worker.retry = set()
        with patch.object(r.named.NamedWorker, "load_states", side_effect=load), patch.object(r.c, "verify"):
            worker.load_states([], "full")
            self.assertEqual(worker.retry, {"fit"})
            state["attempt_result"] = {"path": "different-failed-attempt"}
            with self.assertRaises(ValueError):
                worker.load_states([], "full")

    def test_launcher_keeps_original_native_arguments(self):
        worker = self.worker()
        worker.retry = {"fit"}
        native = ["/python", "-B", "-u", "/frozen/runtime_entry.py", "--task-id", "fit", "--resume", "/output/recovery.pt"]
        invocation = {"argv": ["/python", "-B", "-u", str(r.NAMED / "launch_named.py"),
                               "--dependency-dir", "/dependency", "--", *native[3:]],
                      "native_argv": native}
        with patch.object(r.named.NamedWorker, "invocation", return_value=copy.deepcopy(invocation)):
            changed = worker.invocation({"id": "fit"}, "full", Path("/attempt"))
        self.assertEqual(changed["native_argv"], native)
        self.assertEqual(changed["argv"][changed["argv"].index("--") + 1:], native[3:])
        self.assertTrue(changed["reporting_only_finalization"])
        self.assertIn("--recovery-config", changed["argv"])


if __name__ == "__main__":
    unittest.main()
