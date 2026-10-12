"""Small deterministic CPU engineering tests; these are not scientific evidence."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import (AdapterStepError, capture_checkpoint, inspect_independence,
                    inspect_ownership, restore_checkpoint, resume_trajectory, run_trajectory,
                    _equal_tree, checkpoint_receipt)

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


class TinyAdapter:
    """A deliberately simple fixture with native adapter-owned semantics."""

    def __init__(self, *, batches=1, fail_step=None, nonfinite_step=None):
        torch.manual_seed(711)
        random.seed(713)
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.2),
                                         torch.nn.Linear(3, 2))
        for parameter in self.model[0].parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.SGD((p for p in self.model.parameters() if p.requires_grad),
                                         lr=.03, momentum=.9)
        self.generator = torch.Generator().manual_seed(719)
        self.x = torch.tensor([[1., 0.], [0., 2.], [1., -1.]])
        self.y = torch.tensor([0, 1, 0])
        self.selector = {"best": None, "stale": 0}
        self.batches, self.fail_step, self.nonfinite_step = batches, fail_step, nonfinite_step
        self.calls, self.last_minibatches = [], []

    def train_step(self, step):
        if step == self.fail_step:
            raise AdapterStepError("intentional adapter failure", operation_counts={"optimizer_updates": 0})
        self.calls.append(step)
        self.model.train()
        self.model[0].eval()
        self.last_minibatches = []
        for batch in range(self.batches):
            count = min(batch + 1, len(self.x)) if self.batches > 1 else len(self.x)
            self.optimizer.zero_grad(set_to_none=True)
            jitter = torch.rand((), generator=self.generator) * .01 + random.random() * .01
            logits = self.model(self.x[:count] + jitter)
            ce = torch.nn.functional.cross_entropy(logits, self.y[:count])
            reg = .01 * self.model[2].weight.square().sum()
            objective = ce + reg
            objective.backward()
            self.optimizer.step()
            self.last_minibatches.append({"samples": count, "ce": ce.item(),
                                          "regularization": reg.item(), "objective": objective.item()})
        samples = sum(row["samples"] for row in self.last_minibatches)
        metrics = {key: sum(row[key] * row["samples"] for row in self.last_minibatches) / samples
                   for key in ("objective", "ce", "regularization")}
        if step == self.nonfinite_step:
            metrics["objective"] = float("nan")
        return {**metrics, "sample_count": samples, "operation_counts": {
            "optimizer_updates": self.batches, "backward_calls": self.batches,
            "training_samples": samples}}

    def validate(self, step):
        self.model.eval()
        with torch.no_grad():
            ce = torch.nn.functional.cross_entropy(self.model(self.x), self.y).item()
            reg = (.01 * self.model[2].weight.square().sum()).item()
        # A prescribed validation curve tests selection, never scientific quality.
        curve = [5., 4., 4.5, 4.7, 3., 2., 2.2, 2.4]
        return {"objective": ce + reg, "ce": ce, "regularization": reg,
                "selection_metric": curve[step]}

    def observe_validation(self, step, metrics):
        value = metrics["selection_metric"]
        save = self.selector["best"] is None or value <= self.selector["best"]
        if save:
            self.selector.update(best=value, stale=0)
        else:
            self.selector["stale"] += 1
        return {"save_selected": save, "stop_requested": self.selector["stale"] >= 2,
                "selector": dict(self.selector)}

    def capture_state(self):
        return {"python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "private_rng": self.generator.get_state(), "selector": dict(self.selector)}

    def restore_state(self, state):
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        self.generator.set_state(state["private_rng"])
        self.selector = dict(state["selector"])


def contract(**changes):
    value = {
        "fit_id": "engineering-fit-1", "budget_steps": 6, "budget_unit": "optimizer_updates",
        "validation_interval": 1, "validate_initial": True,
        "recovery_interval": 1,
        "prefixes": [{"prefix_id": "legacy", "cap_step": 6, "close_on_selector_stop": True},
                     {"prefix_id": "cap4", "cap_step": 4, "close_on_selector_stop": False}],
        "inherited_fit_ids": ["engineering-parent-1"],
        "adapter_spec": {"loss_scope": "sample-weighted whole epoch", "fusion": "single body",
                         "selector": "latest tied minimum, two worse validations",
                         "rng": "Python, CPU torch, private CPU generator"},
    }
    value.update(changes)
    return value


@unittest.skipIf(torch is None, "Torch unavailable; do not install a runtime for these tests")
class EngineTests(unittest.TestCase):
    def test_validation_costs_are_counted_without_training_updates(self):
        class CountedValidation(TinyAdapter):
            def validate(self, step):
                return {**super().validate(step), "operation_counts": {
                    "optimizer_updates": 0, "select_query_passes": 3}}
        with tempfile.TemporaryDirectory() as directory:
            result = run_trajectory(torch, CountedValidation(),
                contract(budget_steps=2, prefixes=[]), Path(directory)/"run")
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 2)
            self.assertEqual(result["operation_counts"]["select_query_passes"], 9)
            self.assertEqual(result["validation_attempts"], 3)
            self.assertEqual(result["validation_calls"], 3)

    def test_failed_validation_retains_known_partial_operation_costs(self):
        class FailedValidation(TinyAdapter):
            def validate(self, step):
                raise AdapterStepError("partial SELECT query failure", operation_counts={
                    "optimizer_updates": 0, "select_query_passes": 2})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"run"
            with self.assertRaisesRegex(AdapterStepError, "SELECT"):
                run_trajectory(torch, FailedValidation(),
                    contract(budget_steps=2, validate_initial=False, prefixes=[]), output)
            result = json.loads((output/"result.json").read_text())
            self.assertEqual(result["failure"]["phase"], "validation")
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 1)
            self.assertEqual(result["operation_counts"]["select_query_passes"], 2)
            self.assertEqual(result["validation_attempts"], 1)
            self.assertEqual(result["validation_calls"], 0)
            self.assertNotIn("partial_validation_operation_counts_unknown", result)

    def test_full_budget_and_immutable_prefixes(self):
        adapter = TinyAdapter()
        snapshots = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            def resource(phase, step, result):
                path = output / "prefixes/legacy/selected.pt"
                if phase == "after_step" and step >= 3:
                    snapshots.append(path.read_bytes())
                return {"example_cpu_observation": step}

            result = run_trajectory(torch, adapter, contract(), output, resource_check=resource)
            self.assertEqual(adapter.calls, [1, 2, 3, 4, 5, 6])
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 6)
            self.assertEqual(result["selected"]["step"], 5)
            self.assertEqual(result['selected_checkpoint'], checkpoint_receipt(
                output / 'selected.pt', fit_id=result['fit_id'], step=5))
            self.assertEqual(result["first_selector_stop"], 3)
            legacy, cap = result["prefixes"]
            self.assertEqual((legacy["closure_step"], legacy["selected"]["step"]), (3, 1))
            self.assertEqual((cap["closure_step"], cap["selected"]["step"]), (4, 4))
            self.assertTrue(all(data == snapshots[0] for data in snapshots))
            self.assertNotEqual(legacy["selected"]["path"], result["selected"]["path"])
            self.assertTrue((output / "final_training.pt").is_file())
            self.assertEqual(adapter.selector, {"best": 2., "stale": 0})  # full selected was restored
            self.assertTrue(result["convergence"]["full_budget_completed"])
            self.assertTrue(result["convergence"]["diagnostic_only"])
            with self.assertRaises(FileExistsError):
                run_trajectory(torch, adapter, contract(), output)

    def test_validation_interval_initial_cap_and_last(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_trajectory(torch, TinyAdapter(), contract(
                validation_interval=3, validate_initial=False,
                prefixes=[{"prefix_id": "cap4", "cap_step": 4, "close_on_selector_stop": False}],
                budget_steps=5), Path(directory) / "run")
            records = [json.loads(line) for line in (Path(directory) / "run/trace.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in records if r["kind"] == "validation"], [3, 4, 5])
            self.assertEqual(result["validation_calls"], 3)

    def test_exact_restore_including_mixed_modes_optimizer_and_rng(self):
        adapter = TinyAdapter()
        adapter.train_step(1)
        metrics = adapter.validate(1)
        decision = adapter.observe_validation(1, metrics)
        adapter.model.train()
        adapter.model[0].eval()
        payload = capture_checkpoint(torch, adapter, fit_id="restore-test", step=1,
                                     validation=metrics, decision=decision)
        expected_python = random.random()
        expected_global = torch.rand(4)
        expected_private = torch.rand(4, generator=adapter.generator)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.pt"
            torch.save(payload, path)
            with torch.no_grad():
                for parameter in adapter.model.parameters():
                    parameter.add_(10.)
            for state in adapter.optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.add_(17.)
            adapter.model.eval()
            adapter.model[0].weight.requires_grad_(True)
            random.seed(3)
            torch.manual_seed(5)
            adapter.generator.manual_seed(7)
            adapter.selector = {"best": -100., "stale": 99}
            info = restore_checkpoint(torch, path, adapter, expected_fit_id="restore-test")
            self.assertEqual(info["step"], 1)
            self.assertTrue(adapter.model.training)
            self.assertFalse(adapter.model[0].training)
            self.assertTrue(adapter.model[1].training)
            self.assertFalse(adapter.model[0].weight.requires_grad)
            self.assertTrue(all(p.grad is None for p in adapter.model.parameters()))
            self.assertEqual(random.random(), expected_python)
            self.assertTrue(torch.equal(torch.rand(4), expected_global))
            self.assertTrue(torch.equal(torch.rand(4, generator=adapter.generator), expected_private))
            with self.assertRaises(ValueError):
                restore_checkpoint(torch, path, adapter, expected_fit_id="wrong-fit")

    def test_parameter_and_independent_body_ownership(self):
        adapter = TinyAdapter()
        inspect_ownership(adapter)
        other = TinyAdapter()
        inspect_independence([adapter, other])
        with self.assertRaises(ValueError):
            inspect_independence([adapter, adapter])
        adapter.optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))]})
        with self.assertRaises(ValueError):
            inspect_ownership(adapter)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_trajectory(torch, adapter, contract(), Path(directory) / "run")
            retained = json.loads((Path(directory) / "run/result.json").read_text())
            self.assertEqual(retained["failure"]["phase"], "admission")
            self.assertEqual(retained["operation_counts"]["optimizer_updates"], 0)

    def test_distinct_parameter_objects_with_shared_storage_are_not_independent(self):
        first, second = TinyAdapter(), TinyAdapter()
        second.model[2].weight = torch.nn.Parameter(first.model[2].weight.detach())
        second.optimizer = torch.optim.SGD((p for p in second.model.parameters() if p.requires_grad), lr=.03)
        self.assertIsNot(first.model[2].weight, second.model[2].weight)
        inspect_ownership(second)
        with self.assertRaisesRegex(ValueError, "storage"):
            inspect_independence([first, second])

    def test_epoch_metrics_are_sample_weighted_and_actual_updates_retained(self):
        adapter = TinyAdapter(batches=3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = run_trajectory(torch, adapter, contract(
                budget_unit="epochs", budget_steps=1, prefixes=[]), output)
            records = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
            train = next(r for r in records if r["kind"] == "training")
            expected = sum(row["ce"] * row["samples"] for row in adapter.last_minibatches) / 6
            self.assertAlmostEqual(train["metrics"]["ce"], expected)
            self.assertNotAlmostEqual(train["metrics"]["ce"], adapter.last_minibatches[-1]["ce"])
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 3)
            self.assertEqual(result["operation_counts"]["training_samples"], 6)

    def test_failure_is_retained_with_partial_costs_and_selected_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaisesRegex(AdapterStepError, "intentional"):
                run_trajectory(torch, TinyAdapter(fail_step=3), contract(), output)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed_steps"], 2)
            self.assertEqual(result["attempted_step"], 3)
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 2)
            self.assertFalse(result["partial_operation_counts_unknown"])
            self.assertGreater(result["costs"]["wall_seconds"], 0.)
            self.assertTrue((output / "selected.pt").is_file())
            self.assertTrue((output / "prefixes/legacy/selected.pt").is_file())
            records = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["kind"], "failure")

    def test_nonfinite_failure_keeps_performed_update_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaises(FloatingPointError):
                run_trajectory(torch, TinyAdapter(nonfinite_step=3), contract(), output)
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["completed_steps"], 2)
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 3)
            self.assertFalse(result["partial_operation_counts_unknown"])

    def test_resume_restores_latest_trajectory_and_frozen_prefix_not_selected_best(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interrupted"
            spec = contract()
            with self.assertRaises(AdapterStepError):
                run_trajectory(torch, TinyAdapter(fail_step=4), spec, output)
            frozen = (output / "prefixes/legacy/selected.pt").read_bytes()
            latest = torch.load(output / "latest_trajectory.pt", weights_only=False)
            self.assertEqual(latest["step"], 3)
            self.assertEqual(latest["recovery"]["result"]["selected"]["step"], 1)
            with self.assertRaises(ValueError):
                resume_trajectory(torch, TinyAdapter(), spec, output,
                                  admission={"previous_owner_terminal": False, "previous_owner": "fake-owner"})
            changed = contract(budget_steps=7)
            with self.assertRaises(ValueError):
                resume_trajectory(torch, TinyAdapter(), changed, output,
                                  admission={"previous_owner_terminal": True, "previous_owner": "test-owner"})
            resumed = TinyAdapter()
            result = resume_trajectory(torch, resumed, spec, output,
                                       admission={"previous_owner_terminal": True,
                                                  "previous_owner": {"pid": "engineering-fixture", "status": "reaped"}})
            self.assertEqual(resumed.calls, [4, 5, 6])
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 6)
            self.assertEqual(result["trajectory_operation_counts"]["optimizer_updates"], 6)
            self.assertEqual([a["status"] for a in result["attempts"]], ["failed", "complete"])
            self.assertEqual((output / "prefixes/legacy/selected.pt").read_bytes(), frozen)
            self.assertEqual(json.loads((output / "attempts/1/result.json").read_text())["status"], "failed")
            records = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
            self.assertTrue(any(r["kind"] == "failure" and r["attempt"] == 1 for r in records))
            self.assertTrue(any(r["kind"] == "resume" and r["attempt"] == 2 and r["step"] == 3 for r in records))
            actual = capture_checkpoint(torch, resumed, fit_id=spec["fit_id"], step=5,
                                        validation=result["selected"]["validation"],
                                        decision=torch.load(output / "selected.pt", weights_only=False)["decision"])
            baseline = TinyAdapter()
            run_trajectory(torch, baseline, spec, Path(directory) / "baseline")
            expected = torch.load(Path(directory) / "baseline/selected.pt", weights_only=False)
            self.assertTrue(_equal_tree(torch, actual, expected))

    def test_resume_replay_retains_paid_costs_and_archived_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            spec = contract(prefixes=[], recovery_interval=4)
            with self.assertRaises(AdapterStepError):
                run_trajectory(torch, TinyAdapter(fail_step=4), spec, output)
            self.assertEqual(torch.load(output / "latest_trajectory.pt", weights_only=False)["step"], 0)
            result = resume_trajectory(torch, TinyAdapter(), spec, output,
                                       admission={"previous_owner_terminal": True, "previous_owner": "reaped-test-owner"})
            self.assertEqual(result["replayed_completed_steps"], 3)
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 9)
            self.assertEqual(result["trajectory_operation_counts"]["optimizer_updates"], 6)
            self.assertEqual([a["operation_counts"]["optimizer_updates"] for a in result["attempts"]], [3, 6])
            self.assertEqual(torch.load(output / "attempts/1/selected.pt", weights_only=False)["step"], 1)
            self.assertFalse(result["partial_operation_counts_unknown"])
            self.assertFalse(result["actual_costs_incomplete"])
            self.assertGreater(result["attempts"][1]["costs"]["restore_seconds"], 0.)

    def test_killed_owner_without_terminal_result_is_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            spec = contract()
            with self.assertRaises(AdapterStepError):
                run_trajectory(torch, TinyAdapter(fail_step=3), spec, output)
            # Simulate a hard-killed owner that could not write a terminal record.
            (output / "result.json").unlink()
            result = resume_trajectory(torch, TinyAdapter(), spec, output,
                                       admission={"previous_owner_terminal": True, "previous_owner": "reaped-test-owner"})
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["actual_costs_incomplete"])
            self.assertTrue(result["partial_operation_counts_unknown"])
            self.assertEqual(result["attempts"][0]["status"], "owner_ended_without_terminal_result")

    def test_controlled_boundary_interrupt_saves_current_complete_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            spec = contract(prefixes=[], recovery_interval=100)

            def interrupt(phase, step, result):
                if phase == "before_step" and step == 4:
                    raise RuntimeError("controlled stop at unit boundary")

            with self.assertRaisesRegex(RuntimeError, "controlled stop"):
                run_trajectory(torch, TinyAdapter(), spec, output, resource_check=interrupt)
            failed = json.loads((output / "result.json").read_text())
            self.assertTrue(failed["recovery_saved_at_interrupt"])
            self.assertEqual(torch.load(output / "latest_trajectory.pt", weights_only=False)["step"], 3)
            resumed = TinyAdapter()
            result = resume_trajectory(torch, resumed, spec, output,
                                       admission={"previous_owner_terminal": True, "previous_owner": "reaped-test-owner"})
            self.assertEqual(resumed.calls, [4, 5, 6])
            self.assertEqual(result["operation_counts"]["optimizer_updates"], 6)


if __name__ == "__main__":
    unittest.main()
