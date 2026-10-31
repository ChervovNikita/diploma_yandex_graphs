"""CPU metadata checks for actual queue layout and versioned witnesses."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from queue_support import QueueEvidence, WitnessVerifier, pin, token, reporting_token, HERE, EXPANSION


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return pin(path)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="queue-fixture-", dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.fit = "cell/long_convergence/split-0/seed-0/wrapper-x"
        self.worker = "allocation/gpu0"
        self.task = {"id": self.fit, "worker": self.worker, "kind": "fit", "descriptor": {"fixture": 1}, "runtime_job_hash": "hash"}
        self.plan = {"worker": self.worker, "project_root": str(self.root), "state_root": str(self.root / "runs"), "tasks": [self.task]}
        self.plan_path = self.root / "PLAN.json"
        write(self.plan_path, self.plan)
        self.folder = self.root / "runs/full" / token(self.fit)
        self.attempt = self.folder / "attempts/0001"
        self.output = self.attempt / "output"
        raw = {"TEST_accessed": False, "status": "complete", "job_id": self.fit}
        write(self.output / "RESULT.json", raw)
        native = write(self.attempt / "NATIVE_RESULT.json", raw)
        self.terminal = {"id": self.fit, "worker": self.worker, "phase": "full", "attempt": 1,
                         "status": "complete", "native_result": native, "output": str(self.output)}
        self.save_terminal()

    def save_terminal(self):
        attempt = write(self.attempt / "ATTEMPT_RESULT.json", self.terminal)
        write(self.folder / "STATE.json", {**self.terminal, "attempt_result": attempt})

    def load(self, plans=None):
        return QueueEvidence(self.root, plans or [self.plan_path], expected_pins={self.worker: pin(self.plan_path)["sha256"]})

    def test_exact_terminal_output_and_distinct_reporting_digest(self):
        q = self.load()
        loc = q.locate(self.fit, {"descriptor": self.task["descriptor"], "job_hash": "hash"})
        self.assertEqual(loc["result"], self.output / "RESULT.json")
        self.assertNotEqual(token(self.fit), reporting_token(self.fit))
        self.assertEqual(loc["reporting"], self.folder / "attempts/reporting" / reporting_token(self.fit))
        self.assertEqual(q.receipt()["all_retained_attempts"][0]["status"], "complete")

    def test_missing_or_running_task_is_omitted_without_reading_mutable_result(self):
        write(self.folder / "STATE.json", {"status": "running", "attempt": 2})
        q = self.load()
        self.assertEqual(q.locations, {})
        self.assertEqual(q.omitted, [{"id": self.fit, "status": "running"}])
        self.assertEqual(len(q.attempt_history), 1)

    def test_failed_attempt_with_no_native_result_remains_in_ledger(self):
        self.terminal.update(status="failed", native_result=None)
        self.save_terminal()
        q = self.load()
        self.assertEqual(q.attempt_history[0]["status"], "failed")
        self.assertIsNone(q.attempt_history[0]["optimizer_calls_this_invocation"])
        self.assertFalse(q.locations)

    def test_changed_result_and_changed_state_are_rejected(self):
        write(self.output / "RESULT.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "Live output differs"):
            self.load()
        write(self.output / "RESULT.json", json.loads((self.attempt / "NATIVE_RESULT.json").read_text()))
        state = json.loads((self.folder / "STATE.json").read_text())
        state["status"] = "failed"
        write(self.folder / "STATE.json", state)
        with self.assertRaisesRegex(ValueError, "Queue state differs"):
            self.load()

    def test_duplicate_plan_and_incomplete_registry_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.load([self.plan_path, self.plan_path])
        q = self.load()
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            q.require_registry([])
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            q.require_registry([self.fit, self.fit])

    def test_output_escape_is_rejected(self):
        self.terminal["output"] = str(self.root / "other/output")
        self.save_terminal()
        with self.assertRaisesRegex(ValueError, "output escapes"):
            self.load()


class WitnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = WitnessVerifier()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="witness-fixture-", dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.python = self.root / "ordinary.py"
        self.python.write_text("VALUE = 1\n")
        self.job = {"id": "fixture", "normalization_composition": "composition"}
        self.authority = {"schema": "normalization-reporting-runtime-authority-v1",
            "normalization_composition": "composition", "project_root": str(self.root),
            "producer_manifest": {"path": "fixture"}, "entrypoint": "entry", "normalization_source": "normalization",
            "link_extension_authority": "link", "runtime_sha256": "composed", "helper_runtime_sha256": "helper",
            "callback_sha256": "callback", "extension_sha256": "extension", "policy_sha256": "policy"}
        self.dependencies = {str(self.python): pin(self.python)["sha256"]}
        helper = {k: self.authority[k] for k in ("callback_sha256", "extension_sha256", "policy_sha256")}
        helper.update(runtime_sha256="helper", dependency_sha256=self.dependencies)
        self.actual = {"schema": "server51-composed-normalization-genuine-runtime-binding-v2", **helper,
            "runtime_sha256": "composed", "composition_producer": {"source_manifest": self.authority["producer_manifest"],
            "entrypoint": "entry", "normalization_source": "normalization", "link_extension_authority": "link",
            "unchanged_helper_runtime": copy.deepcopy(helper)}}
        self.spec = {"job_hash": self.verifier.composed.digest(self.job), "composed_runtime_authority": self.authority}

    def base(self):
        self.actual["dependency_sha256"] = self.dependencies
        self.actual["composition_producer"]["unchanged_helper_runtime"]["dependency_sha256"] = self.dependencies
        return self.verifier.original(self.job, self.spec, self.actual, verify_dependency_bytes=False)

    def check(self, recorded):
        before = copy.deepcopy(recorded)
        self.assertEqual(self.verifier.verify(self.job, self.spec, self.actual, recorded), before)
        self.assertEqual(recorded, before)

    def test_original_witness_is_preserved(self):
        self.check(self.base())

    def test_v1_witness_retains_original_producer_without_rewrite(self):
        v1 = self.verifier.modules[1]
        readme = self.root / v1.README_PROJECT_RELATIVE
        readme.parent.mkdir(parents=True)
        readme.write_bytes((EXPANSION / "link_expansion_ports_BUDDY_production_v4/README.md").read_bytes())
        self.dependencies[str(readme)] = pin(readme)["sha256"]
        value = self.base()
        value["reporting_repair"] = {**self.verifier.v1_receipt, "dependency_bytes_verified": True}
        self.check(value)

    def test_v2_accepts_declared_html_bytes_and_rejects_forged_v1(self):
        document = self.root / "declared.html"
        document.write_text("<p>Recorded dependency data.</p>\n")
        self.dependencies[str(document)] = pin(document)["sha256"]
        value = self.base()
        value["reporting_repair"] = {**self.verifier.v2_receipt, "dependency_bytes_verified": True}
        self.check(value)
        value["reporting_repair"] = {**self.verifier.v1_receipt, "dependency_bytes_verified": True}
        with self.assertRaisesRegex(ValueError, "exact pinned BUDDY README"):
            self.check(value)

    def test_unverified_or_modified_repair_receipt_is_rejected(self):
        value = self.base()
        value["reporting_repair"] = {**self.verifier.v2_receipt, "dependency_bytes_verified": False}
        with self.assertRaisesRegex(ValueError, "altered reporting repair"):
            self.check(value)

    def test_dependency_hash_and_runtime_identity_remain_checked(self):
        value = self.base()
        value["reporting_repair"] = {**self.verifier.v2_receipt, "dependency_bytes_verified": True}
        self.python.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "dependency bytes changed"):
            self.check(value)
        value["job_hash"] = "wrong"
        with self.assertRaisesRegex(ValueError, "runtime identity differs"):
            self.check(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
