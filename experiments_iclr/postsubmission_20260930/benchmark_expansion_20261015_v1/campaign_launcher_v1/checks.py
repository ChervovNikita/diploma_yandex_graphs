"""Stdlib synthetic scheduler checks. Popen/system probes are always mocked."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import contracts as c
import supervisor as s


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        def write(name, value):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True) + "\n")
        self.write = write
        for name in c.RUNTIME_FILES:
            path = self.root / c.RUNTIME / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Synthetic source fixture; never imported or executed.\n")
        for name in ("contracts.py", "supervisor.py"):
            path = self.root / c.LAUNCHER / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).with_name(name), path)
        (self.root / "callback.py").write_text("# Synthetic callback; never imported.\n")
        write("evidence.json", {"synthetic_only": True, "not_resource_or_readiness_evidence": True})
        self.resources = {"vram_bytes": 100, "ram_bytes": 200, "disk_bytes": 50,
            "measured": True, "mature_optimizer": True, "pair_qualified": True,
            "profile": "synthetic", "qualified_peer_profiles": ["synthetic"], "max_concurrent_owned_children": 2,
            "evidence": c.pin(root, "evidence.json")}
        self.cost = {"measured": True, "observed_wall_seconds": 0.125, "observed_optimizer_calls": 3,
            "scope": "synthetic scheduler fixture, not measured native work", "evidence": c.pin(root, "evidence.json")}
        self.fits = []
        bindings = {}
        for body in range(4):
            job_id = f"synthetic/long_convergence/split-0/seed-10/native-body-{body}"
            contract = {"body": body, "dataset": "synthetic", "recipe_id": "synthetic",
                "initialization_seed": 10+body, "model_procedure": "native_single", "split": 0,
                "track": "long_convergence", "native_config": {"hidden": 2},
                "training": {"maximum": 1000, "unit": "epochs"}}
            self.fits.append({"id": job_id, "kind": "fit", "contract": contract,
                "contract_hash": c.digest(contract), "dependencies": [], "resources": {},
                "relative_cost": {"measured": False}})
            job = {"schema": "native-full-epoch-fit-v1", "id": job_id, "TEST_access": False,
                "family": "ogb", "device": "cuda:0", "recipe": {"id": "synthetic", "dataset": "synthetic", "native_fixed_config": {"hidden": 2}},
                "procedure": "native_single", "seed": 10+body, "split": 0, "body": body,
                "callback_module": "callback.py", "callback_sha256": c.sha(self.root/"callback.py"),
                "policy": {"track": "matched_extended", "maximum_epochs": 1000}}
            relative = f"jobs/job-{body}.json"
            write(relative, job)
            bindings[job_id] = {"job": c.pin(root, relative), "resources": self.resources, "measured_cost": self.cost}
        endpoint = {"id": "synthetic/evaluate-roster", "kind": "evaluate_roster"}
        self.plan = {"schema": "reference-wrapper-campaign-v2", "workers": list(c.WORKERS), "slots_per_gpu": 2,
            "automatic_retries": 0, "human_GO_required": True, "scientific_launch_enabled": False,
            "jobs": self.fits + [endpoint], "groups": [{"worker": c.WORKERS[0], "job_ids": [job["id"] for job in self.fits]+[endpoint["id"]]}],
            "logical_banks": [{"id": "synthetic/native-I4", "fit_ids": [job["id"] for job in self.fits]}]}
        write("plan.json", self.plan)
        sources = {f"{c.RUNTIME}/{name}": c.sha(self.root/c.RUNTIME/name) for name in c.RUNTIME_FILES}
        sources.update({f"{c.LAUNCHER}/{name}": c.sha(self.root/c.LAUNCHER/name) for name in ("contracts.py", "supervisor.py")})
        sources["callback.py"] = c.sha(self.root/"callback.py")
        self.bindings = {"schema": "finite-campaign-execution-bindings-v1", "source_plan": c.pin(root, "plan.json"),
            "source_sha256": sources, "fits": bindings,
            "python_executable_sha256_by_worker": {worker: c.sha(sys.executable) for worker in c.WORKERS}}
        self.manifest = c.compile_manifest(root, "plan.json", self.bindings, "synthetic", "outputs")
        write("manifest.json", self.manifest)
        budget = {"gpu_headroom_bytes": 10, "host_ram_headroom_bytes": 20, "disk_headroom_bytes": 10}
        self.readiness = {"schema": "finite-campaign-readiness-v1", "manifest_sha256": c.sha(self.root/"manifest.json"),
            "source_bundle_sha256": self.manifest["source_bundle_sha256"], "workers": {c.WORKERS[0]: {
                "gates": {gate: {"status": "PASS", "evidence": c.pin(root, "evidence.json")} for gate in c.GATES}, "budget": budget}}}
        write("readiness.json", self.readiness)
        self.go = {"schema": "finite-campaign-human-GO-v1", "user_explicit_GO": True,
            "manifest_sha256": c.sha(self.root/"manifest.json"), "source_bundle_sha256": self.manifest["source_bundle_sha256"],
            "readiness": c.pin(root, "readiness.json"), "allowed_workers": [c.WORKERS[0]],
            "review_evidence": "SYNTHETIC fixture only: not an actual human campaign GO"}
        write("go.json", self.go)


class FakeProcess:
    counter = 0
    instances = []
    failure_first = False
    def __init__(self, command, **kwargs):
        self.pid = 90_000 + FakeProcess.counter
        self.number = FakeProcess.counter
        FakeProcess.counter += 1
        FakeProcess.instances.append(self)
        self.output = Path(command[command.index("--output")+1])
        self.job = json.loads(Path(command[command.index("--job")+1]).read_text())
        self.polls = 0
        self.code = 1 if self.number == 0 and FakeProcess.failure_first else 0
        self.finished = False
    def finish(self):
        self.output.mkdir(parents=True)
        result = {"schema": "native-full-epoch-fit-result-v1", "job_id": self.job["id"],
            "job_hash": c.digest(self.job), "TEST_accessed": False,
            "status": "failed" if self.code else "complete", "invocation_wall_seconds": 0.125,
            "optimizer_calls_this_invocation": {"attempts": [3], "completed": [3]}, "completed_epochs": 1,
            "resources": {"synthetic_only": True}}
        (self.output/"RESULT.json").write_text(json.dumps(result))
        self.finished = True
        return self.code
    def poll(self):
        self.polls += 1
        if self.finished:
            return self.code
        if self.number == 1 and self.polls == 1:
            return None
        return self.finish()
    def wait(self):
        return self.code if self.finished else self.finish()


def generous_probe(root, active):
    return {"free_vram_bytes": 10_000, "available_ram_bytes": 10_000, "free_disk_bytes": 10_000,
        "allocated_vram_by_pid": {}, "rss_by_pid": {}, "untracked_own_gpu_pids": []}


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-", dir=Path(__file__).parent)
        self.fixture = Fixture(self.temp.name)
        self.root = Path(self.temp.name)
        FakeProcess.counter = 0
        FakeProcess.instances = []
        FakeProcess.failure_first = False
    def tearDown(self):
        self.temp.cleanup()
    def execute(self):
        with patch.object(s.subprocess, "Popen", FakeProcess), patch.object(s, "probe_system", generous_probe), patch.object(s.time, "sleep", lambda seconds: None):
            return s.execute(self.root, "manifest.json", self.fixture.manifest, c.WORKERS[0], "go.json", "readiness.json")

    def test_compile_exact_fit_set_and_immutable_job(self):
        self.assertEqual(len(c.validate_manifest(self.root, self.fixture.manifest)), 4)
        omitted = copy.deepcopy(self.fixture.bindings)
        omitted["fits"].pop(next(iter(omitted["fits"])))
        with self.assertRaisesRegex(ValueError, "Every declared fit"):
            c.compile_manifest(self.root, "plan.json", omitted, "synthetic", "outputs")
        (self.root/"jobs/job-0.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            c.validate_manifest(self.root, self.fixture.manifest)

    def test_no_TEST_arbitrary_command_or_omitted_compiled_fit(self):
        for change in ({"TEST_access": True}, {"schema": "retired-v29"}):
            with self.assertRaises(ValueError):
                c.validate_manifest(self.root, {**self.fixture.manifest, **change})
        altered = copy.deepcopy(self.fixture.manifest)
        altered["jobs"][0]["command"] = ["anything"]
        with self.assertRaisesRegex(ValueError, "arbitrary"):
            c.validate_manifest(self.root, altered)
        altered = copy.deepcopy(self.fixture.manifest)
        altered["jobs"].pop()
        with self.assertRaisesRegex(ValueError, "omits"):
            c.validate_manifest(self.root, altered)

    def test_missing_GO_changed_manifest_readiness_and_source_fail(self):
        with self.assertRaises(ValueError):
            c.require_go(self.root, "manifest.json", self.fixture.manifest, None, None, c.WORKERS[0])
        self.fixture.write("go.json", {**self.fixture.go, "manifest_sha256": "0"*64})
        with self.assertRaisesRegex(ValueError, "hashes mismatch"):
            c.require_go(self.root, "manifest.json", self.fixture.manifest, "go.json", "readiness.json", c.WORKERS[0])
        (self.root/"callback.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            c.validate_manifest(self.root, self.fixture.manifest)

    def test_nonPASS_gate_blocks(self):
        ready = copy.deepcopy(self.fixture.readiness)
        ready["workers"][c.WORKERS[0]]["gates"][c.GATES[0]]["status"] = "PENDING"
        self.fixture.write("readiness.json", ready)
        self.fixture.write("go.json", {**self.fixture.go, "readiness": c.pin(self.root, "readiness.json")})
        with self.assertRaisesRegex(ValueError, "not PASS"):
            c.require_go(self.root, "manifest.json", self.fixture.manifest, "go.json", "readiness.json", c.WORKERS[0])

    def test_loaded_manifest_must_match_exact_reviewed_snapshot(self):
        different = {**self.fixture.manifest, "release_id": "different"}
        with self.assertRaisesRegex(ValueError, "Loaded manifest differs"):
            c.require_go(self.root, "manifest.json", different, "go.json", "readiness.json", c.WORKERS[0])

    def test_unmeasured_resources_costs_and_headroom_block(self):
        bad = {**self.fixture.resources, "pair_qualified": False}
        with self.assertRaisesRegex(ValueError, "pair"):
            c.resource_contract(self.root, bad, self.fixture.cost)
        with self.assertRaises(ValueError):
            c.resource_contract(self.root, self.fixture.resources, {**self.fixture.cost, "measured": False})
        probe = generous_probe(None, [])
        probe["free_vram_bytes"] = 109
        allowed, reason = s.admissible(self.fixture.manifest["jobs"][0], c.WORKERS[0], [], probe,
            self.fixture.readiness["workers"][c.WORKERS[0]]["budget"])
        self.assertEqual((allowed, reason), (False, "VRAM_reservation"))

    def test_two_slots_include_analysis_and_unallocated_startup(self):
        job = self.fixture.manifest["jobs"][0]
        active = [{"pid": pid, "worker": c.WORKERS[0], "resources": self.fixture.resources, "kind": "analysis"} for pid in (1,2)]
        self.assertEqual(s.admissible(job, c.WORKERS[0], active, generous_probe(None, []),
            self.fixture.readiness["workers"][c.WORKERS[0]]["budget"]), (False, "two_slot_limit"))
        probe = generous_probe(None, [])
        probe["untracked_own_gpu_pids"] = [777]
        self.assertEqual(s.admissible(job, c.WORKERS[0], [], probe,
            self.fixture.readiness["workers"][c.WORKERS[0]]["budget"]), (False, "untracked_owned_GPU_process"))

    def test_serial_profiles_and_unqualified_pair_never_overlap(self):
        job = copy.deepcopy(self.fixture.manifest["jobs"][0])
        active = [{"pid": 9, "worker": c.WORKERS[0], "resources": self.fixture.resources}]
        budget = self.fixture.readiness["workers"][c.WORKERS[0]]["budget"]
        job["resources"]["max_concurrent_owned_children"] = 1
        job["resources"]["pair_qualified"] = False
        c.resource_contract(self.root, job["resources"], job["measured_cost"])
        self.assertEqual(s.admissible(job, c.WORKERS[0], active, generous_probe(None, []), budget), (False, "serial_only_resource_profile"))
        job["resources"]["max_concurrent_owned_children"] = 2
        job["resources"]["pair_qualified"] = True
        job["resources"]["qualified_peer_profiles"] = ["different"]
        self.assertEqual(s.admissible(job, c.WORKERS[0], active, generous_probe(None, []), budget), (False, "pair_profile_not_qualified"))

    def test_shared_host_RAM_and_remaining_GPU_reservation(self):
        job = self.fixture.manifest["jobs"][0]
        budget = self.fixture.readiness["workers"][c.WORKERS[0]]["budget"]
        active = [{"pid": 9, "worker": "server77/gpu0", "resources": self.fixture.resources}]
        probe = generous_probe(None, [])
        probe["available_ram_bytes"] = 419
        self.assertEqual(s.admissible(job, "server77/gpu1", active, probe, budget), (False, "host_shared_RAM_reservation"))
        probe = generous_probe(None, [])
        probe["free_vram_bytes"] = 160
        probe["allocated_vram_by_pid"] = {9: 60}
        self.assertTrue(s.admissible(job, "server77/gpu0", active, probe, budget)[0])
        probe["allocated_vram_by_pid"] = {9: 0}
        self.assertEqual(s.admissible(job, "server77/gpu0", active, probe, budget), (False, "VRAM_reservation"))

    def test_atomic_claim_and_existing_output_refusal(self):
        path = self.root/"one-shot.json"
        s.exclusive_json(path, {"claimed": True})
        with self.assertRaises(FileExistsError):
            s.exclusive_json(path, {"claimed": False})
        c.inside(self.root, self.fixture.manifest["jobs"][0]["output"]).mkdir(parents=True)
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(FakeProcess.counter, 0)

    def test_failure_stops_admissions_and_drains_peer_without_retry(self):
        FakeProcess.failure_first = True
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(FakeProcess.counter, 2)
        self.assertTrue(all(process.finished for process in FakeProcess.instances))
        self.assertEqual([row["status"] for row in result["terminal"]], ["failed", "complete"])
        self.assertEqual(len(result["not_admitted"]), 2)
        self.assertEqual(result["terminal"][1]["native_optimizer_calls_this_invocation"]["completed"], [3])

    def test_complete_finite_queue_refuses_second_supervisor(self):
        result = self.execute()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(FakeProcess.counter, 4)
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(FakeProcess.counter, 4)

    def test_failed_launch_drains_already_live_peer(self):
        def failing_second(command, **kwargs):
            if FakeProcess.counter == 1:
                raise OSError("synthetic launch failure")
            return FakeProcess(command, **kwargs)
        with patch.object(s.subprocess, "Popen", failing_second), patch.object(s, "probe_system", generous_probe), patch.object(s.time, "sleep", lambda seconds: None):
            result = s.execute(self.root, "manifest.json", self.fixture.manifest, c.WORKERS[0], "go.json", "readiness.json")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(FakeProcess.counter, 1)
        self.assertTrue(FakeProcess.instances[0].finished)
        self.assertTrue(any(row.get("failure", {}).get("reason") == "launch_failed" for row in result["terminal"] if row.get("failure")))

    def test_missing_terminal_costs_are_unknown(self):
        record = s.terminal_record(self.root, self.fixture.manifest["jobs"][0], 137, 0.5)
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["unobserved_native_costs"])
        self.assertIsNone(record["native_optimizer_calls_this_invocation"])

    def test_authority_change_before_admission_and_source_change_drains(self):
        pins = [c.pin(self.root, path) for path in ("manifest.json", "go.json", "readiness.json")]
        self.fixture.write("go.json", {**self.fixture.go, "review_evidence": "changed"})
        with self.assertRaisesRegex(ValueError, "changed"):
            c.verify_admission_snapshot(self.root, self.fixture.manifest, self.fixture.manifest["jobs"][0], pins)
        self.fixture.write("go.json", self.fixture.go)
        def mutate_after_first(command, **kwargs):
            process = FakeProcess(command, **kwargs)
            (self.root/"callback.py").write_text("# source changed during queue\n")
            return process
        with patch.object(s.subprocess, "Popen", mutate_after_first), patch.object(s, "probe_system", generous_probe), patch.object(s.time, "sleep", lambda seconds: None):
            result = s.execute(self.root, "manifest.json", self.fixture.manifest, c.WORKERS[0], "go.json", "readiness.json")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(FakeProcess.counter, 1)
        self.assertTrue(FakeProcess.instances[0].finished)

    def test_fixed_VALID_analysis_command_and_cross_worker_rejection(self):
        bindings = copy.deepcopy(self.fixture.bindings)
        bindings["VALID_analyses"] = [{"bank_id": "synthetic/native-I4", "resources": self.fixture.resources, "measured_cost": self.fixture.cost}]
        manifest = c.compile_manifest(self.root, "plan.json", bindings, "synthetic", "outputs")
        c.validate_manifest(self.root, manifest)
        analysis = manifest["jobs"][-1]
        command = s.fixed_command(self.root, analysis, {job["id"]: job for job in manifest["jobs"]})
        self.assertIn(str(self.root/c.RUNTIME/"aggregate_selected.py"), command)
        self.assertEqual(len(command[command.index("--members")+1:command.index("--output")]), 4)
        self.assertNotIn("--resume", command)
        analysis["worker"] = "server77/gpu1"
        with self.assertRaisesRegex(ValueError, "bridge changed"):
            c.validate_manifest(self.root, manifest)

    def test_shared_STOP_refuses_all_new_admissions(self):
        (self.root/"outputs").mkdir()
        self.fixture.write("outputs/STOP.json", {"reason": "synthetic other-worker failure"})
        result = self.execute()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(FakeProcess.counter, 0)

    def test_route_guards_and_source_prep_never_probe_GPU(self):
        with patch.object(s.subprocess, "check_output", side_effect=AssertionError("must not probe")):
            for worker in c.WORKERS[1:]:
                with self.assertRaisesRegex(ValueError, "server77"):
                    s.guard_route(worker)
            report = s.preparation(self.root, "plan.json")
            self.assertEqual(report["compiled_runtime_jobs"], 0)
        with patch.object(s.socket, "gethostname", return_value="wrong-host"), patch.object(s.subprocess, "check_output", side_effect=AssertionError("must not probe")):
            with self.assertRaisesRegex(ValueError, "hostname"):
                s.guard_route(c.WORKERS[0])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report = {"schema": "finite-campaign-synthetic-checks-v1", "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "native_models_datasets_or_Torch_imported": False, "actual_process_children_started": 0,
        "system_and_subprocess_calls": "mocked in synthetic queue checks", "GO_receipts": "synthetic temporary fixtures only; no human GO generated",
        "files": {name: c.sha(Path(__file__).with_name(name)) for name in ("contracts.py", "supervisor.py", "checks.py")}}
    Path(__file__).with_name("CHECKS.json").write_text(json.dumps(report, indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
