"""Resource, custody and command checks without native fits, GPUs or network."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import concurrency_worker as q

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE.parent / "qualification/all_fit_resources_20261016_v1"))
import resources

GiB = q.GiB


class ConcurrencyTests(unittest.TestCase):
    def budget(self, vram=8, ram=8, disk=4, cpu=2):
        return {"vram_bytes": vram * GiB, "ram_bytes": ram * GiB,
                "disk_bytes": disk * GiB, "cpu_slots": cpu,
                "thread_count": 2, "profile": "fixture"}

    def fixture(self, count=3):
        candidate = {"gpu_uuid": "gpu", "reservation": self.budget()}
        active = [{"directory": "active-" + str(i), "gpu_uuid": "gpu",
                   "reservation": self.budget()} for i in range(count)]
        live = {"free_vram": {"gpu": 80 * GiB}, "available_ram": 256 * GiB,
                "free_disk": 200 * GiB, "available_cpu_before_owned_reservations": 32,
                "owned_usage": {}}
        return candidate, active, live

    def decide(self, candidate, active, live):
        return q.capacity_decision(candidate, active, live, lambda *_: True)

    def test_third_and_fourth_small_tasks_admitted_fifth_rejected(self):
        for active_count, admitted in ((2, True), (3, True), (4, False)):
            candidate, active, live = self.fixture(active_count)
            with self.subTest(active_count=active_count):
                result = self.decide(candidate, active, live)
                self.assertEqual(result["admitted"], admitted)
                self.assertEqual(result["maximum_concurrent_tasks_per_gpu"], 4)
                self.assertNotIn("two GPU slots occupied", result["reasons"])
                self.assertEqual("four GPU slots occupied" in result["reasons"], active_count == 4)

    def test_gpu_growth_sums_all_active_tasks_and_charges_unknown_usage(self):
        candidate, active, live = self.fixture()
        for free, admitted in ((33 * GiB - 1, False), (33 * GiB, True)):
            result = self.decide(candidate, active, dict(live, free_vram={"gpu": free}))
            self.assertEqual(result["remaining_vram"], 24 * GiB)
            self.assertEqual(result["admitted"], admitted)
        live["owned_usage"] = {row["directory"]: {"vram": 2 * GiB} for row in active}
        live["free_vram"]["gpu"] = 27 * GiB
        result = self.decide(candidate, active, live)
        self.assertEqual(result["remaining_vram"], 18 * GiB)
        self.assertTrue(result["admitted"])

    def test_four_20gib_budgets_cannot_fit_80gib_with_headroom(self):
        candidate, active, live = self.fixture()
        candidate["reservation"] = self.budget(vram=20)
        for row in active:
            row["reservation"] = self.budget(vram=20)
        result = self.decide(candidate, active, live)
        self.assertFalse(result["admitted"])
        self.assertEqual(result["reasons"], ["GPU free capacity/growth"])

    def test_host_ram_cpu_and_disk_include_other_gpu_reservations(self):
        candidate, active, live = self.fixture()
        active[-1]["gpu_uuid"] = "other-gpu"
        cases = (
            ({"available_ram": 40 * GiB - 1}, "host RAM free capacity/growth"),
            ({"available_cpu_before_owned_reservations": 7.94}, "CPU capacity"),
            ({"free_disk": 36 * GiB - 1}, "disk capacity"),
        )
        for changes, reason in cases:
            with self.subTest(reason=reason):
                result = self.decide(candidate, active, dict(live, **changes))
                self.assertFalse(result["admitted"])
                self.assertEqual(result["reasons"], [reason])
                self.assertEqual(result["remaining_ram"], 24 * GiB)
                self.assertEqual(result["cpu_reserved"], 6)
                self.assertEqual(result["disk_reserved"], 12 * GiB)
                self.assertEqual(result["disk_required_free_bytes"], 36 * GiB)
        exact = dict(live, available_ram=40 * GiB,
                     available_cpu_before_owned_reservations=8, free_disk=36 * GiB)
        self.assertTrue(self.decide(candidate, active, exact)["admitted"])

    def test_heavy_serial_and_unknown_budget_exclusions_are_preserved(self):
        profiles = json.loads(resources.CATALOGUE.read_text())["profiles"]
        heavy = next(value for value in profiles.values() if value["vram_bytes"] == 69 * GiB)
        small = next(value for value in profiles.values() if value["vram_bytes"] == 3 * GiB)
        candidate, active, live = self.fixture(1)
        candidate["reservation"] = copy.deepcopy(heavy)
        active[0]["reservation"] = copy.deepcopy(small)
        result = q.capacity_decision(candidate, active, live, resources.can_pair)
        self.assertFalse(result["admitted"])
        self.assertIn("incompatible peak pair budgets", result["reasons"])
        candidate["reservation"] = copy.deepcopy(small)
        candidate["reservation"]["ram_bytes"] += 1
        result = q.capacity_decision(candidate, active, live, resources.can_pair)
        self.assertIn("incompatible peak pair budgets", result["reasons"])

    def test_source_manifest_is_bound_to_new_package(self):
        self.assertEqual(q.HERE, PACKAGE)
        expected = "new-concurrency-manifest"
        checked = []

        def fake_pin(path):
            checked.append(Path(path))
            return {"path": str(path), "sha256": expected}

        with patch.object(q, "pin", side_effect=fake_pin), \
             patch.object(q, "read", return_value={"files": []}), \
             patch.object(q.recovery, "source"), \
             patch.object(q.recovery.named, "own_source"):
            q.own_source(expected)
        self.assertEqual(checked, [PACKAGE / "MANIFEST.json"])

    def test_constructor_keeps_repair_source_for_native_launcher(self):
        repair_source = {"sha256": "original-repair-v2"}
        concurrency_source = {"sha256": "new-concurrency"}
        args = SimpleNamespace()
        with patch.object(q.recovery, "source", return_value=repair_source), \
             patch.object(q.recovery.RecoveryWorker, "__init__", return_value=None) as parent:
            worker = q.FourWorker(args, concurrency_source)
        self.assertEqual(worker.concurrency_source, concurrency_source)
        parent.assert_called_once_with(args, repair_source)

    def worker_for_invocation(self):
        worker = q.FourWorker.__new__(q.FourWorker)
        worker.concurrency_source = {"sha256": "new-concurrency"}
        worker.named_source = {"sha256": q.REPAIR_SHA}
        worker.recovery_authority_pin = {"sha256": "original-recovery-authority"}
        worker.process_title_dependency_dir = "/project/dependency"
        worker.process_title_dependency = {"sha256": "original-title-dependency"}
        worker.scheduler_source = {"changes": ["original launch safeguards"]}
        worker.args = SimpleNamespace(recovery_config="/project/recovery.json",
                                      recovery_config_sha256="recovery-authority")
        worker.retry = set()
        return worker

    def test_native_argv_and_recovery_authority_preserved_with_new_provenance(self):
        worker = self.worker_for_invocation()
        native = ["/python", "-B", "-u", "/frozen/runtime_entry.py",
                  "--task-id", "fit", "--output", "/output"]
        request = {"argv": native, "resume_uses_smoke": False}
        with patch.object(q.safeguards.parent.Worker, "invocation", return_value=request):
            invocation = worker.invocation({"id": "fit"}, "full", Path("/attempt"))
        self.assertEqual(invocation["native_argv"], native)
        self.assertEqual(invocation["argv"][invocation["argv"].index("--") + 1:], native[3:])
        self.assertEqual(invocation["reporting_repair_source"], worker.named_source)
        self.assertEqual(invocation["actual_scheduler_producer"]["actual_outer_source_manifest"],
                         worker.concurrency_source)
        self.assertEqual(invocation["actual_scheduler_producer"]["recovery_authority"],
                         worker.recovery_authority_pin)
        self.assertEqual(invocation["argv"][invocation["argv"].index("--repair-source-sha256") + 1], q.REPAIR_SHA)
        self.assertFalse(invocation["reporting_only_finalization"])
        self.assertNotIn("--recovery-config", invocation["argv"])
        self.assertIs(q.FourWorker.load_states, q.recovery.RecoveryWorker.load_states)
        for name in ("run", "acquire", "collect", "reap", "smoke_summary", "prerequisites"):
            self.assertIs(getattr(q.FourWorker, name), getattr(q.recovery.RecoveryWorker, name))

    def test_fourth_launch_keeps_original_guardian_and_inherited_leases(self):
        with tempfile.TemporaryDirectory(dir=PACKAGE) as temp:
            candidate, active, live = self.fixture()
            worker = q.FourWorker.__new__(q.FourWorker)
            worker.root = Path(temp)
            worker.ledger = worker.root / "ledger"
            worker.config = {"physical_gpu_uuid": "gpu"}
            worker.args = SimpleNamespace(worker="server51/gpu0", poll_seconds=1)
            worker.plan = {"project_root": temp}
            worker.blocked_profiles = {}
            worker.active_ledger = lambda: active
            worker.resources = SimpleNamespace(can_pair=lambda *_: True)
            worker.states = {"candidate": {"status": "pending", "attempt": 0}}
            worker.processes = {}
            worker.retry = set()
            worker.leases = [q.lock(worker.root / ("queue-" + str(i) + ".lock")) for i in range(3)]
            task = {"id": "candidate", "reservation": candidate["reservation"]}
            worker.invocation = lambda task, phase, directory: {"argv": ["unchanged-native"],
                                                              "output": str(directory / "output")}
            try:
                with patch.object(q, "probe", return_value=live), \
                     patch.object(q.subprocess, "Popen") as launch:
                    self.assertTrue(worker.start(task, "full"))
                argv = launch.call_args.args[0]
                self.assertEqual(Path(argv[3]), q.safeguards.parent.HERE / "guardian.py")
                self.assertEqual(len(launch.call_args.kwargs["pass_fds"]), 4)
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
                self.assertEqual(worker.states["candidate"]["status"], "running")
                self.assertEqual(worker.states["candidate"]["attempt"], 1)
            finally:
                for lease in worker.leases:
                    lease.close()

    def test_queue_scans_past_blocked_task_and_starts_four(self):
        class EndCycle(Exception):
            pass

        with tempfile.TemporaryDirectory(dir=PACKAGE) as temp:
            worker = q.FourWorker.__new__(q.FourWorker)
            worker.root = Path(temp)
            worker.args = SimpleNamespace(poll_seconds=1)
            worker.tasks = [{"id": name, "kind": "fit"} for name in ("blocked", "a", "b", "c", "d", "e")]
            worker.states = {task["id"]: {"status": "pending"} for task in worker.tasks}
            worker.processes = {}
            worker.retry = set()
            worker.load_states = lambda *_: None
            worker.reap = lambda *_: None
            worker.report = lambda **_: None

            def start(task, phase):
                if task["id"] == "blocked":
                    return False
                worker.processes[task["id"]] = (None, None)
                worker.states[task["id"]] = {"status": "running"}
                return True

            worker.start = Mock(side_effect=start)
            with patch.object(q.time, "sleep", side_effect=EndCycle), self.assertRaises(EndCycle):
                worker.run_phase("full")
            self.assertEqual(list(worker.processes), ["a", "b", "c", "d"])
            self.assertEqual([call.args[0]["id"] for call in worker.start.call_args_list],
                             ["blocked", "a", "b", "c", "d"])


if __name__ == "__main__":
    unittest.main()
