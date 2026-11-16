"""CPU-only qualification of distributed scheduling and Slurm argument mapping."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from .launch import assigned_gpus, commands
from .schema import validate_plan
from .store import CampaignStore, acquire_lock, release_lock
from .worker import available_gpu_memory, physical_gpu_identity

REPO = Path(__file__).resolve().parents[2]


class CampaignTests(unittest.TestCase):
    def setUp(self):
        parent = REPO / ".transfer_setup_20261110" / "scheduler_tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "run"
        self.plan_path = self.directory / "plan.json"
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.temporary.cleanup()

    def task(self, task_id, **fields):
        fixture = fields.pop("fixture", [])
        result = {
            "id": task_id, "kind": "cpu",
            "argv": ["{python}", "-m", "ablation.campaign.test_fixture",
                     "--output", "{output_dir}", "--registry", "{root}/fixture",
                     *fixture],
            "resources": {"cpus": 1, "memory_gb": 1},
            "required_outputs": ["output/result.json"],
            **fields,
        }
        return result

    def plan(self, tasks):
        plan = {"schema": "gnnm-campaign-v1", "campaign_id": "test", "tasks": tasks}
        self.plan_path.write_text(json.dumps(plan))
        store = CampaignStore(self.root, plan)
        store.prepare()
        return store

    def worker(self, *extra):
        path = self.directory / f"worker-{len(self.processes)}.log"
        with path.open("wb") as log:
            process = subprocess.Popen([
                sys.executable, "-m", "ablation.campaign.worker",
                "--plan", str(self.plan_path), "--root", str(self.root),
                "--repo", str(REPO), "--kind", "cpu", "--slots", "2",
                "--cpus", "4", "--memory-gb", "10", "--poll-seconds", "0.02",
                "--terminate-grace", "3", "--gpu-memory-check", "off", *extra,
            ], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(process)
        return process

    def wait_status(self, store, task_id, desired, timeout=15):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if store.read_state(task_id)["status"] == desired:
                return
            time.sleep(0.03)
        self.fail(f"{task_id} did not reach {desired}: {store.read_state(task_id)}")

    def test_rejects_cycles_and_unsafe_paths(self):
        with self.assertRaises(ValueError):
            self.plan([self.task("../escape")])
        plan = {"schema": "gnnm-campaign-v1", "campaign_id": "cycle",
                "tasks": [self.task("a", dependencies=["b"]),
                          self.task("b", dependencies=["a"])]}
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_immutable_plan_and_incremental_journal(self):
        store = self.plan([self.task("a")])
        reader = CampaignStore(self.root, store.plan)
        self.assertEqual(reader.read_state("a")["status"], "pending")
        store.write_state("a", status="complete", attempts=1)
        self.assertEqual(reader.read_state("a")["status"], "complete")
        with (self.root / "events.jsonl").open("ab") as stream:
            stream.write(b'{"incomplete"')
        store.write_state("a", status="complete", attempts=1, repaired=True)
        self.assertTrue(reader.read_state("a")["repaired"])
        changed = dict(store.plan, campaign_id="other")
        with self.assertRaises(ValueError):
            CampaignStore(self.root, changed).prepare()

    def test_100_workers_claim_each_task_once(self):
        store = self.plan([self.task(f"fit/{i:03d}", fixture=["--duration", "0.1"])
                           for i in range(200)])
        processes = [self.worker("--slots", "1", "--no-wait") for _ in range(100)]
        for process in processes:
            self.assertEqual(process.wait(timeout=60), 0)
        states = store.snapshot()
        self.assertTrue(all(value["status"] == "complete" for value in states.values()))
        self.assertTrue(all(value["attempts"] == 1 for value in states.values()))
        registry = json.loads((self.root / "fixture/registry.json").read_text())
        self.assertEqual(registry["violations"], [])
        self.assertEqual(sum(event["entering"] for event in registry["events"]), 200)
        self.assertGreater(registry["peak"], 1)

    def test_dependencies_gates_optional_analysis_and_failures(self):
        tasks = [
            self.task("gate", fixture=["--gate", "false"]),
            self.task("branch", dependencies=["gate"],
                      when={"task": "gate", "path": "output/decision.json",
                            "field": "expand", "equals": True}),
            self.task("branch_child", dependencies=["branch"]),
            self.task("broken", fixture=["--fail"], max_attempts=1),
            self.task("blocked_child", dependencies=["broken"]),
            self.task("report", optional_dependencies=["branch_child", "blocked_child"]),
        ]
        store = self.plan(tasks)
        self.assertEqual(self.worker().wait(timeout=15), 1)
        states = store.snapshot()
        self.assertEqual(states["branch"]["status"], "skipped")
        self.assertEqual(states["branch_child"]["status"], "skipped")
        self.assertEqual(states["broken"]["status"], "failed")
        self.assertEqual(states["blocked_child"]["status"], "blocked")
        self.assertEqual(states["report"]["status"], "complete")

    def test_failure_retry_and_required_output(self):
        store = self.plan([self.task("retry", fixture=["--fail-first"])])
        self.assertEqual(self.worker().wait(timeout=15), 0)
        state = store.read_state("retry")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["attempts"], 2)
        self.assertEqual(state["failed_attempts"], 1)

    def test_preemption_retains_progress_without_spending_failure_limit(self):
        store = self.plan([self.task("long", fixture=["--duration", "2", "--steps", "20"],
                                    max_attempts=1)])
        worker = self.worker()
        self.wait_status(store, "long", "running")
        progress = self.root / "tasks/long/output/progress.json"
        until = time.monotonic() + 10
        while not progress.exists() and time.monotonic() < until:
            time.sleep(0.03)
        worker.send_signal(signal.SIGTERM)
        self.assertEqual(worker.wait(timeout=10), 2)
        self.assertEqual(store.read_state("long")["status"], "pending")
        self.assertEqual(self.worker().wait(timeout=15), 0)
        result = json.loads((self.root / "tasks/long/output/result.json").read_text())
        self.assertGreater(result["resumed_at"], 0)
        self.assertEqual(result["completed"], 20)
        self.assertEqual(store.read_state("long")["failed_attempts"], 0)

    def test_exclusive_tasks_and_memory_admission(self):
        tasks = [
            self.task("a", kind="gpu", priority=0,
                      resources={"memory_gb": 5, "cpus": 1}, fixture=["--duration", "0.4"]),
            self.task("b", kind="gpu", priority=0,
                      resources={"memory_gb": 5, "cpus": 1}, fixture=["--duration", "0.4"]),
            self.task("cost", kind="gpu", priority=10,
                      resources={"memory_gb": 2, "cpus": 1, "exclusive": True},
                      fixture=["--exclusive", "--duration", "0.2"]),
            self.task("later", kind="gpu", priority=20,
                      resources={"memory_gb": 5, "cpus": 1}, fixture=["--duration", "0.2"]),
        ]
        store = self.plan(tasks)
        self.assertEqual(self.worker("--kind", "gpu", "--device", "simulated",
                                     "--slots", "4").wait(timeout=15), 0)
        registry = json.loads((self.root / "fixture/registry.json").read_text())
        self.assertEqual(registry["violations"], [])
        self.assertEqual(registry["peak"], 2)
        self.assertTrue(all(state["status"] == "complete" for state in store.snapshot().values()))

    def test_child_retains_task_and_gpu_locks_after_supervisor_death(self):
        store = self.plan([self.task("survivor", kind="gpu", fixture=["--duration", "1.5"])])
        process = self.worker("--kind", "gpu", "--device", "simulated")
        self.wait_status(store, "survivor", "running")
        process.kill()
        process.wait(timeout=5)
        lock = acquire_lock(store.task_dir("survivor") / ".ownership.lock")
        self.assertIsNone(lock)
        retry = self.worker("--kind", "gpu", "--device", "simulated")
        self.assertNotEqual(retry.wait(timeout=5), 0)
        until = time.monotonic() + 10
        while time.monotonic() < until:
            lock = acquire_lock(store.task_dir("survivor") / ".ownership.lock")
            if lock is not None:
                release_lock(lock)
                break
            time.sleep(0.05)
        else:
            self.fail("Child task lock was never released")
        self.assertEqual(self.worker("--kind", "gpu", "--device", "simulated").wait(timeout=10), 0)
        self.assertEqual(store.read_state("survivor")["status"], "complete")

    def test_orphan_attempt_directory_and_plan_template(self):
        task = self.task("orphan")
        task["argv"] += ["--plan", "{plan}"]
        store = self.plan([task])
        orphan = store.task_dir("orphan") / "attempts" / "0001"
        orphan.mkdir(parents=True)
        (orphan / "partial-start.txt").write_text("preserved startup evidence")
        self.assertEqual(self.worker().wait(timeout=15), 0)
        self.assertEqual(store.read_state("orphan")["attempts"], 2)
        self.assertTrue((orphan / "partial-start.txt").exists())
        invocation = json.loads((store.task_dir("orphan") / "attempts/0002/invocation.json").read_text())
        self.assertEqual(invocation["argv"][-1], str(self.root / "plan.json"))

    def test_gpu_identity_memory_and_resource_report(self):
        with patch("ablation.campaign.worker.subprocess.check_output", return_value="GPU-example\n"):
            self.assertEqual(physical_gpu_identity("0"), "GPU-example")
        self.assertEqual(physical_gpu_identity("MIG-example"), "MIG-example")
        with patch("ablation.campaign.worker.subprocess.check_output", return_value="40960\n"):
            self.assertEqual(available_gpu_memory("0", "required"), 40)
        with patch("ablation.campaign.worker.subprocess.check_output", side_effect=OSError):
            self.assertIsNone(available_gpu_memory("0", "auto"))
            with self.assertRaises(RuntimeError):
                available_gpu_memory("0", "required")
        self.plan([self.task("oversized", resources={"memory_gb": 20, "cpus": 1})])
        self.assertEqual(self.worker("--no-wait").wait(timeout=15), 0)
        records = list((self.root / "workers").glob("*.json"))
        record = json.loads(records[0].read_text())
        self.assertEqual(record["ineligible_tasks"], {"oversized": ["memory_gb"]})

    def test_filesystem_lock_qualification(self):
        directory = self.directory / "filesystem-check"
        processes = []
        for index in range(4):
            log = (self.directory / f"filesystem-{index}.log").open("wb")
            process = subprocess.Popen([
                sys.executable, "-m", "ablation.campaign.qualify_filesystem",
                "--root", str(directory), "--participants", "4",
                "--rounds", "3", "--hold-seconds", "0.03",
            ], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            self.processes.append(process)
            processes.append(process)
        for process in processes:
            self.assertEqual(process.wait(timeout=15), 0)
        report = json.loads((directory / "qualification.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["participants"], 4)
        self.assertGreater(report["contention_count"], 0)

    def test_slurm_gpu_visibility_and_dry_run(self):
        self.assertEqual(assigned_gpus("auto", {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
                                               "SLURM_JOB_ID": "1"}), ["GPU-a", "GPU-b"])
        self.assertEqual(assigned_gpus("auto", {"SLURM_JOB_ID": "1",
                                               "SLURM_JOB_GPUS": "2,6"}), ["2", "6"])
        with self.assertRaises(ValueError):
            assigned_gpus("auto", {"SLURM_JOB_ID": "1"})
        with self.assertRaises(ValueError):
            assigned_gpus("0,0", {})
        self.plan([self.task("a")])
        result = subprocess.run([
            sys.executable, "-m", "ablation.campaign.launch",
            "--plan", str(self.plan_path), "--root", str(self.root),
            "--gpus", ",".join(str(i) for i in range(100)),
            "--jobs-per-gpu", "4", "--dry-run",
        ], cwd=REPO, capture_output=True, text=True, check=True)
        declared = json.loads(result.stdout)["commands"]
        self.assertEqual(len(declared), 101)
        self.assertTrue(all(command[0] == sys.executable for command in declared))


if __name__ == "__main__":
    unittest.main()
