"""Run bounded asynchronous tasks on one assigned GPU or a CPU worker."""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from .schema import TERMINAL, expand_argv, load_plan, relative_path
from .store import CampaignStore, acquire_lock, atomic_json, release_lock


@dataclass
class RunningTask:
    task: dict
    process: subprocess.Popen
    lock: int
    attempt: int
    started: float
    stdout: object
    stderr: object
    directory: Path


class StopRequested:
    def __init__(self):
        self.requested = False
        for number in (signal.SIGINT, signal.SIGTERM):
            signal.signal(number, self.receive)

    def receive(self, number, frame):
        self.requested = True


def dependency_state(task, states):
    required = [states[name]["status"] for name in task.get("dependencies", [])]
    optional = [states[name]["status"] for name in task.get("optional_dependencies", [])]
    if any(status in {"failed", "blocked"} for status in required):
        return "blocked"
    if any(status == "skipped" for status in required):
        return "skipped"
    if any(status != "complete" for status in required):
        return "waiting"
    if any(status not in TERMINAL for status in optional):
        return "waiting"
    return "ready"


def condition_allows(task, store):
    condition = task.get("when")
    if not condition:
        return True
    path = store.task_dir(condition["task"]) / relative_path(condition["path"])
    value = json.loads(path.read_text())
    for field in condition["field"].split("."):
        value = value[field]
    return value == condition["equals"]


def finish_task(store, running, stop_requested):
    code = running.process.poll()
    if code is None:
        return False
    running.stdout.close()
    running.stderr.close()
    task = running.task
    successful = code == 0 and store.required_outputs_exist(task)
    failures = int(store.read_state(task["id"]).get("failed_attempts", 0))
    if not successful and not stop_requested:
        failures += 1
    if successful:
        status = "complete"
        reason = "subprocess_completed"
    elif stop_requested:
        status = "pending"
        reason = "supervisor_preempted"
    elif failures >= int(task.get("max_attempts", 3)):
        status = "failed"
        reason = "attempt_limit_reached"
    else:
        status = "pending"
        reason = "retry_pending"
    result = {
        "status": status, "attempts": running.attempt, "failed_attempts": failures,
        "exit_code": code, "reason": reason,
        "started": running.started, "finished": time.time(),
        "elapsed_seconds": time.time() - running.started,
        "required_outputs_present": store.required_outputs_exist(task),
    }
    atomic_json(running.directory / "result.json", result)
    store.write_state(task["id"], **result)
    release_lock(running.lock)
    print(f"[campaign] {status} {task['id']} exit={code}", flush=True)
    return True


def start_task(store, task, descriptor, args, worker_id, gpu_lock=None):
    task_id = task["id"]
    previous = store.read_state(task_id)
    attempts_dir = store.task_dir(task_id) / "attempts"
    recorded = [int(path.name) for path in attempts_dir.glob("*")
                if path.is_dir() and path.name.isdigit()]
    # A process may die after creating its log directory and before publishing
    # state. Retain that attempt and allocate the next physical directory.
    attempt = max([int(previous.get("attempts", 0)), *recorded]) + 1
    directory = attempts_dir / f"{attempt:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    output_dir = store.task_dir(task_id) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "python": sys.executable,
        "repo": str(args.repo),
        "root": str(store.root),
        "plan": str(store.root / "plan.json"),
        "task_dir": str(store.task_dir(task_id)),
        "output_dir": str(output_dir),
        "device": "cpu" if args.kind == "cpu" else "cuda:0",
    }
    command = expand_argv(task, context)
    environment = os.environ.copy()
    if args.kind == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = args.device
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    cpus = str(int(task.get("resources", {}).get("cpus", 1)))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = cpus
    environment["PYTHONUNBUFFERED"] = "1"
    environment["GNNM_CAMPAIGN_TASK"] = task_id
    environment["GNNM_CAMPAIGN_ATTEMPT"] = str(attempt)
    environment["GNNM_CAMPAIGN_ROOT"] = str(store.root)
    environment["GNNM_CAMPAIGN_LOCK_FDS"] = ",".join(
        str(value) for value in (descriptor, gpu_lock) if value is not None
    )
    started = time.time()
    invocation = {
        "argv": command, "cwd": str(args.repo), "task_id": task_id,
        "worker_id": worker_id, "hostname": socket.gethostname(),
        "parent_pid": os.getpid(), "started": started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_id": os.environ.get("SLURM_NODEID"),
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "cpus": int(cpus),
        "plan_sha256": store.plan_hash,
        "resume_output_directory": str(output_dir),
    }
    atomic_json(directory / "invocation.json", invocation)
    stdout = (directory / "stdout.log").open("ab", buffering=0)
    stderr = (directory / "stderr.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(command, cwd=args.repo, env=environment,
                                   stdout=stdout, stderr=stderr,
                                   start_new_session=True,
                                   pass_fds=tuple(x for x in (descriptor, gpu_lock) if x is not None))
    except Exception:
        stdout.close()
        stderr.close()
        raise
    try:
        store.write_state(task_id, status="running", attempts=attempt,
                          failed_attempts=int(previous.get("failed_attempts", 0)), worker_id=worker_id,
                          hostname=socket.gethostname(), pid=process.pid,
                          supervisor_pid=os.getpid(), started=started)
    except BaseException:
        # A launched child must not continue after failed state publication.
        signal_process(process, signal.SIGTERM)
        try:
            process.wait(timeout=args.terminate_grace)
        except subprocess.TimeoutExpired:
            signal_process(process, signal.SIGKILL)
            process.wait()
        stdout.close()
        stderr.close()
        raise
    print(f"[campaign] start {task_id} attempt={attempt} pid={process.pid}", flush=True)
    return RunningTask(task, process, descriptor, attempt, started,
                       stdout, stderr, directory)


def signal_process(process, number):
    try:
        os.killpg(process.pid, number)
    except ProcessLookupError:
        pass


def physical_gpu_identity(device):
    # MIG device identifiers already name an isolated physical partition.
    if device.startswith(("GPU-", "MIG-")):
        return device
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--id=" + device, "--query-gpu=uuid",
            "--format=csv,noheader",
        ], text=True, timeout=10, stderr=subprocess.DEVNULL)
        identities = [value.strip() for value in output.splitlines() if value.strip()]
        if len(identities) == 1 and identities[0].startswith("GPU-"):
            return identities[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return device


def resource_shortfall(task, args):
    requested = task.get("resources", {})
    reasons = []
    if float(requested.get("memory_gb", 0)) > args.memory_gb:
        reasons.append("memory_gb")
    if int(requested.get("cpus", 1)) > args.cpus:
        reasons.append("cpus")
    return reasons


def available_gpu_memory(device, mode):
    if mode == "off":
        return None
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--id=" + device,
            "--query-gpu=memory.free", "--format=csv,noheader,nounits",
        ], text=True, timeout=10, stderr=subprocess.DEVNULL)
        values = [float(line.strip()) / 1024 for line in output.splitlines() if line.strip()]
        if len(values) != 1:
            raise ValueError("GPU query did not identify exactly one device")
        return values[0]
    except (OSError, subprocess.SubprocessError, ValueError):
        if mode == "required":
            raise RuntimeError("Unable to inspect free memory of the assigned GPU")
        return None


def status_report(store):
    states = store.snapshot()
    counts = Counter(state["status"] for state in states.values())
    return {"campaign_id": store.plan["campaign_id"], "plan_sha256": store.plan_hash,
            "total": len(states), "counts": dict(counts),
            "tasks": states}


def run(args):
    args.repo = Path(args.repo).resolve()
    args.root = Path(args.root).resolve()
    if not args.root.is_relative_to(args.repo):
        raise ValueError("Campaign output root must be inside the repository")
    plan = load_plan(args.plan)
    store = CampaignStore(args.root, plan)
    store.prepare()
    if args.status:
        print(json.dumps(status_report(store), indent=2, sort_keys=True))
        return 0
    args.repo = Path(args.repo).resolve()
    if not args.repo.is_dir():
        raise ValueError(f"Repository directory is missing: {args.repo}")
    tasks = sorted(
        (task for task in plan["tasks"] if task.get("kind", "gpu") == args.kind),
        key=lambda task: (int(task.get("priority", 100)),
                          -float(task.get("estimated_seconds", 0)), task["id"]),
    )
    if args.kind == "gpu" and not args.device:
        raise ValueError("GPU workers require an explicitly assigned --device token")
    if args.slots < 1 or args.cpus < 1 or args.memory_gb <= 0:
        raise ValueError("Resource budgets and slot count must be positive")
    host = socket.gethostname()
    resource = args.device if args.kind == "gpu" else "cpu"
    worker_id = f"{host}-{args.kind}-{resource}-{uuid.uuid4().hex[:12]}"
    gpu_lock = None
    if args.kind == "gpu":
        safe_resource = physical_gpu_identity(resource).replace("/", "_")
        gpu_lock = acquire_lock(store.root / "resources" / host / f"gpu-{safe_resource}.lock")
        if gpu_lock is None:
            raise RuntimeError(f"Another campaign supervisor owns GPU {resource} on {host}")
    ineligible = {task["id"]: resource_shortfall(task, args) for task in tasks
                  if resource_shortfall(task, args)}
    atomic_json(store.root / "workers" / f"{worker_id}.json", {
        "worker_id": worker_id, "hostname": host, "kind": args.kind,
        "device": args.device, "slots": args.slots,
        "cpus": args.cpus, "memory_gb": args.memory_gb,
        "ineligible_tasks": ineligible, "started": time.time(),
    })
    if ineligible:
        print(f"[campaign] {len(ineligible)} tasks exceed this worker's budgets. "
              f"Details: workers/{worker_id}.json", flush=True)
    stop = StopRequested()
    active = []
    stop_started = None
    try:
        while True:
            active = [running for running in active
                      if not finish_task(store, running, stop.requested)]
            if stop.requested:
                if stop_started is None:
                    stop_started = time.monotonic()
                    for running in active:
                        signal_process(running.process, signal.SIGTERM)
                if not active:
                    return 2
                if time.monotonic() - stop_started > args.terminate_grace:
                    for running in active:
                        signal_process(running.process, signal.SIGKILL)
                time.sleep(min(args.poll_seconds, 1))
                continue
            states = store.snapshot()
            if all(state["status"] in TERMINAL for state in states.values()):
                return 1 if any(state["status"] in {"failed", "blocked"} for state in states.values()) else 0
            free_memory = available_gpu_memory(args.device, args.gpu_memory_check) if args.kind == "gpu" else None
            started_any = False
            ready_exclusive = False
            for task in tasks:
                task_id = task["id"]
                if states[task_id]["status"] in TERMINAL or any(x.task["id"] == task_id for x in active):
                    continue
                dependencies = dependency_state(task, states)
                if dependencies == "waiting":
                    continue
                resources = task.get("resources", {})
                exclusive = bool(resources.get("exclusive", False))
                if dependencies == "ready" and resource_shortfall(task, args):
                    continue
                if dependencies == "ready" and exclusive and states[task_id]["status"] == "pending":
                    ready_exclusive = True
                if dependencies == "ready" and active:
                    if exclusive or any(x.task.get("resources", {}).get("exclusive", False) for x in active):
                        continue
                    if ready_exclusive:
                        continue
                memory = sum(float(x.task.get("resources", {}).get("memory_gb", 0)) for x in active)
                cpus = sum(int(x.task.get("resources", {}).get("cpus", 1)) for x in active)
                if dependencies == "ready" and (
                    len(active) >= args.slots or
                    memory + float(resources.get("memory_gb", 0)) > args.memory_gb or
                    cpus + int(resources.get("cpus", 1)) > args.cpus or
                    (free_memory is not None and float(resources.get("memory_gb", 0)) > max(0, free_memory - args.memory_reserve_gb))
                ):
                    continue
                descriptor = store.task_lock(task_id)
                if descriptor is None:
                    continue
                try:
                    current = store.read_state(task_id)
                    if current["status"] in TERMINAL:
                        continue
                    attempts = int(current.get("attempts", 0))
                    if dependencies in {"blocked", "skipped"}:
                        store.write_state(task_id, status=dependencies, attempts=attempts,
                                          reason="required_dependency_" + dependencies)
                        continue
                    if not condition_allows(task, store):
                        store.write_state(task_id, status="skipped", attempts=attempts,
                                          reason="validation_gate_closed")
                        continue
                    # A free kernel lock and an old running state indicate a terminated
                    # process tree. Its retained training checkpoint supplies recovery.
                    if int(current.get("failed_attempts", 0)) >= int(task.get("max_attempts", 3)):
                        store.write_state(task_id, status="failed", attempts=attempts,
                                          reason="attempt_limit_after_process_loss")
                        continue
                    running = start_task(store, task, descriptor, args, worker_id, gpu_lock)
                    active.append(running)
                    if free_memory is not None:
                        free_memory -= float(resources.get("memory_gb", 0))
                    descriptor = None
                    started_any = True
                except Exception as error:
                    store.write_state(task_id, status="failed",
                                      attempts=int(store.read_state(task_id).get("attempts", 0)),
                                      reason="scheduler_error", error=repr(error))
                    print(f"[campaign] scheduler failure {task_id}: {error!r}", flush=True)
                finally:
                    if descriptor is not None:
                        release_lock(descriptor)
            if args.no_wait and not active and not started_any:
                return 0
            time.sleep(args.poll_seconds)
    finally:
        # Keep task locks in surviving children if the supervisor exits unexpectedly.
        for running in active:
            running.stdout.close()
            running.stderr.close()
            release_lock(running.lock)
        if gpu_lock is not None:
            release_lock(gpu_lock)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", required=True, type=Path)
    result.add_argument("--root", required=True, type=Path)
    result.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--kind", choices=("gpu", "cpu"), default="gpu")
    result.add_argument("--device")
    result.add_argument("--slots", type=int, default=2)
    result.add_argument("--memory-gb", type=float, default=70)
    result.add_argument("--cpus", type=int, default=8)
    result.add_argument("--gpu-memory-check", choices=("auto", "required", "off"), default="auto")
    result.add_argument("--memory-reserve-gb", type=float, default=1)
    result.add_argument("--poll-seconds", type=float, default=2)
    result.add_argument("--terminate-grace", type=float, default=120)
    result.add_argument("--no-wait", action="store_true")
    result.add_argument("--status", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
