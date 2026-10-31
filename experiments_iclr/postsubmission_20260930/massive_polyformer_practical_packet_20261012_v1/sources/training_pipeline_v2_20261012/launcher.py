"""Two supervised experiment slots per GPU, with fixed three-GPU assignments.

The launcher contains no learning code. Children own registry claims, logs and
checkpoints. A lost supervisor never grants permission to repeat a fit.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from registry import Registry
from run_study import owner_identity, require_approval
from spec import canonical, confined, fingerprint, load
from phase_closure import write_host_receipts

ALLOCATION_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"


def hardware():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
                                      "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = []
    for line in output.splitlines():
        index, uuid, total, free = [part.strip() for part in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "total_mib": int(total), "free_mib": int(free)})
    return rows


def host_check(host, plan):
    rows = hardware()
    if host == "allocation":
        if socket.gethostname() != "anogena-2-0" or [r["uuid"] for r in rows] != [ALLOCATION_UUID]:
            raise ValueError("The scientific one-GPU allocation route failed verification")
    elif host == "server77":
        binding = plan.get("host_bindings", {}).get(host)
        if not binding or binding.get("status") != "verified":
            raise ValueError("18.77 needs a verified hostname/UUID binding after synchronization")
        if socket.gethostname() != binding["hostname"] or [r["uuid"] for r in rows] != binding["gpu_uuids"]:
            raise ValueError("18.77 hardware differs from the frozen host binding")
    else:
        raise ValueError("Use allocation or server77")
    return rows


def validate_plan(root, plan):
    if plan.get("schema_version") != 1 or plan.get("slots_per_gpu") != 2:
        raise ValueError("Expected the fixed two-slot execution plan")
    studies, runs, placement = {}, {}, {}
    for row in plan["studies"]:
        path = confined(root, row["path"])
        spec, digest = load(path, root)
        if digest != row["spec_hash"] or spec["study_id"] in studies:
            raise ValueError("Changed or repeated study contract")
        studies[spec["study_id"]] = {"spec": spec, "path": path, "hash": digest}
        for run in spec["runs"]:
            key = spec["study_id"] + "/" + run["run_id"]
            runs[key] = {"study": spec["study_id"], "run": run, "path": path}
    for group in plan["groups"]:
        worker = group["gpu_assignment"]
        if worker not in {"allocation/gpu0", "server77/gpu0", "server77/gpu1"}:
            raise ValueError("Unexpected hardware assignment")
        for key in group["run_keys"]:
            if key not in runs or key in placement:
                raise ValueError("Every job must have exactly one assignment")
            placement[key] = {"worker": worker, "group": group["id"]}
    if set(placement) != set(runs):
        raise ValueError("Unassigned jobs remain")
    # Dependencies are keyed within a study; global barriers are declared by key.
    dependencies = {}
    for key, record in runs.items():
        dependencies[key] = [record["study"] + "/" + p for p in record["run"]["parents"]]
        dependencies[key] += plan.get("barriers", {}).get(key, [])
        if any(parent not in runs for parent in dependencies[key]):
            raise ValueError("Missing parent for " + key)
        if any(placement[p]["worker"].split("/")[0] != placement[key]["worker"].split("/")[0]
               for p in dependencies[key]):
            raise ValueError("Dependencies must be staged on the same host")
    # A cycle must be found before starting any processes.
    done, visiting = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError("Cyclic run dependencies")
        if key in done:
            return
        visiting.add(key)
        for parent in dependencies[key]:
            visit(parent)
        visiting.remove(key)
        done.add(key)
    for key in runs:
        visit(key)
    return studies, runs, placement, dependencies


def load_plan(root, path, approval_path=None):
    plan = json.loads(Path(path).read_text())
    studies, runs, placement, dependencies = validate_plan(root, plan)
    digest = fingerprint(plan)
    if approval_path is not None:
        approval = json.loads(Path(approval_path).read_text())
        if approval.get("approved_plan_hash") != digest:
            raise ValueError("This assignment plan has not received explicit GO")
        for row in studies.values():
            require_approval(approval_path, row["hash"])
    return plan, digest, studies, runs, placement, dependencies


def schedule_ready(runs, states, placement, dependencies, active, capacities, host, *, slots=2):
    """Pure admission policy used by real launcher and engineering checks."""
    result, occupied, reserved = [], {}, {}
    active_keys = {item["key"] for item in active}
    for item in active:
        worker = item["worker"]
        occupied[worker] = occupied.get(worker, 0) + 1
        reserved[worker] = reserved.get(worker, 0) + item["reserve_mib"]
    keys = sorted(runs, key=lambda key: (-runs[key]["run"]["config"].get("estimated_seconds", 0), key))
    for key in keys:
        worker = placement[key]["worker"]
        if worker.split("/")[0] != host or states[key] != "registered" or key in active_keys:
            continue
        terminal_parents = runs[key]["run"].get("kind") == "selection"
        allowed_states = {"complete", "failed"} if terminal_parents else {"complete"}
        if any(states[p] not in allowed_states for p in dependencies[key]):
            continue
        if occupied.get(worker, 0) >= slots:
            continue
        reserve = runs[key]["run"]["config"].get("reservation_mib")
        if type(reserve) is not int or reserve < 1:
            raise ValueError("Each run requires a positive GPU-memory reservation")
        # free_mib is a conservative admission margin. Current external use is
        # respected; reservations prevent simultaneously starting two large fits.
        if reserve + reserved.get(worker, 0) > capacities[worker]:
            continue
        result.append(key)
        occupied[worker] = occupied.get(worker, 0) + 1
        reserved[worker] = reserved.get(worker, 0) + reserve
    return result


def run_supervisor(root, plan_path, approval_path, host, python, *, preflight_only=False):
    plan, digest, studies, runs, placement, dependencies = load_plan(
        root, plan_path, None if preflight_only else approval_path)
    observed = host_check(host, plan)
    if preflight_only:
        print(canonical({"preflight": "passed", "host": host, "hardware": observed,
                         "plan_hash": digest, "jobs": len(runs), "scientific_training_runs": 0}))
        return
    output = confined(root, plan["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / (host + ".supervisor.lock")).open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry = Registry(output / (host + ".sqlite"))
    for study in studies.values():
        registry.register(study["spec"])
    active, stop, closed_local_gates = {}, {"requested": False}, set()
    def request_stop(signum, frame):
        stop["requested"] = True  # Current children finish and retain their state.
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    journal = (output / (host + ".supervisor.jsonl")).open("a", buffering=1)
    def event(kind, payload):
        journal.write(canonical({"time": time.time(), "kind": kind, **payload}) + "\n")
    event("start", {"owner": owner_identity(), "plan_hash": digest, "hardware": observed})
    try:
        while True:
            for key in list(active):
                item = active[key]
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                event("child_terminal", {"key": key, "exit_code": code, "pid": item["process"].pid})
                del active[key]
                row = runs[key]
                if registry.run(row["study"], row["run"]["run_id"])["state"] == "registered":
                    raise RuntimeError("Child terminated before its registry claim; inspect its log before retrying")
            all_states = registry.states()
            states = {key: all_states[key] for key in runs}
            write_host_receipts(root, plan, digest, runs, placement, registry, host, closed_local_gates)
            local = [key for key in runs if placement[key]["worker"].startswith(host + "/")]
            if not active and all(states[key] in {"complete", "failed"} for key in local):
                event("roster_terminal", {"all_successful": all(states[k] == "complete" for k in local)})
                return
            if stop["requested"] and not active:
                event("stopped_after_current_children", {})
                return
            if not stop["requested"]:
                current = hardware()
                capacities = {host + "/gpu" + str(row["index"]): max(0, row["free_mib"] - 1024)
                              for row in current}
                stages = {row["run"]["config"].get("stage", "search") for row in runs.values()}
                phase_allowed = {stage: gate_ready(root, plan, digest, stage) for stage in stages}
                admissible_states = {key: state if phase_allowed[runs[key]["run"]["config"].get("stage", "search")]
                                     else "phase_wait" for key, state in states.items()}
                ready = schedule_ready(runs, admissible_states, placement, dependencies,
                                       list(active.values()), capacities, host)
                for key in ready:
                    row = runs[key]
                    run = row["run"]
                    worker = placement[key]["worker"]
                    index = int(worker.rsplit("gpu", 1)[1])
                    kind = run["kind"]
                    script = Path(__file__).with_name("run_study.py" if kind == "model_fit" else "run_endpoint.py")
                    if not script.is_file():
                        raise ValueError("Missing executable endpoint runner")
                    log_path = output / "logs" / row["study"] / (run["run_id"] + ".log")
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open("a")
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index),
                               GNNM_TORCH_THREADS=str(plan.get("torch_threads_per_fit", 2)),
                               OMP_NUM_THREADS=str(plan.get("torch_threads_per_fit", 2)))
                    command = [str(python), "-B", "-u", str(script), "--project-root", str(root),
                               "--study", str(row["path"]), "--run-id", run["run_id"],
                               "--registry", str(output / (host + ".sqlite")), "--output", str(output / "runs"),
                               "--approval", str(approval_path), "--device", "cuda:0"]
                    process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    active[key] = {"key": key, "process": process, "log": log, "worker": worker,
                                   "reserve_mib": run["config"]["reservation_mib"]}
                    event("spawn", {"key": key, "pid": process.pid, "worker": worker, "command": command})
            # Failed parents and pre-existing unowned running records need action,
            # not an automatic restart or an eternal unchanged loop.
            if not active:
                pending = [k for k in local if states[k] == "registered"]
                blocked = [k for k in pending if runs[k]["run"]["kind"] != "selection"
                           and any(states[p] == "failed" for p in dependencies[k])]
                unowned = [k for k in local if states[k] == "running"]
                if blocked or unowned:
                    event("attention_required", {"failed_parent_jobs": blocked, "existing_running_records": unowned})
                    raise RuntimeError("Inspect failed parents or existing process custody before continuing")
            time.sleep(5)
    finally:
        registry.close()
        journal.close()
        lock.close()


def gate_ready(root, plan, digest, stage):
    for name, gate in plan.get("global_gates", {}).items():
        if stage not in gate["required_for_stages"]:
            continue
        path = confined(root, gate["path"])
        if not path.is_file():
            return False
        value = json.loads(path.read_text())
        if (value.get("status") != "complete" or value.get("plan_hash") != digest
                or value.get("closed_phase") != name):
            raise ValueError("Global phase receipt does not belong to this frozen plan")
        if set(value.get("host_receipts", {})) != set(gate["required_hosts"]):
            raise ValueError("Global phase receipt omits an assigned host")
        for host, receipt in value["host_receipts"].items():
            raw = confined(root, receipt["path"]).read_bytes()
            import hashlib
            if len(raw) != receipt["bytes"] or hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                raise ValueError("Global phase proof changed after collection")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval")
    parser.add_argument("--host", choices=("allocation", "server77"), required=True)
    parser.add_argument("--python")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--detach", action="store_true", help="Start a session-independent supervisor after GO verification")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    if not args.preflight_only and not args.approval:
        raise ValueError("Explicit GO receipt is required")
    for path in (args.plan, args.approval):
        if path and not Path(path).resolve().is_relative_to(root):
            raise ValueError("Plan and approval must stay in the project")
    python = Path(args.python).resolve() if args.python else root / ".venv/bin/python"
    if args.detach:
        if args.preflight_only:
            raise ValueError("A preflight does not start a supervisor")
        plan, digest, _, _, _, _ = load_plan(root, args.plan, args.approval)
        host_check(args.host, plan)
        output = confined(root, plan["output_root"])
        output.mkdir(parents=True, exist_ok=True)
        log = (output / (args.host + ".startup.log")).open("a")
        command = [str(python), "-B", "-u", str(Path(__file__).resolve()),
                   "--project-root", str(root), "--plan", str(Path(args.plan).resolve()),
                   "--approval", str(Path(args.approval).resolve()), "--host", args.host,
                   "--python", str(python)]
        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        time.sleep(1)
        if child.poll() is not None:
            raise RuntimeError("Supervisor exited during startup; inspect " + str(output / (args.host + ".startup.log")))
        print(canonical({"supervisor_pid": child.pid, "host": args.host, "plan_hash": digest,
                         "survives_SSH_or_MacLink_disconnect": True, "slots_per_gpu": 2}))
    else:
        run_supervisor(root, Path(args.plan), Path(args.approval) if args.approval else None,
                       args.host, python, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
