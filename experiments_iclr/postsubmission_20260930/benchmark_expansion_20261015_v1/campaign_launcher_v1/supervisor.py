"""Prepare/dry-run finite manifests; actual execution requires matching human GO.

No remote commands. Server77 is explicitly blocked. Production dispatch has
two fixed runtime-v2 handlers; there is no shell/command/resume/retry option.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from contracts import (BASE, GATES, LAUNCHER, RUNTIME, WORKERS, canonical, compile_manifest,
    digest, inside, pin, read, require, require_go, sha, source_inventory, validate_manifest, verify_admission_snapshot)

ALLOCATION_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def exclusive_json(path, value):
    """Permanent one-shot authority; never steal, clear, or retry a claim."""
    with Path(path).open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def guard_route(worker):
    # Deliberately before project metadata reads in actual --run. The original
    # runtime_v2/common.py guard is left unchanged and runs again in the child.
    require(worker == "allocation/gpu0", "BLOCKED: server77 requires explicit route reauthorization and a separately qualified route-aware adapter")
    require(socket.gethostname() == "anogena-2-0", "BLOCKED: actual execution requires the dedicated allocation hostname")
    observed = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines()
    require(observed == [ALLOCATION_UUID], "BLOCKED: wrong allocation GPU identity")


def process_status(pid):
    fields = {}
    for row in Path(f"/proc/{pid}/status").read_text().splitlines():
        if ":" in row:
            key, value = row.split(":", 1)
            fields[key] = value.split()
    return {"uid": int(fields["Uid"][0]), "rss_bytes": int(fields.get("VmRSS", [0])[0]) * 1024}


def probe_system(root, active):
    """Physical free space plus measured remaining own reservations, not ETA."""
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines()
    require(len(lines) == 1, "Allocation must expose exactly one GPU")
    uuid, free_mib = (value.strip() for value in lines[0].split(","))
    require(uuid == ALLOCATION_UUID, "GPU identity changed during campaign")
    known = {entry["pid"] for entry in active}
    allocated, unknown_own = {}, []
    lines = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines()
    for line in lines:
        if not line.strip():
            continue
        gpu, pid, used = (part.strip() for part in line.split(","))
        require(gpu == ALLOCATION_UUID, "Unexpected GPU compute process")
        pid = int(pid)
        try:
            status = process_status(pid)
        except FileNotFoundError:
            continue
        if pid not in known and status["uid"] == os.getuid():
            unknown_own.append(pid)
        allocated[pid] = int(used) * 1024**2
    memory = {}
    for row in Path("/proc/meminfo").read_text().splitlines():
        key, value = row.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    rss = {}
    for pid in known:
        try:
            rss[pid] = process_status(pid)["rss_bytes"]
        except FileNotFoundError:
            rss[pid] = 0
    return {"free_vram_bytes": int(free_mib) * 1024**2, "available_ram_bytes": memory["MemAvailable"],
            "free_disk_bytes": shutil.disk_usage(root).free, "allocated_vram_by_pid": allocated,
            "rss_by_pid": rss, "untracked_own_gpu_pids": unknown_own, "UTC": utc()}


def admissible(job, worker, active, probe, budget):
    """All owned training/analysis children consume a slot, including startup.

    Same-host reservations include both GPU workers in the pure contract.
    Current production routing admits only allocation/gpu0; server77 needs a
    qualified shared host-ledger/global-stop bridge before this can be reused.
    """
    if probe["untracked_own_gpu_pids"]:
        return False, "untracked_owned_GPU_process"
    gpu = [entry for entry in active if entry["worker"] == worker]
    host = [entry for entry in active if entry["worker"].split("/")[0] == worker.split("/")[0]]
    if len(gpu) >= 2:
        return False, "two_slot_limit"
    if gpu and (job["resources"]["max_concurrent_owned_children"] == 1
            or any(entry["resources"]["max_concurrent_owned_children"] == 1 for entry in gpu)):
        return False, "serial_only_resource_profile"
    if any(entry["resources"]["profile"] not in job["resources"]["qualified_peer_profiles"]
            or job["resources"]["profile"] not in entry["resources"]["qualified_peer_profiles"] for entry in gpu):
        return False, "pair_profile_not_qualified"
    remaining_gpu = sum(max(0, entry["resources"]["vram_bytes"] - probe["allocated_vram_by_pid"].get(entry["pid"], 0)) for entry in gpu)
    remaining_ram = sum(max(0, entry["resources"]["ram_bytes"] - probe["rss_by_pid"].get(entry["pid"], 0)) for entry in host)
    # Disk remains conservatively reserved in full for each live child. Files
    # already written also consume physical free disk; no optimistic release.
    reserved_disk = sum(entry["resources"]["disk_bytes"] for entry in host)
    for needed, available, reason in (
        (job["resources"]["vram_bytes"] + remaining_gpu + budget["gpu_headroom_bytes"], probe["free_vram_bytes"], "VRAM_reservation"),
        (job["resources"]["ram_bytes"] + remaining_ram + budget["host_ram_headroom_bytes"], probe["available_ram_bytes"], "host_shared_RAM_reservation"),
        (job["resources"]["disk_bytes"] + reserved_disk + budget["disk_headroom_bytes"], probe["free_disk_bytes"], "host_shared_disk_reservation"),
    ):
        if needed > available:
            return False, reason
    return True, "admitted"


def fixed_command(root, job, index):
    base = [sys.executable, "-B", "-u"]
    if job["handler"] == "native_train_v2":
        return base + [str(inside(root, f"{RUNTIME}/run_fit.py")), "--project-root", str(root),
            "--job", str(inside(root, job["job"]["path"])), "--output", str(inside(root, job["output"]))]
    require(job["handler"] == "aggregate_VALID_v2", "Unknown handler; arbitrary commands rejected")
    members = [str(inside(root, index[key]["output"]) / "selected_VALID.pt") for key in job["dependencies"]]
    require(len(members) == 4, "Exactly four native selected VALID members required")
    return base + [str(inside(root, f"{RUNTIME}/aggregate_selected.py")), "--project-root", str(root),
        "--members", *members, "--output", str(inside(root, job["output"]))]


def terminal_record(root, job, code, observed_seconds):
    path = inside(root, job["output"]) / "RESULT.json"
    raw = json.loads(path.read_text()) if path.is_file() else None
    valid = raw is not None and raw.get("TEST_accessed") is False
    if job["kind"] == "fit":
        valid = valid and raw.get("schema") == "native-full-epoch-fit-result-v1" \
            and raw.get("job_id") == job["id"] and raw.get("job_hash") == job["job_canonical_sha256"]
    else:
        valid = valid and raw.get("schema") == "independent-own-selected-VALID-bank-result-v1"
    status = raw.get("status") if raw else "missing_terminal"
    successful = code == 0 and valid and status in ("complete", "no_selected_state")
    return {"id": job["id"], "kind": job["kind"], "exit_code": code,
        "status": status if successful else "failed", "native_terminal_status": status,
        "terminal_identity_and_role_valid": bool(valid), "result_sha256": sha(path) if path.is_file() else None,
        "observed_launch_to_reap_seconds": observed_seconds,
        "native_invocation_wall_seconds": None if raw is None else raw.get("invocation_wall_seconds"),
        "native_optimizer_calls_this_invocation": None if raw is None else raw.get("optimizer_calls_this_invocation"),
        "native_completed_epochs": None if raw is None else raw.get("completed_epochs"),
        "native_resources": None if raw is None else raw.get("resources"),
        "unobserved_native_costs": raw is None or (job["kind"] == "fit" and
            (raw.get("optimizer_calls_this_invocation") is None or raw.get("invocation_wall_seconds") is None)),
        "failure": None if successful else {"native": None if raw is None else raw.get("failure"),
            "reason": "nonzero_exit_or_invalid_or_missing_terminal"}, "automatic_retries": 0}


def execute(root, manifest_path, manifest, worker, go_path, readiness_path):
    jobs = validate_manifest(root, manifest)
    go, ready, authority_pins = require_go(root, manifest_path, manifest, go_path, readiness_path, worker)
    expected_python = manifest.get("python_executable_sha256_by_worker", {}).get(worker)
    require(expected_python == sha(Path(sys.executable)), "Qualified Python executable hash is missing or changed")
    own = [job for job in jobs if job["worker"] == worker]
    require(own, "Worker has no explicitly compiled jobs")
    output = inside(root, manifest["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    leases = Path(root) / ".campaign_worker_leases"
    leases.mkdir(exist_ok=True)
    lease = (leases / (worker.replace("/", "_") + ".lock")).open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    folder = output / "workers" / worker.replace("/", "_")
    folder.mkdir(parents=True, exist_ok=False)
    exclusive_json(folder / "START.json", {"UTC": utc(), "pid": os.getpid(), "worker": worker,
        "manifest_sha256": sha(inside(root, manifest_path)), "GO": pin(root, go_path), "readiness": pin(root, readiness_path)})
    claim_dir = output / "claims"
    claim_dir.mkdir(exist_ok=True)
    index = {job["id"]: job for job in jobs}
    pending, active, records, states = list(own), {}, [], {}
    stopped = False
    stop_path = output / "STOP.json"
    started = time.monotonic()

    def stop(reason):
        nonlocal stopped
        stopped = True
        try:
            exclusive_json(stop_path, {"UTC": utc(), "worker": worker, "reason": reason, "manifest_sha256": sha(inside(root, manifest_path))})
        except FileExistsError:
            pass

    def persist():
        atomic_json(folder / "PROGRESS.json", {"UTC": utc(), "status": "failed_draining" if stopped else "running",
            "active": [{"id": entry["job"]["id"], "pid": pid} for pid, entry in active.items()],
            "pending": [job["id"] for job in pending], "terminal": records, "elapsed_seconds": time.monotonic()-started,
            "automatic_retries": 0, "TEST_accessed": False})

    try:
        require(not stop_path.exists(), "Campaign already has STOP evidence; no restart")
        require(all(not inside(root, job["output"]).exists() for job in own), "Existing output refuses the worker; no retry/resume")
        while pending or active:
            for pid, entry in list(active.items()):
                code = entry["process"].poll()
                if code is None:
                    continue
                entry["log"].close()
                active.pop(pid)
                try:
                    record = terminal_record(root, entry["job"], code, time.monotonic()-entry["started"])
                except Exception as error:
                    record = {"id": entry["job"]["id"], "status": "failed", "exit_code": code,
                        "failure": {"reason": "terminal_receipt_error", "message": str(error)}, "native_costs": None}
                records.append(record)
                states[entry["job"]["id"]] = record["status"]
                atomic_json(entry["claim"], {**entry["authority"], "terminal": record})
                if record["status"] == "failed":
                    stop(record)
            if stop_path.exists():
                stopped = True
            if not stopped:
                for job in list(pending):
                    if any(states.get(key) == "no_selected_state" for key in job["dependencies"]):
                        records.append({"id": job["id"], "status": "blocked_dependency_no_selected_state", "observed_launch_to_reap_seconds": 0,
                            "optimizer_transitions": 0, "automatic_retries": 0})
                        states[job["id"]] = "blocked_dependency_no_selected_state"
                        pending.remove(job)
                        continue
                    if not all(states.get(key) == "complete" for key in job["dependencies"]):
                        continue
                    live = [{"pid": pid, "worker": worker, "resources": entry["job"]["resources"]} for pid, entry in active.items()]
                    probe = probe_system(root, live)
                    allowed, reason = admissible(job, worker, live, probe, ready["budget"])
                    if not allowed:
                        if reason == "untracked_owned_GPU_process" or not active:
                            stop({"reason": "resource_admission_blocked", "detail": reason, "probe": probe})
                        break
                    verify_admission_snapshot(root, manifest, job, authority_pins)
                    require(not inside(root, job["output"]).exists(), "Existing own fit output refuses admission")
                    claim = claim_dir / (digest(job["id"]) + ".json")
                    authority = {"UTC": utc(), "id": job["id"], "worker": worker,
                        "manifest_sha256": sha(inside(root, manifest_path)), "resources": job["resources"], "probe": probe}
                    exclusive_json(claim, authority)
                    log = (folder / (digest(job["id"]) + ".log")).open("x")
                    try:
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                                   MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
                        process = subprocess.Popen(fixed_command(root, job, index), cwd=root, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except Exception as launch_error:
                        log.close()
                        record = {"id": job["id"], "status": "failed", "failure": {"reason": "launch_failed", "message": str(launch_error)},
                            "children_started": 0, "optimizer_transitions": 0, "automatic_retries": 0}
                        records.append(record)
                        states[job["id"]] = "failed"
                        pending.remove(job)
                        atomic_json(claim, {**authority, "terminal": record, "traceback": traceback.format_exc(limit=8)})
                        raise
                    active[process.pid] = {"process": process, "job": job, "log": log, "claim": claim,
                        "authority": {**authority, "child_pid": process.pid}, "started": time.monotonic()}
                    atomic_json(claim, active[process.pid]["authority"])
                    pending.remove(job)
                    if len(active) == 2:
                        break
            persist()
            if stopped and not active:
                break
            if pending and not active:
                stop({"reason": "unresolved_dependency_or_admission", "jobs": [job["id"] for job in pending]})
                break
            if pending or active:
                time.sleep(1)
    except BaseException as error:
        stop({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=10)})
        # Do not terminate a live peer or restart failed work. Drain and retain
        # native results even when admission/source/claim/signal handling fails.
        for pid, entry in list(active.items()):
            code = entry["process"].wait()
            entry["log"].close()
            try:
                record = terminal_record(root, entry["job"], code, time.monotonic()-entry["started"])
            except Exception as terminal_error:
                record = {"id": entry["job"]["id"], "status": "failed", "exit_code": code,
                    "failure": {"reason": "terminal_receipt_error_while_draining", "message": str(terminal_error)}, "native_costs": None}
            records.append(record)
            atomic_json(entry["claim"], {**entry["authority"], "terminal": record})
            active.pop(pid)
    final = {"schema": "finite-campaign-worker-result-v1", "worker": worker, "status": "failed" if stopped else "complete",
        "manifest_sha256": sha(inside(root, manifest_path)), "source_bundle_sha256": manifest["source_bundle_sha256"],
        "reviewed_GO": pin(root, go_path), "elapsed_worker_seconds": time.monotonic()-started,
        "terminal": records, "not_admitted": [job["id"] for job in pending], "automatic_retries": 0,
        "TEST_accessed": False, "native_unobserved_costs_are_unknown_not_zero": True}
    atomic_json(folder / "RESULT.json", final)
    lease.close()
    return final


def preparation(root, source_plan):
    plan = read(root, source_plan)
    fits, assignments = source_inventory(plan)
    return {"schema": "finite-campaign-preparation-v1", "status": "BLOCKED_NOT_COMPILED", "source_plan": pin(root, source_plan),
        "workers": list(WORKERS), "slots_per_worker": 2, "historical_source_fit_count": len(fits),
        "historical_fits_by_worker": {worker: sum(assignments[key] == worker for key in fits) for worker in WORKERS},
        "measured_cost_entries": sum(item.get("relative_cost", {}).get("measured") is True for item in fits.values()),
        "fits_with_resource_fields": sum(bool(item.get("resources")) for item in fits.values()),
        "compiled_runtime_jobs": 0, "TEST_access": False, "GO": False, "ETA": None,
        "blockers": ["Exact all-fit native runtime job/source/data bridge absent", "Mature/pair per-fit measured resources and environment receipts absent",
            "Reviewed readiness gates and explicit manifest/source-bound human GO absent",
            "server77 route authorization and qualified route-aware/shared-host claims-stop adapter absent"],
        "readiness_gates": list(GATES), "final_expanded_roster_count_claimed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source-plan", required=True)
    prepare.add_argument("--output", required=True)
    compile_cli = sub.add_parser("compile")
    for flag in ("source-plan", "bindings", "release-id", "output-root", "output"):
        compile_cli.add_argument("--"+flag, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--worker", choices=WORKERS, required=True)
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    run.add_argument("--go")
    run.add_argument("--readiness")
    for child in (prepare, compile_cli, run):
        child.add_argument("--project-root", required=True)
    args = parser.parse_args()
    if args.action == "run" and args.run:
        guard_route(args.worker)
    root = Path(args.project_root).resolve()
    if args.action == "prepare":
        value = preparation(root, args.source_plan)
        exclusive_json(inside(root, args.output), value)
    elif args.action == "compile":
        value = compile_manifest(root, args.source_plan, read(root, args.bindings), args.release_id, args.output_root)
        exclusive_json(inside(root, args.output), value)
    else:
        manifest = read(root, args.manifest)
        if args.dry_run:
            if manifest.get("schema") == "finite-campaign-preparation-v1":
                value = {**manifest, "worker": args.worker, "children_started": 0,
                    "route_execution_status": "requires live allocation guard" if args.worker == WORKERS[0] else "BLOCKED_route_adapter_missing"}
            else:
                jobs = validate_manifest(root, manifest)
                value = {"status": "DRY_RUN_ONLY", "worker": args.worker, "manifest_sha256": sha(inside(root, args.manifest)),
                    "assigned_jobs": sum(job["worker"] == args.worker for job in jobs), "children_started": 0,
                    "GO_checked": False, "route_execution_status": "requires live allocation guard" if args.worker == WORKERS[0] else "BLOCKED_route_adapter_missing"}
        else:
            value = execute(root, args.manifest, manifest, args.worker, args.go, args.readiness)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 1 if value.get("status") == "failed" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}))
        raise SystemExit(2)
