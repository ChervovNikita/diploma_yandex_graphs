"""Drain each existing GPU queue and automatically resume it with four slots."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import concurrency_worker as four

c = four.recovery.c
HERE = Path(__file__).resolve().parent
PLAN_SHAS = {
    "allocation/gpu0": "19fa1a8b96bcf3383965c266c4fa900e0ac09c62ca040553e7669acb659ffaad",
    "server51/gpu0": "b23301b8674faf5dfad2000efeeb067d0c1a2fb97524277c2185d174a307a345",
    "server51/gpu1": "bc025b3209c0a7f5c2f10fd011fd54ca072c78269693b035ca3e53bbf243ad3c",
}


def pane(session):
    rows = subprocess.check_output([
        "tmux", "list-panes", "-t", session, "-F",
        "#{pane_id}\t#{pane_dead}\t#{pane_pid}",
    ], text=True).splitlines()
    c.require(len(rows) == 1, "Expected one campaign pane")
    value = rows[0].split("\t")
    return {"id": value[0], "dead": value[1] == "1", "pid": int(value[2])}


def preflight(worker, source_sha):
    four.recovery.guard()
    source = four.own_source(source_sha)
    config, _ = c.environment(worker)
    project = Path(config["project_root"])
    expansion = project / "experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
    four.recovery.named.verify_dependency(str(expansion / "campaign_process_title_dependency_20261021_v1"))
    plan_path = expansion / "campaign_execution_20261016_v7" / ("PLAN-" + worker.replace("/", "_") + ".json")
    c.require(c.pin(plan_path)["sha256"] == PLAN_SHAS[worker], "Frozen worker plan changed")
    plan = c.read(plan_path)
    root = Path(plan["state_root"])
    c.require(root.is_relative_to(project), "Queue state escapes repository")
    status, owner = c.read(root / "STATUS.json"), c.read(root / "OWNER.json")
    c.require(status["phase"] == "full" and not status.get("error"), "Only a healthy full queue may hand over")
    c.require(status["owner"] == owner["identity"] and c.same_process(owner["identity"]), "Queue owner changed")
    producer = owner["actual_scheduler_producer"]
    c.require(producer["actual_outer_source_manifest"] == c.pin(four.REPAIR / "MANIFEST.json"), "Unexpected old scheduler")
    recovery = producer["recovery_authority"]
    c.verify(recovery, project)
    c.require(c.read(recovery["path"])["plan"] == c.pin(plan_path), "Recovery authority has a different plan")
    c.require(c.global_barrier(plan["global_smoke_pass"], plan, c.pin(root / "SMOKE_SUMMARY.json")), "All-fit smoke release invalid")
    c.require(not (root / "STOP").exists(), "An existing STOP request must be preserved")
    session = "gnnm-allfit-" + worker.replace("/", "-") + "-20261016"
    current = pane(session)
    c.require(not current["dead"] and current["pid"] == owner["identity"]["pid"], "Original tmux owner differs")
    python = str(project / ".venv/bin/python")
    argv = [python, "-B", "-u", str(HERE / "concurrency_worker.py"),
            "--worker", worker, "--plan", str(plan_path), "--plan-sha256", PLAN_SHAS[worker],
            "--source-sha256", source_sha,
            "--process-title-dependency-dir", str(expansion / "campaign_process_title_dependency_20261021_v1"),
            "--recovery-config", recovery["path"], "--recovery-config-sha256", recovery["sha256"]]
    return {"schema": "four-slot-graceful-handover-v1", "worker": worker,
            "project_root": str(project), "state_root": str(root), "source": source,
            "plan": c.pin(plan_path), "old_owner": owner["identity"], "session": session,
            "pane": current, "argv": argv, "recovery_authority": recovery,
            "maximum_concurrent_tasks_per_gpu": 4, "old_limit": 2,
            "native_scientific_changes": False, "existing_children_signalled": False,
            "user_authorization": "2026-10-29: increase to four models on each of all three GPUs",
            "created": time.time()}


def schedule(worker, source_sha):
    request = preflight(worker, source_sha)
    root = Path(request["state_root"])
    journal = HERE.parent / "planning/concurrency4_20261029_v1" / worker.replace("/", "_")
    journal.mkdir(parents=True, exist_ok=False)
    for name in ("OWNER.json", "STATUS.json", "TMUX_LAUNCH.json"):
        if (root / name).exists():
            shutil.copy2(root / name, journal / ("before-" + name))
    request["journal"] = str(journal)
    request_path = journal / "REQUEST.json"
    c.write(request_path, request)
    argv = [request["argv"][0], "-B", "-u", str(HERE / "handover.py"), "watch",
            "--request", str(request_path), "--request-sha256", c.pin(request_path)["sha256"]]
    command = shlex.join(argv) + " >> " + shlex.quote(str(journal / "watcher.log")) + " 2>&1"
    session = "gnnm-concurrency4-" + worker.replace("/", "-")
    c.write(journal / "WATCHER_LAUNCH.json", {"argv": argv, "shell_command": command, "tmux_session": session})
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session, "-c", request["project_root"], command])
    print(json.dumps({"status": "watcher_dispatched", "worker": worker, "journal": str(journal), "tmux_session": session}))


def watch(request_path, request_sha):
    four.recovery.guard()
    c.require(c.pin(request_path)["sha256"] == request_sha, "Handover request changed")
    request = c.read(request_path)
    four.own_source(request["source"]["sha256"])
    c.environment(request["worker"])
    c.verify(request["plan"], request["project_root"])
    c.verify(request["recovery_authority"], request["project_root"])
    root, journal = Path(request["state_root"]), Path(request["journal"])
    c.require(root.is_relative_to(Path(request["project_root"])) and journal.is_relative_to(HERE.parent), "Handover path escapes project")
    lease = c.lock(journal / "WATCHER.lock")
    c.require(lease is not None, "A watcher already owns this handover")
    c.write(journal / "WATCHER.json", {"identity": c.identity(), "request": c.pin(request_path)})
    marker = json.dumps({"reason": "graceful transition to four concurrent tasks per GPU",
                         "request": c.pin(request_path)}, sort_keys=True) + "\n"
    stop = root / "STOP"
    try:
        c.require(c.same_process(request["old_owner"]), "Original owner exited before drain request")
        c.require(c.read(root / "OWNER.json")["identity"] == request["old_owner"], "Owner changed before drain")
        with stop.open("x") as stream:
            stream.write(marker)
            stream.flush()
            os.fsync(stream.fileno())
        c.write(journal / "STATUS.json", {"phase": "draining_existing_fits", "updated": time.time(),
                 "target_concurrency": 4, "training_processes_signalled": False})
        while c.same_process(request["old_owner"]):
            c.require(stop.read_text() == marker, "STOP ownership changed")
            c.require(c.read(root / "OWNER.json")["identity"] == request["old_owner"], "Unexpected queue owner during drain")
            time.sleep(15)
        c.require(stop.read_text() == marker, "STOP ownership changed after drain")
        status = c.read(root / "STATUS.json")
        c.require(status["owner"] == request["old_owner"] and status["phase"] == "stopped"
                  and not status["active_fit_ids"], "Queue did not drain normally")
        c.require(not status["counts"].get("failed", 0), "A fit failed during drain; inspect before continuation")
        current = pane(request["session"])
        c.require(current["id"] == request["pane"]["id"] and current["dead"], "Original pane is not safely stopped")
        plan = c.read(c.verify(request["plan"], request["project_root"]))
        handles = []
        try:
            for path in [root / "SUPERVISOR.lock", *[plan[key] for key in ("slot_lock", "physical_lock", "worker_lock")]]:
                handle = c.lock(path)
                c.require(handle is not None, "A previous owner or child still holds a queue lease")
                handles.append(handle)
            shutil.copy2(root / "STATUS.json", journal / "drained-STATUS.json")
            c.require(stop.read_text() == marker, "STOP ownership changed before archival")
            stop.rename(journal / "CONSUMED_STOP.json")
        finally:
            for handle in handles:
                handle.close()
        command = "exec " + shlex.join(request["argv"]) + " >> " + shlex.quote(str(journal / "new-owner.log")) + " 2>&1"
        c.write(journal / "ACTIVATION_REQUEST.json", {"argv": request["argv"], "shell_command": command,
                "pane": current, "updated": time.time(), "forced_process_stops": 0})
        subprocess.check_call(["tmux", "respawn-pane", "-t", current["id"], "-c", request["project_root"], command])
        c.write(journal / "STATUS.json", {"phase": "four_slot_owner_dispatched", "updated": time.time(), "target_concurrency": 4})
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            status = c.read(root / "STATUS.json")
            producer = status.get("actual_scheduler_producer", {})
            if producer.get("actual_outer_source_manifest") == request["source"]:
                c.require(status["phase"] not in ("supervisor_error", "full_failed", "smoke_failed"), "Replacement scheduler reported failure")
                if status["phase"] == "full" and c.same_process(status["owner"]):
                    c.write(journal / "STATUS.json", {"phase": "active", "updated": time.time(),
                            "target_concurrency": 4, "owner": status["owner"], "counts": status["counts"]})
                    print(json.dumps({"status": "active", "worker": request["worker"], "concurrency": 4}), flush=True)
                    return
            time.sleep(10)
        raise TimeoutError("Replacement scheduler has not reached full phase; inspect retained logs")
    except BaseException as exc:
        c.write(journal / "STATUS.json", {"phase": "handover_error", "updated": time.time(), "error": type(exc).__name__ + ": " + str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "schedule", "watch"))
    parser.add_argument("--worker", choices=c.WORKERS)
    parser.add_argument("--source-sha256")
    parser.add_argument("--request")
    parser.add_argument("--request-sha256")
    args = parser.parse_args()
    if args.action == "watch":
        watch(args.request, args.request_sha256)
    else:
        c.require(args.worker and args.source_sha256, "Worker and source manifest required")
        if args.action == "schedule":
            schedule(args.worker, args.source_sha256)
        else:
            print(json.dumps(preflight(args.worker, args.source_sha256), indent=2))


if __name__ == "__main__":
    main()
