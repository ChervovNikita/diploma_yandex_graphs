"""Prospective standard-library supervisor. Authored only, not executed.

One separately admitted CPU child. Admission ID records root's receipt and
does not itself authorize execution. There is no retry path.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
RESEARCH = BASE.parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_children_usage():
    value = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"user_cpu_seconds": value.ru_utime, "system_cpu_seconds": value.ru_stime,
            "ru_maxrss_raw": value.ru_maxrss,
            "ru_maxrss_unit": "bytes" if sys.platform == "darwin" else "KiB_on_Linux; native_raw_otherwise",
            "scope": "Launcher cumulative RUSAGE_CHILDREN; memory maxima are raw and never subtracted"}


def verify_before_child():
    seal = json.loads((BASE / "AUTHORING_SEAL.json").read_text())
    require(seal["manifest"] == "AUTHORING_MANIFEST.json", "Unexpected manifest name")
    manifest_path = BASE / seal["manifest"]
    require(sha(manifest_path) == seal["manifest_sha256"], "Authoring manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["files"]:
        path = (BASE / row["path"]).resolve()
        require(path.is_relative_to(BASE) and path.is_file(), "Authoring path invalid")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Authoring bytes changed: " + row["path"])
    pins = json.loads((BASE / "SOURCE_PINS.json").read_text())
    for row in pins["files"]:
        path = (RESEARCH / row["research_relative_path"]).resolve()
        require(path.is_relative_to(RESEARCH) and path.is_file(), "External pin path invalid")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "External source changed: " + row["research_relative_path"])
    freeze = json.loads((BASE / "FREEZE.json").read_text())
    require(freeze["runtime"]["hard_child_seconds"] == 180 and freeze["runtime"]["internal_alarm_seconds"] == 170,
            "Pinned timeout changed")
    return seal, pins, freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--admission-id", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name), "Simple new run name required")
    require(args.admission_id.strip(), "Root admission receipt required")
    launcher_started = time.perf_counter_ns()
    utc_started = datetime.now(timezone.utc).isoformat()
    seal, pins, freeze = verify_before_child()
    # Any existing run prevents a second child, including an incomplete run.
    runs = BASE / "runs"
    require(not runs.exists(), "This packet already has a run reservation; retries are forbidden")
    runs.mkdir(mode=0o700, exist_ok=False)
    output = runs / args.run_name
    output.mkdir(mode=0o700, exist_ok=False)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1", BLIS_NUM_THREADS="1")
    command = [sys.executable, "-B", str(BASE / "qualifier.py"), "--output-dir", str(output), "--admission-id", args.admission_id]
    journal = {"status": "ready_for_single_child", "admission_id": args.admission_id,
               "authorization_scope": "Receipt only; root must independently authorize this launch",
               "run_name": args.run_name, "started_utc": utc_started, "command": command,
               "authoring_seal": seal, "source_pins": pins, "hard_child_seconds": 180,
               "timeout_scope": "Child-launch envelope only; prelaunch and kill/reap/finalization work are outside. Root wraps/charges total coordinator envelope.",
               "raw_children_usage_before": raw_children_usage(), "retry": False}
    journal_path = output / "launcher.json"
    def save():
        temporary = output / "launcher.json.tmp"
        temporary.write_text(json.dumps(journal, indent=2) + "\n", encoding="utf-8")
        temporary.replace(journal_path)
    save()
    child = None
    child_started = None
    try:
        with (output / "stdout.txt").open("xb") as stdout, (output / "stderr.txt").open("xb") as stderr:
            child_started = time.perf_counter_ns()
            child = subprocess.Popen(command, cwd=str(BASE), env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
            journal.update(status="child_running", child_pid=child.pid)
            save()
            try:
                remaining = freeze["runtime"]["hard_child_seconds"] - (time.perf_counter_ns() - child_started) / 1e9
                child.wait(timeout=max(0, remaining))
            except subprocess.TimeoutExpired:
                journal["hard_timeout_reached"] = True
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            journal.update(child_returncode=child.returncode,
                           elapsed_child_wall_seconds=(time.perf_counter_ns() - child_started) / 1e9,
                           status="hard_timeout_incomplete" if journal.get("hard_timeout_reached") else "child_exited")
    except BaseException as error:
        journal.update(status="launcher_failed_or_interrupted", error_type=type(error).__name__, error=str(error))
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        if child is not None:
            journal["child_returncode"] = child.returncode
            journal["elapsed_child_wall_seconds"] = (time.perf_counter_ns() - child_started) / 1e9
        raise
    finally:
        journal.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_launcher_wall_seconds=(time.perf_counter_ns() - launcher_started) / 1e9,
                       raw_children_usage_after=raw_children_usage())
        before, after = journal["raw_children_usage_before"], journal["raw_children_usage_after"]
        journal["child_cpu_user_seconds_delta"] = after["user_cpu_seconds"] - before["user_cpu_seconds"]
        journal["child_cpu_system_seconds_delta"] = after["system_cpu_seconds"] - before["system_cpu_seconds"]
        progress = output / "progress.json"
        if progress.is_file():
            journal["child_progress_sha256"] = sha(progress)
            try:
                journal["child_reported_status"] = json.loads(progress.read_text()).get("status")
            except (OSError, ValueError):
                journal["child_reported_status"] = "unreadable_incremental_journal"
        save()
    complete = child is not None and child.returncode == 0 and journal.get("child_reported_status") == "completed_resource_qualification"
    print(json.dumps({"complete": complete, "launcher_journal": str(journal_path)}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
