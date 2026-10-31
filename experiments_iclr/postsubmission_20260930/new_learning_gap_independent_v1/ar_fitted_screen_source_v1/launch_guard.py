"""Unexecuted standard-library guard for exactly one admitted CPU child.

No admission is supplied by this source. Root reviews/seals the packet, admits
it separately, and supervises the whole upload/launch/fetch operation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
RESEARCH = BASE.parents[1]
EXPECTED_FREEZE = "5202fcb2b310377ee1363339a77711ff3447f35295f17b956e45fe61ea25d837"
EXPECTED_LOGIN = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
EXPECTED_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
RSS_CAP = 4 * 1024**3
VIRTUAL_CAP = 16 * 1024**3
POLL_SECONDS = 0.25


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_before_child(args):
    require(args.root_freeze_sha256 == EXPECTED_FREEZE and sha(args.root_freeze) == EXPECTED_FREEZE,
            "Exact root scientific freeze required")
    require(args.root_freeze.resolve().is_relative_to(RESEARCH) and
            args.root_admission.resolve().is_relative_to(RESEARCH), "Root contracts must remain inside phase")
    freeze = json.loads(args.root_freeze.read_text())
    admission = json.loads(args.root_admission.read_text())
    require(freeze["status"] == "scientifically_frozen_by_root", "Scientific freeze required")
    require(admission["status"] == "admitted_by_root" and admission["scientific_freeze_sha256"] == EXPECTED_FREEZE,
            "Separate matching root execution admission required")
    seal = json.loads((BASE / "AUTHORING_SEAL.json").read_text())
    require(seal["manifest"] == "AUTHORING_MANIFEST.json", "Unexpected manifest")
    manifest_path = BASE / seal["manifest"]
    require(sha(manifest_path) == seal["manifest_sha256"] == admission["authoring_manifest_sha256"],
            "Admission did not bind sealed source packet")
    for row in json.loads(manifest_path.read_text())["files"]:
        path = (BASE / row["path"]).resolve()
        require(path.is_relative_to(BASE) and path.is_file(), "Invalid authored path")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Authored source bytes changed")
    pins = json.loads((BASE / "SOURCE_PINS.json").read_text())
    for row in pins["files"]:
        path = (RESEARCH / row["research_relative_path"]).resolve()
        require(path.is_relative_to(RESEARCH) and path.is_file(), "Invalid dependency path")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Pinned dependency bytes changed")
    for row in admission["bindings"]:
        path = (RESEARCH / row["path"]).resolve()
        require(path.is_relative_to(RESEARCH) and sha(path) == row["sha256"], "Root admission binding changed")
    runtime = freeze["runtime"]
    require(runtime["internal_alarm_seconds"] == 2820 and runtime["hard_child_seconds"] == 2850 and
            runtime["whole_coordinator_seconds"] == 3000 and runtime["diagnostic_seconds"] == 60 and
            runtime["new_run_artifact_cap_bytes"] == 1024**3 and runtime["CPU_only"], "Frozen resource bounds changed")
    require(sys.platform.startswith("linux"), "Authorized Linux allocation required")
    require(os.environ.get("GNNM_SSH_DESTINATION") == EXPECTED_LOGIN, "Exact authorized SSH route required")
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True, check=True, timeout=10)
    rows = [line.split(",") for line in query.stdout.splitlines() if line.strip()]
    require(len(rows) == 1, "Exactly one authorized A100 must be visible")
    row = [part.strip() for part in rows[0]]
    require(len(row) == 4 and row == ["0", EXPECTED_UUID, "NVIDIA A100-SXM4-80GB", "81920"],
            "Exact authorized GPU UUID/model/memory guard failed")
    return freeze, admission, seal, {"visible_gpu_count": 1, "gpu_index": 0, "gpu_uuid": row[1],
                                    "gpu_name": row[2], "total_memory_MiB": int(row[3])}


def directory_usage(path):
    apparent = allocated = 0
    for directory, subdirs, files in os.walk(path, followlinks=False):
        for name in subdirs + files:
            item = Path(directory) / name
            require(not item.is_symlink(), "Run output/cache must not contain symlinks")
            try:
                info = item.stat()
            except FileNotFoundError:
                continue  # Atomic journal replacement can remove a temporary.
            apparent += info.st_size
            allocated += info.st_blocks * 512
    return {"apparent_bytes": apparent, "allocated_bytes": allocated}


def published_memory(output):
    try:
        data = json.loads((output / "progress.json").read_text())
    except FileNotFoundError:
        return {"rss_high_water_bytes": 0, "available": False}
    usage = data["latest_usage"]
    require(usage["maxrss_unit"].startswith("KiB_on_Linux"), "Expected Linux child-published high-water RSS")
    return {"rss_high_water_bytes": usage["maxrss_raw"] * 1024, "available": True,
            "scope": "Last published cumulative child maximum; not live RSS or exact preemption"}


def children_usage():
    value = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"user_cpu_seconds": value.ru_utime, "system_cpu_seconds": value.ru_stime,
            "maxrss_KiB": value.ru_maxrss, "scope": "Launcher cumulative children, including allocation query"}


def child_limits():
    resource.setrlimit(resource.RLIMIT_AS, (VIRTUAL_CAP, VIRTUAL_CAP))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def kill_and_reap(child):
    started = time.monotonic()
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=max(0, 15 - (time.monotonic() - started)))
    return time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-freeze", type=Path, required=True)
    parser.add_argument("--root-freeze-sha256", required=True)
    parser.add_argument("--root-admission", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    require(args.run_name == "admitted_run_v1", "Exactly one root-planned run name required")
    started = time.monotonic()
    freeze, admission, seal, hardware = verify_before_child(args)
    runs = BASE / "runs"
    require(not runs.exists(), "A prior reservation forbids overwrite/resume/retry")
    runs.mkdir(mode=0o700)
    output = runs / args.run_name
    output.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", BLIS_NUM_THREADS="1", GNNM_PHASE_ROOT=str(RESEARCH))
    confined = {}
    for name in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "TORCH_HOME",
                 "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                 "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE", "NUMBA_CACHE_DIR", "WANDB_DIR"):
        directory = output / "runtime_paths" / name.lower()
        directory.mkdir(parents=True)
        confined[name] = str(directory)
    environment.update(confined)
    command = [sys.executable, "-B", str(BASE / "entrypoint.py"), "--root-freeze", str(args.root_freeze.resolve()),
        "--root-freeze-sha256", args.root_freeze_sha256, "--root-admission", str(args.root_admission.resolve()),
        "--output-dir", str(output)]
    journal = {"status": "ready_for_single_child", "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name, "command": command, "root_admission_sha256": sha(args.root_admission),
        "authoring_seal": seal, "hardware": hardware, "CPU_only": True, "retry": False,
        "hard_child_seconds": 2850, "internal_child_alarm_seconds": 2820, "kill_reap_seconds_cap": 15,
        "launcher_envelope_seconds": 2885, "RSS_cap_bytes": RSS_CAP, "virtual_memory_RLIMIT_AS_bytes": VIRTUAL_CAP,
        "artifact_cap_bytes": freeze["runtime"]["new_run_artifact_cap_bytes"], "poll_seconds": POLL_SECONDS,
        "resource_enforcement": "Published child cumulative high-water RSS and run disk are polled; unpublished/transient overshoot is possible. RSS observation waits for the next child journal save. Per-file RLIMIT_FSIZE is 1GiB. No exact instantaneous RSS/disk preemption claim.",
        "cache_and_temp_paths": confined, "peak_published_maxrss_bytes": 0,
        "children_usage_before": children_usage()}
    journal_path = output / "launcher.json"
    def save():
        temporary = output / "launcher.json.tmp"
        temporary.write_text(json.dumps(journal, indent=2, allow_nan=False) + "\n")
        temporary.replace(journal_path)
    save()
    def interrupted(signum, frame):
        raise InterruptedError("Launcher interrupted; child is terminated without retry")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    child = None
    child_started = None
    try:
        with (output / "stdout.txt").open("xb") as stdout, (output / "stderr.txt").open("xb") as stderr:
            child_started = time.monotonic()
            child = subprocess.Popen(command, cwd=RESEARCH, env=environment, stdout=stdout, stderr=stderr,
                start_new_session=True, preexec_fn=child_limits)
            journal.update(status="child_running", child_pid=child.pid)
            save()
            deadline = min(child_started + 2850, started + 2870)  # Leave15s for kill/reap.
            last_save = time.monotonic()
            while child.poll() is None:
                memory, disk = published_memory(output), directory_usage(output)
                journal.update(latest_memory=memory, latest_disk=disk,
                    peak_published_maxrss_bytes=max(journal["peak_published_maxrss_bytes"], memory["rss_high_water_bytes"]))
                require(memory["rss_high_water_bytes"] <= RSS_CAP, "Published child maximum RSS exceeds 4GiB; incomplete no-go")
                require(max(disk.values()) <= freeze["runtime"]["new_run_artifact_cap_bytes"], "Observed run artifacts exceed 1GiB; incomplete no-go")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    journal.update(status="hard_timeout_incomplete", hard_timeout_reached=True)
                    journal["kill_reap_seconds"] = kill_and_reap(child)
                    break
                if time.monotonic() - last_save >= 1:
                    save()
                    last_save = time.monotonic()
                try:
                    child.wait(timeout=min(POLL_SECONDS, remaining))
                except subprocess.TimeoutExpired:
                    pass
            if journal["status"] != "hard_timeout_incomplete":
                journal["status"] = "child_exited"
    except BaseException as error:
        journal.update(status="launcher_failed_or_interrupted", error_type=type(error).__name__, error=str(error))
        if child is not None:
            try:
                journal["kill_reap_seconds"] = kill_and_reap(child)
            except BaseException as cleanup_error:
                journal["kill_reap_error"] = str(cleanup_error)
    finally:
        journal.update(child_returncode=None if child is None else child.poll(),
            elapsed_child_wall_seconds=None if child_started is None else time.monotonic() - child_started,
            elapsed_launcher_wall_seconds=time.monotonic() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(), children_usage_after=children_usage())
        if journal["children_usage_after"]["maxrss_KiB"] * 1024 > RSS_CAP:
            journal.update(status="launcher_failed_or_interrupted", terminal_RSS_cap_exceeded=True)
        try:
            journal["final_disk"] = directory_usage(output)
            require(max(journal["final_disk"].values()) <= freeze["runtime"]["new_run_artifact_cap_bytes"],
                    "Terminal run artifacts exceed 1GiB")
        except BaseException as disk_error:
            journal.update(status="launcher_failed_or_interrupted", final_disk_error=str(disk_error))
        for filename, key in (("progress.json", "child_reported_status"), ("diagnostics.json", "diagnostic_snapshot")):
            path = output / filename
            if path.is_file():
                try:
                    data = json.loads(path.read_text())
                    journal[key] = data.get("status") if filename == "progress.json" else {
                        name: data.get(name) for name in ("charged_union_seconds", "cap_seconds", "final_snapshot", "active_depth")}
                    journal[filename + "_sha256"] = sha(path)
                except (OSError, ValueError) as error:
                    journal[key] = {"unreadable": str(error)}
        save()
    meter = journal.get("diagnostic_snapshot", {})
    complete = journal["status"] == "child_exited" and journal["child_returncode"] == 0 and \
        journal.get("child_reported_status") == "completed_frozen_fitted_screen" and meter.get("final_snapshot") is True and \
        meter.get("active_depth") == 0 and meter.get("charged_union_seconds", float("inf")) <= 60
    print(json.dumps({"complete": complete, "status": journal["status"], "launcher_journal": str(journal_path)}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
