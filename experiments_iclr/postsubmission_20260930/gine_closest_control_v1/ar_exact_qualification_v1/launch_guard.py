"""Proposed root-admitted launcher. Authored only; not executed/compiled."""
import argparse
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
import traceback

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parents[1] / "new_learning_gap_independent_v1/ar_mixture_source_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_all_pins():
    seal = json.loads((BASE / "AUTHORING_SEAL.json").read_text(encoding="utf-8"))
    require(sha(BASE / "AUTHORING_MANIFEST.json") == seal["manifest_sha256"], "Authored manifest changed")
    manifest = json.loads((BASE / "AUTHORING_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = BASE / row["file"]
        require(path.resolve().is_relative_to(BASE), "Authored manifest escapes packet")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Authored bytes changed: " + row["file"])
    pins = json.loads((BASE / "SOURCE_PINS.json").read_text(encoding="utf-8"))
    require(sha(SOURCE / "SOURCE_PACKET_MANIFEST_v1.json") == pins["manifest_sha256"], "Source manifest changed")
    require(sha(SOURCE / "SOURCE_PACKET_SEAL_v1.json") == pins["seal_sha256"], "Source seal changed")
    for name, expected in pins["files"].items():
        require(sha(SOURCE / name) == expected, "Pinned source file changed: " + name)
    for packet_name, record in pins["additional_packets"].items():
        folder = SOURCE.parent / packet_name
        require(sha(folder / "SOURCE_PACKET_MANIFEST_v1.json") == record["manifest_sha256"], "Additional source manifest changed")
        require(sha(folder / "SOURCE_PACKET_SEAL_v1.json") == record["seal_sha256"], "Additional source seal changed")
        for name, expected in record["files"].items():
            require(sha(folder / name) == expected, "Pinned additional source file changed: " + name)
    return seal, pins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-id", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    require(args.admission_id.strip(), "Root admission ID must be recorded")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name) is not None, "Simple private run name required")
    output = BASE / "runs" / args.run_name
    require(output.resolve().is_relative_to(BASE), "Run folder escapes packet")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    terminal = {"status": "started", "admission_id": args.admission_id,
                "hard_child_wall_seconds": 120, "child_started": False,
                "launcher_root_permission": "Admission string is a receipt label, not an authorization mechanism; root review must precede launch."}
    child = None
    try:
        seal, pins = verify_all_pins()
        terminal["authored_manifest_sha256"] = seal["manifest_sha256"]
        terminal["source_pins"] = pins
        command = [sys.executable, "-B", str(BASE / "diagnostic.py"),
                   "--output-dir", str(output), "--admission-id", args.admission_id]
        env = dict(os.environ)
        selected = {"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                    "CUDA_VISIBLE_DEVICES": ""}
        env.update(selected)
        terminal["command"] = command
        terminal["explicit_environment_settings"] = selected
        with (output / "child_stdout.txt").open("w", encoding="utf-8") as stdout, (output / "child_stderr.txt").open("w", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=BASE, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            terminal["child_started"] = True
            child_started = time.monotonic()
            try:
                code = child.wait(timeout=120)
                terminal["status"] = "passed" if code == 0 else "failed"
                terminal["returncode"] = code
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                terminal["returncode"] = child.wait()
                terminal["status"] = "hard_timeout"
                terminal["error"] = "Child exceeded 120-second hard wall limit; no retry"
            terminal["child_wall_seconds"] = time.monotonic() - child_started
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        terminal.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        terminal["child_user_cpu_seconds"] = usage_after.ru_utime - usage_before.ru_utime
        terminal["child_system_cpu_seconds"] = usage_after.ru_stime - usage_before.ru_stime
        terminal["launcher_elapsed_wall_seconds"] = time.monotonic() - started
        terminal["stdout_and_stderr_preserved"] = terminal["child_started"]
        (output / "guard_terminal.json").write_text(json.dumps(terminal, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": terminal["status"], "output": str(output)}), flush=True)
    return 0 if terminal["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
