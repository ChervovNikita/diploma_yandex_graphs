"""Detached supervisor for exactly one authorized synthetic CPU suite."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def write_new(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())


def gpu_guard(config):
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True)
    uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert uuids == [config["expected_GPU_UUID"]], uuids
    return uuids


def verify_files(config):
    for record in config["staged_files"]:
        data = (HERE / record["relative_path"]).read_bytes()
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], record["relative_path"]


def main():
    # UUID is checked before opening any task receipt/input file.
    expected = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
    gpu_guard({"expected_GPU_UUID": expected})
    config = json.loads((HERE / "EXECUTION_INPUT.json").read_text())
    write_new("EXECUTION_START.json", dict(pid=os.getpid(), started_unix=time.time(),
        environment={key: os.environ.get(key) for key in config["environment"]},
        GPU_UUID=expected, CUDA_work_authorized=False, synthetic_only=True))
    finish = dict(finished=False, suite_exit_code=None, numerical_status=None)
    try:
        verify_files(config)
        harness = HERE / "phase_tree" / config["harness_folder"]
        argv = [sys.executable, "-B", str(harness / "launch.py"), "--report", "AUTHORIZED_SYNTHETIC_RESULT.json"]
        env = os.environ.copy(); env.update(config["environment"])
        env["TMPDIR"] = str(HERE / "tmp")
        with (HERE / "SUITE_STDOUT.txt").open("x") as log:
            process = subprocess.Popen(argv, cwd=str(HERE), env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            write_new("SUITE_PID_RECEIPT.json", dict(pid=process.pid, supervisor_pid=os.getpid(),
                argv=argv, environment={key: env[key] for key in list(config["environment"]) + ["TMPDIR"]},
                source_manifest_sha256=config["source_manifest_sha256"], harness_manifest_sha256=config["harness_manifest_sha256"]))
            finish["suite_exit_code"] = process.wait()
        report_path = harness / "AUTHORIZED_SYNTHETIC_RESULT.json"
        if report_path.exists():
            data = report_path.read_bytes()
            assert len(data) <= 131072, "Report larger than authorized small result"
            with (HERE / "SYNTHETIC_RESULT.json").open("xb") as stream:
                stream.write(data)
            result = json.loads(data)
            finish["numerical_status"] = result["numerical_status"]
        verify_files(config)
        finish["staged_source_bytes_unchanged_after"] = True
    except BaseException as error:
        write_new("EXECUTION_FAILURE.json", dict(error_type=type(error).__name__, message=str(error),
                                                traceback=traceback.format_exc()))
        finish["supervisor_failure"] = True
    finally:
        finish.update(finished=True, finished_unix=time.time(), supervisor_pid=os.getpid())
        write_new("EXECUTION_FINISH.json", finish)


if __name__ == "__main__":
    main()
