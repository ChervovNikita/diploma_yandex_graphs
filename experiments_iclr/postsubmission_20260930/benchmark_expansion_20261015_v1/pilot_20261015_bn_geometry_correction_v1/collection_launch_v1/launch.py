"""Thin terminal-only launcher for nine unchanged selected-VALID collectors.

No torch/model import. Default mode prints argv only after the terminal gate.
Collect mode runs one original collector at a time, with no automatic retry.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

JOBS_SHA = "f31440cdb44ef8d979cac8898f824407b0ed01f6db2e96c4954ff0dac0ec57ef"
HOST = "anogena-2-0"
GPU = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
EXPANSION = Path("experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def module(path, name):
    definition = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(value)
    return value


def write(path, value, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid())) if replace else path
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if replace:
        temporary.replace(path)


def terminate(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def interrupted(signum, frame):
    raise InterruptedError("Collector launcher received signal " + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--mode", choices=("commands", "collect", "analyze"), default="commands")
    args = parser.parse_args()
    require(socket.gethostname() == HOST and subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines() == [GPU],
        "Dedicated allocation guard required before project reads")
    root = Path(args.project_root).resolve()
    expansion = root / EXPANSION

    def safe(path):
        path = Path(path)
        path = (path if path.is_absolute() else expansion / path).resolve()
        require(path.is_relative_to(root), "Project-contained paths required")
        return path

    def read(path, expected=None):
        path = safe(path)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        require(expected is None or sha == expected, "Input changed: " + str(path))
        return json.loads(raw), {"path": str(path), "sha256": sha, "bytes": len(raw)}

    here = Path(__file__).resolve().parent
    jobs, jobs_pin = read(here / "JOBS.json", JOBS_SHA)
    require(Path(jobs["project_root"]).resolve() == root, "Repository root differs")
    paths = {}
    for pin in jobs["source_pins"]:
        path = safe(pin["path"])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], "Frozen source differs: " + str(path))
        paths[pin["path"]] = path
    admission, admission_pin = read(args.admission, args.admission_sha256)
    require(admission.get("schema") == "root-reviewed-nine-VALID-collections-v1" and
            admission.get("approved") is True and admission.get("resource_admission_approved") is True and
            admission.get("JOBS_sha256") == JOBS_SHA and admission.get("launcher_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest() and
            admission.get("allocation_GPU_UUID") == GPU and admission.get("maximum_concurrent_collectors") == 1,
            "Root review and one-slot resource admission must bind exact source/jobs")
    allowed = admission["allowed_other_GPU_pids"]
    require(isinstance(allowed, list) and len(allowed) <= 1 and all(type(p) is int and p > 0 for p in allowed),
            "Bind at most the one older GRU compute PID, or none")
    study_output, collected, analysis_output = [safe(admission[k]) for k in
        ("study_output", "collection_output", "analysis_output")]
    analyzer = module(paths[jobs["analyzer"]], "_existing_complete_route_analyzer")
    end, end_pin = read(study_output / "END.json", admission["terminal_END_sha256"])
    outcomes, outcomes_pin = read(study_output / "OUTCOMES.json", admission["terminal_OUTCOMES_sha256"])
    analyzer.terminal_gate(end, outcomes, [j["job_id"] for j in jobs["jobs"]])
    # No new score-bearing file is read before all nine END outcomes pass.
    spec, _ = read(paths["pilot_20261015_bn_geometry_correction_v1/SPEC.json"])
    study = module(paths["pilot_20261015_bn_geometry_correction_v1/study.py"], "_existing_nine_fit_study")
    planned = study.jobs(spec)
    require([digest(j) for j in planned] == [j["job_sha256"] for j in jobs["jobs"]] == spec["job_sha256"],
            "Frozen nine-job join differs")
    report, report_pin = read(study_output / "REPORT.json", admission["terminal_REPORT_sha256"])
    analyzer.report_gate(report, spec, planned)
    reports = {r["job_id"]: r for r in report["all_24_cells"] if r["origin"] == "new"}
    fit_inputs, commands = [], []
    for binding, job in zip(jobs["jobs"], planned):
        fit = safe(study_output / binding["fit_relative_to_study_output"])
        require(fit == study.fit_directory(study_output, binding["index"], job), "Fit suffix differs")
        row = reports[binding["job_id"]]
        require(safe(row["authority"]["path"]) == fit / "RESULT.json", "Terminal fit path differs")
        result, result_pin = read(fit / "RESULT.json", row["authority"]["sha256"])
        actual_job, job_pin = read(fit / "JOB.json")
        require(actual_job == job and digest(actual_job) == result["job_hash"] == binding["job_sha256"] and
                result["status"] == "complete" and result["job_id"] == binding["job_id"] and
                result.get("TEST_accessed") is False and actual_job["TEST_access"] is False and
                actual_job["family"] == "ogb", "Collector JOB/RESULT input identity differs")
        expected = dict(spec["expected_runtime_binding"], extension_sha256=job["session_transform"]["sha256"])
        require(result["runtime_binding"] == expected and result["policy"] == job["policy"], "Selected runtime/controller differs")
        for filename, field in (("selected.pt", "selected_checkpoint"), ("selected_VALID.pt", "selected_VALID_export")):
            require(safe(result[field]["path"]) == fit / filename and (fit / filename).is_file(),
                    "Original collector selected-state/export input missing")
        for filename in ("run_fit.py", "adapters.py", "bookkeeping.py", "common.py"):
            pin = result["runtime_sources"][filename]
            require(pin["sha256"] == expected["runtime_sha256"][filename] and
                    hashlib.sha256(safe(pin["path"]).read_bytes()).hexdigest() == pin["sha256"], "Collector runtime input differs")
        command = [sys.executable, "-B", "-u", str(paths[jobs["collector"]]), "--project-root", str(root),
                   "--fit", str(fit), "--output", str(collected / binding["route_collection_relative_to_fresh_root"])]
        commands.append(command)
        fit_inputs.append({"index": binding["index"], "job_id": binding["job_id"], "RESULT": result_pin, "JOB": job_pin})
    analysis_command = [sys.executable, "-B", "-u", str(paths[jobs["analyzer"]]), "--project-root", str(root),
        "--study-output", str(study_output), "--end-sha256", end_pin["sha256"], "--outcomes-sha256", outcomes_pin["sha256"],
        "--report-sha256", report_pin["sha256"], "--new-route-root", str(collected), "--output", str(analysis_output)]
    if args.mode == "commands":
        print(json.dumps({"collectors": commands, "analysis": {"env": {"CUDA_VISIBLE_DEVICES": ""}, "argv": analysis_command}}, indent=2))
        return 0
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    if args.mode == "analyze":
        receipt, _ = read(collected / "END.json")
        require(receipt["status"] == "complete" and receipt["terminal_END"] == end_pin and
                receipt["JOBS"] == jobs_pin and len(receipt["children"]) == 9 and all(
                    r["status"] in ("complete", "complete_with_sanity_warning") and r["exit_code"] == 0 and
                    not r.get("timed_out") for r in receipt["children"]),
                "All nine admitted collections must finish before analysis")
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(analysis_command, cwd=root, stdin=subprocess.DEVNULL, env=env).returncode

    require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "0", GPU), "Collector must see the one authorized GPU")
    lock_path = safe(admission["slot_lock_path"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, interrupted)
        require(not collected.exists(), "Fresh collection root required; never overwrite or retry")
        collected.mkdir(parents=True)
        started = time.monotonic()
        children = [{"index": b["index"], "job_id": b["job_id"], "status": "not_started"} for b in jobs["jobs"]]
        write(collected / "START.json", {"JOBS": jobs_pin, "admission": admission_pin, "terminal_END": end_pin,
              "terminal_OUTCOMES": outcomes_pin, "terminal_REPORT": report_pin, "fit_inputs": fit_inputs,
              "maximum_concurrent_collectors": 1, "allowed_other_GPU_pids": allowed, "automatic_retries": 0})
        launcher_failure = None
        try:
            for index, command in enumerate(commands):
                active = {int(p.strip()) for p in subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=10).splitlines() if p.strip()}
                require(active <= set(allowed), "Unexpected GPU process; one reserved slot is unavailable")
                before = time.monotonic()
                child = None
                children[index].update(status="started", observed_other_GPU_pids=sorted(active))
                write(collected / "OUTCOMES.json", children, True)
                try:
                    with (collected / f"fit_{index:03}.log").open("x") as log:
                        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, env=env, start_new_session=True)
                        children[index]["pid"] = child.pid
                        try:
                            code = child.wait(timeout=jobs["timeout_seconds"])
                        except subprocess.TimeoutExpired:
                            children[index]["timed_out"] = True
                            terminate(child)
                            code = child.returncode
                except BaseException as error:
                    children[index].update(status="launcher_interrupted_or_failed",
                        failure={"type": type(error).__name__, "message": str(error)},
                        unreceipted_forward_cost="unknown, not zero" if child is not None else "no child started")
                    raise
                finally:
                    if child is not None:
                        terminate(child)
                    children[index]["wall_seconds"] = time.monotonic() - before
                children[index]["exit_code"] = code
                result_path = Path(command[-1]) / "RESULT.json"
                if result_path.exists():
                    result, pin = read(result_path)
                    children[index].update(status=result["status"], result=pin, actual_work=result.get("actual_work"),
                        collector_wall_seconds=result.get("wall_seconds"), own_peak_rss_bytes=result.get("own_peak_rss_bytes"),
                        cuda_peak_allocated_bytes=result.get("cuda_peak_allocated_bytes"), cuda_peak_reserved_bytes=result.get("cuda_peak_reserved_bytes"),
                        failure=result.get("failure"), close_error=result.get("close_error"), sanity=result.get("sanity_against_saved_export"))
                    require(result["TEST_accessed"] is False and result["actual_work"]["optimizer_steps"] == 0 and
                            result["actual_work"]["TRAIN_callbacks"] == 0, "Collector performed unexpected work")
                else:
                    children[index].update(status="missing_terminal_RESULT", unreceipted_forward_cost="unknown, not zero")
                write(collected / "OUTCOMES.json", children, True)
                if code or children[index].get("timed_out") or children[index]["status"] not in ("complete", "complete_with_sanity_warning"):
                    break
        except BaseException as error:
            launcher_failure = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            write(collected / "OUTCOMES.json", children, True)
            complete = all(r["status"] in ("complete", "complete_with_sanity_warning") and
                           r.get("exit_code") == 0 and not r.get("timed_out") for r in children)
            write(collected / "END.json", {"status": "complete" if complete else "stopped_no_retry", "children": children,
                  "JOBS": jobs_pin, "terminal_END": end_pin, "wall_seconds": time.monotonic() - started,
                  "launcher_failure": launcher_failure, "automatic_retries": 0, "new_training_updates": 0, "TEST_accessed": False})
        return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
