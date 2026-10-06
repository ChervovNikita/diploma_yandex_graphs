"""Disabled stdlib supervisor for exactly one reviewed streamed native utility V6 child."""
import time
STARTED = time.monotonic()
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import resource
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import traceback

SOURCE_RELEASED = False
HOST_REPOSITORIES = {
    "anogena-2-0": "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs",
    "peptide": "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git",
}
QUALIFIER = {
    "path": "matched_first_order_private_gradient_utility_native_qualification_preparation_20261006_v6/qualify.py",
    "bytes": 55314, "sha256": "8b98dfa18ddd1b9990949984079a3654fd968f55346ee07ab5c29bfa59e0232e",
}
# Pending new independent qualifier review: exact immutable descriptor is root-frozen in scope.
WORKER_PASS = "PASS_NATIVE_UTILITY_FP32_EPISODE_RECOMMIT_AND_ISOLATED_SUPPORT_RESOURCE_ONLY"
LIMIT_KEYS = ("max_elapsed_seconds", "max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def utc():
    return datetime.now(timezone.utc).isoformat()


def descriptor(path, relative=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(relative if relative is not None else path),
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def deliberate(phase, value):
    argument = Path(value).absolute()
    require(".." not in argument.parts and argument.is_relative_to(phase)
            and not argument.is_symlink() and argument.resolve().is_relative_to(phase),
            "Deliberate path escaped authorized phase")
    return argument.resolve()


def bound_json(phase, row):
    relative = Path(row["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Relative root metadata required")
    path = deliberate(phase, phase / relative)
    require(path.stat().st_mode & 0o222 == 0 and descriptor(path, relative) == row, "Immutable root metadata differs")
    return json.loads(path.read_text())


def preflight(args):
    host = socket.gethostname()
    require(sys.platform.startswith("linux") and hasattr(os, "wait4") and host in HOST_REPOSITORIES,
            "Named authorized Linux target with wait4 required")
    repository = Path(HOST_REPOSITORIES[host])
    phase_argument = Path(args.source_root).absolute()
    require(phase_argument == repository / "experiments_iclr/postsubmission_20260930"
            and phase_argument.resolve() == phase_argument and Path.cwd().resolve() == repository,
            "Exact authorized repository/phase/cwd required")
    phase = phase_argument
    scope_path = deliberate(phase, args.supervision_scope)
    require(scope_path.stat().st_mode & 0o222 == 0 and descriptor(scope_path)["sha256"] == args.supervision_scope_sha256,
            "Immutable root supervision scope differs")
    scope = json.loads(scope_path.read_text())
    require(scope["schema"] == "root_native_utility_supervision_scope_v1"
            and scope["root_engineering_supervision_authorized"] is True and scope["fixed_before_execution"] is True
            and scope["no_fit_or_scoring"] is True and scope["closed_A_VALID_TEST"] is True
            and scope["hostname"] == host and scope["repository"] == str(repository)
            and scope["supervisor_sha256"] == descriptor(Path(__file__))["sha256"], "Root supervision authority/source differs")
    require(Path(__file__).resolve() == phase / "matched_first_order_private_gradient_utility_native_supervisor_preparation_20261006_v4/supervise.py",
            "Exact deliberate supervisor source path required")
    require(scope["qualifier"] == QUALIFIER and descriptor(phase / QUALIFIER["path"], QUALIFIER["path"]) == QUALIFIER,
            "Only reviewed exact streamed native V6 qualifier admitted")
    review_row = scope["qualifier_review"]
    require(isinstance(review_row, dict) and set(review_row) == {"path", "bytes", "sha256"}
            and isinstance(review_row["path"], str) and type(review_row["bytes"]) is int and review_row["bytes"] > 0
            and isinstance(review_row["sha256"], str) and len(review_row["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in review_row["sha256"]),
            "Actual root-frozen independent V6 qualifier review descriptor required")
    review = bound_json(phase, review_row)
    require(review["status"].startswith("PASS_SOURCE")
            and review["qualifier"] == QUALIFIER and review["new_source_blockers"] == [], "Qualifier review subject/verdict differs")
    supervisor_review = bound_json(phase, scope["supervisor_review"])
    require(supervisor_review["supervisor_sha256"] == scope["supervisor_sha256"]
            and supervisor_review["status"].startswith("PASS_SOURCE")
            and supervisor_review["execution_or_fit_authorized"] is False, "Independent supervisor source review differs")
    worker = bound_json(phase, scope["worker_scope"])
    require(worker["schema"] == "root_native_utility_qualification_execution_scope_v1"
            and worker["root_engineering_invocation_authorized"] is True and worker["fixed_before_execution"] is True
            and worker["qualifier_sha256"] == QUALIFIER["sha256"] and worker["qualifier_review"] == review_row
            and worker["hostname"] == host and worker["repository"] == str(repository)
            and worker["no_fit_or_scoring"] is True and worker["closed_A_VALID_TEST"] is True
            and worker["external_watchdog_required"] is True and worker["external_terminal_accounting_required"] is True,
            "Exact admitted numerical worker scope required")
    require(scope["resource_limits"] == worker["resource_limits"]
            and set(scope["resource_limits"]) == set(LIMIT_KEYS)
            and all(positive(scope["resource_limits"][k]) for k in LIMIT_KEYS), "Same frozen child resource limits required")
    require(positive(scope["external_watchdog_seconds"])
            and scope["external_watchdog_seconds"] == worker["external_watchdog_seconds"]
            and scope["external_watchdog_seconds"] > scope["resource_limits"]["max_elapsed_seconds"]
            and positive(scope["terminate_grace_seconds"]) and positive(scope["kill_reap_seconds"]),
            "Frozen watchdog/termination/reap bounds required")
    executable = Path(worker["python_executable"]).absolute()
    require(executable.is_file() and os.access(executable, os.X_OK)
            and descriptor(executable)["sha256"] == scope["child_python_sha256"], "Frozen normal child interpreter differs")
    require(worker["device"] == "cuda:0" and worker["CUDA_VISIBLE_DEVICES"] == worker["GPU_UUID"]
            and isinstance(worker["GPU_UUID"], str) and worker["GPU_UUID"].startswith("GPU-"), "Frozen single GPU UUID required")
    relative_output = Path(scope["output_relative_path"])
    require(not relative_output.is_absolute() and ".." not in relative_output.parts, "Relative frozen output required")
    output = deliberate(phase, phase / relative_output)
    require(not output.exists() and output.parent.is_dir(), "New supervision output with existing project parent required")
    child_output = output / "child"
    argv = [str(executable), "-B", str(phase / QUALIFIER["path"]), "--execute-authorized", "--source-root", str(phase),
            "--execution-scope", str(phase / scope["worker_scope"]["path"]),
            "--execution-scope-sha256", scope["worker_scope"]["sha256"], "--output", str(child_output)]
    require(argv == scope["child_argv"] and scope["child_cwd"] == str(repository), "Root-frozen exact child argv/cwd required")
    return phase, repository, scope, worker, output, child_output, argv


def physical(child, argv, repository):
    proc = Path("/proc") / str(child.pid)
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    require(int(fields[1]) == os.getpid() and int(fields[2]) == int(fields[3]) == child.pid,
            "Owned child parent/session differs")
    row = {"PID": child.pid, "start_ticks": int(fields[19]), "state": fields[0]}
    if fields[0] == "Z":
        row["exited_before_identity"] = True
        return row
    actual = [v.decode() for v in (proc / "cmdline").read_bytes().split(b"\0") if v]
    require(actual == argv and (proc / "cwd").resolve() == repository
            and (proc / "exe").resolve() == Path(argv[0]).resolve(), "Actual child argv/cwd/interpreter differs")
    row.update(argv=actual, cwd=str(repository), interpreter=str((proc / "exe").resolve()), verified=True)
    return row


def owned_wait(child, observation, scope, force_reason=None):
    """Sole wait4 reaper; never call Popen.poll/wait/terminate/kill (which can reap)."""
    while True:
        pid, status, usage = os.wait4(child.pid, os.WNOHANG)
        if pid:
            child.returncode = os.waitstatus_to_exitcode(status)
            observation.update(exit_code=child.returncode, signal_number=-child.returncode if child.returncode < 0 else None,
                wait_status=status, wait4_peak_RSS_bytes=int(usage.ru_maxrss * 1024),
                wait4_user_CPU_seconds=usage.ru_utime, wait4_system_CPU_seconds=usage.ru_stime,
                whole_child_wall_seconds=time.monotonic() - observation["monotonic_before_Popen"], reaped=True)
            return
        now = time.monotonic()
        watchdog = now - observation["monotonic_before_Popen"] >= scope["external_watchdog_seconds"]
        if "terminate_at" not in observation and (force_reason is not None or watchdog):
            observation["watchdog_fired"] = watchdog
            observation["stop_reason"] = force_reason or "EXTERNAL_WATCHDOG"
            observation["terminate_at"] = now
            try:
                os.kill(child.pid, signal.SIGTERM)  # Unreaped owned handle cannot be PID-reused.
                observation["signals"].append("SIGTERM")
            except ProcessLookupError:
                pass
        if "terminate_at" in observation and "kill_at" not in observation and now - observation["terminate_at"] >= scope["terminate_grace_seconds"]:
            observation["kill_at"] = now
            try:
                os.kill(child.pid, signal.SIGKILL)
                observation["signals"].append("SIGKILL")
            except ProcessLookupError:
                pass
        if "kill_at" in observation and now - observation["kill_at"] >= scope["kill_reap_seconds"]:
            observation["unreaped_child_pid"] = child.pid
            raise TimeoutError("Owned SIGKILL child did not become reapable within frozen cleanup bound")
        time.sleep(0.05)


def main():
    supervisor_started = STARTED
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--supervision-scope")
    parser.add_argument("--supervision-scope-sha256")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_NATIVE_UTILITY_SUPERVISOR_SOURCE_ONLY", "SOURCE_RELEASED": SOURCE_RELEASED}))
        return 0
    require(SOURCE_RELEASED is False and args.supervision_scope and args.supervision_scope_sha256, "Distinct root scope required")
    phase, repository, scope, worker_scope, output, child_output, argv = preflight(args)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    created = output.stat()
    owner = (created.st_dev, created.st_ino)
    def owned_output():
        require(not output.is_symlink() and output.is_dir() and (output.stat().st_dev, output.stat().st_ino) == owner,
                "Only newly created owned supervision directory may be written/finalized")
    def write(name, value):
        owned_output()
        target, temporary = output / name, output / (name + ".tmp")
        require(not target.exists() and not target.is_symlink(), "Exclusive supervision receipt required")
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        owned_output(); temporary.replace(target); target.chmod(0o444)
    terminal = {"schema": "native_utility_owned_supervision_terminal_v1", "status": "SUPERVISOR_FAILURE", "UTC_start": utc(),
        "supervisor_sha256": descriptor(Path(__file__))["sha256"], "root_scope_sha256": args.supervision_scope_sha256,
        "qualifier": QUALIFIER, "qualifier_review": scope["qualifier_review"], "worker_scope": scope["worker_scope"],
        "resource_limits": scope["resource_limits"], "retry": False, "model_fits": 0, "A_scoring": False, "VALID_TEST_access": False,
        "worker_numerical_status": None, "worker_result": None, "wrapper_resource_closure_PASS": False,
        "whole_child_wall_boundary": "Immediately before Popen through sole wait4 return; includes spawn, final publication, exit and observation latency.",
        "supervisor_terminal_write_tail_measured": False, "cleanup_errors": []}
    child = None
    observation = {"reaped": False, "watchdog_fired": False, "signals": []}
    terminal["owned_child"] = observation
    handlers = {n: signal.getsignal(n) for n in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(number, frame):
        raise InterruptedError("Owned supervisor interrupted by signal " + str(number))
    code = 1
    try:
        write("LAUNCH_CLAIM.json", {"UTC": utc(), "argv": argv, "cwd": str(repository), "child_output": str(child_output),
            "scope_sha256": args.supervision_scope_sha256, "one_child_only": True, "no_retry": True})
        for number in handlers:
            signal.signal(number, interrupted)
        owned_output()
        with (output / "CHILD.stdout.log").open("xb") as stdout, (output / "CHILD.stderr.log").open("xb") as stderr:
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=worker_scope["GPU_UUID"], CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
            observation["monotonic_before_Popen"] = time.monotonic()
            child = subprocess.Popen(argv, cwd=repository, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True)
            observation.update(PID=child.pid, requested_argv=argv, requested_cwd=str(repository), requested_interpreter=argv[0])
            try:
                observation["physical_identity"] = physical(child, argv, repository)
            except (FileNotFoundError, ProcessLookupError) as error:
                observation["physical_identity_unavailable"] = str(error)
            write("CHILD_LAUNCH.json", {"UTC": utc(), "PID": child.pid, "argv": argv, "cwd": str(repository),
                "interpreter_sha256": scope["child_python_sha256"], "qualifier_sha256": QUALIFIER["sha256"]})
            owned_wait(child, observation, scope)
        limits = scope["resource_limits"]
        terminal["external_resource_violations"] = []
        if observation["whole_child_wall_seconds"] > limits["max_elapsed_seconds"]:
            terminal["external_resource_violations"].append("WHOLE_CHILD_ELAPSED")
        if observation["wait4_peak_RSS_bytes"] > limits["max_process_rss_bytes"]:
            terminal["external_resource_violations"].append("WAIT4_CHILD_PEAK_RSS")
        result_path = child_output / "RESULT.json"
        if not result_path.exists():
            terminal["status"] = "EARLY_CHILD_FAILURE_NO_RESULT" if observation["exit_code"] else "MISSING_CHILD_RESULT"
        else:
            require(not child_output.is_symlink() and child_output.resolve().is_relative_to(output)
                    and not result_path.is_symlink() and result_path.stat().st_mode & 0o222 == 0, "Immutable owned child result required")
            terminal["worker_result"] = descriptor(result_path, "child/RESULT.json")
            try:
                result = json.loads(result_path.read_text())
            except (ValueError, OSError) as error:
                terminal["status"] = "INVALID_CHILD_RESULT"
                terminal["result_error"] = str(error)
                result = None
            if result is not None:
                terminal["worker_numerical_status"] = result.get("status")
                require(result["worker_sha256"] == QUALIFIER["sha256"] and result["execution_scope"]["sha256"] == scope["worker_scope"]["sha256"],
                        "Child source/scope identity differs")
                if observation["exit_code"] != 0 or result.get("status") != WORKER_PASS:
                    terminal["status"] = "CHILD_NUMERICAL_FAILURE"
                else:
                    require(observation.get("physical_identity", {}).get("verified") is True, "Live child physical identity was not verified")
                    require(result["restoration_errors"] == [] and result["external_terminal_resource_closure_required"] is True
                            and result["actual_native_forward_total"] == 54 and result["actual_ordinary_grad_API_total"] == 88
                            and result["native_reverse_constructions_excluding_maps_query"] == 77
                            and result["model_fits"] == 0 and result["persistent_updates"] == 0
                            and result["A_scoring"] is False and result["VALID_TEST_access"] is False,
                            "Child numerical/restoration/accounting boundary differs")
                    restoration_flags = ("native_caller_states_aliases_flags_inputs_RNG_and_modes_unchanged",
                        "loaded_source_bytes_and_process_gates_unchanged", "NumPy_and_Torch_CPU_visible_CUDA_rng_restored",
                        "backend_restored_exactly", "intra_op_threads_restored", "workspace_and_visible_device_environment_restored",
                        "Python_rng_path_argv_and_prepared_bindings_restored", "signal_handler_timer_restored")
                    require(all(result[k] is True for k in restoration_flags), "Exact child restoration flags incomplete")
                    peaks = result["whole_process_resources"]
                    terminal["worker_reported_resource_capture"] = peaks
                    for name, cap in (("cuda_peak_allocated_bytes", "max_cuda_allocated_bytes"), ("cuda_peak_reserved_bytes", "max_cuda_reserved_bytes")):
                        value = peaks[name]
                        require(type(value) is int and value >= 0, "Missing/nonfinite reported CUDA maximum")
                        if value > limits[cap]: terminal["external_resource_violations"].append(name)
                    checks = {c["name"]: c["result"] for c in result["checks"]}
                    support = checks["nontrivial_native_both_factor_mixed_credit_support"]["support"]
                    require(all(positive(support[k]) and support[k] > 1e-12 for k in ("h_path_shared_max_abs", "g_path_shared_max_abs")),
                            "Both required shared mixed paths not supported")
                    terminal["status"] = "CHILD_WORKER_PASS_WRAPPER_RESOURCE_FAIL" if terminal["external_resource_violations"] else "PASS_SUPERVISOR_TERMINAL_RESOURCE_CLOSURE_NATIVE_SUPPORT_ONLY"
                    terminal["wrapper_resource_closure_PASS"] = not terminal["external_resource_violations"]
        if observation.get("watchdog_fired"):
            terminal["status"] = "EXTERNAL_WATCHDOG_TIMEOUT"
            terminal["wrapper_resource_closure_PASS"] = False
        elif observation.get("signal_number") is not None:
            terminal["status"] = "CHILD_SIGNAL_TERMINATION"
            terminal["wrapper_resource_closure_PASS"] = False
        code = 0 if terminal["wrapper_resource_closure_PASS"] else 1
    except BaseException as error:
        terminal.update(status="SUPERVISOR_FAILURE", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        terminal["wrapper_resource_closure_PASS"] = False
        code = 1
    finally:
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        if child is not None and not observation["reaped"]:
            try:
                owned_wait(child, observation, scope, force_reason="SUPERVISOR_FAILURE_CLEANUP")
            except BaseException as error:
                terminal["cleanup_errors"].append({"type": type(error).__name__, "error": str(error)})
        for number, handler in handlers.items():
            signal.signal(number, handler)
        if observation.get("watchdog_fired"):
            terminal["status"] = "EXTERNAL_WATCHDOG_TIMEOUT"
            terminal["wrapper_resource_closure_PASS"] = False
            code = 1
        elif observation.get("signal_number") is not None:
            terminal["status"] = "CHILD_SIGNAL_TERMINATION"
            terminal["wrapper_resource_closure_PASS"] = False
            code = 1
        if terminal["cleanup_errors"] or not observation["reaped"]:
            terminal["wrapper_resource_closure_PASS"] = False
            code = 1
        terminal["logs"] = []
        for name in ("CHILD.stdout.log", "CHILD.stderr.log"):
            path = output / name
            if path.exists():
                owned_output(); require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), "Owned ordinary log required")
                path.chmod(0o444); terminal["logs"].append(descriptor(path, name))
        wrapper_usage = resource.getrusage(resource.RUSAGE_SELF)
        terminal.update(UTC_end=utc(), supervisor_capture_seconds=time.monotonic() - supervisor_started,
            supervisor_peak_RSS_bytes=int(wrapper_usage.ru_maxrss * 1024),
            supervisor_user_CPU_seconds=wrapper_usage.ru_utime, supervisor_system_CPU_seconds=wrapper_usage.ru_stime)
        write("TERMINAL.json", terminal)
    print(json.dumps({"status": terminal["status"], "worker_status": terminal["worker_numerical_status"], "fit_authorized": False}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
