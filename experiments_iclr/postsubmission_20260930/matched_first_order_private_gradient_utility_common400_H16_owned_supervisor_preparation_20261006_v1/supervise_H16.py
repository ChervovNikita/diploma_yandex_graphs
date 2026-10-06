"""Disabled one-child common400 utility H16 fit supervisor; no scoring.

Reuse reviewed native V4 physical()/owned_wait() at repository cwd.
The exact H16 caller supplies admission, learner, counts and restoration.
"""
import time
STARTED = time.monotonic()
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import traceback

SOURCE_RELEASED = False
REPOSITORY = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
HOST = "anogena-2-0"
UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
CALLER = ("matched_first_order_private_gradient_utility_common400_H16_preparation_20261006_v4/run_utility_H16.py",
          "d27a2c52f5bd3e5817f0271fe8b31df76ec41f7738de62c5d6ef6596d41b1220")
HELPERS = ("matched_first_order_private_gradient_utility_native_supervisor_preparation_20261006_v4/supervise.py",
           "80611cae8ab2e557a11db72c99321bf5f5d550992dbf16f63374364f48f8f808")
NATIVE_TERMINAL = (6212, "380b245b1440e941353cc127b402ae756fe79f5132d785d1bfb51f97f65fbf14")
NATIVE_RESULT = (41967, "581509b4802c0b10f9326088db5bd2fa8b34360decb8aa90bbf648a372fac619")
ENDPOINTS = {"common400_predictions": "common400_complete_predictions.pt",
             "utility_predictions": "utility_H16_complete_predictions.pt",
             "utility_state": "utility_H16.pt", "B_diagnostics": "utility_H16_B_diagnostics.pt"}


def require(value, message):
    if not value: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): h.update(block)
    return h.hexdigest()


def load(phase, row, name):
    path = phase / row[0]
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(phase)
            and path.stat().st_mode & 0o222 == 0 and sha(path) == row[1] and name not in sys.modules,
            "Exact immutable stdlib source required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    require(module.SOURCE_RELEASED is False and not any(n == "torch" or n.startswith("torch.") for n in sys.modules),
            "Reviewed disabled helpers/caller must remain pre-Torch")
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true"); parser.add_argument("--source-root")
    parser.add_argument("--fit-scope"); parser.add_argument("--fit-scope-sha256")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_COMMON400_UTILITY_H16_OWNED_FIT_SUPERVISOR", "SOURCE_RELEASED": False})); return 0
    require(SOURCE_RELEASED is False and socket.gethostname() == HOST and Path.cwd() == Path(REPOSITORY)
            and args.source_root == REPOSITORY + "/experiments_iclr/postsubmission_20260930"
            and args.fit_scope and args.fit_scope_sha256 and sys.dont_write_bytecode
            and not any(n == "torch" or n.startswith("torch.") for n in sys.modules), "Exact fresh allocation fit supervisor required")
    phase = Path(args.source_root); repository = Path(REPOSITORY)
    helper = load(phase, HELPERS, "_H16_reviewed_native_owned_helpers")
    caller = load(phase, CALLER, "_H16_exact_reviewed_caller")
    # Reuse the exact caller's complete admission/native PASS checks before Popen.
    phase, fit, origin, common, child_output = caller.admission(args)
    admission = caller.bound_json(phase, fit["external_fit_supervision_admission"])
    require(admission["schema"] == "root_matched_utility_H16_supervision_admission_v1"
            and admission["root_supervision_fit_authorized"] is True and admission["fixed_before_launch"] is True
            and admission["reviewed_supervisor_sha256"] == sha(__file__)
            and admission["supervisor_source"]["path"] == str(Path(__file__).resolve().relative_to(phase))
            and admission["caller_sha256"] == CALLER[1] and admission["common400"] == caller.COMMON
            and admission["arm"] == caller.ARM and admission["H"] == caller.HORIZON
            and admission["fit_invocation_id"] == fit["fit_invocation_id"]
            and admission["one_owned_child_no_retry"] is True and admission["closed_A_VALID_TEST"] is True,
            "Exact reviewed once-only fit supervision authority required")
    require((fit["native_utility_terminal"]["bytes"], fit["native_utility_terminal"]["sha256"]) == NATIVE_TERMINAL
            and (fit["native_utility_worker_result"]["bytes"], fit["native_utility_worker_result"]["sha256"]) == NATIVE_RESULT,
            "Actual original-monolithic V6 native PASS required")
    for relative, digest in caller.PINS.values():
        path = phase / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(phase)
                and path.stat().st_mode & 0o222 == 0 and sha(path) == digest, "Exact unchanged H16 source dependency required")
    caps = admission["resource_limits"]
    require(caps == fit["fit_resource_limits"] and set(caps) == set(helper.LIMIT_KEYS)
            and all(helper.positive(caps[k]) for k in helper.LIMIT_KEYS), "Same prospective worker/whole-child caps required")
    require(helper.positive(admission["external_watchdog_seconds"])
            and admission["external_watchdog_seconds"] > caps["max_elapsed_seconds"]
            and helper.positive(admission["terminate_grace_seconds"]) and helper.positive(admission["kill_reap_seconds"]),
            "Finite watchdog/termination/reap bounds required")
    require(fit["GPU_UUID"] == UUID and admission["child_cwd"] == str(repository), "One exact assigned allocation GPU/cwd required")
    relative = Path(admission["supervisor_output_relative_path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Relative assigned supervision output required")
    output = helper.deliberate(phase, phase / relative)
    require(output != child_output and output.parent == child_output.parent and not output.exists()
            and output.parent.is_dir(), "Two fresh distinct assigned outputs under one existing parent required")
    argv = [fit["python_executable"], "-B", str(phase / CALLER[0]), "--execute-authorized", "--source-root", str(phase),
            "--fit-scope", str(Path(args.fit_scope).absolute()), "--fit-scope-sha256", args.fit_scope_sha256]
    require(admission["minimum_initial_cuda_free_bytes"] == 70 * 1024**3, "Prospectively fixed 70GiB H16 free floor required")
    sample = subprocess.run(["nvidia-smi", "-i", UUID, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15)
    require(sample.returncode == 0 and len(sample.stdout.strip().splitlines()) == 1, "Fresh selected-GPU availability required")
    gpu_uuid, free_mib = [v.strip() for v in sample.stdout.strip().split(",")]
    require(gpu_uuid == UUID and int(free_mib) * 1024**2 >= admission["minimum_initial_cuda_free_bytes"],
            "Fresh pre-context H16 GPU free floor failed")
    output.mkdir(mode=0o700, parents=False, exist_ok=False); st = output.stat(); owner = (st.st_dev, st.st_ino)
    def owned():
        require(not output.is_symlink() and output.is_dir() and (output.stat().st_dev, output.stat().st_ino) == owner,
                "Only this newly created supervision output is owned")
    def write(name, row):
        owned(); target = output / name; temp = output / (name + ".tmp")
        require(not target.exists() and not target.is_symlink(), "Exclusive owned supervision receipt required")
        with temp.open("x") as stream:
            json.dump(row, stream, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        owned(); temp.replace(target); target.chmod(0o444)
    terminal = {"schema": "common400_utility_H16_owned_fit_terminal_v1", "status": "FAIL_OWNED_H16_FIT",
        "UTC_start": datetime.now(timezone.utc).isoformat(), "supervisor_sha256": sha(__file__), "caller_sha256": CALLER[1],
        "fit_scope_sha256": args.fit_scope_sha256, "supervision_admission": fit["external_fit_supervision_admission"],
        "fit_invocation_id": fit["fit_invocation_id"], "common400": caller.COMMON, "origin_run": caller.ORIGIN,
        "arm": caller.ARM, "H": caller.HORIZON, "resource_limits": caps, "launch_attempts": 0,
        "native_utility_terminal": fit["native_utility_terminal"], "native_utility_worker_result": fit["native_utility_worker_result"],
        "initial_GPU_free_sample": {"UUID": gpu_uuid, "free_MiB": int(free_mib), "pre_context_floor_bytes": admission["minimum_initial_cuda_free_bytes"]},
        "A_scoring": False, "VALID_TEST_access": False, "comparison_complete": False, "wrapper_resource_closure_PASS": False,
        "numeric_imports_in_supervisor": False, "no_retry": True, "cleanup_errors": [], "outcomes": {}}
    child = None; observation = {"reaped": False, "watchdog_fired": False, "signals": []}; terminal["owned_child"] = observation
    handlers = {n: signal.getsignal(n) for n in (signal.SIGINT, signal.SIGTERM)}
    def interrupted(number, frame): raise InterruptedError("Owned H16 supervisor interrupted: " + str(number))
    code = 1
    try:
        write("LAUNCH_CLAIM.json", {"argv": argv, "cwd": str(repository), "child_output": str(child_output),
            "fit_scope_sha256": args.fit_scope_sha256, "one_child_only": True, "no_retry": True})
        for number in handlers: signal.signal(number, interrupted)
        with (output / "CHILD.stdout.log").open("xb") as stdout, (output / "CHILD.stderr.log").open("xb") as stderr:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
            observation["monotonic_before_Popen"] = time.monotonic(); terminal["launch_attempts"] = 1
            child = subprocess.Popen(argv, cwd=repository, env=env, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True)
            observation.update(PID=child.pid, requested_argv=argv, requested_cwd=str(repository), requested_interpreter=argv[0])
            observation["physical_identity"] = helper.physical(child, argv, repository)
            write("CHILD_LAUNCH.json", observation)
            helper.owned_wait(child, observation, admission)
        require(observation["reaped"] and observation["exit_code"] == 0 and not observation["watchdog_fired"]
                and observation["signal_number"] is None and observation["signals"] == []
                and observation["physical_identity"].get("verified") is True, "Successful owned H16 exit/identity/wait4 required")
        require(child_output.is_dir() and not child_output.is_symlink() and child_output.resolve().is_relative_to(phase),
                "Declared newly owned caller output must remain a real directory")
        child_stat = child_output.stat(); child_owner = (child_stat.st_dev, child_stat.st_ino)
        terminal["child_output_identity"] = {"path": str(child_output), "device": child_owner[0], "inode": child_owner[1]}
        def immutable(path, relative):
            require(not child_output.is_symlink() and (child_output.stat().st_dev, child_output.stat().st_ino) == child_owner
                    and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(child_output)
                    and path.stat().st_mode & 0o222 == 0, "Immutable owned H16 artifact required")
            return helper.descriptor(path, relative)
        result_row = immutable(child_output / "WORKER_TERMINAL.json", "WORKER_TERMINAL.json")
        result = json.loads((child_output / "WORKER_TERMINAL.json").read_text()); terminal["worker_terminal"] = result_row
        require(result["schema"] == "common400_utility_H16_worker_terminal_v1"
                and result["status"] == "COMPLETE_UTILITY_H16_WORKER_CAPTURE_PENDING_EXTERNAL_RESOURCE_CLOSURE"
                and result["arm"] == caller.ARM and result["H"] == result["completed_episodes"] == caller.HORIZON
                and result["common400"] == caller.COMMON and result["origin_run"] == caller.ORIGIN
                and result["fit_scope_sha256"] == args.fit_scope_sha256 and result["primary_error"] is None
                and result["restoration_errors"] == [] and result["scientific_continuations_attempted"] == 1
                and result["committed_core_updates"] == 16 and result["committed_private_row_updates"] == 64
                and result["new_acquisition_updates"] == 0 and result["Adam_history_activated"] is False
                and result["A_scoring"] is False and result["VALID_TEST_access"] is False and result["original_pilot_changed"] is False
                and result["external_fit_terminal_closure_required"] is True, "Complete fixed H16/restoration/custody required")
        counts = result["operation_counts"]["total"]
        require(counts == {k: caller.HORIZON * v for k, v in caller.EPISODE.items()}
                and result["independent_native_callbacks"] == {"continuation": 640, "serving": 8}
                and result["ordinary_grad_API_attempts"] == 1008, "Exact H16 episode and serving work required")
        reverse = sum(counts[k] for k in ("private_gradient_calls", "native_vjp_calls", "utility_margin_phi_vjp_calls",
                                       "utility_dummy_cotangent_reverse_calls", "utility_weighted_margin_private_gradient_calls"))
        require(reverse == 832, "Exact unchanged native reverse bill required")
        peaks = result["worker_resources"]; terminal["worker_reported_resource_capture"] = peaks
        require(observation["whole_child_wall_seconds"] <= caps["max_elapsed_seconds"]
                and observation["wait4_peak_RSS_bytes"] <= caps["max_process_rss_bytes"]
                and all(type(peaks[n]) in (int, float) and peaks[n] >= 0 and peaks[n] <= caps[c] for n, c in
                    (("elapsed_seconds", "max_elapsed_seconds"), ("process_peak_rss_bytes", "max_process_rss_bytes"),
                     ("cuda_peak_allocated_bytes", "max_cuda_allocated_bytes"), ("cuda_peak_reserved_bytes", "max_cuda_reserved_bytes"))),
                "Whole-child wall/kernel RSS/final worker time/RSS/CUDA caps exceeded")
        run_row = immutable(child_output / "RUN.json", "RUN.json"); run = json.loads((child_output / "RUN.json").read_text())
        require(run["schema"] == "common400_utility_H16_run_v1" and run["arm"] == caller.ARM and run["H"] == 16
                and run["common400"] == caller.COMMON and run["origin_run"] == caller.ORIGIN and run["origin_recipe"] == origin["recipe"]
                and run["fit_scope_sha256"] == args.fit_scope_sha256 and run["coefficients"] == caller.COEFFICIENTS
                and run["new_acquisition_updates"] == 0 and run["A_scoring"] is False and run["VALID_TEST_access"] is False
                and run["original_six_arm_pilot_changed"] is False, "Exact common400/recipe/runtime fit custody required")
        require(set(result["endpoints"]) == set(ENDPOINTS), "All four fixed H16 artifacts required")
        for key, name in ENDPOINTS.items():
            row = result["endpoints"][key]; require(row["path"] == name, "Only fixed H16 endpoint names allowed")
            actual = immutable(child_output / name, name); require(actual == row, "Exact immutable H16 endpoint bytes differ")
            terminal["outcomes"][key] = actual
        attempted = immutable(child_output / "ATTEMPTED_OPERATIONS.jsonl", "ATTEMPTED_OPERATIONS.jsonl")
        require(attempted == result["attempted_events"], "All actual attempted-operation metadata retained")
        require(sha(common) == caller.COMMON["sha256"] and sha(phase / CALLER[0]) == CALLER[1]
                and sha(phase / HELPERS[0]) == HELPERS[1] and sha(args.fit_scope) == args.fit_scope_sha256,
                "Common/source/scope bytes changed")
        terminal.update(status="PASS_COMMON400_UTILITY_H16_OWNED_WHOLE_CHILD_RESOURCE_CLOSURE_A_CLOSED",
            wrapper_resource_closure_PASS=True, worker_status=result["status"], worker_restoration_errors=[],
            RUN=run_row, attempted_events=attempted, operation_counts=counts,
            native_callbacks=sum(result["independent_native_callbacks"].values()),
            native_reverse_constructions=reverse, ordinary_grad_API_attempts=1008)
        code = 0
    except BaseException as error:
        terminal.update(status="FAIL_OWNED_H16_FIT", wrapper_resource_closure_PASS=False,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()); code = 1
    finally:
        for number in handlers: signal.signal(number, signal.SIG_IGN)
        if child is not None and not observation["reaped"]:
            try: helper.owned_wait(child, observation, admission, force_reason="SUPERVISOR_FAILURE_CLEANUP")
            except BaseException as error: terminal["cleanup_errors"].append({"type": type(error).__name__, "error": str(error)})
        for number, handler in handlers.items(): signal.signal(number, handler)
        if terminal["cleanup_errors"] or not observation["reaped"]:
            terminal.update(status="FAIL_OWNED_H16_FIT", wrapper_resource_closure_PASS=False); code = 1
        terminal["owned_child_output_artifacts"] = []
        for name in ("WORKER_TERMINAL.json", "RUN.json", "ATTEMPTED_OPERATIONS.jsonl", *ENDPOINTS.values()):
            path = child_output / name
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(child_output):
                terminal["owned_child_output_artifacts"].append(helper.descriptor(path, name))
        terminal["logs"] = []
        for name in ("CHILD.stdout.log", "CHILD.stderr.log"):
            path = output / name
            if path.is_file() and not path.is_symlink():
                owned(); path.chmod(0o444); terminal["logs"].append(helper.descriptor(path, name))
        usage = resource.getrusage(resource.RUSAGE_SELF)
        terminal.update(UTC_end=datetime.now(timezone.utc).isoformat(), supervisor_capture_seconds=time.monotonic() - STARTED,
            supervisor_peak_RSS_bytes=int(usage.ru_maxrss * 1024), supervisor_user_CPU_seconds=usage.ru_utime,
            supervisor_system_CPU_seconds=usage.ru_stime)
        write("TERMINAL.json", terminal)
    print(json.dumps({"status": terminal["status"], "A_scoring": False, "retry": False})); return code


if __name__ == "__main__": raise SystemExit(main())
