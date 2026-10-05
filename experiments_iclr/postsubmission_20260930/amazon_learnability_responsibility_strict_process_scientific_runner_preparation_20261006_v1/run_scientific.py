"""Disabled-by-default strict fresh-process caller for the exact reviewed pilot.

Explicit CLI opt-in never supplies fit authority: the original scientific
worker's immutable root admission plus actual strict all-six evidence is required.
No successor or admission is enabled by this source. A remains closed.
"""
import time
PROCESS_STARTED = time.monotonic()  # Includes every subsequent import/setup/call.
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import socket
import sys
import traceback

SOURCE_RELEASED = False  # This root caller retains its disabled artifact marker.
REPOSITORY = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs"
WORKER_DISABLED_SHA256 = "3cd18127b66dc7024037f64709bb68d64d1f2a5dae5b8b51dacb1cf218d814c5"
WORKER_RELEASED_SHA256 = "0ec5bb6e4d391a990c15657e9fedffd7f02006c75d375bba0a873235ada2ba76"
STRICT_WRAPPER_SHA256 = "af7d7f925e8c6066ff488a7954b9a14e9db2f0fca414f43220fd643f0d54bb9d"
QUALIFIER_SHA256 = "67a898c606d1e94bf680c84023d832ac73d4620081e79fd232800459b041a780"
SEQUENTIAL_SHA256 = "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167"
PUBLIC_B_RELATIVE = "learnability_responsibility_native_full_execution_root_20261005_v1/roles/public_b"
WORKSPACE_CONFIG = ":4096:8"
DEVICE = "cuda:0"
ARMS = ("live", "uniform", "margins", "graph_free", "permuted", "stop_q")
QUALIFICATION_COUNTS = {"native_forward_calls": 328, "private_gradient_calls": 180,
    "native_vjp_calls": 52, "q_map_primal_calls": 220, "q_map_vjp_calls": 40, "small_query_vjp_calls": 6}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_json(root, row):
    root, relative = Path(root).resolve(), Path(row["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Evidence path escape")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root)
            and path.stat().st_mode & 0o222 == 0, "Immutable in-root evidence required")
    raw = path.read_bytes()
    require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "Evidence bytes differ")
    return path, json.loads(raw)


def load_worker(path):
    path = Path(path).absolute()
    require(not path.is_symlink() and sha(path) == WORKER_RELEASED_SHA256
            and path.stat().st_size == 41365 and path.stat().st_mode & 0o222 == 0,
            "Exact future flag-only scientific V2 worker required")
    name = "_strict_amazon_reviewed_scientific_v2"
    require(name not in sys.modules, "Fresh source binding required")
    spec = importlib.util.spec_from_file_location(name, path)
    worker = importlib.util.module_from_spec(spec)
    sys.modules[name] = worker
    spec.loader.exec_module(worker)  # Exact bound worker has only stdlib top-level imports.
    require(worker.SOURCE_RELEASED is True, "Reviewed future worker must be released")
    return worker


def strict_evidence(worker, phase, admission, certificate, public_b):
    """Match the existing admission to actual strict all-six wrapper + inner PASS."""
    row = admission["strict_full_six_wrapper_result"]
    inner_row = admission["strict_full_six_inner_result"]
    require(row in certificate["raw_result_evidence"] and inner_row in certificate["raw_result_evidence"],
            "Strict wrapper/inner must belong to admitted raw evidence")
    _, scope = bound_json(phase, admission["strict_full_six_execution_scope"])
    require(scope["schema"] == "root_strict_all_six_full_FP32_scope_decision_v1"
            and scope["fixed_before_execution"] is True and scope["controls"] == list(ARMS)
            and scope["full_context_FP32"] is True and scope["worker_sha256"] == STRICT_WRAPPER_SHA256
            and scope["original_qualifier_sha256"] == QUALIFIER_SHA256
            and scope["actual_episode_plus_fresh_recompute_counts"] == QUALIFICATION_COUNTS
            and scope["CUBLAS_WORKSPACE_CONFIG_before_Torch"] == WORKSPACE_CONFIG
            and scope["strict_deterministic_algorithms_before_CUDA"] is True and scope["warn_only"] is False
            and scope["original_qualifier_intra_and_interop_threads"] == 1
            and scope["unsupported_operator_fallback"] is False,
            "Actual qualified strict execution scope differs")
    path, strict = bound_json(phase, row)
    require(strict["status"] == "PASS_STRICT_FULL_SIX_FP32_EPISODE_RECOMMIT_RESOURCE_ONLY"
            and strict["wrapper_sha256"] == STRICT_WRAPPER_SHA256
            and strict["original_qualifier_pin"][1] == QUALIFIER_SHA256
            and strict["candidate_pin"][1] == SEQUENTIAL_SHA256
            and strict["inner_exit_code"] == 0 and strict["actual_original_all_six_pass_verified"] is True
            and strict["inner_status"] == "PASS_FULL_SIX_FP32_EPISODE_RECOMMIT_RESOURCE_ONLY"
            and strict["workspace_configured_before_Torch_import"] is True
            and strict["strict_backend_configured_before_CUDA_initialization"] is True
            and strict["backend_restored_exactly"] is True
            and strict["workspace_environment_restored_exactly"] is True
            and strict["CPU_threads_restored"] is True and strict["original_inner_interop_after"] == 1
            and strict["model_fits"] == 0 and strict["persistent_updates"] == 0
            and strict["A_scoring"] is False and strict["VALID_TEST_access"] is False
            and strict["prior_failures_promoted"] is False,
            "Actual restored strict all-six PASS is required; no oneLIVE/nonstrict substitute")
    configured = strict["backend_during_inner"]
    require(configured["deterministic_algorithms_enabled"] is True
            and configured["deterministic_algorithms_warn_only"] is False
            and strict["strict_probe_prerequisite"]["sha256"] ==
            "e6fe1a133b13ef9f7197578503c842c4b71205560fbe05b25e8da2c8692b0245",
            "Qualified strict configuration identity differs")
    require(inner_row["sha256"] == strict["original_inner_RESULT"]["sha256"]
            and inner_row["bytes"] == strict["original_inner_RESULT"]["bytes"],
            "Admitted inner evidence differs from the wrapper's actual inner descriptor")
    inner_path, inner = bound_json(phase, inner_row)
    require(inner["status"] == "PASS_FULL_SIX_FP32_EPISODE_RECOMMIT_RESOURCE_ONLY"
            and inner["worker_sha256"] == QUALIFIER_SHA256
            and inner["candidate_pin"][1] == SEQUENTIAL_SHA256
            and inner["all_six_full_FP32_episodes_and_original_phi_recommits"] is True
            and inner["actual_complete_counter_totals"] == QUALIFICATION_COUNTS
            and inner["caller_states_flags_and_gates_unchanged"] is True
            and inner["restoration_on_exit"]["native_parameters_modes_attributes_buffers_inputs_and_rng_unchanged"] is True
            and inner["nodes"] == 24492 and inner["dtype"] == "torch.float32" and inner["device"] == DEVICE
            and inner["model_fits"] == 0 and inner["persistent_updates"] == 0
            and inner["A_scoring"] is False and inner["VALID_TEST_access"] is False,
            "Exact complete original all-six inner evidence differs")
    checks = {item["name"]: item["result"] for item in inner["checks"]}
    for arm in ARMS:
        item = checks["complete_full_FP32_episode_and_original_phi_recommit_" + arm]
        require(item["arm"] == arm and item["fixed_FP32_atol"] == 2e-6 and item["fixed_FP32_rtol"] == 2e-5
                and item["constructed_states_discarded"] is True
                and item["same_initial_state_restored"]["native_parameters_modes_attributes_buffers_inputs_and_rng_unchanged"] is True,
                "Admitted strict evidence lacks an original complete arm")
    provenance = inner["public_B_binding"]["provenance"]
    require(Path(inner["public_B_binding"]["directory"]).resolve() == public_b
            and scope["public_B_manifest_sha256"] == provenance["public_b_manifest"]["sha256"]
            and certificate["public_graph_sha256"] == provenance["public_graph"]["sha256"]
            and certificate["roles_sha256"] == provenance["roles"]["sha256"]
            and certificate["native_edge_logical_sha256"] == provenance["preprocessing"]["edge_logical_sha256"],
            "Strict evidence and admitted scientific public context differ")
    return strict, {"wrapper": row, "inner": inner_row, "execution_scope": admission["strict_full_six_execution_scope"]}


def backend_snapshot():
    state = {"deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
             "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
             "deterministic_debug_mode": torch.get_deterministic_debug_mode(), "flags": {}}
    for group, owner, names in (("cudnn", torch.backends.cudnn, ("enabled", "deterministic", "benchmark", "allow_tf32")),
        ("cuda_matmul", torch.backends.cuda.matmul, ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"))):
        state["flags"][group] = {name: getattr(owner, name) for name in names if hasattr(owner, name)}
    return state


def backend_restore(before):
    torch.use_deterministic_algorithms(before["deterministic_algorithms_enabled"],
                                       warn_only=before["deterministic_algorithms_warn_only"])
    for group, owner in (("cudnn", torch.backends.cudnn), ("cuda_matmul", torch.backends.cuda.matmul)):
        for name, value in before["flags"][group].items():
            setattr(owner, name, value)
    require(backend_snapshot() == before, "Reversible backend flags not restored")


def runtime_identity():
    import torch_geometric
    extensions = {}
    for name in ("torch-scatter", "torch-sparse", "pyg-lib"):
        try:
            extensions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            extensions[name] = None
    return {"torch_version": str(torch.__version__), "torch_module_path": str(Path(torch.__file__).resolve()),
        "torch_CUDA_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "PyG_version": str(torch_geometric.__version__), "PyG_module_path": str(Path(torch_geometric.__file__).resolve()),
        "extension_versions": extensions, "GPU_name": torch.cuda.get_device_name(0),
        "GPU_capability": list(torch.cuda.get_device_capability(0))}


def resources():
    row = {"elapsed_seconds_including_imports_setup_worker_and_restore": time.monotonic() - PROCESS_STARTED,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
    if "torch" in globals() and torch.cuda.is_initialized():
        row["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(DEVICE)
        row["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(DEVICE)
    return row


def check_limits(limits):
    row = resources()
    require(row["elapsed_seconds_including_imports_setup_worker_and_restore"] <= limits["max_elapsed_seconds"]
            and row["process_peak_rss_bytes"] <= limits["max_process_rss_bytes"], "Existing root process time/RSS limit exceeded")
    if row["cuda_peak_allocated_bytes"] is not None:
        require(row["cuda_peak_allocated_bytes"] <= limits["max_cuda_allocated_bytes"]
                and row["cuda_peak_reserved_bytes"] <= limits["max_cuda_reserved_bytes"], "Existing root CUDA limits exceeded")


def run(args, receipt, save):
    global torch
    phase, repo = Path(args.source_root).resolve(), Path(args.repository).resolve()
    runtime = repo / "experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python"
    require(str(repo) == REPOSITORY and socket.gethostname() == "anogena-2-0"
            and Path(sys.executable).absolute() == runtime and sys.dont_write_bytecode,
            "Fixed native host/interpreter/-B required")
    require(not any(name == "torch" or name.startswith("torch.") for name in sys.modules), "Fresh child required before Torch")
    public_b = Path(args.public_b_dir).resolve()
    require(public_b == (phase / PUBLIC_B_RELATIVE).resolve(), "Fixed existing public+B directory required")
    worker = load_worker(args.released_worker)
    require(not any(name == "torch" or name.startswith("torch.") for name in sys.modules), "Worker import must remain stdlib-only")
    bindings = worker._read(worker.PACKET / "SOURCE_BINDINGS.json")
    queue = worker._read(worker.PACKET / "QUEUE.json")
    admission, full = worker._prerequisites(phase, bindings, queue, DEVICE)
    strict, evidence = strict_evidence(worker, phase, admission, full, public_b)
    receipt["actual_strict_all_six_and_existing_root_admission_verified_before_numeric_or_W_access"] = True
    receipt["actual_strict_evidence"] = evidence
    receipt["admission_sha256"] = sha(worker.PACKET / "PREREQUISITES.json")
    receipt["source_bindings_sha256"] = sha(worker.PACKET / "SOURCE_BINDINGS.json")
    limits = admission["scientific_resource_limits"]
    watchdog = limits["external_watchdog_seconds"]
    require(admission["external_watchdog_required"] is True and type(watchdog) in (int, float)
            and math.isfinite(watchdog) and watchdog > limits["max_elapsed_seconds"],
            "Existing admission must freeze the outer watchdog duration")
    receipt["root_frozen_resource_limits"] = dict(limits)
    check_limits(limits)
    old_handler, old_timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    require(old_timer == (0.0, 0.0), "Fresh child must not replace an active timer")
    def expired(signum, frame):
        raise TimeoutError("Root-frozen scientific whole-process deadline exceeded")
    old_env_present, old_env = "CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    old_path, old_argv, python_rng = list(sys.path), list(sys.argv), random.getstate()
    backend_before, old_threads, rng_before, numpy = None, None, None, None
    receipt["workspace_environment_before"] = {"present": old_env_present, "value": old_env}
    try:
        signal.signal(signal.SIGALRM, expired)
        remaining = limits["max_elapsed_seconds"] - (time.monotonic() - PROCESS_STARTED)
        require(remaining > 0, "Root-frozen whole-process time exhausted before numerical setup")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE_CONFIG
        receipt["workspace_configured_before_Torch_import"] = True
        site = repo / ".venv/lib/python3.11/site-packages"
        require(site.is_dir(), "Fixed normal runtime is missing; no fallback")
        sys.path.insert(0, str(site))
        import torch
        require(Path(torch.__file__).resolve().is_relative_to(repo) and not torch.cuda.is_initialized(),
                "Wrong Torch source or premature CUDA initialization")
        backend_before = backend_snapshot()
        require(not torch.cuda.is_initialized(), "Backend capture prematurely initialized CUDA")
        torch.use_deterministic_algorithms(True, warn_only=False)
        require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
                and not torch.cuda.is_initialized(), "Strict configuration must precede CUDA initialization")
        receipt["strict_backend_configured_before_CUDA_initialization"] = True
        receipt["backend_before"], receipt["backend_during_science"] = backend_before, backend_snapshot()
        require(receipt["backend_during_science"] == strict["backend_during_inner"], "Qualified strict backend flags differ")
        old_threads = torch.get_num_threads()
        receipt["interop_threads_before"] = torch.get_num_interop_threads()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)  # One-time fresh-process initialization, never claimed reversible.
        require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1, "Exact qualifier thread mode required")
        receipt["CPU_threads_during_science"] = {"intra_op": 1, "interop": 1}
        receipt["interop_restoration"] = "Not attempted; one-time initialization belongs to this fresh child and ends with process termination."
        import numpy
        identity = runtime_identity()  # CUDA may initialize only after strict configuration above.
        expected = strict["backend_runtime_metadata"]
        require(all(identity[key] == expected[key] for key in expected), "Qualified runtime/GPU identity differs")
        require({"version": identity["PyG_version"], "path": identity["PyG_module_path"]} == strict["loaded_PyG_identity"],
                "Qualified loaded PyG identity differs")
        receipt["backend_runtime_metadata"] = identity
        rng_before = {"numpy": numpy.random.get_state(), "torch_cpu": torch.get_rng_state().clone(),
                      "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}
        check_limits(limits)
        receipt["phase"] = "calling_exact_reviewed_run_six"
        receipt["scientific_worker_invocations"] = 1
        save()
        # Exactly one original call. All original scientific gates/pre-W identity,
        # W400, warm16, six H16 arms, counters/failures and closed A remain worker-owned.
        inner_dir = Path(args.output).resolve() / "scientific_run"
        result = worker.run_six(phase, public_b, inner_dir, device=DEVICE)
        receipt["worker_returned_COMPLETE"] = result
        receipt["phase"] = "worker_returned"
        require(backend_snapshot() == receipt["backend_during_science"]
                and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == WORKSPACE_CONFIG
                and torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1,
                "Scientific call changed its admitted execution configuration")
        require(worker.SOURCE_RELEASED is True and sha(worker.__file__) == WORKER_RELEASED_SHA256,
                "Original released worker identity changed")
        check_limits(limits)
    except BaseException as error:
        receipt["primary_runner_exception"] = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        errors = []
        def restore_one(name, function):
            try:
                function()
                receipt[name] = True
            except BaseException as error:
                errors.append({"restoration": name, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        if rng_before is not None:
            def restore_rng():
                numpy.random.set_state(rng_before["numpy"])
                torch.set_rng_state(rng_before["torch_cpu"])
                torch.cuda.set_rng_state_all(rng_before["cuda"])
                after_cuda = torch.cuda.get_rng_state_all()
                require(torch.equal(torch.get_rng_state(), rng_before["torch_cpu"])
                        and len(after_cuda) == len(rng_before["cuda"])
                        and all(torch.equal(a, b) for a, b in zip(after_cuda, rng_before["cuda"])),
                        "Torch CPU/visible-CUDA RNG restoration failed")
                restored = numpy.random.get_state(); original = rng_before["numpy"]
                require(restored[0] == original[0] and numpy.array_equal(restored[1], original[1])
                        and restored[2:] == original[2:], "NumPy RNG restoration failed")
            restore_one("numpy_and_Torch_rng_restored_exactly", restore_rng)
        def verify_loaded_gates():
            for suffix, attribute, expected in (("G0V2", "SOURCE_RELEASED", False),
                ("native_port", "PORT_RELEASED", False), ("accessor", "SOURCE_RELEASED", True),
                ("warm_response_probe", "SOURCE_RELEASED", True), ("sequential", "SOURCE_RELEASED", True)):
                module = sys.modules.get("_amazon_G0_WSR_sequential_v1_" + suffix)
                if module is not None:
                    require(getattr(module, attribute) is expected, "Loaded original/released process gate changed: " + suffix)
            require(worker.SOURCE_RELEASED is True and sha(worker.__file__) == WORKER_RELEASED_SHA256,
                    "Exact worker identity changed during call")
        restore_one("loaded_source_gates_verified_unchanged", verify_loaded_gates)
        if backend_before is not None:
            restore_one("backend_restored_exactly", lambda: backend_restore(backend_before))
        if old_threads is not None:
            def restore_threads():
                torch.set_num_threads(old_threads)
                require(torch.get_num_threads() == old_threads, "Reversible intra-op count not restored")
            restore_one("intra_op_threads_restored_exactly", restore_threads)
            receipt["interop_threads_after"] = torch.get_num_interop_threads()
        random.setstate(python_rng)
        sys.path[:], sys.argv[:] = old_path, old_argv
        if old_env_present:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_env
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        receipt["python_rng_restored_exactly"] = random.getstate() == python_rng
        receipt["workspace_environment_restored_exactly"] = ("CUBLAS_WORKSPACE_CONFIG" in os.environ) == old_env_present and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == old_env
        receipt["sys_path_argv_restored_exactly"] = sys.path == old_path and sys.argv == old_argv
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        receipt["signal_handler_timer_restored"] = signal.getsignal(signal.SIGALRM) == old_handler and signal.getitimer(signal.ITIMER_REAL) == old_timer
        receipt["process_restoration_errors"] = errors
        receipt["phase"] = "reversible_process_state_restored"
        save()
    require(not receipt["process_restoration_errors"] and receipt["python_rng_restored_exactly"]
            and receipt["workspace_environment_restored_exactly"] and receipt["sys_path_argv_restored_exactly"]
            and receipt["signal_handler_timer_restored"], "Scientific runner reversible process restoration failed")
    check_limits(limits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--released-worker")
    parser.add_argument("--public-b-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_STRICT_SCIENTIFIC_ROOT_ADMISSION_REQUIRED", "SOURCE_RELEASED": SOURCE_RELEASED, "numeric_imports": False}))
        return
    require(SOURCE_RELEASED is False, "This default-disabled root caller retains its source marker")
    if not all((args.released_worker, args.public_b_dir, args.output)):
        parser.error("Exact future released worker, fixed public+B directory and fresh outer output required")
    output = Path(args.output).resolve(); output.mkdir(parents=False, exist_ok=False)
    receipt = {"status": "RUNNING_STRICT_SCIENTIFIC_CALLER", "UTC": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha(__file__), "expected_worker_sha256": WORKER_RELEASED_SHA256,
        "SOURCE_RELEASED": SOURCE_RELEASED, "scientific_worker_invocations": 0, "A_scoring": False,
        "VALID_TEST_access": False, "fallback_or_shortening_or_selection": False, "phase": "stdlib_admission_preflight"}
    def save():
        receipt["whole_process_resources"] = resources()
        temporary = output / "RUNNER_RESULT.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, output / "RUNNER_RESULT.json")
    code = 1
    try:
        save(); run(args, receipt, save)
        receipt["status"] = "SCIENTIFIC_WORKER_COMPLETE_STRICT_CONFIGURATION_A_CLOSED"
        code = 0
    except BaseException as error:
        receipt.update(status="FAIL_STRICT_SCIENTIFIC_RUNNER", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        save(); (output / "RUNNER_RESULT.json").chmod(0o444)
    print(json.dumps({"status": receipt["status"], "scientific_worker_invocations": receipt["scientific_worker_invocations"],
        "elapsed_seconds": receipt["whole_process_resources"]["elapsed_seconds_including_imports_setup_worker_and_restore"],
        "error": receipt.get("error"), "A_scoring": False}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
