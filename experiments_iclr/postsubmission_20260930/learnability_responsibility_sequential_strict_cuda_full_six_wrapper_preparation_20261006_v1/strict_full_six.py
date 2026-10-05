"""Disabled strict CUDA wrapper around the exact original all-six qualifier.

The two-response probe is a prerequisite, never a substitute for the full six
episodes/outer derivatives/original-phi recommits. Original failures stay FAIL.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import runpy
import socket
import sys
import time
import traceback

SOURCE_RELEASED = False
QUALIFIER = ("learnability_responsibility_sequential_full_six_preparation_20261006_v1/qualify.py",
             "67a898c606d1e94bf680c84023d832ac73d4620081e79fd232800459b041a780")
PROBE = {"path": "learnability_responsibility_sequential_deterministic_cuda_response_probe_execution_root_20261006_v1/RESULT.json",
         "sha256": "e6fe1a133b13ef9f7197578503c842c4b71205560fbe05b25e8da2c8692b0245", "bytes": 13306}
PROBE_SOURCE_SHA256 = "e4c10c30cf5367ee497f79866257e511977e2be1d56f1e4d508575eeb6c3e8e8"
FAILURE = {"path": "learnability_responsibility_sequential_full_six_execution_root_20261006_v1/RESULT.json",
           "sha256": "e5826a3a38b7fe2a1d8b51abf7722488cd16182c1a70b028a1a629c5c063a6b0", "bytes": 20046}
SEQUENTIAL = ("learnability_responsibility_sequential_autograd_vjp_preparation_20261006_v1/sequential_vjp.py",
              "1f9e704a1b95a98cec688d695094058594758a99bac2f057e2fe90e7f9315167")
TOTAL_COUNTS = {"native_forward_calls": 328, "private_gradient_calls": 180, "native_vjp_calls": 52,
                "q_map_primal_calls": 220, "q_map_vjp_calls": 40, "small_query_vjp_calls": 6}
CONTROLS = ("live", "uniform", "margins", "graph_free", "permuted", "stop_q")
WORKSPACE_CONFIG, INNER_DEADLINE, EXTERNAL_DEADLINE = ":4096:8", 900, 950


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def bound_json(phase, row):
    data = (phase / row["path"]).read_bytes()
    require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "Pinned receipt differs: " + row["path"])
    return json.loads(data)


def backend_snapshot():
    result = {"deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
              "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "deterministic_debug_mode": torch.get_deterministic_debug_mode(), "flags": {}}
    for group, owner, names in (("cudnn", torch.backends.cudnn, ("enabled", "deterministic", "benchmark", "allow_tf32")),
        ("cuda_matmul", torch.backends.cuda.matmul, ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"))):
        result["flags"][group] = {name: getattr(owner, name) for name in names if hasattr(owner, name)}
    return result


def backend_restore(before):
    torch.use_deterministic_algorithms(before["deterministic_algorithms_enabled"], warn_only=before["deterministic_algorithms_warn_only"])
    for group, owner in (("cudnn", torch.backends.cudnn), ("cuda_matmul", torch.backends.cuda.matmul)):
        for name, value in before["flags"][group].items():
            setattr(owner, name, value)
    require(backend_snapshot() == before, "Process backend flags were not restored exactly")


def probe_gate(phase):
    probe = bound_json(phase, PROBE)
    comparison = probe["comparison"]
    require(probe["status"] == "DETERMINISTIC_RESPONSE_PROBE_COMPLETE_NO_QUALIFICATION"
            and probe["worker_sha256"] == PROBE_SOURCE_SHA256
            and probe["actual_complete_counts"] == {"native_forward_calls": 24, "private_gradient_calls": 16,
                "native_vjp_calls": 0, "q_map_primal_calls": 20, "q_map_vjp_calls": 0, "small_query_vjp_calls": 0}
            and comparison["tensor_leaves"] == 269 and comparison["coordinates_compared"] == 146102
            and comparison["max_abs_error"] == 0 and comparison["original_tolerance_violation_count"] == 0
            and comparison["structure_mismatches"] == 0 and comparison["exact_scalar_metadata_mismatches"] == 0
            and comparison["first_failing_path"] is None
            and probe["fixed_FP32_atol"] == 2e-6 and probe["fixed_FP32_rtol"] == 2e-5
            and probe["CUBLAS_WORKSPACE_CONFIG"] == WORKSPACE_CONFIG and probe["strict_deterministic_algorithms"] is True
            and probe["warn_only"] is False and probe["workspace_configured_before_Torch_import"] is True
            and probe["strict_backend_configured_before_CUDA_initialization"] is True
            and probe["backend_restored_exactly"] is True and probe["workspace_environment_restored_exactly"] is True
            and probe["caller_states_flags_and_gates_unchanged"] is True
            and probe["restoration_on_exit"]["native_parameters_modes_attributes_buffers_inputs_and_rng_unchanged"] is True
            and probe["qualification_pass_claim"] is False and probe["prior_failures_promoted"] is False
            and probe["model_fits"] == 0 and probe["persistent_updates"] == 0
            and probe["A_scoring"] is False and probe["VALID_TEST_access"] is False,
            "Exact restored strict two-response observation required; it is not all-six qualification")
    failed = bound_json(phase, FAILURE)
    require(failed["status"] == "FAIL_ENGINEERING" and failed["worker_sha256"] == QUALIFIER[1]
            and failed["candidate_pin"] == list(SEQUENTIAL) and failed["last_attempted_arm"] == "live"
            and failed["source_pins"] == probe["source_pins"]
            and failed["public_B_binding"]["provenance"] == probe["public_B_provenance"]
            and failed["model_fits"] == 0 and failed["persistent_updates"] == 0
            and failed["A_scoring"] is False and failed["VALID_TEST_access"] is False,
            "Original all-six failure must remain preserved under the same source/context binding")
    return probe


def run(args, receipt, save, started):
    global torch
    phase, repo, output = Path(args.source_root).resolve(), Path(args.repository).resolve(), Path(args.output).resolve()
    runtime = repo / "experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python"
    require(socket.gethostname() == "anogena-2-0" and Path(sys.executable).absolute() == runtime and sys.dont_write_bytecode,
            "Wrong fixed host/runtime or missing -B")
    require(not any(n == "torch" or n.startswith("torch.") for n in sys.modules), "Fresh process required before Torch/CUDA")
    # Both exact receipt gates are completed using stdlib before any numerical import or public data access.
    probe = probe_gate(phase)
    receipt["probe_verified_before_numeric_and_data_access"] = True
    receipt["preserved_original_failure"] = FAILURE
    qualifier_path = phase / QUALIFIER[0]
    require(hashlib.sha256(qualifier_path.read_bytes()).hexdigest() == QUALIFIER[1], "Original qualifier bytes differ")
    require(Path(args.public_b_dir).resolve() == Path(probe["public_B_provenance"]["public_b_manifest"]["path"]).parent
            and args.device == "cuda:0", "Exact existing publicB directory/device required")
    old_env_present, old_env = "CUBLAS_WORKSPACE_CONFIG" in os.environ, os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    receipt["workspace_environment_before"] = {"present": old_env_present, "value": old_env}
    backend_before, original_threads, inner_started, inner_ended = None, None, None, None
    old_argv = list(sys.argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE_CONFIG
    receipt["workspace_configured_before_Torch_import"] = True
    try:
        site = repo / ".venv/lib/python3.11/site-packages"
        require(site.is_dir(), "Missing normal runtime; no fallback")
        sys.path.insert(0, str(site))
        import torch
        require(Path(torch.__file__).resolve().is_relative_to(repo) and not torch.cuda.is_initialized(), "Wrong Torch runtime or premature CUDA initialization")
        backend_before = backend_snapshot()
        receipt["backend_before"] = backend_before
        torch.use_deterministic_algorithms(True, warn_only=False)
        require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
                and not torch.cuda.is_initialized(), "Strict framework mode must precede CUDA initialization")
        receipt["strict_backend_configured_before_CUDA_initialization"] = True
        receipt["backend_during_inner"] = backend_snapshot()
        require(receipt["backend_during_inner"] == probe["backend_during_probe"], "Exact previously probed process backend configuration required")
        expected = probe["backend_runtime_metadata"]
        extensions = {}
        for distribution in ("torch-scatter", "torch-sparse", "pyg-lib"):
            try:
                extensions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                extensions[distribution] = None
        observed = {"torch_version": str(torch.__version__), "torch_module_path": str(Path(torch.__file__).resolve()),
            "torch_CUDA_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "PyG_version": importlib.metadata.version("torch-geometric"), "extension_versions": extensions,
            "GPU_name": torch.cuda.get_device_name(0), "GPU_capability": list(torch.cuda.get_device_capability(0))}
        require(all(observed[k] == expected[k] for k in observed), "Same probed backend/runtime/GPU metadata required")
        receipt["backend_runtime_metadata"] = observed
        receipt["framework_flag_guarantees_third_party_kernel_determinism"] = False
        original_threads = torch.get_num_threads()
        receipt["inner_thread_setup_is_original_qualifier_owned"] = {"CPU_before": original_threads, "interop_before": torch.get_num_interop_threads()}
        inner_output = output / "original_qualifier"
        sys.argv = [str(qualifier_path), "--execute-authorized", "--source-root", str(phase), "--repository", str(repo),
                    "--device", "cuda:0", "--output", str(inner_output), "--public-b-dir", str(Path(args.public_b_dir).resolve()),
                    "--public-b-sha", probe["public_B_provenance"]["public_b_manifest"]["sha256"]]
        receipt["inner_invocation_argv"] = list(sys.argv)
        receipt["last_phase"] = "starting_exact_original_qualifier"
        save()
        inner_started = time.monotonic()
        receipt["wrapper_pre_inner_seconds"] = inner_started - started
        try:
            runpy.run_path(str(qualifier_path), run_name="__main__")
            code = 0
        except SystemExit as exit:
            code = exit.code if isinstance(exit.code, int) else 0 if exit.code is None else 1
        finally:
            inner_ended = time.monotonic()
            receipt["inner_runpy_wall_seconds"] = inner_ended - inner_started
            sys.argv = old_argv
        receipt["inner_exit_code"] = code
        require(backend_snapshot() == receipt["backend_during_inner"], "Strict backend flags moved during original qualifier")
        loaded_pyg = sys.modules.get("torch_geometric")
        if loaded_pyg is not None:
            receipt["loaded_PyG_identity"] = {"version": str(loaded_pyg.__version__), "path": str(Path(loaded_pyg.__file__).resolve())}
            require(receipt["loaded_PyG_identity"] == {"version": expected["PyG_version"], "path": expected["PyG_module_path"]},
                    "Original qualifier loaded a different PyG module")
        data = (inner_output / "RESULT.json").read_bytes()
        inner = json.loads(data)
        receipt["original_inner_RESULT"] = {"path": "original_qualifier/RESULT.json", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        receipt["inner_status"], receipt["inner_reported_elapsed_seconds"] = inner["status"], inner["elapsed_seconds"]
        receipt["inner_error"], receipt["inner_traceback"] = inner.get("error"), inner.get("traceback")
        receipt["inner_whole_process_resources"] = {k: inner.get(k) for k in ("process_peak_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")}
        require(inner["worker_sha256"] == QUALIFIER[1] and inner["candidate_pin"] == list(SEQUENTIAL)
                and inner["fixed_total_deadline_seconds"] == INNER_DEADLINE
                and inner["model_fits"] == 0 and inner["persistent_updates"] == 0
                and inner["A_scoring"] is False and inner["VALID_TEST_access"] is False,
                "Exact original inner scope/context/gates or outcome record differs")
        if "source_pins" in inner:
            require(inner["source_pins"] == probe["source_pins"], "Inner loaded source identity differs")
        if "public_B_binding" in inner:
            require(inner["public_B_binding"]["provenance"] == probe["public_B_provenance"], "Inner public graph/roles identity differs")
        if code == 0:
            checks = {row["name"]: row["result"] for row in inner["checks"]}
            require(inner["status"] == "PASS_FULL_SIX_FP32_EPISODE_RECOMMIT_RESOURCE_ONLY"
                    and inner["source_pins"] == probe["source_pins"]
                    and inner["public_B_binding"]["provenance"] == probe["public_B_provenance"]
                    and inner["all_six_full_FP32_episodes_and_original_phi_recommits"] is True
                    and inner["actual_complete_counter_totals"] == TOTAL_COUNTS
                    and inner["caller_states_flags_and_gates_unchanged"] is True
                    and inner["restoration_on_exit"]["native_parameters_modes_attributes_buffers_inputs_and_rng_unchanged"] is True,
                    "An actual complete original all-six restored pass is required")
            for arm in CONTROLS:
                row = checks["complete_full_FP32_episode_and_original_phi_recommit_" + arm]
                require(row["arm"] == arm and row["fixed_FP32_atol"] == 2e-6 and row["fixed_FP32_rtol"] == 2e-5
                        and row["constructed_states_discarded"] is True
                        and row["same_initial_state_restored"]["native_parameters_modes_attributes_buffers_inputs_and_rng_unchanged"] is True,
                        "Missing original complete per-arm recommit/restoration pass: " + arm)
            receipt["actual_original_all_six_pass_verified"] = True
        else:
            receipt["inner_error"], receipt["inner_traceback"] = inner.get("error"), inner.get("traceback")
            receipt["actual_original_all_six_pass_verified"] = False
        (inner_output / "RESULT.json").chmod(0o444)
        return code
    except BaseException as error:
        receipt["primary_wrapper_error"] = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        sys.argv = old_argv
        try:
            if backend_before is not None:
                backend_restore(backend_before)
                receipt["backend_restored_exactly"] = True
                receipt["backend_after_restore"] = backend_snapshot()
            if original_threads is not None:
                receipt["original_inner_interop_after"] = torch.get_num_interop_threads()
                torch.set_num_threads(original_threads)
                require(torch.get_num_threads() == original_threads, "Reversible CPU thread count not restored")
                receipt["CPU_threads_restored"] = True
        except BaseException as error:
            receipt["wrapper_backend_restoration_error"] = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
            raise
        finally:
            if old_env_present:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_env
            else:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            require(("CUBLAS_WORKSPACE_CONFIG" in os.environ) == old_env_present and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == old_env,
                    "Process workspace environment not restored")
            receipt["workspace_environment_restored_exactly"] = True
            if inner_ended is not None:
                receipt["wrapper_post_inner_seconds"] = time.monotonic() - inner_ended
            save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-authorized", action="store_true")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--repository", default="/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--public-b-dir")
    args = parser.parse_args()
    if not args.execute_authorized:
        print(json.dumps({"status": "DISABLED_STRICT_CUDA_FULL_SIX_WRAPPER", "numeric_imports": False, "SOURCE_RELEASED": SOURCE_RELEASED}))
        return
    require(SOURCE_RELEASED is False, "This engineering wrapper never flips its source guard")
    if not args.output or not args.public_b_dir:
        parser.error("Future root engineering execution requires fresh output and exact existing publicB directory")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"status": "RUNNING_STRICT_FULL_SIX_WRAPPER", "UTC": datetime.now(timezone.utc).isoformat(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "original_qualifier_pin": QUALIFIER,
        "strict_probe_prerequisite": PROBE, "original_failure_preserved": FAILURE, "candidate_pin": SEQUENTIAL,
        "inner_fixed_deadline_seconds": INNER_DEADLINE, "prospective_external_deadline_seconds": EXTERNAL_DEADLINE,
        "planned_original_counts": TOTAL_COUNTS, "extra_wrapper_native_forward_or_derivative_calls": 0,
        "model_fits": 0, "persistent_updates": 0, "A_scoring": False, "VALID_TEST_access": False,
        "prior_failures_promoted": False, "fits_or_evaluator_release_authorized": False}
    def save():
        receipt["wrapper_total_elapsed_seconds"] = time.monotonic() - started
        receipt["wrapper_peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if "torch" in globals() and torch.cuda.is_initialized():
            receipt["wrapper_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            receipt["wrapper_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        if "inner_runpy_wall_seconds" in receipt:
            receipt["wrapper_overhead_seconds_excluding_inner_runpy"] = receipt["wrapper_total_elapsed_seconds"] - receipt["inner_runpy_wall_seconds"]
        temporary = output / "WRAPPER_RESULT.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, output / "WRAPPER_RESULT.json")
    code = 1
    try:
        save(); code = run(args, receipt, save, started)
        require(receipt["backend_restored_exactly"] is True and receipt["workspace_environment_restored_exactly"] is True,
                "Wrapper process configuration restoration required")
        receipt["status"] = "PASS_STRICT_FULL_SIX_FP32_EPISODE_RECOMMIT_RESOURCE_ONLY" if code == 0 and receipt["actual_original_all_six_pass_verified"] else "FAIL_STRICT_FULL_SIX_ORIGINAL_QUALIFIER"
    except BaseException as error:
        code = 1
        receipt.update(status="FAIL_STRICT_FULL_SIX_WRAPPER", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        save(); (output / "WRAPPER_RESULT.json").chmod(0o444)
    print(json.dumps({"status": receipt["status"], "inner_status": receipt.get("inner_status"),
        "wrapper_total_elapsed_seconds": receipt["wrapper_total_elapsed_seconds"], "error": receipt.get("error"),
        "prior_failures_promoted": False, "fit_authorized": False}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
