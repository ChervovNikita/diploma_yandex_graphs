"""Disabled process-local strict deterministic CUDA response probe.

Two LIVE responses from one initial theta/original-phi bank; no episode, outer
derivative, fallback kernel, tolerance change, qualification or fitting.
"""
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
import sys
import time
import traceback

SOURCE_RELEASED = False
HELPER = ("learnability_responsibility_sequential_same_state_diagnostic_preparation_20261006_v1/diagnose.py",
          "35cfe6269dfb7bf952274a1df4207c6ccc8a18e6b7bb42be76debfa2d87443b4")
PRIOR_DIAGNOSTIC_PATH = "learnability_responsibility_sequential_same_state_diagnostic_execution_root_20261006_v1/RESULT.json"
PRIOR_DIAGNOSTIC_SHA256 = "07e84ef04903d7f3fb574e5c75176214078c002994306c59e6656b5d0dbee194"
WORKSPACE_CONFIG = ":4096:8"
DEADLINE, EXTERNAL_DEADLINE = 300, 350
COUNTS = {"native_forward_calls": 24, "private_gradient_calls": 16,
          "native_vjp_calls": 0, "q_map_primal_calls": 20, "q_map_vjp_calls": 0, "small_query_vjp_calls": 0}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(name, path, digest):
    require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Pinned source differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def backend_snapshot():
    result = {"deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
              "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
              "deterministic_debug_mode": torch.get_deterministic_debug_mode(), "flags": {}}
    for group, owner, names in (("cudnn", torch.backends.cudnn, ("enabled", "deterministic", "benchmark", "allow_tf32")),
        ("cuda_matmul", torch.backends.cuda.matmul, ("allow_tf32", "allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction"))):
        result["flags"][group] = {name: getattr(owner, name) for name in names if hasattr(owner, name)}
    return result


def backend_restore(before):
    torch.use_deterministic_algorithms(before["deterministic_algorithms_enabled"],
                                       warn_only=before["deterministic_algorithms_warn_only"])
    for group, owner in (("cudnn", torch.backends.cudnn), ("cuda_matmul", torch.backends.cuda.matmul)):
        for name, value in before["flags"][group].items():
            setattr(owner, name, value)
    require(backend_snapshot() == before, "Process backend flags were not restored exactly")


def strict_backend_rejection(error):
    message = str(error).lower()
    return isinstance(error, RuntimeError) and ("deterministic" in message or "cublas_workspace_config" in message)


def run(args, receipt, save):
    global torch
    repo, phase, output = Path(args.repository).resolve(), Path(args.source_root).resolve(), Path(args.output).resolve()
    runtime = repo / "experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python"
    require(socket.gethostname() == "anogena-2-0" and Path(sys.executable).absolute() == runtime
            and sys.dont_write_bytecode, "Wrong fixed host/runtime or missing -B")
    require(not any(name == "torch" or name.startswith("torch.") for name in sys.modules),
            "Fresh process required before Torch/CUDA initialization")
    old_env_present = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    old_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    receipt["workspace_environment_before"] = {"present": old_env_present, "value": old_env}
    backend_before, original_threads = None, None
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = WORKSPACE_CONFIG
    receipt["workspace_configured_before_Torch_import"] = True
    try:
        helper = load("deterministic_response_helpers", phase / HELPER[0], HELPER[1])
        full = load("deterministic_response_full_six", phase / helper.FULL_SIX[0], helper.FULL_SIX[1])
        require(full.BASE == helper.BASE and full.SEQUENTIAL == helper.SEQUENTIAL
                and full.ROOT_QUALIFIER == helper.ROOT_QUALIFIER, "Bound qualifier interfaces differ")
        base = load("deterministic_response_base", phase / helper.BASE[0], helper.BASE[1])
        prior_bytes = (phase / PRIOR_DIAGNOSTIC_PATH).read_bytes()
        require(hashlib.sha256(prior_bytes).hexdigest() == PRIOR_DIAGNOSTIC_SHA256, "Completed diagnostic receipt differs")
        prior = json.loads(prior_bytes)
        require(prior["status"] == "DIAGNOSTIC_COMPLETE_NO_QUALIFICATION" and prior["worker_sha256"] == HELPER[1]
                and prior["actual_complete_counts"] == helper.TOTAL_COUNTS
                and prior["candidate_pin"] == list(helper.SEQUENTIAL)
                and prior["preserved_original_failure"] == helper.FAILURE
                and prior["original_failure_status"] == "FAIL_ENGINEERING"
                and prior["caller_states_flags_and_gates_unchanged"] is True
                and prior["original_failure_promoted"] is False and prior["qualification_pass_claim"] is False
                and prior["model_fits"] == 0 and prior["persistent_updates"] == 0
                and prior["A_scoring"] is False and prior["VALID_TEST_access"] is False,
                "Original failure and completed non-qualification diagnostic must remain preserved")
        receipt["prior_non_qualification_diagnostic"] = {"path": PRIOR_DIAGNOSTIC_PATH, "sha256": PRIOR_DIAGNOSTIC_SHA256}
        site = repo / ".venv/lib/python3.11/site-packages"
        require(site.is_dir(), "Missing normal runtime; no fallback")
        sys.path.insert(0, str(site))
        import torch
        import numpy
        require(Path(torch.__file__).resolve().is_relative_to(repo) and not torch.cuda.is_initialized(),
                "Unexpected Torch runtime or premature CUDA initialization")
        backend_before = backend_snapshot()
        require(not torch.cuda.is_initialized(), "Backend flag capture prematurely initialized CUDA")
        receipt["backend_before"] = backend_before
        torch.use_deterministic_algorithms(True, warn_only=False)
        require(torch.are_deterministic_algorithms_enabled()
                and not torch.is_deterministic_algorithms_warn_only_enabled()
                and not torch.cuda.is_initialized(), "Strict deterministic mode must precede CUDA initialization")
        receipt["strict_backend_configured_before_CUDA_initialization"] = True
        receipt["backend_during_probe"] = backend_snapshot()
        # Match the predecessor's one CPU thread for construction/preprocessing;
        # do not change the irreversible interop-thread setting.
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        receipt["process_CPU_threads"] = {"before": original_threads, "during": 1,
                                          "interop_unchanged": torch.get_num_interop_threads()}
        base.torch, base.np, helper.torch = torch, numpy, torch
        device = torch.device(args.device)
        require(str(device) == "cuda:0", "Fixed initial CUDA0 FP32 context required")
        modules = {name: load("deterministic_response_" + name, phase / path, digest)
                   for name, (path, digest) in base.PINS.items()}
        import importlib.metadata
        import torch_geometric
        extension_versions = {}
        for distribution in ("torch-scatter", "torch-sparse", "pyg-lib"):
            try:
                extension_versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                extension_versions[distribution] = None
        receipt["backend_runtime_metadata"] = {"torch_version": str(torch.__version__),
            "torch_module_path": str(Path(torch.__file__).resolve()), "torch_CUDA_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "PyG_version": str(torch_geometric.__version__),
            "PyG_module_path": str(Path(torch_geometric.__file__).resolve()), "extension_versions": extension_versions,
            "GPU_name": torch.cuda.get_device_name(device), "GPU_capability": list(torch.cuda.get_device_capability(device))}
        # A framework flag cannot police third-party extension kernels. The full
        # repeated coordinate comparison remains mandatory even without an error.
        receipt["framework_flag_guarantees_third_party_kernel_determinism"] = False
        receipt["actual_two_response_coordinate_comparison_required"] = True
        op, port = modules["operator"], modules["port"]
        seq = load("deterministic_response_sequential", phase / helper.SEQUENTIAL[0], helper.SEQUENTIAL[1])
        accessor = load("deterministic_response_accessor", phase / full.ACCESSOR[0], full.ACCESSOR[1])
        def gates():
            require(SOURCE_RELEASED is False and helper.SOURCE_RELEASED is False and full.SOURCE_RELEASED is False
                    and op.SOURCE_RELEASED is False and port.PORT_RELEASED is False and seq.SOURCE_RELEASED is False
                    and accessor.SOURCE_RELEASED is True, "Original/reviewed source/process gates changed")
        gates()
        projection = Path(args.public_b_dir).resolve()
        manifest = (projection / "PUBLIC_B_MANIFEST.json").read_bytes()
        require(hashlib.sha256(manifest).hexdigest() == full.PUBLIC_B_MANIFEST["sha256"]
                and len(manifest) == full.PUBLIC_B_MANIFEST["bytes"], "Exact existing public+B projection differs")
        data = accessor.load_public_b(phase, projection, device="cuda:0")
        require(data["provenance"] == prior["public_B_provenance"], "Same native graph/processed edges/roles/projection identity required")
        x, edges = data["features"], data["edge_index"]
        s, ys, r, yr = data["inner_indices"], data["inner_labels"], data["query_indices"], data["query_labels"]
        require(x.shape == (24492, 300) and x.dtype == torch.float32 and s.numel() == 2449 and r.numel() == 2450
                and data["W_ids"].numel() == 4898 and data["B_ids"].numel() == 9797
                and not bool(torch.isin(torch.cat((s, r)), data["A_ids"]).any()), "Exact full native context/roles differ")
        receipt["source_pins"], receipt["public_B_provenance"] = base.PINS, data["provenance"]
        family = base.build_family(modules["native"], modules["boundary"], device, torch.float32, 17)
        forward, theta, phis, _ = port._native_callback_and_state(family, x, edges, expected_nodes=24492, global_stage=True)
        pairs = port._sparse_pairs(op, s, ys, edges, node_count=24492, dtype=torch.float32)
        before = base.snapshot(family, x, edges, device)
        originals = ({n: v.clone() for n, v in theta.items()}, tuple({n: v.clone() for n, v in phi.items()} for phi in phis))
        flags = ([v.requires_grad for v in theta.values()], [[v.requires_grad for v in phi.values()] for phi in phis])
        meters, callback_counts, packages = [], {}, []
        phase_name = "response_1"
        def event(kind, **fields):
            row = {"kind": kind, "phase": phase_name, "monotonic_seconds": time.monotonic(), **fields}
            receipt["last_attempted_operation"] = row
            helper.append_json(output / "ATTEMPTED_OPERATIONS.jsonl", row)
        def counted(core, private):
            callback_counts[phase_name] = callback_counts.get(phase_name, 0) + 1
            event("physical_native_callback_attempt")
            return forward(core, private)
        class DurableCounters(seq.Counters):
            def add(self, key, stage):
                super().add(key, stage)
                event("staged_operation_attempt", operation=key, stage=stage)
        def unchanged():
            result = base.unchanged(before, family, x, edges, device)
            require(flags == ([v.requires_grad for v in theta.values()], [[v.requires_grad for v in phi.values()] for phi in phis])
                    and all(torch.equal(originals[0][n], v) for n, v in theta.items())
                    and all(torch.equal(old[n], v) for old, phi in zip(originals[1], phis) for n, v in phi.items()), "Same original theta/phi values or flags changed")
            gates()
            require(backend_snapshot() == receipt["backend_during_probe"], "Strict backend configuration changed during probe")
            return result
        try:
            for repeat in range(2):
                phase_name = "response_" + str(repeat + 1)
                meter = DurableCounters(); meters.append(meter)
                event("phase_started"); save()
                fresh = seq._engineering_response(op, base.PINS["operator"][1], port, base.PINS["port"][1],
                    theta, phis, counted, pairs, s, ys, r, yr, control="live", engineering_authorized=True, counters=meter)
                require(meter.total == helper.RESPONSE_COUNTS and callback_counts[phase_name] == 12, "Fixed response accounting differs")
                packages.append(helper.response_package(fresh)); del fresh
                unchanged(); receipt["fresh_responses_completed"] = repeat + 1; save()
            receipt["comparison"] = helper.compare_tree(packages[0], packages[1], "strict_deterministic_response1_vs_response2", "$response_package", output)
            receipt["actual_complete_counts"] = {key: sum(m.total[key] for m in meters) for key in COUNTS}
            require(receipt["actual_complete_counts"] == COUNTS and sum(callback_counts.values()) == 24, "Complete probe counters differ")
            receipt["primal_assignment_iterations"] = 160
            del packages
            receipt["constructed_response_states_discarded"] = True
        finally:
            receipt["attempted_counter_snapshots"] = [m.snapshot() for m in meters]
            receipt["independent_physical_callback_attempts"] = dict(callback_counts)
            receipt["restoration_on_exit"] = unchanged()
            receipt["caller_states_flags_and_gates_unchanged"] = True
            save()
        return {"nodes": 24492, "dtype": "torch.float32", "device": "cuda:0", "seed": 17,
                "two_live_responses_same_initial_theta_original_phis": True,
                "episode_or_outer_or_native_VJP_or_map_VJP_or_query_calls": 0,
                "qualification_pass_claim": False, "prior_failures_promoted": False}
    except BaseException as error:
        receipt["primary_probe_exception"] = {"error_type": type(error).__name__, "error": str(error),
                                              "traceback": traceback.format_exc()}
        if strict_backend_rejection(error):
            receipt["precise_backend_reported_unsupported_operation"] = str(error)
            receipt["no_operator_or_precision_fallback_attempted"] = True
        raise
    finally:
        # No host settings or original source files are changed. Restoration is process-local.
        try:
            if backend_before is not None:
                backend_restore(backend_before)
                receipt["backend_restored_exactly"] = True
                receipt["backend_after_restore"] = backend_snapshot()
            if original_threads is not None:
                torch.set_num_threads(original_threads)
                require(torch.get_num_threads() == original_threads, "CPU thread count not restored")
                receipt["CPU_threads_restored"] = True
        except BaseException as error:
            receipt["process_backend_restoration_error"] = {"error_type": type(error).__name__, "error": str(error),
                                                           "traceback": traceback.format_exc()}
            raise
        finally:
            if old_env_present:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_env
            else:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            require(("CUBLAS_WORKSPACE_CONFIG" in os.environ) == old_env_present
                    and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == old_env, "Process workspace environment not restored")
            receipt["workspace_environment_restored_exactly"] = True
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
        print(json.dumps({"status": "DISABLED_DETERMINISTIC_CUDA_RESPONSE_PROBE", "numeric_imports": False, "SOURCE_RELEASED": SOURCE_RELEASED}))
        return
    if not args.output or not args.public_b_dir:
        parser.error("Future engineering execution requires fresh output and exact existing public+B projection")
    require(SOURCE_RELEASED is False, "This engineering probe never flips its guard")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"status": "RUNNING_DETERMINISTIC_RESPONSE_PROBE_NOT_QUALIFICATION", "UTC": datetime.now(timezone.utc).isoformat(),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "comparison_helper_pin": HELPER,
        "prior_diagnostic": {"path": PRIOR_DIAGNOSTIC_PATH, "sha256": PRIOR_DIAGNOSTIC_SHA256},
        "fixed_total_deadline_seconds": DEADLINE, "prospective_external_deadline_seconds": EXTERNAL_DEADLINE,
        "fixed_FP32_atol": 2e-6, "fixed_FP32_rtol": 2e-5, "planned_counts": COUNTS,
        "CUBLAS_WORKSPACE_CONFIG": WORKSPACE_CONFIG, "strict_deterministic_algorithms": True, "warn_only": False,
        "hostname": socket.gethostname(), "python": sys.executable,
        "model_fits": 0, "persistent_updates": 0, "A_scoring": False, "VALID_TEST_access": False,
        "predictive_evidence": False, "qualification_pass_claim": False, "prior_failures_promoted": False}
    def save():
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if "torch" in globals() and torch.cuda.is_initialized():
            receipt["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
            receipt["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
        temporary = output / "RESULT.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, output / "RESULT.json")
    def expired(signum, frame):
        raise TimeoutError("Fixed 300-second deterministic CUDA response deadline exceeded")
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(DEADLINE)
    try:
        save(); receipt.update(run(args, receipt, save))
        receipt["status"] = "DETERMINISTIC_RESPONSE_PROBE_COMPLETE_NO_QUALIFICATION"
    except BaseException as error:
        message = str(error)
        rejected = "precise_backend_reported_unsupported_operation" in receipt or strict_backend_rejection(error)
        receipt.update(status="STRICT_BACKEND_REJECTION_NO_QUALIFICATION" if rejected else "FAIL_DETERMINISTIC_RESPONSE_PROBE_EXECUTION",
                       error_type=type(error).__name__, error=message, traceback=traceback.format_exc())
        if rejected:
            receipt.setdefault("precise_backend_reported_unsupported_operation", message)
            receipt["no_operator_or_precision_fallback_attempted"] = True
        if "process_backend_restoration_error" in receipt:
            receipt["status"] = "FAIL_PROCESS_BACKEND_RESTORATION_NO_QUALIFICATION"
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        save()
        for path in output.iterdir():
            path.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "elapsed_seconds": receipt["elapsed_seconds"],
                      "error": receipt.get("error"), "qualification_pass_claim": False, "prior_failures_promoted": False}))
    raise SystemExit(0 if receipt["status"] == "DETERMINISTIC_RESPONSE_PROBE_COMPLETE_NO_QUALIFICATION" else 1)


if __name__ == "__main__":
    main()
