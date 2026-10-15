"""Bounded actual OGB construction/update/VALID/checkpoint/resume checks.

Engineering only: two CPU optimizer transitions per candidate on the same32
real TRAIN graphs. No benchmark fit, TEST data, selection or GPU execution.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def guard():
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Wrong allocation; no project source or data may be loaded")


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    guard()  # also guarded by the outer literal-route remote_control.call
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    job_path, output, here = Path(args.job).resolve(), Path(args.output).resolve(), Path(__file__).resolve().parent
    if not all(path.is_relative_to(root) for path in (job_path, output, here)) or output.exists():
        raise ValueError("Fresh repository-contained qualification output required")
    output.mkdir(parents=True, exist_ok=False)
    job = json.loads(job_path.read_text())
    if job["family"] != "ogb" or job["procedure"] != "gnnm_boundary_native" \
            or job["recipe"]["dataset"] != "ogbg-molbace" or job["TEST_access"] is not False:
        raise ValueError("Use the actual frozen MolBACE GNNM TRAIN/VALID descriptor")
    callback_path = Path(job["callback_module"]).resolve()
    if not callback_path.is_relative_to(root) or pin(callback_path)["sha256"] != job["callback_sha256"]:
        raise ValueError("Actual frozen OGB callback differs")
    import torch
    from torch_geometric.data import Batch
    torch.set_num_threads(2)
    native = load(callback_path, "_portable_native_ogb_cpu")
    extension = load(here/"extension.py", "_portable_cpu_extension")
    manifest = pin(here/"MANIFEST.json")
    report = {"schema": "portable-private-input-native-CPU-check-v1", "status": "running",
        "job": pin(job_path), "callback": pin(callback_path), "package_manifest": manifest,
        "torch": str(torch.__version__), "device": "cpu", "threads": 2,
        "TEST_accessed": False, "scientific_fit": False, "benchmark_or_selection_claim": False,
        "GPU_models_or_updates": 0, "original_paper_reruns": 0, "cases": []}
    started = time.perf_counter()
    total_steps, total_attempts, constructions = 0, 0, 0

    def observe_optimizer(session, case):
        original = session.optimizer.step
        def step(*arguments, **keywords):
            nonlocal total_steps, total_attempts
            total_attempts += 1
            case["optimizer_attempts"] = case.get("optimizer_attempts", 0)+1
            value = original(*arguments, **keywords)
            total_steps += 1
            case["actual_optimizer_transitions"] += 1
            return value
        session.optimizer.step = step

    def write():
        report.update(actual_optimizer_transitions=total_steps, native_constructions=constructions,
                      optimizer_attempts=total_attempts,
                      elapsed_seconds=time.perf_counter()-started)
        (output/"RESULT.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")

    for method in extension.METHODS:
        case = {"method": method, "status": "running", "actual_optimizer_transitions": 0,
                "VALID_calls": 0, "TRAIN_graphs_per_update": 32, "same_real_TRAIN_batch_reused": True}
        report["cases"].append(case); sessions = []
        try:
            method_job = copy.deepcopy(job)
            method_job.update(procedure=method, device="cpu", method_settings={"rank_cap": 8,
                "frame_policy": "disjoint_if_fit_else_separate_orthonormal_frames", "manifest_sha256": manifest["sha256"]})
            method_job["recipe"]["_repository_root"] = str(root)
            method_job["recipe"]["_seed"] = job["seed"]
            recipe = copy.deepcopy(method_job["recipe"])
            torch.manual_seed(job["seed"])
            source = native.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", job["data_root"], "cpu")
            constructions += 1; sessions.append(source)
            batch = Batch.from_data_list([source.train_view[index] for index in range(32)])
            positions = torch.arange(32, dtype=torch.long)
            case["TRAIN_graph_ids"] = batch.smoke_graph_id.tolist()
            source.model.eval()
            with torch.no_grad():
                reference = torch.stack([source.model.run_member(m, batch)[0] for m in range(4)]) if method == "portable_input_residual_be_v1" \
                    else source.model.native(batch).unsqueeze(0).expand(4, -1, -1).clone()
            source = extension.transform(source, method_job)
            observe_optimizer(source, case)
            source._next_train_batch = lambda: (batch, positions, 0)
            if any(parameter.device.type != "cpu" for parameter in source.model.parameters()):
                raise AssertionError("CPU qualification placed a parameter on another device")
            source.model.eval()
            with torch.no_grad():
                initial = torch.stack([source.model.run_member(m, batch)[0] for m in range(4)])
            case["initial_function_max_abs_difference"] = float((initial-reference).abs().max())
            torch.testing.assert_close(initial, reference, atol=1e-6, rtol=1e-6)
            first = source.train_update()
            case["first_update"] = first
            valid = source.validate(); case["VALID_calls"] += 1
            if not torch.isfinite(valid["outputs"]).all(): raise AssertionError("Nonfinite actual VALID")
            state_path = output/(method+"_after_one.pt")
            payload = {"session": copy.deepcopy(source.state()), "CPU_rng": torch.get_rng_state().clone(), "job": method_job}
            torch.save(payload, state_path)
            case["checkpoint"] = pin(state_path)
            case["VALID_graphs"] = int(valid["outputs"].shape[0])
            case["VALID_graph_ids"] = valid["VALID_graph_ids"]
            case["VALID_is_complete_source_role"] = valid["VALID_scope"] == "complete_source_VALID"
            torch.manual_seed(job["seed"])
            restored = native.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", job["data_root"], "cpu")
            constructions += 1; sessions.append(restored)
            restored = extension.transform(restored, method_job)
            observe_optimizer(restored, case)
            restored._next_train_batch = lambda: (batch, positions, 0)
            native_ids = {name: id(parameter) for name, parameter in restored.source_parameter_references.items()}
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            restored.load_state(saved["session"])
            if native_ids != {name: id(parameter) for name, parameter in restored.source_parameter_references.items()}:
                raise AssertionError("Native Parameter identity changed on restore")
            replay = restored.validate(); case["VALID_calls"] += 1
            case["restored_VALID_max_abs_difference"] = float((valid["outputs"]-replay["outputs"]).abs().max())
            torch.testing.assert_close(replay["outputs"], valid["outputs"], atol=1e-6, rtol=1e-6)
            restored.load_state(saved["session"]); torch.set_rng_state(saved["CPU_rng"])
            second = restored.train_update()
            case["resumed_second_update"] = second
            if restored.updates != 2: raise AssertionError("Resumed source update clock differs")
            final_valid = restored.validate(); case["VALID_calls"] += 1
            if not torch.isfinite(final_valid["outputs"]).all(): raise AssertionError("Nonfinite resumed VALID")
            case.update(status="passed", restored_native_parameter_identity=True,
                        method_metadata=restored.portable_private_input_metadata,
                        native_update_clock=restored.updates, full_fit_or_quality_evidence=False)
        except BaseException as error:
            case.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc(limit=12)})
        finally:
            for session in sessions:
                try: session.close()
                except Exception as error: case.setdefault("close_failures", []).append(str(error))
            write()
    report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"]) else "failed"
    report["limitations"] = ["CPU-only actual MolBACE engineering", "same32 TRAIN graphs reused for two transitions per variant",
        "no convergence, scientific score, CUDA/pairing/resource or node/link qualification", "no automatic scientific retry"]
    write()
    print(json.dumps({"status": report["status"], "cases": [{"method": case["method"], "status": case["status"],
        "optimizer_transitions": case["actual_optimizer_transitions"], "VALID_calls": case["VALID_calls"]} for case in report["cases"]],
        "actual_optimizer_transitions": total_steps, "native_constructions": constructions}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
