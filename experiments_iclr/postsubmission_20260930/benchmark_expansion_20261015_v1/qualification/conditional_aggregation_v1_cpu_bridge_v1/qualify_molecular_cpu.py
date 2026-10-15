"""Actual source MolBACE/GIN CPU bridge qualification, six updates total.

Two optimizer transitions per fixed readout across fresh-session checkpoint
restore, on the same32 real TRAIN graphs. Full official VALID151. No scientific
fit, selection, TEST, CUDA model, worker mutation, or automatic retry.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
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
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA explicitly for this CPU qualification")


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--method-package", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    here = Path(__file__).resolve().parent
    job_path, package, output = map(lambda p: Path(p).resolve(), (args.job, args.method_package, args.output))
    if not all(path.is_relative_to(root) for path in (here, job_path, package, output)) or output.exists():
        raise ValueError("Use exact repository-contained sources and fresh qualification output")
    output.mkdir(parents=True, exist_ok=False)
    job = json.loads(job_path.read_text())
    if job["family"] != "ogb" or job["procedure"] != "gnnm_boundary_native" \
            or job["recipe"]["dataset"] != "ogbg-molbace" or job["TEST_access"] is not False:
        raise ValueError("Use the frozen fits004 native MolBACE GNNM TRAIN/VALID descriptor")
    callback = Path(job["callback_module"]).resolve()
    if not callback.is_relative_to(root) or pin(callback)["sha256"] != job["callback_sha256"]:
        raise ValueError("Frozen source callback differs")
    source_manifest = pin(package / "MANIFEST.json")
    if source_manifest["sha256"] != "edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f":
        raise ValueError("The predeclared conditional aggregation source freeze differs")
    import torch
    from torch_geometric.data import Batch
    torch.set_num_threads(2)
    native = load(callback, "_conditional_native_ogb_cpu")
    extension = load(package / "molecular_transform.py", "_conditional_molecular_cpu_transform")
    core = load(package / "core.py", "_conditional_molecular_cpu_core")
    report = {"schema": "conditional-aggregation-native-molecular-CPU-check-v1", "status": "running",
              "job": pin(job_path), "callback": pin(callback), "package_manifest": source_manifest,
              "qualification_script": pin(__file__), "torch": str(torch.__version__), "device": "cpu", "threads": 2,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "TEST_accessed": False,
              "scientific_fit": False, "selection_or_quality_claim": False, "GPU_models_or_updates": 0,
              "original_paper_reruns": 0, "maximum_optimizer_transitions": 6, "cases": []}
    started = time.perf_counter()
    actual_steps, attempts, constructions = 0, 0, 0

    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(actual_optimizer_transitions=actual_steps, optimizer_attempts=attempts,
                      native_constructions=constructions, elapsed_seconds=time.perf_counter() - started,
                      process_user_seconds=usage.ru_utime, process_system_seconds=usage.ru_stime,
                      process_peak_RSS_KiB=usage.ru_maxrss)
        (output / "RESULT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def count_steps(owner, case):
        original = owner.optimizer.step
        def step(*arguments, **keywords):
            nonlocal actual_steps, attempts
            attempts += 1
            case["optimizer_attempts"] = case.get("optimizer_attempts", 0) + 1
            if actual_steps >= 6 or case["actual_optimizer_transitions"] >= 2:
                raise AssertionError("Qualification optimizer bound reached")
            value = original(*arguments, **keywords)
            actual_steps += 1
            case["actual_optimizer_transitions"] += 1
            return value
        owner.optimizer.step = step

    def assert_valid(value, owner, case):
        case["VALID_calls"] += 1
        if value["VALID_scope"] != "complete_source_VALID" or value["VALID_graphs"] != 151 \
                or value["VALID_graph_ids"] != owner.valid_ids.tolist() or value["TEST_accessed"] is not False:
            raise AssertionError("Exact complete official VALID151/order required")
        if not torch.isfinite(value["outputs"]).all() or not torch.isfinite(value["same_bank_mean_outputs"]).all():
            raise AssertionError("Actual served and same-bank mean VALID must be finite")
        case["VALID_graphs"] = value["VALID_graphs"]
        case["VALID_graph_ids"] = value["VALID_graph_ids"]
        case["VALID_selection_uses"] = value["selection_uses"]

    for method in extension.METHODS:
        case_started = time.perf_counter()
        case = {"method": method, "status": "running", "actual_optimizer_transitions": 0, "VALID_calls": 0,
                "TRAIN_graphs_per_update": 32, "same_real_TRAIN_batch_reused": True}
        report["cases"].append(case)
        sessions = []
        try:
            method_job = copy.deepcopy(job)
            method_job.update(procedure=method, base_procedure="gnnm_boundary_native", device="cpu",
                              method_settings={"manifest_sha256": source_manifest["sha256"]})
            method_job["recipe"]["_repository_root"] = str(root)
            method_job["recipe"]["_seed"] = job["seed"]
            recipe = copy.deepcopy(method_job["recipe"])
            torch.manual_seed(job["seed"])
            owner = native.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", job["data_root"], "cpu")
            constructions += 1
            sessions.append(owner)
            if owner.config["gnn"] != "gin" or owner.config["dataset"] != "ogbg-molbace":
                raise ValueError("Only the requested exact native MolBACE/GIN cell is authorized")
            case["native_config"] = copy.deepcopy(owner.config)
            batch = Batch.from_data_list([owner.train_view[index] for index in range(32)])
            positions = torch.arange(32, dtype=torch.long)
            case["TRAIN_graph_ids"] = batch.smoke_graph_id.tolist()
            owner.model.eval()
            with torch.no_grad():
                reference = torch.stack([owner.model.run_member(member, batch)[1] for member in range(4)]).mean(0)
            rng_before = torch.get_rng_state().clone()
            original_refs = dict(owner.source_parameter_references)
            owner = extension.transform(owner, method_job)
            if not torch.equal(rng_before, torch.get_rng_state()):
                raise AssertionError("Gate attachment consumed global CPU RNG")
            if any(p.device.type != "cpu" for p in owner.model.parameters()):
                raise AssertionError("A model Parameter is not on CPU")
            owner.model.eval()
            with torch.no_grad():
                logits = [owner.model.run_member(member, batch)[1] for member in range(4)]
                initial = owner.model.conditional_aggregation(logits)
            if not torch.equal(initial["served_raw_logits"], reference) \
                    or not torch.equal(initial["weights"], reference.new_full((4, 32), 0.25)):
                raise AssertionError("Exact initial native mean/weights differ")
            case["initial_mean_bit_exact"] = True
            case["attachment_global_RNG_unchanged"] = True
            count_steps(owner, case)
            owner._next_train_batch = lambda: (batch, positions, 0)
            case["first_update"] = owner.train_update()
            valid = owner.validate()
            assert_valid(valid, owner, case)
            if not all(dict(owner.model.named_parameters())[name] is parameter for name, parameter in original_refs.items()):
                raise AssertionError("An original native Parameter was replaced")
            checkpoint = output / (method + "_after_one.pt")
            torch.save({"session": owner.state(), "CPU_rng": torch.get_rng_state().clone(), "job": method_job}, checkpoint)
            case["checkpoint"] = pin(checkpoint)
            first_gate = copy.deepcopy(owner.model.conditional_aggregation.state_dict())
            torch.manual_seed(job["seed"])
            restored = native.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", job["data_root"], "cpu")
            constructions += 1
            sessions.append(restored)
            restored = extension.transform(restored, method_job)
            count_steps(restored, case)
            restored._next_train_batch = lambda: (batch, positions, 0)
            restored_ids = {name: id(parameter) for name, parameter in restored.source_parameter_references.items()}
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            restored.load_state(saved["session"])
            if restored_ids != {name: id(parameter) for name, parameter in restored.source_parameter_references.items()}:
                raise AssertionError("Fresh restore replaced original native Parameter objects")
            gate_after_restore = restored.model.conditional_aggregation.state_dict()
            for key, value in first_gate.items():
                if isinstance(value, torch.Tensor):
                    if not torch.equal(value, gate_after_restore[key]):
                        raise AssertionError("Gate checkpoint tensor differs")
                elif value != gate_after_restore[key]:
                    raise AssertionError("Gate checkpoint configuration differs")
            replay = restored.validate()
            assert_valid(replay, restored, case)
            if not torch.equal(valid["outputs"], replay["outputs"]) \
                    or not torch.equal(valid["same_bank_mean_outputs"], replay["same_bank_mean_outputs"]):
                raise AssertionError("Fresh restore did not reproduce both VALID readouts exactly")
            case["restored_VALID_bit_exact"] = True
            restored.load_state(saved["session"])
            torch.set_rng_state(saved["CPU_rng"])
            case["resumed_second_update"] = restored.train_update()
            if restored.updates != 2:
                raise AssertionError("Native update counter did not resume at two")
            final_valid = restored.validate()
            assert_valid(final_valid, restored, case)
            case.update(status="passed", restored_native_parameter_identity=True, gate_state_restored=True,
                        native_update_clock=restored.updates, method_metadata=restored.conditional_aggregation_metadata,
                        native_bridges_qualified=["MolBACE_GIN_CPU_only"], full_fit_or_quality_evidence=False)
        except BaseException as error:
            case.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc(limit=15)})
        finally:
            for session in sessions:
                try:
                    session.close()
                except Exception as error:
                    case.setdefault("close_failures", []).append(str(error))
            case["elapsed_seconds"] = time.perf_counter() - case_started
            write()
    report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"]) else "failed"
    report["limitations"] = ["CPU-only actual MolBACE/GIN engineering on same32 TRAIN graphs twice per variant",
        "no convergence or predictive benefit claim", "no MolHIV, other molecular architecture, node/link or CUDA qualification",
        "final selected-state TEST reconstruction remains unsupported", "no automatic retry or scientific fit"]
    write()
    print(json.dumps({"status": report["status"], "actual_optimizer_transitions": actual_steps,
        "native_constructions": constructions, "cases": [{"method": c["method"], "status": c["status"],
        "optimizer_transitions": c["actual_optimizer_transitions"], "VALID_calls": c["VALID_calls"],
        "failure": c.get("failure", {}).get("message")} for c in report["cases"]]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
