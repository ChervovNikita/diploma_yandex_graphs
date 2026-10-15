"""Exact source CiteSeer-Cheb: two CPU transitions per C6 mode across restore."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import socket
import subprocess
import sys
import time
import traceback

NODE_MANIFEST = "aa90be4c1f989e6b4f81250eb3f30bb6a56ad75d9b242e445d539821be956a0c"


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
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Wrong allocation; no project source/data may be accessed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA explicitly for this CPU-only check")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--smoke-manifest", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    descriptor_path, package, output = map(lambda s: Path(s).resolve(), (args.smoke_manifest, args.package, args.output))
    if not all(path.is_relative_to(root) for path in (descriptor_path, package, output, Path(__file__).resolve())) or output.exists():
        raise ValueError("Fresh repository-contained exact-case qualification required")
    output.mkdir(parents=True, exist_ok=False)
    descriptor = json.loads(descriptor_path.read_text())
    selected = [case for case in descriptor["cases"] if case["id"] == "polyformer-citeseer-cheb/gnnm_boundary_native"]
    if len(selected) != 1:
        raise ValueError("Exact existing source descriptor required; no Physics/link task")
    source_case = selected[0]
    callback = Path(source_case["callback_module"]).resolve()
    if not callback.is_relative_to(root) or pin(package / "MANIFEST.json")["sha256"] != NODE_MANIFEST:
        raise ValueError("Pinned node package or callback repository binding differs")
    extension = load(package / "transform.py", "_C6_actual_node_transform")
    receipts = extension.source_bindings(NODE_MANIFEST, root)
    if not any(row["path"] == str(callback) and row["sha256"] == pin(callback)["sha256"] for row in receipts):
        raise ValueError("Exact callback is not a frozen source dependency")
    import numpy as np
    import torch
    import torch_geometric
    torch.set_num_threads(2)
    source = load(callback, "_C6_actual_native_CiteSeer")
    report = {"schema": "conditional-aggregation-real-node-CPU-v1", "status": "running",
        "source_descriptor": pin(descriptor_path), "source_case_id": source_case["id"], "callback": pin(callback),
        "node_package_manifest": pin(package / "MANIFEST.json"), "frozen_C6_manifest": extension.C6_MANIFEST,
        "qualification_script": pin(__file__), "torch": str(torch.__version__), "PyG": str(torch_geometric.__version__),
        "device": "cpu", "threads": 2, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "TEST_accessed": False, "scientific_fit": False, "GPU_models_or_updates": 0,
        "maximum_optimizer_transitions": 6, "cases": []}
    started = time.perf_counter()
    steps, attempts, constructions = 0, 0, 0

    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(actual_optimizer_transitions=steps, optimizer_attempts=attempts, native_constructions=constructions,
            elapsed_seconds=time.perf_counter() - started, process_user_seconds=usage.ru_utime,
            process_system_seconds=usage.ru_stime, process_peak_RSS_KiB=usage.ru_maxrss,
            total_complete_VALID_calls=sum(c["VALID_calls"] + c["initial_VALID_calls"] for c in report["cases"]))
        (output / "RESULT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def count(owner, case):
        original = owner.optimizer.step
        def step(*arguments, **keywords):
            nonlocal steps, attempts
            attempts += 1
            case["optimizer_attempts"] += 1
            if steps >= 6 or case["actual_optimizer_transitions"] >= 2:
                raise AssertionError("Fixed engineering transition bound reached")
            value = original(*arguments, **keywords)
            steps += 1
            case["actual_optimizer_transitions"] += 1
            return value
        owner.optimizer.step = step

    def rng():
        return {"CPU": torch.get_rng_state().clone(), "python": random.getstate(), "numpy": np.random.get_state()}

    def restore_rng(value):
        torch.set_rng_state(value["CPU"])
        random.setstate(value["python"])
        np.random.set_state(value["numpy"])

    def check_valid(value, owner, case):
        case["VALID_calls"] += 1
        expected = torch.nonzero(owner.data.val_mask, as_tuple=False).flatten().cpu()
        if value["VALID_scope"] != "complete_source_VALID" or len(expected) != 665 \
                or not torch.equal(value["VALID_node_ids"].cpu(), expected) or value["TEST_accessed"] is not False:
            raise AssertionError("Complete exact official VALID665/order required")
        if not torch.isfinite(value["outputs"]).all() or not torch.isfinite(value["same_bank_mean_outputs"]).all():
            raise AssertionError("Both VALID readouts must be finite")
        case["VALID_records"] = len(expected)
        case["VALID_selection_uses"] = value["selection_uses"]

    for method in extension.METHODS:
        case_started = time.perf_counter()
        case = {"method": method, "status": "running", "actual_optimizer_transitions": 0,
                "optimizer_attempts": 0, "VALID_calls": 0, "initial_VALID_calls": 0}
        report["cases"].append(case)
        sessions = []
        try:
            recipe = copy.deepcopy(source_case["recipe"])
            recipe["_repository_root"] = str(root)
            method_job = {"family": "node", "procedure": method, "base_procedure": "gnnm_boundary_native",
                "seed": int(recipe["_seed"]), "recipe": recipe, "TEST_access": False,
                "policy": {"track": "engineering_two_transition_check"},
                "method_settings": {"manifest_sha256": NODE_MANIFEST}}
            owner = source.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", source_case["data_root"], "cpu")
            constructions += 1
            sessions.append(owner)
            if int(owner.data.num_nodes) != 3327 or len(owner.train_ids) != 1929 \
                    or int(owner.data.x.shape[-1]) != 3703 or owner.source.num_classes != 6 or owner.basis != "cheb":
                raise AssertionError("Exact existing CiteSeer-Cheb source cell differs")
            case.update(graph_nodes=int(owner.data.num_nodes), native_feature_width=int(owner.data.x.shape[-1]),
                TRAIN_records=len(owner.train_ids), source_fixed_config=copy.deepcopy(recipe["native_fixed_config"]))
            reference = owner.validate()["outputs"].clone()  # original source mean VALID
            case["initial_VALID_calls"] += 1
            global_before = rng()
            original_refs = dict(owner.source_parameter_references)
            owner = extension.transform(owner, method_job)
            if not torch.equal(global_before["CPU"], torch.get_rng_state()) \
                    or global_before["python"] != random.getstate() \
                    or not np.array_equal(global_before["numpy"][1], np.random.get_state()[1]):
                raise AssertionError("Gate attachment consumed a global RNG stream")
            if any(p.device.type != "cpu" for p in owner.model.parameters()):
                raise AssertionError("A Parameter is not on CPU")
            initial = owner.validate()
            case["initial_VALID_calls"] += 1
            if not torch.equal(initial["outputs"], reference) or not torch.equal(initial["outputs"], initial["same_bank_mean_outputs"]):
                raise AssertionError("Initial served readout differs from exact original native mean")
            case["initial_native_mean_bit_exact"] = True
            case["attachment_global_RNG_unchanged"] = True
            count(owner, case)
            case["first_update"] = owner.train_update()
            if not case["first_update"]["activation_checkpointing"] or not case["first_update"]["whole_native_train"]:
                raise AssertionError("Native checkpointed fullTRAIN path was not retained")
            case["missing_gradients"] = [name for name, p in owner.model.named_parameters() if p.requires_grad and p.grad is None]
            if case["missing_gradients"]:
                raise AssertionError("A trainable native/boundary/gate parameter lost its J gradient")
            valid = owner.validate()
            check_valid(valid, owner, case)
            if not all(dict(owner.model.named_parameters())[name] is p for name, p in original_refs.items()):
                raise AssertionError("Original source Parameter identity changed")
            path = output / (method + "_after_one.pt")
            torch.save({"session": owner.state(), "global_rng": rng(), "job": method_job}, path)
            case["checkpoint"] = pin(path)
            restored = source.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", source_case["data_root"], "cpu")
            constructions += 1
            sessions.append(restored)
            restored = extension.transform(restored, method_job)
            count(restored, case)
            ids = {name: id(p) for name, p in restored.source_parameter_references.items()}
            saved = torch.load(path, map_location="cpu", weights_only=False)
            restored.load_state(saved["session"])
            if ids != {name: id(p) for name, p in restored.source_parameter_references.items()}:
                raise AssertionError("Fresh restore replaced original native Parameters")
            replay = restored.validate()
            check_valid(replay, restored, case)
            if not torch.equal(valid["outputs"], replay["outputs"]) \
                    or not torch.equal(valid["same_bank_mean_outputs"], replay["same_bank_mean_outputs"]):
                raise AssertionError("Both readouts must reproduce bit exactly after fresh restore")
            case["restored_VALID_bit_exact"] = True
            restored.load_state(saved["session"])
            restore_rng(saved["global_rng"])
            case["resumed_second_update"] = restored.train_update()
            final = restored.validate()
            check_valid(final, restored, case)
            if restored.updates != 2:
                raise AssertionError("Native source update clock did not resume at two")
            case.update(status="passed", native_update_clock=restored.updates, original_native_parameter_identity=True,
                activation_checkpointing=True, whole_native_TRAIN=True, whole_native_VALID=True,
                method_metadata=restored.conditional_aggregation_node_metadata, full_fit_or_quality_evidence=False)
        except BaseException as error:
            case.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc(limit=15)})
        finally:
            for owner in sessions:
                try:
                    owner.core.remove_hooks()
                except Exception as error:
                    case.setdefault("close_failures", []).append(str(error))
            case["elapsed_seconds"] = time.perf_counter() - case_started
            write()
    report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"]) else "failed"
    report["limitations"] = ["exact CiteSeer-Cheb CPU2thread source case only", "two fullTRAIN updates per mode, no convergence or predictive claim",
        "no Physics/link/full scientific fit/CUDA/resource admission or GPU-worker task", "final selected-state TEST reconstruction unsupported",
        "initial parity evaluations are additional forward-only complete VALID passes; three post-update/restore passes per method counted in VALID_calls"]
    write()
    print(json.dumps({"status": report["status"], "actual_optimizer_transitions": steps, "native_constructions": constructions,
        "cases": [{"method": c["method"], "status": c["status"], "steps": c["actual_optimizer_transitions"],
            "VALID_calls": c["VALID_calls"], "failure": c.get("failure", {}).get("message")} for c in report["cases"]]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
