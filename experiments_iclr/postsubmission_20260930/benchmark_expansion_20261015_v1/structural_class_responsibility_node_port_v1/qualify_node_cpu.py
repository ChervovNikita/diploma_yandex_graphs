"""Two CPU source CiteSeer updates and full VALID/checkpoint per candidate."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    assert socket.gethostname() == "anogena-2-0"
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == [
        "GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--smoke-manifest", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    source_path, package, output = Path(args.smoke_manifest).resolve(), Path(args.package).resolve(), Path(args.output).resolve()
    if not all(path.is_relative_to(root) for path in (source_path, package, output, Path(__file__).resolve())) or output.exists():
        raise ValueError("Fresh, explicitly repository-bound engineering output required")
    output.mkdir(parents=True, exist_ok=False)
    descriptor = json.loads(source_path.read_text())
    selected = [case for case in descriptor["cases"] if case["id"] == "polyformer-citeseer-cheb/gnnm_boundary_native"]
    if len(selected) != 1: raise ValueError("Exact existing source smoke descriptor required")
    case_source = selected[0]
    callback = Path(case_source["callback_module"]).resolve()
    if not callback.is_relative_to(root): raise ValueError("Source callback escapes repository")
    manifest = pin(package/"MANIFEST.json")
    if manifest["sha256"] != args.manifest_sha256:
        raise ValueError("This finite check binds the explicitly frozen C2 node package")
    rows = {row["path"]: row for row in json.loads((package/"MANIFEST.json").read_text())["files"]}
    if pin(package/"extension.py")["sha256"] != rows["extension.py"]["sha256"]:
        raise ValueError("Pinned extension source changed")
    extension = load(package/"extension.py", "_structural_class_native_node_extension")
    dependencies = extension.source_bindings(root, manifest["sha256"])
    if not any(row["path"] == str(callback) and row["sha256"] == pin(callback)["sha256"] for row in dependencies):
        raise ValueError("Actual native callback is not the frozen source binding")
    import numpy as np
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
        raise ValueError("This bounded check requires CUDA hidden before importing Torch")
    torch.set_num_threads(2)
    source = load(callback, "_structural_class_actual_citeseer_cpu")
    report = {"schema": "structural-class-responsibility-real-node-CPU-v1", "status": "running",
        "smoke_manifest": pin(source_path), "source_case_id": case_source["id"], "callback": pin(callback),
        "package_manifest": manifest, "script": pin(Path(__file__)), "torch": str(torch.__version__),
        "device": "cpu", "threads": 2, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "TEST_accessed": False, "scientific_fit": False,
        "GPU_models_or_updates": 0, "cases": []}
    mem = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()}
    basis_path = Path(case_source["recipe"]["_redesign_root"])/"smoke_20261014/working_data/node_bases"/(case_source["recipe"]["id"]+".pt")
    report["preflight"] = {"MemAvailable": mem["MemAvailable"], "cached_basis": pin(basis_path)}
    started = time.perf_counter(); total_steps = total_attempts = constructions = 0

    def write():
        report.update(actual_optimizer_transitions=total_steps, optimizer_attempts=total_attempts,
                      native_constructions=constructions, elapsed_seconds=time.perf_counter()-started)
        (output/"RESULT.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")

    def observe(session, case):
        original = session.optimizer.step
        def step(*a, **kw):
            nonlocal total_steps, total_attempts
            total_attempts += 1; case["optimizer_attempts"] += 1
            value = original(*a, **kw)
            total_steps += 1; case["actual_optimizer_transitions"] += 1
            return value
        session.optimizer.step = step

    def rng():
        return {"CPU": torch.get_rng_state().clone(), "python": random.getstate(), "numpy": np.random.get_state()}

    def restore_rng(value):
        torch.set_rng_state(value["CPU"]); random.setstate(value["python"]); np.random.set_state(value["numpy"])

    for method in extension.METHODS:
        case = {"method": method, "status": "running", "optimizer_attempts": 0,
                "actual_optimizer_transitions": 0, "VALID_calls": 0}; report["cases"].append(case)
        sessions = []
        try:
            recipe = copy.deepcopy(case_source["recipe"]); recipe["_repository_root"] = str(root)
            method_job = {"family": "node", "procedure": method, "seed": int(recipe["_seed"]),
                "recipe": recipe, "TEST_access": False, "policy": {"track": "engineering_two_transition_check"},
                "method_settings": {"manifest_sha256": manifest["sha256"]}}
            native = source.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", case_source["data_root"], "cpu")
            constructions += 1; sessions.append(native)
            case.update(graph_nodes=int(native.data.num_nodes), native_feature_width=int(native.data.x.shape[-1]),
                TRAIN_records=len(native.train_ids), VALID_records=int(native.data.val_mask.sum()),
                source_fixed_config=copy.deepcopy(recipe["native_fixed_config"]))
            native.model.eval()
            with torch.no_grad():
                reference = torch.stack([route.raw_logits for route in native._routes(False)])
            groups_before = [{key: copy.deepcopy(value) for key, value in group.items() if key != "params"} for group in native.optimizer.param_groups]
            owners_before = [[id(p) for p in group["params"]] for group in native.optimizer.param_groups]
            native = extension.transform(native, method_job); observe(native, case)
            if groups_before != [{key: value for key, value in group.items() if key != "params"} for group in native.optimizer.param_groups] or owners_before != [[id(p) for p in group["params"]] for group in native.optimizer.param_groups]:
                raise AssertionError("Native optimizer groups/options/owners changed on attachment")
            case["native_optimizer_groups_options_owners_unchanged"] = True
            if any(p.device.type != "cpu" for p in native.model.parameters()): raise AssertionError("Non-CPU parameter")
            native.model.eval()
            with torch.no_grad(): initial = torch.stack([route.raw_logits for route in native._routes(False)])
            case["initial_function_max_abs_difference"] = float((initial-reference).abs().max())
            torch.testing.assert_close(initial, reference, atol=1e-6, rtol=1e-6)
            case["first_update"] = native.train_update()
            if case["first_update"]["work"]["reverse_collections"] != 2 or case["first_update"]["work"]["optimizer_transitions"] != 1:
                raise AssertionError("Native source F/W branch must collect twice and step once")
            valid = native.validate(); case["VALID_calls"] += 1
            if not torch.isfinite(valid["outputs"]).all() or valid["VALID_scope"] != "complete_source_VALID" or valid["VALID_nodes"] != int(native.data.val_mask.sum()):
                raise AssertionError("Nonfinite or incomplete full VALID")
            state_path = output/(method+"_after_one.pt")
            torch.save({"session": native.state(), "global_rng": rng(), "job": method_job}, state_path)
            case["checkpoint"] = pin(state_path)
            restored = source.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", case_source["data_root"], "cpu")
            constructions += 1; sessions.append(restored)
            restored = extension.transform(restored, method_job); observe(restored, case)
            ids = {name: id(p) for name, p in restored.source_parameter_references.items()}
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            bad = copy.deepcopy(saved["session"]); bad["method"]["topology"]["sha256"] = "changed"
            try:
                restored.load_state(bad)
            except ValueError:
                case["changed_topology_checkpoint_rejected"] = True
            else:
                raise AssertionError("Checkpoint accepted mismatched topology identity")
            bad = copy.deepcopy(saved["session"]); bad["native_state"]["responsibility"][0, 0] += 1
            try:
                restored.load_state(bad)
            except ValueError:
                case["changed_native_weight_checkpoint_rejected"] = True
            else:
                raise AssertionError("Checkpoint accepted changed native table")
            restored.load_state(saved["session"])
            if ids != {name: id(p) for name, p in restored.source_parameter_references.items()}:
                raise AssertionError("Native parameter identity changed")
            replay = restored.validate(); case["VALID_calls"] += 1
            case["restored_VALID_max_abs_difference"] = float((replay["outputs"]-valid["outputs"]).abs().max())
            torch.testing.assert_close(replay["outputs"], valid["outputs"], atol=1e-6, rtol=1e-6)
            restored.load_state(saved["session"]); restore_rng(saved["global_rng"])
            case["resumed_second_update"] = restored.train_update()
            final = restored.validate(); case["VALID_calls"] += 1
            if restored.updates != 2 or not torch.isfinite(final["outputs"]).all():
                raise AssertionError("Resumed update clock/whole VALID invalid")
            case.update(status="passed", native_update_clock=restored.updates,
                        restored_native_parameter_identity=True, method_metadata=restored.structural_class_responsibility_metadata,
                        activation_checkpointing=True, whole_native_TRAIN=True, whole_native_VALID=True,
                        full_fit_or_quality_evidence=False)
        except BaseException as error:
            case.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(limit=14)})
        finally:
            for session in sessions:
                try: session.core.remove_hooks()
                except Exception as error: case.setdefault("close_failures", []).append(str(error))
            write()
    report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"]) else "failed"
    report["limitations"] = ["CPU2threads exact CiteSeer-Cheb source descriptor only", "two updates, no convergence or accuracy claim",
        "no Physics/link/CUDA/pair-resource qualification", "no TEST access, selection, or scientific retry"]
    write()
    print(json.dumps({"status": report["status"], "actual_optimizer_transitions": total_steps,
        "cases": [{"method": c["method"], "status": c["status"], "steps": c["actual_optimizer_transitions"]} for c in report["cases"]]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
