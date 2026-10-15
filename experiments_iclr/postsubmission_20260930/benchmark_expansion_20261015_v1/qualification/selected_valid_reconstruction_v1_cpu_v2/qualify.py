"""Bounded existing-state CPU VALID reconstruction; zero optimizer steps."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import traceback

REPLAYER_MANIFEST = "e397d3f1738e763daf8ecfe614833ce64464b974f978f23a67c4eb874c7af27c"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    # Guard before reading project metadata. Importing backend uses stdlib only.
    import os
    import socket
    import subprocess
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Wrong allocation or CUDA not hidden")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.project_root).resolve(), Path(args.output).resolve()
    here = Path(__file__).resolve().parent
    expansion = here.parent.parent
    package = expansion / "selected_valid_reconstruction_v1"
    if not all(path.is_relative_to(root) for path in (here, package, output)) or output.exists():
        raise ValueError("Fresh own repository output required")
    output.mkdir(parents=True, exist_ok=False)
    backend = load(package / "replay.py", "_qualified_selected_replay")
    backend.admit_cpu()
    backend.verify_package(package, REPLAYER_MANIFEST, root)
    inputs_path = here / "INPUTS.json"
    inputs = json.loads(inputs_path.read_text())
    if len(inputs["production"]) != 4 or len(inputs["engineering"]) != 14:
        raise ValueError("Fixed four selected and14 existing engineering cases only")
    registry = json.loads((package / "REGISTRY.json").read_text())
    for row in registry["native_sources"]:
        path = backend.inside(backend.PHASE / row["path_relative_to_phase"], root)
        if backend.receipt(path)["sha256"] != row["sha256"]:
            raise ValueError("Native source changed")
    smoke_path = backend.verified(inputs["native_smoke_manifest"], root)
    smoke = json.loads(smoke_path.read_text())
    node_case = next(case for case in smoke["cases"] if case["id"] == inputs["node_source_case"])
    import torch
    torch.set_num_threads(2)
    report = {"schema": "selected-VALID-reconstruction-CPU-qualification-v1", "status": "running",
        "script": backend.receipt(__file__), "inputs": backend.receipt(inputs_path),
        "replayer_manifest_sha256": REPLAYER_MANIFEST, "torch": str(torch.__version__), "device": "CPU",
        "threads": 2, "CUDA_VISIBLE_DEVICES": "", "optimizer_steps": 0, "optimizer_attempts": 0,
        "scientific_fits": 0, "TEST_accessed": False, "GPU_worker_changes": 0,
        "production_selected_cases": [], "engineering_existing_state_cases": [], "native_constructions": 0,
        "complete_VALID_calls": 0, "C6_mean_diagnostics_saved": 0}
    started = time.perf_counter()

    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter() - started, process_user_seconds=usage.ru_utime,
                      process_system_seconds=usage.ru_stime, process_peak_RSS_KiB=usage.ru_maxrss)
        (output / "RESULT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def blocked(*args, **kwargs):
        report["optimizer_attempts"] += 1
        raise ValueError("Zero optimizer transitions in reconstruction qualification")

    def fingerprint(model):
        hasher = hashlib.sha256()
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            hasher.update(name.encode())
            hasher.update(memoryview(value.detach().contiguous().cpu().numpy()).cast("B"))
        return hasher.hexdigest()

    for index, row in enumerate(inputs["production"]):
        case = {"id": index, "status": "running", "authority": "actual complete run_fit selected checkpoint and saved selected_VALID export"}
        report["production_selected_cases"].append(case)
        try:
            backend.verified(row["origin_result"], root)
            artifact = backend.replay_selected(row["request"], root)
            report["native_constructions"] += 1
            report["complete_VALID_calls"] += 1
            path = output / ("selected_" + str(index) + "_VALID.pt")
            torch.save(artifact, path)
            case.update(status="passed", procedure=artifact["procedure"], runtime_profile=artifact["runtime_profile"],
                        parity=artifact["parity"], artifact=backend.receipt(path), original_inputs=row["request"],
                        optimizer_steps=artifact["optimizer_steps"])
        except BaseException as error:
            case.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc(limit=12)})
        finally:
            gc.collect()
            write()

    profile = next(row for row in registry["runtimes"] if row["id"] == "runtime_v2")
    for index, row in enumerate(inputs["engineering"]):
        case = {"method": row["method"], "family": row["family"], "status": "running",
            "authority": "existing engineering checkpoint; fresh-fresh replay only, not selected-export qualification"}
        report["engineering_existing_state_cases"].append(case)
        owners = []
        try:
            backend.verified(row["origin_result"], root)
            path = backend.verified(row["checkpoint"], root)
            callback = backend.verified(row["callback"], root)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            job = copy.deepcopy(payload["job"])
            if job["procedure"] != row["method"] or job["family"] != row["family"]:
                raise ValueError("Engineering source JOB differs")
            matches = [m for m in registry["methods"] if m["family"] == row["family"] and row["method"] in m["procedures"]]
            if len(matches) != 1:
                raise ValueError("Unknown existing engineering callback method")
            method = matches[0]
            if callback != backend.PHASE / method["callback_relative_to_phase"] or row["callback"]["sha256"] != method["callback_sha256"]:
                raise ValueError("Engineering callback/source registry differs")
            for package_row in method["packages"]:
                backend.verify_package(backend.PHASE / package_row["path_relative_to_phase"], package_row["sha256"], root)
            if job.get("method_settings", {}).get("manifest_sha256") != method["method_manifest_sha256"]:
                raise ValueError("Engineering method generation differs")
            job.setdefault("base_procedure", "gnnm_boundary_native")
            job.setdefault("data_root", node_case["data_root"] if row["family"] == "node" else "")
            declarations = inputs["node_data_files"] if row["family"] == "node" else inputs["ogb_data_files"]
            backend.data_preflight(job, declarations, root)
            results, diagnostics = [], []
            with backend.runtime_namespace(profile, root) as runtime:
                source, _ = runtime["common"].load_module(callback, root, method["callback_sha256"])
                extension, _ = runtime["common"].load_module(backend.PHASE / method["extension_relative_to_phase"], root, method["extension_sha256"])
                for replay_index in range(2):
                    if report["native_constructions"] >= 32:
                        raise ValueError("Fixed qualification construction bound reached")
                    runtime["common"].seed_all(torch, job["seed"])
                    owner = source.create_session(copy.deepcopy(job["recipe"]), "gnnm_boundary_native", job["data_root"], "cpu")
                    report["native_constructions"] += 1
                    owners.append(owner)
                    for optimizer in owner.optimizers:
                        optimizer.step = blocked
                    owner = extension.transform(owner, copy.deepcopy(job))
                    owner.load_state(copy.deepcopy(payload["session"]))  # preserve complete wrapper, never strip
                    owner.train_update = blocked
                    adapter_type = runtime["adapters"].NodeAdapter if row["family"] == "node" else runtime["adapters"].OGBAdapter
                    view = adapter_type.__new__(adapter_type)
                    view.torch, view.session, view.model, view.optimizers = torch, owner, owner.model, owner.optimizers
                    before = fingerprint(owner.model)
                    value, mean = backend.actual_validation(torch, view, method["C6_mean_diagnostic"])
                    report["complete_VALID_calls"] += 1
                    if before != fingerprint(owner.model):
                        raise ValueError("VALID mutated a model parameter/buffer")
                    results.append({**value, "ids": value["ids"].cpu(), "labels": value["labels"].cpu(), "outputs": value["outputs"].cpu()})
                    diagnostics.append(mean)
                    if mean is not None:
                        report["C6_mean_diagnostics_saved"] += 1
                    if row["family"] == "node":
                        owner.core.remove_hooks()
                    else:
                        owner.close()
            first, second = results
            reference = {"ids": first["ids"], "labels": first["labels"], "raw_logits": first["outputs"],
                "validation": {"role": "VALID", "complete_source_VALID": True, "TEST_accessed": False,
                    "records": len(first["ids"]), "loss": first["loss"], "metrics": first["metrics"]}}
            parity = backend.compare_export(torch, second, reference)
            if method["C6_mean_diagnostic"]:
                torch.testing.assert_close(diagnostics[0]["same_bank_mean_raw_logits"], diagnostics[1]["same_bank_mean_raw_logits"],
                                           atol=backend.LOGIT_ATOL, rtol=backend.LOGIT_RTOL)
            artifact = {"schema": "engineering-existing-state-VALID-reconstruction-v1", "role": "VALID", "TEST_accessed": False,
                "source_checkpoint": row["checkpoint"], "method": row["method"], "family": row["family"],
                "first": first, "second": second, "C6_diagnostics": diagnostics, "fresh_fresh_parity": parity,
                "selected_export_parity_exercised": False, "scientific_selection_claim": False, "optimizer_steps": 0}
            artifact_path = output / ("engineering_" + str(index) + "_VALID.pt")
            torch.save(artifact, artifact_path)
            case.update(status="passed", fresh_fresh_parity=parity, artifact=backend.receipt(artifact_path),
                source_checkpoint=row["checkpoint"], C6_same_bank_mean_preserved=method["C6_mean_diagnostic"],
                model_unchanged_by_VALID=True, selected_export_parity_exercised=False, optimizer_steps=0)
        except BaseException as error:
            case.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc(limit=14)})
        finally:
            for owner in owners:
                try:
                    owner.core.remove_hooks() if row["family"] == "node" else owner.close()
                except Exception:
                    pass
            owners.clear()
            gc.collect()
            write()
    all_cases = report["production_selected_cases"] + report["engineering_existing_state_cases"]
    report["status"] = "passed" if all(c["status"] == "passed" for c in all_cases) and report["optimizer_attempts"] == 0 else "failed"
    report["limitations"] = ["four actual selected/export parity cases are existing MolBACE fits, not new scientific fits",
        "fourteen existing engineering states exercise source reconstruction/VALID and fresh-fresh parity, not a saved selected export",
        "BNv3 source mapping prepared; no persisted existing BNv3 selected/engineering state qualified here",
        "C1node sources published after registry freeze are not silently registered", "no GPU execution, TEST, optimizer step or selector update"]
    write()
    print(json.dumps({"status": report["status"], "production_passed": sum(c["status"] == "passed" for c in report["production_selected_cases"]),
        "engineering_passed": sum(c["status"] == "passed" for c in report["engineering_existing_state_cases"]),
        "native_constructions": report["native_constructions"], "complete_VALID_calls": report["complete_VALID_calls"],
        "C6_mean_diagnostics_saved": report["C6_mean_diagnostics_saved"], "optimizer_steps": 0,
        "failures": [{"method": c.get("method", c.get("id")), "message": c.get("failure", {}).get("message")} for c in all_cases if c["status"] != "passed"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
