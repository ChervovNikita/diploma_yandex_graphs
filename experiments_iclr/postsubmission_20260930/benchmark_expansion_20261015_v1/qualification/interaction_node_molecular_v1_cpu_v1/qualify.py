"""Eight explicit interaction cases, two source updates each, fresh restore."""
from __future__ import annotations
import copy
import gc
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
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Allocation/CUDA-hidden guard before metadata")
    root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    here = Path(__file__).resolve().parent
    expansion = here.parent.parent
    package = expansion / "interaction_node_molecular_v1"
    if not all(p.is_relative_to(root) for p in (here, output, package)) or output.exists():
        raise ValueError("Fresh own repository output required")
    inputs = json.loads((here / "INPUTS.json").read_text())
    descriptors = json.loads((here / "SOURCE_DESCRIPTORS.json").read_text())

    def verify(row):
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(root) or pin(path) != row:
            raise ValueError("Pinned authentic input changed")

    for row in (inputs["native_smoke_manifest"], inputs["original_molecular_JOB"],
            *inputs["node_data_files"].values(), *inputs["ogb_data_files"].values()):
        verify(row)
    extension = load(package / "extension.py", "_qualified_explicit_interaction")
    binding = extension.sibling("bindings")
    binding.verify_package(package, inputs["source_manifest_sha256"])
    import torch
    import numpy as np
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"schema": "explicit-node-molecular-interaction-CPU-qualification-v1", "status": "running",
        "source_manifest_sha256": inputs["source_manifest_sha256"], "inputs": pin(here / "INPUTS.json"),
        "script": pin(__file__), "torch": str(torch.__version__), "device": "CPU", "threads": 2,
        "optimizer_attempts": 0, "optimizer_steps": 0, "native_constructions": 0,
        "complete_VALID_calls": 0, "same_call_mean_diagnostics_saved": 0, "cases": [],
        "scientific_fits": 0, "TEST_accessed": False, "GPU_worker_changes": 0, "automatic_retries": 0}

    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter() - started, process_user_seconds=usage.ru_utime,
            process_system_seconds=usage.ru_stime, process_peak_RSS_KiB=usage.ru_maxrss,
            CUDA_context_initialized=torch.cuda.is_initialized())
        (output / "RESULT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def seed(value):
        random.seed(value); np.random.seed(value); torch.random.default_generator.manual_seed(value)

    def rng():
        return {"python": random.getstate(), "numpy": copy.deepcopy(np.random.get_state()), "CPU": torch.get_rng_state().clone()}

    def restore(value):
        random.setstate(value["python"]); np.random.set_state(value["numpy"]); torch.set_rng_state(value["CPU"])

    def fingerprint(model):
        h = hashlib.sha256()
        for name, tensor in (*model.named_parameters(), *model.named_buffers()):
            h.update(name.encode()); h.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B"))
        return h.hexdigest()

    def construct(source, job, case, train_allowed):
        if report["native_constructions"] >= 16:
            raise ValueError("Fixed sixteen construction bound")
        seed(101)
        owner = source.create_session(copy.deepcopy(job["recipe"]), "gnnm_boundary_native", job["data_root"], "cpu")
        report["native_constructions"] += 1
        options = [{k: copy.deepcopy(v) for k, v in g.items() if k != "params"} for g in owner.optimizer.param_groups]
        native_refs = dict(owner.source_parameter_references)
        old_groups = tuple(owner.optimizer.param_groups)
        prior_rng = torch.get_rng_state().clone()
        owner = extension.transform(owner, copy.deepcopy(job))
        if not torch.equal(prior_rng, torch.get_rng_state()):
            raise ValueError("Component initialization consumed global Torch CPU RNG")
        if options != [{k: v for k, v in g.items() if k != "params"} for g in owner.optimizer.param_groups] \
                or any(g is not old for g, old in zip(owner.optimizer.param_groups, old_groups)):
            raise ValueError("Native optimizer groups/options changed")
        if any(dict(owner.model.named_parameters()).get(n) is not p for n, p in native_refs.items()):
            raise ValueError("Native source parameter identity changed")
        original_step = owner.optimizer.step

        def observed_step(*args, **kwargs):
            report["optimizer_attempts"] += 1; case["optimizer_attempts"] += 1
            if not train_allowed or case["optimizer_steps"] >= 2 or report["optimizer_steps"] >= 16:
                raise ValueError("No restore updates; maximum2source updates per case/16total")
            source_gradients = [p.grad for p in native_refs.values() if p.grad is not None]
            private_gradients = [p.grad for p in owner.model.private_parameters() if p.grad is not None]
            if not source_gradients or not private_gradients or not any(g.abs().sum() > 0 for g in source_gradients) \
                    or not any(g.abs().sum() > 0 for g in private_gradients):
                raise ValueError("Native shared and private gradient support required")
            if any(not torch.isfinite(g).all() for g in source_gradients + private_gradients):
                raise ValueError("Nonfinite real shared/private gradient")
            result = original_step(*args, **kwargs)
            report["optimizer_steps"] += 1; case["optimizer_steps"] += 1
            return result

        owner.optimizer.step = observed_step
        return owner

    def complete_valid(owner, family):
        if report["complete_VALID_calls"] >= 16:
            raise ValueError("Fixed sixteen complete VALID bound")
        before = fingerprint(owner.model)
        value = owner.validate()
        if fingerprint(owner.model) != before:
            raise ValueError("VALID mutated model parameters/buffers")
        n = value["VALID_nodes"] if family == "node" else value["VALID_graphs"]
        if n != (665 if family == "node" else 151) or value["VALID_scope"] != "complete_source_VALID":
            raise ValueError("Authentic complete native VALID665/151 required")
        report["complete_VALID_calls"] += 1
        report["same_call_mean_diagnostics_saved"] += 1
        return extension.selected_mean_sidecar(owner)

    for family in ("node", "ogb"):
        descriptor = descriptors["node_case"] if family == "node" else descriptors["molecular_job"]
        source = load(Path(descriptor["callback_module"]), "_interaction_actual_native_" + family)
        for index, method in enumerate(extension.METHODS):
            case = {"family": family, "method": method, "status": "running", "optimizer_steps": 0,
                "optimizer_attempts": 0, "TRAIN_updates": [], "fresh_restore_steps": 0}
            report["cases"].append(case)
            owners = []
            try:
                recipe = copy.deepcopy(descriptor["recipe"])
                recipe.update(_seed=101, seed=101, _repository_root=str(root))
                job = {"schema": "native-full-epoch-fit-v1", "family": family, "id": family + "/" + method + "/engineering101",
                    "cohort_id": "explicit_interaction_CPU_v1/" + family + "/seed101", "procedure": method,
                    "base_procedure": "gnnm_boundary_native", "recipe": recipe, "seed": 101,
                    "callback_module": descriptor["callback_module"], "data_root": descriptor["data_root"],
                    "policy": {"track": "matched_pilot"}, "device": "cpu", "execution_mode": "cpu_engineering_v1",
                    "engineering_only": True, "TEST_access": False,
                    "method_settings": {"manifest_sha256": inputs["source_manifest_sha256"]}}
                owner = construct(source, job, case, True); owners.append(owner)
                try:
                    extension.selected_mean_sidecar(owner)
                except ValueError:
                    pass
                else:
                    raise ValueError("Getter fabricated uncached VALID outputs")
                for step in range(2):
                    value = owner.train_update()
                    case["TRAIN_updates"].append(value)
                    if value["work"]["optimizer_transitions"] != 1 or value["work"]["reverse_collections"] != (
                            2 if method == "interaction_responsibility_private_pool" else 1):
                        raise ValueError("Declared gradient collection/one-transition recipe changed")
                    if family == "node" and (not value["whole_native_train"] or value["train_records"] != 1929):
                        raise ValueError("Complete authentic native TRAIN1929 required")
                    if family == "ogb" and (value["native_batch_graphs"] != 32 or any(
                            count == 0 for count in value["source_batch_observed_class_counts"].values())):
                        raise ValueError("First two original source32-graph batches must have authentic mixed-class support; no batch/seed search")
                if method != "interaction_responsibility_private_pool":
                    gradient_rows = case["TRAIN_updates"][-1]["component_gradients"]["input_residual"]
                    if len(gradient_rows) != 2 or any(not g["grad_present"] or g["gradient_L2"] <= 0 for g in gradient_rows):
                        raise ValueError("After two real updates, both residual down/up gradients must be live")
                if method == "interaction_input_conditionalJ":
                    rows = case["TRAIN_updates"][-1]["component_gradients"]["readout"]
                    if len(rows) != 4 or any(not g["grad_present"] or g["gradient_L2"] <= 0 for g in rows):
                        raise ValueError("After two actual updates, all conditional scorer parameter blocks must have live gradients")
                actual = complete_valid(owner, family)
                state = owner.state(); checkpoint = output / (family + "_" + str(index) + "_after_two.pt")
                torch.save({"session": state, "job": job, "global_rng": rng()}, checkpoint)
                restored = construct(source, job, case, False); owners.append(restored)
                bad = copy.deepcopy(state); bad["method"]["ingredient_bindings"]["helper_files"][0]["sha256"] = "changed"
                try:
                    restored.load_state(bad)
                except ValueError:
                    case["changed_ingredient_rejected"] = True
                else:
                    raise ValueError("Restore accepted changed ingredient binding")
                if state["fixed_weights"] is not None:
                    bad = copy.deepcopy(state); bad["fixed_weights"][0, 0] += .1
                    try:
                        restored.load_state(bad)
                    except ValueError:
                        case["changed_fixed_table_rejected"] = True
                    else:
                        raise ValueError("Restore accepted modified responsibility weight")
                restored.load_state(copy.deepcopy(state))
                try:
                    extension.selected_mean_sidecar(restored)
                except ValueError:
                    case["restore_requires_fresh_VALID_sidecar"] = True
                else:
                    raise ValueError("Restore fabricated selected/VALID sidecar")
                restore(torch.load(checkpoint, map_location="cpu", weights_only=False)["global_rng"])
                fresh = complete_valid(restored, family)
                if not torch.equal(actual["ids"], fresh["ids"]) or not torch.equal(actual["labels"], fresh["labels"]):
                    raise ValueError("Fresh restore VALID IDs/order/labels differ")
                for key in ("actual_served_raw_logits", "same_bank_mean_raw_logits"):
                    torch.testing.assert_close(actual[key], fresh[key], atol=1e-4, rtol=1e-4)
                if actual["actual_metrics"] != fresh["actual_metrics"] or actual["same_bank_mean_metrics"] != fresh["same_bank_mean_metrics"]:
                    raise ValueError("Fresh restored VALID metrics differ")
                artifact = output / (family + "_" + str(index) + "_VALID.pt")
                torch.save({"schema": "existing-engineering-interaction-VALID-pair-v1", "first": actual, "restored": fresh,
                    "selected_run_fit_export_qualified": False, "TEST_accessed": False}, artifact)
                case.update(status="passed", checkpoint=pin(checkpoint), VALID_artifact=pin(artifact),
                    complete_VALID_records=len(actual["ids"]), actual_restore_max_abs_difference=float((
                        actual["actual_served_raw_logits"] - fresh["actual_served_raw_logits"]).abs().max()),
                    mean_restore_max_abs_difference=float((actual["same_bank_mean_raw_logits"] - fresh["same_bank_mean_raw_logits"]).abs().max()),
                    mean_diagnostic_no_extra_forward=True, native_optimizer_groups_options_preserved=True,
                    complete_parameter_ownership=True, selected_run_fit_export_qualified=False)
            except BaseException as error:
                case.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                    "traceback": traceback.format_exc(limit=16)})
            finally:
                for owner in owners:
                    try:
                        owner.core.remove_hooks() if family == "node" else owner.close()
                    except Exception:
                        pass
                owners.clear(); gc.collect(); write()
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) and (
        report["optimizer_steps"], report["optimizer_attempts"], report["native_constructions"], report["complete_VALID_calls"]) == (16, 16, 16, 16) else "failed"
    report["limitations"] = ["two source updates per recipe/task only; no convergence or scientific accuracy claim",
        "exact CiteSeer-Cheb and MolBACE GIN/dropout.5 CPU cells only, not GPU or other datasets",
        "direct native serving/fresh-restore outputs qualified; actual run_fit selected export/selector not exercised",
        "future production finalizer must bind/persist same-call mean sidecar after own selected-state validation; frozen v3 drops extra fields",
        "no TEST, new scientific fit, legacy checkpoint alias or frozen parent modification"]
    binding.verify_package(package, inputs["source_manifest_sha256"])
    write()
    print(json.dumps({"status": report["status"], "optimizer_steps": report["optimizer_steps"],
        "native_constructions": report["native_constructions"], "complete_VALID_calls": report["complete_VALID_calls"],
        "cases": [{"family": c["family"], "method": c["method"], "status": c["status"], "failure": c.get("failure", {}).get("message")} for c in report["cases"]]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
