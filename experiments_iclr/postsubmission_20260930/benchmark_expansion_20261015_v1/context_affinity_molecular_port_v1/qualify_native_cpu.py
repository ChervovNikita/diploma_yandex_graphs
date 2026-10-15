"""Bounded actual molecular alignment-only OGB bridge/checkpoint checks.

Engineering only: two CPU optimizer transitions per candidate on the same32
real TRAIN graphs. No benchmark fit, TEST data, selection or GPU execution.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
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
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--method-package", required=True)
    parser.add_argument("--qualifier-manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    job_path, output, here = Path(args.job).resolve(), Path(args.output).resolve(), Path(__file__).resolve().parent
    if not all(path.is_relative_to(root) for path in (job_path, output, here)) or output.exists():
        raise ValueError("Fresh repository-contained qualification output required")
    output.mkdir(parents=True, exist_ok=False)
    package=Path(args.method_package).resolve()
    if not package.is_relative_to(root):raise ValueError("Method source outside actual repository")
    qualifier_manifest=pin(here/"MANIFEST.json")
    if qualifier_manifest["sha256"]!=args.qualifier_manifest_sha256:raise ValueError("Unfrozen active-support qualifier")
    for row in json.loads((here/"MANIFEST.json").read_text())["files"]:
        actual=pin(here/row["path"])
        if any(actual[k]!=row[k] for k in ("bytes","sha256")):raise ValueError("Active-support qualifier source changed")
    job = json.loads(job_path.read_text())
    if job["family"] != "ogb" or job["procedure"] != "gnnm_boundary_native" \
            or job["recipe"]["dataset"] != "ogbg-molbace" or job["TEST_access"] is not False:
        raise ValueError("Use the actual frozen MolBACE GNNM TRAIN/VALID descriptor")
    callback_path = Path(job["callback_module"]).resolve()
    if not callback_path.is_relative_to(root) or pin(callback_path)["sha256"] != job["callback_sha256"]:
        raise ValueError("Actual frozen OGB callback differs")
    import torch
    from torch_geometric.data import Batch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():raise ValueError("CPU qualification requires hidden CUDA")
    torch.set_num_threads(2)
    native = load(callback_path, "_context_affinity_native_ogb_cpu")
    extension = load(package/"extension.py", "_context_affinity_cpu_extension")
    manifest = pin(package/"MANIFEST.json")
    if manifest["sha256"] != args.manifest_sha256:raise ValueError("Explicit frozen molecular control package required")
    extension.source_bindings(root,manifest["sha256"])
    report = {"schema": "context-affinity-native-molecular-active-CPU-v1", "status": "running",
        "job": pin(job_path), "callback": pin(callback_path), "package_manifest": manifest,
        "torch": str(torch.__version__), "device": "cpu", "threads": 2,
        "TEST_accessed": False, "scientific_fit": False, "benchmark_or_selection_claim": False,
        "GPU_models_or_updates": 0, "original_paper_reruns": 0, "cases": [],
        "qualifier_manifest":qualifier_manifest,"extra_auxiliary_gradient_collections":0}
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

    def instrument_auxiliary(session,case):
        core=extension.core_module()
        original=getattr(core.context_auxiliary,"_unwrapped_context_callable",core.context_auxiliary)
        shared=tuple(session.source_parameter_references.values());private=tuple(session.model.private_parameters())
        output_ids={id(p) for p in session.model.output_bank.parameters()}
        context=extension._load(package/"context.py","native_molecular_context_"+pin(package/"context.py")["sha256"])
        def observed(torch,**kwargs):
            value=original(torch,**kwargs)
            target=context.target_diagnostics(torch,kwargs["context_a"],kwargs["context_b"],kwargs["labels"],kwargs["object_ids"])
            if not kwargs["preheads"].requires_grad or value["diagnostics"]["eligible_anchors"]==0 or float(value["loss"].detach())<=0:
                raise ValueError("Supported live factual context loss required")
            if target["A_nonuniform_anchors_at_1e-8"]<1 or target["B_nonuniform_anchors_at_1e-8"]<1 or target["distinct_target_anchors_at_1e_8"]<1:
                raise ValueError("Both factual channels need nonuniform,distinct supported donor targets")
            gradients=torch.autograd.grad(value["loss"],shared+private,retain_graph=True,allow_unused=True)
            report["extra_auxiliary_gradient_collections"]+=1
            def norm(values):return sum(float(g.detach().square().sum()) for g in values if g is not None)**.5
            shared_norm=norm(gradients[:len(shared)]);private_norm=norm(gradients[len(shared):])
            output_norm=norm(g for p,g in zip(shared+private,gradients) if id(p) in output_ids)
            info={"native_update_before":session.updates,"shared_auxiliary_gradient_norm":shared_norm,
                "private_auxiliary_gradient_norm":private_norm,"output_bank_auxiliary_gradient_norm":output_norm,
                "weighted_context_loss":float(value["loss"].detach()),"target_support":target,
                "retained_graph_query_only":True,"optimizer_or_parameter_mutation":False}
            case["auxiliary_gradient_observations"].append(info)
            if not(shared_norm>0 and private_norm>0 and output_norm==0):raise AssertionError("Unexpected live context gradient support")
            return value
        observed._unwrapped_context_callable=original
        core.context_auxiliary=observed

    for method in extension.METHODS:
        case = {"method": method, "status": "running", "actual_optimizer_transitions": 0,
                "VALID_calls": 0, "TRAIN_graphs_per_update": 32, "same_real_TRAIN_batch_reused": True,
                "source_shuffled_batch_selection":[],"auxiliary_gradient_observations":[]}
        report["cases"].append(case); sessions = []
        try:
            method_job = copy.deepcopy(job)
            method_job.update(procedure=method, device="cpu", method_settings={"manifest_sha256": manifest["sha256"]})
            method_job["recipe"]["_repository_root"] = str(root)
            method_job["recipe"]["_seed"] = job["seed"]
            recipe = copy.deepcopy(method_job["recipe"])
            torch.manual_seed(job["seed"])
            source = native.create_session(copy.deepcopy(recipe), "gnnm_boundary_native", job["data_root"], "cpu")
            constructions += 1; sessions.append(source)
            if source.config["gnn"]!="gin":raise AssertionError("Exact actual MolBACE GIN descriptor required")
            batch=positions=None
            for scan in range((len(source.train_view)+31)//32):
                candidate,where,skipped=source._next_train_batch()
                observed=candidate.y.reshape(-1);observed=observed[~torch.isnan(observed)]
                counts={str(cls):int((observed==cls).sum()) for cls in (0,1)}
                case["source_shuffled_batch_selection"].append({"scan":scan,"TRAIN_positions":where.tolist(),
                    "TRAIN_ids":candidate.smoke_graph_id.reshape(-1).tolist(),"observed_class_counts":counts,"native_skipped_batches":skipped})
                if len(where)==32 and all(counts.values()):batch,positions=candidate,where;break
            if batch is None:raise ValueError("No both-class full32 source-shuffled batch in bounded one-epoch scan")
            case["chosen_batch_policy"]="first actual source-shuffled32 TRAIN batch with both observed classes; reused for2engineeringupdates, not a scientific trajectory"

            case["TRAIN_graph_ids"] = batch.smoke_graph_id.tolist()
            source.model.eval()
            with torch.no_grad():
                reference = torch.stack([source.model.run_member(m, batch)[0] for m in range(4)])
            group_options=[{k:copy.deepcopy(v) for k,v in g.items() if k!="params"} for g in source.optimizer.param_groups]
            group_owners=[[id(p) for p in g["params"]] for g in source.optimizer.param_groups]
            source = extension.transform(source, method_job)
            if group_options!=[{k:v for k,v in g.items() if k!="params"} for g in source.optimizer.param_groups] or group_owners!=[[id(p) for p in g["params"]] for g in source.optimizer.param_groups]:raise AssertionError("Native optimizer ownership/options changed")
            case["native_optimizer_groups_options_owners_unchanged"]=True
            instrument_auxiliary(source,case)
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
            if first["work"]["reverse_collections"] != 1:
                raise AssertionError("Native all-block alignment-only branch was not used")
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
            instrument_auxiliary(restored,case)
            observe_optimizer(restored, case)
            restored._next_train_batch = lambda: (batch, positions, 0)
            native_ids = {name: id(parameter) for name, parameter in restored.source_parameter_references.items()}
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            bad=copy.deepcopy(saved["session"]);bad["method"]["mode"]="changed"
            try:restored.load_state(bad)
            except ValueError:case["changed_context_mode_rejected"]=True
            else:raise AssertionError("Changed context mode restored")
            bad=copy.deepcopy(saved["session"]);bad["last_context_receipt"]["A_sha256"]="changed"
            try:restored.load_state(bad)
            except ValueError:case["changed_context_signature_rejected"]=True
            else:raise AssertionError("Changed factual context signature restored")
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
                        method_metadata=restored.context_affinity_metadata,
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
    report["limitations"] = ["CPU-only actual MolBACE engineering", "first actual source-shuffled both-class32 TRAIN graphs reused for2transitions, not a scientific trajectory", "extra retained-graph auxiliary autograd collections are charged separately",
        "no convergence, scientific score, CUDA/pairing/resource or node/link qualification", "no automatic scientific retry"]
    write()
    print(json.dumps({"status": report["status"], "cases": [{"method": case["method"], "status": case["status"],
        "optimizer_transitions": case["actual_optimizer_transitions"], "VALID_calls": case["VALID_calls"]} for case in report["cases"]],
        "actual_optimizer_transitions": total_steps, "native_constructions": constructions}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
