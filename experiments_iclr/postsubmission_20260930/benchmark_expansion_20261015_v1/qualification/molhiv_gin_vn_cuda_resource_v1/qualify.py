"""One unchanged v4 MolHIV GIN-VN shared epoch, production finalization included.

Root admission is mandatory. This script performs no work when imported.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
import traceback


class WallLimit(BaseException):
    pass


def main():
    process_started = time.perf_counter()
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Exact allocation host, sole GPU UUID and visibility0 required before project reads")
    if len(sys.argv) != 3:
        raise ValueError("Usage: qualify.py EXACT_REPOSITORY_ROOT EXACT_FRESH_OUTPUT")
    here = Path(__file__).resolve().parent
    expansion = here.parent.parent
    root, output = (Path(x).resolve() for x in sys.argv[1:])
    inputs = json.loads((here / "INPUTS.json").read_text())
    if str(root) != inputs["repository_root"] or str(output) != inputs["output"] or output.exists() \
            or not all(p.is_relative_to(root) for p in (here, expansion, output)):
        raise ValueError("Literal repository and new own output required; no retry or resume")

    def receipt(path):
        path = Path(path)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Input escapes repository")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    admission = json.loads((here / "ROOT_ADMISSION.json").read_text())
    if admission.get("status") != "admitted" or admission.get("INPUTS_sha256") != receipt(here / "INPUTS.json")["sha256"] \
            or admission.get("qualifier_sha256") != receipt(__file__)["sha256"] \
            or admission.get("output") != str(output) or not admission.get("published_source_commit") \
            or admission.get("one_owned_GPU_slot") is not True or admission.get("automatic_retries") != 0:
        raise ValueError("Matching root source/slot admission required before native imports")
    for pin in inputs["file_pins"]:
        if receipt(pin["path"]) != pin:
            raise ValueError("Frozen input changed: " + pin["path"])
    for name in inputs["required_absent_paths"]:
        path = Path(name)
        if not path.resolve().is_relative_to(root) or path.exists():
            raise ValueError("Native role-loader precedence changed: " + name)
    runtime = expansion / "runtime_v4"
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(expansion / "qualification/runtime_v4_molecular_selected_mean_lifecycle_v1"))
    import common
    import adapters
    import run_fit
    import torch
    from resource_sampler import Sampler, preflight
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    job = json.loads((here / "JOB.json").read_text())
    run_fit.validate_job(job, root)
    preflight_receipt = preflight("cuda:0", output, inputs["resource_envelope"])
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = {"schema": "MolHIV-GIN-VN-one-epoch-current-v4-resource-v1", "status": "running",
        "engineering_only": True, "scientific_fits": 0, "scientific_convergence_inferred": False,
        "full_campaign_GO": False, "TEST_forward_or_scoring": False, "automatic_retries": 0,
        "resume_calls": 0, "interruption_or_floating_replay_program": False,
        "inputs": receipt(here / "INPUTS.json"), "root_admission": admission,
        "preflight": preflight_receipt, "job_hash": common.digest(job), "torch": str(torch.__version__),
        "pre_native_preparation_wall_seconds": started - process_started,
        "constructions_attempted": 0, "constructions_completed": 0, "epochs_started": 0,
        "epochs_completed": 0, "updates_started": 0, "updates_completed": 0, "exhaustion_calls": 0,
        "optimizer_attempts": 0, "optimizer_completed": 0, "backward_attempts": 0, "backward_completed": 0,
        "native_forward_attempts": {"TRAIN": 0, "VALID": 0}, "native_forward_completed": {"TRAIN": 0, "VALID": 0},
        "whole_VALID_started": 0, "whole_VALID_completed": 0, "selected_mean_exports": 0,
        "ordinary_finalization_adapter_loads": 0, "ordinary_finalization_RNG_restores": 0,
        "max_native_batch_nodes": {"TRAIN": 0, "VALID": 0}, "max_native_batch_edges": {"TRAIN": 0, "VALID": 0},
        "artifact_writes": [], "phase_wall_seconds": {}, "validation_receipts": []}
    sampler = Sampler(output, torch, "cuda:0")
    sampler.start()
    hooks, adapter = [], None

    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter() - started,
            process_entry_wall_seconds=time.perf_counter() - process_started,
            process_peak_RSS_KiB=usage.ru_maxrss, process_user_seconds=usage.ru_utime,
            process_system_seconds=usage.ru_stime, resource_observations=sampler.summary())
        common.atomic_json(output / "RESULT.json", report)

    def limited(*_):
        raise WallLimit("Frozen1800second resource gate watchdog; no retry")

    signal.signal(signal.SIGALRM, limited)
    signal.signal(signal.SIGTERM, limited)
    signal.alarm(inputs["bounds"]["wall_seconds"])
    original_create, original_backward = adapters.create_adapter, torch.Tensor.backward
    original_save, original_rng = run_fit.atomic_torch, run_fit.restore_rng

    def measured_backward(tensor, *args, **kwargs):
        if report["backward_attempts"] >= 1029:
            raise ValueError("1029native backward bound exceeded")
        report["backward_attempts"] += 1
        value = original_backward(tensor, *args, **kwargs)
        report["backward_completed"] += 1
        return value

    def measured_save(library, path, value):
        sampler.phase("atomic_checkpoint_or_export")
        begin = time.perf_counter()
        pin = original_save(library, path, value)
        report["artifact_writes"].append({"receipt": pin, "wall_seconds": time.perf_counter() - begin})
        sampler.sample()
        write()
        return pin

    def measured_rng(*args, **kwargs):
        report["ordinary_finalization_RNG_restores"] += 1
        if report["ordinary_finalization_RNG_restores"] > 1:
            raise ValueError("One ordinary selected finalization RNG restore only")
        return original_rng(*args, **kwargs)

    def measured_create(library, descriptor, policy, repository):
        nonlocal adapter
        if report["constructions_attempted"]:
            raise ValueError("One native construction only")
        report["constructions_attempted"] += 1
        sampler.phase("native_GIN_VN_construction")
        adapter = original_create(library, descriptor, policy, repository)
        report["constructions_completed"] += 1
        if adapter.session.config != inputs["native_session_config"] or len(adapter.session.train_ids) != 32901 \
                or len(adapter.session.valid_ids) != 4113 or adapter.session.recipe.get("smoke_valid_batch_limit") is not None:
            raise ValueError("Source-bound full MolHIV GIN-VN setting/roles required")
        step, update, train, valid, load, mean = (adapter.optimizers[0].step, adapter.session.train_update,
            adapter.train_epoch, adapter.validate, adapter.load_state, adapter.selected_mean_export)

        def forward_start(module, args):
            role = "TRAIN" if module.training else "VALID"
            bound = 4116 if role == "TRAIN" else 1032
            if report["native_forward_attempts"][role] >= bound:
                raise ValueError("Native route forward bound exceeded: " + role)
            report["native_forward_attempts"][role] += 1
            batch = args[0]
            report["max_native_batch_nodes"][role] = max(report["max_native_batch_nodes"][role], int(batch.x.shape[0]))
            report["max_native_batch_edges"][role] = max(report["max_native_batch_edges"][role], int(batch.edge_index.shape[1]))

        def forward_end(module, args, value):
            role = "TRAIN" if module.training else "VALID"
            report["native_forward_completed"][role] += 1

        hooks.extend([adapter.model.native.register_forward_pre_hook(forward_start),
                      adapter.model.native.register_forward_hook(forward_end)])

        def measured_step(*args, **kwargs):
            if report["optimizer_attempts"] >= 1029:
                raise ValueError("1029Adam transition bound exceeded")
            report["optimizer_attempts"] += 1
            value = step(*args, **kwargs)
            report["optimizer_completed"] += 1
            return value

        def measured_update():
            if report["updates_started"] >= 1030:
                raise ValueError("1029maps plus one epoch exhaustion sentinel only")
            report["updates_started"] += 1
            before = time.perf_counter()
            try:
                value = update()
            except adapters.EpochComplete:
                report["exhaustion_calls"] += 1
                raise
            if value["native_batch_graphs"] not in (32, 5) or value["optimizer_updates"] != 1 \
                    or value["native_route_forward_views"] != 4 or value["native_skipped_batches"] != 0 \
                    or value["work"] != {"reverse_collections": 1, "optimizer_transitions": 1, "extra_forward_views": 0}:
                raise ValueError("Exact native1029batch one-backward conditional interaction work required")
            report["updates_completed"] += 1
            common.event(output / "UPDATE_COSTS.jsonl", {"ordinal": report["updates_completed"],
                "wall_seconds": time.perf_counter() - before, "graphs": value["native_batch_graphs"],
                "work": value["work"], "loss": value["loss"], "component_gradients": value["component_gradients"]})
            if report["updates_completed"] % 32 == 0:
                write()
            return value

        def measured_train():
            if report["epochs_started"]:
                raise ValueError("One full native TRAIN epoch only")
            report["epochs_started"] += 1
            sampler.phase("whole_TRAIN_epoch1")
            before = time.perf_counter()
            value = train()
            report["phase_wall_seconds"]["whole_TRAIN_epoch"] = time.perf_counter() - before
            if value["TRAIN_records_traversed"] != 32901 or value["raw_batches"] != 1029 \
                    or value["optimizer_maps"] != 1029 or value["source_skipped_batches"] != 0:
                raise ValueError("Complete source TRAIN32901/1029maps required")
            report["epochs_completed"] += 1
            report["native_epoch_receipt"] = value
            sampler.sample(); write()
            return value

        def measured_valid():
            if report["whole_VALID_started"] >= 2:
                raise ValueError("One epoch VALID plus one production selected VALID only")
            report["whole_VALID_started"] += 1
            sampler.phase("whole_VALID_" + str(report["whole_VALID_started"]))
            before = time.perf_counter()
            value = valid()
            if len(value["ids"]) != 4113 or value["native_skipped_batches"] != 0:
                raise ValueError("Whole source VALID4113 required")
            report["whole_VALID_completed"] += 1
            report["validation_receipts"].append({"records": 4113, "role": "VALID",
                "wall_seconds": time.perf_counter() - before, "summary": run_fit.compact_validation(value)})
            sampler.sample(); write()
            return value

        def measured_load(value):
            report["ordinary_finalization_adapter_loads"] += 1
            if report["ordinary_finalization_adapter_loads"] > 1:
                raise ValueError("One ordinary selected finalization state load only")
            return load(value)

        def measured_mean(*args, **kwargs):
            if report["selected_mean_exports"]:
                raise ValueError("One authenticated same-call selected mean export only")
            value = mean(*args, **kwargs)
            if value is None:
                raise ValueError("Conditional interaction mean export missing")
            report["selected_mean_exports"] += 1
            return value

        adapter.optimizers[0].step = measured_step
        adapter.session.train_update, adapter.train_epoch, adapter.validate = measured_update, measured_train, measured_valid
        adapter.load_state, adapter.selected_mean_export = measured_load, measured_mean
        sampler.sample(); write()
        return adapter

    adapters.create_adapter, torch.Tensor.backward = measured_create, measured_backward
    run_fit.atomic_torch, run_fit.restore_rng = measured_save, measured_rng
    try:
        result = run_fit.run(job, root, output / "own_fit")
        report["production_result"] = result
        if result["status"] != "complete" or result["completed_epochs"] != 1 or result["selected_epoch"] != 1:
            raise ValueError("Unchanged production one-epoch selected finalization did not complete")
        expected = {"constructions_attempted": 1, "constructions_completed": 1, "epochs_started": 1,
            "epochs_completed": 1, "updates_started": 1030, "updates_completed": 1029, "exhaustion_calls": 1,
            "optimizer_attempts": 1029, "optimizer_completed": 1029, "backward_attempts": 1029,
            "backward_completed": 1029, "whole_VALID_started": 2, "whole_VALID_completed": 2,
            "selected_mean_exports": 1, "ordinary_finalization_adapter_loads": 1,
            "ordinary_finalization_RNG_restores": 1}
        if any(report[k] != value for k, value in expected.items()) \
                or report["native_forward_attempts"] != {"TRAIN": 4116, "VALID": 1032} \
                or report["native_forward_completed"] != report["native_forward_attempts"]:
            raise ValueError("Exact charged native work differs from frozen one-epoch plan")
        for pin in inputs["source_pins_after_run"]:
            if receipt(pin["path"]) != pin:
                raise ValueError("Frozen source changed during the gate")
        report.update(status="passed", scientific_metric_promotion=False,
            limitations=inputs["non_dominance_limits"])
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(limit=12)}, retry_authorized=False)
    finally:
        signal.alarm(0)
        adapters.create_adapter, torch.Tensor.backward = original_create, original_backward
        run_fit.atomic_torch, run_fit.restore_rng = original_save, original_rng
        for hook in hooks:
            hook.remove()
        sampler.stop()
        report["own_output_files"] = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*")) if p.is_file()]
        report["own_output_disk_bytes_final_observation"] = sum(r["bytes"] for r in report["own_output_files"])
        report["atomic_write_disk_peak_limit"] = "Final file sizes and sampled process data are observations; no continuous exact temporary-file disk high-water mark is inferred."
        write()
    print(json.dumps({key: report.get(key) for key in ("status", "elapsed_seconds", "updates_completed", "whole_VALID_completed", "failure")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
