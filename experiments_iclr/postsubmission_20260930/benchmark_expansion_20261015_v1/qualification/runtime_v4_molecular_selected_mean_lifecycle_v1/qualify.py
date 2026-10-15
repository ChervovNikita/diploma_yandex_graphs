"""Two total native molecular epochs; actual v4 mean export and resume lifecycle."""
from __future__ import annotations
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import math


class ControlledCompletedEpochStop(BaseException):
    pass


class EngineeringWallLimit(BaseException):
    pass


def guard(device):
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"] or os.environ.get("CUDA_VISIBLE_DEVICES") != ("" if device == "cpu" else "0"):
        raise ValueError("Literal allocation/UUID and exact CPU or GPU visibility required before project reads")


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in ("cpu", "cuda:0"):
        raise ValueError("Usage: qualify.py PROJECT_ROOT EXACT_OUTPUT cpu|cuda:0; choose one alternative only")
    device = sys.argv[3]
    guard(device)
    here = Path(__file__).resolve().parent
    expansion = here.parent.parent
    root, output = (Path(p).resolve() for p in sys.argv[1:3])
    runtime = expansion / "runtime_v4"
    if not all(p.is_relative_to(root) for p in (here, runtime, output)) or output.exists():
        raise ValueError("Fresh own repository output only")
    inputs = json.loads((here / "INPUTS.json").read_text())
    if str(output) != inputs["exclusive_output"] or str(root) != inputs["repository_root"]:
        raise ValueError("One literal output and repository binding shared by both alternatives required")
    sys.path.insert(0, str(runtime))
    import common
    import adapters
    import run_fit
    import torch
    import numpy as np
    from resource_sampler import Sampler, preflight
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    def wall_limit(*_):
        raise EngineeringWallLimit("Declared 600-second native engineering wall watchdog")

    signal.signal(signal.SIGTERM, wall_limit)
    signal.signal(signal.SIGALRM, wall_limit)
    signal.alarm(600)

    def verify(pin):
        if common.receipt(common.inside(pin["path"], root)) != pin:
            raise ValueError("Pinned input bytes changed: " + str(pin["path"]))

    def package(folder, expected):
        manifest = folder / "MANIFEST.json"
        if common.receipt(manifest)["sha256"] != expected:
            raise ValueError("Frozen package manifest changed: " + str(folder))
        rows = [common.receipt(manifest)]
        for pin in json.loads(manifest.read_text())["files"]:
            path = common.inside(folder / pin["path"], root)
            if not path.is_relative_to(folder):
                raise ValueError("Manifest member escapes package")
            now = common.receipt(path)
            if any(now[key] != pin[key] for key in ("bytes", "sha256")):
                raise ValueError("Frozen package source changed: " + str(path))
            rows.append(now)
        return rows

    def source_check():
        rows = package(runtime, inputs["runtime_manifest_sha256"])
        rows += package(expansion / "interaction_node_molecular_v1", inputs["interaction_manifest_sha256"])
        parent = json.loads((runtime / "PARENT_SOURCE.json").read_text())
        parent_dir = expansion / parent["directory"]
        if common.receipt(parent_dir / "MANIFEST.json")["sha256"] != parent["manifest_sha256"]:
            raise ValueError("Frozen parent runtime changed")
        for pin in parent["files"]:
            now = common.receipt(parent_dir / pin["path"])
            if any(now[key] != pin[key] for key in ("bytes", "sha256")):
                raise ValueError("Frozen parent runtime producer changed")
        for pin in (inputs["native_smoke_manifest"], *inputs["data_files"].values()):
            verify(pin)
        for pin in inputs["qualifier_sources"]:
            verify(pin)
        for item in json.loads((expansion / "interaction_node_molecular_v1/BINDINGS.json").read_text())["packages"]:
            rows += package(expansion / item["path"], item["manifest_sha256"])
        for pin in inputs["native_dependencies"]:
            verify(pin)
        return rows

    before_sources = source_check()
    job_path = here / ("JOB_CPU.json" if device == "cpu" else "JOB_CUDA.json")
    if common.receipt(job_path)["sha256"] != inputs["job_sha256"][device]:
        raise ValueError("Frozen engineering descriptor changed")
    job = json.loads(job_path.read_text())
    if job["policy"]["maximum_epochs"] != 2 or job["policy"]["minimum_epochs"] != 2 \
            or job["seed"] != 101 or job["split"] != 0 or job["procedure"] != "interaction_input_conditionalJ" \
            or job["device"] != device or job["family"] != "ogb" or job["engineering_only"] is not True:
        raise ValueError("Exact admitted finite engineering descriptor required")
    run_fit.validate_job(job, root)
    admission = preflight(device, output, inputs["resource_envelopes"][device])
    choice_path = common.inside(inputs["exclusive_choice_claim"], root)
    choice_fd = os.open(choice_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(choice_fd, "w") as handle:
        json.dump({"device": device, "job_sha256": common.receipt(job_path)["sha256"], "output": str(output),
                   "pid": os.getpid(), "UTC": common.utc(), "native_retry_authorized": False}, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sampler = Sampler(output, torch, device)
    sampler.start()
    report = {"schema": "runtime-v4-native-molecular-selected-mean-lifecycle-v1", "status": "running",
        "runtime_manifest_sha256": inputs["runtime_manifest_sha256"],
        "interaction_manifest_sha256": inputs["interaction_manifest_sha256"],
        "script": common.receipt(__file__), "inputs": common.receipt(here / "INPUTS.json"),
        "job_hash": common.digest(job), "torch": str(torch.__version__), "device": device, "threads": 2,
        "torch_interop_threads": 1, "native_constructions_attempted": 0, "native_constructions_completed": 0,
        "TRAIN_epochs_started": 0, "complete_native_TRAIN_epochs": 0, "source_optimizer_updates_observed": 0,
        "adapter_VALID_calls_started": 0, "complete_adapter_VALID_calls": 0,
        "native_VALID_calls_started": 0, "complete_native_VALID_calls": 0,
        "same_call_mean_diagnostics": 0, "selected_mean_exports": 0, "TRAIN_diagnostic_calls": 0,
        "scientific_fits": 0, "TEST_accessed": False, "other_GPU_workers_modified": 0,
        "own_GPU_worker": device != "cpu", "automatic_retries": 0,
        "invocations": [], "train_epoch_receipts": [], "VALID_receipts": [],
        "emitted_mean_receipts": [], "snapshots": [], "checks": []}
    report.update(exclusive_choice_claim=common.receipt(choice_path), preflight=admission,
        engineering_only=True, alternative_device_not_executed="cuda:0" if device == "cpu" else "cpu",
        optimizer_step_attempts_observed=0, optimizer_steps_completed_observed=0,
        native_update_calls_started=0, native_update_calls_completed=0, epoch_exhaustion_calls=0,
        native_backward_calls=0, exact_adapter_state_restores=0, exact_global_RNG_restores=0,
        update_receipts=[], float_reexecution_diagnostics=[], native_TRAIN_graphs=1210, native_VALID_graphs=151)

    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter() - started, process_user_seconds=usage.ru_utime,
            process_system_seconds=usage.ru_stime, process_peak_RSS_KiB=usage.ru_maxrss,
            CUDA_context_initialized=torch.cuda.is_initialized(), resource_observations=sampler.summary())
        common.atomic_json(output / "RESULT.json", report)

    def equal(a, b, name):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                raise ValueError("Exact tensor differs: " + name)
        elif isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                raise ValueError("Exact NumPy state differs: " + name)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                raise ValueError("Exact mapping differs: " + name)
            for key in a:
                equal(a[key], b[key], name + "/" + str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError("Exact sequence differs: " + name)
            for index, value in enumerate(a):
                equal(value, b[index], name + "/" + str(index))
        elif a != b:
            raise ValueError("Exact scalar differs: " + name)

    def repeated_float(a, b, name):
        if not torch.is_tensor(a) or not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype \
                or not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
            raise ValueError("Reexecuted native output tensor/shape/dtype/nonfinite failure: " + name)
        delta = a.double() - b.double()
        row = {"name": name, "bit_equal": torch.equal(a, b), "maximum_absolute_difference": float(delta.abs().max()),
               "RMS_difference": float(delta.square().mean().sqrt()), "rtol": 1e-4, "atol": 1e-5}
        row["passed"] = row["bit_equal"] if device == "cpu" else bool(torch.allclose(a, b, rtol=1e-4, atol=1e-5))
        report["float_reexecution_diagnostics"].append(row)
        if not row["passed"]:
            raise ValueError("Reexecuted output outside the fixed chosen-device policy: " + name)

    def repeated_validation(a, b, name):
        if device == "cpu":
            equal(a, b, name)
            return
        for key in ("role", "complete_source_VALID", "records", "TEST_accessed", "metrics"):
            equal(a[key], b[key], name + "/" + key)
        row = {"name": name + "/loss", "absolute_difference": abs(a["loss"] - b["loss"]),
               "rtol": 1e-4, "atol": 1e-5, "passed": math.isclose(a["loss"], b["loss"], rel_tol=1e-4, abs_tol=1e-5)}
        report["float_reexecution_diagnostics"].append(row)
        if not row["passed"]:
            raise ValueError("Reexecuted native loss outside fixed CUDA diagnostics")

    original_backward = torch.Tensor.backward
    original_restore_rng = run_fit.restore_rng

    def measured_backward(tensor, *args, **kwargs):
        if report["native_backward_calls"] >= 76:
            raise ValueError("Fixed 76 native backward bound")
        report["native_backward_calls"] += 1
        return original_backward(tensor, *args, **kwargs)

    def measured_restore_rng(library, value, **kwargs):
        if report["exact_global_RNG_restores"] >= 4:
            raise ValueError("Fixed four global RNG restore bound")
        original_restore_rng(library, value, **kwargs)
        equal(common.cpu_clone(torch, common.global_rng(torch, device=device)), value, "restored_global_RNG")
        report["exact_global_RNG_restores"] += 1

    torch.Tensor.backward = measured_backward
    run_fit.restore_rng = measured_restore_rng

    original_create = adapters.create_adapter

    def measured_create(tensor_library, descriptor, policy, repository):
        if report["native_constructions_attempted"] >= 3:
            raise ValueError("Fixed three native construction bound")
        report["native_constructions_attempted"] += 1
        sampler.phase("native_construction_" + str(report["native_constructions_attempted"]))
        write()
        owner = original_create(tensor_library, descriptor, policy, repository)
        report["native_constructions_completed"] += 1
        sampler.sample()
        if owner.recipe["_repository_root"] != str(root) or owner.session.recipe["_repository_root"] != str(root):
            raise ValueError("Actual repository root was not bound")
        if owner.session.config != inputs["native_session_config"] or len(owner.session.train_ids) != 1210 \
                or len(owner.session.valid_ids) != 151 or owner.session.recipe.get("smoke_valid_batch_limit") is not None:
            raise ValueError("Frozen full native MolBACE GIN configuration and role populations required")
        original_step, original_update, original_load = owner.optimizers[0].step, owner.session.train_update, owner.load_state

        def step(*args, **kwargs):
            if report["optimizer_step_attempts_observed"] >= 76:
                raise ValueError("Fixed 76 native Adam attempt bound")
            report["optimizer_step_attempts_observed"] += 1
            value = original_step(*args, **kwargs)
            report["optimizer_steps_completed_observed"] += 1
            return value

        def update():
            if report["native_update_calls_started"] >= 78:
                raise ValueError("Fixed 78 update calls including two epoch exhaustion sentinels")
            report["native_update_calls_started"] += 1
            try:
                value = original_update()
            except adapters.EpochComplete:
                report["epoch_exhaustion_calls"] += 1
                raise
            if value["native_batch_graphs"] not in (32, 26) or value["optimizer_updates"] != 1 \
                    or value["native_route_forward_views"] != 4 or value["work"]["reverse_collections"] != 1 \
                    or value["native_skipped_batches"] != 0 or value["work"]["extra_forward_views"] != 0:
                raise ValueError("Actual original native batch/one transition/one backward source work required")
            report["native_update_calls_completed"] += 1
            report["update_receipts"].append({"ordinal": report["native_update_calls_completed"],
                "graphs": value["native_batch_graphs"], "loss": value["loss"],
                "source_batch_observed_class_counts": value["source_batch_observed_class_counts"],
                "component_gradients": value["component_gradients"], "work": value["work"]})
            write()
            return value

        def load(value):
            if report["exact_adapter_state_restores"] >= 4:
                raise ValueError("Fixed four adapter-state restore bound")
            expected = common.cpu_clone(torch, value)
            original_load(value)
            equal(common.cpu_clone(torch, owner.state()), expected, "fresh_loaded_native_Adam_loader_wrapper_state")
            report["exact_adapter_state_restores"] += 1

        owner.optimizers[0].step, owner.session.train_update, owner.load_state = step, update, load
        original_train, original_valid = owner.train_epoch, owner.validate
        native_validate, original_export = owner.session.validate, owner.selected_mean_export

        def train():
            if report["TRAIN_epochs_started"] >= 2:
                raise ValueError("Fixed two total native TRAIN epoch bound")
            report["TRAIN_epochs_started"] += 1
            sampler.phase("whole_TRAIN_epoch_" + str(report["TRAIN_epochs_started"]))
            before = owner.session.updates
            try:
                value = original_train()
            finally:
                report["source_optimizer_updates_observed"] += owner.session.updates - before
                write()
            if value["TRAIN_records_traversed"] != 1210 or value["raw_batches"] != 38 \
                    or value["optimizer_maps"] != 38 or value["source_skipped_batches"] != 0 \
                    or value["complete_native_TRAIN"] is not True:
                raise ValueError("Complete native TRAIN1210 /38original updates required")
            report["complete_native_TRAIN_epochs"] += 1
            report["train_epoch_receipts"].append(copy.deepcopy(value))
            return value

        def native_valid():
            if report["native_VALID_calls_started"] >= 4:
                raise ValueError("Fixed four native VALID call bound")
            report["native_VALID_calls_started"] += 1
            sampler.phase("whole_VALID_call_" + str(report["native_VALID_calls_started"]))
            model_before = common.cpu_clone(torch, owner.model.state_dict())
            value = native_validate()
            equal(common.cpu_clone(torch, owner.model.state_dict()), model_before, "VALID_does_not_mutate_model_or_BN_buffers")
            if value["native_skipped_batches"] != 0 or value["VALID_graphs"] != 151:
                raise ValueError("Exactly 151 native VALID graphs and no source skip required")
            report["complete_native_VALID_calls"] += 1
            sampler.sample()
            return value

        def valid():
            if report["adapter_VALID_calls_started"] >= 4:
                raise ValueError("Fixed four adapter VALID call bound")
            report["adapter_VALID_calls_started"] += 1
            value = original_valid()
            expected_ids = owner.session.valid_ids.detach().cpu()
            store = getattr(owner.session.valid_view.dataset, "_data", None)
            if store is None:
                store = owner.session.valid_view.dataset.data
            expected_labels = store.y.index_select(0, expected_ids).detach().cpu()
            if len(value["ids"]) != 151 or value["TEST_accessed"] is not False:
                raise ValueError("Actual complete VALID151 / closed TEST required")
            equal(value["ids"], expected_ids, "native_VALID_IDs")
            equal(value["labels"], expected_labels, "native_VALID_labels")
            mean = value["same_bank_mean_VALID"]
            if mean["extra_forward_calls"] != 0:
                raise ValueError("Mean helper introduced an extra forward")
            report["complete_adapter_VALID_calls"] += 1
            report["same_call_mean_diagnostics"] += 1
            report["VALID_receipts"].append({"source_updates": owner.session.updates,
                "adapter_completed_epochs": owner.completed_epochs,
                "ids": owner._tensor_identity(value["ids"]), "labels": owner._tensor_identity(value["labels"]),
                "actual_logits": owner._tensor_identity(value["outputs"]), "mean_logits": owner._tensor_identity(mean["outputs"]),
                "actual_loss": value["loss"], "actual_metrics": value["metrics"],
                "mean_loss": mean["loss"], "mean_metrics": mean["metrics"], "authority": mean["authority"]})
            write()
            return value

        def selected_export(actual, **context):
            if report["selected_mean_exports"] >= 2:
                raise ValueError("Fixed two selected-mean export bound")
            value = original_export(actual, **context)
            if value is None or value["diagnostic_only"] is not True or value["selection_changed"] is not False:
                raise ValueError("Actual native selected mean was dropped or changed selection")
            equal(value["ids"], actual["ids"], "emitted_mean_IDs")
            equal(value["labels"], actual["labels"], "emitted_mean_labels")
            equal(value["raw_logits"], actual["same_bank_mean_VALID"]["outputs"], "emitted_same_call_mean")
            report["selected_mean_exports"] += 1
            sampler.phase("selected_mean_export_" + str(report["selected_mean_exports"]))
            report["emitted_mean_receipts"].append({"job_hash": value["job_hash"], "selected_epoch": value["selected_epoch"],
                "selected_checkpoint": value["selected_checkpoint"], "runtime_binding": value["runtime_binding"],
                "same_call_authority": value["same_call_authority"], "validation": value["validation"]})
            write()
            return value

        def forbidden_diagnostic():
            report["TRAIN_diagnostic_calls"] += 1
            raise ValueError("No TRAIN diagnostic admitted")

        owner.train_epoch, owner.validate = train, valid
        owner.session.validate = native_valid
        owner.selected_mean_export, owner.train_diagnostic = selected_export, forbidden_diagnostic
        return owner

    adapters.create_adapter = measured_create

    def snapshot(folder, name):
        destination = output / "snapshots" / name
        destination.mkdir(parents=True, exist_ok=False)
        rows = []
        for filename in ("JOB.json", "RESULT.json", "CURVE.jsonl", "EVENTS.jsonl", "PROGRESS.json",
                         "selected.pt", "recovery.pt", "selected_VALID.pt"):
            path = folder / filename
            if path.exists():
                copied = destination / filename
                shutil.copy2(path, copied)
                rows.append({"original": common.receipt(path), "preserved_copy": common.receipt(copied)})
        report["snapshots"].append({"name": name, "files": rows})
        write()

    folder = output / "own_fit"

    def invoke(name, resume=None, interrupt=False):
        original_atomic_json = run_fit.atomic_json
        interrupted = False

        def controlled_write(path, value):
            nonlocal interrupted
            original_atomic_json(path, value)
            if interrupt and Path(path) == folder / "PROGRESS.json" and value["train"]["epoch"] == 1:
                interrupted = True
                raise ControlledCompletedEpochStop("Declared engineering fault after durable complete epoch1 recovery/CURVE/PROGRESS")

        if interrupt:
            run_fit.atomic_json = controlled_write
        invocation_start = time.perf_counter()
        try:
            value = run_fit.run(copy.deepcopy(job), root, folder, resume=resume)
        finally:
            run_fit.atomic_json = original_atomic_json
        report["invocations"].append({"name": name, "controlled_completed_epoch_interruption": interrupt,
            "outer_invocation_elapsed_seconds": time.perf_counter() - invocation_start,
            "result": copy.deepcopy(value), "result_receipt_at_completion": common.receipt(folder / "RESULT.json")})
        write()
        if interrupt:
            if not interrupted or value["status"] != "failed" or value["failure"]["type"] != "ControlledCompletedEpochStop" \
                    or value["completed_epochs"] != 1 or value["training_epochs_this_invocation"] != 1 \
                    or value["optimizer_calls_this_invocation"] != {"attempts": [38], "completed": [38]}:
                raise ValueError("Controlled interruption/reporting did not retain the complete epoch1 boundary")
        elif value["status"] != "complete":
            raise ValueError("Unexpected native runtime failure: " + str(value.get("failure")))
        if common.digest(json.loads((folder / "JOB.json").read_text())) != common.digest(job):
            raise ValueError("Actual JOB hash changed across invocation/resume")
        gc.collect()  # Release observer/session reference cycles before the next fresh construction.
        return value

    def own_export():
        value = torch.load(folder / "selected_VALID.pt", map_location="cpu", weights_only=False)
        checkpoint = torch.load(folder / "selected.pt", map_location="cpu", weights_only=False)
        mean = value["same_bank_mean_VALID"]
        pin = common.receipt(folder / "selected.pt")
        epoch = checkpoint["selector"]["selected_epoch"]
        if value["schema"] != "own-selected-VALID-logits-v1" or mean["schema"] != "own-selected-same-bank-mean-VALID-v1" \
                or checkpoint["job_hash"] != common.digest(job) or value["job_hash"] != common.digest(job) \
                or mean["job_hash"] != common.digest(job) or mean["job_id"] != job["id"] \
                or value["selected_epoch"] != epoch or mean["selected_epoch"] != epoch \
                or checkpoint["adapter"]["completed_epochs"] != epoch or value["selected_checkpoint"] != pin \
                or mean["selected_checkpoint"] != pin or mean["runtime_binding"] != checkpoint["runtime_binding"] \
                or value["TEST_accessed"] is not False or mean["TEST_accessed"] is not False \
                or len(value["ids"]) != 151 or mean["extra_forward_calls"] != 0:
            raise ValueError("Persisted actual/mean own selected authority failed")
        for key in ("ids", "labels"):
            equal(value[key], mean[key], "persisted_actual_vs_mean/" + key)
        tensor_hashes = {"ids": adapters.BaseAdapter._tensor_identity(None, value["ids"]),
            "labels": adapters.BaseAdapter._tensor_identity(None, value["labels"]),
            "actual_logits": adapters.BaseAdapter._tensor_identity(None, value["raw_logits"]),
            "mean_logits": adapters.BaseAdapter._tensor_identity(None, mean["raw_logits"])}
        for key, identity in tensor_hashes.items():
            equal(identity, mean["same_call_authority"][key], "persisted_authority/" + key)
        emitted = report["emitted_mean_receipts"][-1]
        for key in ("job_hash", "selected_epoch", "selected_checkpoint", "runtime_binding", "same_call_authority", "validation"):
            equal(mean[key], emitted[key], "persisted_vs_emitted_mean/" + key)
        if not torch.isfinite(value["raw_logits"]).all().item() or not torch.isfinite(mean["raw_logits"]).all().item():
            raise ValueError("Persisted selected logits nonfinite")
        return value

    try:
        invoke("controlled_completed_epoch1", interrupt=True)
        snapshot(folder, "after_controlled_epoch1_before_resume")
        report["checks"].append("reporting repair: intentional post-PROGRESS failure retains completed epoch1 and invocation epoch1/38 Adam steps")
        resumed = invoke("resume_remaining_epoch2", resume=folder / "recovery.pt")
        if resumed["training_epochs_this_invocation"] != 1 or resumed["optimizer_calls_this_invocation"] != {"attempts": [38], "completed": [38]}:
            raise ValueError("Resume did not perform exactly one remaining full epoch/38 Adam steps")
        first_export = own_export()
        epochs = report["VALID_receipts"][:2]
        expected_epoch = 1 + max(range(2), key=lambda i: epochs[i]["actual_metrics"]["rocauc"])
        if first_export["selected_epoch"] != expected_epoch or resumed["selected_epoch"] != expected_epoch:
            raise ValueError("Own selector did not preserve first strict best actual VALID epoch")
        report["independent_selection_audit"] = {"actual_epoch_rocauc": [e["actual_metrics"]["rocauc"] for e in epochs],
            "first_strict_best_epoch": expected_epoch, "same_bank_mean_used_for_selection": False}
        snapshot(folder, "after_resumed_epoch2_before_stopped_finalize")
        selected_pin, recovery_pin = common.receipt(folder / "selected.pt"), common.receipt(folder / "recovery.pt")
        final = invoke("stopped_finalization_resume", resume=folder / "recovery.pt")
        if not final["finalization_only_resume"] or final["training_epochs_this_invocation"] != 0 \
                or final["optimizer_calls_this_invocation"] != {"attempts": [0], "completed": [0]}:
            raise ValueError("Stopped finalization performed extra training/optimizer work")
        second_export = own_export()
        for key in ("job_hash", "selected_epoch", "selected_checkpoint", "ids", "labels", "policy"):
            equal(first_export[key], second_export[key], "stopped_reloaded_actual/" + key)
        repeated_float(first_export["raw_logits"], second_export["raw_logits"], "stopped_reloaded_actual/logits")
        repeated_validation(first_export["validation"], second_export["validation"], "stopped_reloaded_actual/validation")
        for key in ("job_hash", "job_id", "selected_epoch", "selected_checkpoint", "runtime_binding", "ids", "labels"):
            equal(first_export["same_bank_mean_VALID"][key], second_export["same_bank_mean_VALID"][key], "stopped_reloaded_mean/" + key)
        repeated_float(first_export["same_bank_mean_VALID"]["raw_logits"], second_export["same_bank_mean_VALID"]["raw_logits"],
                       "stopped_reloaded_mean/logits")
        repeated_validation(first_export["same_bank_mean_VALID"]["validation"], second_export["same_bank_mean_VALID"]["validation"],
                            "stopped_reloaded_mean/validation")
        if common.receipt(folder / "selected.pt") != selected_pin or common.receipt(folder / "recovery.pt") != recovery_pin:
            raise ValueError("Stopped finalization changed own saved checkpoints")
        report["checks"].append("native resumed selected mean persists under actual JOB/runtime/source/checkpoint/IDs/labels and exact same-call tensor authority")
        report["checks"].append("stopped recovery resume: zero epochs/steps; exact own selected authority/IDs/labels/native metrics; repeated logits follow the frozen chosen-device policy")
        attempts = sum(sum(v["result"]["optimizer_calls_this_invocation"]["attempts"]) for v in report["invocations"])
        completed = sum(sum(v["result"]["optimizer_calls_this_invocation"]["completed"]) for v in report["invocations"])
        counts = tuple(report[k] for k in ("complete_native_TRAIN_epochs", "native_constructions_attempted", "native_constructions_completed",
            "complete_adapter_VALID_calls", "complete_native_VALID_calls", "same_call_mean_diagnostics", "selected_mean_exports",
            "source_optimizer_updates_observed", "TRAIN_diagnostic_calls"))
        if counts != (2, 3, 3, 4, 4, 4, 2, 76, 0) or (attempts, completed) != (76, 76):
            raise ValueError("Actual work counters differ from exact admitted workload")
        extra_counts = tuple(report[k] for k in ("optimizer_step_attempts_observed", "optimizer_steps_completed_observed",
            "native_update_calls_started", "native_update_calls_completed", "epoch_exhaustion_calls", "native_backward_calls",
            "exact_adapter_state_restores", "exact_global_RNG_restores"))
        if extra_counts != (76, 76, 78, 76, 2, 76, 4, 4):
            raise ValueError("Actual native step/backward/exhaustion/restore counters differ from frozen limits")
        after_sources = source_check()
        if after_sources != before_sources or (device == "cpu" and torch.cuda.is_initialized()):
            raise ValueError("Frozen source/data changed or CUDA context initialized")
        report.update(status="passed", optimizer_attempts=attempts, optimizer_completed=completed,
            source_rechecks_unchanged=True, full_native_TRAIN_records_per_epoch=1210, full_VALID_records_per_call=151,
            exact_actual_and_mean_exports_after_fresh_stopped_resume=all(r.get("bit_equal", True) for r in report["float_reexecution_diagnostics"]),
            floating_reexecution_policy="CPU bit equality; CUDA fixed rtol1e-4/atol1e-5 outputs/loss, exact native metrics; no tolerance search",
            selected_checkpoint=selected_pin, recovery_checkpoint=recovery_pin,
            selected_VALID_export=common.receipt(folder / "selected_VALID.pt"),
            limitations=["one finite molecular engineering trajectory on the chosen device, not scientific convergence or accuracy evidence",
                "no duplicate uninterrupted training; unchanged training AST and v3 lifecycle evidence remain separate",
                "controlled exception after durable completed epoch, not abrupt partial-epoch kill",
                "other native operators, link, source-reference policies and unchosen device not qualified; no earlier checkpoint migration"])
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(limit=20)})
    finally:
        adapters.create_adapter = original_create
        run_fit.restore_rng, torch.Tensor.backward = original_restore_rng, original_backward
        signal.alarm(0)
        try:
            sampler.phase("terminal_resource_and_disk_accounting")
            sampler.stop()
            report["own_output_disk_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            report["own_output_files"] = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size}
                                           for p in sorted(output.rglob("*")) if p.is_file()]
        except BaseException as error:
            report.update(status="failed", resource_finalization_failure={"type": type(error).__name__, "message": str(error)})
        write()
        print(json.dumps({k: report.get(k) for k in ("status", "failure", "complete_native_TRAIN_epochs", "native_constructions_completed",
            "complete_native_VALID_calls", "same_call_mean_diagnostics", "selected_mean_exports", "optimizer_attempts", "optimizer_completed",
            "elapsed_seconds", "process_peak_RSS_KiB", "CUDA_context_initialized")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
