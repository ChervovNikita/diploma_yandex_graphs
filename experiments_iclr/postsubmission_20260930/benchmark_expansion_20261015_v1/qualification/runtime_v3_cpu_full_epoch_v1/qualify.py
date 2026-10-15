"""Five full native TRAIN epochs maximum; actual v3 run/stop/resume paths."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time
import traceback


class ControlledCompletedEpochStop(BaseException):
    pass


def guard():
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Guarded allocation and hidden CUDA required before project reads")


def main():
    guard()
    here = Path(__file__).resolve().parent
    expansion = here.parent.parent
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    runtime = expansion / "runtime_v3"
    if not all(p.is_relative_to(root) for p in (here, runtime, output)) or output.exists():
        raise ValueError("Fresh own repository output only")
    inputs = json.loads((here / "INPUTS.json").read_text())
    sys.path.insert(0, str(runtime))
    import common
    import adapters
    import run_fit
    import torch
    import numpy as np
    torch.set_num_threads(2)

    def verify(row):
        path = common.inside(row["path"], root)
        if common.receipt(path) != row:
            raise ValueError("Pinned input changed: " + str(path))
        return path

    def source_check():
        rows = []
        manifest = runtime / "MANIFEST.json"
        if common.receipt(manifest)["sha256"] != inputs["runtime_manifest_sha256"]:
            raise ValueError("Frozen v3 source manifest changed")
        for row in json.loads(manifest.read_text())["files"]:
            path = runtime / row["path"]
            actual = common.receipt(path)
            if any(actual[k] != row[k] for k in ("bytes", "sha256")):
                raise ValueError("Frozen v3 source changed")
            rows.append(actual)
        parent = json.loads((runtime / "PARENT_SOURCE.json").read_text())
        if common.receipt(expansion / "runtime_v2/MANIFEST.json")["sha256"] != parent["manifest_sha256"]:
            raise ValueError("Parent v2 manifest changed")
        for row in parent["files"]:
            actual = common.receipt(expansion / "runtime_v2" / row["path"])
            if any(actual[k] != row[k] for k in ("bytes", "sha256")):
                raise ValueError("Immutable v2 runtime changed")
        return rows

    before_sources = source_check()
    verify(inputs["original_native_job"])
    for row in inputs["data_files"].values():
        verify(row)
    native_job = json.loads((here / "NATIVE2_JOB.json").read_text())
    boundary_job = json.loads((here / "BOUNDARY1_JOB.json").read_text())
    if native_job["policy"]["maximum_epochs"] != 2 or boundary_job["policy"]["maximum_epochs"] != 1:
        raise ValueError("Frozen two/one epoch engineering horizons only")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"schema": "runtime-v3-actual-CPU-full-epoch-qualification-v1", "status": "running",
        "runtime_manifest_sha256": inputs["runtime_manifest_sha256"], "inputs": common.receipt(here / "INPUTS.json"),
        "script": common.receipt(__file__), "torch": str(torch.__version__), "device": "CPU", "threads": 2,
        "TRAIN_epochs_started": 0, "complete_native_TRAIN_epochs": 0, "native_constructions": 0,
        "VALID_calls_started": 0, "complete_VALID_calls": 0, "source_optimizer_updates_observed": 0,
        "scientific_fits": 0, "TEST_accessed": False, "GPU_worker_changes": 0,
        "invocations": [], "train_epoch_receipts": [], "snapshots": [], "checks": []}

    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter() - started,
            process_user_seconds=usage.ru_utime, process_system_seconds=usage.ru_stime,
            process_peak_RSS_KiB=usage.ru_maxrss, CUDA_context_initialized=torch.cuda.is_initialized())
        common.atomic_json(output / "RESULT.json", report)

    original_create = adapters.create_adapter

    def measured_create(tensor_library, job, policy, repository):
        if report["native_constructions"] >= 6:
            raise ValueError("Fixed six construction bound")
        owner = original_create(tensor_library, job, policy, repository)
        report["native_constructions"] += 1
        if owner.recipe["_repository_root"] != str(root) or owner.session.recipe["_repository_root"] != str(root):
            raise ValueError("Actual repository root was not bound to construction recipe")
        original_train, original_valid = owner.train_epoch, owner.validate

        def train():
            if report["TRAIN_epochs_started"] >= 5:
                raise ValueError("Five total source TRAIN epochs maximum")
            report["TRAIN_epochs_started"] += 1
            before = owner.session.updates
            try:
                value = original_train()
            finally:
                report["source_optimizer_updates_observed"] += owner.session.updates - before
            if value["TRAIN_records_traversed"] != 1210 or value["raw_batches"] != 38 or value[
                    "complete_native_TRAIN"] is not True:
                raise ValueError("Complete source TRAIN1210 with38 source batches required")
            report["complete_native_TRAIN_epochs"] += 1
            report["train_epoch_receipts"].append({"job_hash": common.digest(job), **value})
            return value

        def valid():
            if report["VALID_calls_started"] >= 10:
                raise ValueError("Ten complete VALID calls maximum")
            report["VALID_calls_started"] += 1
            value = original_valid()
            if len(value["ids"]) != 151 or not torch.equal(value["ids"], owner.session.valid_ids) or value[
                    "complete_source_VALID"] is not True or value["TEST_accessed"] is not False:
                raise ValueError("Complete actual native VALID151 required")
            report["complete_VALID_calls"] += 1
            return value

        owner.train_epoch, owner.validate = train, valid
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

    def invoke(name, job, folder, resume=None, interrupt=False):
        original_atomic_json = run_fit.atomic_json
        interrupted = False

        def controlled_write(path, value):
            nonlocal interrupted
            original_atomic_json(path, value)
            if interrupt and Path(path) == folder / "PROGRESS.json" and value["train"]["epoch"] == 1:
                interrupted = True
                raise ControlledCompletedEpochStop("Declared engineering fault after complete epoch1 recovery, curve and progress")

        if interrupt:
            run_fit.atomic_json = controlled_write
        try:
            result = run_fit.run(copy.deepcopy(job), root, folder, resume=resume)
        finally:
            run_fit.atomic_json = original_atomic_json
        report["invocations"].append({"name": name, "job_hash": common.digest(job),
            "controlled_completed_epoch_interruption": interrupt, "result": copy.deepcopy(result),
            "result_receipt_at_completion": common.receipt(folder / "RESULT.json")})
        write()
        if interrupt:
            if not interrupted or result["status"] != "failed" or result["failure"]["type"] != "ControlledCompletedEpochStop" \
                    or result["completed_epochs"] != 1 or result["optimizer_calls_this_invocation"]["completed"] != [38]:
                raise ValueError("Controlled completed-epoch interruption did not preserve the expected boundary")
        elif result["status"] != "complete":
            raise ValueError("Unexpected actual runtime failure: " + str(result.get("failure")))
        if common.digest(json.loads((folder / "JOB.json").read_text())) != common.digest(job):
            raise ValueError("Immutable JOB hash changed across actual execution/resume")
        return result

    def equal(a, b, name):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                raise ValueError("Fresh/resume tensor differs: " + name)
        elif isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.dtype != b.dtype or not np.array_equal(a, b):
                raise ValueError("Fresh/resume numpy state differs: " + name)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                raise ValueError("Fresh/resume mapping differs: " + name)
            for key in a:
                equal(a[key], b[key], name + "/" + str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError("Fresh/resume sequence differs: " + name)
            for index, value in enumerate(a):
                equal(value, b[index], name + "/" + str(index))
        elif a != b:
            raise ValueError("Fresh/resume scalar differs: " + name)

    def own_export(folder, job):
        export = torch.load(folder / "selected_VALID.pt", map_location="cpu", weights_only=False)
        selected = torch.load(folder / "selected.pt", map_location="cpu", weights_only=False)
        if export["schema"] != "own-selected-VALID-logits-v1" or selected["schema"] != "native-full-epoch-recovery-v1" \
                or export["job_hash"] != common.digest(job) or selected["job_hash"] != common.digest(job) \
                or export["selected_checkpoint"] != common.receipt(folder / "selected.pt") \
                or export["selected_epoch"] != selected["selector"]["selected_epoch"] \
                or selected["adapter"]["completed_epochs"] != export["selected_epoch"] \
                or len(export["ids"]) != 151 or export["TEST_accessed"] is not False \
                or not torch.isfinite(export["raw_logits"]).all():
            raise ValueError("Own selected checkpoint/export custody failed")
        return export

    def equal_exports(a, b):
        for key in ("job_hash", "selected_epoch", "ids", "labels", "raw_logits", "validation", "policy"):
            equal(a[key], b[key], "own_selected_export/" + key)

    try:
        full, split, boundary = (output / name for name in ("native_uninterrupted", "native_interrupted_resume", "boundary_one_epoch"))
        uninterrupted = invoke("native_uninterrupted2", native_job, full)
        full_export = own_export(full, native_job)
        full_recovery = torch.load(full / "recovery.pt", map_location="cpu", weights_only=False)
        snapshot(full, "native_before_stopped_finalize_resume")
        finalized = invoke("native_stopped_finalize_resume", native_job, full, resume=full / "recovery.pt")
        if not finalized["finalization_only_resume"] or finalized["training_epochs_this_invocation"] != 0 or finalized[
                "optimizer_calls_this_invocation"] != {"attempts": [0], "completed": [0]}:
            raise ValueError("Stopped resume performed extra training/optimizer work")
        equal_exports(full_export, own_export(full, native_job))
        report["checks"].append("native stopped finalization resume:0extraepochs/steps, own selected export exact")
        invoke("native_controlled_stop_after_epoch1", native_job, split, interrupt=True)
        snapshot(split, "native_interrupted_epoch1_before_resume")
        resumed = invoke("native_resume_remaining_epoch2", native_job, split, resume=split / "recovery.pt")
        if resumed["training_epochs_this_invocation"] != 1 or resumed["optimizer_calls_this_invocation"]["completed"] != [38]:
            raise ValueError("Completed-epoch resume did not execute exactly its one remaining native epoch")
        split_recovery = torch.load(split / "recovery.pt", map_location="cpu", weights_only=False)
        for key in ("job_hash", "adapter", "selector", "global_rng", "optimizer_calls", "runtime_binding", "validation"):
            equal(full_recovery[key], split_recovery[key], "uninterrupted_vs_resumed_recovery/" + key)
        equal_exports(full_export, own_export(split, native_job))
        report["checks"].append("uninterrupted2epoch vs same-job completed-epoch1 resume: exact whole native/Adam/loader/RNG/selector state and selected export")
        invoke("boundary_one_full_epoch", boundary_job, boundary)
        boundary_export = own_export(boundary, boundary_job)
        snapshot(boundary, "boundary_before_stopped_finalize_resume")
        boundary_finalized = invoke("boundary_stopped_finalize_resume", boundary_job, boundary, resume=boundary / "recovery.pt")
        if not boundary_finalized["finalization_only_resume"] or boundary_finalized["training_epochs_this_invocation"] != 0 \
                or boundary_finalized["optimizer_calls_this_invocation"] != {"attempts": [0], "completed": [0]}:
            raise ValueError("Stopped boundary resume performed extra work")
        equal_exports(boundary_export, own_export(boundary, boundary_job))
        report["checks"].append("boundary full epoch and stopped finalization: own selected export exact,0extraepochs/steps")
        optimizer_attempts = sum(sum(i["result"]["optimizer_calls_this_invocation"]["attempts"]) for i in report["invocations"])
        optimizer_completed = sum(sum(i["result"]["optimizer_calls_this_invocation"]["completed"]) for i in report["invocations"])
        if (report["complete_native_TRAIN_epochs"], report["native_constructions"], report["complete_VALID_calls"],
                report["source_optimizer_updates_observed"], optimizer_attempts, optimizer_completed) != (5, 6, 10, 190, 190, 190):
            raise ValueError("Actual work counters differ from fixed complete-source workload")
        report.update(optimizer_attempts=optimizer_attempts, optimizer_completed=optimizer_completed,
            checkpoint_receipts=[common.receipt(p) for folder in (full, split, boundary)
                for p in (folder / "selected.pt", folder / "recovery.pt", folder / "selected_VALID.pt")])
        if source_check() != before_sources or torch.cuda.is_initialized():
            raise ValueError("Frozen sources changed or a CUDA context initialized")
        report.update(status="passed", sources_unchanged=True, exact_job_hash_on_resume=True,
            controlled_interruption_preserved=True, complete_native_TRAIN_records_per_epoch=1210,
            complete_VALID_records_per_call=151,
            limitations=["five bounded CPU engineering epochs; no convergence or accuracy claim",
                "MolBACE native GIN/dropout.5 and one boundary bank only; GPU/node/link/extension paths not executed",
                "controlled exception after a completed recovery boundary, not an abrupt partial-epoch kill",
                "no TEST, scientific fit, pilot/launcher migration or v2 checkpoint migration",
                "runtime_v3 is outside the frozen selected-VALID replay registry"])
    except BaseException as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(limit=18)}, automatic_retries=0)
    finally:
        adapters.create_adapter = original_create
        write()
    print(json.dumps({k: report.get(k) for k in ("status", "complete_native_TRAIN_epochs", "native_constructions",
        "complete_VALID_calls", "optimizer_attempts", "optimizer_completed", "checks", "failure")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
