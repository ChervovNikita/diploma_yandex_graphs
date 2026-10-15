"""Train one own native body or shared bank through complete TRAIN epochs.

CLI execution is allocation-only. TEST access and automatic scientific retries
are disabled. Full-matrix launch is not authorized by this bounded-pilot runner.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from controllers import controller, create_selector
from common import (EXCLUDED_DATASETS, atomic_json, atomic_torch, cpu_clone, digest,
                    event, global_rng, inside, receipt, require_allocation,
                    restore_rng, seed_all, utc)


class OptimizerObserver:
    def __init__(self, torch, owners, previous=None):
        self.torch, self.owners = torch, owners
        self.attempts = list((previous or {}).get("attempts", [0] * len(owners)))
        self.completed = list((previous or {}).get("completed", [0] * len(owners)))
        self.originals = []
        for index, owner in enumerate(owners):
            original = owner.step
            self.originals.append(original)
            def wrap(i, function):
                def step(*args, **kwargs):
                    self.attempts[i] += 1
                    for group in self.owners[i].param_groups:
                        for parameter in group["params"]:
                            gradient = parameter.grad
                            if gradient is not None:
                                values = gradient.coalesce().values() if gradient.is_sparse else gradient
                                if not self.torch.isfinite(values).all().item():
                                    raise ValueError("Nonfinite native/private gradient before optimizer transition")
                    result = function(*args, **kwargs)
                    self.completed[i] += 1
                    return result
                return step
            owner.step = wrap(index, original)

    def state(self):
        return {"attempts": list(self.attempts), "completed": list(self.completed)}

    def close(self):
        for owner, original in zip(self.owners, self.originals):
            owner.step = original


def validate_job(job, root=None):
    if job.get("schema") != "native-full-epoch-fit-v1" or job.get("family") not in ("node", "ogb", "link"):
        raise ValueError("Frozen node/OGB/explicit-link full-epoch job schema required")
    if job.get("TEST_access") is not False:
        raise ValueError("TEST must remain explicitly closed for these pilots")
    dataset = job["recipe"]["dataset"].lower().replace("_", "-").replace(" ", "-")
    if dataset in EXCLUDED_DATASETS:
        raise ValueError("Original-paper experiment reruns are excluded")
    if job.get("device", "cuda:0") != "cuda:0":
        raise ValueError("This prepared pilot worker targets only the authorized allocation GPU0")
    policy, _, _ = controller(job, root)
    if policy.track == "source_reference" and job["procedure"] != "native_single":
        raise ValueError("Source-reference track is native M1 only")
    if job.get("source_rng_emulation") is not None and (job["family"] != "ogb"
            or policy.track != "source_reference" or job["procedure"] != "native_single"):
        raise ValueError("Source RNG emulation cannot be silently applied to another task/track/method")
    if type(job["seed"]) is not int or job["seed"] < 0:
        raise ValueError("Explicit independent nonnegative fit seed required")
    return policy


def compact_validation(value):
    if value.get("role") != "VALID" or value.get("complete_source_VALID") is not True \
            or value.get("TEST_accessed") is not False:
        raise ValueError("Only complete native VALID may reach selection")
    metrics = {k: float(v) for k, v in value["metrics"].items()}
    loss = float(value["loss"])
    if not math.isfinite(loss) or any(not math.isfinite(v) for v in metrics.values()):
        raise ValueError("Nonfinite complete-VALID loss/metric")
    return {"role": "VALID", "complete_source_VALID": True, "loss": loss,
            "metrics": metrics, "records": len(value["ids"]), "TEST_accessed": False}


def resource_record(torch):
    import resource
    rss_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return {"own_peak_rss_bytes": rss_peak,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def claim_lock(output, job_hash, resume):
    lock = output / "OWN_WORKER.lock"
    if lock.exists():
        if resume is None:
            raise ValueError("Own fit already has a worker lock")
        prior = json.loads(lock.read_text())
        if prior.get("job_hash") != job_hash:
            raise ValueError("Lock belongs to another immutable fit")
        try:
            os.kill(int(prior["pid"]), 0)  # liveness only; no process is stopped
        except ProcessLookupError:
            history = output / "history"
            history.mkdir(exist_ok=True)
            lock.replace(history / ("terminated_owner_" + str(time.time_ns()) + ".lock.json"))
        else:
            raise ValueError("Recorded owner PID is still live; do not steal or duplicate its fit")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock_fd, json.dumps({"pid": os.getpid(), "job_hash": job_hash, "UTC": utc()}).encode())
    os.close(lock_fd)
    return lock


def run(job, root, output, resume=None):
    import torch
    from adapters import create_adapter
    root, output = Path(root).resolve(), inside(output, root)
    policy = validate_job(job, root)
    policy, policy_module, policy_receipt = controller(job, root)
    job_hash = digest(job)
    if output.exists() and resume is None:
        raise ValueError("Choose a fresh fit output; explicit --resume is required for existing work")
    if resume is not None and not output.is_dir():
        raise ValueError("Resume output must be the existing fit evidence directory")
    output.mkdir(parents=True, exist_ok=resume is not None)
    if resume is not None and (output / "JOB.json").is_file() \
            and digest(json.loads((output / "JOB.json").read_text())) != job_hash:
        raise ValueError("Existing fit descriptor differs; preserve it and choose another output")
    lock = claim_lock(output, job_hash, resume)
    if resume is not None and (output / "RESULT.json").is_file():
        history = output / "history"
        history.mkdir(exist_ok=True)
        shutil.copy2(output / "RESULT.json", history / ("before_resume_" + str(time.time_ns()) + "_RESULT.json"))
    atomic_json(output / "JOB.json", job)
    adapter = observer = None
    calls_at_invocation_start = None
    started = time.perf_counter()
    selector = create_selector(policy, policy_module)
    runtime_sources = {name: receipt(Path(__file__).with_name(name))
                       for name in ("run_fit.py", "adapters.py", "bookkeeping.py", "common.py", "controllers.py", "source_rng.py")}
    result = {"schema": "native-full-epoch-fit-result-v1", "status": "running", "job_id": job["id"],
              "job_hash": job_hash, "cohort_id": job["cohort_id"], "procedure": job["procedure"],
              "seed": job["seed"], "split": job.get("split", 0), "body": job.get("body"),
              "dataset": job["recipe"]["dataset"], "family": job["family"], "policy": job["policy"],
              "TEST_accessed": False, "full_campaign_GO": False,
              "scientific_scope": "bounded frozen validation-only pilot" if policy.track != "source_reference"
                   else "native source-horizon reference with disclosed no-TEST diagnostic adaptation",
              "runtime_sources": runtime_sources, "started_UTC": utc(), "resume_requested": resume is not None,
              "completed_epochs": 0, "selected_epoch": None, "automatic_retries": 0,
              "recovery_granularity": "complete native epoch; an interrupted partial epoch is not silently counted"}
    try:
        torch.set_num_threads(int(job.get("torch_threads", 2)))
        torch.cuda.reset_peak_memory_stats()
        seed_all(torch, job["seed"])
        adapter = create_adapter(torch, job, policy, root)
        result["source_metadata"] = adapter.metadata()
        runtime_binding = {"runtime_sha256": {name: pin["sha256"] for name, pin in runtime_sources.items()},
                           "policy_sha256": None if policy_receipt is None else policy_receipt["sha256"],
                           "callback_sha256": adapter.callback_receipt["sha256"],
                           "extension_sha256": None if adapter.extension_receipt is None else adapter.extension_receipt["sha256"],
                           "dependency_sha256": {pin["path"]: pin["sha256"] for pin in adapter.dependency_receipts}}
        result["runtime_binding"] = runtime_binding
        result["construction_seconds"] = time.perf_counter() - started
        previous_calls = None
        inherited_wall = 0.0
        if resume is not None:
            checkpoint = torch.load(inside(resume, root), map_location="cpu", weights_only=False)
            if checkpoint["job_hash"] != job_hash:
                raise ValueError("Resume job/settings/seed/source binding changed")
            if checkpoint.get("runtime_binding") != runtime_binding:
                raise ValueError("Resume runtime/native source changed; preserve this attempt and freeze any repair explicitly")
            adapter.load_state(checkpoint["adapter"])
            selector.load_state(checkpoint["selector"])
            restore_rng(torch, checkpoint["global_rng"])
            previous_calls = checkpoint["optimizer_calls"]
            inherited_wall = float(checkpoint.get("fit_wall_seconds_through_checkpoint", 0.0))
            if adapter.completed_epochs != selector.completed_epoch:
                raise ValueError("Native completed-epoch and selector clocks disagree")
            result["resumed_checkpoint"] = receipt(inside(resume, root))
            result["partial_work_after_saved_checkpoint"] = "unknown_if_prior_process_died_without_terminal_receipt; prior records retained"
        observer = OptimizerObserver(torch, adapter.optimizers, previous_calls)
        calls_at_invocation_start = copy.deepcopy(observer.state())
        completed_at_invocation_start = selector.completed_epoch
        result.update(completed_epochs=selector.completed_epoch, selected_epoch=selector.selected_epoch,
                      best_validation_metric=selector.best_metric, training_epochs_this_invocation=0)
        event(output / "EVENTS.jsonl", {"event": "worker_started", "pid": os.getpid(), "job_hash": job_hash,
                                      "resume": resume is not None, "completed_epoch": selector.completed_epoch})
        decision = selector.stopping_status()
        result["finalization_only_resume"] = resume is not None and decision["stop"]
        if result["finalization_only_resume"]:
            event(output / "EVENTS.jsonl", {"event": "resume_already_stopped_finalize_selected",
                                          "selection": decision, "TRAIN_epochs": 0})
        while not decision["stop"]:
            epoch_start = time.perf_counter()
            train = adapter.train_epoch()
            if train["epoch"] != selector.completed_epoch + 1 or train.get("complete_native_TRAIN") is not True:
                raise ValueError("Only completed native epochs may advance training/selection")
            diagnostic = adapter.train_diagnostic() if policy.evaluate_TRAIN_each_epoch else None
            raw_validation = adapter.validate()
            validation = compact_validation(raw_validation)
            post_validation = getattr(adapter, "after_epoch_diagnostics", lambda: None)()
            decision = selector.observe_validation(train["epoch"], raw_validation) if job["family"] == "link" \
                else selector.observe(train["epoch"], validation["metrics"])
            payload = {"schema": "native-full-epoch-recovery-v1", "job_hash": job_hash,
                       "adapter": cpu_clone(torch, adapter.state()), "selector": selector.state(),
                       "global_rng": cpu_clone(torch, global_rng(torch)), "optimizer_calls": observer.state(),
                       "runtime_binding": runtime_binding,
                       "fit_wall_seconds_through_checkpoint": inherited_wall + time.perf_counter() - started,
                       "validation": validation, "TEST_accessed": False}
            selected_receipt = None
            if decision["save_selected"]:
                selected_receipt = atomic_torch(torch, output / "selected.pt", payload)
            recovery = atomic_torch(torch, output / "recovery.pt", payload)
            row = {"event": "completed_epoch", "train": train, "TRAIN_diagnostic": diagnostic,
                   "post_VALID_rng_bookkeeping": post_validation,
                   "validation": validation, "selection": decision, "optimizer_calls": observer.state(),
                   "epoch_wall_seconds": time.perf_counter() - epoch_start,
                   "resources": resource_record(torch), "recovery_receipt": recovery,
                   "new_selected_receipt": selected_receipt}
            event(output / "CURVE.jsonl", row)
            atomic_json(output / "PROGRESS.json", {"job_id": job["id"], "job_hash": job_hash, **row})
            result.update(completed_epochs=selector.completed_epoch, selected_epoch=selector.selected_epoch,
                          best_validation_metric=selector.best_metric,
                          training_epochs_this_invocation=selector.completed_epoch-completed_at_invocation_start)
        stop_reason = decision["stop_reason"]
        selected_path = output / "selected.pt"
        if selector.selected_epoch is None and job["family"] == "link" and policy.track == "source_reference":
            # In the pinned source BUDDY controller, VALID must exceed the
            # initial zero sentinel. An all-zero curve has no native winner.
            result.update(status="no_selected_state", stop_reason=stop_reason,
                          selected_checkpoint=None, selected_VALID_export=None,
                          selected_VALID=None, best_validation_metric=selector.best_metric,
                          convergence_established=False,
                          source_selection_outcome="native selector produced no selected checkpoint; no fallback")
            return result
        if not selected_path.is_file():
            raise ValueError("No own selected state exists; never substitute final/current weights")
        selected = torch.load(selected_path, map_location="cpu", weights_only=False)
        if selected["job_hash"] != job_hash:
            raise ValueError("Selected checkpoint belongs to another fit")
        if selected.get("runtime_binding") != runtime_binding:
            raise ValueError("Selected checkpoint runtime/native source differs")
        adapter.load_state(selected["adapter"])
        restore_rng(torch, selected["global_rng"])
        final_valid = adapter.validate()
        valid_summary = compact_validation(final_valid)
        export = {"schema": "own-selected-VALID-logits-v1", "role": "VALID", "TEST_accessed": False,
                  "job_id": job["id"], "job_hash": job_hash, "cohort_id": job["cohort_id"],
                  "dataset": job["recipe"]["dataset"], "recipe_id": job["recipe"]["id"],
                  "family": job["family"], "procedure": job["procedure"], "body": job.get("body"),
                  "seed": job["seed"], "split": job.get("split", 0),
                  "selected_epoch": selected["selector"]["selected_epoch"],
                  "selected_checkpoint": receipt(selected_path),
                  "ids": final_valid["ids"].detach().cpu(), "labels": final_valid["labels"].detach().cpu(),
                  "raw_logits": final_valid["outputs"].detach().cpu(), "validation": valid_summary,
                  "source_metadata": result["source_metadata"], "policy": job["policy"]}
        if job["family"] == "link":
            if (final_valid.get("complete_VALID_positive_records"), final_valid.get("complete_VALID_negative_records")) != (60084, 100000):
                raise ValueError("Selected link export requires the complete official VALID queries")
            export.update(query_endpoints=final_valid["query_endpoints"].detach().cpu(),
                          complete_VALID_positive_records=60084, complete_VALID_negative_records=100000)
        export_receipt = atomic_torch(torch, output / "selected_VALID.pt", export)
        result.update(status="complete", stop_reason=stop_reason, selected_checkpoint=receipt(selected_path),
                      selected_VALID_export=export_receipt, selected_VALID=valid_summary,
                      convergence_established=stop_reason == "patience",
                      selected_epoch=selected["selector"]["selected_epoch"],
                      best_validation_metric=selected["selector"]["best_metric"])
    except BaseException as error:
        result.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc(limit=14)},
                      completed_epochs=selector.completed_epoch, selected_epoch=selector.selected_epoch)
    finally:
        result.update(finished_UTC=utc(), invocation_wall_seconds=time.perf_counter() - started,
                      optimizer_calls=None if observer is None else observer.state(), resources=resource_record(torch))
        result["optimizer_calls_this_invocation"] = None if observer is None else {
            key: [a - b for a, b in zip(observer.state()[key], calls_at_invocation_start[key])]
            for key in ("attempts", "completed")}
        result["optimizer_calls_are_trajectory_cumulative"] = True
        if observer is not None:
            observer.close()
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                result["close_error"] = str(error)
        atomic_json(output / "RESULT.json", result)
        event(output / "EVENTS.jsonl", {"event": "worker_ended", "status": result["status"],
                                      "optimizer_calls": result["optimizer_calls"], "pid": os.getpid()})
        lock.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    require_allocation()  # before project metadata/source/data reads
    root = Path(args.project_root).resolve()
    job = json.loads(inside(args.job, root).read_text())
    result = run(job, root, args.output, args.resume)
    print(json.dumps({k: result.get(k) for k in ("status", "job_id", "completed_epochs", "selected_epoch",
                                               "selected_VALID", "stop_reason", "optimizer_calls")}))
    sys.exit(0 if result["status"] in ("complete", "no_selected_state") else 1)


if __name__ == "__main__":
    main()
