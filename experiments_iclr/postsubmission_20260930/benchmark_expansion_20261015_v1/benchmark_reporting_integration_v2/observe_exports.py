"""Observe existing run_fit exports and invocation metadata, without native work.

Use `with observe(run_fit_module, job, output, binding_spec=None) as observer:`,
then observer.record_invocation(result, label) after each unchanged run_fit.run.
Only scalar/receipt fields are read from the existing export payload.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_receipt(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def scalar_copy(value):
    """Reject tensor/library objects instead of inspecting or converting them."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if type(value) is list:
        return [scalar_copy(v) for v in value]
    if type(value) is dict:
        return {str(k): scalar_copy(v) for k, v in value.items()}
    raise TypeError("Observer accepts scalar/JSON metadata only: " + type(value).__name__)


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class ExportObserver:
    def __init__(self, runtime, job, output, binding_spec):
        self.runtime, self.job, self.output = runtime, job, Path(output)
        self.job_hash = digest(job)
        self.binding_spec = binding_spec
        self.started = time.perf_counter()
        self.exports, self.invocations = [], []
        self.data_signatures = {}
        self.verified_files = []
        if self.output.exists():
            raise ValueError("Preserve earlier reporting observations; fresh output required")
        self.output.mkdir(parents=True)
        if binding_spec is not None:
            if binding_spec["job_hash"] != self.job_hash:
                raise ValueError("Observer binding spec belongs to another job")
            for pin in binding_spec["observed_input_files"]:
                now = file_receipt(pin["path"])
                if any(now[k] != pin[k] for k in ("path", "bytes", "sha256")):
                    raise ValueError("Observed source/data/evaluator bytes differ: " + pin["path"])
                stat = Path(pin["path"]).stat()
                self.data_signatures[pin["path"]] = (stat.st_size, stat.st_mtime_ns)
                self.verified_files.append(now)
        self.original = runtime.atomic_torch
        self.producer_receipt = file_receipt(__file__)

    def save(self, library, path, payload):
        # Execute the original writer first, with the identical payload object.
        # The observer never clones, converts, indexes, mutates or serializes a
        # tensor; only the original writer performs its existing serialization.
        pin = self.original(library, path, payload)
        if payload.get("schema") != "own-selected-VALID-logits-v1":
            return pin
        if payload["job_hash"] != self.job_hash or payload["job_id"] != self.job["id"] \
                or payload.get("role") != "VALID" or payload.get("TEST_accessed") is not False:
            raise ValueError("Observed selected export job/role mismatch")
        row = {"schema": "actual-selected-export-scalar-sidecar-v1", "producer_receipt": self.producer_receipt,
            "job_hash": self.job_hash, "job_id": payload["job_id"], "selected_export": scalar_copy(pin),
            "selected_checkpoint": scalar_copy(payload["selected_checkpoint"]),
            "selected_epoch": payload["selected_epoch"], "actual_validation": scalar_copy(payload["validation"]),
            "extra_native_calls": 0, "payload_mutated": False, "RNG_calls": 0}
        mean = payload.get("same_bank_mean_VALID")
        if mean is not None:
            if mean.get("job_hash") != self.job_hash or mean.get("selected_checkpoint") != payload["selected_checkpoint"] \
                    or mean.get("selected_epoch") != payload["selected_epoch"] or mean.get("extra_forward_calls") != 0:
                raise ValueError("Observed same-bank mean changed selection/state or used extra forwards")
            validation = mean.get("validation")
            if validation is None:
                validation = {"role": "VALID", "complete_source_VALID": True,
                    "records": payload["validation"]["records"], "loss": mean["loss"], "metrics": mean["metrics"], "TEST_accessed": False}
            row["same_selected_bank_mean"] = {"schema": mean["schema"], "validation": scalar_copy(validation),
                "same_call_authority": scalar_copy(mean.get("same_call_authority", mean.get("authority"))),
                "runtime_binding": scalar_copy(mean["runtime_binding"]), "diagnostic_only": True,
                "selection_changed": False, "extra_forward_calls": 0}
        self.exports.append(row)
        atomic_json(self.output / ("SELECTED_EXPORT_" + str(len(self.exports)).zfill(3) + ".json"), row)
        return pin

    def record_invocation(self, result, label):
        if result.get("job_hash") != self.job_hash or result.get("job_id") != self.job["id"]:
            raise ValueError("Invocation result is not the observed job")
        bindings, reasons = {}, []
        if self.binding_spec is None:
            reasons.append("No campaign binding spec attached; engineering observation only")
        else:
            for name, before in self.data_signatures.items():
                stat = Path(name).stat()
                if (stat.st_size, stat.st_mtime_ns) != before:
                    reasons.append("Verified input changed during execution: " + name)
            actual = result.get("runtime_binding") or {}
            if actual.get("callback_sha256") != self.job.get("callback_sha256") or result.get("policy") != self.job["policy"]:
                reasons.append("Actual callback/controller metadata differs from the verified descriptor")
            if not reasons:
                bindings = scalar_copy(self.binding_spec["bindings"])
        selected = result.get("selected_VALID_export")
        sidecars = [r for r in self.exports if r["selected_export"] == selected]
        row = {"schema": "observed-runtime-invocation-reporting-v1", "label": str(label),
            "job_hash": self.job_hash, "job_id": self.job["id"], "producer_receipt": self.producer_receipt,
            "selected_export_sha256": None if selected is None else selected["sha256"],
            "bindings": bindings, "binding_unresolved_reasons": reasons,
            "selected_sidecar": None if not sidecars else scalar_copy(sidecars[-1]),
            "actual_cost": {"status": result.get("status"), "worker_wall_seconds": result.get("invocation_wall_seconds"),
                "optimizer_calls_this_invocation": scalar_copy(result.get("optimizer_calls_this_invocation")),
                "optimizer_calls_trajectory_cumulative": scalar_copy(result.get("optimizer_calls")),
                "resources": scalar_copy(result.get("resources")), "automatic_retries": result.get("automatic_retries"),
                "resume_requested": result.get("resume_requested"), "finalization_only_resume": result.get("finalization_only_resume")},
            "extra_model_forwards_or_updates": 0, "RNG_calls": 0, "tensor_deserializations": 0}
        self.invocations.append(row)
        atomic_json(self.output / ("INVOCATION_" + str(len(self.invocations)).zfill(3) + ".json"), row)
        return row

    def close(self):
        atomic_json(self.output / "INDEX.json", {"schema": "observed-runtime-reporting-index-v1",
            "job_hash": self.job_hash, "producer_receipt": self.producer_receipt,
            "exports": len(self.exports), "invocations": len(self.invocations),
            "observer_elapsed_seconds": time.perf_counter() - self.started,
            "verified_input_files": self.verified_files, "binding_spec_attached": self.binding_spec is not None,
            "extra_native_work": 0, "RNG_calls": 0, "tensor_deserializations": 0})


@contextmanager
def observe(runtime, job, output, binding_spec=None):
    observer = ExportObserver(runtime, job, output, binding_spec)
    runtime.atomic_torch = observer.save
    try:
        yield observer
    finally:
        runtime.atomic_torch = observer.original
        observer.close()
