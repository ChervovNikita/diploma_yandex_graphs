"""Observed entrypoint for a future explicitly admitted unchanged native task.

This is source preparation, not an admission or campaign launcher.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    from host_extension import guard, install_native_authority
    config = guard()  # Genuine new identity; frozen scientific runtime remains unchanged.
    from observe_exports import observe, file_receipt, scalar_copy, atomic_json, digest
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--preset-file", required=True)
    parser.add_argument("--preset-sha256", required=True); parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--resume")
    parser.add_argument("--members", nargs=4)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    preset_path, output = Path(args.preset_file).resolve(), Path(args.output).resolve()
    if not all(p.is_relative_to(root) for p in (preset_path, output)) or file_receipt(preset_path)["sha256"] != args.preset_sha256:
        raise ValueError("Pinned own registry/output required")
    preset = json.loads(preset_path.read_text())
    is_fit = args.task_id in preset["fit_specs"]
    spec = preset["fit_specs"][args.task_id] if is_fit else preset["I4_specs"][args.task_id]
    reporting = output.parent.parent / "reporting" / digest(args.task_id) / str(time.time_ns())
    if not reporting.resolve().is_relative_to(root):
        raise ValueError("Reporting output escapes repository")
    entry = spec["runtime_entrypoint"] if is_fit else spec["aggregator_source"]
    if file_receipt(entry["path"]) != entry:
        raise ValueError("Qualified underlying runtime source changed")
    authority = install_native_authority('/composed/' in entry['path'])
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output.parent / 'ACTUAL_HOST_AUTHORITY.json', authority)
    sys.path.insert(0, str(Path(entry["path"]).parent))
    module_spec = importlib.util.spec_from_file_location("run_fit" if is_fit else "aggregate_selected", entry["path"])
    runtime = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runtime)
    if is_fit:
        if args.members is not None or file_receipt(spec["descriptor"]["path"]) != spec["descriptor"]:
            raise ValueError("Exact fit descriptor and no I4 member arguments required")
        job = json.loads(Path(spec["descriptor"]["path"]).read_text())
        with observe(runtime, job, reporting, binding_spec=spec) as observer:
            result = runtime.run(job, root, output, args.resume)
            observer.record_invocation(result, "production_invocation")
    else:
        if args.resume is not None or args.members is None:
            raise ValueError("I4 consumes four existing exports and never resumes training")
        reporting.mkdir(parents=True)
        original = runtime.atomic_torch
        captured = []
        started = time.perf_counter()

        def save(library, path, payload):
            pin = original(library, path, payload)
            if payload.get("schema") == "independent-own-selected-VALID-bank-v1":
                row = {"schema": "observed-I4-aggregate-scalar-sidecar-v1", "bank_receipt": scalar_copy(pin),
                    "producer_receipt": file_receipt(__file__), "aggregator_source": entry,
                    "cohort_id": payload["cohort_id"], "metrics": scalar_copy(payload["metrics"]), "loss": payload["loss"],
                    "members": scalar_copy(payload["members"]), "aggregation": payload["aggregation"],
                    "extra_training_fits": payload["extra_training_fits"], "extra_native_forwards": 0,
                    "RNG_calls": 0, "tensor_deserializations_added_by_observer": 0, "payload_mutated": False}
                captured.append(row); atomic_json(reporting / "AGGREGATE_SCALARS.json", row)
            return pin

        runtime.atomic_torch = save
        try:
            result = runtime.aggregate(args.members, root, output)
        finally:
            runtime.atomic_torch = original
            atomic_json(reporting / "INDEX.json", {"schema": "observed-I4-reporting-index-v1", "task_id": args.task_id,
                "aggregator_source": entry, "producer_receipt": file_receipt(__file__),
                "worker_wall_seconds": time.perf_counter() - started, "captured_aggregate_exports": len(captured)})
    print(json.dumps({key: result.get(key) for key in ("status", "job_id", "cohort_id", "completed_epochs")}))
    return 0 if result["status"] in ("complete", "no_selected_state") else 1


if __name__ == "__main__":
    sys.exit(main())
