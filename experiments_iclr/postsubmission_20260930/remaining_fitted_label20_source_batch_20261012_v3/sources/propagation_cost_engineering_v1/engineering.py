"""Supervised source entrypoint: prepare/profile/forecast, never real fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from propagation_cost_engineering_v1.contract import (
    ROOT, SCHEMA, CPU_SECONDS, DISK_BYTES, WallBudget, live_context, load_phase,
    output_path, sha256, verify_own_seal)


def exclusive_json(path, value):
    with output_path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "profile", "forecast"), required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    args = parser.parse_args()
    live = live_context(args.supervisor, Path(__file__))
    phase, output, phase_sha = load_phase(args.phase_file, args.mode)
    own_seal = verify_own_seal()
    started = time.perf_counter()
    try:
        if args.mode == "forecast":
            from propagation_cost_engineering_v1.profile import forecast_family
            reports = []
            if len(phase["profile_receipts"]) != 2:
                raise ValueError("Forecast requires both completed complete-graph profiles")
            for record in phase["profile_receipts"]:
                path = output_path(Path(phase["repo_root"]) / record["profile_relative_to_repo"])
                audit = output_path(Path(phase["repo_root"]) / record["completed_audit_relative_to_repo"])
                receipt = json.loads(audit.read_text())
                if (receipt.get("schema") != "gnnm-propagation-profile-completed-admission-v1"
                        or receipt.get("qualified") is not True
                        or receipt.get("profile_sha256") != sha256(path)):
                    raise ValueError("Completed profile supervisor/root receipt required")
                report = json.loads(path.read_text())
                if report.get("schema") != SCHEMA or report.get("status") != "completed_engineering_profiles":
                    raise ValueError("Profile incomplete")
                reports.append(report)
            result = forecast_family(reports, phase["prior_gpu_seconds"])
            output.mkdir(parents=True)
            exclusive_json(output / "FORECAST_RESULT.json", {**result, "live": live,
                           "phase_sha256": phase_sha, "source_manifest_sha256": own_seal})
        else:
            budget = WallBudget(CPU_SECONDS - phase["prior_cpu_seconds"], phase.get("deadline_utc"))
            from propagation_cost_engineering_v1.data_contract import load_acquired
            export, binding = load_acquired(phase)
            input_audit_seconds = time.perf_counter() - started
            budget.check()
            if args.mode == "prepare":
                from propagation_cost_engineering_v1.prepare import prepare_complete
                result = prepare_complete(export, binding, output, budget)
                if result["actual_prepared_file_bytes"] + phase["prior_disk_bytes"] > DISK_BYTES:
                    raise RuntimeError("Total acquired/prepared extra disk budget exceeded")
                exclusive_json(output / "PREPARATION_RECEIPT.json", {
                    "schema": SCHEMA, "status": "completed_complete_graph_preparation",
                    "phase_sha256": phase_sha, "source_manifest_sha256": own_seal,
                    "live": live, "data_binding": binding,
                    "input_audit_CPU_wall_seconds": input_audit_seconds,
                    "prepare_input_audit_and_CSR_wall_seconds": budget.elapsed(),
                    "prepared_manifest_sha256": sha256(output / "PREPARED_CSR.json"),
                    "actual_prepared_file_bytes": result["actual_prepared_file_bytes"],
                    "prior_cpu_seconds": phase["prior_cpu_seconds"], "prior_disk_bytes": phase["prior_disk_bytes"],
                    "CUDA_context_created": False, "scientific_fits": 0,
                    "heldout_targets_loaded": False})
            else:
                from propagation_cost_engineering_v1.profile import profile_complete
                result = profile_complete(phase, export, binding, output, phase_sha, live)
                exclusive_json(output / "PROFILE_ENTRY_RECEIPT.json", {
                    "schema": SCHEMA, "source_manifest_sha256": own_seal,
                    "profile_result_sha256": sha256(output / "PROFILE_RESULT.json"),
                    "pre_GPU_input_audit_CPU_wall_seconds": input_audit_seconds,
                    "whole_child_wall_seconds": time.perf_counter() - started,
                    "scientific_fits": 0, "heldout_targets_loaded": False})
        print(json.dumps({"status": result.get("status", "prepared"), "output": str(output),
                          "scientific_training_admitted": False}), flush=True)
    except BaseException as error:
        if not output.exists():
            output.mkdir(parents=True)
        exclusive_json(output / "ENGINEERING_FAILURE.json", {
            "schema": SCHEMA, "mode": args.mode, "type": type(error).__name__, "message": str(error),
            "phase_sha256": phase_sha, "source_manifest_sha256": own_seal, "live": live,
            "partial_artifacts_preserved": True, "automatic_resume": False,
            "scientific_training_admitted": False})
        raise


if __name__ == "__main__":
    main()
