#!/usr/bin/env python3
"""One separately released complete-family selected-state replay; no fit/resume."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import os
import tempfile
from replay_gate import preflight, family_gate, admitted_input_custody, require, ARMS, descriptor
from replay_contract import public_cells, summaries


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class Accounting:
    def __init__(self, context, started):
        self.context, self.started = context, started
        self.phase_name, self.phase_started = "startup", perf_counter()
        self.phases, self.cuda_started = [], False
        self.work = {"cells_attempted": 0, "score_valid_calls": 0}
        self.flush("IN_PROGRESS")

    def flush(self, status):
        atomic_json(self.context["output"] / "STATUS.json", {"schema": "ncnc-selected-state-replay-count-status-v1", "status": status, "phase": self.phase_name, "work": self.work, "observed_wall_seconds": perf_counter() - self.started, "predictive_values_exposed": False})
        atomic_json(self.context["output"] / "REPLAY_ATTEMPT.json", {"schema": "ncnc-selected-state-replay-inclusive-attempt-v1", "identity": self.context["identity"], "root_release_sha256": self.context["release_sha256"], "sidecar_manifest_sha256": self.context["sidecar_manifest_sha256"], "status": status, "phase": self.phase_name, "phases": self.phases, "work": self.work, "observed_wall_seconds": perf_counter() - self.started, "automatic_retry": False})

    def phase(self, name):
        now = perf_counter()
        self.phases.append({"phase": self.phase_name, "wall_seconds": now - self.phase_started})
        self.phase_name, self.phase_started = name, now
        self.flush("IN_PROGRESS")

    def progress(self, cells, calls):
        self.work.update(cells_attempted=cells, score_valid_calls=calls)
        self.flush("IN_PROGRESS")

    def private(self, value):
        atomic_json(self.context["output"] / "PRIVATE_REPLAY_DETAILS.json", value)

    def finish(self, status):
        self.phase("finalization")
        self.flush(status)
        return {"inclusive_wall_seconds": perf_counter() - self.started, "phases": self.phases, "terminal_write_tail_measured": False, "scope": "startup_release_source_hashes_all_custody_load_restore_transfer_full_VALID_replay_and_finalization", "training_updates": 0, "automatic_retry": False}


def execute(context, started, *, fabricated_data=None, synthetic_after_metadata=None, synthetic_before_custody=None):
    """Shared production/qualification flow; injection is synthetic-entry-only."""
    require((fabricated_data is not None) == bool(context.get("synthetic_only")), "Synthetic data cannot enter the production CLI")
    require((synthetic_after_metadata is None and synthetic_before_custody is None) or context.get("synthetic_only") is True, "Failure injection is confined to admitted qualification")
    output = context["output"]
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    import fcntl
    fd = os.open(output / ".RUN_LOCK", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    accounting = Accounting(context, started)
    gate = None
    cells = [{"arm": arm, "base_seed": seed, "status": "NOT_REPLAYED_METADATA_GATE"} for arm in ARMS for seed in range(5)]
    result = {"schema": "ncnc-selected-state-replay-result-v1", "identity": context["identity"], "sidecar_manifest_sha256": context["sidecar_manifest_sha256"], "root_release_sha256": context["release_sha256"], "family_lock": context["release"]["family_lock"], "denominator": {"units": 20, "unique_fits": 35, "fit_epochs": 3500, "optimizer_steps": 59500, "served_cells": 25, "nominal_score_valid_calls": 40}, "summary": None, "training_costs": None, "TEST_opened": False, "training_updates": 0, "new_checkpoint_selection": False, "no_success_only_subset_summary": True, "fabricated_inputs_only": bool(context.get("synthetic_only")), "scientific_replay": not bool(context.get("synthetic_only"))}
    result["denominator_scope"] = "fabricated_metadata_contract_only_not_training_execution" if context.get("synthetic_only") else "frozen_complete_study_family"
    exit_code = 1
    try:
        accounting.phase("complete_family_stdlib_metadata_all_custody_gate")
        gate = family_gate(context)
        result["training_costs"] = gate["costs"]
        cells = gate["cells"]
        if not gate["full_success"]:
            result.update(status="TERMINAL_FAILED_FAMILY_METADATA_ONLY", numerical_runtime_imported=False, complete_served_cells=gate["complete_served_cells"], unique_fits_completed=gate["unique_fits_completed"], numerical_replay={"score_valid_calls": 0, "no_partial_replay": True})
        else:
            if synthetic_after_metadata is not None:
                synthetic_after_metadata(context)
            from replay_numeric import run
            cells, replay = run(context, gate, accounting, fabricated_data=fabricated_data)
            result["numerical_replay"] = replay
            accounting.phase("immutable_source_input_custody_recheck")
            if synthetic_before_custody is not None:
                synthetic_before_custody(context)
            result["final_input_custody"] = admitted_input_custody(context)
            after = family_gate(context)
            require(after["lock"] == gate["lock"], "Immutable family lock changed during replay")
            passed = replay["status"] == "PASS" and len(cells) == 25 and all(c["status"] == "PASS" for c in cells)
            result["status"] = "ALL25_SELECTED_STATE_REPLAY_PASS" if passed else "FAILED_SELECTED_STATE_REPLAY"
            if passed:
                result["summary"] = summaries(gate, cells)
                exit_code = 0
        result["cells"] = public_cells(cells, full_pass=exit_code == 0)
    except Exception as error:
        result.update(status="FAILED_REPLAY_OR_METADATA_GATE", summary=None, failure={"exception_type": type(error).__name__, "condition": str(error)}, cells=public_cells(cells, full_pass=False))
    finally:
        if accounting.cuda_started:
            try:
                import torch
                torch.cuda.synchronize(0)
                result["replay_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
                result["replay_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            except Exception as error:
                result["CUDA_accounting_failure"] = {"exception_type": type(error).__name__, "condition": str(error)}
                result["status"], result["summary"], exit_code = "FAILED_REPLAY_CUDA_ACCOUNTING", None, 1
                result["cells"] = public_cells(cells, full_pass=False)
        result["UTC"] = datetime.now(timezone.utc).isoformat()
        # Never serialize public PASS/metrics until terminal accounting succeeds.
        provisional = {**result, "status": "FINALIZATION_PENDING", "summary": None,
                       "cells": public_cells(cells, full_pass=False)}
        atomic_json(output / "REPLAY_RESULT.json", provisional)
        try:
            result["replay_accounting"] = accounting.finish(result["status"])
        except Exception as error:
            result["status"], result["summary"], exit_code = "FAILED_REPLAY_ACCOUNTING", None, 1
            result["cells"] = public_cells(cells, full_pass=False)
            result["accounting_failure"] = {"exception_type": type(error).__name__, "condition": str(error)}
            result["replay_accounting"] = {"observed_wall_lower_bound_seconds": perf_counter() - started,
                                           "inclusive_wall_seconds": None, "unknown_finalization_remainder": True}
            accounting.flush("FAILED_REPLAY_ACCOUNTING")
        result["replay_attempt_receipt"] = descriptor(output / "REPLAY_ATTEMPT.json")
        if (output / "PRIVATE_REPLAY_DETAILS.json").exists():
            result["private_replay_details_receipt"] = descriptor(output / "PRIVATE_REPLAY_DETAILS.json")
        atomic_json(output / "REPLAY_RESULT.json", result)
        os.close(fd)
    print("REPLAY_TERMINAL status=" + result["status"] + " denominator=25 receipt=" + str(output / "REPLAY_RESULT.json"), flush=True)
    return exit_code, result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root-release", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = perf_counter()
    context = preflight(args.root_release, args.output)
    code, _ = execute(context, started)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
