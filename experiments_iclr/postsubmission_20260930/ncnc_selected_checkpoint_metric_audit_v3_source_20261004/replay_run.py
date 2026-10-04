#!/usr/bin/env python3
"""Ordinary root-released all25 checkpoint/metric audit; zero updates, no retry."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import os
import tempfile
from replay_gate import preflight, family_gate, admitted_input_custody, require, ARMS, descriptor
from replay_contract import public_cells, summaries
from audit_contract import SlotLedger, PROFILE_CONTRACT, PASS, terminal


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


class Accounting:
    def __init__(self, context, started):
        self.context, self.started = context, started
        self.phase_name, self.phase_started = "startup", perf_counter()
        self.phases, self.cuda_started, self.ledger = [], False, SlotLedger()
        self.flush("IN_PROGRESS")

    def flush(self, status):
        shared = {"status": status, "phase": self.phase_name, "work": self.ledger.counts(), "observed_wall_seconds": perf_counter() - self.started}
        atomic_json(self.context["output"] / "STATUS.json", {"schema": "ncnc-selected-checkpoint-metric-count-status-v3", **shared, "predictive_values_exposed": False})
        atomic_json(self.context["output"] / "AUDIT_ATTEMPT.json", {"schema": "ncnc-selected-checkpoint-metric-inclusive-attempt-v3", **shared,
            "identity": self.context["identity"], "root_release_sha256": self.context["release_sha256"], "sidecar_manifest_sha256": self.context["sidecar_manifest_sha256"],
            "phases": self.phases, "slot_ledger": self.ledger.rows, "automatic_retry": False})

    def phase(self, name):
        now = perf_counter(); self.phases.append({"phase": self.phase_name, "wall_seconds": now - self.phase_started})
        self.phase_name, self.phase_started = name, now; self.flush("IN_PROGRESS")

    def progress(self): self.flush("IN_PROGRESS")

    def private(self, value): atomic_json(self.context["output"] / "PRIVATE_AUDIT_DETAILS.json", value)

    def finish(self, status):
        self.phase("terminal_accounting"); self.flush(status)
        return {"inclusive_wall_seconds_through_accounting_finish": perf_counter() - self.started, "phases": self.phases,
                "scope": "startup_admission_load_restore_all_reached_full_VALID_calls_CUDA_and_terminal_accounting",
                "post_accounting_custody_and_terminal_write_tail_measured": False, "training_updates": 0, "automatic_retry": False}


def final_guard(context, gate, *, expected_strict_profile=False):
    """Called after finish and receipts, immediately before all25 public serialization."""
    receipt = admitted_input_custody(context)
    after = family_gate(context)
    require(gate is not None and after["lock"] == gate["lock"], "Frozen full-family/journal/selector/selected-byte custody changed")
    if "loaded_tensor_guard" in context: context["loaded_tensor_guard"]()
    if expected_strict_profile:
        import torch
        from replay_numeric import profile_receipt
        profile_receipt(torch, "True2")
    return {**receipt, "family_journal_selected_bytes": "PASS", "loaded_typed_canonical_rows": "PASS" if "loaded_tensor_guard" in context else "not_loaded",
            "timing": "after_Accounting.finish_and_receipt_hashes_immediately_before_public_serialization",
            "terminal_write_tail_measured": False}


def execute(context, started, *, fabricated_data=None, hooks=None, synthetic_after_metadata=None, synthetic_after_accounting=None):
    require((fabricated_data is not None) == bool(context.get("synthetic_only")), "Separate admitted fabricated entry required")
    require(not (hooks or synthetic_after_metadata or synthetic_after_accounting) or context.get("synthetic_only") is True, "Ordinary supervisor has no failure injection")
    output = context["output"]; output.mkdir(parents=True, mode=0o700, exist_ok=False)
    import fcntl
    fd = os.open(output / ".RUN_LOCK", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    accounting = Accounting(context, started)
    gate, failures = None, []
    cells = [{"arm": arm, "base_seed": seed, "status": "NOT_ATTEMPTED_ADMISSION_GATE"} for arm in ARMS for seed in range(5)]
    result = {"schema": "ncnc-selected-checkpoint-metric-audit-result-v3", "identity": context["identity"], "sidecar_manifest_sha256": context["sidecar_manifest_sha256"],
              "root_release_sha256": context["release_sha256"], "family_lock": context["release"]["family_lock"], "profile_contract": PROFILE_CONTRACT,
              "denominator": {"units": 20, "unique_fits": 35, "original_optimizer_steps": 59500, "served_cells": 25, "scoring_snapshot_slots": 40,
                              "planned_original_scorer_slots": 120, "scheduled_served_metric_checks": 75, "conditional_I4_reference_mean_checks_maximum": 5},
              "old_exact_replay_qualification_status": "FAILED", "old_exact_replay_qualified": False, "old_failed_qualification_and_results_overwritten": False,
              "claim": "All25 authenticated selected checkpoints yield exact saved Hits50 under fixed strict True1/True2 and observed byte-identical repeats in this admitted invocation",
              "historical_raw_reference_evidence": "conditional on exact authenticated full-array digest; unavailable pools have no historical error claim",
              "engineering_128eps_acceptance_authority": False, "summary": None, "training_costs": None,
              "TEST_opened": False, "training_updates": 0, "new_checkpoint_selection": False, "no_success_only_subset_summary": True,
              "fabricated_inputs_only": bool(context.get("synthetic_only")),
              "denominator_scope": "authenticated_fabricated_metadata_only_not_training_execution" if context.get("synthetic_only") else "frozen_original_complete_family"}
    try:
        accounting.phase("complete_family_metadata_custody")
        gate = family_gate(context); cells = gate["cells"]; result["training_costs"] = gate["costs"]
        require(gate["full_success"], "Complete authenticated20/35/25 family required; no subset inference")
        if synthetic_after_metadata: synthetic_after_metadata(context)
        from replay_numeric import run
        cells, audit = run(context, gate, accounting, fabricated_data=fabricated_data, hooks=hooks)
        result["checkpoint_metric_audit"] = audit; failures.extend(audit["failures"])
    except Exception as error:
        accounting.ledger.stop("admission_or_core_exception")
        failures.append({"category": "FAILED_ADMISSION_OR_SELECTED_STATE", "exception_type": type(error).__name__, "condition": str(error)})
    finally:
        if accounting.cuda_started:
            try:
                import torch
                torch.cuda.synchronize(0)
                result["audit_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
                result["audit_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            except Exception as error:
                failures.append({"category": "FAILED_CUDA_ACCOUNTING", "exception_type": type(error).__name__, "condition": str(error)})
        result.update(status="FINALIZATION_PENDING", summary=None, cells=public_cells(cells, full_pass=False), work=accounting.ledger.counts())
        atomic_json(output / "AUDIT_RESULT.json", result)
        try:
            result["audit_accounting"] = accounting.finish(terminal(failures))
        except Exception as error:
            failures.append({"category": "FAILED_ACCOUNTING", "exception_type": type(error).__name__, "condition": str(error)})
            result["audit_accounting"] = {"observed_wall_lower_bound_seconds": perf_counter() - started, "inclusive_wall_seconds_through_accounting_finish": None,
                                          "unknown_finalization_remainder": True, "terminal_write_tail_measured": False}
        result["audit_attempt_receipt"] = descriptor(output / "AUDIT_ATTEMPT.json")
        if (output / "PRIVATE_AUDIT_DETAILS.json").exists(): result["private_audit_details_receipt"] = descriptor(output / "PRIVATE_AUDIT_DETAILS.json")
        result["UTC"] = datetime.now(timezone.utc).isoformat()
        if synthetic_after_accounting:
            try: synthetic_after_accounting(context)
            except Exception as error: failures.append({"category": "FAILED_FINAL_CUSTODY", "exception_type": type(error).__name__, "condition": str(error)})
        eligible = not failures and len(cells) == 25 and all(c["status"] == "PASS" for c in cells) and accounting.ledger.counts()["completed_validated"] == 120
        if not eligible and not failures: failures.append({"category": "FAILED_ADMISSION_OR_SELECTED_STATE", "condition": "Incomplete full25 terminal"})
        # Compute the all25 candidate privately before the final guard. The disk remains null.
        candidate_cells = public_cells(cells, full_pass=eligible)
        candidate_summary = summaries(gate, cells) if eligible else None
        try:
            result["final_input_custody"] = final_guard(context, gate, expected_strict_profile=eligible)
        except Exception as error:
            failures.append({"category": "FAILED_FINAL_CUSTODY", "exception_type": type(error).__name__, "condition": str(error)})
        # No source/data/runtime/accounting work follows this guard; only terminal payload assembly/write.
        result.update(status=terminal(failures), failures=failures,
                      cells=candidate_cells if not failures else public_cells(cells, full_pass=False),
                      summary=candidate_summary if not failures else None)
        atomic_json(output / "AUDIT_RESULT.json", result)
        os.close(fd)
    code = 0 if result["status"] == PASS else 1
    print("CHECKPOINT_METRIC_AUDIT_TERMINAL status=" + result["status"] + " denominator=25 receipt=" + str(output / "AUDIT_RESULT.json"), flush=True)
    return code, result


def main():
    parser = ArgumentParser(description=__doc__); parser.add_argument("--root-release", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); started = perf_counter()
    return execute(preflight(args.root_release, args.output), started)[0]


if __name__ == "__main__": raise SystemExit(main())
