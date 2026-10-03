"""Durable attempts, phase timing, failures, and count-only live status."""
from pathlib import Path
from time import perf_counter
import json
from pilot_common import atomic_json, require, utc


class Attempts:
    def __init__(self, output, context, stage, unit, seed, *, started):
        self.output = Path(output)
        self.started = started
        self.phase_started = perf_counter()
        self.phase_name = "startup"
        path = self.output / "ATTEMPTS.json"
        self.ledger = json.loads(path.read_text()) if path.exists() else {
            "schema": "ncnc-pilot-inclusive-attempts-v1", "identity": context["identity"], "attempts": []}
        require(self.ledger["identity"] == context["identity"], "Attempt ledger identity differs")
        # A killed process can leave an unclosed attempt. Its observed time and
        # completed batches remain; unknown remainder is never called complete.
        for prior in self.ledger["attempts"]:
            if prior["status"] == "IN_PROGRESS":
                prior["status"] = "INTERRUPTED_UNCLOSED"
                prior["inclusive_wall_seconds"] = None
        self.row = {"attempt": len(self.ledger["attempts"]) + 1, "stage": stage, "unit": unit,
                    "base_seed": seed, "UTC_started": utc(), "status": "IN_PROGRESS", "phases": [],
                    "observed_wall_seconds": perf_counter() - started, "work": {},
                    "root_release_sha256": context["release_sha256"]}
        self.ledger["attempts"].append(self.row)
        self.flush()

    def flush(self):
        self.row["observed_wall_seconds"] = perf_counter() - self.started
        atomic_json(self.output / "ATTEMPTS.json", self.ledger)
        atomic_json(self.output / "STATUS.json", {"schema": "ncnc-pilot-count-only-status-v1",
            **{k: self.row[k] for k in ("attempt", "stage", "unit", "base_seed", "status", "work", "observed_wall_seconds")},
            "phase": self.phase_name, "UTC": utc(), "predictive_values_exposed": False})

    def phase(self, name, **work):
        now = perf_counter()
        self.row["phases"].append({"phase": self.phase_name, "wall_seconds": now - self.phase_started,
                                    "work_at_end": dict(self.row["work"])})
        self.phase_name, self.phase_started = name, now
        self.row["work"].update(work)
        self.flush()

    def progress(self, detail):
        self.row["work"].update(detail)
        self.flush()

    def finish(self, status, error=None):
        self.phase("final_accounting")
        self.row["status"] = status
        self.row["UTC_finished"] = utc()
        self.row["inclusive_wall_seconds"] = perf_counter() - self.started
        if error is not None:
            # Store the actual failure without dumping tensors or metric values.
            self.row["failure"] = {"exception_type": type(error).__name__, "condition": str(error),
                                    "at_phase": self.row["phases"][-1]["phase"]}
        self.flush()
        return self.receipt()

    def receipt(self):
        closed = [r for r in self.ledger["attempts"] if r.get("inclusive_wall_seconds") is not None]
        unknown = [r["attempt"] for r in self.ledger["attempts"] if r["status"] == "INTERRUPTED_UNCLOSED"]
        return {"attempts": len(self.ledger["attempts"]), "failed_or_interrupted_attempts": sum(r["status"] in ("FAILED", "INTERRUPTED_UNCLOSED") for r in self.ledger["attempts"]),
                "closed_attempt_wall_seconds": sum(r["inclusive_wall_seconds"] for r in closed),
                "observed_interrupted_wall_seconds": sum(r["observed_wall_seconds"] for r in self.ledger["attempts"] if r["status"] == "INTERRUPTED_UNCLOSED"),
                "unclosed_attempt_ids": unknown, "total_cost_exact": not unknown,
                "terminal_accounting_write_tail_measured": False,
                "scope": "all_attempts_setup_reads_hashes_transfers_training_VALID_selection_journals_and_finalization"}
