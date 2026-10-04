"""Shared literal schedule, counts and engineering evidence; no numerical imports."""
from collections import Counter
from replay_gate import ARMS, require, HERE
import json

LABELS = ("False_anchor", "True1", "True2")
PASS = "ALL25_SELECTED_CHECKPOINT_METRIC_CONSISTENCY_PASS"
PROFILE_CONTRACT = {
    "labels": list(LABELS), "strict_candidates": ["True1", "True2"],
    "cublas_workspace_config": ":4096:8", "warn_only": False,
    "dtype": "torch.float32", "TF32": False, "autocast_cuda": False,
    "autocast_cpu": False, "float32_matmul_precision": "highest",
    "cudnn_benchmark": False, "cudnn_deterministic": False,
    "anchor_metric_acceptance_authority": False,
    "engineering_128eps_acceptance_authority": False,
}
TERMINAL_PRECEDENCE = (
    "FAILED_FINAL_CUSTODY", "FAILED_ACCOUNTING", "FAILED_CUDA_ACCOUNTING",
    "FAILED_ADMISSION_OR_SELECTED_STATE", "FAILED_SCORER_OR_SLOT_GUARD",
    "FAILED_I4_REFERENCE_METRIC", "FAILED_STRICT_REPEAT", "FAILED_STRICT_METRIC",
)


def schedule():
    rows = []
    for arm in ARMS:
        for seed in range(5):
            for member in range(4 if arm == ARMS[1] else 1):
                for label in LABELS:
                    route = "native64" if arm in ARMS[:2] else "native70" if arm == ARMS[4] else "private" if arm == ARMS[2] else "pooled_after_clamp"
                    rows.append({"slot": len(rows) + 1, "id": f"{arm}/seed{seed}/member{member}/{label}",
                                 "arm": arm, "base_seed": seed, "member": member, "label": label, "route": route})
    require(len(rows) == 120 and Counter(r["route"] for r in rows) == {"native64": 75, "native70": 15, "private": 15, "pooled_after_clamp": 15}, "Fixed120 route denominator differs")
    require(rows == json.loads((HERE / "SLOT_PLAN.json").read_text())["slots"], "Manifest-bound predeclared120 identities differ")
    return rows


class SlotLedger:
    def __init__(self):
        self.rows = [{**r, "attempted": False, "entered_original_scorer": False,
                      "returned": False, "completed_validated": False, "reason": "not_reached"} for r in schedule()]

    def event(self, slot, key):
        row = self.rows[slot - 1]
        require(key in ("attempted", "entered_original_scorer", "returned", "completed_validated") and not row[key], "Duplicate/unknown slot event; retries forbidden")
        required = {"entered_original_scorer": "attempted", "returned": "entered_original_scorer", "completed_validated": "returned"}
        require(key not in required or row[required[key]], "Slot event ordering differs")
        row[key], row["reason"] = True, "in_progress"
        if key == "completed_validated": row["reason"] = "complete_validated"

    def fail(self, slot, error):
        self.rows[slot - 1]["reason"] = type(error).__name__ + ": " + str(error)

    def stop(self, reason):
        for row in self.rows:
            if not row["attempted"]: row["reason"] = "not_attempted_terminal_stop: " + reason

    def counts(self):
        def count(rows):
            return {"planned": len(rows), **{k: sum(bool(r[k]) for r in rows) for k in ("attempted", "entered_original_scorer", "returned", "completed_validated")}}
        cells = {(r["arm"], r["base_seed"]) for r in self.rows}
        return {**count(self.rows), "cells_planned": 25,
                "cells_attempted": sum(any(r["attempted"] for r in self.rows if (r["arm"], r["base_seed"]) == c) for c in cells),
                "cells_completed_validated": sum(all(r["completed_validated"] for r in self.rows if (r["arm"], r["base_seed"]) == c) for c in cells),
                "by_profile": {label: count([r for r in self.rows if r["label"] == label]) for label in LABELS},
                "by_route": {route: count([r for r in self.rows if r["route"] == route]) for route in ("native64", "native70", "private", "pooled_after_clamp")}}


def terminal(failures):
    found = {r["category"] for r in failures}
    return next((name for name in TERMINAL_PRECEDENCE if name in found), PASS)


def sum_case_work(cases):
    keys = ("planned", "attempted", "entered_original_scorer", "returned", "completed_validated")
    return {"new_training_updates": 0, "fixture_creation_calls": 0,
            **{k: sum(r.get("work", {}).get(k, 0) for r in cases) for k in keys}}
