#!/usr/bin/env python3
"""Pure metadata recommendations for a successor dispatcher; never a launch gate.

JSON input mode is 'disk' or 'pair'. Disk inputs are current incremental bytes,
so already retained outputs and transferred files are not reserved twice.
"""
import argparse
import json
import math
from pathlib import Path


def nonnegative(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Nonnegative finite " + name + " required")
    return value


def disk_admission(state):
    keys = ("free_disk_bytes", "floor_bytes", "pending_transfer_net_bytes", "transfer_temporary_extra_bytes",
            "peer_growth_allowance_bytes", "current_campaign_retained_bytes", "campaign_retained_cap_bytes")
    for key in keys:
        nonnegative(state[key], key)
    rows = state["active"] + [state["candidate"]]
    unknown = [r["id"] for r in rows if r.get("additional_to_peak_bytes") is None or r.get("additional_retained_bytes") is None]
    if unknown:
        return {"recommendation": "HOLD_FOR_PROFILE", "unknown_profiles": unknown, "launch_admitted": False}
    for r in rows:
        for key in ("additional_to_peak_bytes", "additional_retained_bytes"):
            nonnegative(r[key], key)
        if r["additional_to_peak_bytes"] < r["additional_retained_bytes"]:
            raise ValueError("Incremental peak must cover retained growth")
    required = (state["floor_bytes"] + state["pending_transfer_net_bytes"] + state["transfer_temporary_extra_bytes"]
                + state["peer_growth_allowance_bytes"] + sum(r["additional_to_peak_bytes"] for r in rows))
    retained = state["current_campaign_retained_bytes"] + sum(r["additional_retained_bytes"] for r in rows)
    reasons = []
    if state["free_disk_bytes"] < required:
        reasons.append("insufficient free bytes for floor plus incremental peaks")
    if retained > state["campaign_retained_cap_bytes"]:
        reasons.append("campaign retained byte cap would be exceeded")
    return {"recommendation": "PAUSE_NEW_CLAIMS" if reasons else "CAPACITY_ONLY_PASS",
            "required_free_bytes": required, "free_after_peak_bytes": state["free_disk_bytes"] - required + state["floor_bytes"],
            "retained_after_active_and_candidate_bytes": retained, "reasons": reasons,
            "launch_admitted": False, "freshness_and_source_and_GPU_gates_external": True}


def pair_evaluation(state):
    if len(state["jobs"]) != 2:
        raise ValueError("Exactly two jobs required")
    normalized = []
    pair_seconds = [j["pair_seconds"] for j in state["jobs"]]
    if pair_seconds[0] != pair_seconds[1]:
        raise ValueError("Pair rates require the same concurrent measurement interval")
    for job in state["jobs"]:
        for key in ("solo_work", "solo_seconds", "pair_work", "pair_seconds"):
            if nonnegative(job[key], key) == 0:
                raise ValueError("Positive measured work and time required")
        solo_rate = job["solo_work"] / job["solo_seconds"]
        pair_rate = job["pair_work"] / job["pair_seconds"]
        normalized.append(pair_rate / solo_rate)
    # Sum the solo-equivalent progress rates for heterogeneous jobs. 1.0 is
    # serial capacity; 2.0 is the unattainable-with-contention ideal, not assumed.
    speedup = sum(normalized)
    correctness = all(state[k] is True for k in ("source_and_work_identical", "lifecycle_reused_or_passed",
                                                  "both_active_throughout_pair_window",
                                                  "resource_headroom_pass", "disk_pass", "peer_identity_stable"))
    passed = correctness and speedup >= state.get("minimum_speedup", 1.15) and min(normalized) >= state.get("minimum_per_job_ratio", 0.40)
    return {"recommendation": "PAIR_PROFILE_CANDIDATE" if passed else "KEEP_ONE_CHILD",
            "normalized_per_job_progress": normalized, "aggregate_solo_equivalent_speedup": speedup,
            "correctness_and_resource_checks": correctness, "launch_admitted": False,
            "scope": "Only tested source/profile pair, host, UUID, package and thread settings; no blanket campaign admission"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("disk", "pair"))
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    state = json.loads(args.input.read_text())
    print(json.dumps((disk_admission if args.mode == "disk" else pair_evaluation)(state), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
