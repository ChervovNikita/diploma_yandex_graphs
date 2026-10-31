"""Deterministic assignment of complete dependency groups to three GPUs."""
from __future__ import annotations

import hashlib

DEFAULT_WORKERS = ("allocation/gpu0", "server77/gpu0", "server77/gpu1")


def assign(groups, capacities=None):
    """Balance declared cost while keeping cached ancestors on the same host.

    Each group has id, estimated_seconds, maximum_cuda_bytes and a list of run
    IDs. Estimates are scheduling inputs, never measured performance claims.
    """
    capacities = capacities or {name: None for name in DEFAULT_WORKERS}
    loads = {name: 0. for name in capacities}
    seen = set()
    result = []
    ordered = sorted(groups, key=lambda g: (-g["estimated_seconds"], g["id"]))
    for group in ordered:
        if group["id"] in seen or group["estimated_seconds"] < 0:
            raise ValueError("Unique groups and nonnegative cost estimates required")
        seen.add(group["id"])
        eligible = [name for name, memory in capacities.items()
                    if memory is None or group["maximum_cuda_bytes"] <= memory]
        if not eligible:
            raise ValueError("No GPU can accommodate group: " + group["id"])
        worker = min(eligible, key=lambda name: (loads[name], name))
        loads[worker] += group["estimated_seconds"]
        result.append({**group, "gpu_assignment": worker})
    return {"groups": sorted(result, key=lambda row: row["id"]), "estimated_gpu_seconds": loads,
            "target_parallel_slots_per_gpu": 2, "estimates_are_measurements": False}


def stable_fallback(group_id):
    """Stable placement when no validated cost profile is available."""
    index = int(hashlib.sha256(group_id.encode()).hexdigest()[:16], 16) % len(DEFAULT_WORKERS)
    return DEFAULT_WORKERS[index]
