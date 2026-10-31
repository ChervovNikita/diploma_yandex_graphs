"""Independent integrity check and anonymous table for the optional profile.

This reads only the already locked selection and measured resource profile.
It neither trains models nor evaluates labels. The raw profile, which contains
GPU context including UUIDs, remains an author-only file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
ROUNDS = 3
REPETITIONS = 100
PROFILE_SHA = "427b828a8be63526922e1f1a98d1f4d5ca5c09bf95100388b1364e8cf1ec90d2"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise RuntimeError(explanation)


def near(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def main() -> None:
    source = ROOT / "profile_selected_inference.py"
    profile_file = ROOT / "INFERENCE_PROFILE.json"
    lock_file = ROOT / "VALIDATION_SELECTION_LOCK.json"
    frozen_file = ROOT / "FROZEN_STUDY.json"
    for path in (source, profile_file, lock_file, frozen_file):
        require(path.is_file(), f"Missing {path.name}")
    require(digest(source) == PROFILE_SHA, "Profiler source differs from predeclared script")
    profile = json.loads(profile_file.read_text())
    lock = json.loads(lock_file.read_text())
    frozen = json.loads(frozen_file.read_text())
    require(profile["protocol"] == "selected_inference_profile_v1", "Profile protocol differs")
    require(profile["source_sha256"] == PROFILE_SHA, "Profile source SHA differs")
    require(profile["freeze_sha256"] == digest(frozen_file) == lock["freeze_sha256"],
            "Study freeze differs")
    require(profile["selection_lock_sha256"] == digest(lock_file), "Selection lock differs")
    require(len(lock["cells"]) == 432 and len(lock["selections"]) == 4,
            "Incomplete global validation lock")
    require(tuple(frozen["matrix"]["datasets"]) == DATASETS and
            tuple(frozen["matrix"]["arms"]) == ARMS, "Frozen graph or arm set differs")
    require(set(profile["graphs"]) == set(DATASETS), "Profile graph set differs")
    require(profile["optimization_seed"] == 0 and profile["rounds"] == ROUNDS and
            profile["warmups_per_block"] == 20 and
            profile["repetitions_per_block"] == REPETITIONS,
            "Profile measurement schedule differs")
    require("A100" in profile["device"], "Expected common A100 device")

    rows = []
    for dataset in DATASETS:
        graph = profile["graphs"][dataset]
        require(graph["descriptor"] == frozen["graphs"][dataset]["descriptor"],
                f"Graph descriptor differs: {dataset}")
        require(set(graph["arms"]) == set(ARMS), f"Arm set differs: {dataset}")
        for arm in ARMS:
            rec = graph["arms"][arm]
            selected = lock["selections"][dataset][arm]["selected_candidate"]
            require(rec["selected_candidate"] == selected, f"Selection differs: {dataset}/{arm}")
            lr, wd = selected
            key = f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed0"
            checkpoint = ROOT / "results" / key / "checkpoint.pt"
            require(checkpoint.is_file(), f"Missing selected checkpoint: {key}")
            checkpoint_sha = digest(checkpoint)
            require(rec["checkpoint_sha256"] == checkpoint_sha ==
                    lock["cells"][key]["checkpoint_sha256"],
                    f"Selected checkpoint differs: {key}")
            result_path = ROOT / "results" / key / "result.json"
            require(digest(result_path) == lock["cells"][key]["result_sha256"],
                    f"Selected training result differs: {key}")
            trained = json.loads(result_path.read_text())
            require(len(rec["blocks"]) == ROUNDS, f"Block count differs: {dataset}/{arm}")
            vals = []
            peaks = []
            increments = []
            block_medians = []
            for rnd, block in enumerate(rec["blocks"]):
                order = list(ARMS)
                if rnd == 1:
                    order.reverse()
                elif rnd == 2:
                    order = order[3:] + order[:3]
                require(block["round"] == rnd and block["position"] == order.index(arm),
                        f"Block order differs: {dataset}/{arm}/{rnd}")
                measured = block["latency_ms"]
                require(len(measured) == REPETITIONS and all(
                    isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in measured),
                    f"Invalid latency samples: {dataset}/{arm}/{rnd}")
                idle = block["idle_allocated_bytes"]
                peak = block["peak_allocated_bytes"]
                increment = block["peak_increment_bytes"]
                require(all(isinstance(v, int) for v in (idle, peak, increment)) and
                        0 <= idle <= peak and increment == peak - idle,
                        f"Invalid allocated memory: {dataset}/{arm}/{rnd}")
                vals.extend(measured)
                peaks.append(peak)
                increments.append(increment)
                block_medians.append(statistics.median(measured))
            q25, q75 = np.quantile(vals, [.25, .75]).tolist()
            require(near(rec["latency_ms_median"], statistics.median(vals)) and
                    near(rec["latency_ms_q25"], q25) and
                    near(rec["latency_ms_q75"], q75) and
                    all(near(a, b) for a, b in zip(
                        rec["block_latency_medians_ms"], block_medians)) and
                    len(rec["block_latency_medians_ms"]) == ROUNDS,
                    f"Latency summaries differ: {dataset}/{arm}")
            require(rec["peak_allocated_bytes_max"] == max(peaks) and
                    rec["peak_increment_bytes_max"] == max(increments),
                    f"Memory summaries differ: {dataset}/{arm}")
            require(isinstance(rec["parameter_count"], int) and rec["parameter_count"] > 0 and
                    isinstance(rec["parameter_bytes"], int) and rec["parameter_bytes"] > 0,
                    f"Invalid parameter count: {dataset}/{arm}")
            require(rec["parameter_count"] == trained["initialization"]["parameter_count"] and
                    rec["parameter_bytes"] ==
                    trained["initialization"]["parameter_storage_bytes"],
                    f"Profiled parameter count differs from training: {dataset}/{arm}")
            rows.append({
                "dataset": dataset, "arm": arm, "selected_lr": lr,
                "selected_weight_decay": wd, "seed": 0,
                "parameter_count": rec["parameter_count"],
                "parameter_bytes": rec["parameter_bytes"],
                "latency_ms_median": rec["latency_ms_median"],
                "latency_ms_q25": rec["latency_ms_q25"],
                "latency_ms_q75": rec["latency_ms_q75"],
                "peak_allocated_bytes_max": max(peaks),
                "peak_increment_bytes_max": max(increments),
            })
    require(len(rows) == 24, "Expected exactly 24 profiled graph/arm pairs")
    csv_file = ROOT / "INFERENCE_PROFILE_ANONYMOUS_TABLE.csv"
    audit_file = ROOT / "INFERENCE_PROFILE_AUDIT.json"
    require(not csv_file.exists() and not audit_file.exists(), "Refusing to overwrite prior audit")
    with csv_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "status": "PASS", "source_sha256": PROFILE_SHA,
        "raw_profile_sha256": digest(profile_file),
        "selection_lock_sha256": digest(lock_file),
        "anonymous_table_sha256": digest(csv_file),
        "scope": "24 validation-selected seed-0 graph/arm profiles, 300 timings each",
        "raw_gpu_context_author_only": True,
        "rows": len(rows), "samples": len(rows) * ROUNDS * REPETITIONS,
    }
    audit_file.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
