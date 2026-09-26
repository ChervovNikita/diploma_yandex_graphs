"""Recompute all selected-checkpoint timing summaries from anonymous samples.

This public verifier needs NumPy but no GPU, raw graph data, or checkpoint.
The full GPU context and original raw profile are deliberately absent.
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
ALLOWED = {
    "PUBLIC_PROFILE_SAMPLES.json", "PUBLIC_PROFILE_MANIFEST.json",
    "INFERENCE_PROFILE_ANONYMOUS_TABLE.csv", "INFERENCE_PROFILE_AUDIT.json",
    "INFERENCE_PROFILE_RUNTIME_ANONYMOUS.json", "profile_selected_inference.py",
    "VALIDATION_SELECTION_LOCK.json", "FROZEN_STUDY.json",
    "project_inference_profile_public.py", "verify_inference_profile_public.py",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def near(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def main() -> None:
    actual_files = {p.name for p in ROOT.iterdir() if p.is_file()}
    require(actual_files == ALLOWED, "Public profile file set differs")
    require({p.name for p in ROOT.iterdir() if p.is_dir()} == {"selected_results"},
            "Unexpected public profile subdirectory")
    read = lambda name: json.loads((ROOT / name).read_text())
    manifest = read("PUBLIC_PROFILE_MANIFEST.json")
    samples = read("PUBLIC_PROFILE_SAMPLES.json")
    audit = read("INFERENCE_PROFILE_AUDIT.json")
    runtime = read("INFERENCE_PROFILE_RUNTIME_ANONYMOUS.json")
    lock = read("VALIDATION_SELECTION_LOCK.json")
    frozen = read("FROZEN_STUDY.json")
    require(manifest["protocol"] == samples["protocol"] ==
            "selected_inference_profile_public_v1" and
            manifest["raw_profile_sha256"] == samples["raw_profile_sha256"] ==
            audit["raw_profile_sha256"] and audit["status"] == "PASS" and
            manifest["selection_lock_sha256"] ==
            samples["selection_lock_sha256"] == audit["selection_lock_sha256"] ==
            sha(ROOT / "VALIDATION_SELECTION_LOCK.json") and
            manifest["frozen_study_sha256"] == samples["freeze_sha256"] ==
            lock["freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            manifest["anonymous_table_sha256"] == audit["anonymous_table_sha256"] ==
            sha(ROOT / "INFERENCE_PROFILE_ANONYMOUS_TABLE.csv") and
            manifest["audit_sha256"] == sha(ROOT / "INFERENCE_PROFILE_AUDIT.json") and
            manifest["samples_sha256"] == sha(ROOT / "PUBLIC_PROFILE_SAMPLES.json") and
            manifest["projection_source_sha256"] ==
            samples["projection_source_sha256"] ==
            sha(ROOT / "project_inference_profile_public.py") and
            manifest["verifier_source_sha256"] ==
            sha(ROOT / "verify_inference_profile_public.py") and
            manifest["profiler_source_sha256"] ==
            samples["profiler_source_sha256"] == audit["source_sha256"] ==
            sha(ROOT / "profile_selected_inference.py") and
            manifest["runtime_sha256"] ==
            sha(ROOT / "INFERENCE_PROFILE_RUNTIME_ANONYMOUS.json") and
            runtime["raw_profile_sha256"] == audit["raw_profile_sha256"],
            "Public profile manifest, lock, source, or table binding differs")
    require(set(runtime) == {"protocol", "raw_profile_sha256", "python", "torch",
                             "pyg", "cuda", "device_class", "cpu_threads",
                             "memory_measurement", "timing_measurement"} and
            runtime["protocol"] == "selected_inference_profile_runtime_public_v1" and
            "A100" in runtime["device_class"] and
            isinstance(runtime["cpu_threads"], int) and runtime["cpu_threads"] > 0 and
            all(isinstance(runtime[name], str) and runtime[name]
                for name in ("python", "torch", "pyg", "cuda")) and
            all("GPU-" not in str(value) and "/" not in str(value)
                for value in runtime.values() if isinstance(value, str)),
            "Anonymous runtime version/class record differs")
    require(lock["protocol"] == frozen["protocol"] ==
            "validation_tuning_sensitivity_v1" and
            len(lock["cells"]) == 432 and
            tuple(frozen["matrix"]["datasets"]) == DATASETS and
            tuple(frozen["matrix"]["arms"]) == ARMS and
            samples["optimization_seed"] == 0 and
            samples["rounds"] == 3 and samples["warmups_per_block"] == 20 and
            samples["repetitions_per_block"] == 100 and
            audit["rows"] == 24 and audit["samples"] == 7200 and
            manifest["rows"] == 24 and manifest["timings"] == 7200 and
            set(samples["graphs"]) == set(DATASETS),
            "Frozen experiment matrix or public timing schedule differs")
    require(set(samples) == {"protocol", "raw_profile_sha256",
                             "projection_source_sha256", "profiler_source_sha256",
                             "selection_lock_sha256", "freeze_sha256",
                             "optimization_seed", "rounds", "warmups_per_block",
                             "repetitions_per_block", "graphs"},
            "Unexpected public profile metadata field")
    require(samples["profiler_source_sha256"] == audit["source_sha256"] ==
            "427b828a8be63526922e1f1a98d1f4d5ca5c09bf95100388b1364e8cf1ec90d2",
            "Original profiler source differs")

    with (ROOT / "INFERENCE_PROFILE_ANONYMOUS_TABLE.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = ("dataset", "arm", "selected_lr", "selected_weight_decay", "seed",
                  "parameter_count", "parameter_bytes", "latency_ms_median",
                  "latency_ms_q25", "latency_ms_q75", "peak_allocated_bytes_max",
                  "peak_increment_bytes_max")
        require(tuple(reader.fieldnames or ()) == fields, "Anonymous table schema differs")
        rows = list(reader)
    require(len(rows) == 24 and {(r["dataset"], r["arm"]) for r in rows} ==
            {(d, a) for d in DATASETS for a in ARMS},
            "Missing or duplicate graph/arm table row")
    table = {(r["dataset"], r["arm"]): r for r in rows}
    n_timings = 0
    for dataset in DATASETS:
        graph = samples["graphs"][dataset]
        require(set(graph) == {"arms"} and set(graph["arms"]) == set(ARMS),
                f"Unexpected graph profile fields: {dataset}")
        for arm in ARMS:
            rec = graph["arms"][arm]
            require(set(rec) == {"selected_candidate", "checkpoint_sha256",
                                 "parameter_count", "parameter_bytes", "blocks"},
                    f"Unexpected arm profile fields: {dataset}/{arm}")
            selected = lock["selections"][dataset][arm]["selected_candidate"]
            require(rec["selected_candidate"] == selected and len(rec["blocks"]) == 3,
                    f"Selection or block count differs: {dataset}/{arm}")
            lr, wd = selected
            key = f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed0"
            selected_cell = lock["cells"][key]
            result_path = ROOT / "selected_results" / dataset / arm / "result.json"
            require(result_path.is_file() and
                    sha(result_path) == selected_cell["result_sha256"] ==
                    manifest["selected_result_sha256"][key],
                    f"Selected training result differs from lock: {dataset}/{arm}")
            result = json.loads(result_path.read_text())
            require(result["protocol"] == lock["protocol"] and
                    result["freeze_sha256"] == lock["freeze_sha256"] and
                    (result["dataset"], result["arm"], result["seed"]) ==
                    (dataset, arm, 0) and
                    (result["lr"], result["weight_decay"]) == (lr, wd) and
                    result["checkpoint_sha256"] == selected_cell["checkpoint_sha256"] ==
                    rec["checkpoint_sha256"] and
                    rec["parameter_count"] ==
                    result["initialization"]["parameter_count"] and
                    rec["parameter_bytes"] ==
                    result["initialization"]["parameter_storage_bytes"],
                    f"Selected checkpoint/parameter identity differs: {dataset}/{arm}")
            durations, peaks, increments = [], [], []
            for rnd, block in enumerate(rec["blocks"]):
                require(set(block) == {"round", "position", "latency_ms",
                                       "idle_allocated_bytes", "peak_allocated_bytes",
                                       "peak_increment_bytes"},
                        f"Unexpected timing block field: {dataset}/{arm}/{rnd}")
                order = list(ARMS)
                if rnd == 1:
                    order.reverse()
                elif rnd == 2:
                    order = order[3:] + order[:3]
                vals = block["latency_ms"]
                idle, peak, increment = (block[k] for k in
                                         ("idle_allocated_bytes", "peak_allocated_bytes",
                                          "peak_increment_bytes"))
                require(block["round"] == rnd and block["position"] == order.index(arm) and
                        len(vals) == 100 and
                        all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                            for x in vals) and
                        all(isinstance(x, int) for x in (idle, peak, increment)) and
                        0 <= idle <= peak and increment == peak - idle,
                        f"Invalid timing or allocation block: {dataset}/{arm}/{rnd}")
                durations.extend(vals)
                peaks.append(peak)
                increments.append(increment)
            n_timings += len(durations)
            q25, q75 = np.quantile(durations, [.25, .75]).tolist()
            row = table[(dataset, arm)]
            require(int(row["seed"]) == 0 and
                    near(row["selected_lr"], lr) and near(row["selected_weight_decay"], wd) and
                    int(row["parameter_count"]) == rec["parameter_count"] and
                    int(row["parameter_bytes"]) == rec["parameter_bytes"] and
                    near(row["latency_ms_median"], statistics.median(durations)) and
                    near(row["latency_ms_q25"], q25) and
                    near(row["latency_ms_q75"], q75) and
                    int(row["peak_allocated_bytes_max"]) == max(peaks) and
                    int(row["peak_increment_bytes_max"]) == max(increments),
                    f"Anonymous timing/allocation summary differs: {dataset}/{arm}")
    require(n_timings == 7200, "Not all 7,200 fixed timing samples retained")
    result_files = {str(p.relative_to(ROOT)) for p in
                    (ROOT / "selected_results").rglob("*") if p.is_file()}
    require(result_files == {f"selected_results/{d}/{a}/result.json"
                             for d in DATASETS for a in ARMS} and
            len(manifest["selected_result_sha256"]) == 24,
            "Selected result provenance set differs")
    print(json.dumps({"status": "PASS", "rows": 24, "timings": n_timings,
                      "scope": "Exact retained timing samples, per-block allocations, selected checkpoint/count/lock binding; original raw GPU context omitted"},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
