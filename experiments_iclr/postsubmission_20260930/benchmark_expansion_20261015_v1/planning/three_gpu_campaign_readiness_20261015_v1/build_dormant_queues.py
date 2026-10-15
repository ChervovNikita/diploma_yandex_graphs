#!/usr/bin/env python3
"""Read pinned campaign metadata and write dormant queue views; never dispatch."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCHEMA = "normalization-portable-canonical-plan-v1"
COPY_FIELDS = (
    "id", "kind", "cell_id", "family", "protocol_track", "scientific_method_id",
    "split", "cohort_seed", "body", "initialization_seed", "group_id",
    "canonical_runtime_job_hash", "runtime_job_hash", "science_contract_hash",
    "source_descriptor", "dependencies", "output_relative", "M1_exact_reuse",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_pin(path):
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def read_pinned(path, expected):
    raw = Path(path).read_bytes()
    require(len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"],
            "Input bytes/hash differ: " + str(path))
    return json.loads(raw)


def source_check(package, expected_sha):
    package = Path(package).resolve()
    manifest_path = package / "MANIFEST.json"
    require(file_pin(manifest_path)["sha256"] == expected_sha, "Source manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    paths = set()
    for row in manifest["files"]:
        path = (package / row["path"]).resolve()
        require(path.is_relative_to(package.resolve()) and path not in paths, "Invalid source manifest path")
        paths.add(path)
        actual = file_pin(path)
        require(all(actual[k] == row[k] for k in ("bytes", "sha256")), "Source member changed: " + row["path"])
    require({package / "build_dormant_queues.py", package / "TARGETS_AND_WEIGHTS.json"}.issubset(paths),
            "Builder/config missing from source closure")
    return file_pin(manifest_path)


def validate_config(config):
    require(config["schema"] == "dormant-three-gpu-queue-config-v1", "Wrong config schema")
    targets = config["targets"]
    require(len(targets) == 3 and len({t["worker"] for t in targets}) == 3, "Exactly three distinct workers required")
    require(len({t["physical_gpu_uuid"] for t in targets}) == 3, "Distinct physical GPU UUIDs required")
    require(all(t["worker"] == "allocation/gpu0" or t["worker"].startswith("server51/") for t in targets),
            "Only allocation and server51 belong in this queue plan")
    require(all(t["launch_admitted"] is False for t in targets), "Dormant targets cannot carry admission")
    weights = config["relative_work"]
    require(weights["measured_time"] is False and weights["seconds_per_unit"] is None, "Relative units are not seconds")
    require(type(weights["ticks_per_unit"]) is int and weights["ticks_per_unit"] > 0, "Positive integer scale required")
    require(all(type(weights[k]) is int and weights[k] > 0 for k in
                ("native_fit_ticks", "shared_fit_ticks", "export_ticks")), "Positive integer proxy weights required")
    require(all(t["relative_speed_assumption"] == 1 for t in targets), "No unmeasured hardware speed conversion")


def build_views(plan, config):
    """Pure metadata transform. All rows refer to, and do not modify, canonical tasks."""
    validate_config(config)
    before = digest(plan)
    require(plan.get("schema") == SCHEMA, "Wrong canonical plan schema")
    tasks = plan["tasks"]
    require(isinstance(tasks, list), "Tasks must be a list")
    index = {}
    outputs = set()
    for t in tasks:
        require(isinstance(t.get("id"), str) and t["id"] and t["id"] not in index, "Duplicate/invalid task ID")
        require(t.get("kind") in ("fit", "aggregate_I4_exports"), "Unexpected task kind")
        require(isinstance(t.get("dependencies"), list) and len(t["dependencies"]) == len(set(t["dependencies"])),
                "Invalid or duplicate dependencies")
        require(t.get("output_relative") and t["output_relative"] not in outputs, "Duplicate/missing canonical output")
        require(t.get("science_contract_hash"), "Missing scientific hash")
        if t["kind"] == "fit":
            require(t.get("runtime_job_hash") and t.get("source_descriptor", {}).get("sha256"), "Missing fit/source hash")
            require("split" in t and "cohort_seed" in t and t.get("cell_id"), "Fit lacks cohort key")
        index[t["id"]] = t
        outputs.add(t["output_relative"])
    counts = {"tasks": len(tasks), "fits": sum(t["kind"] == "fit" for t in tasks),
              "exports": sum(t["kind"] == "aggregate_I4_exports" for t in tasks)}
    require(counts == config["expected_counts"], "Canonical inventory counts changed")

    keys = {}
    for t in tasks:
        require(all(dep in index for dep in t["dependencies"]), "Unknown dependency")
        if t["kind"] == "fit":
            key = (t["cell_id"], t["split"], t["cohort_seed"])
            require(all(isinstance(v, (str, int)) and not isinstance(v, bool) for v in key), "Non-scalar cohort key")
            keys[t["id"]] = key
    for t in tasks:
        if t["kind"] != "aggregate_I4_exports":
            continue
        deps = [index[x] for x in t["dependencies"]]
        require(len(deps) == 4 and all(x["kind"] == "fit" and x.get("scientific_method_id") == "native_single" for x in deps),
                "I4 must depend on four native fits")
        require([x.get("body") for x in deps] == [0, 1, 2, 3], "Preserve ordered I4 body0..3 dependencies")
        dep_keys = {keys[x["id"]] for x in deps}
        require(len(dep_keys) == 1, "I4 dependencies cross comparison cohorts")
        key = next(iter(dep_keys))
        for field, value in zip(("cell_id", "split", "cohort_seed"), key):
            require(field not in t or t[field] == value, "Export cohort field disagrees with native dependencies")
        require(t.get("M1_exact_reuse") == deps[0]["id"], "M1 must reuse I4 body0")
        keys[t["id"]] = key

    groups = defaultdict(list)
    old_groups = {}
    for t in tasks:
        key = keys[t["id"]]
        require(all(keys[dep] == key for dep in t["dependencies"]), "Dependency crosses merged cohort")
        require(old_groups.setdefault(t["group_id"], key) == key, "Canonical group split by merged cohort key")
        groups[key].append(t)

    weights = config["relative_work"]
    scale = weights["ticks_per_unit"]
    def ticks(t):
        if t["kind"] == "aggregate_I4_exports":
            return weights["export_ticks"]
        return weights["native_fit_ticks"] if t["scientific_method_id"] == "native_single" else weights["shared_fit_ticks"]

    def topological(rows):
        local = {t["id"]: t for t in rows}
        indegree = {t["id"]: len(t["dependencies"]) for t in rows}
        children = defaultdict(list)
        for t in rows:
            for dep in t["dependencies"]:
                children[dep].append(t["id"])
        ready = [tid for tid, n in indegree.items() if n == 0]
        heapq.heapify(ready)
        result = []
        while ready:
            tid = heapq.heappop(ready)
            result.append(local[tid])
            for child in sorted(children[tid]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        require(len(result) == len(rows), "Dependency cycle")
        return result

    ordered_groups = {key: topological(rows) for key, rows in groups.items()}
    group_ticks = {key: sum(ticks(t) for t in rows) for key, rows in groups.items()}
    strata = defaultdict(list)
    for key in groups:
        strata[key[0]].append(key)
    workers = [t["worker"] for t in config["targets"]]
    loads = {w: 0 for w in workers}
    group_counts = Counter()
    queues = {w: [] for w in workers}
    assignment = {}
    strata_summary = []
    # Every cell/profile gets one turn per worker in each three-cohort round.
    # Greedy selection within a round uses current global load, not wall seconds.
    for cell in sorted(strata, key=lambda c: (-sum(group_ticks[k] for k in strata[c]), c)):
        cell_keys = sorted(strata[cell], key=lambda k: (-group_ticks[k], canonical(list(k))))
        per_worker = Counter()
        eligible = set(workers)
        for key in cell_keys:
            if not eligible:
                eligible = set(workers)
            worker = min(eligible, key=lambda w: (loads[w], group_counts[w], workers.index(w)))
            eligible.remove(worker)
            assignment[key] = worker
            loads[worker] += group_ticks[key]
            group_counts[worker] += 1
            per_worker[worker] += 1
            for t in ordered_groups[key]:
                queues[worker].append({"canonical_task": {k: t[k] for k in COPY_FIELDS if k in t},
                                       "canonical_task_sha256": digest(t), "canonical_worker": t.get("worker"),
                                       "comparison_cohort": list(key), "relative_work_ticks": ticks(t)})
        distribution = {w: per_worker[w] for w in workers}
        require(max(distribution.values()) - min(distribution.values()) <= 1, "Cell stratification imbalance")
        strata_summary.append({"cell_id": cell, "comparison_cohorts": len(cell_keys), "cohorts_by_worker": distribution})

    all_rows = [r for rows in queues.values() for r in rows]
    require(len(all_rows) == len(index) and {r["canonical_task"]["id"] for r in all_rows} == set(index), "Queue coverage differs")
    for worker, rows in queues.items():
        seen = set()
        for row in rows:
            t = row["canonical_task"]
            require(all(d in seen for d in t["dependencies"]), "Dependency not earlier on same worker")
            require(row["canonical_task_sha256"] == digest(index[t["id"]]), "Canonical task digest changed")
            require(all(t[k] == index[t["id"]][k] for k in t), "Canonical field changed")
            seen.add(t["id"])
    require(digest(plan) == before, "Input plan mutated")
    by_worker = []
    for target in config["targets"]:
        worker = target["worker"]
        rows = queues[worker]
        by_worker.append({"worker": worker, "tasks": len(rows),
                          "fits": sum(r["canonical_task"]["kind"] == "fit" for r in rows),
                          "exports": sum(r["canonical_task"]["kind"] != "fit" for r in rows),
                          "comparison_cohorts": group_counts[worker], "relative_work_ticks": loads[worker],
                          "relative_work_units": loads[worker] / scale, "estimated_seconds": None})
    summary = {"schema": "dormant-three-gpu-queue-summary-v1", "status": "dormant_planning_views_only",
               "launch_admitted": False, "claims_created": 0, "children_started": 0,
               "canonical_counts": counts, "canonical_groups": len(old_groups), "comparison_cohorts": len(groups),
               "grouping": ["cell_id", "split", "cohort_seed"], "stratification": "cell_id (canonical profile)",
               "algorithm": "descending cell-total work; descending cohort work; every three cohorts within cell visit each worker once; choose lowest current total among eligible workers",
               "workers": by_worker, "strata": strata_summary, "targets": config["targets"],
               "relative_work": weights, "total_relative_work_units": sum(loads.values()) / scale,
               "relative_work_spread_units": (max(loads.values()) - min(loads.values())) / scale,
               "max_to_min_relative_work_ratio": max(loads.values()) / min(loads.values()) if min(loads.values()) else None,
               "time_balance_or_ETA_claim": False, "runtime_descriptors_rewritten": False,
               "canonical_ownership_changed": False, "canonical_input_mutated": False,
               "coverage_and_dependency_checks": "PASS", "input_order_independent": True,
               "release_readiness": "not created; destination source/reporting/resource/custody gates remain separate"}
    return queues, summary


def generate(plan_path, output, config, source_pin):
    plan_path, output = Path(plan_path), Path(output)
    require(not output.exists(), "Output must be a fresh directory")
    plan = read_pinned(plan_path, config["canonical_plan"])
    queues, summary = build_views(plan, config)
    summary["canonical_plan"] = file_pin(plan_path)
    summary["builder_source_manifest"] = source_pin
    blobs = {}
    for target in config["targets"]:
        worker = target["worker"]
        filename = "QUEUE_" + worker.replace("/", "_") + ".json"
        blobs[filename] = canonical({"schema": "dormant-three-gpu-queue-v1", "worker": worker,
                                     "target": target, "launch_admitted": False, "dispatch_supported": False,
                                     "canonical_plan": summary["canonical_plan"], "tasks": queues[worker]}) + b"\n"
    summary["queue_files"] = [{"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                              for name, raw in sorted(blobs.items())]
    blobs["SUMMARY.json"] = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    receipts = [{"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                for name, raw in sorted(blobs.items())]
    blobs["MANIFEST.json"] = json.dumps({"schema": "dormant-three-gpu-output-manifest-v1", "files": receipts,
                                        "source_manifest": source_pin, "canonical_plan": summary["canonical_plan"],
                                        "launch_admitted": False}, indent=2, sort_keys=True).encode() + b"\n"
    # All semantic validation completes before creating the output directory.
    output.mkdir()
    try:
        for name, raw in blobs.items():
            with (output / name).open("xb") as stream:
                stream.write(raw)
    except BaseException:
        with (output / "FAILED.json").open("x") as stream:
            json.dump({"status": "partial_write_failed", "launch_admitted": False}, stream)
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-plan", help="Optional local copy of the exact pinned canonical JSON")
    parser.add_argument("--output", required=True, help="Fresh dormant-output directory; existing directories refuse")
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    source_pin = source_check(HERE, args.source_manifest_sha256)
    config = json.loads((HERE / "TARGETS_AND_WEIGHTS.json").read_text())
    summary = generate(args.canonical_plan or config["canonical_plan"]["path"], args.output, config, source_pin)
    print(json.dumps({"status": summary["status"], "workers": summary["workers"],
                      "relative_work_spread_units": summary["relative_work_spread_units"],
                      "time_balance_or_ETA_claim": False, "output": str(Path(args.output).resolve())}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit("DORMANT_QUEUE_PREPARATION_REFUSED: " + str(exc))
