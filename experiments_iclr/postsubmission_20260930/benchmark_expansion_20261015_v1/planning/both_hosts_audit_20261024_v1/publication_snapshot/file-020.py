"""Audit frozen local plan copies and source-only dependency builders; no fits."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from queue_support import EXPECTED_COUNTS, EXPANSION, HERE, PLAN_SHAS, pin, read, require


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    proof = EXPANSION / "campaign_readable_coordinator_20261021_v2/proofs/b605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96"
    inventories = [read(proof / ("INVENTORY-" + host + ".json")) for host in ("allocation", "server51")]
    inventory_rows = [row for inventory in inventories for row in inventory["fits"]]
    index = {row["id"]: row for row in inventory_rows}
    require(len(index) == len(inventory_rows) == 6460, "Inventory IDs overlap or coverage differs")
    require(set(index) == set(inventories[0]["canonical_fit_ids"]) == set(inventories[1]["canonical_fit_ids"]), "Canonical fit union differs")
    resource = load(EXPANSION / "qualification/all_fit_resources_20261016_v1/resources.py", "_audit_resources")
    observed, workers, task_ids, dependency_count = {}, [], set(), 0
    for worker, expected_sha in PLAN_SHAS.items():
        path = HERE / "plans" / ("PLAN-" + worker.replace("/", "_") + ".json")
        require(pin(path)["sha256"] == expected_sha, "Plan hash changed")
        plan = read(path)
        tasks = plan["tasks"]
        local = {row["id"]: row for row in tasks}
        require(len(local) == len(tasks), "Duplicate task in one worker")
        require(not task_ids.intersection(local), "Task assigned to more than one worker")
        task_ids.update(local)
        fits = [row for row in tasks if row["kind"] == "fit"]
        exports = [row for row in tasks if row["kind"] == "aggregate_I4_exports"]
        require((len(fits), len(exports)) == EXPECTED_COUNTS[worker], "Exact task counts differ")
        seen = set()
        for task in tasks:
            require(task["worker"] == worker, "Wrong queue ownership")
            require(task["reservation"] == resource.reserve_for(task), "Reservation differs from frozen catalogue")
            require(all(dep in seen for dep in task.get("dependencies", [])), "Dependency is absent or not earlier in queue")
            seen.add(task["id"])
            if task["kind"] == "fit":
                smoke = index[task["id"]]
                for key in ("worker", "runtime_job_hash", "descriptor", "runtime_entrypoint"):
                    require(task[key] == smoke[key], "Fit differs from all-fit smoke inventory: " + key)
                observed[task["id"]] = task
            else:
                deps = [local[key] for key in task["dependencies"]]
                require(len(deps) == 4 and {row["body"] for row in deps} == {0, 1, 2, 3}, "I4 body set differs")
                require(all(row["kind"] == "fit" and row["scientific_method_id"] == "native_single"
                            and row["group_id"] == task["group_id"] for row in deps), "I4 mixes cohorts or methods")
                dependency_count += len(deps)
        workers.append({"worker": worker, "plan": pin(path), "fits": len(fits), "I4_export_tasks": len(exports),
            "sum_task_disk_reservations_GiB": sum(row["reservation"]["disk_bytes"] for row in tasks) / 1024**3,
            "reservation_sum_is_not_a_retained_storage_forecast": True})
    require(set(observed) == set(index), "Plans differ from complete 6460-fit inventory")
    closures = []
    for folder in ("link_expansion_ports_BUDDY_production_v4", "interaction_BUDDY_production_v5",
                   "context_affinity_BUDDY_port_v1", "buddy_private_bn_production_v5"):
        module = load(EXPANSION / folder / "session.py", "_audit_" + folder)
        manifest = pin(EXPANSION / folder / "MANIFEST.json")
        pins = module.bindings(EXPANSION.parent.parent, manifest["sha256"])
        if isinstance(pins, tuple):
            pins = pins[0]
        rejected = [{"path_relative_to_phase": str(Path(row["path"]).relative_to(EXPANSION.parent)),
                     "bytes": row["bytes"], "sha256": row["sha256"]}
                    for row in pins if Path(row["path"]).suffix not in (".py", ".json")]
        closures.append({"provider": folder, "source": pin(EXPANSION / folder / "session.py"),
                         "manifest": manifest, "verified_dependency_entries": len(pins),
                         "old_suffix_gate_rejections": rejected})
    result = {"schema": "future-queue-static-and-source-audit-v1", "status": "PASS-with-identified-reporting-defects",
        "workers": workers, "fits": len(observed), "total_tasks": len(task_ids), "I4_dependency_edges": dependency_count,
        "duplicate_task_assignments": 0, "cross_worker_dependencies": 0, "cyclic_or_later_dependencies": 0,
        "fit_smoke_identity_mismatches": 0, "reservation_mismatches": 0,
        "fit_runtime_families": dict(Counter(row["family"] for row in observed.values())),
        "frozen_original_reporting_defect": "dependency suffix rejects genuine documentation/data members at finalization",
        "reporting_layout_defect": "canonical collector paths differ from execution7 attempt output paths",
        "dependency_closures": closures, "new_training_updates": 0, "TEST_accessed": False,
        "scope": "Local immutable queue/source audit; runtime success and final fit quality are not established"}
    (HERE / "QUEUE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "fits", "total_tasks", "I4_dependency_edges", "duplicate_task_assignments", "fit_runtime_families")}, sort_keys=True))


if __name__ == "__main__":
    main()
