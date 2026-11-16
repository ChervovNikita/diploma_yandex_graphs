"""Combine node calibration with explicit transfer and auxiliary scenarios."""
import csv
import json
import math
from pathlib import Path
from .plan import DATASETS

ROOT = Path(__file__).resolve().parents[2]

def main():
    directory = ROOT / "ablation/campaign"
    plan = json.loads((directory / "plans/rebuttal.json").read_text())
    counts = json.loads((directory / "plans/rebuttal.counts.json").read_text())["scientific_fit_counts"]
    tasks = {task["id"]: task for task in plan["tasks"]}
    node = json.loads((directory / "evidence/node_capacity.json").read_text())
    def score_rate(identifier):
        return sum(tasks[name].get("estimated_seconds", 1500) / 7500
                   for name in tasks[identifier]["metadata"].get("fits", []))
    auxiliary_seconds = 0
    for task in tasks.values():
        meta = task["metadata"]
        action = meta["action"]
        if action == "budget_profile":
            rates = [tasks[name]["estimated_seconds"] / 7500 for name in meta["fits"]]
            auxiliary_seconds += (rates[0] + sum(rates[:4]) + rates[4]) * 120 * 1.25
        elif action in ("profile", "scaling_profile"):
            candidates = meta.get("candidates", [])
            score = meta.get("score") or candidates[len(candidates) // 2]
            auxiliary_seconds += score_rate(score) * (235 if action == "scaling_profile" else 250) * 1.25
        elif action in ("node_score", "diversity", "initialization"):
            auxiliary_seconds += 15 if action == "node_score" else 45
        elif action == "transfer_score" and task["kind"] == "gpu":
            auxiliary_seconds += 300  # Planning allowance per full transfer scorer.
    resources = dict(
        gpu="NVIDIA A100-SXM4-80GB", potential_training_fits=counts["total"],
        core_training_fits=counts["planned_core"], conditional_expansion_fits=counts["conditional_expansion"],
        nodes=node, auxiliary_allowance_gpu_hours=auxiliary_seconds / 3600,
        transfer_scoring_allowance_seconds_per_fit=300,
        transfer_scenarios_gpu_hours={"150epochs": 156, "1000epochs": 1040},
        concurrency_measurement=dict(one_process_updates_per_second=2.727,
            two_processes_updates_per_second=3.375, four_processes_updates_per_second=3.287,
            scope="one GAT Roman Empire depth3 workload, two-update engineering probe"),
        four_day_scenarios=[])
    for row in node["scenarios"]:
        hours = row["all_node"]["isolated_gpu_hours"] + 1040 + auxiliary_seconds / 3600
        resources["four_day_scenarios"].append(dict(
            average_node_updates=row["updates_per_fit"], assumed_transfer_epochs=1000,
            isolated_gpu_hours=hours,
            gpus_at_80pct_availability_and_1p2_throughput=math.ceil(hours / (96 * .8 * 1.2)),
            gpus_at_80pct_availability_no_concurrency_gain=math.ceil(hours / (96 * .8)),
            days_on_32gpus_with_1p2_gain=hours / (32 * 24 * .8 * 1.2)))
    resources["interpretation"] = (
        "Capacity scenarios from bounded engineering measurements, with interpolation "
        "and extrapolation. Stopping times and heterogeneous-workload concurrency remain unknown.")
    resources["storage_estimate"] = dict(
        preliminary_selected_node_weight_gib=329,
        preliminary_training_prediction_gib=49,
        planning_reservation_gib=1000,
        assumptions="Pre-budget-expansion parameter-count interpolation. Reserve 1TB with headroom for added scoring and retained unresolved optimizer states. Many final-cap hits can exceed the reservation.")
    (directory / "evidence/resource_estimate.json").write_text(json.dumps(resources, indent=2) + "\n")
    fields = ["id", "action", "stage", "dataset", "model", "method", "variant",
              "num_layers", "split", "priority", "memory_gb", "estimated_seconds", "dependency_count"]
    with (directory / "plans/tasks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in tasks.values():
            meta = task["metadata"]
            spec = meta.get("spec", {})
            row = {key: spec.get(key, meta.get(key, "")) for key in fields}
            row.update(id=task["id"], action=meta["action"], priority=task["priority"],
                memory_gb=task["resources"]["memory_gb"], estimated_seconds=task.get("estimated_seconds", ""),
                dependency_count=len(task.get("dependencies", [])) + len(task.get("optional_dependencies", [])))
            writer.writerow(row)
    print(json.dumps(resources["four_day_scenarios"][:3], indent=2))

if __name__ == "__main__":
    main()
