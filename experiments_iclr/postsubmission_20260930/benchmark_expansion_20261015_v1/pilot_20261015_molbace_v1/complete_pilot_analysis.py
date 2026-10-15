"""Summarize the frozen 24-fit development pilot without a model or TEST read.

Selected exports were already scored by the frozen analysis watcher. This script
verifies its result authorities and audits all complete training curves. It does
not change checkpoints, readouts, training or candidate selection.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


LABELS = {
    "M1": "Native single",
    "I4": "Independent ensemble",
    "gnnm_boundary_native": "Base GNNM",
    "gnnm_graph_alignment_variance": "Alignment + variance",
    "gnnm_private_input_residual": "Private input correction",
    "gnnm_input_residual_plus_graph_alignment_variance": "Both ingredients",
}


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def summary(values):
    return {"values": values, "mean": statistics.mean(values), "sample_SD": statistics.stdev(values)}


def run(expansion):
    root = Path(expansion).resolve()
    pilot = root / "results/molbace_pilot_v1"
    output = root / "analysis_outputs/molbace_complete_pilot_v1"
    terminal_path = pilot / "QUEUE_RESULT.json"
    comparison_path = pilot / "selected_analysis_v1/COMPARISON.json"
    costs_path = pilot / "COSTS.json"
    terminal, comparison, costs = [json.loads(p.read_text()) for p in (terminal_path, comparison_path, costs_path)]
    assert terminal["status"] == "complete" and len(terminal["completed"]) == 24 and not terminal["failed"]
    assert comparison["complete_paired_trials"] == comparison["planned_paired_trials"] == 3
    assert all(d["TEST_accessed"] is False for d in (terminal, comparison, costs))
    assert costs["complete_fits"] == 24 and costs["failed_fits"] == 0
    sources = [pin(p) for p in (terminal_path, comparison_path, costs_path)]
    records, curves, fit_summaries = [], [], []
    for cohort in comparison["cohorts"]:
        seed = int(cohort["cohort"].rsplit("seed-", 1)[1])
        for name, metrics in [("M1", cohort["M1"]), ("I4", cohort["I4"])] + [
                (r["method"], r["metrics"]) for r in cohort["methods"]]:
            assert metrics["observed_graphs"] == 151 and metrics["missing_labels"] == 0
            records.append({"seed": seed, "method": name, "VALID_AUC_percent": 100 * metrics["rocauc"],
                            "VALID_BCE": metrics["BCE"], "VALID_Brier": metrics["Brier"]})
        for authority in cohort["authorities"]:
            result_path = Path(authority["path"]).resolve()
            assert result_path.is_relative_to(pilot / "fits") and pin(result_path) == authority
            result = json.loads(result_path.read_text())
            curve_path = result_path.with_name("CURVE.jsonl")
            events = [json.loads(line) for line in curve_path.read_text().splitlines()]
            assert result["status"] == "complete" and result["TEST_accessed"] is False
            assert len(events) == result["completed_epochs"]
            assert [e["train"]["epoch"] for e in events] == list(range(1, len(events) + 1))
            for event in events:
                train, valid = event["train"], event["validation"]
                assert train["complete_native_TRAIN"] and train["TRAIN_records_traversed"] == 1210
                assert valid["complete_source_VALID"] and valid["records"] == 151 and not valid["TEST_accessed"]
                curves.append({"seed": seed, "job_id": result["job_id"], "procedure": result["procedure"],
                               "body": result["body"], "epoch": train["epoch"],
                               "VALID_AUC_percent": 100 * valid["metrics"]["rocauc"], "VALID_BCE": valid["loss"],
                               "mean_update_TRAIN_loss": train["mean_update_loss"],
                               "epoch_wall_seconds": event["epoch_wall_seconds"]})
            best = max(events, key=lambda e: e["validation"]["metrics"]["rocauc"])
            assert best["train"]["epoch"] == result["selected_epoch"]
            assert best["validation"]["metrics"]["rocauc"] == result["best_validation_metric"]
            tail = events[-20:]
            preceding = events[-40:-20]
            fit_summaries.append({
                "job_id": result["job_id"], "seed": seed, "procedure": result["procedure"], "body": result["body"],
                "trained_epochs": len(events), "selected_epoch": result["selected_epoch"],
                "stop_reason": result["stop_reason"], "last_selection": events[-1]["selection"],
                "selected_AUC_percent": 100 * result["best_validation_metric"],
                "last20_best_AUC_percent": 100 * max(e["validation"]["metrics"]["rocauc"] for e in tail),
                "last20_mean_TRAIN_update_loss": statistics.mean(e["train"]["mean_update_loss"] for e in tail),
                "preceding20_mean_TRAIN_update_loss": statistics.mean(e["train"]["mean_update_loss"] for e in preceding),
                "invocation_wall_seconds": result["invocation_wall_seconds"], "resources": result["resources"],
                "optimizer_calls": result["optimizer_calls"],
                "TRAIN_loss_comparability": "Native versus augmented losses have different definitions; these are within-fit diagnostics.",
            })
            sources.extend([authority, pin(curve_path)])
    assert len(fit_summaries) == 24 and len({r["job_id"] for r in fit_summaries}) == 24
    assert sorted({r["seed"] for r in records}) == [101, 202, 303]
    aggregates = []
    for method in LABELS:
        selected = sorted((r for r in records if r["method"] == method), key=lambda r: r["seed"])
        assert len(selected) == 3
        row = {"method": method, "label": LABELS[method],
               "AUC_percent": summary([r["VALID_AUC_percent"] for r in selected]),
               "BCE": summary([r["VALID_BCE"] for r in selected]), "paired_comparisons": {}}
        for baseline in ["M1", "I4", "gnnm_boundary_native"]:
            base = sorted((r for r in records if r["method"] == baseline), key=lambda r: r["seed"])
            deltas = [a["VALID_AUC_percent"] - b["VALID_AUC_percent"] for a, b in zip(selected, base)]
            value = summary(deltas)
            half_width = 4.302652729749462 * value["sample_SD"] / math.sqrt(3)
            value.update({"wins_ties_losses": [sum(v > 0 for v in deltas), sum(v == 0 for v in deltas), sum(v < 0 for v in deltas)],
                          "approximate_95pct_paired_t_interval": [value["mean"] - half_width, value["mean"] + half_width],
                          "interval_scope": "Descriptive, assumes independent approximately normal seed differences on this fixed split; n=3, selected VALID, no multiple-comparison adjustment or confirmation claim."})
            row["paired_comparisons"][baseline] = value
        aggregates.append(row)
    report = {"schema": "completed-molbace-development-pilot-analysis-v1", "scope": "Three paired training seeds, one selected VALID scaffold split",
              "all_planned_fits_complete": True, "fits": fit_summaries, "aggregate": aggregates,
              "costs": {k: v for k, v in costs.items() if k != "jobs"}, "pilot_elapsed_seconds": terminal["elapsed_seconds"],
              "sources": sources, "new_optimizer_steps": 0, "new_model_forwards": 0, "TEST_accessed": False,
              "convergence": "Every actual epoch/selector is audited below. Patience shows validation stopping under the frozen rule; it does not establish optimization stationarity or remove seed/split uncertainty.",
              "decision": "Private input correction improves base GNNM in 3/3 seeds, remains behind I4 on average; geometry helps 2/3; their combination helps less. Retain planned ingredient comparisons and inspect private BN follow-up. No method promoted as an ensemble improvement."}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [("SELECTED_SCORES.csv", records), ("CURVES.csv", curves)]:
        with (output / name).open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Complete MolBACE development pilot", "", report["scope"] + ". Test data remain closed.", "",
             "| Method | VALID AUC mean ± sample SD (%) | Δ vs base (points) | Δ vs I4 (points) | Mean BCE |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in aggregates:
        lines.append(f"| {row['label']} | {row['AUC_percent']['mean']:.2f} ± {row['AUC_percent']['sample_SD']:.2f} | "
                     f"{row['paired_comparisons']['gnnm_boundary_native']['mean']:+.2f} | {row['paired_comparisons']['I4']['mean']:+.2f} | {row['BCE']['mean']:.3f} |")
    lines += ["", report["decision"], "", "The selected-score CSV contains all three seeds for every method. REPORT.json includes paired differences and approximate t intervals with their assumptions. Three seeds and selected validation measurements cannot establish significance or transfer.", "",
              f"All 24 fits completed; {len(curves):,} whole TRAIN epochs are recorded. Pilot elapsed time was {terminal['elapsed_seconds']/3600:.2f} hours, with another existing training worker sharing the GPU. Worker costs are preserved separately.", "", report["convergence"], "",
              "The last-20-epoch training-loss summaries compare a fit with itself. Auxiliary and native losses are not interchangeable. Selecting early and stopping later can indicate overfitting; a lower last training loss does not by itself justify training longer.", "",
              "This analysis reads existing JSON/JSONL measurements only. It starts no training, scores no new predictions, and changes no frozen decisions."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return {"output": str(output), "fits": 24, "epochs": len(curves), "new_optimizer_steps": 0, "new_model_forwards": 0,
            "report_sha256": pin(output / "REPORT.json")["sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--expansion", required=True)
    print(json.dumps(run(parser.parse_args().expansion)))
