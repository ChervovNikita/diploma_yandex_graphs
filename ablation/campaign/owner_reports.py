"""Tables and per-dataset figures for the two selected-configuration studies."""
import csv
import os
from pathlib import Path
import statistics

from .tasks import REPO, output, result
from .analyses import paired_summary


def table(rows, path):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def figures(rows, out, kind):
    # Keep matplotlib's cache within the portable repository.
    os.environ.setdefault("MPLCONFIGDIR", str(REPO / "ablation/campaign/evidence/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    artifacts = []
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        fig, ax = plt.subplots(figsize=(8, 4))
        if kind == "contrast":
            models = sorted({row["model"] for row in selected})
            values = [[row["test_delta"] for row in selected if row["model"] == model]
                      for model in models]
            ax.boxplot(values, tick_labels=models)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Contrastive minus archived GNNM test metric")
        else:
            for method in ("gnnm", "ens"):
                group = [row for row in selected if row["method"] == method]
                ax.scatter([row["pair_disagreement"] for row in group],
                           [row["ensemble_gain"] for row in group], label=method)
                for row in group:
                    ax.annotate(row["model"], (row["pair_disagreement"], row["ensemble_gain"]),
                                fontsize=7)
            ax.set_xlabel("Mean pairwise prediction disagreement")
            ax.set_ylabel("Ensemble metric minus mean member metric")
            ax.legend()
        ax.set_title(dataset + " (descriptive)")
        fig.tight_layout()
        path = out / (dataset + ".png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        artifacts.append(path.name)
    return artifacts


def build_report(action, meta, root, out):
    missing = [name for name in meta["results"] if not (output(root, name) / "done.json").exists()]
    available = [(name, result(root, name)) for name in meta["results"] if name not in missing]
    failed = [name for name, row in available if row.get("scientific_status") in
              {"numerical_failure", "infeasible"}]
    rows, summaries = [], {}
    if action == "historical_contrast_report":
        with (REPO / "ablation/p13_node_contrast/historical_baseline.csv").open(newline="") as stream:
            frozen = {(r["dataset"], r["model"], int(r["split"])): r for r in csv.DictReader(stream)}
        for name, row in available:
            if name in failed:
                continue
            baseline = frozen[(row["dataset"], row["model"], row["split"])]
            if row["num_layers"] != int(baseline["num_layers"]):
                raise ValueError("Contrastive fit differs from the frozen historical depth")
            rows.append(dict(dataset=row["dataset"], model=row["model"], split=row["split"],
                             num_layers=row["num_layers"], archived_source=baseline["source"],
                             archived_val=float(baseline["val_metric"]),
                             archived_test=float(baseline["test_metric"]),
                             val_metric=row["val_metric"], test_metric=row["test_metric"],
                             val_delta=row["val_metric"]-float(baseline["val_metric"]),
                             test_delta=row["test_metric"]-float(baseline["test_metric"]),
                             unresolved=bool(row.get("unresolved_convergence"))))
        for dataset, model in sorted({(r["dataset"], r["model"]) for r in rows}):
            group = [r for r in rows if (r["dataset"], r["model"]) == (dataset, model)]
            summaries[dataset+"/"+model] = dict(paired_summary([r["test_delta"] for r in group]),
                                                splits=[r["split"] for r in group])
        kind = "contrast"
    else:
        for name, row in available:
            if name in failed:
                continue
            method = name.rsplit("/", 1)[-1]
            rows.append(dict(dataset=row["dataset"], model=row["model"], method=method,
                             split=row["split"], num_layers=row["num_layers"], role=row["role"],
                             ensemble_metric=row["ensemble_metric"],
                             mean_member_metric=row["mean_member_metric"],
                             ensemble_gain=row["ensemble_gain"],
                             pair_disagreement=row["pair_disagreement"],
                             error_correlation=row["error_correlation"],
                             probability_correlation=row["probability_correlation"],
                             js_divergence=row["js_divergence"]))
        kind = "diversity"
    table(rows, out / "runs.csv")
    plots = figures(rows, out, kind) if rows else []
    return dict(study=action, rows=len(rows), missing_tasks=missing, failed_tasks=failed,
                complete=not missing and not failed and not any(r.get("unresolved") for r in rows),
                summaries=summaries, plots=plots, table="runs.csv",
                interpretation="Descriptive per-dataset comparisons; no pooled cross-metric or causal claim",
                baseline="Frozen archived published training results" if kind == "contrast" else
                         "Each method uses its own validation-selected depth on official split 0")


def training_tables(tasks, root, out):
    """Retain actual stop counts, not merely the selected best epoch."""
    out.mkdir(parents=True, exist_ok=True)
    fits, scores = [], []
    for name, task in tasks.items():
        meta = task.get("metadata", {})
        action = meta.get("action")
        path = output(root, name) / "done.json"
        if action not in {"node_fit", "transfer_fit", "node_score"} or not path.exists():
            continue
        row = result(root, name)
        if action == "node_score":
            scores.append({key: row.get(key) for key in (
                "dataset", "model", "num_layers", "split", "setting", "stage",
                "val_metric", "test_metric", "scientific_status", "unresolved_convergence")})
            continue
        fit = row.get("fit", row)
        spec = meta.get("spec", {})
        fits.append(dict(task=name, action=action, stage=meta.get("stage"),
                         dataset=fit.get("dataset", spec.get("dataset", meta.get("pair"))),
                         model=fit.get("model", spec.get("model")), split=fit.get("split"),
                         cohort=meta.get("cohort"), method=meta.get("method", spec.get("method")),
                         variant=spec.get("variant"), depth=fit.get("num_layers"),
                         stopped_step=fit.get("stopped_step"), best_step=fit.get("best_step"),
                         stopped_epoch=fit.get("stopped_epoch"), best_epoch=fit.get("best_epoch"),
                         total_updates=fit.get("global_step",fit.get("updates")),
                         train_seconds=fit.get("train_seconds"), stop_reason=fit.get("stop_reason"),
                         unresolved=fit.get("needs_convergence_extension",fit.get("stop_reason")=="epoch_cap"),
                         scientific_status=fit.get("scientific_status","complete")))
    table(fits, out/"training_steps.csv")
    table(scores, out/"depth_scores.csv")
    return dict(training_steps="training_steps.csv", depth_scores="depth_scores.csv",
                training_rows=len(fits), depth_rows=len(scores),
                interpretation="Actual stopped counts per physical fit; sum four native member counts for ENS compute. Best checkpoint counts are separate.")
