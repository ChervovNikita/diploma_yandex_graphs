"""Describe selected ogbn-arxiv test predictions without selecting models.

Reads the checkpoints' already-saved predictions. Writes one row per completed
seed/variant, a breakdown by the number of correct members, and seed summaries.
All quantities are post-selection diagnostics on the official test set.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from ogbn_arxiv_pilot import repo_path, write_json

FIELDS = (
    "seed", "variant", "num_members", "test_nodes", "mean_member_accuracy",
    "pooled_accuracy", "pool_minus_mean_member_accuracy",
    "pairwise_prediction_disagreement", "all_members_wrong_fraction",
    "all_members_correct_fraction", "pooled_correct_with_no_correct_member_fraction",
)
COUNT_FIELDS = ("seed", "variant", "correct_members", "node_count",
                "node_fraction", "pooled_accuracy_within_group")


def analyze_predictions(member_logits: np.ndarray, labels: np.ndarray) -> tuple[dict, list[dict]]:
    if member_logits.ndim != 3:
        raise ValueError("Expected [members, test nodes, classes] logits")
    members, nodes, classes = member_logits.shape
    if labels.shape != (nodes,) or members < 1 or classes < 2:
        raise ValueError("Logit and label shapes do not match")
    member_preds = member_logits.argmax(axis=-1)
    member_correct = member_preds == labels[None, :]
    pooled_preds = member_logits.mean(axis=0).argmax(axis=-1)
    pooled_correct = pooled_preds == labels
    correct_counts = member_correct.sum(axis=0)
    pair_rates = [float(np.mean(member_preds[a] != member_preds[b]))
                  for a, b in combinations(range(members), 2)]
    row = {
        "num_members": int(members),
        "test_nodes": int(nodes),
        "member_accuracies": [float(x) for x in member_correct.mean(axis=1)],
        "mean_member_accuracy": float(member_correct.mean()),
        "pooled_accuracy": float(pooled_correct.mean()),
        "pool_minus_mean_member_accuracy": float(pooled_correct.mean() - member_correct.mean()),
        "pairwise_prediction_disagreement": float(np.mean(pair_rates)) if pair_rates else None,
        "all_members_wrong_fraction": float(np.mean(correct_counts == 0)),
        "all_members_correct_fraction": float(np.mean(correct_counts == members)),
        "pooled_correct_with_no_correct_member_fraction": float(np.mean((correct_counts == 0) & pooled_correct)),
    }
    by_count = []
    for count in range(members + 1):
        mask = correct_counts == count
        by_count.append({
            "correct_members": count,
            "node_count": int(mask.sum()),
            "node_fraction": float(mask.mean()),
            "pooled_accuracy_within_group": float(pooled_correct[mask].mean()) if mask.any() else None,
        })
    return row, by_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="experiments_iclr/ogbn_arxiv_results")
    args = parser.parse_args()
    root = repo_path(args.result_root)
    rows: list[dict] = []
    by_count_rows: list[dict] = []
    for selected_path in sorted(root.glob("seed_*/**/selected.json")):
        selected = json.loads(selected_path.read_text())
        archive_path = selected_path.parent / "selected_predictions.npz"
        if not archive_path.exists():
            raise FileNotFoundError(f"Predictions were not saved: {archive_path}")
        with np.load(archive_path) as archive:
            stats, by_count = analyze_predictions(
                archive["test_member_logits"], archive["test_labels"])
        if abs(stats["pooled_accuracy"] - selected["test_accuracy"]) > 1e-6:
            raise RuntimeError(f"Selected accuracy mismatch: {archive_path}")
        key = {"seed": int(selected["seed"]), "variant": selected["variant"]}
        rows.append({**key, **stats})
        by_count_rows.extend([{**key, **entry} for entry in by_count])
    if not rows:
        raise RuntimeError(f"No completed selected predictions found under {root}")
    with (root / "diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (root / "diagnostics_by_correct_count.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COUNT_FIELDS)
        writer.writeheader()
        writer.writerows(by_count_rows)
    summary = {
        "scope": "Descriptive diagnostics on the official test nodes after validation-only checkpoint selection",
        "accuracy_units": "fraction correct",
        "replication_unit": "optimization seed on one graph and one official split",
        "variants": {},
    }
    for variant in ("base", "ens", "gnnm"):
        vr = [row for row in rows if row["variant"] == variant]
        if not vr:
            continue
        metrics = ("mean_member_accuracy", "pooled_accuracy",
                   "pool_minus_mean_member_accuracy", "pairwise_prediction_disagreement")
        entry = {"seeds": [row["seed"] for row in vr], "num_seeds": len(vr)}
        for metric in metrics:
            values = [row[metric] for row in vr if row[metric] is not None]
            entry[metric + "_mean"] = float(np.mean(values)) if values else None
            entry[metric + "_sample_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None)
        summary["variants"][variant] = entry
    write_json(root / "diagnostics_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
