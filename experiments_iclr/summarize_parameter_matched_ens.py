"""Descriptive paired summary of width-512 GNNM and two pooled ensembles."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / "experiments_iclr" / "results" / "projector_controls.csv"
MATCHED = ROOT / "experiments_iclr" / "parameter_matched_ens_results"


def read_rows(path: Path, variant: str, width: int) -> dict[int, dict[str, str]]:
    with path.open(newline="") as f:
        matches = [r for r in csv.DictReader(f)
                   if r["dataset"] == "roman-empire" and r["model"] == "SAGE"
                   and r["variant"] == variant and int(r["hidden_dim"]) == width
                   and int(r["num_layers"]) == 5 and float(r["lr"]) == 3e-5]
    by_split = {int(r["split"]): r for r in matches}
    if len(matches) != 5 or tuple(sorted(by_split)) != tuple(range(5)):
        raise RuntimeError(f"Expected one {variant} width {width} result for each split 0-4")
    return by_split


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def main() -> None:
    selection = json.loads((MATCHED / "selection.json").read_text())
    width = int(selection["selected_width"])
    gnnm = read_rows(STANDARD, "gnnm", 512)
    full = read_rows(STANDARD, "ens_pooled", 512)
    narrow = read_rows(MATCHED / "projector_controls.csv", "ens_pooled", width)
    rows = []
    for split in range(5):
        a, b, c = gnnm[split], full[split], narrow[split]
        row = {
            "split": split,
            "gnnm_test_acc_percent": 100 * float(a["test_acc"]),
            "ens_512_test_acc_percent": 100 * float(b["test_acc"]),
            "ens_matched_test_acc_percent": 100 * float(c["test_acc"]),
            "ens_512_minus_gnnm_pp": 100 * (float(b["test_acc"]) - float(a["test_acc"])),
            "ens_matched_minus_gnnm_pp": 100 * (float(c["test_acc"]) - float(a["test_acc"])),
            "gnnm_validation_acc_percent": 100 * float(a["val_metric"]),
            "ens_512_validation_acc_percent": 100 * float(b["val_metric"]),
            "ens_matched_validation_acc_percent": 100 * float(c["val_metric"]),
            "gnnm_train_seconds": float(a["train_seconds"]),
            "ens_512_train_seconds": float(b["train_seconds"]),
            "ens_matched_train_seconds": float(c["train_seconds"]),
            "gnnm_selected_step": int(a["best_step"]),
            "ens_512_selected_step": int(b["best_step"]),
            "ens_matched_selected_step": int(c["best_step"]),
        }
        rows.append(row)
    with (MATCHED / "paired_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = {
        "dataset": "roman-empire",
        "model": "SAGE",
        "official_masks": list(range(5)),
        "all_variants_depth": 5,
        "all_variants_learning_rate": 3e-5,
        "checkpoint_rule": "Pooled validation accuracy selects one joint ensemble checkpoint; held-out test read after selection",
        "parameter_match_rule": selection["selection_rule"],
        "gnnm_parameters": int(gnnm[0]["num_params"]),
        "ens_512_parameters": int(full[0]["num_params"]),
        "ens_matched_width": width,
        "ens_matched_parameters": int(narrow[0]["num_params"]),
        "gnnm_mean_test_acc_percent": mean(rows, "gnnm_test_acc_percent"),
        "ens_512_mean_test_acc_percent": mean(rows, "ens_512_test_acc_percent"),
        "ens_matched_mean_test_acc_percent": mean(rows, "ens_matched_test_acc_percent"),
        "ens_512_minus_gnnm_mean_paired_pp": mean(rows, "ens_512_minus_gnnm_pp"),
        "ens_matched_minus_gnnm_mean_paired_pp": mean(rows, "ens_matched_minus_gnnm_pp"),
        "ens_matched_test_mask_wins": sum(r["ens_matched_minus_gnnm_pp"] > 0 for r in rows),
        "ens_matched_test_mask_ties": sum(r["ens_matched_minus_gnnm_pp"] == 0 for r in rows),
        "ens_matched_test_mask_losses": sum(r["ens_matched_minus_gnnm_pp"] < 0 for r in rows),
        "gnnm_mean_full_training_seconds": mean(rows, "gnnm_train_seconds"),
        "ens_512_mean_full_training_seconds": mean(rows, "ens_512_train_seconds"),
        "ens_matched_mean_full_training_seconds": mean(rows, "ens_matched_train_seconds"),
        "scope_note": "Five official masks overlap on one graph. These are descriptive paired differences, not independent task replicates. Full training seconds include validation and checkpoint I/O.",
    }
    (MATCHED / "comparison.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
