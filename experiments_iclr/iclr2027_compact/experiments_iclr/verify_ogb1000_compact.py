"""Recalculate 1,000-epoch OGB scores from compact selected decisions.

This verifier does not replay checkpoints or recover logits from class decisions.
"""

import csv
import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments_iclr" / "ogbn_arxiv_1000_results"
scores = {}
reference_ids = None
reference_labels = None
for seed in range(3):
    for arm in ("base", "ens", "gnnm"):
        location = RESULT / f"seed_{seed}" / arm
        selected = json.loads((location / "selected.json").read_text())
        with (location / "epochs.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1000
        best = max(
            rows,
            key=lambda row: (
                float(row["valid_pooled_accuracy"]),
                -float(row["valid_pooled_ce"]),
                -int(row["epoch"]),
            ),
        )
        assert int(best["epoch"]) == selected["selected_epoch"]
        with np.load(location / "selected_decisions.npz") as data:
            ids = data["test_indices"]
            labels = data["test_labels"]
            valid_labels = data["valid_labels"]
            valid_pool = data["valid_pool"]
            members = data["test_members"]
            pooled = data["test_pool"]
            if reference_ids is None:
                reference_ids = ids.copy()
                reference_labels = labels.copy()
            assert np.array_equal(ids, reference_ids)
            assert np.array_equal(labels, reference_labels)
            assert len(ids) == 48603 and len(valid_labels) == 29799
            assert members.shape == (1 if arm == "base" else 4, len(ids))
            test_accuracy = float(np.mean(pooled == labels))
            valid_accuracy = float(np.mean(valid_pool == valid_labels))
            mean_member = float(np.mean(members == labels))
        assert abs(test_accuracy - selected["test_accuracy"]) < 1e-6
        assert abs(valid_accuracy - selected["valid_accuracy"]) < 1e-6
        assert abs(float(best["valid_pooled_accuracy"]) - selected["valid_accuracy"]) < 1e-6
        scores[seed, arm] = test_accuracy
        print(seed, arm, selected["selected_epoch"], f"{100*test_accuracy:.4f}",
              f"member={100*mean_member:.4f}")

summary = json.loads((RESULT / "summary.json").read_text())
for arm in ("base", "ens", "gnnm"):
    actual = [scores[seed, arm] for seed in range(3)]
    assert abs(statistics.mean(actual) -
               summary["variants"][arm]["test_accuracy_mean"]) < 1e-6
for pair, left, right in (
    ("gnnm_minus_ens", "gnnm", "ens"),
    ("gnnm_minus_base", "gnnm", "base"),
    ("ens_minus_base", "ens", "base"),
):
    differences = [scores[seed, left] - scores[seed, right]
                   for seed in range(3)]
    expected = summary["paired_differences"][pair]
    assert all(abs(a - b) < 1e-6 for a, b in
               zip(differences, expected["differences"]))
    assert abs(statistics.mean(differences) - expected["mean"]) < 1e-6
print("OGB1000_COMPACT_SCORE_AUDIT_PASS")
