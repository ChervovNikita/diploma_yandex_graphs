"""Verify compact selected decisions for the 12-cell Roman budget study."""

import csv
import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "roman_budget1000"
index = json.loads((ROOT / "index.json").read_text())
assert len(index["cells"]) == 12
scores = {}
for cell in index["cells"]:
    run = ROOT / cell["run_dir"]
    result = json.loads((run / "result.json").read_text())
    with (run / "validation_trace.csv").open(newline="") as stream:
        trace = list(csv.DictReader(stream))
    assert len(trace) == 1000
    best = max(trace, key=lambda row: (
        float(row["valid_accuracy"]),
        -float(row["valid_ce"]),
        -int(row["epoch"]),
    ))
    assert int(best["epoch"]) == result["selected_epoch"]
    with np.load(run / "selected_decisions.npz") as data:
        assert len(data["test_indices"]) == len(data["test_labels"]) == 5666
        assert data["test_member_classes"].shape == (4, 5666)
        test = float(np.mean(data["test_pooled_classes"] == data["test_labels"]))
        valid = float(np.mean(data["valid_pooled_classes"] == data["valid_labels"]))
    assert abs(test - result["test_accuracy"]) < 1e-6
    assert abs(valid - result["valid_accuracy"]) < 1e-6
    scores[cell["depth"], cell["seed"], cell["arm"]] = test

for depth in (2, 5):
    paired = [100 * (scores[depth, seed, "tied"] -
                     scores[depth, seed, "untied_propagation"])
              for seed in range(3)]
    print(depth, [round(x, 3) for x in paired], round(statistics.mean(paired), 3))
print("ROMAN_BUDGET1000_COMPACT_SCORE_AUDIT_PASS")
