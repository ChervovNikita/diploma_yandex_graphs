"""Deterministic descriptive aggregation of the saved complete 72-cell CSV."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "test_results" / "TEMPERATURE_TEST_ALL72.csv"
OUTPUT = ROOT / "test_results" / "TEMPERATURE_ALL24_MEANS.csv"
GRAPHS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
METRICS = ("accuracy", "nll", "brier", "ece10")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert not OUTPUT.exists(), "Refusing to overwrite saved aggregation"
    original = json.loads((ROOT / "test_results" / "TEMPERATURE_TEST_SUMMARY.json").read_text())
    assert sha(SOURCE) == original["test_all72_csv_sha256"]
    with SOURCE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 72
    indexed = {(r["graph"], r["arm"], int(r["seed"])): r for r in rows}
    assert len(indexed) == 72
    fields = ["graph", "arm", "seeds", "test_nodes", "beta_mean", "beta_sample_sd"]
    for metric in METRICS:
        for label in ("before", "after", "delta"):
            fields.extend((f"{label}_{metric}_mean", f"{label}_{metric}_sample_sd"))
    output = []
    for graph in GRAPHS:
        for arm in ARMS:
            group = [indexed[(graph, arm, seed)] for seed in range(3)]
            node_counts = {int(r["test_nodes"]) for r in group}
            assert len(node_counts) == 1
            record = {"graph": graph, "arm": arm, "seeds": 3,
                      "test_nodes": node_counts.pop()}
            beta = np.array([float(r["beta"]) for r in group])
            record["beta_mean"] = float(beta.mean())
            record["beta_sample_sd"] = float(beta.std(ddof=1))
            for metric in METRICS:
                for label in ("before", "after", "delta"):
                    values = np.array([float(r[f"{label}_{metric}"]) for r in group])
                    record[f"{label}_{metric}_mean"] = float(values.mean())
                    record[f"{label}_{metric}_sample_sd"] = float(values.std(ddof=1))
            output.append(record)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"status": "ALL24_COMPLETE_MEANS_PASS", "groups": len(output),
                      "input_sha256": sha(SOURCE), "output_sha256": sha(OUTPUT)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
