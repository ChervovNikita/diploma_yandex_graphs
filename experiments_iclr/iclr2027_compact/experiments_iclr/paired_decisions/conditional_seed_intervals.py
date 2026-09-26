"""Conditional three-seed t intervals for selected partial versus ENS.

These intervals describe optimizer-seed variation on a fixed graph split and
frozen selected configurations. They are not intervals across graph tasks,
graph splits, nodes, or the selection process itself.
"""
from __future__ import annotations

import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path

from analyze_paired_decisions import calculate, require


HERE = Path(__file__).resolve().parent
T_CRITICAL_95_DF2 = 4.3026527299


def build() -> dict:
    source = calculate()
    require(json.loads((HERE / "PAIRED_DECISIONS.json").read_text()) == source,
            "Paired decision source differs")
    rows = []
    for summary in source["setting_summaries"]:
        values = list(summary["seed_delta_pp"])
        require(len(values) == 3, f"Expected three seeds: {summary['graph']}")
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        se = sd / math.sqrt(3)
        require(abs(mean - summary["mean_accuracy_delta_pp"]) < 1e-12,
                f"Paired-count mean differs: {summary['graph']}")
        half = T_CRITICAL_95_DF2 * se
        rows.append({"study": summary["study"], "graph": summary["graph"],
                     "partial_arm": summary["partial_arm"],
                     "n_optimizer_seeds": 3, "seed_deltas_pp": values,
                     "mean_delta_pp": mean, "sample_sd_pp": sd,
                     "standard_error_pp": se, "t_critical_df2": T_CRITICAL_95_DF2,
                     "conditional_95_low_pp": mean - half,
                     "conditional_95_high_pp": mean + half})
    require(len(rows) == 8, "Expected eight settings")
    return {"scope": "Illustrative t intervals conditional on the fixed graph split and frozen validation-selected configurations. Assumes independent optimizer-seed differences and approximately normal seed effects. With n=3, estimates are unstable. Candidate selection pooled these same seeds' validation results, so the intervals do not account for selection. They are not graph-task, split, node, or simultaneous confidence intervals.",
            "t_critical_95_df2": T_CRITICAL_95_DF2,
            "rows": rows}


def main() -> None:
    require(len(sys.argv) == 1 or sys.argv == [sys.argv[0], "--check"],
            "Usage: python conditional_seed_intervals.py [--check]")
    report = build()
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    fields = ("study", "graph", "partial_arm", "n_optimizer_seeds",
              "mean_delta_pp", "sample_sd_pp", "standard_error_pp",
              "conditional_95_low_pp", "conditional_95_high_pp")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields}
                     for row in report["rows"])
    outputs = (("CONDITIONAL_SEED_INTERVALS.json", json_text),
               ("CONDITIONAL_SEED_INTERVALS.csv", stream.getvalue()))
    for name, data in outputs:
        path = HERE / name
        if "--check" in sys.argv:
            require(path.read_bytes() == data.encode(),
                    f"Packaged interval output differs: {name}")
        else:
            path.write_text(data)
    print(json.dumps({"status": "PASS", "settings": 8, "seeds_each": 3},
                     sort_keys=True))


if __name__ == "__main__":
    main()
