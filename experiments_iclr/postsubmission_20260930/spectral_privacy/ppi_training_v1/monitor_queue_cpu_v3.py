"""Record this project's CPU queue progress without inspecting other jobs."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if HERE not in output.parents or output.exists():
        raise ValueError("Require a new progress record inside this study")
    launch = json.loads((HERE / "launch_cpu_v3_run01/launch.json").read_text())
    try:
        os.kill(launch["pid"], 0)
        alive = True
    except ProcessLookupError:
        alive = False
    cells = []
    for folder in sorted((HERE / "fits_cpu_v3").iterdir()):
        trace = folder / "validation_trace.csv"
        epochs = []
        if trace.is_file():
            with trace.open() as stream:
                for row in csv.DictReader(stream):
                    if row.get("epoch") is not None and row.get("elapsed_seconds") is not None:
                        try:
                            epochs.append(int(row["epoch"]))
                        except (TypeError, ValueError):
                            pass
        cells.append({"cell": folder.name, "last_complete_validation_epoch": max(epochs) if epochs else None,
                      "fit_completed": (folder / "result.json").is_file(),
                      "failure_recorded": (folder / "FAILURE.json").is_file(),
                      "replay_completed": (HERE / "replays_cpu_v3" / (folder.name + ".json")).is_file()})
    queue_complete = (HERE / "queue_cpu_v3_run01/completion.json").is_file()
    record = {"schema": "gnnm-cpu-queue-progress-v3", "UTC": datetime.now(timezone.utc).isoformat(),
              "supervisor_pid": launch["pid"], "supervisor_pid_exists": alive,
              "queue_completed": queue_complete, "fits_started": len(cells),
              "fits_completed": sum(c["fit_completed"] for c in cells),
              "failures": sum(c["failure_recorded"] for c in cells),
              "replays_completed": sum(c["replay_completed"] for c in cells),
              "development_gate_exists": (HERE / "DEVELOPMENT_GATE_cpu_v3.json").is_file(),
              "cells": cells, "predictive_metrics_read": False, "other_jobs_inspected": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
