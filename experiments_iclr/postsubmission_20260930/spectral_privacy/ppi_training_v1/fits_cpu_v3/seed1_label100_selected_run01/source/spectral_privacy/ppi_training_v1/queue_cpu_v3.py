"""Complete CPU spectral matrix; bounded concurrency and preserved failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
REPO = PHASE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    protocol = args.protocol.resolve()
    if HERE not in output.parents or output.exists() or HERE not in protocol.parents:
        raise ValueError("Require new queue output and exact phase protocol")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("GNNM_PHASE_ROOT") != str(PHASE):
        raise ValueError("Require repository wrapper and no visible CUDA devices")
    v = json.loads(protocol.read_text())
    if (v["schema"] != "gnnm-ppi-frozen-basis-predictive-cpu-v3"
            or v["status"] != "runtime_frozen_no_predictive_fit"
            or v["source_sha256"][str(Path(__file__).relative_to(PHASE))] != sha(__file__)):
        raise ValueError("Require frozen CPU queue source")
    python = REPO / ".venv/bin/python"
    supervisor = PHASE / "protocols/run_logged.py"
    runner = HERE / "runner_cpu_v3.py"
    summarize = HERE / "summarize_cpu_v3.py"
    fits = HERE / "fits_cpu_v3"
    replays = HERE / "replays_cpu_v3"
    if fits.exists() or replays.exists():
        raise ValueError("Refuse an existing matrix; inspect prior queue rather than overwrite")
    output.mkdir(parents=True)
    fits.mkdir()
    replays.mkdir()
    with (output / "queue_contract.json").open("x") as stream:
        json.dump({"schema": "gnnm-complete-cpu-spectral-queue-v3",
                   "pid": os.getpid(), "start_utc": datetime.now(timezone.utc).isoformat(),
                   "protocol_file_sha256": sha(protocol), "source_sha256": sha(__file__),
                   "parallel_fits_maximum": 8, "fits_required": 24, "replays_required": 24,
                   "per_fit_CPU_threads": 1, "CUDA_VISIBLE_DEVICES": "",
                   "stop_for_failure": "failed cells retained; others continue; gate withheld"}, stream, indent=2)
        stream.write("\n")

    def logged(name, command):
        log = PHASE / "protocols/logged_runs" / name
        result = subprocess.run([str(python), str(supervisor), "--output", str(log), "--",
                                 str(python), *map(str, command)], text=True, capture_output=True)
        return {"exit_code": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr, "supervisor_output": str(log.relative_to(PHASE))}

    def cell(label, seed, arm):
        name = f"seed{seed}_label{label}_{arm}_run01"
        fit = logged("spectral_cpu_v3_fit_" + name,
                     [runner, "fit", "--protocol", protocol, "--seed", seed,
                      "--label-fraction", label / 100, "--arm", arm,
                      "--device", "cpu", "--output", fits / name])
        record = {"cell": name, "fit": fit, "replay": None}
        if fit["exit_code"] == 0:
            record["replay"] = logged("spectral_cpu_v3_replay_" + name,
                                      [runner, "replay", "--protocol", protocol,
                                       "--run", fits / name, "--output", replays / (name + ".json")])
        with (output / (name + ".json")).open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return record

    records = []
    jobs = [(label, seed, arm) for label in (10, 100) for seed in (0, 1, 2)
            for arm in ("tied", "selected", "random", "raw")]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(cell, *job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"event": "cell_completed", "cell": record["cell"],
                              "fit_exit": record["fit"]["exit_code"],
                              "replay_exit": None if record["replay"] is None else record["replay"]["exit_code"],
                              "cells_completed": len(records)}), flush=True)
    passed = all(r["fit"]["exit_code"] == 0 and r["replay"] is not None
                 and r["replay"]["exit_code"] == 0 for r in records)
    aggregation = None
    if passed:
        aggregation = logged("spectral_cpu_v3_summarize_run01",
                             [summarize, "--protocol", protocol, "--fits", fits, "--replays", replays,
                              "--output", HERE / "DEVELOPMENT_GATE_cpu_v3.json"])
    completion = {"status": "complete" if passed and aggregation["exit_code"] == 0 else "failure_preserved",
                  "completion_utc": datetime.now(timezone.utc).isoformat(), "cells": records,
                  "aggregation": aggregation, "test_scored": False, "acceptance_claimed": False}
    with (output / "completion.json").open("x") as stream:
        json.dump(completion, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": completion["status"], "cells_completed": len(records)}), flush=True)
    sys.exit(0 if completion["status"] == "complete" else 1)


if __name__ == "__main__":
    main()
