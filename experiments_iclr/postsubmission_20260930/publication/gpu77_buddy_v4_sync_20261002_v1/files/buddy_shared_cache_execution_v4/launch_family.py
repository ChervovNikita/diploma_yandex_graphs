"""Two-device local queue. Dry-run by default; no SSH or Git operations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads((HERE / "CONFIG.json").read_text())
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2:
        parser.error("Specify two distinct local GPU IDs.")
    root = Path(args.runs).resolve()
    pending = [(arm, seed) for seed in config["seeds"] for arm in config["arms"]]
    jobs = []
    for arm, seed in pending:
        folder = root / f"{arm}_seed{seed}"
        command = [sys.executable, str(HERE / "run.py"), "fit", "--cache", str(Path(args.cache).resolve()), "--output", str(folder), "--arm", arm, "--seed", str(seed), "--device", "cuda:0", "--qualification", str(Path(args.qualification).resolve())]
        jobs.append((arm, seed, folder, command))
    if not args.execute:
        print(json.dumps({"cells": len(jobs), "optimizer_fits": 24, "gpu_slots": gpus, "jobs": [{"arm": a, "seed": s, "argv": c} for a, s, _, c in jobs]}, indent=2))
        return
    # Validate the whole output namespace before starting either process.
    for arm, seed, folder, _ in jobs:
        if folder.exists() or (root / f"{arm}_seed{seed}.log").exists():
            raise RuntimeError(f"Existing cell requires explicit review/replay: {folder}")
    certificate = json.loads(Path(args.qualification).read_text())
    if certificate.get("status") != "synthetic_cpu_pass" or certificate.get("test_count") != 7:
        raise RuntimeError("CPU numerical qualification is required before dispatch.")
    root.mkdir(parents=True, exist_ok=True)
    active = {}
    failed = False
    while jobs or active:
        for gpu in gpus:
            if failed or gpu in active or not jobs:
                continue
            arm, seed, folder, command = jobs.pop(0)
            if folder.exists():
                raise RuntimeError(f"Existing cell requires explicit review/replay: {folder}")
            log = (root / f"{arm}_seed{seed}.log").open("x")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            process = subprocess.Popen(command, cwd=HERE, env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (process, log, arm, seed)
        for gpu, (process, log, arm, seed) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close(); del active[gpu]
                print(json.dumps({"arm": arm, "seed": seed, "gpu": gpu, "exit_code": code}), flush=True)
                failed |= code != 0
        if failed and not active:
            raise SystemExit("Family incomplete: a cell failed. Remaining cells were not silently discarded.")
        if active:
            time.sleep(1)


if __name__ == "__main__":
    main()
