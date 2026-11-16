"""Start one GPU supervisor per visible assigned device and one CPU supervisor.

Under Slurm, invoke this once per node, using srun --ntasks-per-node=1.
Every node receives the identical --plan and shared --root.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .schema import load_plan
from .store import CampaignStore


def assigned_gpus(requested, environment=None):
    environment = os.environ if environment is None else environment
    if requested != "auto":
        tokens = requested.split(",") if requested else []
    elif environment.get("CUDA_VISIBLE_DEVICES"):
        tokens = environment["CUDA_VISIBLE_DEVICES"].split(",")
    elif environment.get("SLURM_JOB_ID"):
        declared = environment.get("SLURM_STEP_GPUS") or environment.get("SLURM_JOB_GPUS")
        if not declared:
            raise ValueError("Slurm GPU visibility is absent. Supply assigned --gpus explicitly.")
        tokens = declared.split(",")
    else:
        raise ValueError("Set CUDA_VISIBLE_DEVICES or supply --gpus explicitly.")
    tokens = [token.strip() for token in tokens]
    if not tokens or any(not token or token in {"-1", "NoDevFiles"} for token in tokens):
        raise ValueError("No assigned GPU devices")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Assigned GPU list contains duplicates")
    return tokens


def commands(args):
    common = [
        sys.executable, "-u", "-m", "ablation.campaign.worker",
        "--plan", str(args.plan.resolve()), "--root", str(args.root.resolve()),
        "--repo", str(args.repo.resolve()), "--poll-seconds", str(args.poll_seconds),
        "--terminate-grace", str(args.terminate_grace),
    ]
    result = []
    for token in assigned_gpus(args.gpus):
        result.append(common + [
            "--kind", "gpu", "--device", token,
            "--slots", str(args.jobs_per_gpu),
            "--memory-gb", str(args.memory_gb_per_gpu),
            "--cpus", str(args.cpus_per_gpu),
            "--gpu-memory-check", args.gpu_memory_check,
            "--memory-reserve-gb", str(args.memory_reserve_gb),
        ])
    if args.cpu_workers > 0:
        result.append(common + [
            "--kind", "cpu", "--slots", str(args.cpu_workers),
            "--memory-gb", str(args.cpu_memory_gb),
            "--cpus", str(args.cpu_cpus),
        ])
    return result


def guard_fixed_protocol_root(plan, root, repo):
    """The final campaign may only write into a new, protocol-marked root."""
    from ablation.fixed_protocol.protocol import CAMPAIGN_ID, OUTPUT_PREFIX, mark_output_root
    if plan["campaign_id"] == CAMPAIGN_ID:
        mark_output_root(root, repo)
    elif OUTPUT_PREFIX in Path(root).name:
        raise ValueError("Only the fixed-protocol plan may write into a fixed_protocol root")


def main(args):
    if not args.root.resolve().is_relative_to(args.repo.resolve()):
        raise ValueError("Campaign output root must be inside the repository")
    plan = load_plan(args.plan)
    guard_fixed_protocol_root(plan, args.root, args.repo)
    launch_commands = commands(args)
    if args.dry_run:
        print(json.dumps({"commands": launch_commands, "shared_root": str(args.root.resolve()),
                          "campaign_id": plan["campaign_id"]}, indent=2))
        return 0
    CampaignStore(args.root, plan).prepare()
    processes = [subprocess.Popen(command, cwd=args.repo) for command in launch_commands]
    interrupted = False

    def receive(number, frame):
        nonlocal interrupted
        interrupted = True
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, receive)
    while any(process.poll() is None for process in processes):
        # A failed supervisor must stop its peers instead of leaving them waiting
        # forever for work requiring the missing resource class.
        if any(process.poll() not in (None, 0) for process in processes):
            receive(signal.SIGTERM, None)
        time.sleep(0.5)
    return 2 if interrupted else max(process.returncode for process in processes)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", required=True, type=Path)
    result.add_argument("--root", required=True, type=Path)
    result.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--gpus", default="auto")
    result.add_argument("--jobs-per-gpu", type=int, default=4)
    result.add_argument("--memory-gb-per-gpu", type=float, default=70)
    result.add_argument("--cpus-per-gpu", type=int, default=8)
    result.add_argument("--gpu-memory-check", choices=("auto", "required", "off"), default="auto")
    result.add_argument("--memory-reserve-gb", type=float, default=1)
    result.add_argument("--cpu-workers", type=int, default=2)
    result.add_argument("--cpu-cpus", type=int, default=4)
    result.add_argument("--cpu-memory-gb", type=float, default=64)
    result.add_argument("--poll-seconds", type=float, default=2)
    result.add_argument("--terminate-grace", type=float, default=120)
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
