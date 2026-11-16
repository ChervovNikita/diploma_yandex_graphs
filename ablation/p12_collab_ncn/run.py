"""Run five paired Collab cohorts through the existing shared job queue."""
import argparse
from pathlib import Path

import torch
from ablation.common.worker import Job, print_status, run_worker
from .protocol import METHODS, REPOSITORY, config_for, job_key, source_identity


def make_jobs(args):
    identity = source_identity()
    jobs = [
        Job(job_key(cohort, method),
            config_for(cohort, method, args.smoke_updates, args.data_dir, identity))
        for cohort in args.cohorts for method in args.methods
    ]
    for job in jobs:
        job.config.update(epochs=args.epochs, min_epochs=args.min_epochs,
                          patience=args.patience, defer_test=args.defer_test,
                          cap_schedule=([args.initial_cap, args.epochs] if args.initial_cap
                                        else [args.epochs]))
        if args.defer_test and not args.smoke_updates:
            job.config["test_role"] = "deferred"
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=REPOSITORY / "data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cohorts", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--smoke-updates", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--initial-cap", type=int)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--defer-test", action="store_true")
    parser.add_argument("--save_every_sec", type=float, default=120)
    parser.add_argument("--max_failures", type=int, default=3)
    parser.add_argument("--stale_seconds", type=float, default=3600)
    parser.add_argument("--no_wait", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or not 0 <= args.min_epochs <= args.epochs:
        parser.error("Invalid convergence horizon")
    if args.initial_cap is not None and not 0 < args.initial_cap < args.epochs:
        parser.error("Initial cap must precede the final epoch cap")
    if args.patience is not None and args.patience < 1:
        parser.error("Patience must be positive")
    if args.smoke_updates is not None and args.smoke_updates < 1:
        parser.error("--smoke-updates must be positive")
    if len(set(args.cohorts)) != len(args.cohorts) or any(c < 0 or c > 4 for c in args.cohorts):
        parser.error("--cohorts must contain distinct values from 0 through 4")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must be distinct")
    for path in (args.root, args.data_dir):
        if not path.resolve().is_relative_to(REPOSITORY):
            parser.error("All input/output directories must be inside the authorized repository")
    return args


def main(argv=None):
    args = parse_args(argv)
    jobs = make_jobs(args)
    if args.status:
        print_status(args.root, jobs, args.max_failures, args.stale_seconds)
        return 0
    from ablation.common.resumable import JobStore
    for job in jobs:
        existing = JobStore(args.root / job.key)
        if existing.config_path.exists():
            existing.ensure_config(job.config)
    from .data import load_bundle
    from .train import train_fit
    device = torch.device(args.device)
    bundle = load_bundle(args.data_dir, device)

    def execute(job, store, stop):
        return train_fit(job.config, store, stop, bundle, device, args.save_every_sec)

    return run_worker(args.root, jobs, execute, args.max_failures,
                      args.stale_seconds, wait_for_others=not args.no_wait)


if __name__ == "__main__":
    raise SystemExit(main())
