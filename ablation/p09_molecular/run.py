"""Run the 150 paired molecular fits with the shared resumable worker queue."""

import argparse
import json
import sys
from pathlib import Path

import torch

from ablation.common.resumable import JobStore, atomic_write_json
from ablation.common.worker import print_status, run_worker
from .protocol import EXTENDED_METHODS, METHODS, RECIPES, REPOSITORY, confined_path, make_jobs
from .training import load_dataset, train, score_selected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, default=REPOSITORY / 'data')
    parser.add_argument('--datasets', nargs='+', choices=list(RECIPES), default=list(RECIPES))
    parser.add_argument('--models', nargs='+', choices=['gcn', 'gin'])
    parser.add_argument('--cohorts', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--methods', nargs='+', choices=EXTENDED_METHODS, default=METHODS)
    parser.add_argument('--convergence', action='store_true')
    parser.add_argument('--defer-test', action='store_true')
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke_epochs', type=int)
    parser.add_argument('--smoke_updates', type=int)
    parser.add_argument('--save_every_sec', type=float, default=120.0)
    parser.add_argument('--max_failures', type=int, default=3)
    parser.add_argument('--stale_seconds', type=float, default=600.0)
    parser.add_argument('--no_wait', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    for name in ('smoke_epochs', 'smoke_updates'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f'--{name} must be positive')
    if any(cohort < 0 for cohort in args.cohorts) or len(set(args.cohorts)) != len(args.cohorts):
        parser.error('Cohorts must be distinct nonnegative integers')
    if args.smoke_epochs and args.smoke_epochs > 100:
        parser.error('--smoke_epochs must fit the 100-epoch reference horizon')
    if args.save_every_sec <= 0:
        parser.error('--save_every_sec must be positive')
    args.root = confined_path(args.root)
    args.data_dir = confined_path(args.data_dir)
    return args


def main(argv=None):
    args = parse_args(argv)
    jobs = make_jobs(args)
    if not jobs:
        raise ValueError('Dataset/model selection produced no jobs')
    for job in jobs:
        store = JobStore(args.root / job.key)
        if store.config_path.exists():
            if json.loads(store.config_path.read_text()) != job.config:
                raise ValueError(f'Existing job has different source/config binding: {store.dir}')
        elif store.is_done() or store.has_state():
            raise ValueError(f'Existing state lacks a bound config: {store.dir}')
    if args.status:
        print_status(args.root, jobs, args.max_failures, args.stale_seconds)
        return 0
    args.root.mkdir(parents=True, exist_ok=True)
    role = {'study': 'p09_molecular', 'smoke': jobs[0].config['smoke'],
            'smoke_epochs': args.smoke_epochs, 'smoke_updates': args.smoke_updates}
    marker = args.root / 'protocol_role.json'
    if marker.exists():
        if json.loads(marker.read_text()) != role:
            raise ValueError('Use a separate root for smoke and scientific configurations')
    else:
        atomic_write_json(role, marker)
    device = torch.device(args.device)
    loaded = {}

    def bundle_for(name):
        if name not in loaded:
            loaded.clear()
            loaded[name] = load_dataset(name, args.data_dir)
        return loaded[name]

    # Completed fits must still match the current graph inputs before the shared
    # worker can skip them. Resumable fits enforce this binding inside train().
    for job in jobs:
        store = JobStore(args.root / job.key)
        if store.is_done():
            current_binding = bundle_for(job.config['dataset'])[3]
            if store.read_done().get('data_binding') != current_binding:
                raise ValueError(f'Completed job data binding changed: {store.dir}')

    if args.score_only:
        if not args.defer_test:
            raise ValueError('--score-only requires the training --defer-test binding')
        for job in jobs:
            parent = JobStore(args.root / job.key)
            if not parent.is_done() or parent.read_done().get('smoke'):
                raise ValueError('Scoring requires a completed scientific training fit')
        def score_job(job, store, stop):
            dataset, splits, evaluator, binding = bundle_for(job.config['dataset'])
            return score_selected(job.config, args.root / job.key, store, stop,
                                  dataset, splits, evaluator, binding, device)
        return run_worker(args.root / 'scoring', jobs, score_job,
                          max_failures=args.max_failures, stale_seconds=args.stale_seconds,
                          wait_for_others=not args.no_wait)

    def execute(job, store, stop):
        dataset, splits, evaluator, data_binding = bundle_for(job.config['dataset'])
        return train(job.config, store, stop, dataset, splits, evaluator, data_binding,
                     device, save_every_sec=args.save_every_sec)

    return run_worker(args.root, jobs, execute, max_failures=args.max_failures,
                      stale_seconds=args.stale_seconds, wait_for_others=not args.no_wait)


if __name__ == '__main__':
    sys.exit(main())
