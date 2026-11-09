"""Step 1 of the diversity study: retrain the winning depths and keep the full models.

Run from the repository root; safe to start repeatedly and in parallel:

    python -m ablation.p06_member_diversity.run --root RUN_DIR --device cuda:0

Each job stores the state dict of its best checkpoint in model.pt.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from datasets import load_dataset
from run_common import PATIENCE

from ablation.common.resumable import source_fingerprint
from ablation.common.study import HIDDEN_DIM, LEARNING_RATE, REPOSITORY
from ablation.common.tabm_trainer import TrainSettings, train_resumable
from ablation.common.worker import Job, print_status, run_worker
from ablation.p05_hparam_grid import grid

from .models import build_model
from .select_depths import SELECTED_PATH, SPLIT


SOURCES = ('models.py', 'datasets.py', 'run_common.py', 'ablation/common/resumable.py',
           'ablation/common/tabm_trainer.py', 'ablation/common/worker.py',
           'ablation/p05_hparam_grid/grid.py', 'ablation/p05_hparam_grid/models.py',
           'ablation/p06_member_diversity/models.py', 'ablation/p06_member_diversity/run.py',
           'ablation/p06_member_diversity/selected_depths.csv')


def job_methods(method):
    return ['gnnm'] if method == 'gnnm' else [f'ens_m{k}' for k in range(grid.ENSEMBLE_MEMBERS)]


def make_jobs(selected, args):
    wanted = selected[selected.dataset.isin(args.datasets) & selected.model.isin(args.models)]
    return [
        Job(
            key=f'{row.dataset}/{method}/{row.model}/L{row.num_layers}/split{SPLIT}',
            config={
                'dataset': row.dataset, 'method': method, 'model': row.model,
                'num_layers': int(row.num_layers), 'split': SPLIT,
                'seed': grid.seed_for(method, SPLIT), 'hidden_dim': HIDDEN_DIM,
                'members': grid.members_for(method), 'lr': LEARNING_RATE,
                'num_steps': args.num_steps or grid.steps_for(row.dataset),
                'patience': PATIENCE * 100,
            },
        )
        for row in wanted.sort_values(['dataset', 'method', 'model']).itertuples()
        for method in job_methods(row.method)
    ]


def parse_args(selected):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent / 'runs')
    parser.add_argument('--datasets', nargs='+', default=sorted(selected.dataset.unique()))
    parser.add_argument('--models', nargs='+', default=sorted(selected.model.unique()))
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Override the step limit, for smoke tests only.')
    parser.add_argument('--save_every_sec', type=float, default=120.0)
    parser.add_argument('--max_failures', type=int, default=3)
    parser.add_argument('--stale_seconds', type=float, default=600.0)
    parser.add_argument('--no_wait', action='store_true',
                        help='Exit instead of waiting for jobs held by other workers.')
    parser.add_argument('--status', action='store_true', help='Print progress and exit.')
    return parser.parse_args()


def main():
    selected = pd.read_csv(SELECTED_PATH)
    args = parse_args(selected)
    jobs = make_jobs(selected, args)
    if args.status:
        print_status(args.root, jobs, args.max_failures, args.stale_seconds)
        return 0

    device = torch.device(args.device)
    fingerprint = source_fingerprint(REPOSITORY / name for name in SOURCES)
    loaded = {}

    def bundle_for(dataset):
        if dataset not in loaded:
            loaded.clear()  # Jobs are grouped by dataset; keep one graph on the GPU.
            torch.cuda.empty_cache()
            data, train, val, test, _, targets, binary = load_dataset(
                dataset, add_self_loops=True, device=device, data_dir=args.data_dir)
            loaded[dataset] = ((data, train, val, test, binary), targets)
        return loaded[dataset]

    def execute(job, store, stop):
        config = job.config
        bundle, targets = bundle_for(config['dataset'])
        data = bundle[0]
        settings = TrainSettings(
            dataset=config['dataset'], split=config['split'], seed=config['seed'],
            members=config['members'], lr=config['lr'], num_steps=config['num_steps'],
            patience=config['patience'], save_every_sec=args.save_every_sec,
        )
        result = train_resumable(
            settings, store, bundle, device,
            lambda: build_model(
                config['method'], config['model'], config['num_layers'], data.x.size(1),
                config['hidden_dim'], targets, config['members'], device,
            ),
            stop=stop, model_path=store.dir / 'model.pt',
        )
        result.update(config)
        result['source_sha256'] = fingerprint
        return result

    return run_worker(args.root, jobs, execute, max_failures=args.max_failures,
                      stale_seconds=args.stale_seconds, wait_for_others=not args.no_wait)


if __name__ == '__main__':
    sys.exit(main())
