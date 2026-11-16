"""GAT hyperparameter pilot. Worker for the fixed grid; every job is restartable.

Run from the repository root; safe to start repeatedly and in parallel:

    python -m ablation.p05_hparam_grid.run --root RUN_DIR --device cuda:0
"""

import argparse
import sys
from pathlib import Path

import torch

from datasets import load_dataset
from run_common import PATIENCE

from ablation.common.resumable import source_fingerprint
from ablation.common.study import REPOSITORY, add_training_arguments
from ablation.common.tabm_trainer import TrainSettings, train_resumable
from ablation.common.worker import Job, print_status, run_worker

from . import grid
from .models import build_model


SOURCES = ('models.py', 'datasets.py', 'run_common.py', 'ablation/common/resumable.py',
           'ablation/common/tabm_trainer.py', 'ablation/common/worker.py',
           'ablation/p05_hparam_grid/grid.py', 'ablation/p05_hparam_grid/models.py',
           'ablation/p05_hparam_grid/run.py')


def make_jobs(args):
    configs = grid.configs_for(args.grid_family)
    return [
        Job(
            key=f'{dataset}/{method}/{model}/{name}/L{layers}/split{split}',
            config={
                'dataset': dataset, 'method': method, 'model': model, 'config_name': name,
                'model_module': 'ablation.p05_hparam_grid.models',
                'num_layers': layers, 'split': split,
                'seed': args.seed if args.seed is not None else grid.seed_for(method, split),
                'hidden_dim': grid.HIDDEN_DIM, 'members': grid.members_for(method),
                'num_steps': args.num_steps or grid.steps_for(dataset),
                'patience': args.patience, **configs[name],
                'min_steps': args.min_steps, 'selection_interval': args.selection_interval,
                'weight_decay': (args.weight_decay if args.weight_decay is not None
                                 else configs[name].get('weight_decay', 0.0)),
                'defer_test': args.defer_test, 'save_selected': args.save_selected,
            },
        )
        for dataset in args.datasets
        for method in args.methods
        for model in args.models
        for name in args.configs
        for layers in args.layers
        for split in args.splits
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent / 'runs')
    parser.add_argument('--datasets', nargs='+', default=grid.DATASETS)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--methods', nargs='+', default=grid.METHODS, choices=grid.METHODS)
    parser.add_argument('--models', nargs='+', default=['GAT'])
    parser.add_argument('--grid_family', choices=['legacy', 'lrwd', 'combined'], default='legacy')
    parser.add_argument('--configs', nargs='+', default=None)
    parser.add_argument('--layers', nargs='+', type=int, default=grid.LAYERS)
    parser.add_argument('--splits', nargs='+', type=int, default=grid.PILOT_SPLITS)
    parser.add_argument('--num_steps', type=int, default=None,
                        help='Explicit maximum updates, shared by every method in this task.')
    add_training_arguments(parser)
    parser.add_argument('--save_every_sec', type=float, default=120.0)
    parser.add_argument('--max_failures', type=int, default=3)
    parser.add_argument('--stale_seconds', type=float, default=600.0)
    parser.add_argument('--no_wait', action='store_true',
                        help='Exit instead of waiting for jobs held by other workers.')
    parser.add_argument('--status', action='store_true', help='Print progress and exit.')
    args = parser.parse_args()
    available = grid.configs_for(args.grid_family)
    args.configs = args.configs or list(available)
    unknown = set(args.configs) - set(available)
    if unknown:
        parser.error(f'Unknown configs for {args.grid_family}: {sorted(unknown)}')
    return args


def main():
    args = parse_args()
    jobs = make_jobs(args)
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
            min_steps=config['min_steps'], selection_interval=config['selection_interval'],
            weight_decay=config['weight_decay'], defer_test=config['defer_test'],
            save_selected=config['save_selected'],
        )
        logits_path = (store.dir / 'logits.pt'
                       if config['method'].startswith('ens_m') and not config['defer_test']
                       else None)
        result = train_resumable(
            settings, store, bundle, device,
            lambda: build_model(
                config['method'], config['model'], config['num_layers'], data.x.size(1),
                config['hidden_dim'], targets, config['members'], device, config['dropout'],
                config['attention_dropout'], config['num_heads'],
            ),
            stop=stop, logits_path=logits_path,
        )
        result.update(config)
        result['source_sha256'] = fingerprint
        return result

    return run_worker(args.root, jobs, execute, max_failures=args.max_failures,
                      stale_seconds=args.stale_seconds, wait_for_others=not args.no_wait)


if __name__ == '__main__':
    sys.exit(main())
