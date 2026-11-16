"""Command line shared by the studies: one fixed matrix of restartable jobs."""

import argparse
from pathlib import Path

import torch

from datasets import load_dataset
from run_common import NUM_STEPS, PATIENCE

from .resumable import source_fingerprint
from .tabm_trainer import TrainSettings, train_resumable
from .worker import Job, print_status, run_worker


REPOSITORY = Path(__file__).resolve().parents[2]
COMMON_SOURCES = (
    'models.py', 'datasets.py', 'run_common.py', 'ablation/common/resumable.py',
    'ablation/common/study.py', 'ablation/common/tabm_trainer.py', 'ablation/common/worker.py',
)
PAPER_MODELS = ['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep']
HIDDEN_DIM = 512
LEARNING_RATE = 3e-5


def make_jobs(args, variants):
    return [
        Job(
            key=f'{args.dataset}/{variant}/{model}/L{num_layers}/split{split}',
            config={
                'dataset': args.dataset, 'variant': variant, 'model': model,
                'num_layers': num_layers, 'split': split,
                'seed': args.seed if args.seed is not None else split,
                'hidden_dim': HIDDEN_DIM, 'lr': LEARNING_RATE, 'm': variants[variant],
                'num_steps': args.num_steps, 'patience': args.patience,
                'min_steps': args.min_steps, 'selection_interval': args.selection_interval,
                'weight_decay': args.weight_decay or 0.0, 'defer_test': args.defer_test,
                'save_selected': args.save_selected,
            },
        )
        for variant in args.variants
        for model in args.models
        for num_layers in args.layers
        for split in args.splits
    ]


def add_training_arguments(parser):
    """Shared protocol flags for singleton campaign tasks and legacy study workers."""
    parser.add_argument('--min_steps', type=int, default=0)
    parser.add_argument('--patience', type=int, default=PATIENCE * 100)
    parser.add_argument('--selection_interval', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--defer_test', action='store_true')
    parser.add_argument('--save_selected', action='store_true')


def parse_args(description, variants, default_variants, default_root):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=default_root)
    parser.add_argument('--dataset', default='roman-empire')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--variants', nargs='+', default=default_variants, choices=list(variants))
    parser.add_argument('--models', nargs='+', default=PAPER_MODELS)
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--splits', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS)
    add_training_arguments(parser)
    parser.add_argument('--save_every_sec', type=float, default=120.0)
    parser.add_argument('--max_failures', type=int, default=3)
    parser.add_argument('--stale_seconds', type=float, default=600.0)
    parser.add_argument('--no_wait', action='store_true',
                        help='Exit instead of waiting for jobs held by other workers.')
    parser.add_argument('--status', action='store_true', help='Print progress and exit.')
    return parser.parse_args()


def run_study(description, variants, build_model, study_sources, default_root,
              default_variants=None):
    """Run a worker. ``variants`` maps a variant name to its number of members."""
    args = parse_args(description, variants, default_variants or list(variants), default_root)
    jobs = make_jobs(args, variants)
    for job in jobs:
        job.config['model_module'] = build_model.__module__
    if args.status:
        print_status(args.root, jobs, args.max_failures, args.stale_seconds)
        return 0

    device = torch.device(args.device)
    fingerprint = source_fingerprint(
        REPOSITORY / name for name in (*COMMON_SOURCES, *study_sources))
    data, train_masks, val_masks, test_masks, _, num_targets, is_binary = load_dataset(
        args.dataset, add_self_loops=True, device=device, data_dir=args.data_dir,
    )
    bundle = (data, train_masks, val_masks, test_masks, is_binary)

    def execute(job, store, stop):
        config = job.config
        settings = TrainSettings(
            dataset=config['dataset'], split=config['split'], seed=config['seed'],
            members=config['m'], lr=config['lr'], num_steps=config['num_steps'],
            patience=config['patience'], save_every_sec=args.save_every_sec,
            min_steps=config['min_steps'], selection_interval=config['selection_interval'],
            weight_decay=config['weight_decay'], defer_test=config['defer_test'],
            save_selected=config['save_selected'],
        )
        result = train_resumable(
            settings, store, bundle, device,
            lambda: build_model(
                config['variant'], config['model'], config['num_layers'], data.x.size(1),
                config['hidden_dim'], num_targets, config['m'], device,
            ),
            stop=stop,
        )
        result.update(config)
        result['source_sha256'] = fingerprint
        return result

    return run_worker(args.root, jobs, execute, max_failures=args.max_failures,
                      stale_seconds=args.stale_seconds, wait_for_others=not args.no_wait)
