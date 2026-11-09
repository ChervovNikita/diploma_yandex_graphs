"""Measure the cost of one training step and of inference, in a short full-graph run.

Run one worker per GPU from the repository root, with nothing else on that GPU:

    python -m ablation.p07_cost_small_setup.run --root RUN_DIR --device cuda:0

A step is what the archived training loop does at every step: the training update, a
forward pass and the validation metric. The protocol follows Appendix C: 50 warmup
steps, then 200 measured steps with CUDA synchronization. A job is short, so after a
restart it simply starts again; measurements are never joined across a restart.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset

from ablation.common.resumable import source_fingerprint
from ablation.common.study import HIDDEN_DIM, PAPER_MODELS, REPOSITORY
from ablation.common.tabm_trainer import evaluate, member_logits, train_step
from ablation.common.worker import Job, Preempted, print_status, run_worker
from ablation.p05_hparam_grid import grid

from .models import VARIANTS, build_units


WARMUP_STEPS = 50
MEASURED_STEPS = 200
INFERENCE_REPEATS = 20
ALL_MODELS = PAPER_MODELS + ['TAG']
HEARTBEAT_EVERY_STEPS = 10
SOURCES = ('models.py', 'datasets.py', 'run_common.py', 'ablation/common/tabm_trainer.py',
           'ablation/common/worker.py', 'ablation/p01_placement/models.py',
           'ablation/p04_shared_heads/models.py', 'ablation/p06_member_diversity/models.py',
           'ablation/p07_cost_small_setup/models.py', 'ablation/p07_cost_small_setup/run.py')


def clock(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


def reset_peak(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device):
    return torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == 'cuda' else 0.0


def memory_of_others_mb(device):
    """GPU memory held outside this process; timings are unreliable if it is large."""
    if device.type != 'cuda':
        return 0.0
    free, total = torch.cuda.mem_get_info(device)
    return (total - free - torch.cuda.memory_reserved(device)) / 2 ** 20


def one_step(units, data, is_binary, dataset, device):
    """One step of the archived loop for every unit, in milliseconds per part."""
    train = forward = metric = 0.0
    for model, members, optimizer in units:
        started = clock(device)
        train_step(model, data, optimizer, is_binary, members)
        trained = clock(device)
        logits = member_logits(model, data, members)
        predicted = clock(device)
        evaluate(logits, data, data.val_mask, is_binary, dataset)
        train += trained - started
        forward += predicted - trained
        metric += time.perf_counter() - predicted
    return train * 1000, forward * 1000, metric * 1000


def pooled_logits(units, data):
    return torch.cat([member_logits(model, data, members) for model, members, _ in units]).mean(dim=0)


def summary(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def measure(units, data, is_binary, dataset, device, warmup, measured, tick):
    reset_peak(device)
    for step in range(warmup):
        one_step(units, data, is_binary, dataset, device)
        tick(step)
    parts = []
    for step in range(measured):
        parts.append(one_step(units, data, is_binary, dataset, device))
        tick(step)
    train, forward, metric = (np.array(values) for values in zip(*parts))
    training_peak = peak_mb(device)

    reset_peak(device)
    inference = []
    for _ in range(INFERENCE_REPEATS):
        started = clock(device)
        pooled_logits(units, data)
        inference.append((clock(device) - started) * 1000)

    full = train + forward + metric
    return {
        'num_params': sum(p.numel() for model, _, _ in units for p in model.parameters()),
        'warmup_steps': warmup, 'measured_steps': measured,
        'train_step_ms': summary(train)[0], 'train_step_std_ms': summary(train)[1],
        'forward_ms': summary(forward)[0], 'metric_ms': summary(metric)[0],
        'full_step_ms': summary(full)[0], 'full_step_std_ms': summary(full)[1],
        'inference_ms': summary(inference)[0], 'inference_std_ms': summary(inference)[1],
        'train_peak_mb': training_peak, 'inference_peak_mb': peak_mb(device),
        'full_step_series_ms': [round(float(value), 3) for value in full],
    }


def make_jobs(args):
    return [
        Job(
            key=f'{dataset}/{variant}/{model}/L{layers}/split{split}',
            config={'dataset': dataset, 'variant': variant, 'model': model,
                    'num_layers': layers, 'split': split, 'hidden_dim': HIDDEN_DIM,
                    'members': VARIANTS[variant]},
        )
        for dataset in args.datasets
        for variant in args.variants
        for model in args.models
        for layers in args.layers
        for split in args.splits
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).parent / 'runs')
    parser.add_argument('--datasets', nargs='+', default=grid.DATASETS)
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument('--models', nargs='+', default=ALL_MODELS)
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--splits', nargs='+', type=int, default=[0])
    parser.add_argument('--warmup_steps', type=int, default=WARMUP_STEPS)
    parser.add_argument('--measured_steps', type=int, default=MEASURED_STEPS)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_failures', type=int, default=3)
    parser.add_argument('--stale_seconds', type=float, default=600.0)
    parser.add_argument('--no_wait', action='store_true',
                        help='Exit instead of waiting for jobs held by other workers.')
    parser.add_argument('--status', action='store_true', help='Print progress and exit.')
    return parser.parse_args()


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
            loaded[dataset] = load_dataset(dataset, add_self_loops=True, device=device,
                                           data_dir=args.data_dir)
        return loaded[dataset]

    def execute(job, store, stop):
        config = job.config
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        data, train, val, test, _, targets, is_binary = bundle_for(config['dataset'])
        data.train_mask = train[:, config['split']].to(device)
        data.val_mask = val[:, config['split']].to(device)
        data.test_mask = test[:, config['split']].to(device)
        units = build_units(config['variant'], config['model'], config['num_layers'],
                            data.x.size(1), config['hidden_dim'], targets, config['split'], device)
        others = memory_of_others_mb(device)

        def tick(step):
            if stop.requested:
                raise Preempted()
            if step % HEARTBEAT_EVERY_STEPS == 0:
                store.heartbeat()

        result = measure(units, data, is_binary, config['dataset'], device,
                         args.warmup_steps, args.measured_steps, tick)
        result.update(config)
        result.update(source_sha256=fingerprint, other_gpu_memory_mb=others,
                      device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu')
        return result

    return run_worker(args.root, jobs, execute, max_failures=args.max_failures,
                      stale_seconds=args.stale_seconds, wait_for_others=not args.no_wait)


if __name__ == '__main__':
    sys.exit(main())
