"""Bounded real-data engineering checks and GPU throughput calibration.

Every fit performs two updates and validation only. These artifacts are marked
engineering_smoke and are excluded from scientific result collection.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from datasets import load_dataset
from run_common import set_seed
from ablation.common.resumable import atomic_write_json
from ablation.common.tabm_trainer import evaluate, member_logits, train_step
from ablation.p05_hparam_grid.selected import build_selected
from .plan import MODELS, VARIANTS, node_spec

REPO = Path(__file__).resolve().parents[2]


def specification(dataset, model, depth, method):
    return node_spec(dataset, model, depth, 0, method)


def cases():
    selected = {}
    for model in MODELS:
        for depth in [1, 3, 5]:
            for method in ['native0', 'gnnm']:
                selected[("roman-empire", model, depth, method)] = specification(
                    'roman-empire', model, depth, method)
    for model in ['GCN', 'GAT']:
        for method in VARIANTS:
            selected[('roman-empire', model, 3, method)] = specification(
                'roman-empire', model, 3, method)
    for dataset in ['amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        for depth in [1, 5]:
            for method in ['native0', 'gnnm']:
                selected[(dataset, 'GAT', depth, method)] = specification(dataset, 'GAT', depth, method)
    return [{'name': '/'.join(map(str, identity)), 'spec': spec}
            for identity, spec in selected.items()]


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def prepare(spec, device):
    data, train, val, _, _, targets, binary = load_dataset(
        spec['dataset'], add_self_loops=True, device=device, data_dir=str(REPO / 'data'))
    data.train_mask = train[:, spec['split']].to(device)
    data.val_mask = val[:, spec['split']].to(device)
    set_seed(spec['seed'])
    model, members = build_selected(spec, data, targets, device, calibrate=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'],
                                  weight_decay=spec['weight_decay'])
    return data, binary, model, members, optimizer


def measure(spec, prepared, device, updates=2):
    data, binary, model, members, optimizer = prepared
    torch.cuda.reset_peak_memory_stats(device)
    update_seconds = []
    for _ in range(updates):
        synchronize(device)
        started = time.perf_counter()
        loss = train_step(model, data, optimizer, binary, members)
        logits = member_logits(model, data, members)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite calibration predictions')
        metrics = evaluate(logits, data, data.val_mask, binary, spec['dataset'])
        if not math.isfinite(metrics['metric']) or not math.isfinite(loss):
            raise FloatingPointError('Nonfinite calibration training/validation metric')
        synchronize(device)
        update_seconds.append(time.perf_counter() - started)
        del logits
    return dict(role='engineering_smoke', scientific_results=False, updates=updates,
                heldout_scored=False, spec=spec, status='passed',
                full_step_seconds=update_seconds,
                steady_step_seconds=update_seconds[-1],
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
                nodes=int(data.num_nodes), edges=int(data.edge_index.shape[1]),
                parameters=sum(parameter.numel() for parameter in model.parameters()))


def single_case(case, out, device, barrier=None, ready=None):
    prepared = prepare(case['spec'], device)
    if ready:
        atomic_write_json({'ready': True}, ready)
        deadline = time.monotonic() + 180
        while not barrier.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Parallel calibration release barrier timed out')
            time.sleep(.05)
    result = measure(case['spec'], prepared, device)
    if hasattr(prepared[2], 'initialization_calibration'):
        result['initialization_calibration'] = prepared[2].initialization_calibration
    atomic_write_json(dict(name=case['name'], **result), out)
    return result


def parallel_trial(out, concurrency, device, peak_gib):
    folder = out / f'parallel_{concurrency}'
    folder.mkdir(parents=True, exist_ok=True)
    free, _ = torch.cuda.mem_get_info(device)
    required = concurrency * (peak_gib + 1.5) * 2**30
    if required > free * .9:
        return dict(concurrency=concurrency, status='skipped_memory',
                    required_bytes=required, free_bytes=free)
    case = {'name': 'parallel/roman-empire/GAT/L3/gnnm',
            'spec': specification('roman-empire', 'GAT', 3, 'gnnm')}
    release = folder / 'release'
    if release.exists():
        raise ValueError('Parallel calibration folder already contains a release marker')
    processes, streams = [], []
    try:
        for index in range(concurrency):
            stream = (folder / f'worker{index}.log').open('w')
            streams.append(stream)
            argv = [sys.executable, '-m', 'ablation.campaign.calibration',
                    '--out', str(folder / f'worker{index}.json'),
                    '--device', str(device), '--single-case', json.dumps(case),
                    '--barrier', str(release), '--ready', str(folder / f'ready{index}.json')]
            environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            processes.append(subprocess.Popen(argv, cwd=REPO, stdout=stream,
                                               stderr=subprocess.STDOUT, env=environment))
        deadline = time.monotonic() + 180
        while not all((folder / f'ready{index}.json').exists() for index in range(concurrency)):
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('Parallel calibration worker exited before readiness')
            if time.monotonic() > deadline:
                raise TimeoutError('Parallel calibration workers did not become ready')
            time.sleep(.1)
        started = time.perf_counter()
        release.write_text('start\n')
        codes = [process.wait(timeout=180) for process in processes]
        elapsed = time.perf_counter() - started
        if any(codes):
            raise RuntimeError(f'Parallel calibration exits {codes}')
        rows = [json.loads((folder / f'worker{index}.json').read_text())
                for index in range(concurrency)]
        critical = max(sum(row['full_step_seconds']) for row in rows)
        return dict(concurrency=concurrency, status='passed', updates=2 * concurrency,
                    release_to_exit_seconds=elapsed, measured_critical_seconds=critical,
                    measured_updates_per_gpu_second=2 * concurrency / critical,
                    end_to_end_updates_per_gpu_second=2 * concurrency / elapsed,
                    workers=rows)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--single-case')
    parser.add_argument('--barrier', type=Path)
    parser.add_argument('--ready', type=Path)
    parser.add_argument('--skip-parallel', action='store_true')
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(REPO):
        raise ValueError('Calibration artifacts must remain inside the repository')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('GPU calibration requires a CUDA device')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.single_case:
        single_case(json.loads(args.single_case), args.out, device, args.barrier, args.ready)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = []
    for index, case in enumerate(cases()):
        target = args.out / (case['name'].replace('/', '__') + '.json')
        if target.exists():
            previous = json.loads(target.read_text())
            if previous.get('spec') != case['spec']:
                raise ValueError(f'Calibration config changed for {case["name"]}; use a new output directory')
            records.append(previous)
            continue
        try:
            result = single_case(case, target, device)
            row = dict(name=case['name'], **result)
        except Exception as error:
            row = dict(name=case['name'], spec=case['spec'], status='failed',
                       error=f'{type(error).__name__}: {error}',
                       role='engineering_smoke', scientific_results=False)
            atomic_write_json(row, target)
        records.append(row)
        print(json.dumps({key: row.get(key) for key in
                          ('name', 'status', 'steady_step_seconds', 'peak_allocated_gib', 'error')}),
              flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    parallel = []
    if not args.skip_parallel:
        reference = next(row for row in records if row['name'] == 'roman-empire/GAT/3/gnnm')
        for concurrency in [1, 2, 4]:
            try:
                trial = parallel_trial(args.out, concurrency, device,
                                       reference['peak_allocated_gib'])
            except Exception as error:
                trial = dict(concurrency=concurrency, status='failed',
                             error=f'{type(error).__name__}: {error}')
            parallel.append(trial)
            atomic_write_json(trial, args.out / f'throughput{concurrency}.json')
            print(json.dumps({key: value for key, value in trial.items() if key != 'workers'}),
                  flush=True)
    summary = dict(role='engineering_smoke', scientific_results=False,
                   gpu=torch.cuda.get_device_name(device),
                   gpu_total_gib=torch.cuda.get_device_properties(device).total_memory / 2**30,
                   started_unix=started, elapsed_seconds=time.time() - started,
                   cases=len(records), passed=sum(row['status'] == 'passed' for row in records),
                   failed=sum(row['status'] == 'failed' for row in records),
                   records=records, parallel=parallel,
                   caveat='Two-update engineering measurements provide planning estimates. Long-run throughput and convergence must be measured after launch.')
    atomic_write_json(summary, args.out / 'summary.json')
    if summary['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
