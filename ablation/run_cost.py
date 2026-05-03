"""Cost analysis: parameter count, mean step time and peak GPU memory for
``BASE`` (single deterministic), ``ENS`` (k independently trained models, the
deep-ensemble baseline) and ``TABM`` (BatchEnsemble) with several values of k.

We do not need full training: a short warmup + measurement loop on split 0 is
enough to read off the per-step compute and memory profile.
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import DATASET_CHOICES, load_dataset
from models import Model, TABMModel
from run_common import ALL_MODELS, set_seed


VARIANTS = ['BASE', 'ENS_k4', 'TABM_k2', 'TABM_k4', 'TABM_k8']
WARMUP_STEPS = 50
MEASURE_STEPS = 200
ENS_K = 4

FIELDNAMES = [
    'dataset', 'model', 'variant', 'k', 'num_layers', 'hidden_dim', 'lr',
    'num_params', 'mean_step_ms', 'std_step_ms', 'peak_mem_mb',
]


def make_base(model_name, num_layers, input_dim, num_targets, hidden_dim, device):
    return Model(
        model_name=model_name, num_layers=num_layers, input_dim=input_dim,
        hidden_dim=hidden_dim, output_dim=num_targets,
        hidden_dim_multiplier=1, num_heads=8,
        normalization='LayerNorm', dropout=0.2,
    ).to(device)


def make_tabm(model_name, num_layers, input_dim, num_targets, hidden_dim, k, device):
    return TABMModel(
        model_name=model_name, num_layers=num_layers, input_dim=input_dim,
        hidden_dim=hidden_dim, output_dim=num_targets,
        hidden_dim_multiplier=1, num_heads=8,
        normalization='LayerNorm', dropout=0.2,
        tabm_inits=k, device=device,
    ).to(device)


def loss_fn(out, data, is_binary):
    if is_binary:
        return F.binary_cross_entropy_with_logits(
            out[data.train_mask].squeeze(-1), data.y[data.train_mask]
        )
    return F.cross_entropy(out[data.train_mask], data.y[data.train_mask].long())


def step_base(model, data, optimizer, is_binary):
    model.train()
    optimizer.zero_grad()
    out = model(data, data.x)
    loss = loss_fn(out, data, is_binary)
    loss.backward()
    optimizer.step()


def step_tabm(model, data, optimizer, is_binary, k):
    model.train()
    optimizer.zero_grad()
    total = 0.0
    for i in range(k):
        out = model(data, data.x, tabm_seed=i)
        total = total + loss_fn(out, data, is_binary) / k
    total.backward()
    optimizer.step()


def measure(step_fn, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times_ms = []
    for _ in range(WARMUP_STEPS):
        step_fn()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    for _ in range(MEASURE_STEPS):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step_fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 / 1024) if device.type == 'cuda' else 0.0
    return times_ms, peak_mb


def num_params(model):
    return sum(p.numel() for p in model.parameters())


def profile_variant(variant, model_name, num_layers, hidden_dim, lr, data, num_targets,
                    is_binary, device):
    set_seed(0)
    if variant == 'BASE':
        model = make_base(model_name, num_layers, data.x.size(1), num_targets, hidden_dim, device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        params = num_params(model)
        times, peak = measure(lambda: step_base(model, data, opt, is_binary), device)
        del model, opt
    elif variant == 'ENS_k4':
        models = [make_base(model_name, num_layers, data.x.size(1), num_targets, hidden_dim, device)
                  for _ in range(ENS_K)]
        opts = [torch.optim.AdamW(m.parameters(), lr=lr) for m in models]
        params = sum(num_params(m) for m in models)

        def _step():
            for m, o in zip(models, opts):
                step_base(m, data, o, is_binary)
        times, peak = measure(_step, device)
        del models, opts
    else:
        k = int(variant.split('_k')[-1])
        model = make_tabm(model_name, num_layers, data.x.size(1), num_targets, hidden_dim, k, device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        params = num_params(model)
        times, peak = measure(lambda: step_tabm(model, data, opt, is_binary, k), device)
        del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return params, times, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_CHOICES)
    parser.add_argument('--models', nargs='+', default=['GCN', 'SAGE', 'GAT-sep', 'GT-sep'],
                        choices=ALL_MODELS)
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=[512])
    parser.add_argument('--lr', nargs='+', type=float, default=[3e-5])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--variants', nargs='+', default=VARIANTS, choices=VARIANTS)
    args = parser.parse_args()

    device = torch.device(args.device)
    if os.path.dirname(args.log_path):
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)

    pyg_data, train_masks, val_masks, test_masks, num_classes, num_targets, is_binary = load_dataset(
        args.dataset, add_self_loops=True, device=device, data_dir=args.data_dir
    )
    pyg_data.train_mask = train_masks[:, 0].to(device)
    pyg_data.val_mask = val_masks[:, 0].to(device)
    pyg_data.test_mask = test_masks[:, 0].to(device)

    write_header = not os.path.exists(args.log_path)
    log_file = open(args.log_path, 'a', newline='')
    writer = csv.DictWriter(log_file, fieldnames=FIELDNAMES, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    for model_name in args.models:
        for num_layers in args.layers:
            for hidden_dim in args.hidden_dim:
                for lr in args.lr:
                    print(f'\n[{model_name}] L={num_layers} h={hidden_dim} lr={lr}')
                    for variant in args.variants:
                        print(f'  -> {variant}', flush=True)
                        params, times, peak_mb = profile_variant(
                            variant, model_name, num_layers, hidden_dim, lr,
                            pyg_data, num_targets, is_binary, device,
                        )
                        mean_ms = sum(times) / len(times)
                        var = sum((t - mean_ms) ** 2 for t in times) / max(len(times) - 1, 1)
                        std_ms = var ** 0.5
                        row = {
                            'dataset': args.dataset, 'model': model_name, 'variant': variant,
                            'k': 1 if variant == 'BASE' else (
                                ENS_K if variant == 'ENS_k4' else int(variant.split('_k')[-1])),
                            'num_layers': num_layers, 'hidden_dim': hidden_dim, 'lr': lr,
                            'num_params': params, 'mean_step_ms': round(mean_ms, 3),
                            'std_step_ms': round(std_ms, 3), 'peak_mem_mb': round(peak_mb, 1),
                        }
                        writer.writerow(row)
                        log_file.flush()
                        print(f'     params={params/1e6:.2f}M  step={mean_ms:.2f}ms  peak={peak_mb:.0f}MB')

    log_file.close()
    print(f'\nDone. Results saved to {args.log_path}')


if __name__ == '__main__':
    main()
