"""Numerical parity check for the TABM training step.

Trains the same TABM configuration twice from the same seed: once with the
``old`` sum-then-backward loop and once with the ``new`` per-member backward
loop (the memory-efficient version that ships in run_tabm/_runner). Compares
final parameters and final val/test metrics; both should match within a tiny
floating-point tolerance because the two formulations are mathematically
equivalent (sums commute, ``.grad`` accumulates additively).
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import compute_metrics, get_validation_metric_name, load_dataset
from models import TABMModel
from run_common import set_seed


def loss_for_member(model, data, is_binary, member_idx):
    out = model(data, data.x, tabm_seed=member_idx)
    if is_binary:
        return F.binary_cross_entropy_with_logits(
            out[data.train_mask].squeeze(-1), data.y[data.train_mask]
        )
    return F.cross_entropy(out[data.train_mask], data.y[data.train_mask].long())


def step_old(model, data, optimizer, is_binary, k):
    model.train()
    optimizer.zero_grad()
    total = 0.0
    for i in range(k):
        loss = loss_for_member(model, data, is_binary, i)
        total = total + loss / k
    total.backward()
    optimizer.step()
    return total.item()


def step_new(model, data, optimizer, is_binary, k):
    model.train()
    optimizer.zero_grad()
    total = 0.0
    for i in range(k):
        loss = loss_for_member(model, data, is_binary, i) / k
        loss.backward()
        total += loss.item()
    optimizer.step()
    return total


@torch.no_grad()
def evaluate(model, data, mask, is_binary, dataset_name, k):
    model.eval()
    outs = [model(data, data.x, tabm_seed=i) for i in range(k)]
    avg = torch.stack(outs).mean(dim=0)
    return compute_metrics(avg, data.y, mask, is_binary, dataset_name=dataset_name)


def make_model(model_name, num_layers, input_dim, num_targets, hidden_dim, k, device):
    return TABMModel(
        model_name=model_name, num_layers=num_layers, input_dim=input_dim,
        hidden_dim=hidden_dim, output_dim=num_targets,
        hidden_dim_multiplier=1, num_heads=8,
        normalization='LayerNorm', dropout=0.0,
        tabm_inits=k, device=device,
    ).to(device)


def train(step_fn, model_name, num_layers, hidden_dim, lr, k, num_steps,
          data, num_targets, is_binary, device, dataset_name, seed):
    set_seed(seed)
    model = make_model(model_name, num_layers, data.x.size(1), num_targets, hidden_dim, k, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    for _ in range(num_steps):
        step_fn(model, data, optimizer, is_binary, k)
    val = evaluate(model, data, data.val_mask, is_binary, dataset_name, k)
    test = evaluate(model, data, data.test_mask, is_binary, dataset_name, k)
    state = {name: param.detach().clone() for name, param in model.named_parameters()}
    return state, val, test


def compare_states(state_old, state_new):
    max_abs = 0.0
    max_rel = 0.0
    worst = None
    for name in state_old:
        a = state_old[name]
        b = state_new[name]
        diff = (a - b).abs()
        m = diff.max().item()
        denom = a.abs().max().item() + 1e-12
        r = m / denom
        if m > max_abs:
            max_abs = m
            worst = name
        if r > max_rel:
            max_rel = r
    return max_abs, max_rel, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--model', type=str, default='SAGE')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=200)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--abs_tol', type=float, default=1e-4)
    parser.add_argument('--rel_tol', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    pyg_data, train_masks, val_masks, test_masks, _, num_targets, is_binary = load_dataset(
        args.dataset, add_self_loops=True, device=device, data_dir=args.data_dir
    )
    pyg_data.train_mask = train_masks[:, args.split].to(device)
    pyg_data.val_mask = val_masks[:, args.split].to(device)
    pyg_data.test_mask = test_masks[:, args.split].to(device)

    common = dict(
        model_name=args.model, num_layers=args.num_layers, hidden_dim=args.hidden_dim,
        lr=args.lr, k=args.k, num_steps=args.num_steps,
        data=pyg_data, num_targets=num_targets, is_binary=is_binary,
        device=device, dataset_name=args.dataset, seed=args.seed,
    )

    print(f'[parity-check] model={args.model} L={args.num_layers} h={args.hidden_dim} '
          f'k={args.k} steps={args.num_steps} dataset={args.dataset} split={args.split}')
    print('Training OLD step (sum-then-backward)...')
    state_old, val_old, test_old = train(step_old, **common)
    print('Training NEW step (per-member backward, memory-efficient)...')
    state_new, val_new, test_new = train(step_new, **common)

    metric_name = get_validation_metric_name(args.dataset, is_binary)
    val_diff = abs(val_old['metric'] - val_new['metric'])
    test_diff = abs(test_old['metric'] - test_new['metric'])
    max_abs, max_rel, worst = compare_states(state_old, state_new)

    print('\n=== Parity report ===')
    print(f'val_{metric_name}: old={val_old["metric"]:.6f}  new={val_new["metric"]:.6f}  '
          f'|diff|={val_diff:.2e}')
    print(f'test_{metric_name}: old={test_old["metric"]:.6f} new={test_new["metric"]:.6f} '
          f'|diff|={test_diff:.2e}')
    print(f'params  max |old - new| = {max_abs:.3e}   max relative = {max_rel:.3e}')
    print(f'        worst tensor    = {worst}')

    ok_metrics = val_diff < args.abs_tol and test_diff < args.abs_tol
    ok_params = max_abs < args.abs_tol or max_rel < args.rel_tol
    if ok_metrics and ok_params:
        print('\nPASS: old and new train_step are numerically equivalent.')
        sys.exit(0)
    print('\nFAIL: differences exceed tolerance.')
    sys.exit(1)


if __name__ == '__main__':
    main()
