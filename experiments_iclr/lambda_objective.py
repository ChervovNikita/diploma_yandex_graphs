"""Predeclared test of the member-loss diversity penalty coefficient.

For M=4 BatchEnsemble members with logits Z_m, the training loss is
(1-lambda) * CE(mean_m Z_m, y) + lambda * mean_m CE(Z_m, y).
Lambda=1 reproduces the original GNNM objective. The difference between the
second and first terms is the mean KL from pooled prediction to each member.
Only Roman Empire SAGE is run here, with fixed 5 layers/512/3e-5 and official
splits 0--4. Test results are read only after validation checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import compute_metrics, load_dataset
from projector_controls import (all_logits, compute_loss, make_model,
                                model_output, test_analysis)
from run_common import set_seed

FIELDS = [
    'dataset', 'model', 'variant', 'lambda', 'split', 'seed', 'num_layers',
    'hidden_dim', 'lr', 'm', 'max_steps', 'best_step', 'num_params',
    'train_seconds', 'val_acc', 'test_acc', 'test_nll',
    'test_member_loss_minus_pooled_loss', 'mean_member_acc',
    'at_least_one_member_correct_rate', 'pair_disagreement', 'error_jaccard',
    'brier', 'ece10', 'checkpoint', 'prediction_file',
]


def train_step_lambda(model, optimizer, data, m, lam):
    """Use the original sequential GNNM update when lambda is exactly one."""
    model.train()
    optimizer.zero_grad()
    if lam == 1.0:
        # Follow projector_controls.train_one exactly for the measured
        # GNNM baseline: four sequential forwards and backwards.
        for member in range(m):
            loss = compute_loss(
                model_output(model, 'gnnm', data, member), data, False
            ) / m
            loss.backward()
    else:
        logits = all_logits(model, 'gnnm', data, m)
        pooled_loss = compute_loss(logits.mean(0), data, False)
        member_loss = torch.stack([compute_loss(z, data, False)
                                   for z in logits]).mean()
        loss = (1 - lam) * pooled_loss + lam * member_loss
        loss.backward()
    optimizer.step()


def run_one(args, lam, split, data, masks, output_dim, device, root):
    data.train_mask = masks[0][:, split].to(device)
    data.val_mask = masks[1][:, split].to(device)
    data.test_mask = masks[2][:, split].to(device)
    set_seed(split)
    model = make_model(args, 'gnnm', data.x.size(1), output_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    slug = f'{args.dataset}_{args.model}_lambda{str(lam).replace(".", "p")}_split{split}'
    ckpt = root / 'checkpoints' / f'{slug}.pt'
    pred_path = root / 'predictions' / f'{slug}.npz'
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    best_val, best_step, idle = -float('inf'), 0, 0
    t0 = time.monotonic()
    for step in range(1, args.max_steps + 1):
        train_step_lambda(model, optimizer, data, args.m, lam)
        if step % 10 != 0 and step != 1:
            continue
        model.eval()
        with torch.no_grad():
            val_logits = all_logits(model, 'gnnm', data, args.m).mean(0)
            val = compute_metrics(val_logits, data.y, data.val_mask,
                                  False, args.dataset)['acc']
        if val > best_val:
            best_val, best_step, idle = val, step, 0
            torch.save(model.state_dict(), ckpt)
        else:
            idle += 10
        if step % 100 == 0 or step == 1:
            print(f'{slug} step={step} val={val:.6f} best={best_val:.6f}',
                  flush=True)
        if idle >= 300:
            break
    train_seconds = time.monotonic() - t0
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = all_logits(model, 'gnnm', data, args.m)
        pooled = logits.mean(0)
        val = compute_metrics(pooled, data.y, data.val_mask, False,
                              args.dataset)['acc']
        test = compute_metrics(pooled, data.y, data.test_mask, False,
                               args.dataset)
        descriptive, pred = test_analysis(logits, data.y, data.test_mask,
                                          data.edge_index)
        np.savez_compressed(pred_path, **pred)
        labels = data.y[data.test_mask].long()
        member_loss_test = torch.stack([
            torch.nn.functional.cross_entropy(z[data.test_mask], labels)
            for z in logits
        ]).mean()
        pooled_loss_test = torch.nn.functional.cross_entropy(
            pooled[data.test_mask], labels
        )
    out = {
        'dataset': args.dataset, 'model': args.model, 'variant': 'gnnm',
        'lambda': lam, 'split': split, 'seed': split,
        'num_layers': args.num_layers, 'hidden_dim': args.hidden_dim,
        'lr': args.lr, 'm': args.m, 'max_steps': args.max_steps,
        'best_step': best_step,
        'num_params': sum(p.numel() for p in model.parameters()),
        'train_seconds': train_seconds,
        'val_acc': val, 'test_acc': test['acc'], 'test_nll': test['loss'],
        'test_member_loss_minus_pooled_loss':
            (member_loss_test - pooled_loss_test).item(),
        'mean_member_acc': descriptive['mean_member_acc'],
        'at_least_one_member_correct_rate':
            descriptive['at_least_one_member_correct_rate'],
        'pair_disagreement': descriptive['pair_disagreement'],
        'error_jaccard': descriptive['error_jaccard'],
        'brier': descriptive['brier'], 'ece10': descriptive['ece10'],
        'checkpoint': str(ckpt), 'prediction_file': str(pred_path),
    }
    del model, optimizer, logits, pooled
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='roman-empire')
    parser.add_argument('--model', default='SAGE', choices=['SAGE'])
    parser.add_argument('--lambdas', nargs='+', type=float,
                        default=[0.0, 0.5, 1.0])
    parser.add_argument('--splits', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--result_root', default='experiments_iclr/lambda_results')
    args = parser.parse_args()
    if sorted(args.lambdas) != [0.0, 0.5, 1.0]:
        parser.error('Predeclared lambda set is {0, 0.5, 1}')
    if any(s < 0 or s > 9 for s in args.splits):
        parser.error('Split indices must be official 0..9')
    root = Path(args.result_root)
    root.mkdir(parents=True, exist_ok=True)
    result_file = root / 'lambda_objective.csv'
    existing = set()
    if result_file.exists():
        with result_file.open(newline='') as f:
            existing = {(float(r['lambda']), int(r['split']))
                        for r in csv.DictReader(f)}
    device = torch.device(args.device)
    data, train, val, test, _, output_dim, is_binary = load_dataset(
        args.dataset, add_self_loops=True, device=device, data_dir='data'
    )
    if is_binary:
        raise ValueError('This predeclared study is for multiclass Roman Empire')
    print('CONFIG', json.dumps(vars(args), sort_keys=True), flush=True)
    with result_file.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        for split in args.splits:
            for lam in args.lambdas:
                if (lam, split) in existing:
                    print('SKIP', lam, split, flush=True)
                    continue
                row = run_one(args, lam, split, data, (train, val, test),
                              output_dim, device, root)
                writer.writerow(row)
                f.flush()
                existing.add((lam, split))
                print('RESULT', json.dumps(row, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
