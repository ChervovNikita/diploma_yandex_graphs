"""Matched Roman Empire controls for projector-level GNN ensembling.

Run from the repository root with the repository-local Python environment.
The default protocol is fixed in advance: five layers, width 512, AdamW
learning rate 3e-5, official splits 0--4, five thousand maximum steps, and
validation checks every ten steps. Test labels affect the final reported
metrics and descriptive analyses only. They are never used to select a model.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation.models_ablation import BEBlockAblation, TABMAblationModel
from datasets import compute_metrics, load_dataset
from models import MODULES, NORMALIZATION, Model, ResidualModuleWrapper
from run_common import set_seed


VARIANTS = (
    'base',
    'gnnm',
    'untied_backbone',
    'freeze_output_factors',
    'gnnm_m1',
    'input_only',
    'output_only',
    'independent_projectors',
    'heads_only',
    'ens_pooled',
)
FIELDS = [
    'dataset', 'model', 'variant', 'split', 'seed', 'num_layers', 'hidden_dim',
    'lr', 'm', 'num_steps', 'best_step', 'num_params', 'train_seconds',
    'val_metric', 'test_metric', 'test_acc', 'test_loss', 'n_test',
    'mean_member_acc', 'at_least_one_member_correct_rate', 'pair_disagreement',
    'error_jaccard', 'mean_entropy', 'nll', 'brier', 'ece10',
    'checkpoint', 'prediction_file',
]


def init_projector(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


class ProjectorControlModel(nn.Module):
    """Four predictions with one residual backbone and selected projectors."""

    def __init__(self, variant: str, model_name: str, num_layers: int,
                 input_dim: int, hidden_dim: int, output_dim: int,
                 m: int, device: torch.device):
        super().__init__()
        if variant not in ('input_only', 'output_only', 'independent_projectors', 'heads_only'):
            raise ValueError(variant)
        self.variant = variant
        self.m = m
        if variant in ('input_only', 'independent_projectors'):
            if variant == 'input_only':
                self.input_be = BEBlockAblation(input_dim, hidden_dim, True,
                                                device, m, 'default')
            else:
                self.input_heads = nn.ModuleList(
                    [nn.Linear(input_dim, hidden_dim) for _ in range(m)]
                )
                for layer in self.input_heads:
                    init_projector(layer)
        else:
            self.input_shared = nn.Linear(input_dim, hidden_dim)
            init_projector(self.input_shared)

        self.dropout = nn.Dropout(p=0.2)
        self.act = nn.GELU()
        normalization = NORMALIZATION['LayerNorm']
        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                self.residual_modules.append(ResidualModuleWrapper(
                    module=module, normalization=normalization, dim=hidden_dim,
                    hidden_dim_multiplier=1, num_heads=8, dropout=0.2,
                ))
        self.output_normalization = normalization(hidden_dim)
        if variant in ('output_only', 'independent_projectors', 'heads_only'):
            if variant == 'output_only':
                self.output_be = BEBlockAblation(hidden_dim, output_dim, False,
                                                 device, m, 'default')
            else:
                self.output_heads = nn.ModuleList(
                    [nn.Linear(hidden_dim, output_dim) for _ in range(m)]
                )
                for layer in self.output_heads:
                    init_projector(layer)
        else:
            self.output_shared = nn.Linear(hidden_dim, output_dim)
            init_projector(self.output_shared)

    def forward(self, graph, x, member: int):
        if self.variant == 'input_only':
            x = self.input_be(x, tabm_seed=member)
        elif self.variant == 'independent_projectors':
            x = self.input_heads[member](x)
        else:
            x = self.input_shared(x)
        x = self.dropout(x)
        x = self.act(x)
        for module in self.residual_modules:
            x = module(graph, x)
        x = self.output_normalization(x)
        if self.variant == 'output_only':
            x = self.output_be(x, tabm_seed=member)
        elif self.variant in ('independent_projectors', 'heads_only'):
            x = self.output_heads[member](x)
        else:
            x = self.output_shared(x)
        return x.squeeze(1)


class IndependentBackbones(nn.Module):
    """Four ordinary networks whose checkpoint is chosen from pooled validation."""

    def __init__(self, args, input_dim, output_dim):
        super().__init__()
        self.members = nn.ModuleList([
            Model(args.model, args.num_layers, input_dim, args.hidden_dim,
                  output_dim, 1, 8, 'LayerNorm', 0.2)
            for _ in range(args.m)
        ])

    def forward(self, graph, x, member):
        return self.members[member](graph, x)


class UntiedBackboneGNNM(nn.Module):
    """GNNM projectors/readout with one initially identical backbone per member.

    The deep copies start all four stacks from the same parameters. This
    runtime advances RNG during the copies, so the constructor restores the
    post-anchor RNG state to match the tied arm. After updates, only the
    propagation stacks are untied; the two BatchEnsemble projectors and
    output norm remain shared.
    """

    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()
        anchor = TABMAblationModel(
            model_name=args.model, num_layers=args.num_layers,
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            output_dim=output_dim, hidden_dim_multiplier=1, num_heads=8,
            normalization='LayerNorm', dropout=0.2, tabm_inits=args.m,
            init_scheme='default', device=device,
        ).to(device)
        # Under the GPU77 PyG runtime, deepcopy of the residual stack advances
        # the library RNG. Preserve the state immediately after constructing
        # the common tied anchor so both arms start training from the same
        # parameter tensors, initial logits, and dropout RNG state.
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = (torch.cuda.get_rng_state(device).clone()
                    if torch.device(device).type == 'cuda' else None)
        self.input_be_block = anchor.input_be_block
        self.dropout = anchor.dropout
        self.act = anchor.act
        self.residual_modules_by_member = nn.ModuleList([
            copy.deepcopy(anchor.residual_modules) for _ in range(args.m)
        ])
        self.output_normalization = anchor.output_normalization
        self.output_be_block = anchor.output_be_block
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)

    def forward(self, graph, x, tabm_seed):
        x = self.input_be_block(x, tabm_seed=tabm_seed)
        x = self.dropout(x)
        x = self.act(x)
        for module in self.residual_modules_by_member[tabm_seed]:
            x = module(graph, x)
        x = self.output_normalization(x)
        return self.output_be_block(x, tabm_seed=tabm_seed).squeeze(1)


def make_model(args, variant, input_dim, output_dim, device):
    if variant == 'ens_pooled':
        return IndependentBackbones(args, input_dim, output_dim).to(device)
    if variant == 'base':
        return Model(args.model, args.num_layers, input_dim, args.hidden_dim,
                     output_dim, 1, 8, 'LayerNorm', 0.2).to(device)
    if variant == 'untied_backbone':
        return UntiedBackboneGNNM(args, input_dim, output_dim, device).to(device)
    m = 1 if variant == 'gnnm_m1' else args.m
    if variant in ('gnnm', 'gnnm_m1', 'freeze_output_factors'):
        model = TABMAblationModel(
            model_name=args.model, num_layers=args.num_layers,
            input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=output_dim,
            hidden_dim_multiplier=1, num_heads=8, normalization='LayerNorm',
            dropout=0.2, tabm_inits=m, init_scheme='default', device=device,
        ).to(device)
        if variant == 'freeze_output_factors':
            for factor in (model.output_be_block.R, model.output_be_block.S,
                           model.output_be_block.B):
                factor.requires_grad_(False)
        return model
    return ProjectorControlModel(variant, args.model, args.num_layers,
                                 input_dim, args.hidden_dim, output_dim,
                                 m, device).to(device)


def model_output(model, variant, data, member):
    if variant == 'base':
        return model(data, data.x)
    if variant in ('gnnm', 'gnnm_m1', 'untied_backbone',
                   'freeze_output_factors'):
        return model(data, data.x, tabm_seed=member)
    if variant == 'ens_pooled':
        return model(data, data.x, member)
    return model(data, data.x, member)


def all_logits(model, variant, data, m):
    return torch.stack([model_output(model, variant, data, i)
                        for i in range(m)], dim=0)


def compute_loss(logits, data, is_binary):
    if is_binary:
        return F.binary_cross_entropy_with_logits(
            logits[data.train_mask].squeeze(-1), data.y[data.train_mask]
        )
    return F.cross_entropy(logits[data.train_mask],
                           data.y[data.train_mask].long())


def test_analysis(logits, labels, mask, edge_index):
    """Descriptive held-out analysis, computed after model selection."""
    logits = logits[:, mask]
    y = labels[mask].long()
    m, n, c = logits.shape
    probs = torch.softmax(logits.mean(dim=0), dim=-1)
    member_preds = logits.argmax(dim=-1)
    avg_pred = probs.argmax(dim=-1)
    correct = member_preds.eq(y.unsqueeze(0))
    mean_member_acc = correct.float().mean().item()
    at_least_one_member_correct_rate = correct.any(dim=0).float().mean().item()
    pairs = list(combinations(range(m), 2))
    pair_disagreement = float(np.mean([
        member_preds[i].ne(member_preds[j]).float().mean().item()
        for i, j in pairs
    ])) if pairs else 0.0
    error_jaccard_values = []
    for i, j in pairs:
        ei, ej = ~correct[i], ~correct[j]
        union = (ei | ej).sum().item()
        error_jaccard_values.append((ei & ej).sum().item() / union if union else 1.0)
    error_jaccard = float(np.mean(error_jaccard_values)) if pairs else 1.0
    conf, _ = probs.max(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    nll = F.nll_loss(probs.clamp_min(1e-12).log(), y).item()
    brier = ((probs - F.one_hot(y, c)) ** 2).sum(dim=-1).mean().item()
    bins = torch.linspace(0, 1, 11, device=conf.device)
    ece = 0.0
    for i in range(10):
        in_bin = (conf > bins[i]) & (conf <= bins[i + 1])
        if in_bin.any():
            ece += in_bin.float().mean().item() * abs(
                avg_pred[in_bin].eq(y[in_bin]).float().mean().item()
                - conf[in_bin].mean().item()
            )
    # Undirected input has both directions. Remove self loops before counting.
    src, dst = edge_index
    keep = src != dst
    src, dst = src[keep], dst[keep]
    num_nodes = labels.numel()
    degree = torch.bincount(dst, minlength=num_nodes).float()
    same = labels[src].eq(labels[dst]).float()
    same_count = torch.zeros(num_nodes, device=labels.device)
    same_count.index_add_(0, dst, same)
    homophily = same_count / degree.clamp_min(1)
    indices = mask.nonzero(as_tuple=False).flatten()
    prediction = {
        'node_index': indices.cpu().numpy(),
        'y_true': y.cpu().numpy(),
        'ensemble_pred': avg_pred.cpu().numpy(),
        'member_pred': member_preds.cpu().numpy(),
        'member_logits': logits.cpu().numpy(),
        'ensemble_prob': probs.cpu().numpy(),
        'confidence': conf.cpu().numpy(),
        'degree': degree[mask].cpu().numpy(),
        'local_homophily': homophily[mask].cpu().numpy(),
    }
    return {
        'n_test': n,
        'mean_member_acc': mean_member_acc,
        'at_least_one_member_correct_rate': at_least_one_member_correct_rate,
        'pair_disagreement': pair_disagreement,
        'error_jaccard': error_jaccard,
        'mean_entropy': entropy.mean().item(),
        'nll': nll,
        'brier': brier,
        'ece10': ece,
    }, prediction


def apply_common_backbone_init(model, args, input_dim, output_dim, split, device):
    """Copy one independently seeded backbone into a projector variant."""
    set_seed(100000 + split)
    anchor = Model(args.model, args.num_layers, input_dim, args.hidden_dim,
                   output_dim, 1, 8, 'LayerNorm', 0.2).to(device)
    model.residual_modules.load_state_dict(anchor.residual_modules.state_dict())
    model.output_normalization.load_state_dict(anchor.output_normalization.state_dict())
    del anchor
    set_seed(200000 + split)


def train_one(args, variant, split, data, masks, output_dim, is_binary,
              device, result_root):
    data.train_mask = masks[0][:, split].to(device)
    data.val_mask = masks[1][:, split].to(device)
    data.test_mask = masks[2][:, split].to(device)
    set_seed(split)
    m = 1 if variant in ('base', 'gnnm_m1') else args.m
    model = make_model(args, variant, data.x.shape[1], output_dim, device)
    if args.common_backbone_init:
        apply_common_backbone_init(model, args, data.x.shape[1],
                                   output_dim, split, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_dir = result_root / 'checkpoints'
    pred_dir = result_root / 'predictions'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    record_variant = variant + '_common_init' if args.common_backbone_init else variant
    stem = f'{args.dataset}_{args.model}_{record_variant}_split{split}'
    ckpt = ckpt_dir / f'{stem}.pt'
    pred_file = pred_dir / f'{stem}.npz'
    best_val = -float('inf')
    best_step = 0
    since_improvement = 0
    start = time.monotonic()
    for step in range(1, args.num_steps + 1):
        model.train()
        optimizer.zero_grad()
        for member in range(m):
            loss = compute_loss(model_output(model, variant, data, member),
                                data, is_binary)
            loss = loss / m
            loss.backward()
        optimizer.step()
        if step % 10 != 0 and step != 1:
            continue
        model.eval()
        with torch.no_grad():
            mean_val_logits = all_logits(model, variant, data, m).mean(0)
            val = compute_metrics(mean_val_logits, data.y, data.val_mask,
                                  is_binary, args.dataset)['metric']
        if val > best_val:
            best_val = val
            best_step = step
            since_improvement = 0
            torch.save(model.state_dict(), ckpt)
        else:
            since_improvement += 10
        if step % 100 == 0 or step == 1:
            print(f'{stem} step={step} val={val:.6f} best={best_val:.6f}',
                  flush=True)
        if since_improvement >= 300:
            break
    train_seconds = time.monotonic() - start
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = all_logits(model, variant, data, m)
        mean_logits = logits.mean(0)
        val_metric = compute_metrics(mean_logits, data.y, data.val_mask,
                                     is_binary, args.dataset)['metric']
        test_metrics = compute_metrics(mean_logits, data.y, data.test_mask,
                                       is_binary, args.dataset)
        if not is_binary:
            descriptive, pred = test_analysis(logits, data.y, data.test_mask,
                                              data.edge_index)
            np.savez_compressed(pred_file, **pred)
        else:
            descriptive = {k: float('nan') for k in FIELDS[18:26]}
            descriptive['n_test'] = int(data.test_mask.sum().item())
    out = {
        'dataset': args.dataset,
        'model': args.model,
        'variant': record_variant,
        'split': split,
        'seed': split,
        'num_layers': args.num_layers,
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'm': m,
        'num_steps': args.num_steps,
        'best_step': best_step,
        'num_params': num_params,
        'train_seconds': train_seconds,
        'val_metric': val_metric,
        'test_metric': test_metrics['metric'],
        'test_acc': test_metrics['acc'],
        'test_loss': test_metrics['loss'],
        'checkpoint': str(ckpt),
        'prediction_file': str(pred_file) if not is_binary else '',
    }
    out.update(descriptive)
    del model, optimizer, logits, mean_logits
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='roman-empire')
    parser.add_argument('--model', choices=['SAGE', 'GAT', 'GAT-sep', 'GT-sep'], required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        default=['gnnm', 'independent_projectors',
                                 'input_only', 'output_only', 'base', 'gnnm_m1'])
    parser.add_argument('--splits', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=5000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--result_root', default='experiments_iclr/results')
    parser.add_argument('--common_backbone_init', action='store_true',
                        help='Overwrite residual and output-normalization parameters with one split-specific anchor state, then reset the training RNG. Use only for the predeclared GNNM versus independent-projector diagnostic.')
    args = parser.parse_args()
    if args.common_backbone_init and set(args.variants) != {'gnnm', 'independent_projectors'}:
        parser.error('The common initialization diagnostic requires exactly gnnm and independent_projectors')
    if any(s < 0 or s > 9 for s in args.splits):
        parser.error('Official split indices must be between 0 and 9')
    root = Path(args.result_root)
    root.mkdir(parents=True, exist_ok=True)
    result_file = root / 'projector_controls.csv'
    existing = set()
    if result_file.exists():
        with result_file.open(newline='') as f:
            existing = {(r['dataset'], r['model'], r['variant'], int(r['split']))
                        for r in csv.DictReader(f)}
    device = torch.device(args.device)
    data, train_masks, val_masks, test_masks, _, output_dim, is_binary = load_dataset(
        args.dataset, add_self_loops=True, device=device, data_dir='data'
    )
    masks = (train_masks, val_masks, test_masks)
    print('CONFIG', json.dumps(vars(args), sort_keys=True), flush=True)
    print('DEVICE', torch.cuda.get_device_name(device), flush=True)
    with result_file.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        for split in args.splits:
            for variant in args.variants:
                record_variant = variant + '_common_init' if args.common_backbone_init else variant
                key = (args.dataset, args.model, record_variant, split)
                if key in existing:
                    print('SKIP', key, flush=True)
                    continue
                result = train_one(args, variant, split, data, masks,
                                   output_dim, is_binary, device, root)
                writer.writerow(result)
                f.flush()
                existing.add(key)
                print('RESULT', json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
