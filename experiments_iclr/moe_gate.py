"""Frozen-expert gate diagnostic for the Roman Empire GNNM checkpoint.

Run from the repository root after copying this file to
``experiments_iclr/moe_gate.py``. All data, checkpoints, and outputs stay
inside the repository. The predeclared comparison is uniform logit mean,
global convex member weights, and an entropy-conditioned node gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / 'experiments_iclr' / 'moe_gate_results'
CACHE = RESULTS / 'roman_empire_sage_gnnm_split0_all_logits.npz'
METADATA = RESULTS / 'roman_empire_sage_gnnm_split0_export.json'
REPORT = RESULTS / 'roman_empire_sage_gnnm_split0_gate.json'


def selected_control_row() -> dict[str, str]:
    path = ROOT / 'experiments_iclr' / 'results' / 'projector_controls.csv'
    with path.open(newline='') as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if (row['dataset'], row['model'], row['variant'], row['split'])
            == ('roman-empire', 'SAGE', 'gnnm', '0')
        ]
    if len(rows) != 1:
        raise RuntimeError(f'Expected one completed GNNM SAGE split-0 row, got {len(rows)}')
    return rows[0]


def export_logits() -> None:
    from ablation.models_ablation import TABMAblationModel
    from datasets import load_dataset

    torch.set_num_threads(2)
    row = selected_control_row()
    checkpoint = ROOT / row['checkpoint']
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sha256 = hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest()
    start = time.monotonic()
    data, train_masks, val_masks, test_masks, _, out_dim, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device='cpu',
        data_dir=str(ROOT / 'data'),
    )
    if is_binary:
        raise RuntimeError('This prespecified diagnostic requires multiclass Roman Empire')
    m = int(row['m'])
    model = TABMAblationModel(
        model_name='SAGE', num_layers=int(row['num_layers']),
        input_dim=data.x.shape[1], hidden_dim=int(row['hidden_dim']),
        output_dim=out_dim, hidden_dim_multiplier=1, num_heads=8,
        normalization='LayerNorm', dropout=0.2, tabm_inits=m,
        init_scheme='default', device='cpu',
    )
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = torch.stack(
            [model(data, data.x, tabm_seed=i) for i in range(m)], dim=1,
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        member_logits=logits.cpu().numpy().astype(np.float32),
        labels=data.y.cpu().numpy().astype(np.int64),
        train_mask=train_masks[:, 0].cpu().numpy().astype(bool),
        val_mask=val_masks[:, 0].cpu().numpy().astype(bool),
        test_mask=test_masks[:, 0].cpu().numpy().astype(bool),
    )
    metadata = {
        'dataset': 'roman-empire', 'model': 'SAGE', 'variant': 'gnnm',
        'official_split': 0, 'checkpoint': str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256': sha256, 'checkpoint_best_step': int(row['best_step']),
        'num_layers': int(row['num_layers']), 'hidden_dim': int(row['hidden_dim']),
        'member_count': m, 'nodes': int(logits.shape[0]),
        'classes': int(logits.shape[2]),
        'split_sizes': {
            'train': int(train_masks[:, 0].sum()),
            'validation': int(val_masks[:, 0].sum()),
            'test': int(test_masks[:, 0].sum()),
        },
        'export_seconds_cpu': time.monotonic() - start,
        'aggregation': 'raw logit average',
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


class GlobalGate(nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(m))

    def forward(self, logits: torch.Tensor, entropy_feature: torch.Tensor):
        del entropy_feature
        weights = self.alpha.softmax(dim=0).expand(logits.shape[0], -1)
        return (logits * weights.unsqueeze(-1)).sum(dim=1), weights


class EntropyGate(nn.Module):
    def __init__(self, m: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(m))
        self.beta = nn.Parameter(torch.zeros(m))

    def forward(self, logits: torch.Tensor, entropy_feature: torch.Tensor):
        weights = (self.alpha + self.beta * entropy_feature).softmax(dim=1)
        return (logits * weights.unsqueeze(-1)).sum(dim=1), weights


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        return {
            'accuracy': (logits.argmax(dim=1) == labels).float().mean().item(),
            'cross_entropy': F.cross_entropy(logits, labels).item(),
        }


def fit_one(
    name: str, model: nn.Module, logits: torch.Tensor,
    entropy_feature: torch.Tensor, labels: torch.Tensor,
    train_mask: torch.Tensor, val_mask: torch.Tensor,
) -> tuple[dict, nn.Module]:
    # The expert checkpoint is frozen. Only 4 or 8 gate parameters can change.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0)
    max_epochs, patience = 1000, 100
    best_key = (-float('inf'), -float('inf'))
    best_state = None
    best_epoch = 0
    stagnant = 0
    start = time.monotonic()
    for epoch in range(max_epochs + 1):
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(logits[val_mask], entropy_feature[val_mask])
            val = metrics(val_logits, labels[val_mask])
            key = (val['accuracy'], -val['cross_entropy'])
        if key > best_key:
            best_key = key
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stagnant = 0
        else:
            stagnant += 1
        if epoch == max_epochs or stagnant >= patience:
            break
        model.train()
        optimizer.zero_grad()
        train_logits, _ = model(logits[train_mask], entropy_feature[train_mask])
        loss = F.cross_entropy(train_logits, labels[train_mask])
        loss.backward()
        optimizer.step()
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits, _ = model(logits[train_mask], entropy_feature[train_mask])
        val_logits, _ = model(logits[val_mask], entropy_feature[val_mask])
    result = {
        'name': name,
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'effective_free_parameters': sum(p.numel() for p in model.parameters()) - 1,
        'optimizer': 'Adam', 'learning_rate': 0.02,
        'weight_decay': 0, 'maximum_epochs': max_epochs,
        'patience_epochs': patience, 'best_epoch': best_epoch,
        'last_epoch': epoch, 'fit_seconds_cpu': time.monotonic() - start,
        'train': metrics(train_logits, labels[train_mask]),
        'validation': metrics(val_logits, labels[val_mask]),
        'alpha': model.alpha.detach().tolist(),
    }
    if isinstance(model, EntropyGate):
        result['beta'] = model.beta.detach().tolist()
    return result, model


def evaluate_selected_gate_on_test(
    result: dict, model: nn.Module, logits: torch.Tensor,
    entropy_feature: torch.Tensor, labels: torch.Tensor,
    test_mask: torch.Tensor,
) -> None:
    model.eval()
    with torch.no_grad():
        test_logits, test_weights = model(
            logits[test_mask], entropy_feature[test_mask],
        )
    result['test'] = metrics(test_logits, labels[test_mask])
    result['mean_member_weight_test'] = test_weights.mean(dim=0).tolist()
    result['std_member_weight_test'] = test_weights.std(dim=0).tolist()


def fit_gates() -> None:
    torch.set_num_threads(2)
    if not CACHE.is_file() or not METADATA.is_file():
        raise FileNotFoundError('Run `export` first')
    source = json.loads(METADATA.read_text())
    with np.load(CACHE) as data:
        logits = torch.tensor(data['member_logits'], dtype=torch.float32)
        labels = torch.tensor(data['labels'], dtype=torch.long)
        train_mask = torch.tensor(data['train_mask'], dtype=torch.bool)
        val_mask = torch.tensor(data['val_mask'], dtype=torch.bool)
        test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)
    n, m, c = logits.shape
    if (n, m, c) != (source['nodes'], source['member_count'], source['classes']):
        raise RuntimeError('Cache shape does not match export metadata')
    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise RuntimeError('Masks are not disjoint')
    # Confidence is calculated from frozen member predictions, without labels.
    probs = logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    confidence = 1 - entropy / np.log(c)
    feature_mean = confidence[train_mask].mean(dim=0)
    feature_std = confidence[train_mask].std(dim=0).clamp_min(1e-6)
    feature = (confidence - feature_mean) / feature_std
    # Predeclared baseline. Test labels are used only after both gates are fit.
    uniform_logits = logits.mean(dim=1)
    uniform = {
        'name': 'uniform_logit_mean', 'trainable_parameters': 0,
        'train': metrics(uniform_logits[train_mask], labels[train_mask]),
        'validation': metrics(uniform_logits[val_mask], labels[val_mask]),
        'mean_member_weight_test': [1 / m] * m,
        'std_member_weight_test': [0.0] * m,
    }
    global_gate, selected_global = fit_one(
        'global_convex_logit_gate', GlobalGate(m), logits,
        feature, labels, train_mask, val_mask,
    )
    node_gate, selected_node = fit_one(
        'entropy_conditioned_node_gate', EntropyGate(m), logits,
        feature, labels, train_mask, val_mask,
    )
    # All model selection is complete before any held-out test labels are used.
    uniform['test'] = metrics(uniform_logits[test_mask], labels[test_mask])
    evaluate_selected_gate_on_test(
        global_gate, selected_global, logits, feature, labels, test_mask,
    )
    evaluate_selected_gate_on_test(
        node_gate, selected_node, logits, feature, labels, test_mask,
    )
    report = {
        'protocol': {
            'frozen_checkpoint': source['checkpoint'],
            'checkpoint_sha256': source['checkpoint_sha256'],
            'dataset': source['dataset'], 'model': source['model'],
            'official_split': source['official_split'],
            'train_nodes': int(train_mask.sum()),
            'validation_nodes': int(val_mask.sum()),
            'test_nodes': int(test_mask.sum()),
            'node_gate_feature': 'per-member normalized 1 - softmax entropy / log(classes)',
            'feature_standardization': 'per member, train nodes only',
            'node_gate_score': 'alpha_m + beta_m * standardized_confidence_i_m',
            'global_gate_score': 'alpha_m',
            'weight_rule': 'softmax across members, then weighted raw logit sum',
            'training_loss': 'cross-entropy on training nodes only',
            'model_selection': 'maximum validation accuracy, then lower validation cross-entropy',
            'validation_reuse_caveat': 'The frozen expert checkpoint was already selected using this split validation mask.',
            'claim_scope': 'one split, diagnostic only; no method novelty or generalization claim',
        },
        'uniform': uniform,
        'global': global_gate,
        'node': node_gate,
    }
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['export', 'fit'])
    args = parser.parse_args()
    if args.stage == 'export':
        export_logits()
    else:
        fit_gates()


if __name__ == '__main__':
    main()
