"""Validation-only selection of the fixed GAT grid and independent ensembles.

Every ensemble member has its own validation-selected checkpoint. The first
native fit may also serve as BASE when its seed and training protocol match.
Held-out metrics are evaluated only when reveal_test is explicitly requested.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from datasets import compute_metrics, load_dataset

from . import grid


CELL = ['dataset', 'model', 'config_name', 'num_layers', 'split']


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob('done.json')):
        row = json.loads(path.read_text())
        row['job_dir'] = str(path.parent)
        rows.append(row)
    if not rows:
        raise SystemExit(f'No finished grid fits under {root}')
    return pd.DataFrame(rows)


def _role_logits(group, role):
    payloads = [
        torch.load(Path(directory) / 'predictions.pt', map_location='cpu', weights_only=True)
        for directory in group.job_dir
    ]
    reference = payloads[0]['roles'][role]
    tensors = []
    for payload in payloads:
        item = payload['roles'][role]
        if not torch.equal(item['indices'], reference['indices']):
            raise ValueError(f'Ensemble members have different {role} node indices')
        if not torch.equal(item['labels'], reference['labels']):
            raise ValueError(f'Ensemble members have different {role} labels')
        if item['logits'].shape[0] != 1:
            raise ValueError('Each independent ensemble member must contain one model')
        tensors.append(item['logits'][0])
    logits = torch.stack(tensors).mean(dim=0)
    mask = torch.ones(len(reference['labels']), dtype=torch.bool)
    payload = payloads[0]
    return compute_metrics(logits, reference['labels'], mask, payload['is_binary'],
                           payload['dataset'])['metric']


def compose_ensembles(runs, load_split_data, reveal_test=True, reuse_base=False):
    """Compose exactly four native members, validating identity before averaging."""
    members = runs[runs.method.str.startswith('ens_m')].copy()
    if reuse_base:
        base = runs[runs.method == 'base'].copy()
        base['method'] = 'ens_m0'
        members = pd.concat([members, base], ignore_index=True)
    rows = []
    expected = {f'ens_m{i}' for i in range(grid.ENSEMBLE_MEMBERS)}
    for cell, group in members.groupby(CELL):
        if group.method.duplicated().any():
            raise ValueError(f'Duplicate independent member for {cell}')
        if set(group.method) != expected:
            continue
        group = group.sort_values('method')
        for key in ('lr', 'weight_decay', 'dropout', 'attention_dropout', 'num_heads',
                    'min_steps', 'num_steps', 'patience', 'selection_interval'):
            if key in group and group[key].nunique(dropna=False) != 1:
                raise ValueError(f'Independent members differ in {key} for {cell}')
        if 'seed' in group and group.seed.nunique() != grid.ENSEMBLE_MEMBERS:
            raise ValueError(f'Independent members must have distinct seeds for {cell}')
        row = {**dict(zip(CELL, cell)), 'method': 'ens'}
        roles = ['val', 'test'] if reveal_test else ['val']
        if all((Path(directory) / 'predictions.pt').exists() for directory in group.job_dir):
            for role in roles:
                row[f'{role}_metric'] = _role_logits(group, role)
        else:
            tensors = [torch.load(Path(directory) / 'logits.pt', map_location='cpu',
                                  weights_only=True) for directory in group.job_dir]
            if any(tensor.shape[0] != 1 for tensor in tensors):
                raise ValueError('Independent member logits must have leading dimension one')
            logits = torch.stack(tensors).squeeze(1).mean(dim=0)
            data, val_mask, test_mask, is_binary = load_split_data(cell[0], cell[4])
            row['val_metric'] = compute_metrics(logits, data.y, val_mask, is_binary, cell[0])['metric']
            if reveal_test:
                row['test_metric'] = compute_metrics(logits, data.y, test_mask, is_binary, cell[0])['metric']
        for name in ('fit_seconds', 'train_seconds', 'validation_seconds', 'num_params'):
            if name in group:
                row[name] = float(group[name].sum())
        if 'stopped_step' in group:
            row['member_steps_total'] = int(group.stopped_step.sum())
            row['steps_run'] = float(group.stopped_step.mean())
        if 'best_step' in group:
            row['steps_to_best'] = float(group.best_step.mean())
        if 'needs_convergence_extension' in group:
            row['needs_convergence_extension'] = bool(group.needs_convergence_extension.any())
        row['member_job_dirs'] = json.dumps(group.job_dir.tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def score_table(runs, ensembles, reveal_test=True):
    plain = runs[~runs.method.str.startswith('ens_m')].copy()
    columns = ['method', *CELL, 'val_metric']
    if reveal_test:
        columns.append('test_metric')
    for name in ('fit_seconds', 'train_seconds', 'validation_seconds', 'num_params',
                 'needs_convergence_extension'):
        if name in plain:
            columns.append(name)
    frames = [plain.reindex(columns=columns)]
    if len(ensembles):
        frames.append(ensembles.reindex(columns=columns))
    return pd.concat(frames, ignore_index=True)


def choose_depth(table):
    keys = ['dataset', 'method', 'model', 'config_name', 'split']
    if table.duplicated(keys + ['num_layers']).any():
        raise ValueError('Duplicate physical fit in depth comparison')
    ordered = table.sort_values(keys + ['num_layers'])
    return ordered.loc[ordered.groupby(keys, sort=False).val_metric.idxmax()]


def choose_configuration(selected, expected_splits):
    grouped = selected.groupby(['dataset', 'method', 'model', 'config_name'])
    aggregation = {'val_metric': ('val_metric', 'mean'), 'splits': ('split', 'nunique')}
    if 'test_metric' in selected:
        aggregation['test_metric'] = ('test_metric', 'mean')
    scores = grouped.agg(**aggregation).reset_index()
    complete = scores[scores.splits == expected_splits]
    if complete.empty:
        return complete, complete
    ordered = complete.sort_values(['dataset', 'method', 'model', 'config_name'])
    best = ordered.loc[ordered.groupby(['dataset', 'method', 'model'], sort=False).val_metric.idxmax()]
    return scores, best


def decide(best, reference):
    """Historical validation gate. Multi-backbone callers evaluate each backbone separately."""
    if 'model' in best and best.model.nunique() > 1:
        return {model: decide(best[best.model == model], reference[reference.model == model])
                for model in sorted(best.model.unique())}

    def pick(frame, method):
        return 100 * frame[frame.method == method].set_index('dataset').val_metric

    gap = pick(best, 'gnnm') - pick(best, 'ens')
    reference_gap = pick(reference, 'gnnm') - pick(reference, 'ens')
    wins = int((gap >= 0).sum())
    improvement = float(gap.mean() - reference_gap.mean())
    return {
        'datasets': int(gap.notna().sum()), 'gnnm_not_worse_on': wins,
        'mean_gap': float(gap.mean()), 'mean_reference_gap': float(reference_gap.mean()),
        'improvement_points': improvement, 'continue': wins >= 3 or improvement >= 0.5,
        'gap_by_dataset': gap.round(3).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--splits', type=int, default=len(grid.PILOT_SPLITS))
    parser.add_argument('--reveal_test', action='store_true')
    parser.add_argument('--reuse_base', action='store_true')
    args = parser.parse_args()
    runs = load_rows(args.root)
    if not all(key in runs for key in CELL):
        raise ValueError('Grid collection requires dataset/model/config_name/depth/split fields')
    cache = {}

    def load_split_data(dataset, split):
        if dataset not in cache:
            cache[dataset] = load_dataset(dataset, add_self_loops=True, device='cpu',
                                          data_dir=args.data_dir)
        data, _, val, test, _, _, is_binary = cache[dataset]
        return data, val[:, split], test[:, split], is_binary

    ensembles = compose_ensembles(runs, load_split_data, args.reveal_test, args.reuse_base)
    table = score_table(runs, ensembles, args.reveal_test)
    selected = choose_depth(table)
    scores, best = choose_configuration(selected, args.splits)
    if best.empty:
        raise SystemExit('No configuration is complete on all pilot splits yet.')
    reference = scores[(scores.config_name == grid.REFERENCE_NAME) & (scores.splits == args.splits)]
    decision = decide(best, reference)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / 'per_depth.csv', index=False)
    scores.to_csv(args.out / 'config_scores.csv', index=False)
    best.to_csv(args.out / 'selected.csv', index=False)
    (args.out / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(best.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
