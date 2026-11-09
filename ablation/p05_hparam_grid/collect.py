"""Pilot decision for the GAT grid, made on validation only.

    python -m ablation.p05_hparam_grid.collect --root RUN_DIR --out RESULT_DIR

Each ENS result averages the logits of its four members. Depth is chosen on
validation for each split, then one configuration per dataset and method by the
mean of those validation scores. Test scores are shown only with --reveal_test.
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
        raise SystemExit(f'No finished jobs under {root}')
    return pd.DataFrame(rows)


def compose_ensembles(runs, load_split_data):
    """One row per complete ensemble; ``load_split_data`` gives (data, masks, is_binary)."""
    members = runs[runs.method.str.startswith('ens_m')]
    rows = []
    for cell, group in members.groupby(CELL):
        if group.method.nunique() < grid.ENSEMBLE_MEMBERS:
            continue
        dataset = cell[0]
        logits = torch.stack([torch.load(Path(d) / 'logits.pt', map_location='cpu', weights_only=True)
                              for d in group.job_dir]).squeeze(1).mean(dim=0)
        data, val_mask, test_mask, is_binary = load_split_data(dataset, cell[4])
        rows.append({
            **dict(zip(CELL, cell)),
            'method': 'ens',
            'val_metric': compute_metrics(logits, data.y, val_mask, is_binary, dataset)['metric'],
            'test_metric': compute_metrics(logits, data.y, test_mask, is_binary, dataset)['metric'],
        })
    return pd.DataFrame(rows)


def score_table(runs, ensembles):
    plain = runs[~runs.method.str.startswith('ens_m')]
    columns = ['method', *CELL, 'val_metric', 'test_metric']
    frames = [plain[columns]]
    if len(ensembles):
        frames.append(ensembles[columns])
    return pd.concat(frames, ignore_index=True)


def choose_depth(table):
    keys = ['dataset', 'method', 'model', 'config_name', 'split']
    ordered = table.sort_values(keys + ['num_layers'])
    return ordered.loc[ordered.groupby(keys, sort=False).val_metric.idxmax()]


def choose_configuration(selected, expected_splits):
    grouped = selected.groupby(['dataset', 'method', 'model', 'config_name'])
    scores = grouped.agg(val_metric=('val_metric', 'mean'), test_metric=('test_metric', 'mean'),
                         splits=('split', 'nunique')).reset_index()
    complete = scores[scores.splits == expected_splits]
    if complete.empty:
        return complete, complete
    ordered = complete.sort_values(['dataset', 'method', 'model', 'config_name'])
    best = ordered.loc[ordered.groupby(['dataset', 'method', 'model'], sort=False).val_metric.idxmax()]
    return scores, best


def decide(best, reference):
    """Gaps in points; the decision rule was fixed before any pilot run."""
    pick = lambda frame, method: 100 * frame[frame.method == method].set_index('dataset').val_metric
    gap = pick(best, 'gnnm') - pick(best, 'ens')
    reference_gap = pick(reference, 'gnnm') - pick(reference, 'ens')
    wins = int((gap >= 0).sum())
    improvement = float(gap.mean() - reference_gap.mean())
    return {
        'datasets': int(gap.notna().sum()),
        'gnnm_not_worse_on': wins,
        'mean_gap': float(gap.mean()),
        'mean_reference_gap': float(reference_gap.mean()),
        'improvement_points': improvement,
        'continue': wins >= 3 or improvement >= 0.5,
        'gap_by_dataset': gap.round(3).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--splits', type=int, default=len(grid.PILOT_SPLITS))
    parser.add_argument('--reveal_test', action='store_true')
    args = parser.parse_args()

    runs = load_rows(args.root)
    cache = {}

    def load_split_data(dataset, split):
        if dataset not in cache:
            cache[dataset] = load_dataset(dataset, add_self_loops=True, device='cpu',
                                          data_dir=args.data_dir)
        data, _, val, test, _, _, is_binary = cache[dataset]
        return data, val[:, split], test[:, split], is_binary

    table = score_table(runs, compose_ensembles(runs, load_split_data))
    selected = choose_depth(table)
    scores, best = choose_configuration(selected, args.splits)
    if best.empty:
        raise SystemExit('No configuration is complete on all pilot splits yet.')
    reference = scores[(scores.config_name == grid.REFERENCE_NAME) & (scores.splits == args.splits)]
    decision = decide(best, reference)

    hidden = [] if args.reveal_test else ['test_metric']
    args.out.mkdir(parents=True, exist_ok=True)
    table.drop(columns=hidden).to_csv(args.out / 'per_depth.csv', index=False)
    scores.drop(columns=hidden).to_csv(args.out / 'config_scores.csv', index=False)
    best.drop(columns=hidden).to_csv(args.out / 'selected.csv', index=False)
    (args.out / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(best.drop(columns=hidden).to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
