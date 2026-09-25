"""Summarize matched control runs by official graph split.

All numbers are descriptive. The five masks come from the same graph and are
not five independent datasets. Run after or during projector_controls.py.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments_iclr' / 'results'
SOURCE = RESULTS / 'projector_controls.csv'
SUMMARY = RESULTS / 'control_summary.csv'
PAIRED = RESULTS / 'paired_control_effects.csv'
SUMMARY_FIELDS = ['dataset', 'model', 'variant', 'n_splits', 'splits',
                  'mean_test_acc_percent', 'population_sd_test_acc_percent',
                  'mean_validation_acc_percent', 'mean_params_m',
                  'mean_member_acc_percent', 'mean_pooling_gain_pp',
                  'mean_pair_disagreement_percent']
PAIRED_FIELDS = ['dataset', 'model', 'comparison', 'n_paired_splits', 'splits',
                 'mean_paired_delta_pp', 'min_paired_delta_pp',
                 'max_paired_delta_pp', 'mean_member_delta_pp',
                 'mean_pooling_gain_delta_pp', 'mean_disagreement_delta_pp',
                 'split_wins', 'split_ties', 'split_losses']


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with SOURCE.open(newline='') as f:
        source = list(csv.DictReader(f))
    seen = set()
    groups = defaultdict(dict)
    for row in source:
        key = (row['dataset'], row['model'], row['variant'], int(row['split']))
        if key in seen:
            raise ValueError(f'Duplicate run: {key}')
        seen.add(key)
        groups[key[:3]][key[3]] = row
    summary = []
    paired = []
    dataset_models = sorted({key[:2] for key in groups})
    for dataset, model in dataset_models:
        variants = sorted(v for d, m, v in groups if (d, m) == (dataset, model))
        reference = groups.get((dataset, model, 'gnnm'), {})
        for variant in variants:
            runs = groups[(dataset, model, variant)]
            splits = sorted(runs)
            acc = np.asarray([float(runs[s]['test_acc']) for s in splits])
            val = np.asarray([float(runs[s]['val_metric']) for s in splits])
            member = np.asarray([float(runs[s]['mean_member_acc']) for s in splits])
            disagreement = np.asarray([float(runs[s]['pair_disagreement']) for s in splits])
            summary.append({
                'dataset': dataset, 'model': model, 'variant': variant,
                'n_splits': len(splits), 'splits': ' '.join(map(str, splits)),
                'mean_test_acc_percent': 100 * acc.mean(),
                'population_sd_test_acc_percent': 100 * acc.std(ddof=0),
                'mean_validation_acc_percent': 100 * val.mean(),
                'mean_params_m': np.mean([float(runs[s]['num_params']) for s in splits]) / 1e6,
                'mean_member_acc_percent': 100 * member.mean(),
                'mean_pooling_gain_pp': 100 * (acc.mean() - member.mean()),
                'mean_pair_disagreement_percent': 100 * disagreement.mean(),
            })
            if variant == 'gnnm':
                continue
            overlap = sorted(set(splits) & set(reference))
            if not overlap:
                continue
            delta = np.asarray([float(runs[s]['test_acc']) - float(reference[s]['test_acc'])
                                for s in overlap])
            member_delta = np.asarray([float(runs[s]['mean_member_acc'])
                                       - float(reference[s]['mean_member_acc'])
                                       for s in overlap])
            disagreement_delta = np.asarray([float(runs[s]['pair_disagreement'])
                                             - float(reference[s]['pair_disagreement'])
                                             for s in overlap])
            tied = np.isclose(delta, 0.0, atol=1e-12, rtol=0.0)
            paired.append({
                'dataset': dataset, 'model': model,
                'comparison': f'{variant}-gnnm',
                'n_paired_splits': len(overlap),
                'splits': ' '.join(map(str, overlap)),
                'mean_paired_delta_pp': 100 * delta.mean(),
                'min_paired_delta_pp': 100 * delta.min(),
                'max_paired_delta_pp': 100 * delta.max(),
                'mean_member_delta_pp': 100 * member_delta.mean(),
                'mean_pooling_gain_delta_pp': 100 * (delta - member_delta).mean(),
                'mean_disagreement_delta_pp': 100 * disagreement_delta.mean(),
                'split_wins': int(np.sum((delta > 0) & ~tied)),
                'split_ties': int(np.sum(tied)),
                'split_losses': int(np.sum((delta < 0) & ~tied)),
            })
    write_csv(SUMMARY, SUMMARY_FIELDS, summary)
    write_csv(PAIRED, PAIRED_FIELDS, paired)
    print(f'Wrote {len(summary)} variant summaries and {len(paired)} paired comparisons')


if __name__ == '__main__':
    main()
