"""Retrospective validation-only depth-selection policy sensitivity.

No policy uses test values to select a depth. All results use the same archived
400-row-per-method, eight-backbone manifest as reproduce_main.py. The ten official masks of a
graph overlap, so global or split-0 depth selection can use validation labels
that are in another mask's test set. These are sensitivity analyses, not new
independent test datasets or a prospectively selected benchmark protocol.
"""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np

from reproduce_main import BACKBONES, METRIC, SOURCE, VARIANTS, read_csv_with_repeated_headers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments_iclr' / 'recomputed_main'
POLICIES = ('per_split_validation', 'mean_validation', 'split0_validation')
SPLIT_FIELDS = ['policy', 'dataset', 'backbone', 'variant', 'split',
                'selected_depth', 'test_metric', 'metric_name']
CELL_FIELDS = ['policy', 'dataset', 'backbone', 'variant', 'metric_name',
               'n_splits', 'mean_percent', 'population_sd_percent']
EFFECT_FIELDS = ['policy', 'dataset', 'comparison', 'mean_paired_delta_pp',
                 'backbone_wins', 'backbone_losses']


def choose_depth(scores):
    best = max(scores.values())
    return min(depth for depth, value in scores.items() if best - value <= 1e-12)


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    selected = []
    table = []
    score = {}
    for dataset, directory in SOURCE.items():
        for variant, filename in VARIANTS.items():
            rows = read_csv_with_repeated_headers(directory / filename)
            for backbone in BACKBONES:
                grid = {(int(r['num_layers']), int(float(r['split']))): r
                        for r in rows if r['model'] == backbone}
                if len(grid) != 50:
                    raise ValueError((dataset, variant, backbone, len(grid)))
                val = {(depth, split): float(grid[depth, split]['val_metric'])
                       for depth in range(1, 6) for split in range(10)}
                chosen = {
                    'per_split_validation': [choose_depth({d: val[d, s]
                                                            for d in range(1, 6)})
                                             for s in range(10)],
                    'mean_validation': [choose_depth({d: np.mean([val[d, s]
                                                                  for s in range(10)])
                                                       for d in range(1, 6)})] * 10,
                    'split0_validation': [choose_depth({d: val[d, 0]
                                                        for d in range(1, 6)})] * 10,
                }
                for policy in POLICIES:
                    tests = []
                    for split, depth in enumerate(chosen[policy]):
                        result = float(grid[depth, split][METRIC[dataset]])
                        tests.append(result)
                        score[policy, dataset, backbone, variant, split] = result
                        selected.append({
                            'policy': policy, 'dataset': dataset,
                            'backbone': backbone, 'variant': variant,
                            'split': split, 'selected_depth': depth,
                            'test_metric': result,
                            'metric_name': METRIC[dataset].removeprefix('test_'),
                        })
                    arr = np.asarray(tests)
                    table.append({
                        'policy': policy, 'dataset': dataset,
                        'backbone': backbone, 'variant': variant,
                        'metric_name': METRIC[dataset].removeprefix('test_'),
                        'n_splits': len(arr), 'mean_percent': 100 * arr.mean(),
                        'population_sd_percent': 100 * arr.std(ddof=0),
                    })
    effects = []
    for policy, dataset, (better, reference) in product(
            POLICIES, SOURCE, (('GNNM', 'ENS'), ('GNNM', 'BASE'), ('ENS', 'BASE'))):
        by_backbone = [np.mean([score[policy, dataset, backbone, better, s]
                                - score[policy, dataset, backbone, reference, s]
                                for s in range(10)]) for backbone in BACKBONES]
        effects.append({
            'policy': policy, 'dataset': dataset,
            'comparison': f'{better}-{reference}',
            'mean_paired_delta_pp': 100 * float(np.mean(by_backbone)),
            'backbone_wins': sum(value > 0 for value in by_backbone),
            'backbone_losses': sum(value < 0 for value in by_backbone),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'depth_policy_per_split.csv', SPLIT_FIELDS, selected)
    write_csv(OUT / 'depth_policy_cell_summary.csv', CELL_FIELDS, table)
    write_csv(OUT / 'depth_policy_dataset_effects.csv', EFFECT_FIELDS, effects)
    print(f'Wrote {len(selected)} selections, {len(table)} cells, and {len(effects)} dataset effects')
    for policy in POLICIES:
        vals = [float(r['mean_paired_delta_pp']) for r in effects
                if r['policy'] == policy and r['comparison'] == 'GNNM-ENS']
        print(policy, 'GNNM-ENS dataset mean pp', np.mean(vals),
              'dataset wins', sum(value > 0 for value in vals))


if __name__ == '__main__':
    main()
