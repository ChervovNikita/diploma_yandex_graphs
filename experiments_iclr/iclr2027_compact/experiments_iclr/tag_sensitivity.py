"""Independent TAG cross-check from the complete archived fixed grid.

TAG is included in the eight-backbone main reconstruction. This script
recomputes its five rows separately using the same validation-only depth rule.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import reproduce_main as main


OUT = Path(__file__).resolve().parent / 'recomputed_main' / 'tag_sensitivity.csv'
FIELDS = (
    'dataset', 'backbone', 'metric_name', 'n_splits',
    'base_mean_percent', 'base_population_sd_percent',
    'ens_mean_percent', 'ens_population_sd_percent',
    'gnnm_mean_percent', 'gnnm_population_sd_percent',
    'gnnm_minus_ens_mean_pp', 'gnnm_minus_ens_min_pp',
    'gnnm_minus_ens_max_pp', 'gnnm_minus_ens_split_wins',
    'gnnm_minus_base_mean_pp', 'gnnm_minus_base_split_wins',
)


def select(path: Path, metric: str) -> np.ndarray:
    rows = main.read_csv_with_repeated_headers(path)
    expected = {(model, depth, split)
                for model in main.FULL_GRID_BACKBONES
                for depth in range(1, 6) for split in range(10)}
    observed = {(r['model'], int(r['num_layers']), int(float(r['split'])))
                for r in rows}
    if len(rows) != 400 or observed != expected:
        raise ValueError(f'Incomplete fixed grid: {path}')
    if any(int(float(r['hidden_dim'])) != 512 or float(r['lr']) != 3e-5
           for r in rows):
        raise ValueError(f'Width or learning rate differs: {path}')
    scores = []
    for split in range(10):
        candidates = [r for r in rows
                      if r['model'] == 'TAG' and int(float(r['split'])) == split]
        highest = max(float(r['val_metric']) for r in candidates)
        tied = [r for r in candidates
                if abs(float(r['val_metric']) - highest) <= 1e-12]
        chosen = min(tied, key=lambda r: int(r['num_layers']))
        scores.append(float(chosen[metric]))
    return np.asarray(scores)


def main_entry() -> None:
    rows = []
    for dataset, directory in main.SOURCE.items():
        selected = {variant: select(directory / filename, main.METRIC[dataset])
                    for variant, filename in main.VARIANTS.items()}
        base, ens, gnnm = (selected[name] for name in ('BASE', 'ENS', 'GNNM'))
        versus_ens = 100 * (gnnm - ens)
        versus_base = 100 * (gnnm - base)
        rows.append({
            'dataset': dataset,
            'backbone': 'TAG',
            'metric_name': main.METRIC[dataset].removeprefix('test_'),
            'n_splits': 10,
            'base_mean_percent': 100 * base.mean(),
            'base_population_sd_percent': 100 * base.std(ddof=0),
            'ens_mean_percent': 100 * ens.mean(),
            'ens_population_sd_percent': 100 * ens.std(ddof=0),
            'gnnm_mean_percent': 100 * gnnm.mean(),
            'gnnm_population_sd_percent': 100 * gnnm.std(ddof=0),
            'gnnm_minus_ens_mean_pp': versus_ens.mean(),
            'gnnm_minus_ens_min_pp': versus_ens.min(),
            'gnnm_minus_ens_max_pp': versus_ens.max(),
            'gnnm_minus_ens_split_wins': int(np.sum(versus_ens > 0)),
            'gnnm_minus_base_mean_pp': versus_base.mean(),
            'gnnm_minus_base_split_wins': int(np.sum(versus_base > 0)),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} TAG sensitivity rows to {OUT}')


if __name__ == '__main__':
    main_entry()
