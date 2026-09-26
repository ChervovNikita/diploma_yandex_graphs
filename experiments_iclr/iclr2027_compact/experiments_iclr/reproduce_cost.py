"""Reconstruct the archived cost table from measured per-configuration CSVs."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ablation' / 'results_cost'
OUT = ROOT / 'experiments_iclr' / 'recomputed_main' / 'cost_table_recomputed.csv'
FIELDS = ['dataset', 'variant', 'included_backbones', 'n_configurations',
          'mean_params_m', 'mean_step_ms', 'mean_peak_allocated_mb']


def main():
    rows = []
    for path in sorted(SOURCE.glob('*.csv')):
        with path.open(newline='') as f:
            source = list(csv.DictReader(f))
        if len(source) != 200:
            raise ValueError(f'{path}: expected 200 measured rows, found {len(source)}')
        models = {r['model'] for r in source}
        if len(models) != 8:
            raise ValueError(f'{path}: expected eight backbones, got {models}')
        for subset_name, keep in [('all_eight', models), ('without_TAG', models - {'TAG'})]:
            for variant in ('BASE', 'ENS_k4', 'TABM_k2', 'TABM_k4', 'TABM_k8'):
                selected = [r for r in source if r['variant'] == variant
                            and r['model'] in keep]
                expected = len(keep) * 5
                if len(selected) != expected:
                    raise ValueError(f'{path}: {variant} has {len(selected)} rows, expected {expected}')
                mean = lambda col: sum(float(r[col]) for r in selected) / len(selected)
                rows.append({
                    'dataset': path.stem,
                    'variant': variant,
                    'included_backbones': subset_name,
                    'n_configurations': len(selected),
                    'mean_params_m': mean('num_params') / 1e6,
                    'mean_step_ms': mean('mean_step_ms'),
                    'mean_peak_allocated_mb': mean('peak_mem_mb'),
                })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
