"""Recompute archived Roman Empire K and initialization ablation summaries."""
from __future__ import annotations

import csv
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments_iclr' / 'recomputed_main' / 'ablations_recomputed.csv'
SOURCES = {
    'members': sorted((ROOT / 'ablation' / 'results_k').glob('roman-empire_*.csv')),
    'initialization': sorted((ROOT / 'ablation' / 'results_init').glob('roman-empire_*.csv')),
}
BACKBONES = ('SAGE', 'GAT-sep', 'GT-sep')
FIELDS = ['study', 'setting', 'backbone', 'n_splits', 'mean_test_acc_percent',
          'sample_sd_test_acc_percent', 'source']


def main():
    rows = []
    for study, files in SOURCES.items():
        if len(files) != 3:
            raise ValueError(f'Expected three files for {study}, got {len(files)}')
        for path in files:
            with path.open(newline='') as f:
                source = list(csv.DictReader(f))
            if len(source) != 15:
                raise ValueError(f'{path}: expected 15 rows, got {len(source)}')
            for backbone in BACKBONES:
                selected = [r for r in source if r['model'] == backbone]
                if sorted(int(r['split']) for r in selected) != list(range(5)):
                    raise ValueError(f'{path}: missing or repeated split for {backbone}')
                if any(int(r['num_layers']) != 5 or int(r['hidden_dim']) != 512
                       or float(r['lr']) != 3e-5 for r in selected):
                    raise ValueError(f'{path}: unexpected setup for {backbone}')
                values = [100 * float(r['test_acc']) for r in selected]
                rows.append({
                    'study': study,
                    'setting': path.stem.removeprefix('roman-empire_'),
                    'backbone': backbone,
                    'n_splits': len(values),
                    'mean_test_acc_percent': statistics.mean(values),
                    'sample_sd_test_acc_percent': statistics.stdev(values),
                    'source': str(path.relative_to(ROOT)),
                })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} archived ablation cells to {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
