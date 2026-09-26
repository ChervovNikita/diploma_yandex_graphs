"""Compare a timing-selected wider ordinary SAGE with four-member GNNM.

The width is loaded from the timing-only selection manifest. The script
requires the same five official masks in both result files and makes no
model-selection decisions from validation or test scores.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STANDARD = ROOT / 'experiments_iclr' / 'results' / 'projector_controls.csv'
STRONG = ROOT / 'experiments_iclr' / 'strong_base_results'
SELECTION = STRONG / 'strong_base_selection.json'
OUTPUT = STRONG / 'strong_base_comparison.json'
PAIRED = STRONG / 'strong_base_paired_scores.csv'
SPLITS = tuple(range(5))


def read_rows(path, variant, width):
    with path.open(newline='') as f:
        matches = [r for r in csv.DictReader(f)
                   if r['dataset'] == 'roman-empire' and r['model'] == 'SAGE'
                   and r['variant'] == variant and int(r['hidden_dim']) == width
                   and int(r['num_layers']) == 5 and float(r['lr']) == 3e-5]
    by_split = {int(r['split']): r for r in matches}
    if len(matches) != len(by_split) or tuple(sorted(by_split)) != SPLITS:
        raise ValueError(f'Expected one {variant} width {width} row per split 0–4 in {path}')
    return by_split


def main():
    selection = json.loads(SELECTION.read_text())
    width = int(selection['selected_width'])
    if width not in (896, 1024, 1152):
        raise ValueError(f'Unexpected timing-selected width: {width}')
    gnnm = read_rows(STANDARD, 'gnnm', 512)
    base = read_rows(STRONG / 'projector_controls.csv', 'base', width)
    rows = []
    for split in SPLITS:
        a, b = gnnm[split], base[split]
        rows.append({
            'split': split,
            'gnnm_val_acc_percent': 100 * float(a['val_metric']),
            'gnnm_test_acc_percent': 100 * float(a['test_acc']),
            'gnnm_train_seconds': float(a['train_seconds']),
            'gnnm_best_step': int(a['best_step']),
            'base_val_acc_percent': 100 * float(b['val_metric']),
            'base_test_acc_percent': 100 * float(b['test_acc']),
            'base_train_seconds': float(b['train_seconds']),
            'base_best_step': int(b['best_step']),
            'base_minus_gnnm_test_pp': 100 * (float(b['test_acc']) - float(a['test_acc'])),
        })
    with PAIRED.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        'dataset': 'roman-empire', 'model': 'SAGE', 'official_splits': list(SPLITS),
        'fixed_depth': 5, 'learning_rate': 3e-5,
        'gnnm_width': 512, 'base_width_selected_from_timing_only': width,
        'gnnm_num_params': int(selection['gnnm_num_params']),
        'base_num_params': int(selection['selected_num_params']),
        'gnnm_synchronized_step_ms': float(selection['target_step_ms']),
        'base_synchronized_step_ms': float(selection['selected_step_ms']),
        'base_to_gnnm_step_ratio': float(selection['selected_to_target_step_ratio']),
        'gnnm_mean_test_acc_percent': float(np.mean([r['gnnm_test_acc_percent'] for r in rows])),
        'base_mean_test_acc_percent': float(np.mean([r['base_test_acc_percent'] for r in rows])),
        'base_minus_gnnm_mean_paired_test_pp': float(np.mean([r['base_minus_gnnm_test_pp'] for r in rows])),
        'base_test_split_wins': sum(r['base_minus_gnnm_test_pp'] > 0 for r in rows),
        'base_test_split_ties': sum(r['base_minus_gnnm_test_pp'] == 0 for r in rows),
        'base_test_split_losses': sum(r['base_minus_gnnm_test_pp'] < 0 for r in rows),
        'gnnm_mean_full_training_seconds': float(np.mean([r['gnnm_train_seconds'] for r in rows])),
        'base_mean_full_training_seconds': float(np.mean([r['base_train_seconds'] for r in rows])),
        'scope_note': 'Five official masks share one graph and overlap. Paired scores are descriptive, not independent replicates. Full training seconds include validation and checkpoint I/O and are separate from synchronized step time.',
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
