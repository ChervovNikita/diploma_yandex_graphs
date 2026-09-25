"""Recompute the fixed-grid main comparison from the tracked CSV files.

The archived CSVs use per-split depth selection, unlike the current default
training runner settings. This script selects depth 1--5 for each
(dataset, backbone, variant, official split) by validation score, then reads
the corresponding test score once. Hidden width 512 and LR 3e-5 are fixed.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments_iclr' / 'recomputed_main'
SOURCE = {
    'roman-empire': ROOT / 'results2' / 'roman_empire',
    'amazon-ratings': ROOT / 'results2' / 'amazon_ratings',
    'minesweeper': ROOT / 'results' / 'minesweeper',
    'questions': ROOT / 'results' / 'questions',
    'tolokers': ROOT / 'results3' / 'tolokers',
}
BACKBONES = ['GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep', 'ResNet', 'TAG']
FULL_GRID_BACKBONES = BACKBONES
VARIANTS = {'BASE': 'base.csv', 'ENS': 'ensemble.csv', 'GNNM': 'tabm.csv'}
METRIC = {
    'roman-empire': 'test_acc',
    'amazon-ratings': 'test_acc',
    'minesweeper': 'test_roc_auc',
    'questions': 'test_roc_auc',
    'tolokers': 'test_roc_auc',
}


def read_csv_with_repeated_headers(path: Path):
    with path.open(newline='') as stream:
        raw = list(csv.reader(stream))
    header = next((row for row in raw if row and row[0] == 'dataset'), None)
    if header is None:
        raise ValueError(f'No CSV header in {path}')
    rows = []
    for line, fields in enumerate(raw, start=1):
        if not fields or fields[0] == 'dataset':
            continue
        if len(fields) != len(header):
            raise ValueError(f'{path}:{line} has {len(fields)} fields, expected {len(header)}')
        rows.append(dict(zip(header, fields)))
    return rows


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def signflip_p(values):
    """Exact one-sided random-sign p for five dataset-level effects."""
    observed = sum(values)
    total = 0
    extreme = 0
    for signs in product((-1, 1), repeat=len(values)):
        total += 1
        score = sum(s * x for s, x in zip(signs, values))
        extreme += score >= observed - 1e-14
    return float(extreme / total)


def main(output_root=None):
    global OUT
    if output_root is not None:
        OUT = Path(output_root)
    OUT.mkdir(parents=True, exist_ok=True)
    selected = []
    grouped = {}
    source_counts = {}
    source_manifest = []
    tie_cases = []
    for dataset, directory in SOURCE.items():
        for variant, filename in VARIANTS.items():
            path = directory / filename
            rows = read_csv_with_repeated_headers(path)
            expected_grid = {(name, depth, split) for name in FULL_GRID_BACKBONES
                             for depth in range(1, 6) for split in range(10)}
            actual_grid = {(r['model'], int(r['num_layers']), int(float(r['split'])))
                           for r in rows}
            if len(rows) != 400 or actual_grid != expected_grid:
                raise ValueError(f'{path}: not the full 8-backbone × 5-depth × 10-split grid')
            if any(int(float(r['hidden_dim'])) != 512 or float(r['lr']) != 3e-5
                   for r in rows):
                raise ValueError(f'{path}: width/LR differ from fixed grid')
            relpath = str(path.relative_to(ROOT))
            source_counts[relpath] = len(rows)
            source_manifest.append({
                'path': relpath,
                'rows': len(rows),
                'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            })
            for backbone in BACKBONES:
                for split in range(10):
                    candidates = [r for r in rows if r['model'] == backbone
                                  and int(float(r['split'])) == split]
                    depths = sorted(int(r['num_layers']) for r in candidates)
                    if depths != [1, 2, 3, 4, 5]:
                        raise ValueError(f'{path}: {backbone} split {split} depths {depths}')
                    if any(int(float(r['hidden_dim'])) != 512 or
                           float(r['lr']) != 3e-5 for r in candidates):
                        raise ValueError(f'{path}: non-fixed width/LR for {backbone} split {split}')
                    max_val = max(float(r['val_metric']) for r in candidates)
                    tied = [r for r in candidates
                            if abs(float(r['val_metric']) - max_val) <= 1e-12]
                    best = min(tied, key=lambda r: int(r['num_layers']))
                    if len(tied) > 1:
                        tie_cases.append({
                            'dataset': dataset, 'backbone': backbone,
                            'variant': variant, 'split': split,
                            'candidate_depths': sorted(int(r['num_layers']) for r in tied),
                            'chosen_depth': int(best['num_layers']),
                            'exact_float_winner_depth': int(max(
                                candidates, key=lambda r: float(r['val_metric'])
                            )['num_layers']),
                        })
                    score = float(best[METRIC[dataset]])
                    selected.append({
                        'dataset': dataset, 'backbone': backbone,
                        'variant': variant, 'split': split,
                        'selected_depth': int(best['num_layers']),
                        'val_metric': float(best['val_metric']),
                        'test_metric': score,
                        'metric_name': METRIC[dataset].removeprefix('test_'),
                        'source': str(path.relative_to(ROOT)),
                    })
                    grouped[(dataset, backbone, variant, split)] = score
    write_csv(OUT / 'per_split_selected.csv',
              ['dataset', 'backbone', 'variant', 'split', 'selected_depth',
               'val_metric', 'test_metric', 'metric_name', 'source'], selected)

    summary = []
    for dataset, backbone, variant in product(SOURCE, BACKBONES, VARIANTS):
        values = np.array([grouped[(dataset, backbone, variant, s)]
                           for s in range(10)])
        summary.append({
            'dataset': dataset, 'backbone': backbone, 'variant': variant,
            'metric_name': METRIC[dataset].removeprefix('test_'),
            'n_splits': 10,
            'mean_percent': 100 * values.mean(),
            'population_sd_percent': 100 * values.std(ddof=0),
        })
    write_csv(OUT / 'table_recomputed.csv',
              ['dataset', 'backbone', 'variant', 'metric_name', 'n_splits',
               'mean_percent', 'population_sd_percent'], summary)

    effects = []
    backbone_rows = []
    comparisons = [('GNNM', 'ENS'), ('GNNM', 'BASE'), ('ENS', 'BASE')]
    for dataset, (better, reference) in product(SOURCE, comparisons):
        backbone_effects = []
        for backbone in BACKBONES:
            split_effects = [grouped[(dataset, backbone, better, s)]
                             - grouped[(dataset, backbone, reference, s)]
                             for s in range(10)]
            backbone_effects.append(np.mean(split_effects))
            delta = np.asarray(split_effects)
            tied = np.isclose(delta, 0.0, atol=1e-12, rtol=0.0)
            backbone_rows.append({
                'dataset': dataset, 'backbone': backbone,
                'comparison': f'{better}-{reference}',
                'metric_name': METRIC[dataset].removeprefix('test_'),
                'n_splits': len(delta),
                'mean_paired_delta_pp': 100 * float(delta.mean()),
                'split_wins': int(np.sum((delta > 0) & ~tied)),
                'split_ties': int(np.sum(tied)),
                'split_losses': int(np.sum((delta < 0) & ~tied)),
            })
        effects.append({
            'dataset': dataset,
            'comparison': f'{better}-{reference}',
            'mean_paired_delta_pp': 100 * np.mean(backbone_effects),
            'backbone_wins': sum(x > 0 for x in backbone_effects),
            'backbone_losses': sum(x < 0 for x in backbone_effects),
        })
    write_csv(OUT / 'backbone_paired_effects.csv',
              ['dataset', 'backbone', 'comparison', 'metric_name', 'n_splits',
               'mean_paired_delta_pp', 'split_wins', 'split_ties', 'split_losses'],
              backbone_rows)
    write_csv(OUT / 'dataset_level_effects.csv',
              ['dataset', 'comparison', 'mean_paired_delta_pp',
               'backbone_wins', 'backbone_losses'], effects)

    appendix_summary = []
    for dataset in SOURCE:
        for better, reference in comparisons:
            comparison = f'{better}-{reference}'
            values = [row['mean_paired_delta_pp'] for row in backbone_rows
                      if row['dataset'] == dataset
                      and row['comparison'] == comparison]
            if len(values) != len(BACKBONES):
                raise ValueError(f'Incomplete appendix summary: {dataset} {comparison}')
            appendix_summary.append({
                'dataset': dataset, 'comparison': comparison,
                'metric_name': METRIC[dataset].removeprefix('test_'),
                'n_backbones': len(BACKBONES), 'n_splits_per_backbone': 10,
                'backbone_wins': sum(value > 1e-12 for value in values),
                'backbone_ties': sum(abs(value) <= 1e-12 for value in values),
                'backbone_losses': sum(value < -1e-12 for value in values),
                'mean_backbone_delta_pp': statistics.mean(values),
                'median_backbone_delta_pp': statistics.median(values),
                'minimum_backbone_delta_pp': min(values),
                'maximum_backbone_delta_pp': max(values),
            })
    write_csv(OUT / 'all_eight_candidate_summary.csv',
              list(appendix_summary[0]), appendix_summary)

    appendix_spread = []
    for dataset in SOURCE:
        mask_effects = []
        for split in range(10):
            deltas = [100 * (grouped[(dataset, backbone, 'GNNM', split)]
                             - grouped[(dataset, backbone, 'ENS', split)])
                      for backbone in BACKBONES]
            mask_effects.append(statistics.mean(deltas))
        appendix_spread.append({
            'dataset': dataset, 'comparison': 'GNNM-ENS',
            'n_backbones': len(BACKBONES), 'n_official_masks': 10,
            'mean_paired_delta_pp': statistics.mean(mask_effects),
            'minimum_mask_delta_pp': min(mask_effects),
            'maximum_mask_delta_pp': max(mask_effects),
            'mask_wins': sum(value > 0 for value in mask_effects),
        })
    write_csv(OUT / 'all_eight_candidate_dataset_spread.csv',
              list(appendix_spread[0]), appendix_spread)

    clustered_tests = {}
    for better, reference in comparisons:
        label = f'{better}-{reference}'
        values = [next(r['mean_paired_delta_pp'] for r in effects
                       if r['dataset'] == dataset and r['comparison'] == label)
                  for dataset in SOURCE]
        clustered_tests[label] = {
            'dataset_effects_pp': {name: float(value) for name, value in zip(SOURCE, values)},
            'mean_effect_pp': float(np.mean(values)),
            'dataset_wins': int(sum(x > 0 for x in values)),
            'exact_one_sided_signflip_p': signflip_p(values),
        }
    # A rank statistic across the 40 cells is descriptive only because eight
    # backbones share each dataset and its ten official splits.
    ranks = []
    for dataset, backbone in product(SOURCE, BACKBONES):
        score = np.array([np.mean([grouped[(dataset, backbone, variant, s)]
                                   for s in range(10)]) for variant in VARIANTS])
        rank = np.empty(3, dtype=int)
        rank[np.argsort(-score)] = [1, 2, 3]
        ranks.append(rank)
    ranks = np.stack(ranks)
    rank_means = dict(zip(VARIANTS, ranks.mean(axis=0).tolist()))
    friedman_q = 12 * len(ranks) / (3 * 4) * float(np.sum(ranks.mean(0) ** 2)) \
                 - 3 * len(ranks) * 4
    report = {
        'protocol': 'Archived per-split depth selection; width=512; lr=3e-5; '
                    '10 official splits; population SD across splits',
        'source_counts': source_counts,
        'num_selected_records': len(selected),
        'tie_rule': 'Validation scores within 1e-12 are tied; choose smaller depth',
        'tie_rule_provenance': 'Retrospective reconstruction rule; the original run manifest does not document ties',
        'tie_cases': tie_cases,
        'clustered_tests': clustered_tests,
        'descriptive_40_cell_mean_ranks': rank_means,
        'descriptive_40_cell_friedman_q': friedman_q,
        'warning': 'The 40 cells share five datasets. The Friedman statistic is '
                   'not a valid independent-task significance test here.',
    }
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    provenance = {
        'status': 'Retrospective reconstruction from archived CSV folders, not a prospectively recorded source manifest',
        'selection_basis': 'Among results/, results2/, and results3/, select for each dataset the unique folder containing base.csv, ensemble.csv, and tabm.csv with exactly 400 distinct rows each: eight backbones (including TAG), depths 1–5, ten official splits, width 512, and learning rate 3e-5. This inclusion rule uses setup and completeness, not test scores. results_grid/ is a separate broader search. The rule was identified retrospectively, and the archive does not establish whether the original authors used it before seeing test results.',
        'csv_parser_note': 'The first row beginning dataset supplies column names, wherever it occurs. Valid data rows before it are retained, and repeated header rows are skipped. This handles the leading data row in Minesweeper BASE and repeated headers in Tolokers BASE and GNNM.',
        'files': source_manifest,
    }
    (OUT / 'source_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=None,
                        help='Write reconstructed tables to this directory')
    args = parser.parse_args()
    main(args.output_root)
