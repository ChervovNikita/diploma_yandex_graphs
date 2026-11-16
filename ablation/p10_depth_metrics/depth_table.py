"""Scores at every depth from the archived results; no training is needed.

    python -m ablation.p10_depth_metrics.depth_table --out RESULT_DIR

Scores are in percentage points. Gaps are computed on the splits that exist for all three
methods, so a missing run cannot change a comparison. Output files:

    per_depth.csv   mean validation and test score of every cell and depth
    by_depth.csv    mean gaps and GNNM win counts over backbones, per depth
    by_dataset.csv  the same, per dataset and depth
    by_model.csv    the same, per backbone and depth
"""

import argparse
from pathlib import Path

import pandas as pd

from ablation.p08_training_steps.estimate import DATASET_FOLDERS, METHODS, PAPER_MODELS
from ablation.p08_training_steps.history import read_results_csv


KEYS = ['dataset', 'model', 'num_layers', 'split']


def load_method(results, method):
    frames = [read_results_csv(Path(results) / folder / METHODS[method][0]) for folder in DATASET_FOLDERS]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.model.isin(PAPER_MODELS)][KEYS + ['val_metric', 'test_metric']]
    frame = frame.drop_duplicates(KEYS + ['val_metric', 'test_metric'])
    if frame.duplicated(KEYS).any():
        raise ValueError(f'Conflicting historical results for {method}; supply one authoritative source')
    columns = {'val_metric': f'val_{method}', 'test_metric': f'test_{method}'}
    return frame.rename(columns=columns)


def paired_scores(results):
    merged = load_method(results, 'base')
    for method in ('ens', 'gnnm'):
        merged = merged.merge(load_method(results, method), on=KEYS, how='inner')
    return merged


def cell_means(paired, role='test'):
    score_columns = [column for column in paired if column.startswith(('val_', 'test_'))]
    table = paired.groupby(['dataset', 'model', 'num_layers']).agg(
        splits=('split', 'nunique'), **{column: (column, 'mean') for column in score_columns},
    ).reset_index()
    table[score_columns] *= 100
    table['gnnm_minus_ens'] = table[f'{role}_gnnm'] - table[f'{role}_ens']
    table['gnnm_minus_base'] = table[f'{role}_gnnm'] - table[f'{role}_base']
    table['ens_minus_base'] = table[f'{role}_ens'] - table[f'{role}_base']
    return table


def summarise(table, group):
    return table.groupby(group).agg(
        cells=('gnnm_minus_ens', 'size'),
        gnnm_minus_ens=('gnnm_minus_ens', 'mean'),
        gnnm_minus_base=('gnnm_minus_base', 'mean'),
        ens_minus_base=('ens_minus_base', 'mean'),
        gnnm_beats_ens=('gnnm_minus_ens', lambda s: int((s > 0).sum())),
        gnnm_beats_base=('gnnm_minus_base', lambda s: int((s > 0).sum())),
    ).reset_index()


def pair_current_scores(scores, role='val'):
    metric = f'{role}_metric'
    required = set(KEYS + ['method', metric])
    if not required.issubset(scores):
        raise ValueError(f'Missing current score columns {sorted(required - set(scores))}')
    merged = None
    for method in ('base', 'ens', 'gnnm'):
        frame = scores[scores.method == method][KEYS + [metric]]
        if frame.duplicated(KEYS).any():
            raise ValueError(f'Duplicate score cells for {method}; select one config first')
        frame = frame.rename(columns={metric: f'{role}_{method}'})
        merged = frame if merged is None else merged.merge(frame, on=KEYS, how='inner', validate='one_to_one')
    if merged.empty:
        raise ValueError('No complete paired BASE/ENS/GNNM depth cells')
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--results', type=Path, help='Explicit historical score directory')
    sources.add_argument('--scores', type=Path, help='Current normalized per-depth score CSV')
    parser.add_argument('--config_name', help='Explicit common hyperparameter config for current scores')
    parser.add_argument('--role', choices=['val', 'test'], default='val')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()

    if args.scores is not None:
        scores = pd.read_csv(args.scores)
        if 'config_name' in scores:
            if args.config_name is not None:
                scores = scores[scores.config_name == args.config_name]
            elif scores.config_name.nunique() != 1:
                parser.error('--config_name is required when several configurations are present')
        table = cell_means(pair_current_scores(scores, args.role), role=args.role)
    else:
        table = cell_means(paired_scores(args.results), role=args.role)
    by_depth = summarise(table, ['num_layers'])
    by_dataset = summarise(table, ['dataset', 'num_layers'])
    by_model = summarise(table, ['model', 'num_layers'])
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / 'per_depth.csv', index=False)
    by_depth.to_csv(args.out / 'by_depth.csv', index=False)
    by_dataset.to_csv(args.out / 'by_dataset.csv', index=False)
    by_model.to_csv(args.out / 'by_model.csv', index=False)
    print(by_depth.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
