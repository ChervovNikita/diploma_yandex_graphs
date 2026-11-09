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


KEYS = ['dataset', 'model', 'num_layers', 'split']


def load_method(results, method):
    frames = [pd.read_csv(Path(results) / folder / METHODS[method][0]) for folder in DATASET_FOLDERS]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.model.isin(PAPER_MODELS)][KEYS + ['val_metric', 'test_metric']]
    columns = {'val_metric': f'val_{method}', 'test_metric': f'test_{method}'}
    return frame.rename(columns=columns)


def paired_scores(results):
    merged = load_method(results, 'base')
    for method in ('ens', 'gnnm'):
        merged = merged.merge(load_method(results, method), on=KEYS, how='inner')
    return merged


def cell_means(paired):
    score_columns = [column for column in paired if column.startswith(('val_', 'test_'))]
    table = paired.groupby(['dataset', 'model', 'num_layers']).agg(
        splits=('split', 'nunique'), **{column: (column, 'mean') for column in score_columns},
    ).reset_index()
    table[score_columns] *= 100
    table['gnnm_minus_ens'] = table.test_gnnm - table.test_ens
    table['gnnm_minus_base'] = table.test_gnnm - table.test_base
    table['ens_minus_base'] = table.test_ens - table.test_base
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=Path('results_merged'))
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'results')
    args = parser.parse_args()

    table = cell_means(paired_scores(args.results))
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
