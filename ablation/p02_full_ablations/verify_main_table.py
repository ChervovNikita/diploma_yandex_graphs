"""Compare finished ``m4`` jobs with the archived GNNM runs of the main table.

Run this before the full launch:

    python -m ablation.p02_full_ablations.verify_main_table --root RUN_DIR \
        --reference results_merged/roman_empire/tabm.csv

Differences are expected to be small but not zero: GPU attention scatter is not
deterministic, and the archive was produced on different hardware.
"""

import argparse
from pathlib import Path

import pandas as pd

from ablation.common.collect import load_runs


KEYS = ['model', 'num_layers', 'split']


def compare(runs, reference):
    ours = runs[runs.variant == 'm4'][KEYS + ['best_step', 'val_metric', 'test_metric']]
    reference = reference[KEYS + ['best_step', 'val_metric', 'test_metric']]
    merged = ours.merge(reference, on=KEYS, suffixes=('', '_main'), how='left')
    merged['test_difference_points'] = 100 * (merged.test_metric - merged.test_metric_main)
    merged['val_difference_points'] = 100 * (merged.val_metric - merged.val_metric_main)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, nargs='+', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    merged = compare(load_runs(args.root), pd.read_csv(args.reference))
    missing = merged.test_metric_main.isna().sum()
    if missing:
        print(f'WARNING: {missing} runs have no archived counterpart')
    merged = merged.sort_values(KEYS)
    print(merged.round(5).to_string(index=False))
    summary = merged.groupby(['model', 'num_layers']).agg(
        runs=('split', 'nunique'),
        test_new=('test_metric', lambda x: 100 * x.mean()),
        test_main=('test_metric_main', lambda x: 100 * x.mean()),
        mean_difference=('test_difference_points', 'mean'),
        largest_difference=('test_difference_points', lambda x: x.abs().max()),
    )
    print(summary.round(3).to_string())
    complete = merged.dropna(subset=['val_metric_main', 'test_metric_main'])
    new = complete.loc[complete.groupby(['model', 'split']).val_metric.idxmax()]
    main_table = complete.loc[complete.groupby(['model', 'split']).val_metric_main.idxmax()]
    print('\nWith depth selected on validation for each split:')
    for model in sorted(new.model.unique()):
        ours, archived = new[new.model == model], main_table[main_table.model == model]
        print(f'{model}: new {100 * ours.test_metric.mean():.3f}, '
              f'main table {100 * archived.test_metric_main.mean():.3f}, '
              f'splits {ours.split.nunique()}, '
              f'same depth on {(ours.num_layers.values == archived.num_layers.values).sum()} splits')
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
