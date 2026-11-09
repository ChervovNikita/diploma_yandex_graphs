"""Build result tables from finished jobs of one or more studies; safe at any moment.

    python -m ablation.common.collect --root RUN_DIR [RUN_DIR ...] --out RESULT_DIR

Depth is chosen on validation separately for each split, as in the main table.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


COLUMNS = ['test_metric', 'test_mean_member_metric', 'test_pair_disagreement',
           'test_error_jaccard', 'best_step', 'stopped_step', 'train_seconds', 'num_params']


def load_runs(roots):
    rows = [json.loads(path.read_text())
            for root in roots for path in sorted(Path(root).rglob('done.json'))]
    if not rows:
        raise SystemExit(f'No finished jobs under {roots}')
    return pd.DataFrame(rows)


def select_depth(runs):
    keys = ['dataset', 'variant', 'model', 'split']
    ordered = runs.sort_values(keys + ['num_layers'])
    return ordered.loc[ordered.groupby(keys, sort=False).val_metric.idxmax()]


def summarize(selected):
    grouped = selected.groupby(['dataset', 'variant', 'model'])
    summary = grouped[COLUMNS].mean()
    summary['test_metric_std'] = grouped.test_metric.std(ddof=0)
    summary['splits'] = grouped.split.nunique()
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    runs = load_runs(args.root)
    selected = select_depth(runs)
    summary = summarize(selected)
    args.out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out / 'all_runs.csv', index=False)
    selected.to_csv(args.out / 'selected_by_validation.csv', index=False)
    summary.to_csv(args.out / 'summary.csv', index=False)

    print(f'jobs={len(runs)} source_versions={runs.source_sha256.nunique()}')
    incomplete = summary[summary.splits < 10]
    if len(incomplete):
        print(f'WARNING: {len(incomplete)} cells have fewer than 10 splits')
    table = 100 * summary.pivot_table(index='model', columns='variant', values='test_metric')
    print(table.round(2).to_string())


if __name__ == '__main__':
    main()
