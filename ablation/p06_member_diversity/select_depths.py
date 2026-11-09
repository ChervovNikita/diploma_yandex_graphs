"""Freeze the winning depth of every GNNM and ENS cell on split 0.

    python -m ablation.p06_member_diversity.select_depths

The result, selected_depths.csv, is stored next to this file so that the runner does
not need the archived results. The depth is chosen on validation, as in Table 1. When
split 0 is absent from the archive, the depth with the best mean validation score over
the available splits is used and the row is marked in the `source` column.
"""

import argparse
from pathlib import Path

import pandas as pd

from ablation.common.study import PAPER_MODELS


SPLIT = 0
METHOD_FILES = {'gnnm': 'tabm.csv', 'ens': 'ensemble.csv'}
DATASET_FOLDERS = ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']
SELECTED_PATH = Path(__file__).parent / 'selected_depths.csv'


def select_cell(frame):
    """Return the chosen row of one dataset, backbone and method."""
    on_split = frame[frame.split == SPLIT]
    if len(on_split):
        best = on_split.sort_values('num_layers').set_index('num_layers').val_metric.idxmax()
        row = on_split[on_split.num_layers == best].iloc[0]
        return {'num_layers': int(best), 'source': 'split0_validation',
                'archive_val_metric': float(row.val_metric),
                'archive_test_metric': float(row.test_metric)}
    means = frame.groupby('num_layers').val_metric.mean().sort_index()
    return {'num_layers': int(means.idxmax()), 'source': 'mean_validation_other_splits',
            'archive_val_metric': float('nan'), 'archive_test_metric': float('nan')}


def select_depths(results_root):
    rows = []
    for folder in DATASET_FOLDERS:
        for method, name in METHOD_FILES.items():
            frame = pd.read_csv(Path(results_root) / folder / name)
            for model in PAPER_MODELS:
                cell = frame[frame.model == model]
                if cell.empty:
                    raise SystemExit(f'No archived runs for {folder}/{method}/{model}')
                rows.append({'dataset': cell.dataset.iloc[0], 'model': model, 'method': method,
                             **select_cell(cell)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=Path('results_merged'))
    parser.add_argument('--out', type=Path, default=SELECTED_PATH)
    args = parser.parse_args()
    selected = select_depths(args.results)
    selected.to_csv(args.out, index=False)
    print(selected.to_string(index=False))


if __name__ == '__main__':
    main()
