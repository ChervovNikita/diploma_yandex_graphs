"""Main-table runs whose result can change under the frozen protocol.

    python -m ablation.fixed_protocol.capped_runs

The frozen protocol keeps the paper's patience, so a run that stopped by
patience before the old step limit would stop at the same step again. Only two
kinds of runs need training:

    step_limit  at least one model of the run stopped at the old limit
                (5000 steps; 20000 on Tolokers), read from the training logs;
    missing     the run is absent from the archived results.

An ensemble is retrained with all four members, because the archive keeps no
member predictions to rebuild it. The list is written once to rerun_list.csv
and committed, so the cluster does not need the archived logs.
"""

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from ablation.p08_training_steps.estimate import DATASET_FOLDERS, METHODS, PAPER_MODELS, archived_runs

KEYS = ['dataset', 'model', 'num_layers', 'split']
DEFAULT_OUT = Path(__file__).with_name('rerun_list.csv')


def rerun_list(results):
    frames = []
    for folder, method in product(DATASET_FOLDERS, METHODS):
        runs = pd.DataFrame(archived_runs(results, folder, method))
        frames.append(runs[runs.model.isin(PAPER_MODELS)])
    runs = pd.concat(frames, ignore_index=True)
    capped = runs[runs.members_at_limit > 0][KEYS + ['method', 'members_at_limit']].copy()
    capped['reason'] = 'step_limit'
    expected = pd.DataFrame(
        [dict(dataset=dataset, model=model, num_layers=depth, split=split, method=method)
         for dataset in runs.dataset.unique() for model, depth, split, method
         in product(PAPER_MODELS, range(1, 6), range(10), METHODS)])
    present = runs[KEYS + ['method']].assign(present=True)
    missing = expected.merge(present, on=KEYS + ['method'], how='left')
    missing = missing[missing.present.isna()].drop(columns='present')
    missing['members_at_limit'] = 0
    missing['reason'] = 'missing'
    table = pd.concat([capped, missing], ignore_index=True)
    return table.sort_values(['dataset', 'method', 'model', 'num_layers', 'split']).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=Path('results_merged'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    table = rerun_list(args.results)
    table.to_csv(args.out, index=False)
    print(table.groupby(['dataset', 'method', 'reason']).size().to_string())
    print(f'{len(table)} runs written to {args.out}')


if __name__ == '__main__':
    main()
