"""Full training cost: measured step time multiplied by the archived number of steps.

    python -m ablation.p08_training_steps.estimate --cost_root P07_RUNS --out RESULT_DIR

Run where results_merged is available. The stop step comes from the training logs; a run
that never reached early stopping used its whole step limit. ENS costs one step time per
member multiplied by the steps of that member. Output files:

    per_run.csv            every archived run with its steps and estimated cost
    by_dataset.csv         steps and GPU-hours per dataset and method for the whole grid
    selected.csv           the depth chosen on validation for each split
    selected_summary.csv   quality and training cost of those selected configurations
    summary.json           GNNM relative to ENS, per dataset
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


# The seven backbones of the main table; TAG is reported only in the cost tables.
PAPER_MODELS = ['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep']
METHODS = {
    'base': ('base.csv', 'logs_base'),
    'ens': ('ensemble.csv', 'logs_ensemble'),
    'gnnm': ('tabm.csv', 'logs_tabm'),
}
DATASET_FOLDERS = ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']
ENSEMBLE_MEMBERS = 4

HEADER = re.compile(r'\b(\d+)L h=\d+ lr=\S+ split=(\d+)')
MEMBER = re.compile(r'Training model seed (\d+) \(seed=\d+\)')
STOP = re.compile(r'Early stopping at step (\d+)')


def step_limit(dataset):
    return 20000 if dataset == 'tolokers' else 5000


def parse_log(text):
    """Map (depth, split) to {member: stop step or None}; a later section replaces an earlier one."""
    sections = {}
    current, member = None, 0
    for line in text.splitlines():
        header = HEADER.search(line)
        if header:
            current = (int(header.group(1)), int(header.group(2)))
            sections[current] = {}
            member = 0
            continue
        if current is None:
            continue
        started = MEMBER.search(line)
        if started:
            member = int(started.group(1))
            sections[current][member] = None
            continue
        stopped = STOP.search(line)
        if stopped:
            sections[current][member] = int(stopped.group(1))
    return sections


def read_logs(folder):
    logs = {}
    for path in sorted(Path(folder).glob('*.log')):
        model = path.name.split('_')[0]
        for (depth, split), members in parse_log(path.read_text()).items():
            logs[(model, depth, split)] = members
    return logs


def member_best_steps(value):
    return [int(number) for number in re.findall(r'\d+', str(value))]


def archived_runs(results, folder, method):
    file_name, log_name = METHODS[method]
    frame = pd.read_csv(Path(results) / folder / file_name)
    logs = read_logs(Path(results) / folder / log_name)
    rows = []
    for row in frame.itertuples():
        limit = step_limit(row.dataset)
        best = member_best_steps(row.best_steps) if method == 'ens' else [int(row.best_step)]
        logged = logs.get((row.model, int(row.num_layers), int(row.split)))
        expected = ENSEMBLE_MEMBERS if method == 'ens' else 1
        if logged is None or (method == 'ens' and len(logged) != expected):
            stops, source = [min(step + 300, limit) for step in best], 'best_step_plus_patience'
        else:
            members = sorted(logged) if method == 'ens' else [0]
            logged = logged if logged else {0: None}
            stops = [logged[k] if logged[k] is not None else limit for k in members]
            source = 'log'
        rows.append({
            'dataset': row.dataset, 'model': row.model, 'method': method,
            'num_layers': int(row.num_layers), 'split': int(row.split),
            'val_metric': float(row.val_metric), 'test_metric': float(row.test_metric),
            'steps_to_best': sum(best) / len(best), 'steps_run': sum(stops) / len(stops),
            'member_steps_total': sum(stops), 'members_at_limit': sum(s >= limit for s in stops),
            'members': len(stops), 'stop_source': source,
        })
    return rows


def load_step_times(root):
    rows = [json.loads(path.read_text()) for path in sorted(Path(root).rglob('done.json'))]
    if not rows:
        raise SystemExit(f'No timing jobs under {root}')
    table = pd.DataFrame(rows)
    table = table[table.variant.isin(METHODS)]
    return (table.groupby(['dataset', 'model', 'variant', 'num_layers']).full_step_ms.mean()
            .reset_index().rename(columns={'variant': 'method'}))


def attach_cost(runs, step_times):
    frame = runs.merge(step_times, on=['dataset', 'model', 'method', 'num_layers'], how='left')
    # One ENS step time covers all four members; each member pays for its own steps only.
    per_member = frame.full_step_ms.where(frame.method != 'ens', frame.full_step_ms / ENSEMBLE_MEMBERS)
    frame['cost_hours'] = per_member * frame.member_steps_total / 1000 / 3600
    return frame


def select_depths(frame):
    keys = ['dataset', 'model', 'method', 'split']
    ordered = frame.sort_values(keys + ['num_layers'])
    return ordered.loc[ordered.groupby(keys, sort=False).val_metric.idxmax()]


def by_dataset(frame):
    return frame.groupby(['dataset', 'method']).agg(
        runs=('model', 'size'), mean_steps_to_best=('steps_to_best', 'mean'),
        mean_steps_run=('steps_run', 'mean'), runs_with_a_member_at_limit=('members_at_limit', lambda s: int((s > 0).sum())),
        total_hours=('cost_hours', 'sum'), mean_hours=('cost_hours', 'mean'),
    ).reset_index()


def selected_summary(selected):
    return selected.groupby(['dataset', 'method']).agg(
        test_metric=('test_metric', 'mean'), val_metric=('val_metric', 'mean'),
        mean_hours=('cost_hours', 'mean'), mean_steps_run=('steps_run', 'mean'),
    ).reset_index()


def relative_to_ensemble(grid, selected):
    result = {}
    for dataset in grid.dataset.unique():
        table = grid[grid.dataset == dataset].set_index('method')
        chosen = selected[selected.dataset == dataset].set_index('method')
        result[dataset] = {
            'gnnm_over_ens_steps_to_best': float(table.mean_steps_to_best['gnnm'] / table.mean_steps_to_best['ens']),
            'gnnm_over_ens_steps_run': float(table.mean_steps_run['gnnm'] / table.mean_steps_run['ens']),
            'gnnm_over_ens_grid_hours': float(table.total_hours['gnnm'] / table.total_hours['ens']),
            'gnnm_over_ens_selected_hours': float(chosen.mean_hours['gnnm'] / chosen.mean_hours['ens']),
            'gnnm_minus_ens_selected_test': float(chosen.test_metric['gnnm'] - chosen.test_metric['ens']),
            'gnnm_runs_at_limit': int(table.runs_with_a_member_at_limit['gnnm']),
            'ens_runs_with_a_member_at_limit': int(table.runs_with_a_member_at_limit['ens']),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cost_root', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=Path('results_merged'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    runs = pd.DataFrame([row for folder in DATASET_FOLDERS for method in METHODS
                         for row in archived_runs(args.results, folder, method)])
    runs = runs[runs.model.isin(PAPER_MODELS)]
    frame = attach_cost(runs, load_step_times(args.cost_root))
    missing = int(frame.full_step_ms.isna().sum())
    if missing:
        print(f'[warning] {missing} archived runs have no timing job and are left out')
        frame = frame[frame.full_step_ms.notna()]
    fallback = int((frame.stop_source != 'log').sum())
    if fallback:
        print(f'[warning] {fallback} runs have no log entry; their stop is best step + 300')

    grid = by_dataset(frame)
    selected = select_depths(frame)
    chosen = selected_summary(selected)
    summary = relative_to_ensemble(grid, chosen)

    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / 'per_run.csv', index=False)
    grid.to_csv(args.out / 'by_dataset.csv', index=False)
    selected.to_csv(args.out / 'selected.csv', index=False)
    chosen.to_csv(args.out / 'selected_summary.csv', index=False)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(grid.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
