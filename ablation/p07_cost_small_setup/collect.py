"""Cost tables and linear-scaling checks from the short timing runs.

    python -m ablation.p07_cost_small_setup.collect --root RUN_DIR --out RESULT_DIR

Output files:

    per_cell.csv     every measured job, with its time-versus-steps fit
    by_variant.csv   means over backbones and depths, and ratios to BASE and ENS, per dataset
    linearity.csv    linearity in the number of steps, per dataset and variant
    m_scaling.csv    linearity of GNNM cost in the number of members, per dataset
    summary.json     overall checks
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COST_COLUMNS = ['num_params', 'train_step_ms', 'forward_ms', 'metric_ms', 'full_step_ms',
                'inference_ms', 'train_peak_mb', 'inference_peak_mb']
CELL = ['dataset', 'model', 'num_layers', 'split']
SHARED_GPU_MB = 200.0


def load_jobs(root):
    rows = []
    for path in sorted(Path(root).rglob('done.json')):
        rows.append(json.loads(path.read_text()))
    if not rows:
        raise SystemExit(f'No finished jobs under {root}')
    return rows


def r_squared(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = ((y - (slope * x + intercept)) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return float(1 - residual / total) if total > 0 else 1.0, float(slope), float(intercept)


def step_linearity(series):
    """Fit cumulative time on the step count and extrapolate from the first half."""
    series = np.asarray(series, dtype=float)
    fit, _, _ = r_squared(np.arange(1, len(series) + 1), np.cumsum(series))
    half = len(series) // 2
    predicted = series[:half].sum() * len(series) / half
    actual = series.sum()
    return fit, float(100 * (predicted - actual) / actual)


def per_cell(rows):
    table = []
    for row in rows:
        fit, error = step_linearity(row['full_step_series_ms'])
        kept = {key: value for key, value in row.items() if key != 'full_step_series_ms'}
        table.append({**kept, 'time_vs_steps_r2': fit, 'half_run_prediction_error_pct': error})
    return pd.DataFrame(table)


def by_variant(frame):
    table = frame.groupby(['dataset', 'variant'])[COST_COLUMNS].mean()
    for reference, suffix in (('base', 'vs_base'), ('ens', 'vs_ens')):
        baseline = frame[frame.variant == reference].set_index(CELL)[COST_COLUMNS]
        indexed = frame.set_index(CELL)
        ratios = indexed[COST_COLUMNS].div(baseline.reindex(indexed.index))
        ratios = ratios.assign(variant=indexed.variant.values).reset_index()
        means = ratios.groupby(['dataset', 'variant'])[COST_COLUMNS].mean()
        table = table.join(means.add_suffix(f'_{suffix}'))
    return table.reset_index()


def linearity_table(frame):
    return frame.groupby(['dataset', 'variant']).agg(
        min_r2=('time_vs_steps_r2', 'min'),
        median_r2=('time_vs_steps_r2', 'median'),
        median_abs_error_pct=('half_run_prediction_error_pct', lambda s: float(s.abs().median())),
        max_abs_error_pct=('half_run_prediction_error_pct', lambda s: float(s.abs().max())),
        jobs=('variant', 'size'),
    ).reset_index()


def member_scaling(frame):
    """Fit train step time on M = 2, 4, 8 for every backbone and depth."""
    sizes = {'gnnm_m2': 2, 'gnnm': 4, 'gnnm_m8': 8}
    subset = frame[frame.variant.isin(sizes)]
    rows = []
    for cell, group in subset.groupby(CELL):
        if group.variant.nunique() < len(sizes):
            continue
        fit, slope, intercept = r_squared(group.variant.map(sizes), group.train_step_ms)
        rows.append({'dataset': cell[0], 'r2': fit, 'ms_per_member': slope, 'intercept_ms': intercept})
    if not rows:
        return pd.DataFrame(columns=['dataset', 'cells', 'min_r2', 'median_r2'])
    table = pd.DataFrame(rows)
    return table.groupby('dataset').agg(
        cells=('r2', 'size'), min_r2=('r2', 'min'), median_r2=('r2', 'median'),
        mean_ms_per_member=('ms_per_member', 'mean'), mean_intercept_ms=('intercept_ms', 'mean'),
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    frame = per_cell(load_jobs(args.root))
    variants = by_variant(frame)
    linearity = linearity_table(frame)
    scaling = member_scaling(frame)
    shared = int((frame.other_gpu_memory_mb > SHARED_GPU_MB).sum())
    summary = {
        'jobs': int(len(frame)),
        'devices': sorted(frame.device_name.unique().tolist()),
        'jobs_on_a_shared_gpu': shared,
        'worst_time_vs_steps_r2': float(frame.time_vs_steps_r2.min()),
        'worst_half_run_error_pct': float(frame.half_run_prediction_error_pct.abs().max()),
        'worst_member_scaling_r2': float(scaling.min_r2.min()) if len(scaling) else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / 'per_cell.csv', index=False)
    variants.to_csv(args.out / 'by_variant.csv', index=False)
    linearity.to_csv(args.out / 'linearity.csv', index=False)
    scaling.to_csv(args.out / 'm_scaling.csv', index=False)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if shared:
        print(f'[warning] {shared} jobs shared their GPU with another process; re-measure them.')
    if len(summary['devices']) > 1:
        print(f"[warning] jobs ran on different devices: {summary['devices']}")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
