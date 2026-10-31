"""Describe all fixed paired comparisons after completed BUDDY final evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

PHASE = Path(__file__).resolve().parents[1]
REMOTE = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
PACKET = PHASE / 'buddy_gpu77_postfamily_eval_preparation_v4'
FAMILY = PHASE / 'buddy_gpu77_resource_family_launcher_v3/root_family_v2'
ARMS = ('native1024', 'single256', 'factorized4', 'independent4', 'matched_single')
SEEDS = (0, 1, 2)
CONTROLS = ('single256', 'independent4', 'matched_single')


def require(value, message):
    if not value:
        raise ValueError(message)


def confined(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE.resolve()), 'Path outside project phase')
    return path


def sha(path):
    return hashlib.sha256(confined(path).read_bytes()).hexdigest()


def read(path):
    path = confined(path)
    require(path.suffix == '.json', 'JSON metadata only')
    return json.loads(path.read_text())


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid ' + name)
    return value


def t_critical_df2(coverage):
    require(0 < coverage < 1, 'Invalid coverage')
    # For df=2, P(|T| <= t) = t/sqrt(t*t + 2).
    return math.sqrt(2 * coverage * coverage / (1 - coverage * coverage))


def paired(values):
    require(len(values) == 3, 'Exactly three paired seeds required')
    for value in values:
        finite(value, 'difference')
    mean, sd = statistics.mean(values), statistics.stdev(values)
    se = sd / math.sqrt(3)
    positive, negative = sum(v > 0 for v in values), sum(v < 0 for v in values)
    n = positive + negative
    p = min(1., 2 * sum(math.comb(n, j) for j in range(min(positive, negative) + 1)) / (2 ** n)) if n else 1.
    return dict(differences_pp=values, paired_mean_pp=mean, sample_sd_pp=sd,
                range_pp=[min(values), max(values)],
                paired_t95_pp=[mean - t_critical_df2(.95) * se, mean + t_critical_df2(.95) * se],
                bonferroni_family95_pp=[mean - t_critical_df2(1 - .05 / 3) * se,
                                       mean + t_critical_df2(1 - .05 / 3) * se],
                positive_pairs=positive, negative_pairs=negative, zero_pairs=3 - n,
                sign_reference_p=p)


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    result = [None] * len(pvalues)
    previous = 0.
    for rank, position in enumerate(order):
        previous = max(previous, min(1., (len(pvalues) - rank) * pvalues[position]))
        result[position] = previous
    return result


def collect():
    terminal_path = FAMILY / 'FAMILY_LAUNCH_RECEIPT.json'
    terminal = read(terminal_path)
    require(terminal.get('schema') == 'buddy77-family-launch-receipt-v2'
            and type(terminal.get('exit_code')) is int and terminal['exit_code'] == 0,
            'Successful family terminal required')
    receipt_path = PACKET / 'root_eval_v1/EVALUATION_RECEIPT.json'
    receipt = read(receipt_path)
    require(receipt.get('schema') == 'buddy77-postfamily-evaluation-receipt-v2'
            and receipt.get('status') == 'all15_locked_cells_scored_once'
            and type(receipt.get('scoring_exit_code')) is int
            and receipt.get('scoring_exit_code') == 0,
            'Completed fifteen-cell evaluation required')
    lock_path = PACKET / 'root_lock_v1/FAMILY_LOCK.json'
    audit_path = PACKET / 'root_lock_v1/LOCK_AUDIT.json'
    require(receipt['family_lock_sha256'] == sha(lock_path)
            and receipt['lock_audit_sha256'] == sha(audit_path)
            and receipt['terminal_family_receipt_sha256'] == sha(terminal_path),
            'Evaluation lineage changed')
    audit = read(audit_path)
    require(audit.get('status') == 'all15_production_lock_and_selected_checkpoints_audited'
            and audit.get('all_selected_checkpoints_runtime_validated') is True
            and type(audit.get('exit_code')) is int
            and audit.get('exit_code') == 0, 'Completed selected-artifact audit required')
    lock = read(lock_path)
    require(lock.get('schema') == 'buddy-family-lock-v2'
            and lock.get('status') == 'family_locked', 'Locked cohort required')
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    rows = lock['runs']
    results = receipt['final_results']
    require(audit.get('family_lock_sha256') == sha(lock_path)
            and audit.get('locked_runs') == rows, 'Audit belongs to a different lock/cohort')
    test_manifest_path = PHASE / 'buddy_complete_data_cache_preparation_v3/root_run_77_v1/cache/test_manifest.json'
    test_manifest_hash = sha(test_manifest_path)
    require(receipt['final_cache_qualification']['test_manifest_sha256'] == test_manifest_hash,
            'Qualified final-test metadata changed')
    test_manifest = read(test_manifest_path)
    require(test_manifest['family_lock_sha256'] == sha(lock_path), 'Final-test metadata belongs to another lock')
    require(len(rows) == len(results) == 15
            and {(r['arm'], r['seed']) for r in rows} == expected
            and {(r['arm'], r['seed']) for r in results} == expected,
            'Missing, duplicated or unexpected arm/seed cell')
    hashes = {(r['arm'], r['seed']): r['result_sha256'] for r in results}
    scores, inputs = {}, []
    for row in rows:
        key = row['arm'], row['seed']
        remote = Path(row['run_directory'])
        require(remote.is_absolute() and '..' not in remote.parts and remote.is_relative_to(REMOTE),
                'Unexpected scientific run path')
        path = confined(PHASE / remote.relative_to(REMOTE) / 'final_test.json')
        require(path == FAMILY / 'runs' / f'{row["arm"]}_seed{row["seed"]}' / 'final_test.json',
                'Unexpected family output')
        require(sha(path) == hashes[key], 'Final-result bytes changed')
        result = read(path)
        require(result['arm'] == row['arm'] and type(result['seed']) is int
                and result['seed'] == row['seed']
                and result['family_lock_sha256'] == sha(lock_path)
                and result['test_manifest_sha256'] == test_manifest_hash
                and result['checkpoint_sha256'] == row['selected_checkpoint_sha256'],
                'Final result belongs to different artifacts')
        value = finite(result['hits50'], 'Hits@50')
        require(0 <= value <= 1, 'Hits@50 outside range')
        scores[key] = value
        inputs.append(dict(path=str(path.relative_to(PHASE)), sha256=sha(path)))
    for path in [terminal_path, receipt_path, lock_path, audit_path, test_manifest_path]:
        inputs.append(dict(path=str(path.relative_to(PHASE)), sha256=sha(path)))
    return scores, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = confined(PHASE / args.output)
    require(not out.exists(), 'Fresh report directory required')
    scores, inputs = collect()
    arms = {}
    for arm in ARMS:
        values = [100 * scores[arm, seed] for seed in SEEDS]
        arms[arm] = dict(seeds=list(SEEDS), hits50_percent=values,
                         mean_percent=statistics.mean(values), sample_sd_percent=statistics.stdev(values))
    contrasts = []
    for control in CONTROLS:
        values = [100 * (scores['factorized4', seed] - scores[control, seed]) for seed in SEEDS]
        contrasts.append(dict(candidate='factorized4', control=control, **paired(values)))
    adjusted = holm([row['sign_reference_p'] for row in contrasts])
    for row, value in zip(contrasts, adjusted):
        row['holm_sign_reference_p'] = value
    report = dict(schema='buddy-paired-descriptive-analysis-v1', UTC=datetime.now(timezone.utc).isoformat(),
                  arms=arms, contrasts=contrasts, source_sha256=sha(Path(__file__)),
                  plan_sha256=sha(Path(__file__).with_name('PLAN.md')), inputs=inputs,
                  conditional_scope='Three optimizer seeds on one fixed graph/split/negative pool.',
                  t_interval_assumption='Independent, approximately normal seed differences. Not checkable with n=3.',
                  sign_reference_assumption='Independent, equiprobable signs under a zero-median null. Not treatment randomization.',
                  graph_or_node_population_uncertainty=False,
                  labels_logits_checkpoints_or_models_read=False, additional_model_forwards=0,
                  original_scores_changed=False, superiority_or_novelty_certified=False)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    text = ['# Complete BUDDY pilot: paired summary', '',
            'All values below concern three optimizer seeds on the same graph and split.', '',
            '| Arm | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |',
            '|---|---:|---:|---:|---:|']
    for arm, row in arms.items():
        values = row['hits50_percent']
        text.append(f'| {arm} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {row["mean_percent"]:.3f} ± {row["sample_sd_percent"]:.3f} |')
    text += ['', 'Hits@50 is reported in percent. Differences are percentage points.', '',
             '| Factorized minus control | Paired differences | Mean | 95% t interval | Family 95% interval | Holm sign reference p |',
             '|---|---|---:|---|---|---:|']
    for row in contrasts:
        fmt = lambda values: ', '.join(f'{value:.3f}' for value in values)
        text.append(f'| {row["control"]} | {fmt(row["differences_pp"])} | {row["paired_mean_pp"]:.3f} | {fmt(row["paired_t95_pp"])} | {fmt(row["bonferroni_family95_pp"])} | {row["holm_sign_reference_p"]:.3f} |')
    text += ['', 'The t intervals require independent, approximately normal seed differences. '
             'Three seeds cannot check that assumption. Family intervals use Bonferroni over the three fixed contrasts. '
             'The sign reference assumes equiprobable independent signs under a zero-median null. '
             'Its smallest two-sided p-value with three nonzero pairs is 0.25. '
             'These summaries do not establish graph generalization, state of the art or a new method.', '']
    (out / 'REPORT.md').write_text('\n'.join(text))
    print(json.dumps(dict(report=str(out.relative_to(PHASE)), source_sha256=report['source_sha256'],
                          complete_cells=15, paired_contrasts=3, additional_model_forwards=0)))


if __name__ == '__main__':
    main()
