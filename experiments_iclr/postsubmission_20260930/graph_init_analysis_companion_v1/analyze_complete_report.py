"""Uncertainty/cost companion; only complete admitted report and closed cohort."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy')
SETTINGS = ('Squirrel', 'Photo')
SEEDS = (17, 29, 43)
REMOTE_PHASE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(values):
    require(len(values) == 3 and all(math.isfinite(v) for v in values), 'Three finite paired differences required')
    mean, sd = statistics.mean(values), statistics.stdev(values)
    # Closed-form 97.5% quantile of t with df2, rather than a new dependency.
    t975_df2 = math.sqrt(2 * .95**2 / (1 - .95**2))
    half = t975_df2 * sd / math.sqrt(3)
    positive = sum(v > 0 for v in values)
    negative = sum(v < 0 for v in values)
    n = positive + negative
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n) if n else 1.
    return {'paired_mean': mean, 'minimum': min(values), 'maximum': max(values),
            'sample_sd': sd, 'paired_t95_low': mean - half, 'paired_t95_high': mean + half,
            'positive_pairs': positive, 'negative_pairs': negative, 'zero_pairs': 3 - n,
            'sign_reference_p': p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--coordinator-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    require(report.get('schema') == 'graph-init-narrow-final-report-v1'
            and report.get('final_labels_selected_nothing') is True
            and report.get('all_fallbacks_retained') is True
            and report.get('pooling') == 'softmax(mean raw member logits)', 'Admitted complete report contract differs')
    expected = {(g, s, a) for g in SETTINGS for s in SEEDS for a in ARMS}
    keys = [(r['graph'], r['seed'], r['arm']) for r in report['rows']]
    require(len(keys) == 30 and set(keys) == expected, 'Exact 30-row cohort required')
    pairs = report['paired_graph_differences']
    pair_keys = [(r['graph'], r['seed'], r['comparator']) for r in pairs]
    require(len(pair_keys) == 24 and set(pair_keys) == expected - {(g, s, 'graph') for g in SETTINGS for s in SEEDS},
            'Exact 24 paired contrasts required')
    source_rows = {(r['graph'], r['seed'], r['arm']): r for r in report['rows']}
    for r in pairs:
        graph = source_rows[(r['graph'], r['seed'], 'graph')]['metrics']
        control = source_rows[(r['graph'], r['seed'], r['comparator'])]['metrics']
        for metric, field in [('primary_nll', 'graph_minus_control_nll'), ('primary_accuracy', 'graph_minus_control_accuracy')]:
            require(abs((graph[metric] - control[metric]) - r[field]) < 1e-12, 'Paired difference does not match retained metrics')

    statistics_rows = []
    for g in SETTINGS:
        for arm in ARMS[1:]:
            chosen = sorted((r for r in pairs if r['graph'] == g and r['comparator'] == arm), key=lambda r: r['seed'])
            for metric, field in [('NLL', 'graph_minus_control_nll'), ('accuracy', 'graph_minus_control_accuracy')]:
                values = [r[field] for r in chosen]
                statistics_rows.append({'graph': g, 'comparator': arm, 'metric': metric,
                    'differences_in_seed_order': values, **summaries(values)})
    primary = [r for r in statistics_rows if r['metric'] == 'NLL']
    previous = 0.
    for rank, row in enumerate(sorted(primary, key=lambda r: r['sign_reference_p'])):
        previous = max(previous, min(1., (8 - rank) * row['sign_reference_p']))
        row['sign_reference_holm_p'] = previous
        row['sign_reference_holm_reject_at_05'] = previous <= .05

    receipts = list(args.coordinator_run.glob('*_COMPLETED.json'))
    require(len(receipts) == 72, 'All 72 supervised phase completions required')
    # Both local and authorized remote trees have a common phase-relative layout.
    phase_root = args.coordinator_run.resolve().parents[1]
    counts = {}
    cost_rows = []
    seen = set()
    for path in receipts:
        key = path.name.removesuffix('_COMPLETED.json')
        record = json.loads(path.read_text())
        claim = json.loads((args.coordinator_run / (key + '_LAUNCH_CLAIM.json')).read_text())['attempt']
        require(key not in seen and claim['key'] == key, 'Repeated or mismatched phase')
        seen.add(key)
        terminal_record = record['phase_terminal']
        require(terminal_record['path'].startswith(REMOTE_PHASE), 'Unexpected remote terminal scope')
        terminal_path = phase_root / terminal_record['path'][len(REMOTE_PHASE):]
        terminal = json.loads(terminal_path.read_text())
        require(sha(terminal_path) == terminal_record['sha256'] and terminal.get('completed') is True
                and terminal.get('final_labels_read') is False, 'Unverified or failed phase terminal')
        seconds = record['whole_supervised_seconds']
        require(math.isfinite(seconds) and seconds >= 0, 'Invalid measured duration')
        context = next(g for g in SETTINGS if '/' + g + '_seed' in claim['output'])
        counts[claim['phase']] = counts.get(claim['phase'], 0) + 1
        cost_rows.append({'setting': context, 'phase': claim['phase'], 'arm': claim.get('arm'),
                          'key': key, 'outer_supervised_seconds': seconds})
    require(counts == {'qualify': 6, 'warm': 6, 'initialize': 30, 'fit': 30}, 'Phase count differs')
    cost_summary = []
    for g in SETTINGS:
        shared = sum(r['outer_supervised_seconds'] for r in cost_rows if r['setting'] == g and r['phase'] in ('qualify', 'warm'))
        for arm in ARMS:
            own = sum(r['outer_supervised_seconds'] for r in cost_rows if r['setting'] == g and r['arm'] == arm)
            cost_summary.append({'setting': g, 'arm': arm, 'shared_qualify_warm_seconds': shared,
                'arm_init_fit_seconds': own, 'standalone_registered_phase_replay_seconds': shared + own})
    output = args.output.resolve()
    require(not output.exists(), 'New output directory required')
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'gnnm-complete-initializer-uncertainty-cost-companion-v1',
        'report_sha256': sha(args.report), 'source_sha256': sha(Path(__file__)),
        'plan_sha256': sha(Path(__file__).with_name('PLAN.md')), 'statistics': statistics_rows,
        'costs': cost_summary, 'cohort_outer_supervised_seconds': sum(r['outer_supervised_seconds'] for r in cost_rows),
        'cost_scope': '72 registered phases only; setup, acquisition, prior failures and final scoring excluded',
        'inference_scope': 'conditional exploratory context summaries; not a new-graph confidence claim',
        'sign_reference_assumptions': 'independent symmetric signs; not a randomized causal test',
        'new_originality_or_superiority_claim': False, 'labels_checkpoints_logits_opened': False}
    (output / 'ANALYSIS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'PAIRED_DIFFERENCES.csv').open('w', newline='') as stream:
        fields = ['graph', 'seed', 'comparator', 'graph_minus_control_nll', 'graph_minus_control_accuracy']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps({'output': str(output), 'primary_contrasts': 8, 'secondary_contrasts': 8, 'complete_phases': 72}))


if __name__ == '__main__':
    main()
