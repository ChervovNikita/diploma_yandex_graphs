"""Paired uncertainty and measured costs after complete, admitted reporting."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

PHASE = Path(__file__).resolve().parents[1]
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
REGISTRY = PHASE / 'graph_init_precision_execution_root_v2/study_v2/GRAPH_INIT_ATTEMPT_REGISTRY.json'
REGISTRY_SHA = '715c361c82b543d4c436a565ccaa86470a9204433998d410954f2f268300307f'
COORDINATOR = PHASE / 'graph_init_precision_continuation_v3_cap_binding/coordinator_run_v3'
ARMS = ('graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy')
SETTINGS = ('Squirrel', 'Photo')
SEEDS = (17, 29, 43)


def require(value, message):
    if not value:
        raise ValueError(message)


def confined(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE.resolve()), 'Path escapes project phase')
    return path


def sha(path):
    return hashlib.sha256(confined(path).read_bytes()).hexdigest()


def read(path):
    path = confined(path)
    require(path.suffix == '.json', 'Only JSON metadata may be opened')
    return json.loads(path.read_text())


def mirror(remote):
    value = Path(remote)
    require(value.is_absolute() and '..' not in value.parts and value.is_relative_to(REMOTE),
            'Unexpected remote project scope')
    return confined(PHASE / value.relative_to(REMOTE))


def bound(record):
    path = mirror(record['path'])
    require(path.suffix == '.json', 'Descriptors must reference JSON metadata')
    require(sha(path) == record['sha256'], 'Descriptor digest differs: ' + str(path))
    return path


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def verified_whole_cost(run, key, claim, record):
    outer = run / 'supervision' / (key + '_outer')
    whole_path = bound(record['whole_terminal'])
    require(whole_path == outer / 'TERMINAL.json', 'Cost terminal is not the canonical per-attempt outer terminal')
    whole = read(whole_path)
    start_path = outer / 'START.json'
    require(whole.get('schema') == 'gnnm-whole-process-bound-terminal-v1'
            and whole.get('START_sha256') == sha(start_path), 'Whole terminal changed START identity')
    start = read(start_path)
    request_path = bound(claim['request'])
    require(request_path == run / 'requests' / key / 'REQUEST.json', 'Canonical launch request required')
    request = read(request_path)
    request_ref = start['root_request']
    relative = Path(request_ref['path'])
    require(not relative.is_absolute() and '..' not in relative.parts
            and confined(PHASE / relative) == request_path
            and request_ref['sha256'] == sha(request_path), 'START does not bind this launch request')
    remote_outer = str(REMOTE / outer.relative_to(PHASE))
    remote_inner = str(REMOTE / (run / 'supervision' / (key + '_inner')).relative_to(PHASE))
    require(start.get('schema') == 'gnnm-whole-process-bound-start-v1'
            and start.get('ssh_destination') == 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
            and start['outer_output'] == remote_outer and start['inner_supervisor_directory'] == remote_inner
            and request['outer_supervisor_directory'] == remote_outer
            and request['supervisor_directory'] == remote_inner
            and request.get('root_admitted') is True and request['attempt'] == claim['attempt']
            and request['action'] == claim['attempt']['phase'] and request['output'] == claim['attempt']['output'],
            'Whole-process source request identity differs')
    cap = claim['attempt']['whole_cap_seconds']
    require(type(cap) in (int, float) and math.isfinite(cap) and cap > 0
            and cap == record['whole_cap_seconds'] == start['whole_cap_seconds'] == request['whole_cap_seconds'],
            'Whole-process cap binding differs')
    require(whole.get('complete') is True and whole.get('within_whole_cap') is True
            and whole.get('root_request_unchanged') is True and whole.get('timed_out') is False
            and whole.get('child_exit_code') == 0 and record.get('whole_process_costs_charged') is True,
            'Unverified whole-process completion')
    seconds = record['whole_supervised_seconds']
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= cap
            and seconds == whole['whole_supervised_seconds'], 'Cost disagrees with verified whole terminal')
    return seconds, whole_path, {'start_sha256': sha(start_path), 'request_sha256': sha(request_path)}


def summaries(values):
    require(len(values) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in values),
            'Three finite paired differences required')
    mean, sd = statistics.mean(values), statistics.stdev(values)
    t975_df2 = math.sqrt(2 * .95**2 / (1 - .95**2))
    half = t975_df2 * sd / math.sqrt(3)
    positive, negative = sum(v > 0 for v in values), sum(v < 0 for v in values)
    n = positive + negative
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n) if n else 1.
    return {'paired_mean': mean, 'minimum': min(values), 'maximum': max(values), 'sample_sd': sd,
            'paired_t95_low': mean - half, 'paired_t95_high': mean + half,
            'positive_pairs': positive, 'negative_pairs': negative, 'zero_pairs': 3 - n,
            'sign_reference_p': p}


def closed_cohort(run):
    require(confined(run) == COORDINATOR.resolve(), 'Exact registered coordinator required')
    require(not (run / 'FAILED.json').exists(), 'Failed coordinator cannot be analyzed')
    completed = read(run / 'COMPLETED.json')
    require(completed.get('schema') == 'graph-init-finite-coordinator-terminal-v1'
            and completed.get('completed') is True and completed.get('registered_successful_terminals') == 72
            and completed.get('compare_report_or_final_labels_executed') is False,
            'Genuine complete coordinator closure required before report access')
    require(sha(REGISTRY) == REGISTRY_SHA, 'Registered source cohort changed')
    registry = read(REGISTRY)
    attempts = registry['attempts']
    expected = {r['key']: r for r in attempts}
    receipts = sorted(run.glob('*_COMPLETED.json'))
    require(len(attempts) == len(expected) == len(receipts) == 72,
            'Exact 72 unique registered attempts and successful completions required')
    require({p.name.removesuffix('_COMPLETED.json') for p in receipts} == set(expected),
            'Completed phases differ from registered cohort')
    contexts = {object_hash(r): r['graph'] for r in registry['contexts']}
    require(len(contexts) == 6, 'Exact six registered contexts required')
    rows = []
    bindings = []
    for path in receipts:
        key = path.name.removesuffix('_COMPLETED.json')
        claim_path = run / (key + '_LAUNCH_CLAIM.json')
        launch_claim = read(claim_path)
        claim = launch_claim['attempt']
        attempt = expected[key]
        for field in ('key', 'context_sha256', 'phase', 'arm', 'output'):
            require(claim[field] == attempt[field], 'Launch disagrees with registered attempt')
        require(claim['context_sha256'] in contexts, 'Unknown phase context')
        record = read(path)
        terminal = read(bound(record['phase_terminal']))
        require(terminal.get('completed') is True and terminal.get('final_labels_read') is False,
                'Unverified or failed phase terminal')
        require(bound(record['phase_terminal']) == REGISTRY.parent / 'terminals' / (key + '.json'),
                'Wrong phase terminal identity')
        require(bound(terminal['claim']) == REGISTRY.parent / 'claims' / (key + '.json'),
                'Wrong phase claim identity')
        freeze = read(bound(terminal['freeze']))
        require(bound(terminal['freeze']).parent == mirror(attempt['output']) and freeze['phase'] == attempt['phase'],
                'Phase freeze differs from registered output')
        seconds, whole_path, whole_identity = verified_whole_cost(run, key, launch_claim, record)
        rows.append({'setting': contexts[attempt['context_sha256']], 'phase': attempt['phase'],
                     'arm': attempt['arm'], 'key': key, 'outer_supervised_seconds': seconds})
        bindings.append({'key': key, 'completion_sha256': sha(path), 'launch_sha256': sha(claim_path), **whole_identity,
                         'phase_terminal': record['phase_terminal'], 'whole_terminal': record['whole_terminal']})
    require(Counter(r['phase'] for r in rows) == {'qualify': 6, 'warm': 6, 'initialize': 30, 'fit': 30},
            'Complete phase counts differ')
    require(len({b['whole_terminal']['path'] for b in bindings}) == 72, 'Whole cost terminals must be distinct')
    for setting in SETTINGS:
        require(Counter(r['phase'] for r in rows if r['setting'] == setting) ==
                {'qualify': 3, 'warm': 3, 'initialize': 15, 'fit': 15}, 'Setting phase counts differ')
        for arm in ARMS:
            require(Counter(r['phase'] for r in rows if r['setting'] == setting and r['arm'] == arm) ==
                    {'initialize': 3, 'fit': 3}, 'Arm phase counts differ')
    return registry, rows, bindings


def validated_report(path, registry):
    path = confined(path)
    require(path == mirror(registry['final_report_output']) / 'FINAL_REPORT.json', 'Canonical final report required')
    report = read(path)
    require(report.get('schema') == 'graph-init-narrow-final-report-v1'
            and report.get('final_labels_selected_nothing') is True and report.get('all_fallbacks_retained') is True
            and report.get('pooling') == 'softmax(mean raw member logits)', 'Final report contract differs')
    admission = read(bound(report['report_admission']))
    require(admission.get('schema') == 'graph-init-final-report-admission-v1'
            and admission.get('execution_authorized') is True and admission.get('authorized_phase') == 'report'
            and admission['comparison_freeze'] == report['comparison_freeze'], 'Separate final-report admission differs')
    comparison_path = bound(report['comparison_freeze'])
    comparison = read(comparison_path)
    require(comparison_path == mirror(registry['comparison_output']) / 'COMPARISON_FREEZE.json'
            and comparison['schema'] == 'graph-init-comparison-freeze-v1'
            and comparison['source_selection_closed'] is True and comparison['complete_cells'] == 30
            and comparison['final_labels_read'] is False
            and bound(comparison['attempt_registry']) == REGISTRY, 'Source comparison closure differs')
    claim = read(bound(report['claim']))
    require(claim['report_admission'] == report['report_admission']
            and claim['comparison_freeze'] == report['comparison_freeze']
            and claim['report_output'] == registry['final_report_output'], 'Final report claim differs')
    expected = {(g, s, a) for g in SETTINGS for s in SEEDS for a in ARMS}
    keys = [(r['graph'], r['seed'], r['arm']) for r in report['rows']]
    require(len(keys) == 30 and set(keys) == expected, 'Exact 30-row cohort required')
    sources = {(r['graph'], r['seed'], r['arm']): r for r in comparison['rows']}
    require(len(comparison['rows']) == len(sources) == 30 and set(sources) == expected, 'Complete comparison rows required')
    for row in report['rows']:
        source = sources[(row['graph'], row['seed'], row['arm'])]
        require(all(row.get(k) == v for k, v in source.items()), 'Report changed a source selection')
        for metric in ('primary_nll', 'primary_accuracy'):
            value = row['metrics'][metric]
            require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite primary metric')
        require(row['metrics']['primary_nll'] >= 0 and 0 <= row['metrics']['primary_accuracy'] <= 1,
                'Invalid metric domain')
    pairs = report['paired_graph_differences']
    pair_keys = [(r['graph'], r['seed'], r['comparator']) for r in pairs]
    require(len(pair_keys) == 24 and set(pair_keys) == expected - {(g, s, 'graph') for g in SETTINGS for s in SEEDS},
            'Exact 24 paired contrasts required')
    source_rows = {(r['graph'], r['seed'], r['arm']): r for r in report['rows']}
    for row in pairs:
        graph = source_rows[(row['graph'], row['seed'], 'graph')]['metrics']
        control = source_rows[(row['graph'], row['seed'], row['comparator'])]['metrics']
        for metric, field in [('primary_nll', 'graph_minus_control_nll'), ('primary_accuracy', 'graph_minus_control_accuracy')]:
            require(type(row[field]) in (int, float) and math.isfinite(row[field])
                    and abs((graph[metric] - control[metric]) - row[field]) < 1e-12, 'Paired difference differs')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--coordinator-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = confined(args.output)
    require(not output.exists(), 'New project-confined output directory required')
    registry, cost_rows, bindings = closed_cohort(confined(args.coordinator_run))
    report = validated_report(args.report, registry)
    statistics_rows = []
    for setting in SETTINGS:
        for arm in ARMS[1:]:
            chosen = sorted((r for r in report['paired_graph_differences'] if r['graph'] == setting and r['comparator'] == arm),
                            key=lambda r: r['seed'])
            for metric, field in [('NLL', 'graph_minus_control_nll'), ('accuracy', 'graph_minus_control_accuracy')]:
                values = [r[field] for r in chosen]
                statistics_rows.append({'graph': setting, 'comparator': arm, 'metric': metric,
                                        'differences_in_seed_order': values, **summaries(values)})
    primary = [r for r in statistics_rows if r['metric'] == 'NLL']
    previous = 0.
    for rank, row in enumerate(sorted(primary, key=lambda r: r['sign_reference_p'])):
        previous = max(previous, min(1., (8 - rank) * row['sign_reference_p']))
        row['sign_reference_holm_p'] = previous
        row['sign_reference_holm_reject_at_05'] = previous <= .05
    costs = []
    for setting in SETTINGS:
        shared = sum(r['outer_supervised_seconds'] for r in cost_rows if r['setting'] == setting and r['phase'] in ('qualify', 'warm'))
        for arm in ARMS:
            own = sum(r['outer_supervised_seconds'] for r in cost_rows if r['setting'] == setting and r['arm'] == arm)
            costs.append({'setting': setting, 'arm': arm, 'shared_qualify_warm_seconds': shared,
                          'arm_init_fit_seconds': own, 'standalone_registered_phase_replay_seconds': shared + own})
    result = {'schema': 'gnnm-complete-initializer-uncertainty-cost-companion-v4',
              'report_sha256': sha(args.report), 'source_sha256': sha(Path(__file__)),
              'plan_sha256': sha(Path(__file__).with_name('PLAN.md')), 'registry_sha256': REGISTRY_SHA,
              'statistics': statistics_rows, 'costs': costs, 'phase_cost_rows': cost_rows, 'cost_bindings': bindings,
              'cohort_outer_supervised_seconds': sum(r['outer_supervised_seconds'] for r in cost_rows),
              'cost_scope': '72 registered phases only; setup, acquisition, prior failures and final scoring excluded',
              'inference_scope': 'conditional exploratory context summaries; not a new-graph confidence claim',
              'sign_reference_assumptions': 'independent symmetric signs; not a randomized causal test',
              'standalone_accounts_overlap_and_must_not_be_summed': True,
              'new_originality_or_superiority_claim': False, 'labels_checkpoints_logits_opened': False}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'ANALYSIS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'PAIRED_DIFFERENCES.csv').open('w', newline='') as stream:
        fields = ['graph', 'seed', 'comparator', 'graph_minus_control_nll', 'graph_minus_control_accuracy']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['paired_graph_differences'])
    print(json.dumps({'output': str(output), 'primary_contrasts': 8, 'secondary_contrasts': 8, 'complete_phases': 72}))


if __name__ == '__main__':
    main()
