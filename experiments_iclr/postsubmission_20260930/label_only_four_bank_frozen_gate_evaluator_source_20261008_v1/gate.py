"""Disabled exact frozen gate; descriptive intervals across three paired seeds."""
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time

HERE = Path(__file__).resolve().parent
SEEDS = (6101, 6203, 6307)
ARMS = ('C4', 'S_joint4head', 'U4_sharedB', 'S_one_path')
MEMBERS = dict(C4=4, S_joint4head=1, U4_sharedB=4, S_one_path=1)
COHORTS = ('whole', 'covered', 'no_visible_TRAIN_neighbor')
T95_DF2 = 4.302652729911275
METRICS = ('correctcount', 'accuracy', 'NLL', 'Brier',
           'member_correctcount', 'member_accuracy', 'member_NLL')
BASE_FIELDS = ('seed', 'arm', 'selected_epoch', 'global_mode', 'selected_sha256', 'members', 'cohort', 'support')
EXTRA_FIELDS = ('correctcount_change', 'accuracy_change_pp', 'NLL_change', 'Brier_change',
    'no_neighbor_logits_max_abs', 'no_neighbor_probability_max_abs', 'repairs', 'introduced_errors',
    'wrong_to_different_wrong', 'persistent_wrong_same_label', 'persistent_errors', 'correct_to_correct')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def exact_file(path, digest, limit):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= limit and sha(path) == digest,
            'Exact bounded complete-collector file required')
    return path


def source_seal(root, digest):
    require(sha(root / 'MANIFEST.json') == digest, 'Exact source manifest required')
    for row in read(root / 'MANIFEST.json')['files']:
        path = root / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Frozen source payload changed')


def release(path, exact_sha, authorized):
    require(authorized is True, 'Disabled gate helper; root source review and whole-family opening required')
    require(sha(path) == exact_sha, 'Exact enabled root release required')
    cfg, fixed = read(path), read(HERE / 'ROOT_RELEASE_TEMPLATE_DISABLED.json')
    variable = {'enabled', 'source_review_approved', 'root_whole_family_opened', 'helper_manifest_sha256',
                'collector_input_directory', 'collector_complete_sha256', 'collector_cost_terminal_sha256',
                'collector_table_sha256', 'family_closure_sha256'}
    require(set(cfg) == set(fixed) and all(cfg[k] == fixed[k] for k in fixed if k not in variable)
            and all(cfg[k] is True for k in ('enabled', 'source_review_approved', 'root_whole_family_opened')),
            'Disabled or changed exact frozen gate release')
    for key in variable - {'enabled', 'source_review_approved', 'root_whole_family_opened', 'collector_input_directory'}:
        require(isinstance(cfg[key], str) and len(cfg[key]) == 64
                and all(c in '0123456789abcdef' for c in cfg[key]), 'Exact custody hash required')
    require(isinstance(cfg['collector_input_directory'], str) and Path(cfg['collector_input_directory']).is_absolute(),
            'Root-bound complete collector directory required')
    source_seal(HERE, cfg['helper_manifest_sha256'])
    return cfg


def inputs(cfg):
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    for row in (pins['protocol'], pins['collector_manifest'], pins['collector_program']):
        path = HERE.parent / row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Exact protocol/collector source binding')
    source_seal(HERE.parent / pins['collector_directory'], pins['collector_manifest']['sha256'])
    protocol = read(HERE.parent / pins['protocol']['path'])
    require(protocol['native_seeds'] == list(SEEDS) and protocol['arms'] == list(ARMS)
            and protocol['required_complete_correction_records'] == 12
            and protocol['frozen_whole_family_gate']['co_primary_contrasts'] == ['C4-S_joint4head', 'C4-U4_sharedB'],
            'Exact frozen protocol cardinality and both co-primaries')
    root = Path(cfg['collector_input_directory'])
    done = read(exact_file(root / 'COMPLETE.json', cfg['collector_complete_sha256'], 262144))
    cost = read(exact_file(root / 'COST_TERMINAL.json', cfg['collector_cost_terminal_sha256'], 262144))
    require(done['complete'] is True and cost['complete'] is True and cost['error'] is None
            and cost['source_manifest_sha256'] == pins['collector_manifest']['sha256']
            and done['family_closure_sha256'] == cost['family_closure_sha256'] == cfg['family_closure_sha256']
            and done['selected_states'] == cost['selected_states'] and done['work'] == cost['work']
            and done['work']['reconstructions_completed'] == done['work']['native_forwards_completed'] ==
                done['work']['raw_files_completed'] == 12, 'Complete root-opened 12-state collector custody only')
    require([(r['seed'], r['arm']) for r in done['selected_states']] ==
            [(seed, arm) for seed in SEEDS for arm in ARMS], 'All 12 ordered existing selected-state identities')
    table = exact_file(root / pins['collector_table'], cfg['collector_table_sha256'], 2097152)
    require(done['tables'][pins['collector_table']] == cfg['collector_table_sha256'], 'Collector-owned exact summary CSV')
    return protocol, table, {(r['seed'], r['arm']): r['sha256'] for r in done['selected_states']}


def number(value, nullable=False):
    if nullable and value == '':
        return None
    result = float(value)
    require(math.isfinite(result), 'Finite recorded scalar required; no dropping values')
    return result


def rows_from_csv(path, state_hashes):
    expected = set(BASE_FIELDS) | {p + '_' + m for p in ('native', 'corrected') for m in METRICS} | set(EXTRA_FIELDS)
    result = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(len(reader.fieldnames) == len(expected) and set(reader.fieldnames) == expected, 'Exact collector CSV field schema')
        for raw in reader:
            require(set(raw) == expected and None not in raw.values(), 'Complete recorded CSV row')
            row = dict(raw); seed, arm, cohort = int(raw['seed']), raw['arm'], raw['cohort']
            key = (seed, arm, cohort)
            require(seed in SEEDS and arm in ARMS and cohort in COHORTS and key not in result, 'Unique fixed seed/arm/cohort row')
            row.update(seed=seed, selected_epoch=int(raw['selected_epoch']), members=int(raw['members']), support=int(raw['support']))
            support = row['support']
            require(row['members'] == MEMBERS[arm] and 1 <= row['selected_epoch'] <= 1100
                    and raw['global_mode'] in ('True', 'False') and raw['selected_sha256'] == state_hashes[seed, arm]
                    and 0 <= support <= 5274 and (cohort != 'whole' or support == 5274), 'Exact selected endpoint and support')
            row['global_mode'] = raw['global_mode'] == 'True'
            for prefix, members in (('native', 1), ('corrected', MEMBERS[arm])):
                count = int(raw[prefix + '_correctcount']); require(0 <= count <= support, 'Recorded correctcount domain')
                row[prefix + '_correctcount'] = count
                for metric in ('accuracy', 'NLL', 'Brier'):
                    row[prefix + '_' + metric] = number(raw[prefix + '_' + metric], nullable=support == 0)
                counts = json.loads(raw[prefix + '_member_correctcount'])
                acc = json.loads(raw[prefix + '_member_accuracy']); nll = json.loads(raw[prefix + '_member_NLL'])
                require(len(counts) == len(acc) == len(nll) == members
                        and all(type(x) is int and 0 <= x <= support for x in counts), 'Every declared prediction member retained')
                require(row[prefix + '_accuracy'] == (count / support if support else None)
                        and acc == [x / support if support else None for x in counts]
                        and all((x is None if support == 0 else type(x) in (int, float) and math.isfinite(x)) for x in nll),
                        'Exact count-derived accuracy and finite complete member risks')
                for name, value in (('member_correctcount', counts), ('member_accuracy', acc), ('member_NLL', nll)):
                    row[prefix + '_' + name] = value
            for name in EXTRA_FIELDS:
                row[name] = int(raw[name]) if name in ('correctcount_change', 'repairs', 'introduced_errors',
                    'wrong_to_different_wrong', 'persistent_wrong_same_label', 'persistent_errors', 'correct_to_correct') else number(raw[name], nullable=True)
            result[key] = row
    require(set(result) == {(s, a, c) for s in SEEDS for a in ARMS for c in COHORTS}, 'All 36 rows; no partial arm/seed/cohort opening')
    for seed in SEEDS:
        for arm in ARMS:
            whole, covered, missing = [result[seed, arm, c] for c in COHORTS]
            require(covered['support'] + missing['support'] == 5274
                    and all((r['selected_epoch'], r['global_mode']) == (whole['selected_epoch'], whole['global_mode']) for r in (covered, missing)),
                    'Full same-state structural cohort partition')
            for prefix in ('native', 'corrected'):
                require(covered[prefix + '_correctcount'] + missing[prefix + '_correctcount'] == whole[prefix + '_correctcount'],
                        'Whole correctcount retained across cohorts')
            require(covered['support'] == result[SEEDS[0], ARMS[0], 'covered']['support'], 'One structural cohort across all seed/arms')
    return result


def endpoint(row):
    support = row['support']; counts = row['corrected_member_correctcount']
    risks = [Fraction.from_float(float(x)) for x in row['corrected_member_NLL']]
    return dict(accuracy_pp=100 * Fraction(row['corrected_correctcount'], support),
        served_NLL=Fraction.from_float(row['corrected_NLL']), Brier=Fraction.from_float(row['corrected_Brier']),
        mean_member_accuracy_pp=100 * Fraction(sum(counts), len(counts) * support),
        worst_member_accuracy_pp=100 * Fraction(min(counts), support),
        mean_member_NLL=sum(risks) / len(risks), worst_member_NLL=max(risks))


def exact(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def describe(values):
    require(len(values) == 3, 'Exactly three paired optimizer-seed effects')
    mean = sum(values) / 3; floating = [float(x) for x in values]
    sd = statistics.stdev(floating); se = sd / math.sqrt(3)
    return dict(per_seed=[dict(seed=seed, value=float(value), exact=exact(value)) for seed, value in zip(SEEDS, values)],
        mean=float(mean), exact_mean=exact(mean), sample_SD=sd, standard_error=se,
        descriptive_95_percent_t_interval=[float(mean) - T95_DF2 * se, float(mean) + T95_DF2 * se],
        n_optimizer_seeds=3, df=2, t_critical=T95_DF2, interval_used_for_gate=False)


def evaluate(rows, protocol):
    frozen = protocol['frozen_whole_family_gate']; contrasts = []; failures = []
    safeguards = frozen['member_safeguards_for_EACH_reference']
    limits = (('accuracy_mean', 'accuracy_pp', '>=', frozen['accuracy_for_EACH_contrast']['minimum_equal_weighted_mean_gain_pp']),
        ('served_NLL_mean', 'served_NLL', '<=', frozen['served_NLL_for_EACH_contrast']['maximum_equal_weighted_mean_C4_minus_reference_nats']),
        ('mean_member_accuracy', 'mean_member_accuracy_pp', '>=', safeguards['minimum_mean_over3seeds_delta_mean_member_accuracy_pp']),
        ('worst_member_accuracy', 'worst_member_accuracy_pp', '>=', safeguards['minimum_mean_over3seeds_delta_worst_member_accuracy_pp']),
        ('mean_member_NLL', 'mean_member_NLL', '<=', safeguards['maximum_mean_over3seeds_delta_mean_member_NLL_nats']),
        ('worst_member_NLL', 'worst_member_NLL', '<=', safeguards['maximum_mean_over3seeds_delta_worst_member_NLL_nats']))
    require(frozen['accuracy_for_EACH_contrast']['paired_gain_strictly_positive_all3seeds'] is True,
            'Exact all-three positive-sign condition')
    for reference in ('S_joint4head', 'U4_sharedB'):
        paired = [(endpoint(rows[s, 'C4', 'whole']), endpoint(rows[s, reference, 'whole'])) for s in SEEDS]
        effects = {name: [c[name] - r[name] for c, r in paired] for name in paired[0][0]}
        summary = {name: describe(values) for name, values in effects.items()}
        signs = [value > 0 for value in effects['accuracy_pp']]
        components = [dict(name='accuracy_positive_all3seeds', passed=all(signs), operator='>0 in every seed',
            per_seed_positive=[dict(seed=s, passed=ok) for s, ok in zip(SEEDS, signs)],
            failed_seeds=[s for s, ok in zip(SEEDS, signs) if not ok])]
        for name, metric, operator, threshold_value in limits:
            threshold = Fraction(str(threshold_value)); observed = sum(effects[metric]) / 3
            passed = observed >= threshold if operator == '>=' else observed <= threshold
            components.append(dict(name=name, metric=metric, passed=passed, operator=operator,
                threshold=threshold_value, exact_threshold=exact(threshold), observed_mean=float(observed), exact_mean=exact(observed)))
        for component in components:
            if not component['passed']:
                failures.append(dict(contrast='C4-' + reference, **component))
        contrasts.append(dict(contrast='C4-' + reference, whole_support_per_seed=[5274] * 3,
            passed=all(c['passed'] for c in components), components=components, paired_effects=summary))
    return dict(family_gate_passed=all(c['passed'] for c in contrasts), contrasts=contrasts,
        failed_gate_components=failures, per_seed_arm_cohort_values=[rows[s, a, c] for s in SEEDS for a in ARMS for c in COHORTS],
        exact_protocol_gate=frozen, evaluated_gate_components=14, gate_population='whole5274development only',
        secondary_or_cohort_rescue=False, intervals_used_for_gate=False, node_independence_assumed=False,
        significance_claimed=False, promotion_authorized=False, exploratory_consumed_development=True,
        interval_interpretation='Descriptive model-based t intervals across3paired optimizer seeds; df2. Not independent graph/node replications; assumptions not established and no significance or confirmation claim.')


def run(*, release_path, release_sha256, later_execution_authorized=False):
    cfg = release(release_path, release_sha256, later_execution_authorized)
    output = HERE.parent / cfg['output_relative']
    require(not output.exists(), 'Fresh once-only gate output; no overwrite/retry')
    output.mkdir(); started = time.monotonic(); usage = resource.getrusage(resource.RUSAGE_SELF)
    error = None; complete = False
    try:
        protocol, table, states = inputs(cfg)
        rows = rows_from_csv(table, states); result = evaluate(rows, protocol)
        result['custody'] = {k: cfg[k] for k in ('helper_manifest_sha256', 'collector_complete_sha256',
            'collector_cost_terminal_sha256', 'collector_table_sha256', 'family_closure_sha256')}
        result['custody']['release_sha256'] = release_sha256
        write(output / 'GATE_RESULT.json', result); complete = True
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        write(output / 'FAILURE.json', dict(error=error, family_gate_evaluated=False,
            no_pass_for_incomplete_custody=True, automatic_retry=False, release_sha256=release_sha256))
        raise
    finally:
        final = resource.getrusage(resource.RUSAGE_SELF)
        write(output / 'TERMINAL.json', dict(complete=complete, error=error, wall_seconds=time.monotonic() - started,
            CPU_user_seconds=final.ru_utime - usage.ru_utime, CPU_system_seconds=final.ru_stime - usage.ru_stime,
            max_RSS_bytes=final.ru_maxrss * (1024 if os.uname().sysname == 'Linux' else 1),
            release_sha256=release_sha256, no_training=True, no_raw_prediction_reads=True, automatic_retry=False))
    return dict(complete=True, family_gate_passed=result['family_gate_passed'], output=str(output),
                gate_result_sha256=sha(output / 'GATE_RESULT.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--authorized', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(release_path=args.release, release_sha256=args.release_sha256,
                        later_execution_authorized=args.authorized), sort_keys=True))
