"""Fixed descriptive CPU degree profile of six CLOSED Wiki24 prediction banks.

No framework/model/checkpoint import, collection, fitting, or prediction change.
Numeric imports occur only after the exact saved input custody is authenticated.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import zipfile

SEEDS = (6101, 6203, 6307)
ARMS = ('be_unit_contrastive', 'independent4')
NODES, DEVELOP, TRAIN, CLASSES, MEMBERS = 11701, 5274, 580, 10, 4
ALLOWED_REPOS = (
    Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'),
    Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'),
)
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def bound(phase, binding):
    relative = Path(binding['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative custody required')
    path = (phase / relative).resolve(strict=True)
    require(path.is_relative_to(phase) and path.is_file(), 'Input outside bound phase')
    require(sha(path) == binding['sha256'], 'Input SHA256 changed: ' + binding['path'])
    if 'bytes' in binding:
        require(path.stat().st_size == binding['bytes'], 'Input byte count changed')
    return path


def same_binding(a, b):
    return all(a.get(k) == b.get(k) for k in ('path', 'sha256'))


def authenticate(phase, contract):
    require(contract['schema'] == 'closed-wiki24-degree-profile-input-v1', 'Exact contract schema')
    documents = {name: read(bound(phase, binding)) for name, binding in contract['documents'].items()}
    collection, metadata, cost = (documents[x] for x in ('collection', 'metadata_export', 'collection_cost'))
    require(collection['status'] == 'complete' and collection['whole24_accounted'] is True
            and collection['TEST_access'] is False, 'Complete original collection required')
    require(metadata['whole24_accounted'] is True and metadata['TEST_access'] is False
            and metadata['reselected'] is False, 'Complete original metadata required')
    gate = metadata['gate']
    require(gate['passed'] is True and gate['family_closed'] is True and gate['complete_cells'] == 24
            and gate['failed_or_not_launched_cells'] == 0, 'Whole24 saved closure required')
    require(gate['source_manifest_sha256'] == contract['original_suite_manifest_sha256'], 'Source family custody')
    closure = read(bound(phase, gate['closure']))
    owner = read(bound(phase, gate['owner']))
    require(closure['schema'] == 'internal-be-WikiCS-family-closure-v1' and closure['closed'] is True
            and closure['complete_cells'] == 24 and closure['TEST_access'] is False
            and closure['all_full_endpoints_or_retained_failure'] is True
            and owner == gate['owner_metadata'] == closure['owner'], 'Saved actual owner/closure custody')
    require(same_binding(cost['data_manifest'], contract['documents']['data_manifest'])
            and same_binding(gate['data_manifest_binding'], contract['documents']['data_manifest']), 'Original graph role custody')
    require(cost['original_serving'] == 'mean_probability' and cost['TEST_access'] is False,
            'Original mean-probability serving convention')
    role = documents['data_manifest']
    require(role['schema'] == 'internal-be-official-role-projection-v2' and role['task'] == 'wikics'
            and role['split_index'] == 0 and role['train_count'] == TRAIN and role['valid_count'] == DEVELOP
            and role['TEST_values_in_payload'] is False and role['official_split_preserved'] is True
            and role['format'] == 'NPZ_numeric_only', 'Complete official TEST-free split0 role')
    require(role['source_manifest_sha256'] == contract['original_suite_manifest_sha256'], 'Safe role source family')
    for name in ('train', 'valid'):
        require(same_binding(role['payloads'][name], contract['roles'][name])
                and same_binding(cost['data_payloads'][name], contract['roles'][name]), 'Role payload custody')
    rows = {r['cell']: r for r in collection['cells']}
    meta = {r['cell']: r for r in metadata['cells']}
    require(len(rows) == len(meta) == 24, 'Complete unique original roster')
    paths = {}
    for seed in SEEDS:
        for arm in ARMS:
            cell = arm + '_' + str(seed)
            specification, actual, selected = contract['cells'][cell], rows[cell], meta[cell]
            require(actual['family_status'] == actual['collection_status'] == selected['status'] == 'complete',
                    'All six original complete cells required')
            require(actual['seed'] == selected['seed'] == seed and actual['arm'] == selected['arm'] == arm
                    and actual['members'] == MEMBERS and actual['scored_merged_development_nodes'] == DEVELOP,
                    'Exact seed/arm/member/population custody')
            require(actual['raw_prediction_archive'] == specification['raw_prediction_archive']
                    and actual['selected_checkpoint'] == selected['selected_checkpoint'] == specification['selected_checkpoint'],
                    'Exact saved prediction and selected-state identity')
            require(actual['selected_member_modes'] == selected['member_global'] == specification['selected_member_modes']
                    and selected['selection'] == specification['selection'], 'Selected serving modes and selector identity')
            paths[cell] = bound(phase, specification['raw_prediction_archive'])
    return documents, paths, {k: gate[k] for k in ('closure', 'owner', 'source_manifest_sha256')}


def load_npz(np, path, schema, wanted=None):
    # Inspect every entry before loading any numeric values. TEST/extra fields fail.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == {k + '.npy' for k in schema}, 'Exact NPZ field set')
        for name in names:
            with archive.open(name) as entry:
                version = np.lib.format.read_magic(entry)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(entry)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(entry)
                else:
                    raise ValueError('Unsupported NPY header')
            expected_shape, expected_dtype = schema[name[:-4]]
            require(shape == expected_shape and dtype == np.dtype(expected_dtype)
                    and not dtype.hasobject and not order, 'Exact safe NPZ shape/dtype/order')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in (wanted or schema)}


def degree_cohorts(np, phase, contract):
    train = load_npz(np, bound(phase, contract['roles']['train']),
                     {'x': ((NODES, 300), 'float32'), 'edge_index': ((2, 442907), 'int64'),
                      'ids': ((TRAIN,), 'int64'), 'y': ((TRAIN,), 'int64')}, wanted=('edge_index', 'ids', 'y'))
    valid = load_npz(np, bound(phase, contract['roles']['valid']),
                     {'ids': ((DEVELOP,), 'int64'), 'y': ((DEVELOP,), 'int64')})
    for role, count in ((train, TRAIN), (valid, DEVELOP)):
        require(np.unique(role['ids']).size == count and (role['ids'] >= 0).all()
                and (role['ids'] < NODES).all() and (role['y'] >= 0).all() and (role['y'] < CLASSES).all(),
                'Unique full role IDs and finite class domain')
    require(not np.isin(train['ids'], valid['ids']).any(), 'TRAIN/development overlap')
    edge = train['edge_index']
    require((edge >= 0).all() and (edge < NODES).all(), 'Full graph endpoint domain')
    nonself = edge[0] != edge[1]
    unique = np.unique(edge[0, nonself] * NODES + edge[1, nonself])
    degree = np.bincount(unique % NODES, minlength=NODES).astype(np.int64)
    ordered = np.sort(degree[train['ids']])
    ranks = [math.ceil(q * TRAIN / 4) for q in (1, 2, 3)]
    boundaries = [int(ordered[r - 1]) for r in ranks]
    bins = np.searchsorted(np.array(boundaries), degree, side='left')
    ids, truth = valid['ids'], valid['y']
    masks = [('full_population', None, np.ones(DEVELOP, dtype=bool))]
    masks += [('degree_quartile', q + 1, bins[ids] == q) for q in range(4)]
    masks += [('zero_neighbor', True, degree[ids] == 0), ('zero_neighbor', False, degree[ids] != 0)]
    expanded = [(kind, value, c, mask & (truth == c) if c is not None else mask)
                for kind, value, mask in masks for c in (None, *range(CLASSES))]
    freeze = dict(schema='closed-wiki24-degree-cohorts-v1', graph_nodes=NODES,
        edge_convention='edge_index[0] source -> edge_index[1] receiver; incoming unique nonself source count',
        prepared_edge_records=int(edge.shape[1]), self_edge_records=int((~nonself).sum()),
        unique_directed_nonself_edges=int(unique.size), removed_duplicate_nonself_records=int(nonself.sum() - unique.size),
        boundaries_train_only=True, nearest_rank_one_based=ranks, quartile_upper_boundaries=boundaries,
        bins=['degree <= b1', 'b1 < degree <= b2', 'b2 < degree <= b3', 'degree > b3'],
        tie_policy='Equal degrees stay together; tied boundaries retain empty bins; no jitter or bin merging',
        train_nodes=TRAIN, development_nodes=DEVELOP,
        train_quartile_counts=[int((bins[train['ids']] == q).sum()) for q in range(4)],
        development_quartile_counts=[int((bins[ids] == q).sum()) for q in range(4)],
        train_zero_neighbor_count=int((degree[train['ids']] == 0).sum()),
        development_zero_neighbor_count=int((degree[ids] == 0).sum()),
        matched_cohorts='Both banks use identical complete original rows for every cohort and truth class',
        ordered_development_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        ordered_development_truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),
        degree_boundaries_frozen_before_prediction_numeric_load=True)
    return ids, truth, expanded, freeze


def prediction_schema():
    return {'valid_ids': ((DEVELOP,), 'int64'), 'truth': ((DEVELOP,), 'int64'),
            'member_logits': ((MEMBERS, DEVELOP, CLASSES), 'float32'),
            'member_probability': ((MEMBERS, DEVELOP, CLASSES), 'float32'),
            'pool_probability': ((DEVELOP, CLASSES), 'float32'),
            'member_prediction': ((MEMBERS, DEVELOP), 'int64'), 'pool_prediction': ((DEVELOP,), 'int64'),
            'member_nll': ((MEMBERS, DEVELOP), 'float32'), 'pool_nll': ((DEVELOP,), 'float32')}


def bank(np, path, ids, truth):
    arrays = load_npz(np, path, prediction_schema())
    require(np.array_equal(arrays['valid_ids'], ids) and np.array_equal(arrays['truth'], truth),
            'Exact full original ordered development rows/truth')
    for key in ('member_logits', 'member_probability', 'pool_probability'):
        require(np.isfinite(arrays[key]).all(), 'Nonfinite original prediction values')
    for key in ('member_probability', 'pool_probability'):
        require((arrays[key] >= 0).all() and (arrays[key] <= 1).all(), 'Probability domain')
    require(np.array_equal(arrays['member_prediction'], arrays['member_logits'].argmax(axis=2))
            and np.array_equal(arrays['pool_prediction'], arrays['pool_probability'].argmax(axis=1)),
            'Saved actual argmax/logit/pool inconsistency')
    # Retain the recorded GPU float32 pool. Do not recompute softmax or demand
    # tiny CPU/GPU floating-point parity of the saved mean probabilities.
    correct = arrays['member_prediction'] == truth[None, :]
    pool_correct = arrays['pool_prediction'] == truth
    truth_logits = np.take_along_axis(arrays['member_logits'], truth[None, :, None], axis=2)
    common_classes = (arrays['member_logits'] > truth_logits).all(axis=0)
    arrays.update(member_correct=correct, correct_member_count=correct.sum(axis=0),
        common_classes=common_classes, truth_logits=truth_logits,
        flags=dict(pool_correct=pool_correct, pool_wrong=~pool_correct,
                   any_member_correct=correct.any(axis=0), all_member_wrong=~correct.any(axis=0),
                   all_member_correct=correct.all(axis=0), common_rival=common_classes.any(axis=1),
                   unanimous_wrong=~correct.any(axis=0) & (arrays['member_prediction'] == arrays['member_prediction'][0]).all(axis=0),
                   pool_harm=correct.any(axis=0) & ~pool_correct, pool_rescue=~correct.any(axis=0) & pool_correct))
    arrays['flags']['common_rival_and_pool_correct'] = arrays['flags']['common_rival'] & pool_correct
    return arrays


def rate(value, denominator):
    return value / denominator if denominator else None


def summarize_bank(np, arrays, mask):
    n = int(mask.sum())
    counts = {key: int(values[mask].sum()) for key, values in arrays['flags'].items()}
    members = [int(x[mask].sum()) for x in arrays['member_correct']]
    common = arrays['flags']['common_rival'] & mask
    first_class = arrays['common_classes'].argmax(axis=1)
    return dict(counts=counts, node_denominator=n, rates={k: rate(v, n) for k, v in counts.items()},
        member_correct_counts=members, member_node_denominator=n,
        member_accuracy=[rate(v, n) for v in members], mean_member_accuracy=rate(sum(members), MEMBERS * n),
        correct_member_count_histogram=np.bincount(arrays['correct_member_count'][mask], minlength=5).tolist(),
        smallest_strict_common_rival_class_counts=np.bincount(first_class[common], minlength=CLASSES).tolist())


def summarize_pair(np, old, ordinary, mask):
    n = int(mask.sum()); oc, ic = old['flags']['pool_correct'], ordinary['flags']['pool_correct']
    oa, ia = old['flags']['any_member_correct'], ordinary['flags']['any_member_correct']
    both_wrong = ~oc & ~ic
    conditions = dict(served_both_correct=oc & ic, served_repairs_old_to_ordinary=~oc & ic,
        served_harms_old_to_ordinary=oc & ~ic, served_both_wrong_same_class=both_wrong & (old['pool_prediction'] == ordinary['pool_prediction']),
        served_both_wrong_different_class=both_wrong & (old['pool_prediction'] != ordinary['pool_prediction']),
        coverage_both_have_correct_member=oa & ia, ordinary_acquired_correct_member_ranking=~oa & ia,
        old_acquired_correct_member_ranking=oa & ~ia, coverage_neither_has_correct_member=~oa & ~ia,
        both_have_strict_common_rival=old['flags']['common_rival'] & ordinary['flags']['common_rival'],
        identical_strict_common_rival_in_both_banks=(old['common_classes'] & ordinary['common_classes']).any(axis=1))
    counts = {key: int(values[mask].sum()) for key, values in conditions.items()}
    acquisitions = {}
    for name, origin, destination in (('ordinary_on_old_errors', old, ordinary), ('old_on_ordinary_errors', ordinary, old)):
        wrong_mask = mask & origin['flags']['all_member_wrong']
        rivals_mask = mask & origin['flags']['common_rival']
        rivals = origin['common_classes']
        beaten = destination['truth_logits'] > destination['member_logits']
        same_route = (beaten | ~rivals[None, :, :]).all(axis=2).any(axis=0)
        different_routes = (beaten.any(axis=0) | ~rivals).all(axis=1)
        rival_conditions = dict(any_correct_member=destination['flags']['any_member_correct'],
            one_member_strictly_beats_every_origin_common_rival=same_route,
            every_origin_common_rival_strictly_beaten_by_some_member=different_routes,
            every_origin_common_rival_beaten_but_no_correct_member=different_routes & ~destination['flags']['any_member_correct'])
        rival_counts = {k: int(v[rivals_mask].sum()) for k, v in rival_conditions.items()}
        acq_n, rival_n = int(wrong_mask.sum()), int(rivals_mask.sum())
        acquisitions[name] = dict(origin_all_member_wrong_denominator=acq_n,
            destination_member_correct_counts=[int(x[wrong_mask].sum()) for x in destination['member_correct']],
            destination_correct_member_count_histogram=np.bincount(destination['correct_member_count'][wrong_mask], minlength=5).tolist(),
            any_correct_member_count=int(destination['flags']['any_member_correct'][wrong_mask].sum()),
            any_correct_member_rate=rate(int(destination['flags']['any_member_correct'][wrong_mask].sum()), acq_n),
            origin_common_rival_denominator=rival_n, common_rival_acquisition_counts=rival_counts,
            common_rival_acquisition_rates={k: rate(v, rival_n) for k, v in rival_counts.items()})
    return dict(node_denominator=n, counts=counts, rates={k: rate(v, n) for k, v in counts.items()},
        served_net_repairs=counts['served_repairs_old_to_ordinary'] - counts['served_harms_old_to_ordinary'],
        served_accuracy_change=rate(counts['served_repairs_old_to_ordinary'] - counts['served_harms_old_to_ordinary'], n),
        acquisitions=acquisitions)


def additive(row):
    """Only counts enter partition identities; no sum of rates or overlapping cohorts."""
    result = {'nodes': row['nodes']}
    for arm in ARMS:
        value = row['banks'][arm]
        result.update({arm + '/' + k: v for k, v in value['counts'].items()})
        for key in ('member_correct_counts', 'correct_member_count_histogram', 'smallest_strict_common_rival_class_counts'):
            result.update({arm + '/' + key + '/' + str(i): v for i, v in enumerate(value[key])})
    result.update({'paired/' + k: v for k, v in row['paired']['counts'].items()})
    for name, value in row['paired']['acquisitions'].items():
        for key in ('origin_all_member_wrong_denominator', 'any_correct_member_count', 'origin_common_rival_denominator'):
            result['acquisition/' + name + '/' + key] = value[key]
        for key in ('destination_member_correct_counts', 'destination_correct_member_count_histogram'):
            result.update({'acquisition/' + name + '/' + key + '/' + str(i): v for i, v in enumerate(value[key])})
        result.update({'acquisition/' + name + '/' + k: v for k, v in value['common_rival_acquisition_counts'].items()})
    return result


def reconcile(rows, reference):
    indexed = {(r['cohort_kind'], r['cohort_value'], r['truth_class']): r for r in rows}
    identities = []
    def check(name, target, parts):
        expected = additive(target); summed = {k: sum(additive(r)[k] for r in parts) for k in expected}
        require(expected == summed, 'Partition count reconciliation failed: ' + name)
        identities.append(dict(identity=name, matched=True, checked_count_fields=len(expected), nodes=target['nodes']))
    full = indexed[('full_population', None, None)]
    for kind, values in (('degree_quartile', range(1, 5)), ('zero_neighbor', (True, False))):
        for c in (None, *range(CLASSES)):
            check(kind + '_partition_class_' + str(c), indexed[('full_population', None, c)],
                  [indexed[(kind, value, c)] for value in values])
    for kind, value, c in indexed:
        if c is None:
            check('truth_class_partition_' + kind + '_' + str(value), indexed[(kind, value, None)],
                  [indexed[(kind, value, label)] for label in range(CLASSES)])
    for row in rows:
        n = row['nodes']; pair = row['paired']['counts']
        require(sum(pair[k] for k in ('served_both_correct', 'served_repairs_old_to_ordinary', 'served_harms_old_to_ordinary',
                    'served_both_wrong_same_class', 'served_both_wrong_different_class')) == n, 'Served transition partition')
        require(sum(pair[k] for k in ('coverage_both_have_correct_member', 'ordinary_acquired_correct_member_ranking',
                    'old_acquired_correct_member_ranking', 'coverage_neither_has_correct_member')) == n, 'Coverage transition partition')
        for arm in ARMS:
            value = row['banks'][arm]; count = value['counts']
            require(sum(value['correct_member_count_histogram']) == n
                    and count['any_member_correct'] + count['all_member_wrong'] == n
                    and count['pool_correct'] + count['pool_wrong'] == n
                    and count['pool_correct'] == count['any_member_correct'] - count['pool_harm'] + count['pool_rescue'],
                    'Full within-cohort count accounting')
    # Integer event reconciliation against earlier analysis of these same exact
    # archives is custody checking; historical float32 scores are not a gate.
    for arm in ARMS:
        saved = reference[arm]['full_population']['counts']; current = full['banks'][arm]['counts']
        for key in ('pool_correct', 'any_member_correct', 'all_member_wrong', 'unanimous_wrong', 'pool_harm', 'pool_rescue'):
            require(current[key] == saved[key], 'Existing complete-population count inconsistency: ' + arm + '/' + key)
        require(current['common_rival'] == saved['common_competitor'], 'Existing common-rival count inconsistency')
    return dict(all_partitions_reconciled=True, identities=identities,
                existing_complete_population_integer_counts_reconciled=True,
                historical_score_recalculation_or_float_parity_gate=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--execute-authorized-closed-wiki24-profile', action='store_true', required=True)
    args = parser.parse_args()
    phase = args.phase.resolve(strict=True)
    require(any(phase == (p / 'experiments_iclr/postsubmission_20260930').resolve() for p in ALLOWED_REPOS),
            'Only authorized scientific repository phase')
    require(sha(HERE / 'INPUT_CONTRACT.json') == args.contract_sha256
            and sha(Path(__file__)) == args.source_sha256, 'Frozen source/contract digests required')
    output = args.output.resolve()
    require(output.is_relative_to(phase) and not output.exists(), 'Fresh phase-local output required')
    os.umask(0o077); output.mkdir(parents=True)
    started, cpu_started = time.monotonic(), time.process_time()
    try:
        contract = read(HERE / 'INPUT_CONTRACT.json')
        documents, paths, custody = authenticate(phase, contract)
        authentication_seconds = time.monotonic() - started
        import numpy as np
        ids, truth, cohorts, freeze = degree_cohorts(np, phase, contract)
        write(output / 'COHORT_FREEZE.json', freeze)
        reference = {s['seed']: {c['arm']: c for c in s['cells']} for s in documents['per_seed']['seeds']}
        require(tuple(reference) == SEEDS, 'Complete original seed roster')
        results = []
        for seed in SEEDS:
            banks = {arm: bank(np, paths[arm + '_' + str(seed)], ids, truth) for arm in ARMS}
            rows = [dict(seed=seed, cohort_kind=kind, cohort_value=value, truth_class=c,
                        nodes=int(mask.sum()), banks={arm: summarize_bank(np, banks[arm], mask) for arm in ARMS},
                        paired=summarize_pair(np, banks[ARMS[0]], banks[ARMS[1]], mask))
                    for kind, value, c, mask in cohorts]
            results.append(dict(seed=seed, rows=rows, reconciliation=reconcile(rows, reference[seed])))
        write(output / 'PROFILE.json', dict(schema='closed-wiki24-degree-error-profile-v1', status='complete',
            serving='Saved actual member argmax and float32 mean-softmax pooled predictions/probabilities',
            comparison_direction='old be_unit_contrastive -> ordinary independent4', seeds=results,
            new_fits=0, new_model_calls=0, prediction_changes=0, original_paper_scores_changed=False,
            limits=contract['limits']))
        write(output / 'INPUTS.json', dict(contract_sha256=args.contract_sha256, source_sha256=args.source_sha256,
            documents=contract['documents'], roles=contract['roles'], cells=contract['cells'], saved_terminal_custody=custody))
        write(output / 'TERMINAL.json', dict(status='complete', seeds=list(SEEDS), cells=6,
            nodes_per_seed=DEVELOP, cohort_rows_per_seed=len(cohorts), authentication_seconds=authentication_seconds,
            inclusive_wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time() - cpu_started,
            linux_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            numpy_version=np.__version__, new_fits=0, new_model_calls=0, TEST_access=False))
    except Exception as error:
        write(output / 'FAILURE.json', dict(status='failed', error_type=type(error).__name__, error=str(error),
            inclusive_wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time() - cpu_started,
            no_partial_population_result=True))
        raise


if __name__ == '__main__':
    main()
