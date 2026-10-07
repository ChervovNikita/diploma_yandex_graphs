"""Descriptive selected-prediction summaries; numerical imports occur only in run()."""
import hashlib
import json
from pathlib import Path

ARMS = ['single', 'single_contrastive', 'independent4', 'independent4_contrastive',
        'be_unit', 'be_init', 'be_unit_contrastive', 'be_init_contrastive']
SEEDS = [6101, 6203, 6307]
COHORTS = ['full_population', 'baseline_pool_error', 'baseline_all_member_wrong',
           'baseline_common_competitor', 'baseline_unanimous_wrong']
ROLE = 'WikiCS split0 merged development selection role: official val_mask union stopping_mask; all5274 nodes'
LIMITS = {
    'role': ROLE,
    'VALID_selection_optimism': 'The same merged development role selected checkpoints and supplies these error analyses; these are selected-state descriptive results.',
    'independent_TEST_evidence': False,
    'unit_of_replication': 'Three fixed seed blocks; nodes and members are not independent fitted-model replicates.',
    'cohorts': 'Frozen from be_init for each seed before any be_init_contrastive prediction is inspected; no favorable node selection.',
    'mechanism': 'be_init to be_init_contrastive compares the complete alignment plus residual contrastive package; it does not isolate either term, internal GNCL, or a graph mechanism.',
    'causal_claim': False,
    'metric_reselection_or_calibration': False,
}


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def error_masks(np, arrays):
    y = arrays['truth']; predictions = arrays['member_prediction']; pool = arrays['pool_prediction']
    correct = predictions == y[None, :]
    all_wrong = ~correct.any(axis=0)
    # A common competitor is one same nontruth class whose original logit is
    # strictly greater than the truth logit in EVERY member. Ties are excluded.
    logits = arrays['member_logits']
    truth_logits = np.take_along_axis(logits, y[None, :, None], axis=2)
    common_classes = (logits > truth_logits).all(axis=0)
    common = common_classes.any(axis=1)
    common_class = np.where(common, common_classes.argmax(axis=1), -1).astype(np.int64)
    pool_correct = pool == y
    return dict(member_correct=correct, pool_correct=pool_correct,
                any_member_correct=correct.any(axis=0), all_member_correct=correct.all(axis=0),
                all_member_wrong=all_wrong, common_competitor=common,
                common_competitor_class=common_class,
                unanimous_wrong=all_wrong & (predictions == predictions[0]).all(axis=0),
                pool_rescue=all_wrong & pool_correct,
                pool_harm=correct.any(axis=0) & ~pool_correct,
                pool_harm_despite_all_member_correct=correct.all(axis=0) & ~pool_correct)


def baseline_cohorts(np, arrays):
    masks = error_masks(np, arrays)
    return dict(valid_ids=arrays['valid_ids'].copy(), truth=arrays['truth'].copy(),
                full_population=np.ones(len(arrays['truth']), dtype=np.bool_),
                baseline_pool_error=~masks['pool_correct'],
                baseline_all_member_wrong=masks['all_member_wrong'],
                baseline_common_competitor=masks['common_competitor'],
                baseline_unanimous_wrong=masks['unanimous_wrong'],
                baseline_common_competitor_class=masks['common_competitor_class'])


def mean(np, values):
    values = np.asarray(values)
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return float(values.mean(dtype=np.float64))


def scalar_errors(np, arrays, mask):
    n = int(mask.sum()); flags = error_masks(np, arrays)
    counts = {key: int(flags[key][mask].sum()) for key in (
        'pool_correct', 'any_member_correct', 'all_member_correct', 'all_member_wrong',
        'common_competitor', 'unanimous_wrong', 'pool_rescue', 'pool_harm',
        'pool_harm_despite_all_member_correct')}
    counts['pool_wrong'] = n - counts['pool_correct']
    counts['common_competitor_and_pool_correct'] = int((flags['common_competitor'] & flags['pool_correct'])[mask].sum())
    members = []
    for i, correct in enumerate(flags['member_correct']):
        loss = arrays['member_nll'][i, mask]
        members.append(dict(member_index0=i, correct=int(correct[mask].sum()),
                            accuracy=mean(np, correct[mask]), nll=mean(np, loss),
                            nonfinite_nll=int((~np.isfinite(loss)).sum())))
    accuracies = [row['accuracy'] for row in members]
    nlls = [row['nll'] for row in members]
    pool_loss = arrays['pool_nll'][mask]
    return dict(nodes=n, counts=counts, rates={key: value / n if n else None for key, value in counts.items()},
                members=members, mean_member_accuracy=mean(np, np.array(accuracies, dtype=np.float64)),
                minimum_member_accuracy=min(accuracies) if n else None,
                maximum_member_accuracy=max(accuracies) if n else None,
                member_accuracy_range=max(accuracies) - min(accuracies) if n else None,
                mean_member_nll=mean(np, np.array(nlls, dtype=np.float64)),
                served_accuracy=mean(np, flags['pool_correct'][mask]), served_nll=mean(np, pool_loss),
                nonfinite_served_nll=int((~np.isfinite(pool_loss)).sum()))


def validate(np, arrays, members):
    n = 5274
    expected = {'valid_ids': ((n,), np.dtype('int64')), 'truth': ((n,), np.dtype('int64')),
                'member_logits': ((members, n, 10), np.dtype('float32')),
                'member_probability': ((members, n, 10), np.dtype('float32')),
                'pool_probability': ((n, 10), np.dtype('float32')),
                'member_prediction': ((members, n), np.dtype('int64')),
                'pool_prediction': ((n,), np.dtype('int64')),
                'member_nll': ((members, n), np.dtype('float32')),
                'pool_nll': ((n,), np.dtype('float32'))}
    if set(arrays) != set(expected):
        raise ValueError('Exact complete numeric prediction archive required')
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError('Prediction array shape/dtype: ' + name)
    for name in ('member_logits', 'member_probability', 'pool_probability'):
        if not np.isfinite(arrays[name]).all():
            raise ValueError('Nonfinite source predictions: ' + name)


def load(np, path, expected):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected['sha256'] or Path(path).stat().st_size != expected['bytes']:
        raise ValueError('Saved server-only prediction/cohort bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def paired(np, baseline, candidate, cohorts, seed):
    if not np.array_equal(baseline['valid_ids'], candidate['valid_ids']) or not np.array_equal(baseline['truth'], candidate['truth']):
        raise ValueError('Paired predictions must retain identical complete original IDs/truth')
    if not np.array_equal(cohorts['valid_ids'], baseline['valid_ids']) or not np.array_equal(cohorts['truth'], baseline['truth']):
        raise ValueError('Original baseline cohort custody')
    before = error_masks(np, baseline); after = error_masks(np, candidate); rows = []
    for name in COHORTS:
        mask = cohorts[name]; n = int(mask.sum())
        bc = before['pool_correct']; cc = after['pool_correct']
        repairs = int((~bc & cc)[mask].sum()); harms = int((bc & ~cc)[mask].sum())
        both_wrong = ~bc & ~cc
        same_wrong = both_wrong & (baseline['pool_prediction'] == candidate['pool_prediction'])
        delta_nll = candidate['pool_nll'][mask].astype(np.float64) - baseline['pool_nll'][mask].astype(np.float64)
        member_delta = []
        for m in range(4):
            member_delta.append(dict(member_index0=m,
                accuracy_change=mean(np, after['member_correct'][m, mask].astype(np.float64) - before['member_correct'][m, mask].astype(np.float64)),
                nll_change=mean(np, candidate['member_nll'][m, mask].astype(np.float64) - baseline['member_nll'][m, mask].astype(np.float64))))
        rows.append(dict(seed=seed, cohort=name, nodes=n, repairs=repairs, harms=harms,
            net_repairs=repairs - harms, accuracy_change=(repairs - harms) / n if n else None,
            both_correct=int((bc & cc)[mask].sum()), both_wrong=int(both_wrong[mask].sum()),
            both_wrong_same_class=int(same_wrong[mask].sum()),
            both_wrong_different_class=int((both_wrong & ~same_wrong)[mask].sum()),
            served_nll_change=mean(np, delta_nll), nonfinite_nll_changes=int((~np.isfinite(delta_nll)).sum()),
            member_changes=member_delta, baseline=scalar_errors(np, baseline, mask), candidate=scalar_errors(np, candidate, mask)))
    return dict(seed=seed, available=True, comparison='be_init -> be_init_contrastive', cohorts=rows)


def run(output, collection):
    """Read bounded server-only arrays after baseline cohort seals already exist."""
    import numpy as np
    output = Path(output); compact = output / 'compact'; raw = output / 'raw'
    rows = []; complete = {}
    for record in collection['cells']:
        row = {key: record[key] for key in ('arm', 'seed', 'cell', 'family_status', 'collection_status')}
        row['historical_selected_metadata'] = record['historical_selected_metadata']
        if record['collection_status'] == 'complete':
            arrays = load(np, raw / (record['cell'] + '.npz'), record['raw_prediction_archive'])
            validate(np, arrays, record['members'])
            row['full_population'] = scalar_errors(np, arrays, np.ones(5274, dtype=np.bool_))
            flags = error_masks(np, arrays)
            row['served_confusion_counts_truth_rows_prediction_columns'] = np.bincount(
                arrays['truth'] * 10 + arrays['pool_prediction'], minlength=100).reshape(10, 10).tolist()
            row['common_competitor_class_counts_smallest_qualifying_class'] = np.bincount(
                flags['common_competitor_class'][flags['common_competitor']], minlength=10).tolist()
            row['pool_rescues_by_truth_class'] = np.bincount(arrays['truth'][flags['pool_rescue']], minlength=10).tolist()
            row['pool_harms_by_truth_class'] = np.bincount(arrays['truth'][flags['pool_harm']], minlength=10).tolist()
            row['source_float32_reevaluation'] = record['source_float32_reevaluation']
            row['selected_body_modes'] = record['selected_body_modes']
            # Stored versus re-evaluated VALID values are separate events. These
            # differences are diagnostics, never an admission/parity condition.
            row['reevaluation_minus_historical_served_VALID'] = record['source_float32_reevaluation']['selected_VALID'] - record['historical_selected_metadata']['selected_VALID']
            row['reevaluation_minus_historical_member_VALID'] = [new - old for new, old in zip(record['source_float32_reevaluation']['member_VALID'], record['historical_selected_metadata']['member_VALID'])]
            complete[(record['arm'], record['seed'])] = record
            del arrays
        else:
            row['unavailable_reason'] = record['unavailable_reason']
        rows.append(row)
    write(compact / 'PER_CELL.json', dict(schema='internal-be-Wiki24-selected-per-cell-v1', cells=rows, limits=LIMITS))
    pairs = []
    for seed in SEEDS:
        before = complete.get(('be_init', seed)); after = complete.get(('be_init_contrastive', seed))
        cohort = collection['cohorts'][str(seed)]
        if before is None or after is None or not cohort['available']:
            pairs.append(dict(seed=seed, available=False, comparison='be_init -> be_init_contrastive',
                unavailable_reason='Complete baseline, complete candidate and original baseline cohort seal are all required',
                baseline_available=before is not None, candidate_available=after is not None, cohort_available=cohort['available']))
            continue
        baseline = load(np, raw / (before['cell'] + '.npz'), before['raw_prediction_archive'])
        candidate = load(np, raw / (after['cell'] + '.npz'), after['raw_prediction_archive'])
        masks = load(np, raw / ('be_init_cohorts_' + str(seed) + '.npz'), cohort['cohort_archive'])
        pairs.append(paired(np, baseline, candidate, masks, seed))
        del baseline, candidate, masks
    write(compact / 'PAIRED_REPAIRS.json', dict(schema='internal-be-Wiki24-baseline-cohort-repairs-v1', paired=pairs, limits=LIMITS))
    seed_rows = [dict(seed=seed, cells=[row for row in rows if row['seed'] == seed],
                      primary_pair=next(row for row in pairs if row['seed'] == seed)) for seed in SEEDS]
    write(compact / 'PER_SEED.json', dict(schema='internal-be-Wiki24-selected-per-seed-v1', seeds=seed_rows, limits=LIMITS))
    descriptive = []
    for name in COHORTS:
        blocks = [next(row for row in pair['cohorts'] if row['cohort'] == name) for pair in pairs if pair['available']]
        nonempty = [row for row in blocks if row['nodes']]
        descriptive.append(dict(cohort=name, available_seed_blocks=len(blocks), nonempty_seed_blocks=len(nonempty),
            equal_seed_mean_accuracy_change=mean(np, np.array([row['accuracy_change'] for row in nonempty], dtype=np.float64)),
            equal_seed_mean_nll_change=mean(np, np.array([row['served_nll_change'] for row in nonempty], dtype=np.float64)),
            repairs_sum_over_seed_node_instances=sum(row['repairs'] for row in blocks),
            harms_sum_over_seed_node_instances=sum(row['harms'] for row in blocks),
            net_repairs_sum_over_seed_node_instances=sum(row['net_repairs'] for row in blocks),
            nodes_sum_over_seed_node_instances=sum(row['nodes'] for row in blocks),
            interpretation='Descriptive only; repeated seed-node instances are not independent observations. No node-level inferential test or interval.'))
    write(compact / 'ERROR_SUMMARY.json', dict(schema='internal-be-Wiki24-selected-error-summary-v1',
        exact_roster_cells=24, complete_prediction_cells=len(complete),
        excluded_prediction_cells=24 - len(complete), primary_comparison=descriptive, limits=LIMITS,
        common_competitor='Same nontruth class strictly outranks truth in every original member logit vector; ties excluded. Smallest class index is retained server-only when multiple competitors qualify.',
        pool_rescue='All members predict incorrectly, served mean-softmax predicts truth.',
        pool_harm='At least one member predicts truth, served mean-softmax predicts incorrectly.',
        nll='Float32 log_softmax of original member logits; pooled true-class log probability is logsumexp(member true-class log probabilities) minus log(member count). No clamp, epsilon, fitting, or probability recalibration. Serving stays original float32 mean-softmax.',
        numerical_diagnostics='Counts use actual source predictions. Common-competitor pool successes and harm despite all-member correctness are reported; no tiny floating-point parity/invariant gate.'))
