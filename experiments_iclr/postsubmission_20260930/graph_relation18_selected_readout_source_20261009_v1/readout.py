"""M1/M4 adapter using audited Wiki24 errors and Wiki12 summary/df2 arithmetic."""
import ast
import copy
import hashlib
from pathlib import Path
import time

SEEDS = (6101, 6203, 6307)
CONDITIONS = ('alphaF', 'allJ', 'phiJ', 'relationJ', 'single', 'independent4')
COMPARISONS = (('relationJ-alphaF', 'alphaF', 'relationJ'), ('relationJ-allJ', 'allJ', 'relationJ'),
    ('relationJ-phiJ', 'phiJ', 'relationJ'), ('allJ-alphaF', 'alphaF', 'allJ'), ('phiJ-alphaF', 'alphaF', 'phiJ'),
    ('relationJ-single', 'single', 'relationJ'), ('relationJ-independent4', 'independent4', 'relationJ'),
    ('alphaF-single', 'single', 'alphaF'), ('alphaF-independent4', 'independent4', 'alphaF'))
METRICS = ('served_accuracy', 'served_nll', 'served_brier', 'mean_member_accuracy', 'worst_member_accuracy',
    'mean_member_nll', 'worst_member_nll', 'mean_member_brier', 'worst_member_brier')
LIMITS = dict(exploratory=True, independent_TEST_evidence=False, novelty_or_confirmation_claimed=False,
    selection_population_caveat='All5274 original development nodes selected these states and supply this readout.',
    replication='Three fixed optimizer seeds on one graph; nodes/members are not independent model replicates.',
    historical_controls='Original practical single and ordinary own-selected four-bank; provider/hardware/selector differences preserved; no fresh same-execution claim.',
    missing_controls='Historical controls do not replace single8, objective-matched untied credit, unused confirmation or capable efficiency controls.',
    cohorts='Same-seed alphaF only, all three frozen before candidate arrays; five overlapping cohorts are not additive.',
    member_pairing='Pooled flows plus each actual bank competence; no arbitrary member-index pairing between banks.',
    mechanism='RelationJ changes local scorer and tied-QK recipients together; phiJ overlaps tied-QK factors; no separate local/global causal attribution.',
    numerical_parity_claimed=False, reselection_or_calibration=False)


def pooled_pair_tree(source):
    """Keep original pooled flow arithmetic, removing its hardcoded range(4)."""
    tree = ast.parse(source); function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'paired'))
    removed = 0
    for node in ast.walk(function):
        if isinstance(node, ast.For):
            loops = [n for n in node.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'm']
            for loop in loops:
                if ast.unparse(loop.iter) != 'range(4)':
                    raise ValueError('Original member-pair loop changed')
                node.body.remove(loop); removed += 1
    if removed != 1:
        raise ValueError('Unique original hardcoded member-pair loop required')
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def configure(g, pins):
    analysis = g.module(g.bound(pins['reuse']['legacy_analyse']), '_relation18_reused_Wiki12_analysis')
    contract = analysis.contracts(pins)
    tree = pooled_pair_tree(Path(contract.__file__).read_text()); namespace = dict(vars(contract))
    exec(compile(tree, contract.__file__ + ':pool-flows-only', 'exec'), namespace)
    contract.paired = namespace['paired']
    analysis.SEEDS = SEEDS
    original_summary = analysis.summary
    def summary(np, arrays, mask, original):
        result = original_summary(np, arrays, mask, original)
        result['actual_member_count'] = arrays['member_prediction'].shape[0]
        result['pool_minus_mean_member_accuracy'] = (result['served_accuracy'] - result['mean_member_accuracy']
            if result['served_accuracy'] is not None else None)
        result['within_bank_pairs'] = within_pairs(np, arrays, mask, original)
        return result
    analysis.summary = summary
    analysis.validate = validate
    analysis.LIMITS = LIMITS
    return analysis, contract, hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def validate(np, arrays, original):
    base = original_keys(); members = arrays['member_prediction'].shape[0]
    if members not in (1, 4):
        raise ValueError('Actual M1 or M4 required')
    original.validate(np, {key: arrays[key] for key in base}, members)
    expected = set(base) | {'member_brier', 'pool_brier'}
    if 'member_representations' in arrays:
        expected.add('member_representations')
    if set(arrays) != expected:
        raise ValueError('Exact original archive plus derived Brier/optional original representations')
    if not ((arrays['truth'] >= 0) & (arrays['truth'] < 10)).all():
        raise ValueError('Ten original classes')
    for key, shape in (('member_brier', (members, 5274)), ('pool_brier', (5274,))):
        if arrays[key].shape != shape or arrays[key].dtype != np.dtype('float32') or not np.isfinite(arrays[key]).all():
            raise ValueError('Finite sum-over10 Brier array: ' + key)
    if 'member_representations' in arrays:
        values = arrays['member_representations']
        if values.shape != (members, 5274, 512) or values.dtype != np.dtype('float32') or not np.isfinite(values).all():
            raise ValueError('Complete original relation representations')


def original_keys():
    return ('valid_ids', 'truth', 'member_logits', 'member_probability', 'pool_probability',
        'member_prediction', 'pool_prediction', 'member_nll', 'pool_nll')


def arrays_for(np, g, record, contract):
    arrays = contract.load(np, g.bound(record['raw_archive']), record['raw_archive'])
    if record['historical_reference']:
        contract.validate(np, arrays, record['members'])
        target = np.eye(10, dtype=np.float32)[arrays['truth']]
        arrays['member_brier'] = np.square(arrays['member_probability'] - target[None]).sum(-1, dtype=np.float32)
        arrays['pool_brier'] = np.square(arrays['pool_probability'] - target).sum(-1, dtype=np.float32)
    validate(np, arrays, contract)
    g.require(arrays['member_prediction'].shape[0] == record['members'], 'Actual roster member count')
    return arrays


def within_pairs(np, arrays, mask, contract):
    p = arrays['member_prediction']; correct = p == arrays['truth'][None]; rows = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            disagreement = p[i] != p[j]; one = correct[i] ^ correct[j]; both_wrong = ~correct[i] & ~correct[j]
            count = lambda values: int(values[mask].sum())
            d, o, w = count(disagreement), count(disagreement & one), count(disagreement & both_wrong)
            if d != o + w:
                raise ValueError('Disagreement splits exactly into one-correct and both-wrong')
            intersection, union = count(both_wrong), count(~correct[i] | ~correct[j])
            rows.append(dict(member_indices0=[i, j], nodes=int(mask.sum()), disagreement=d,
                disagreement_exactly_one_correct=o, disagreement_both_wrong=w,
                error_intersection=intersection, error_union=union,
                error_Jaccard=intersection / union if union else None,
                disagreement_rate=contract.mean(np, disagreement[mask]),
                exactly_one_correct_disagreement_rate=contract.mean(np, (disagreement & one)[mask]),
                both_wrong_disagreement_rate=contract.mean(np, (disagreement & both_wrong)[mask])))
    return rows


def rival_acquisition(np, arrays, frozen, mask):
    eligible = mask & frozen['baseline_common_competitor']; classes = frozen['baseline_common_competitor_class'].clip(0)
    logits = arrays['member_logits']; truth = arrays['truth']
    true_values = np.take_along_axis(logits, truth[None, :, None], axis=2)[..., 0]
    rival_values = np.take_along_axis(logits, classes[None, :, None], axis=2)[..., 0]
    win = true_values > rival_values; lose = true_values < rival_values; tie = true_values == rival_values
    return dict(eligible_nodes=int(eligible.sum()), baseline_rival_identity_preserved=True,
        each_member=[dict(member_index0=m, truth_strictly_above_alphaF_rival=int(win[m, eligible].sum()),
            truth_tied_alphaF_rival=int(tie[m, eligible].sum()), alphaF_rival_strictly_above_truth=int(lose[m, eligible].sum())) for m in range(len(logits))],
        any_member_truth_strictly_above_alphaF_rival=int(win.any(0)[eligible].sum()),
        all_members_truth_strictly_above_alphaF_rival=int(win.all(0)[eligible].sum()),
        alphaF_rival_still_strictly_above_truth_in_all_members=int(lose.all(0)[eligible].sum()))


def pair(np, before, after, frozen, seed, label, analysis, contract):
    result = analysis.pair(np, before, after, frozen, seed, label, contract)
    a, b = contract.error_masks(np, before), contract.error_masks(np, after)
    result['member_index_pairing_reported'] = False
    for row in result['cohorts']:
        mask = frozen[row['cohort']]; count = lambda x: int(x[mask].sum())
        row['coverage_flows'] = dict(gained=count(~a['any_member_correct'] & b['any_member_correct']),
            lost=count(a['any_member_correct'] & ~b['any_member_correct']),
            retained=count(a['any_member_correct'] & b['any_member_correct']),
            neither=count(~a['any_member_correct'] & ~b['any_member_correct']))
        row['coverage_flows']['net_gained'] = row['coverage_flows']['gained'] - row['coverage_flows']['lost']
        for key in ('pool_rescue', 'pool_harm'):
            row[key + '_flows'] = dict(appeared=count(~a[key] & b[key]), cleared=count(a[key] & ~b[key]),
                persisted=count(a[key] & b[key]))
        row['alphaF_rival_acquisition'] = dict(before=rival_acquisition(np, before, frozen, mask),
            after=rival_acquisition(np, after, frozen, mask))
        row['bank_competence_changes'] = {metric: (row['candidate'][metric] - row['baseline'][metric]
            if row['candidate'][metric] is not None and row['baseline'][metric] is not None else None) for metric in METRICS}
    return result


def gate_result(contrasts, whole18, protocol):
    by_label = {row['comparison']: row for row in contrasts}; primary = by_label['relationJ-alphaF']['metrics']
    source_rule = protocol['unchanged_pilot_screen']
    expected = dict(positive_primary_accuracy_all3seeds=True, minimum_mean_primary_gain_pp=.2,
        mean_pool_NLL_not_worse=True, maximum_mean_member_accuracy_degradation_pp=.1,
        maximum_worst_member_accuracy_degradation_pp=.2, same_development_optimizer_replicates_not_confirmation=True)
    if source_rule != expected:
        raise ValueError('Unchanged relation primary screen required')
    screen_metrics = ('served_accuracy', 'served_nll', 'mean_member_accuracy', 'worst_member_accuracy')
    available = all(primary[m]['available_seeds'] == 3 for m in screen_metrics)
    checks = dict(positive_primary_accuracy_all3seeds=available and all(x > 0 for x in primary['served_accuracy']['values']),
        minimum_mean_primary_gain_pp=available and primary['served_accuracy']['mean'] >= .2,
        mean_pool_NLL_not_worse=available and primary['served_nll']['mean'] <= 0,
        mean_member_accuracy_degradation_limit=available and primary['mean_member_accuracy']['mean'] >= -.1,
        worst_member_accuracy_degradation_limit=available and primary['worst_member_accuracy']['mean'] >= -.2)
    placement = {c: by_label['relationJ-' + c]['metrics']['served_accuracy'] for c in ('allJ', 'phiJ')}
    placement_available = available and all(row['available_seeds'] == 3 for row in placement.values())
    return dict(available=available, whole_fixed18_readout_available=whole18, original_primary_rule=source_rule, checks=checks,
        primary_screen_passed=all(checks.values()), original_placement_rule=protocol['attribution_gate'],
        placement_comparison='Equal-seed mean served accuracy; exact matches count as matches; all seed deltas retained.',
        allJ_or_phiJ_matches_or_exceeds_relationJ=(any(row['mean'] <= 0 for row in placement.values()) if placement_available else None),
        recipient_placement_supported=placement_available and all(checks.values()) and all(row['mean'] > 0 for row in placement.values()),
        historical_comparisons_are_descriptive_not_added_gates=True, further_execution_authorized=False)


def run(np, g, pins, output, collection, analysis, contract):
    compact = output / 'compact'; rows = []; summaries = {}; records = {}; pairs = []; frozen = {}
    for seed in SEEDS:
        cohort = collection['cohorts'][str(seed)]
        if cohort.get('available'):
            masks = contract.load(np, g.bound(cohort['archive']), cohort['archive'])
            alpha = next(r for r in collection['cells'] if (r['condition'], r['seed']) == ('alphaF', seed))
            baseline = arrays_for(np, g, alpha, contract); expected = contract.baseline_cohorts(np, baseline)
            g.require(set(masks) == set(expected) and all(np.array_equal(masks[k], expected[k]) for k in expected), 'Immutable complete alphaF cohort/rival seal')
            frozen[seed] = masks; del baseline
    for record in collection['cells']:
        row = {k: record[k] for k in ('cell', 'seed', 'condition', 'members', 'family_status', 'collection_status', 'historical_reference')}
        row['selected_metadata'] = record.get('selected_metadata')
        if record['collection_status'] == 'complete':
            try:
                arrays = arrays_for(np, g, record, contract)
                whole = analysis.summary(np, arrays, np.ones(5274, dtype=np.bool_), contract)
                row['full_population'] = whole
                row['served_confusion_counts_truth_rows_prediction_columns'] = np.bincount(arrays['truth'] * 10 + arrays['pool_prediction'], minlength=100).reshape(10, 10).tolist()
                row['source_float32_reevaluation'] = record.get('source_float32_reevaluation')
                row['historical_original_metadata'] = record.get('historical_original_metadata')
                summaries[(record['seed'], record['condition'])] = whole; records[(record['seed'], record['condition'])] = record
                del arrays
            except Exception as error:
                row['collection_status'] = record['collection_status'] = 'retained_analysis_archive_failure'
                row['unavailable_reason'] = record['failure'] = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
        else:
            row['unavailable_reason'] = record.get('failure')
        rows.append(row)
    contrasts = []
    for label, left, right in COMPARISONS:
        metrics = {}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                a, b = summaries.get((seed, left), {}).get(metric), summaries.get((seed, right), {}).get(metric)
                value = b - a if a is not None and b is not None else None
                values.append(value * 100 if value is not None and 'accuracy' in metric else value)
            metrics[metric] = dict(unit='percentage points' if 'accuracy' in metric else ('nats' if 'nll' in metric else 'Brier sum over10classes'), **analysis.seed_interval(values))
        contrasts.append(dict(comparison=label, from_condition=left, to_condition=right, metrics=metrics))
        for seed in SEEDS:
            a, b = records.get((seed, left)), records.get((seed, right))
            if a is None or b is None or seed not in frozen:
                pairs.append(dict(seed=seed, comparison=label, available=False, reason='Complete banks and all-three-frozen alphaF cohorts required')); continue
            try:
                before, after = arrays_for(np, g, a, contract), arrays_for(np, g, b, contract)
                result = pair(np, before, after, frozen[seed], seed, label, analysis, contract)
                result.update(from_condition=left, to_condition=right); pairs.append(result); del before, after
            except Exception as error:
                pairs.append(dict(seed=seed, comparison=label, available=False,
                    failure=dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)))
    seed_flows = []
    flow_names = ('repairs', 'harms', 'net_repairs', 'both_wrong_same_class', 'both_wrong_different_class',
        'accuracy_change', 'served_nll_change', 'served_brier_change')
    flow_names += tuple('coverage_flows.' + k for k in ('gained', 'lost', 'retained', 'neither', 'net_gained'))
    flow_names += tuple(name + '.' + k for name in ('pool_rescue_flows', 'pool_harm_flows') for k in ('appeared', 'cleared', 'persisted'))
    flow_names += tuple('common_wrong_competitor_flows.' + k for k in ('cleared', 'appeared', 'persisted', 'persisted_same_smallest_competitor'))
    flow_names += tuple('bank_competence_changes.' + k for k in METRICS)
    for label, _, _ in COMPARISONS:
        for cohort in contract.COHORTS:
            values_by_name = {name: [None] * 3 for name in flow_names}
            for seed in SEEDS:
                pair_row = next(r for r in pairs if r['comparison'] == label and r['seed'] == seed)
                block = next((r for r in pair_row.get('cohorts', []) if r['cohort'] == cohort), None)
                scalars = {}
                if block is not None and block['nodes']:
                    scalars.update({k: block[k] for k in ('repairs', 'harms', 'net_repairs', 'both_wrong_same_class', 'both_wrong_different_class', 'accuracy_change', 'served_nll_change', 'served_brier_change')})
                    for name in ('coverage_flows', 'pool_rescue_flows', 'pool_harm_flows', 'common_wrong_competitor_flows', 'bank_competence_changes'):
                        scalars.update({name + '.' + k: v for k, v in block[name].items()})
                for name, value in scalars.items():
                    values_by_name.setdefault(name, [None] * 3)[SEEDS.index(seed)] = value
            seed_flows.append(dict(comparison=label, cohort=cohort, statistics={k: analysis.seed_interval(v) for k, v in values_by_name.items()}))
    whole18 = len(summaries) == 18 and len(frozen) == 3 and all(p.get('available') for p in pairs)
    screen = gate_result(contrasts, whole18, g.read(g.bound(pins['reuse']['relation_protocol'])))
    for filename, value in (
        ('PER_CELL.json', dict(cells=rows, limits=LIMITS)),
        ('PER_SEED.json', dict(seeds=[dict(seed=s, cells=[r for r in rows if r['seed'] == s]) for s in SEEDS], limits=LIMITS)),
        ('CONTRASTS.json', dict(ordered_comparisons=contrasts, limits=LIMITS)),
        ('PAIRED_ERRORS.json', dict(paired=pairs, limits=LIMITS, Brier='Sum over ten classes; no division, clipping or fitting.',
            common_rival='Same nontruth class strictly above truth in every member; ties excluded; smallest qualifying class identity kept in sealed alphaF raw cohorts.')),
        ('PAIRED_SEED_SUMMARY.json', dict(comparisons=seed_flows, limits=LIMITS)),
        ('UNCHANGED_SCREEN.json', screen)):
        contract.write(compact / filename, value)
    lines = ['# Relation18 selected-development readout', '',
        'Original selected states; historical plain single and own-selected ordinary4. Same5274 selection nodes, three fixed optimizer seeds, one graph. No TEST evidence, fresh historical-control claim or confirmation.', '',
        '| Comparison | Accuracy mean (pp) | SD | Descriptive df2 95% interval | NLL mean | Brier mean |',
        '|---|---:|---:|---|---:|---:|']
    number = lambda x: 'unavailable' if x is None else format(x, '.6g')
    for row in contrasts:
        m = row['metrics']; a = m['served_accuracy']; interval = a['exploratory_95pct_df2_interval']
        ci = 'unavailable' if interval is None else '[' + number(interval[0]) + ', ' + number(interval[1]) + ']'
        lines.append('| ' + row['comparison'] + ' | ' + number(a['mean']) + ' | ' + number(a['SD']) + ' | ' + ci + ' | ' + number(m['served_nll']['mean']) + ' | ' + number(m['served_brier']['mean']) + ' |')
    lines.extend(['', 'Primary screen passed: ' + str(screen['primary_screen_passed']) + '. Recipient placement supported: ' + str(screen['recipient_placement_supported']) + '.', '',
        'All three paired values, all actual members, mean/worst competence, repairs, introduced errors, coverage, pool rescue/harm and common-rival flows are retained in compact JSON. Worst is defined independently per metric. Raw arrays and per-node rival identities remain server-only.', '',
        'Historical original metrics, training/selection/resource/owner costs and provider/hardware differences are preserved in ORIGINAL_COSTS.json and selected metadata. COST.json adds this reader’s actual work and retained failures. No survivor-only intervals, node bootstrap, calibration or reselection.', ''])
    (compact / 'REPORT.md').write_text('\n'.join(lines))
    collection['analysis_complete'] = whole18
    return screen
