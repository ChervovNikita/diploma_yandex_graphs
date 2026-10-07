"""Wiki12 descriptive analysis; audited Wiki24 error/NLL/cohort contracts reused."""
import math
from pathlib import Path
import statistics
from gate import CONDITIONS, SEEDS, bound, module, require

ROLE = 'WikiCS split0 development: official validation union stopping masks;5274 nodes'
LIMITS = dict(role=ROLE, selection_population_caveat='The same development population selected these original checkpoints and supplies this readout; selected-state optimism applies.',
    independent_TEST_evidence=False, exploratory=True, novelty_or_confirmation_claimed=False,
    replication='Three reused fixed optimizer seeds on one graph; nodes and members are not independent fitted-model replicates.',
    numerical_parity_claimed=False, reselection_or_calibration=False,
    cohorts='Frozen from same-seed plain before all candidate predictions; overlapping cohorts are not additive.',
    mechanism='Interpret component contrasts after fresh C-P; no rescue selection or extra fits if the reference gain fails to reproduce.')
BASE_KEYS = ('valid_ids', 'truth', 'member_logits', 'member_probability', 'pool_probability',
             'member_prediction', 'pool_prediction', 'member_nll', 'pool_nll')
CONTRASTS = [('C-P', {'combined': 1, 'plain': -1}),
             ('A-P', {'alignment_only': 1, 'plain': -1}),
             ('R-P', {'residual_only': 1, 'plain': -1}),
             ('C-A', {'combined': 1, 'alignment_only': -1}),
             ('C-R', {'combined': 1, 'residual_only': -1}),
             ('C-A-R+P', {'combined': 1, 'alignment_only': -1, 'residual_only': -1, 'plain': 1})]
METRICS = ('served_accuracy', 'served_nll', 'served_brier', 'mean_member_accuracy', 'worst_member_accuracy',
           'mean_member_nll', 'worst_member_nll', 'mean_member_brier', 'worst_member_brier')


def contracts(pins):
    return module(bound(pins['audited_Wiki24_analysis']), '_wiki12_audited_Wiki24_contracts')


def validate(np, arrays, original):
    require(set(arrays) == set(BASE_KEYS) | {'member_representations', 'member_brier', 'pool_brier'}, 'Exact complete Wiki12 archive')
    original.validate(np, {key: arrays[key] for key in BASE_KEYS}, 4)
    for key, shape in (('member_representations', (4, 5274, 512)), ('member_brier', (4, 5274)), ('pool_brier', (5274,))):
        require(arrays[key].shape == shape and arrays[key].dtype == np.dtype('float32')
                and np.isfinite(arrays[key]).all(), 'Finite fixed raw array shape/dtype: ' + key)
    require(((arrays['truth'] >= 0) & (arrays['truth'] < 10)).all(), 'Ten original truth classes')


def summary(np, arrays, mask, original):
    result = original.scalar_errors(np, arrays, mask)
    n = int(mask.sum())
    result['worst_member_accuracy'] = result['minimum_member_accuracy']
    result['served_brier'] = original.mean(np, arrays['pool_brier'][mask])
    for member, row in enumerate(result['members']):
        row['brier'] = original.mean(np, arrays['member_brier'][member, mask])
    result['mean_member_brier'] = original.mean(np, arrays['member_brier'][:, mask].mean(0, dtype=np.float64))
    nlls = [row['nll'] for row in result['members']]; briers = [row['brier'] for row in result['members']]
    result['worst_member_nll'] = max(nlls) if n and all(x is not None for x in nlls) else None
    result['worst_member_brier'] = max(briers) if n and all(x is not None for x in briers) else None
    result['worst_member_definition'] = 'Minimum accuracy, maximum NLL/Brier separately; worst indices can differ by metric.'
    return result


def pair(np, before, after, cohorts, seed, label, original):
    result = original.paired(np, before, after, cohorts, seed)
    result['comparison'] = label
    flags_before, flags_after = original.error_masks(np, before), original.error_masks(np, after)
    for row in result['cohorts']:
        mask = cohorts[row['cohort']]
        row['baseline'] = summary(np, before, mask, original); row['candidate'] = summary(np, after, mask, original)
        row['served_brier_change'] = original.mean(np, after['pool_brier'][mask].astype(np.float64) - before['pool_brier'][mask].astype(np.float64))
        for member, change in enumerate(row['member_changes']):
            change['brier_change'] = original.mean(np, after['member_brier'][member, mask].astype(np.float64) - before['member_brier'][member, mask].astype(np.float64))
        bc, ac = flags_before['common_competitor'], flags_after['common_competitor']
        row['common_wrong_competitor_flows'] = dict(
            cleared=int((bc & ~ac)[mask].sum()), appeared=int((~bc & ac)[mask].sum()), persisted=int((bc & ac)[mask].sum()),
            persisted_same_smallest_competitor=int((bc & ac & (flags_before['common_competitor_class'] == flags_after['common_competitor_class']))[mask].sum()))
    return result


def seed_interval(values):
    require(len(values) == 3, 'All three fixed seed slots required')
    finite = [x for x in values if x is not None and math.isfinite(x)]
    if len(finite) != 3:
        return dict(values=values, fixed_seeds=list(SEEDS), available_seeds=len(finite), mean=None, SD=None,
                    exploratory_95pct_df2_interval=None, reason='No survivor-only interval; all3 required')
    average = statistics.mean(finite); sd = statistics.stdev(finite)
    margin = 4.302652729911275 * sd / math.sqrt(3)
    return dict(values=values, fixed_seeds=list(SEEDS), available_seeds=3, mean=average, SD=sd, df=2,
        exploratory_95pct_df2_interval=[average-margin, average+margin],
        interpretation='Descriptive paired optimizer-seed interval on one graph; no node/member independence or confirmation claim.')


def run(output, collection, pins):
    import numpy as np
    original = contracts(pins); output = Path(output); raw = output / 'raw'; compact = output / 'compact'
    rows = []; summaries = {}; records = {}; pairs = []
    for record in collection['cells']:
        row = {key: record[key] for key in ('cell', 'seed', 'condition', 'method_identity', 'family_status', 'collection_status')}
        if record['collection_status'] != 'complete':
            row['unavailable_reason'] = record.get('failure'); rows.append(row); continue
        arrays = original.load(np, raw / (record['cell'] + '.npz'), record['raw_archive'])
        validate(np, arrays, original)
        whole = summary(np, arrays, np.ones(5274, dtype=np.bool_), original)
        row.update(full_population=whole, selected_metadata=record['selected_metadata'],
            source_float32_reevaluation=record['source_float32_reevaluation'],
            reevaluation_minus_stored_served_accuracy=record['source_float32_reevaluation']['accuracy'] - record['selected_metadata']['stored_accuracy'],
            reevaluation_minus_stored_member_accuracy=[a-b for a, b in zip(record['source_float32_reevaluation']['members'], record['selected_metadata']['stored_member_accuracy'])])
        flags = original.error_masks(np, arrays)
        row['common_competitor_class_counts_smallest_qualifying_class'] = np.bincount(flags['common_competitor_class'][flags['common_competitor']], minlength=10).tolist()
        row['served_confusion_counts_truth_rows_prediction_columns'] = np.bincount(arrays['truth']*10+arrays['pool_prediction'], minlength=100).reshape(10,10).tolist()
        summaries[(record['seed'], record['condition'])] = whole; records[(record['seed'], record['condition'])] = record
        rows.append(row); del arrays
    contrasts = []
    for label, weights in CONTRASTS:  # C-P is first throughout reports.
        metric_rows = {}
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                terms = [summaries.get((seed, condition), {}).get(metric) for condition in weights]
                value = None if any(x is None for x in terms) else sum(weights[c]*summaries[(seed,c)][metric] for c in weights)
                values.append(value * 100 if value is not None and 'accuracy' in metric else value)
            metric_rows[metric] = dict(unit='percentage points' if 'accuracy' in metric else ('nats' if 'nll' in metric else 'Brier sum over10classes'), **seed_interval(values))
        contrasts.append(dict(comparison=label, coefficients=weights, metrics=metric_rows))
        if len(weights) != 2:
            continue  # Interaction has no invented repair/harms interpretation.
        left = next(c for c, w in weights.items() if w == -1); right = next(c for c, w in weights.items() if w == 1)
        for seed in SEEDS:
            before, after = records.get((seed,left)), records.get((seed,right)); cohort = collection['cohorts'][str(seed)]
            if before is None or after is None or not cohort.get('available'):
                pairs.append(dict(seed=seed, comparison=label, available=False, reason='Complete pair and frozen plain cohort seal required')); continue
            a = original.load(np, raw / (before['cell']+'.npz'), before['raw_archive'])
            b = original.load(np, raw / (after['cell']+'.npz'), after['raw_archive'])
            masks = original.load(np, raw / ('plain_cohorts_'+str(seed)+'.npz'), cohort['archive'])
            validate(np,a,original); validate(np,b,original)
            paired = pair(np,a,b,masks,seed,label,original); paired['from_condition']=left; paired['to_condition']=right
            pairs.append(paired); del a,b,masks
    original.write(compact / 'PER_CELL.json', dict(cells=rows, limits=LIMITS))
    original.write(compact / 'CONTRASTS.json', dict(ordered_comparisons=contrasts, limits=LIMITS, accuracy_higher_better=True, NLL_and_Brier_lower_better=True))
    original.write(compact / 'PAIRED_ERRORS.json', dict(paired=pairs, limits=LIMITS,
        common_wrong_competitor='Same nontruth class strictly outranks truth in every original member logit vector; ties excluded; smallest qualifying class is retained server-side.',
        Brier='Sum_k(probability_k - onehot_truth_k)^2; no division by10, clipping, epsilon or fitting.'))
    original.write(compact / 'PER_SEED.json', dict(seeds=[dict(seed=seed,cells=[row for row in rows if row['seed']==seed]) for seed in SEEDS],limits=LIMITS))
    lines = ['# Wiki12 exploratory component attribution', '', ROLE + '.', '',
        'Original selected states; the selection population is reused for this analysis. No independent TEST evidence. Three optimizer seeds on one graph; nodes and members are not independent model replicates.', '',
        '| Contrast | Accuracy mean (pp) | SD | Exploratory df2 95% interval | NLL mean | Brier mean |',
        '|---|---:|---:|---|---:|---:|']
    def number(value):
        return 'unavailable' if value is None else format(value,'.6g')
    for row in contrasts:
        m=row['metrics']; a=m['served_accuracy']; interval=a['exploratory_95pct_df2_interval']
        ci='unavailable' if interval is None else '['+number(interval[0])+', '+number(interval[1])+']'
        lines.append('| '+row['comparison']+' | '+number(a['mean'])+' | '+number(a['SD'])+' | '+ci+' | '+number(m['served_nll']['mean'])+' | '+number(m['served_brier']['mean'])+' |')
    lines += ['', 'Assess fresh C-P before component attribution. These intervals are descriptive and do not establish novelty, confirmation or a graph mechanism. No seed/checkpoint rescue selection.', '',
              'PER_CELL/PER_SEED contain accuracy, NLL, Brier and all four members plus mean/worst competence. PAIRED_ERRORS contains repairs, harms and common-wrong-competitor flows on plain-frozen overlapping cohorts. Raw logits, representations, IDs, truth and masks remain server-side.', '']
    (compact/'REPORT.md').write_text('\n'.join(lines))
