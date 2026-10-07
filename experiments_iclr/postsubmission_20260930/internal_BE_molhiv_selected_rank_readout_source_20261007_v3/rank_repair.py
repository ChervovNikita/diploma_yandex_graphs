"""O-frozen complete-development diagnostics; numerical imports occur only on call."""
import hashlib
from protocol import CONDITIONS, SEEDS, bound, paired, write

NAMES = ('loss', 'tie', 'win')
MAX_BLOCK_PAIRS = 65536
PAIR_ID_SCHEMA = 'ordered-positive-negative-original-ids-int64le-v1'
CALIBRATION_BINS = 10
CONTRASTS = (('I', 'independent4'), ('I', 'single'), ('I', 'O'), ('I', 'P'), ('P', 'G'), ('I', 'G'))
DIAGNOSTIC_DIRECTIONS = {
    'average_precision': 'higher', 'PR_AUC_trapezoidal': 'higher',
    'Brier': 'lower', 'BCE': 'lower', 'ECE_equal_width_10': 'lower',
    'MCE_occupied_bins': 'lower', 'absolute_mean_probability_minus_prevalence': 'lower',
    'mean_probability_minus_prevalence': 'signed; magnitude measures marginal mismatch',
    'positive_BCE': 'lower', 'negative_BCE': 'lower',
}


def blocks(positive, negative):
    for a in range(0, len(positive), 32):
        for b in range(0, len(negative), 2048):
            yield slice(a, a + 32), slice(b, b + 2048)


def states(scores, positive, negative, a, b, np):
    p, n = scores[positive[a], None], scores[negative[b]][None, :]
    return (p > n).astype(np.uint8) * 2 + (p == n).astype(np.uint8)


def ordered_pair_identity(positive_ids, negative_ids, np):
    """Hash original ID pairs in positive-major row order, not block traversal order."""
    digest = hashlib.sha256((PAIR_ID_SCHEMA + '\0').encode('ascii'))
    digest.update(np.asarray([len(positive_ids), len(negative_ids)], dtype='<i8').tobytes())
    for positive_id in positive_ids:
        for start in range(0, len(negative_ids), 2048):
            negatives = negative_ids[start:start + 2048]
            pairs = np.empty((len(negatives), 2), dtype='<i8')
            pairs[:, 0], pairs[:, 1] = positive_id, negatives
            digest.update(pairs.tobytes(order='C'))
    return digest.hexdigest()


def load_raw(path, np):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'ids', 'labels', 'member_logits', 'pool_logits'}:
            raise ValueError('Exact raw prediction fields required')
        raw = {key: archive[key].copy() for key in archive.files}
    labels, ids, logits, pool = (raw[k] for k in ('labels', 'ids', 'member_logits', 'pool_logits'))
    if ids.shape != (4113,) or ids.dtype != np.int64 or len(np.unique(ids)) != 4113 or labels.shape != (4113,) \
            or labels.dtype != np.float32 or logits.shape not in ((1, 4113), (4, 4113)) \
            or pool.shape != (4113,) or logits.dtype != np.float32 or pool.dtype != np.float32 \
            or not all(np.isfinite(v).all() for v in (labels, logits, pool)) or not np.isin(labels, (0, 1)).all():
        raise ValueError('Complete finite original4113 raw predictions required; no sampled rescue')
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError('Both complete finite target classes required')
    return raw


def freeze_baseline(raw, destination):
    """All fields and ordered pair identity saved before any non-O prediction collection."""
    import numpy as np
    positive, negative = np.flatnonzero(raw['labels'] == 1), np.flatnonzero(raw['labels'] == 0)
    if not len(positive) or not len(negative):
        raise ValueError('Both finite target classes required')
    positive_ids, negative_ids = raw['ids'][positive], raw['ids'][negative]
    identity = ordered_pair_identity(positive_ids, negative_ids, np)
    base = np.empty((len(positive), len(negative)), dtype=np.uint8)
    common = np.ones_like(base, dtype=np.bool_)
    for a, b in blocks(positive, negative):
        base[a, b] = states(raw['pool_logits'], positive, negative, a, b, np)
        for member in raw['member_logits']:
            common[a, b] &= member[positive[a], None] < member[negative[b]][None, :]
    np.savez(destination, positive=positive, negative=negative, positive_ids=positive_ids,
             negative_ids=negative_ids, baseline_state=base, common_inversion=common,
             ordered_pair_identity_sha256=np.asarray(identity, dtype='S64'),
             pair_identity_schema=np.asarray(PAIR_ID_SCHEMA, dtype='S64'))
    common_states = np.bincount(base[common], minlength=3)
    return {'pairs': int(base.size), 'O_pool_states': dict(zip(NAMES, map(int, np.bincount(base.ravel(), minlength=3)))),
            'O_same_pair_strictly_inverted_by_all_members': int(common.sum()),
            'O_common_inversion_pool_states': dict(zip(NAMES, map(int, common_states))),
            'O_common_inversion_pool_nonloss_rounding_boundary_count': int(common_states[1:].sum()),
            'ordered_pair_identity_sha256': identity, 'pair_identity_schema': PAIR_ID_SCHEMA,
            'pair_order': 'positive positions in original role order, then negative positions in original role order',
            'definition': 'same positive-negative pair strictly inverted by every O member; ties excluded'}


def probabilities(scores, np):
    z = scores.astype(np.float64)
    p = np.empty_like(z)
    nonnegative = z >= 0
    p[nonnegative] = 1. / (1. + np.exp(-z[nonnegative]))
    exponential = np.exp(z[~nonnegative])
    p[~nonnegative] = exponential / (1. + exponential)
    return p


def probability_diagnostics(scores, labels, np):
    """No fitting or threshold selection: grouped raw-logit PR and fixed probability bins."""
    z, y = scores.astype(np.float64), labels.astype(np.float64)
    p = probabilities(scores, np)
    n, positives = len(y), int(y.sum())
    if not 0 < positives < n:
        raise ValueError('Both classes required for PR/calibration summaries')
    # Score ties enter together. AP is a step integral; trapezoidal PR is separate.
    order = np.argsort(-z, kind='stable')
    ranked_z, ranked_y = z[order], y[order]
    ends = np.r_[np.flatnonzero(ranked_z[:-1] != ranked_z[1:]), n - 1]
    tp = np.cumsum(ranked_y)[ends]
    precision, recall = tp / (ends + 1), tp / positives
    recall_steps = np.diff(np.r_[0., recall])
    ap = float(np.sum(recall_steps * precision))
    pr_auc = float(np.sum(np.diff(np.r_[0., recall]) *
                          (np.r_[1., precision][:-1] + np.r_[1., precision][1:]) / 2.))
    loss = np.logaddexp(0., z) - y * z
    bins = np.minimum((p * CALIBRATION_BINS).astype(np.int64), CALIBRATION_BINS - 1)
    reliability, ece, gaps = [], 0., []
    for index in range(CALIBRATION_BINS):
        mask = bins == index
        count = int(mask.sum())
        confidence = float(p[mask].mean()) if count else None
        frequency = float(y[mask].mean()) if count else None
        gap = abs(confidence - frequency) if count else None
        if count:
            ece += count / n * gap
            gaps.append(gap)
        reliability.append({'bin': index, 'lower': index / CALIBRATION_BINS,
                            'upper': (index + 1) / CALIBRATION_BINS, 'upper_inclusive': index == CALIBRATION_BINS - 1,
                            'count': count, 'positive_count': int(y[mask].sum()),
                            'mean_probability': confidence, 'observed_positive_frequency': frequency,
                            'absolute_gap': gap})
    prevalence, mean_probability = positives / n, float(p.mean())
    return {'average_precision': ap, 'PR_AUC_trapezoidal': pr_auc,
            'Brier': float(np.square(p - y).mean()), 'BCE': float(loss.mean()),
            'positive_BCE': float(loss[y == 1].mean()), 'negative_BCE': float(loss[y == 0].mean()),
            'ECE_equal_width_10': float(ece), 'MCE_occupied_bins': max(gaps),
            'prevalence': prevalence, 'mean_probability': mean_probability,
            'mean_probability_minus_prevalence': mean_probability - prevalence,
            'absolute_mean_probability_minus_prevalence': abs(mean_probability - prevalence),
            'reliability_equal_width_10': reliability,
            'distinct_raw_logit_score_groups': len(ends), 'tied_score_groups': int(np.sum(np.diff(np.r_[-1, ends]) > 1)),
            'logit_mean': float(z.mean()), 'logit_SD_population': float(z.std()),
            'logit_min': float(z.min()), 'logit_max': float(z.max())}


def competence(raw, np):
    positive, negative = np.flatnonzero(raw['labels'] == 1), np.flatnonzero(raw['labels'] == 0)
    counts = np.zeros((len(raw['member_logits']) + 1, 3), dtype=np.int64)
    bank = [*raw['member_logits'], raw['pool_logits']]
    for a, b in blocks(positive, negative):
        for index, scores in enumerate(bank):
            counts[index] += np.bincount(states(scores, positive, negative, a, b, np).ravel(), minlength=3)
    total = len(positive) * len(negative)
    aucs = [(int(row[2]) + .5 * int(row[1])) / total for row in counts]
    diagnostics = [probability_diagnostics(scores, raw['labels'], np) for scores in bank]
    member_summary = {}
    for metric in DIAGNOSTIC_DIRECTIONS:
        values = [row[metric] for row in diagnostics[:-1]]
        member_summary[metric] = {'mean': sum(values) / len(values), 'min': min(values), 'max': max(values),
                                  'pool_minus_member_mean': diagnostics[-1][metric] - sum(values) / len(values),
                                  'favorable_direction': DIAGNOSTIC_DIRECTIONS[metric]}
    return {'positive_count': len(positive), 'negative_count': len(negative), 'finite_pair_count': total,
            'member_pair_AUC': aucs[:-1], 'member_AUC_mean': sum(aucs[:-1]) / (len(aucs) - 1),
            'member_AUC_min': min(aucs[:-1]), 'member_AUC_max': max(aucs[:-1]),
            'pool_pair_AUC': aucs[-1], 'member_BCE': [row['BCE'] for row in diagnostics[:-1]],
            'served_mean_raw_logit_BCE': diagnostics[-1]['BCE'],
            'member_pair_states': [dict(zip(NAMES, map(int, row))) for row in counts[:-1]],
            'pool_pair_states': dict(zip(NAMES, map(int, counts[-1]))),
            'probabilistic_diagnostics': {'members': diagnostics[:-1], 'served': diagnostics[-1],
                                        'member_summary': member_summary,
                                        'probability_definition': 'stable sigmoid of each source raw logit; served is sigmoid(source mean raw logits)',
                                        'PR_definition': 'raw-logit grouped ties; AP step integral and separately labeled trapezoidal PR area',
                                        'calibration_definition': 'descriptive positive-event probabilities in ten fixed equal-width bins; no fitted calibrator'}}


def validate_cohort(base, cohort, np):
    expected = {'positive', 'negative', 'positive_ids', 'negative_ids', 'baseline_state', 'common_inversion',
                'ordered_pair_identity_sha256', 'pair_identity_schema'}
    if set(cohort) != expected:
        raise ValueError('Exact O-only frozen v3 cohort fields required')
    positive, negative = cohort['positive'], cohort['negative']
    shape = (len(positive), len(negative))
    if not np.array_equal(positive, np.flatnonzero(base['labels'] == 1)) \
            or not np.array_equal(negative, np.flatnonzero(base['labels'] == 0)) \
            or positive.dtype != np.int64 or negative.dtype != np.int64 \
            or cohort['positive_ids'].dtype != np.int64 or cohort['negative_ids'].dtype != np.int64 \
            or not np.array_equal(cohort['positive_ids'], base['ids'][positive]) \
            or not np.array_equal(cohort['negative_ids'], base['ids'][negative]) \
            or cohort['baseline_state'].shape != shape or cohort['baseline_state'].dtype != np.uint8 \
            or (cohort['baseline_state'] > 2).any() or cohort['common_inversion'].shape != shape \
            or cohort['common_inversion'].dtype != np.bool_:
        raise ValueError('Frozen complete O-only ordered pair populations/state domains differ')
    if cohort['pair_identity_schema'].shape != () or cohort['ordered_pair_identity_sha256'].shape != () \
            or cohort['pair_identity_schema'].item() != PAIR_ID_SCHEMA.encode('ascii') \
            or cohort['ordered_pair_identity_sha256'].item() != ordered_pair_identity(base['ids'][positive], base['ids'][negative], np).encode('ascii'):
        raise ValueError('Frozen O ordered pair identity/hash differs')
    return positive, negative


def transitions(base, candidate, cohort, np, candidate_name='I'):
    """Exact O->candidate rank credit plus per-pair coverage on the fixed O cohort."""
    if not np.array_equal(base['ids'], candidate['ids']) or not np.array_equal(base['labels'], candidate['labels']):
        raise ValueError('Paired full development IDs/labels differ')
    positive, negative = validate_cohort(base, cohort, np)
    tables = {name: np.zeros((3, 3), dtype=np.int64)
              for name in ('all_pairs', 'O_pool_loss', 'O_pool_tie', 'O_pool_win', 'O_common_inversion')}
    coverage = np.zeros((2, 3), dtype=np.int64)  # strict member coverage false/true by pool loss/tie/win.
    win_histogram = np.zeros(len(candidate['member_logits']) + 1, dtype=np.int64)
    tie_histogram = np.zeros_like(win_histogram)
    weak_coverage_count = 0
    matched = len(base['member_logits']) == len(candidate['member_logits'])
    member_tables = [np.zeros((3, 3), dtype=np.int64) for _ in candidate['member_logits']] if matched else None
    for a, b in blocks(positive, negative):
        old, new = cohort['baseline_state'][a, b], states(candidate['pool_logits'], positive, negative, a, b, np)
        code = old * 3 + new
        masks = {'all_pairs': np.ones_like(old, dtype=np.bool_), 'O_pool_loss': old == 0,
                 'O_pool_tie': old == 1, 'O_pool_win': old == 2,
                 'O_common_inversion': cohort['common_inversion'][a, b]}
        for name, mask in masks.items():
            tables[name] += np.bincount(code[mask], minlength=9).reshape(3, 3)
        wins, ties = np.zeros_like(old, dtype=np.uint8), np.zeros_like(old, dtype=np.uint8)
        for member, scores in enumerate(candidate['member_logits']):
            current = states(scores, positive, negative, a, b, np)
            wins += current == 2; ties += current == 1
            if matched:
                old_member = states(base['member_logits'][member], positive, negative, a, b, np)
                member_tables[member] += np.bincount((old_member * 3 + current).ravel(), minlength=9).reshape(3, 3)
        mask = masks['O_common_inversion']
        coverage += np.bincount(((wins > 0).astype(np.uint8) * 3 + new)[mask], minlength=6).reshape(2, 3)
        win_histogram += np.bincount(wins[mask], minlength=len(win_histogram))
        tie_histogram += np.bincount(ties[mask], minlength=len(tie_histogram))
        weak_coverage_count += int(((wins > 0) | (ties > 0))[mask].sum())
    all_pair_count = len(positive) * len(negative)
    result = {}
    for name, table in tables.items():
        total = int(table.sum())
        gain = sum(int(table[i, j]) * (j - i) / 2 for i in range(3) for j in range(i + 1, 3))
        harm = sum(int(table[i, j]) * (i - j) / 2 for i in range(3) for j in range(i))
        result[name] = {'pairs': total, 'O_to_' + candidate_name: {NAMES[i] + '_to_' + NAMES[j]: int(table[i, j])
                                                               for i in range(3) for j in range(3)},
                        'improved_pair_count': sum(int(table[i, j]) for i in range(3) for j in range(i + 1, 3)),
                        'harmed_pair_count': sum(int(table[i, j]) for i in range(3) for j in range(i)),
                        'gained_AUC_credit': gain, 'lost_AUC_credit': harm, 'net_AUC_credit': gain - harm,
                        'net_AUC_change_within_cohort': (gain - harm) / total if total else None,
                        'net_AUC_contribution_full_population': (gain - harm) / all_pair_count}
    cohort_count = int(coverage.sum())
    attribution = {'pairs': cohort_count, 'candidate': candidate_name,
                   'ordered_pair_identity_sha256': cohort['ordered_pair_identity_sha256'].item().decode('ascii'),
                   'strict_coverage_acquired_count': int(coverage[1].sum()),
                   'weak_coverage_any_tie_or_win_count': weak_coverage_count,
                   'strict_coverage_acquired_and_pool_win': int(coverage[1, 2]),
                   'strict_coverage_acquired_but_pool_tie': int(coverage[1, 1]),
                   'strict_coverage_acquired_but_pool_loss': int(coverage[1, 0]),
                   'strict_coverage_acquired_not_fully_served': int(coverage[1, :2].sum()),
                   'no_strict_coverage_pool_loss': int(coverage[0, 0]),
                   'no_strict_coverage_pool_tie': int(coverage[0, 1]),
                   'no_strict_coverage_pool_win': int(coverage[0, 2]),
                   'strict_member_win_count_histogram': list(map(int, win_histogram)),
                   'member_tie_count_histogram': list(map(int, tie_histogram)),
                   'interpretation': 'O had no strict member wins; new coverage is pair-ranking acquisition, not causal graph-evidence proof. Pool ties retain half ROC credit.'}
    for field in ('strict_coverage_acquired_count', 'strict_coverage_acquired_and_pool_win',
                  'strict_coverage_acquired_but_pool_tie', 'strict_coverage_acquired_but_pool_loss',
                  'strict_coverage_acquired_not_fully_served'):
        attribution[field + '_rate_within_O_cohort'] = attribution[field] / cohort_count if cohort_count else None
    result['O_common_inversion_attribution'] = attribution
    result['member_pair_order_comparison'] = {
        'matching_member_slots_available': matched,
        'matched_initialization_intervention_slots': candidate_name in ('O', 'I', 'P', 'G'),
        'members': [{'member': m, 'changed_pair_state_count': int(t.sum() - np.trace(t)),
                     'strict_loss_to_win': int(t[0, 2]), 'strict_loss_to_tie': int(t[0, 1]),
                     'strict_win_to_loss': int(t[2, 0]), 'strict_win_to_tie': int(t[2, 1]),
                     'tie_to_win': int(t[1, 2]), 'tie_to_loss': int(t[1, 0])}
                    for m, t in enumerate(member_tables)] if matched else None,
        'all_same_slot_full_pair_orders_preserved': all(int(t.sum() - np.trace(t)) == 0 for t in member_tables) if matched else None,
        'interpretation': 'If every corresponding member pair order is preserved while pooled ranks change, rank-preserving score transformations (including relative positive scales) suffice; this does not identify acquired member ordering. Changed rankings are not proof of a graph-specific mechanism.'}
    return result


def analyze(phase, output, collection, baselines):
    import numpy as np
    cells, raws, errors = {}, {}, []
    for row in collection:
        key = (row['condition'], row['seed'])
        cell = dict(row)
        if row['status'] == 'collected':
            try:
                raw = load_raw(bound(phase, row['raw']), np)
                if len(raw['member_logits']) != (1 if row['condition'] == 'single' else 4):
                    raise ValueError('Member count differs from frozen condition')
                cell['competence'] = competence(raw, np)
                raws[key] = raw
            except Exception as error:
                cell['status'], cell['error'] = 'analysis_failed', type(error).__name__ + ': ' + str(error)
                errors.append({'condition': key[0], 'seed': key[1], 'error': cell['error']})
        cells[key] = cell
    seeds = []
    for seed in SEEDS:
        row = {'seed': seed, 'rank_repair_available': False, 'frozen_O_candidate_comparisons': {}}
        o = cells['O', seed]
        if o['status'] == 'collected' and baselines[str(seed)]['available']:
            try:
                base = raws['O', seed]
                if baselines[str(seed)]['O_prediction'] != o['raw']:
                    raise ValueError('Frozen cohort was not bound to this exact O prediction event')
                with np.load(bound(phase, baselines[str(seed)]['raw']), allow_pickle=False) as archive:
                    cohort = {key: archive[key].copy() for key in archive.files}
                validate_cohort(base, cohort, np)
                for condition in CONDITIONS:
                    candidate = cells[condition, seed]
                    if candidate['status'] != 'collected':
                        row['frozen_O_candidate_comparisons'][condition] = {'available': False, 'reason': candidate['status']}
                        continue
                    try:
                        comparison = transitions(base, raws[condition, seed], cohort, np, condition)
                        row['frozen_O_candidate_comparisons'][condition] = {'available': True, 'cohorts': comparison}
                    except Exception as error:
                        message = type(error).__name__ + ': ' + str(error)
                        row['frozen_O_candidate_comparisons'][condition] = {'available': False, 'error': message}
                        # Malformed paired identity/state evidence must not remain a usable contrast cell.
                        cells[condition, seed]['status'] = 'analysis_failed'
                        cells[condition, seed]['error'] = message
                        errors.append({'condition': condition, 'seed': seed, 'error': message})
                i = cells['I', seed]
                comparison = row['frozen_O_candidate_comparisons']['I']
                if comparison['available']:
                    row.update(rank_repair_available=True, cohorts=comparison['cohorts'])
                    row['full_population_net_AUC_repair'] = row['cohorts']['all_pairs']['net_AUC_change_within_cohort']
                    row['member_AUC_changes_I_minus_O'] = [b - a for a, b in zip(o['competence']['member_pair_AUC'], i['competence']['member_pair_AUC'])]
                    row['member_BCE_changes_I_minus_O'] = [b - a for a, b in zip(o['competence']['member_BCE'], i['competence']['member_BCE'])]
                    row['mean_raw_logit_BCE_change_I_minus_O'] = i['competence']['served_mean_raw_logit_BCE'] - o['competence']['served_mean_raw_logit_BCE']
            except Exception as error:
                row['error'] = type(error).__name__ + ': ' + str(error)
                errors.append({'seed': seed, 'error': row['error']})
        else:
            row['reason'] = 'Complete O predictions and O-only frozen population are required; no substitution'
        seeds.append(row)
    contrasts, diagnostic_contrasts, member_contrasts, attribution_contrasts = {}, {}, {}, {}
    for left, right in CONTRASTS:
        name = left + '_minus_' + right
        contrasts[name] = paired([cells[left, s]['source_served_AUC'] - cells[right, s]['source_served_AUC']
                                  if cells[left, s]['status'] == cells[right, s]['status'] == 'collected' else None for s in SEEDS])
        diagnostic_contrasts[name] = {}
        member_contrasts[name] = {}
        for metric in ('member_AUC_mean', 'member_AUC_min', 'member_AUC_max'):
            values = [cells[left, s]['competence'][metric] - cells[right, s]['competence'][metric]
                      if cells[left, s]['status'] == cells[right, s]['status'] == 'collected' else None for s in SEEDS]
            member_contrasts[name][metric] = {**paired(values), 'favorable_direction': 'higher'}
        for metric, direction in DIAGNOSTIC_DIRECTIONS.items():
            values = [cells[left, s]['competence']['probabilistic_diagnostics']['served'][metric] -
                      cells[right, s]['competence']['probabilistic_diagnostics']['served'][metric]
                      if cells[left, s]['status'] == cells[right, s]['status'] == 'collected' else None for s in SEEDS]
            diagnostic_contrasts[name][metric] = {**paired(values), 'favorable_direction': direction,
                                                  'contrast_sign': 'left minus right; negative favors left for lower-is-better metrics'}
            if direction in ('higher', 'lower'):
                for statistic in ('mean', 'worst'):
                    key = statistic if statistic == 'mean' else ('min' if direction == 'higher' else 'max')
                    member_values = [cells[left, s]['competence']['probabilistic_diagnostics']['member_summary'][metric][key] -
                                     cells[right, s]['competence']['probabilistic_diagnostics']['member_summary'][metric][key]
                                     if cells[left, s]['status'] == cells[right, s]['status'] == 'collected' else None for s in SEEDS]
                    member_contrasts[name][metric + '_' + statistic] = {**paired(member_values), 'favorable_direction': direction}
        attribution_contrasts[name] = {}
        for metric in ('strict_coverage_acquired_count_rate_within_O_cohort',
                       'strict_coverage_acquired_and_pool_win_rate_within_O_cohort',
                       'strict_coverage_acquired_but_pool_tie_rate_within_O_cohort',
                       'strict_coverage_acquired_but_pool_loss_rate_within_O_cohort'):
            values = []
            for row in seeds:
                a, b = (row['frozen_O_candidate_comparisons'].get(c, {}) for c in (left, right))
                va = a['cohorts']['O_common_inversion_attribution'][metric] if a.get('available') else None
                vb = b['cohorts']['O_common_inversion_attribution'][metric] if b.get('available') else None
                values.append(va - vb if va is not None and vb is not None else None)
            attribution_contrasts[name][metric] = paired(values)
    complete18 = (all(row['status'] == 'collected' for row in cells.values()) and
                  all(row['rank_repair_available'] and all(x.get('available') for x in row['frozen_O_candidate_comparisons'].values()) for row in seeds) and not errors)
    practical = [contrasts['I_minus_independent4'], contrasts['I_minus_single']]
    decision = ('Unavailable: entire three-seed paired practical comparison required' if not complete18 or not all(x['available'] for x in practical)
                else 'This frozen recipe does not demonstrate practical superiority' if any(x['mean'] <= 0 for x in practical)
                else 'Positive exploratory development contrasts; separately frozen independent evidence required')
    result = {'schema': 'internal-be-molhiv-rank-readout-v3', 'cells': list(cells.values()), 'seeds': seeds,
              'paired_source_served_AUC': contrasts, 'paired_served_PR_probability_calibration': diagnostic_contrasts,
              'paired_member_competence': member_contrasts,
              'paired_O_common_inversion_attribution_rates': attribution_contrasts,
              'paired_full_population_net_AUC_repair': paired([row.get('full_population_net_AUC_repair') for row in seeds]),
              'errors': errors, 'complete18_readout': complete18, 'candidate_fixed_before_outcomes': 'I',
              'selection_optimism': True, 'independent_TEST_evidence': False, 'frozen_practical_decision': decision,
              'rescue_tuning_or_best_policy_substitution': False,
              'claim': 'exploratory selected scaffold-development evidence; three paired optimizer seeds, no molecule/pair/member iid inference',
              'pool_semantics': 'source float32 mean raw logits; stable sigmoid only for proper-score/calibration summaries; no probability averaging or calibration fit',
              'diagnostic_limits': ['PR depends on prevalence; AP and trapezoidal area use different interpolation',
                                    'ECE depends on fixed bins and is descriptive; no independent calibration evidence',
                                    'member pair-order acquisition is not causal graph evidence; all-unchanged orders permit score-scale explanations',
                                    'common-inversion coverage is a pair oracle; pool ties earn half ROC credit',
                                    'mean-logit BCE Jensen gap is not a standalone benefit or competence guarantee',
                                    'extra diagnostic contrasts are exploratory; no recipe changes or PR/calibration-based acceptance substitution'],
              'pair_processing': {'sampled': False, 'max_temporary_block_pairs': MAX_BLOCK_PAIRS,
                                  'max_pairs_per_seed': 4229192, 'identity_schema': PAIR_ID_SCHEMA,
                                  'O_identity_fixed_before_non_O_serving': True, 'credit': {'win': 1, 'tie': .5, 'loss': 0},
                                  'additional_inference_calls': 0}}
    write(output / 'compact' / 'ANALYSIS.json', result)
    return result
