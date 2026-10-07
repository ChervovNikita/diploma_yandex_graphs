"""Bounded complete-development rank analysis; imports numerical code only on call."""
from protocol import SEEDS, bound, paired, write

NAMES = ('loss', 'tie', 'win')
MAX_BLOCK_PAIRS = 65536


def blocks(positive, negative):
    for a in range(0, len(positive), 32):
        for b in range(0, len(negative), 2048):
            yield slice(a, a + 32), slice(b, b + 2048)


def states(scores, positive, negative, a, b, np):
    p, n = scores[positive[a], None], scores[negative[b]][None, :]
    return (p > n).astype(np.uint8) * 2 + (p == n).astype(np.uint8)


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
    return raw


def freeze_baseline(raw, destination):
    """Called for O before any candidate prediction collection/inspection."""
    import numpy as np
    positive, negative = np.flatnonzero(raw['labels'] == 1), np.flatnonzero(raw['labels'] == 0)
    if not len(positive) or not len(negative):
        raise ValueError('Both finite target classes required')
    base = np.empty((len(positive), len(negative)), dtype=np.uint8)
    common = np.ones_like(base, dtype=np.bool_)
    for a, b in blocks(positive, negative):
        base[a, b] = states(raw['pool_logits'], positive, negative, a, b, np)
        for member in raw['member_logits']:
            common[a, b] &= member[positive[a], None] < member[negative[b]][None, :]
    np.savez(destination, positive=positive, negative=negative, baseline_state=base, common_inversion=common)
    return {'pairs': int(base.size), 'O_pool_states': dict(zip(NAMES, map(int, np.bincount(base.ravel(), minlength=3)))),
            'O_same_pair_strictly_inverted_by_all_members': int(common.sum()),
            'definition': 'same positive-negative pair with positive logit strictly below negative in every O member'}


def competence(raw, np):
    positive, negative = np.flatnonzero(raw['labels'] == 1), np.flatnonzero(raw['labels'] == 0)
    counts = np.zeros((len(raw['member_logits']) + 1, 3), dtype=np.int64)
    bank = [*raw['member_logits'], raw['pool_logits']]
    for a, b in blocks(positive, negative):
        for index, scores in enumerate(bank):
            counts[index] += np.bincount(states(scores, positive, negative, a, b, np).ravel(), minlength=3)
    total = len(positive) * len(negative)
    aucs = [(int(row[2]) + .5 * int(row[1])) / total for row in counts]
    losses = [float((np.logaddexp(0., scores.astype(np.float64))
                     - raw['labels'] * scores.astype(np.float64)).mean()) for scores in bank]
    return {'positive_count': len(positive), 'negative_count': len(negative), 'finite_pair_count': total,
            'member_pair_AUC': aucs[:-1], 'member_AUC_mean': sum(aucs[:-1]) / (len(aucs) - 1),
            'member_AUC_min': min(aucs[:-1]), 'member_AUC_max': max(aucs[:-1]),
            'pool_pair_AUC': aucs[-1], 'member_BCE': losses[:-1], 'served_mean_raw_logit_BCE': losses[-1],
            'member_pair_states': [dict(zip(NAMES, map(int, row))) for row in counts[:-1]],
            'pool_pair_states': dict(zip(NAMES, map(int, counts[-1])))}


def transitions(base, candidate, cohort, np):
    """All nine O->I transitions, exact strict/equality comparisons, no tolerance."""
    if set(cohort) != {'positive', 'negative', 'baseline_state', 'common_inversion'}:
        raise ValueError('Exact O-only frozen cohort fields required')
    positive, negative = cohort['positive'], cohort['negative']
    shape = (len(positive), len(negative))
    if not np.array_equal(positive, np.flatnonzero(base['labels'] == 1)) \
            or not np.array_equal(negative, np.flatnonzero(base['labels'] == 0)) \
            or cohort['baseline_state'].shape != shape or cohort['baseline_state'].dtype != np.uint8 \
            or (cohort['baseline_state'] > 2).any() or cohort['common_inversion'].shape != shape \
            or cohort['common_inversion'].dtype != np.bool_:
        raise ValueError('Frozen complete O-only pair populations/state domains differ')
    tables = {name: np.zeros((3, 3), dtype=np.int64)
              for name in ('all_pairs', 'O_pool_loss', 'O_pool_tie', 'O_pool_win', 'O_common_inversion')}
    for a, b in blocks(positive, negative):
        old = cohort['baseline_state'][a, b]
        new = states(candidate['pool_logits'], positive, negative, a, b, np)
        code = old * 3 + new
        masks = {'all_pairs': np.ones_like(old, dtype=np.bool_), 'O_pool_loss': old == 0,
                 'O_pool_tie': old == 1, 'O_pool_win': old == 2,
                 'O_common_inversion': cohort['common_inversion'][a, b]}
        for name, mask in masks.items():
            tables[name] += np.bincount(code[mask], minlength=9).reshape(3, 3)
    result = {}
    for name, table in tables.items():
        total = int(table.sum())
        gain = sum(int(table[i, j]) * (j - i) / 2 for i in range(3) for j in range(i + 1, 3))
        harm = sum(int(table[i, j]) * (i - j) / 2 for i in range(3) for j in range(i))
        result[name] = {'pairs': total, 'O_to_I': {NAMES[i] + '_to_' + NAMES[j]: int(table[i, j])
                                                 for i in range(3) for j in range(3)},
                        'improved_pair_count': sum(int(table[i, j]) for i in range(3) for j in range(i + 1, 3)),
                        'harmed_pair_count': sum(int(table[i, j]) for i in range(3) for j in range(i)),
                        'gained_AUC_credit': gain, 'lost_AUC_credit': harm, 'net_AUC_credit': gain - harm,
                        'net_AUC_change_within_cohort': (gain - harm) / total if total else None}
    return result


def analyze(phase, output, collection, baselines):
    import numpy as np
    cells, errors = {}, []
    for row in collection:
        key = (row['condition'], row['seed'])
        cell = dict(row)
        if row['status'] == 'collected':
            try:
                raw = load_raw(bound(phase, row['raw']), np)
                expected = 1 if row['condition'] == 'single' else 4
                if len(raw['member_logits']) != expected:
                    raise ValueError('Member count differs from frozen condition')
                cell['competence'] = competence(raw, np)
            except Exception as error:
                cell['status'] = 'analysis_failed'
                cell['error'] = type(error).__name__ + ': ' + str(error)
                errors.append({'condition': key[0], 'seed': key[1], 'error': cell['error']})
        cells[key] = cell
    seeds = []
    for seed in SEEDS:
        row = {'seed': seed, 'rank_repair_available': False}
        o, i = cells['O', seed], cells['I', seed]
        if o['status'] == i['status'] == 'collected' and baselines[str(seed)]['available']:
            try:
                base = load_raw(bound(phase, o['raw']), np)
                candidate = load_raw(bound(phase, i['raw']), np)
                if not np.array_equal(base['ids'], candidate['ids']) or not np.array_equal(base['labels'], candidate['labels']):
                    raise ValueError('Paired full development IDs/labels differ')
                if baselines[str(seed)]['O_prediction'] != o['raw']:
                    raise ValueError('Frozen cohort was not bound to this exact O prediction event')
                with np.load(bound(phase, baselines[str(seed)]['raw']), allow_pickle=False) as archive:
                    cohort = {key: archive[key].copy() for key in archive.files}
                row.update(rank_repair_available=True, cohorts=transitions(base, candidate, cohort, np))
                row['full_population_net_AUC_repair'] = row['cohorts']['all_pairs']['net_AUC_change_within_cohort']
                row['member_AUC_changes_I_minus_O'] = [b - a for a, b in zip(o['competence']['member_pair_AUC'], i['competence']['member_pair_AUC'])]
                row['member_BCE_changes_I_minus_O'] = [b - a for a, b in zip(o['competence']['member_BCE'], i['competence']['member_BCE'])]
                row['mean_raw_logit_BCE_change_I_minus_O'] = i['competence']['served_mean_raw_logit_BCE'] - o['competence']['served_mean_raw_logit_BCE']
            except Exception as error:
                row['error'] = type(error).__name__ + ': ' + str(error)
                errors.append({'seed': seed, 'error': row['error']})
        else:
            row['reason'] = 'Complete O/I predictions and an O-only frozen full-population cohort are required; no sampling/substitution'
        seeds.append(row)
    contrasts = {}
    for left, right in (('I', 'independent4'), ('I', 'single'), ('I', 'O'), ('I', 'P'), ('P', 'G'), ('I', 'G')):
        values = []
        for seed in SEEDS:
            a, b = cells[left, seed], cells[right, seed]
            values.append(a['source_served_AUC'] - b['source_served_AUC']
                          if a['status'] == b['status'] == 'collected' else None)
        contrasts[left + '_minus_' + right] = paired(values)
    complete18 = all(row['status'] == 'collected' for row in cells.values()) and all(row['rank_repair_available'] for row in seeds)
    practical = [contrasts['I_minus_independent4'], contrasts['I_minus_single']]
    decision = ('Unavailable: the entire three-seed paired practical comparison is required'
                if not complete18 or not all(x['available'] for x in practical) else
                'This frozen recipe does not demonstrate practical superiority'
                if any(x['mean'] <= 0 for x in practical) else
                'Positive exploratory development contrasts; a separately frozen independent-evidence protocol is required')
    result = {'schema': 'internal-be-molhiv-rank-readout-v1', 'cells': list(cells.values()), 'seeds': seeds,
              'paired_source_served_AUC': contrasts,
              'paired_full_population_net_AUC_repair': paired([row.get('full_population_net_AUC_repair') for row in seeds]),
              'errors': errors, 'complete18_readout': complete18,
              'candidate_fixed_before_outcomes': 'I', 'selection_optimism': True, 'independent_TEST_evidence': False,
              'frozen_practical_decision': decision, 'rescue_tuning_or_best_policy_substitution': False,
              'claim': 'exploratory development evidence only; n3 paired optimizer seeds; pairs/molecules/members are not model replicates',
              'pool_semantics': 'mean raw logits; mean-logit BCE; no Wiki probability responsibility',
              'pair_processing': {'sampled': False, 'max_temporary_block_pairs': MAX_BLOCK_PAIRS,
                                  'max_pairs_per_seed': 4229192, 'credit': {'win': 1, 'tie': .5, 'loss': 0}}}
    write(output / 'compact' / 'ANALYSIS.json', result)
    return result
