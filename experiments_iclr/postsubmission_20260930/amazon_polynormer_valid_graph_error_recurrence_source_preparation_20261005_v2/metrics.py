"""Frozen descriptive VALID-to-VALID recurrence; no fitting, replay or propagation."""
import csv
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch

C = 5
EVENTS = ('shared_pool_wrong', 'shared_unanimous_wrong', 'shared_wrong_member_fraction',
          'independent_pool_wrong', 'independent_unanimous_wrong',
          'independent_wrong_member_fraction', 'single_wrong')
CONTEXTS = ('raw_adjacency', 'raw_all_nonneighbors', 'matched_adjacency', 'matched_control')
REASONS = ('no_signature_nonneighbor', 'no_confidence_candidate', 'controls_exhausted')
STRATA = ('both_correct', 'shared_loss_not_unanimous', 'shared_loss_unanimous',
          'shared_recovery_not_unanimous', 'shared_recovery_unanimous') + tuple(
    'both_wrong_%s_S%d_I%d' % (relation, su, iu)
    for relation in ('same', 'different') for su in (0, 1) for iu in (0, 1))
PRIMARY = STRATA.index('both_wrong_same_S1_I1')
COLUMNS = ('split', 'scope', 'stratum', 'true_class', 'competitor', 'event', 'context',
           'weighting', 'metric', 'target_population_n', 'support_targets', 'numerator',
           'denominator', 'value', 'status')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


class Table:
    def __init__(self):
        self.rows = []

    def add(self, *, numerator, denominator, **dims):
        value = float(numerator / denominator) if denominator else None
        require(value is None or math.isfinite(value), 'Nonfinite descriptive ratio')
        row = {k: None for k in COLUMNS}
        row.update(dims, numerator=float(numerator), denominator=int(denominator),
                   value=value, status='descriptive_defined' if denominator else 'undefined_empty')
        self.rows.append(row)


def native(z):
    require(z.device.type == 'cpu' and z.dtype == torch.float32 and
            tuple(z.shape) == (4, 6123, C), 'Complete VALID FP32 bank required')
    with torch.no_grad():
        p = torch.softmax(z, dim=-1)
        pred = p.mean(0).argmax(-1).numpy().copy()
        member = z.argmax(-1).numpy().T.copy()
        confidence = p.max(-1).values.mean(0).numpy().astype(np.float64)
    require(np.isfinite(confidence).all() and np.all((confidence >= 0) & (confidence <= 1)),
            'Invalid FP32 confidence')
    return {'pred': pred, 'member': member, 'confidence': confidence}


def classify(shared, independent, y):
    se, ie = shared['pred'] != y, independent['pred'] != y
    # FP32 softmax rounding can create a pooled argmax tie absent in raw logits.
    # Target flags are unanimity at the native pool's wrong competitor; raw events
    # remain separately retained in events_for and precision counts below.
    su = se & np.all(shared['member'] == shared['pred'][:, None], axis=1)
    iu = ie & np.all(independent['member'] == independent['pred'][:, None], axis=1)
    s = np.zeros(len(y), dtype=np.int64)
    loss, recovery, both = se & ~ie, ~se & ie, se & ie
    s[loss] = 1 + su[loss].astype(np.int64)
    s[recovery] = 3 + iu[recovery].astype(np.int64)
    same = shared['pred'] == independent['pred']
    s[both] = 5 + 4 * (~same[both]).astype(np.int64) + 2 * su[both] + iu[both]
    ks = np.sum(shared['member'] == y[:, None], axis=1)
    ki = np.sum(independent['member'] == y[:, None], axis=1)
    k0 = (independent['member'][:, 0] == y).astype(np.int64)
    return s, ks, ki, k0


def events_for(shared, independent, y):
    values = np.zeros((len(y), C, len(EVENTS)), dtype=np.float64)
    for c in range(C):
        for bank, offset in ((shared, 0), (independent, 3)):
            count = np.sum(bank['member'] == c, axis=1)
            values[:, c, offset] = bank['pred'] == c
            values[:, c, offset + 1] = count == 4
            values[:, c, offset + 2] = count / 4.0
        values[:, c, 6] = independent['member'][:, 0] == c
    values[np.arange(len(y)), y, :] = 0.0
    return values


def stratum_counts(shared, independent, y, s, ks, ki, k0):
    n = len(y)
    by_class = np.zeros((len(STRATA), C), dtype=np.int64)
    joint = np.zeros((len(STRATA), C, 5, 5, 2), dtype=np.int64)
    wrong_count = np.zeros((len(STRATA), C, C, 2, 5), dtype=np.int64)
    np.add.at(by_class, (s, y), 1)
    np.add.at(joint, (s, y, ks, ki, k0), 1)
    pool, member, precision = {}, {}, {}
    for name, bank, f in (('shared4', shared, 0), ('independent4', independent, 1)):
        raw_unanimous = np.all(bank['member'] == bank['member'][:, :1], axis=1)
        discrepancy = raw_unanimous & (bank['pred'] != bank['member'][:, 0])
        precision[name] = {
            'raw_unanimous_pool_class_discrepancy_count': int(discrepancy.sum()),
            'raw_unanimous_wrong_pool_correct_count': int(np.sum(raw_unanimous & (bank['member'][:, 0] != y) & (bank['pred'] == y))),
            'discrepancy_raw_member_class_vs_pool_class_confusion': np.bincount(
                bank['member'][discrepancy, 0]*C + bank['pred'][discrepancy], minlength=C*C).reshape(C, C).tolist()}
        pool[name] = np.bincount(y * C + bank['pred'], minlength=C*C).reshape(C, C).tolist()
        member[name] = [np.bincount(y * C + bank['member'][:, m], minlength=C*C).reshape(C, C).tolist()
                        for m in range(4)]
        for c in range(C):
            mask = y != c
            count = np.sum(bank['member'] == c, axis=1)
            np.add.at(wrong_count, (s[mask], y[mask], c, f, count[mask]), 1)
    single = independent['member'][:, 0]
    pool['single'] = np.bincount(y*C + single, minlength=C*C).reshape(C, C).tolist()
    require(int(by_class.sum()) == int(joint.sum()) == n and
            int(wrong_count.sum()) == 2*(C-1)*n, 'Complete stratum accounting failure')
    return {'strata': list(STRATA), 'true_classes': list(range(C)),
            'pool_stratum_by_true_class': by_class.tolist(), 'pool_confusions': pool,
            'individual_member_confusions': member, 'precision_discrepancies': precision,
            'erroneous_unanimity_flag': 'all raw members equal native pool wrong competitor',
            'correct_member_joint_axes': ['pool_stratum', 'true_class', 'shared_k', 'independent_k', 'single_correct'],
            'correct_member_joint_counts': joint.tolist(),
            'wrong_competitor_member_histogram_axes': ['pool_stratum', 'true_class', 'competitor', 'bank_shared_then_independent', 'members_predicting_competitor'],
            'wrong_competitor_member_histogram_counts': wrong_count.tolist(),
            'correct_class_competitor_cells': 'not applicable; structural zero, not a wrong-event estimate'}


def same_class_neighbors(edge, ids, y, public_n):
    lookup = np.full(public_n, -1, dtype=np.int64)
    lookup[ids] = np.arange(len(ids))
    a, b = lookup[edge[0]], lookup[edge[1]]
    keep = (a >= 0) & (b >= 0) & (a != b)
    a, b = a[keep], b[keep]
    keep = y[a] == y[b]
    a, b = a[keep], b[keep]
    require(len(np.unique(a*len(ids)+b)) == len(a), 'Duplicate VALID directed adjacency')
    order = np.argsort(a, kind='stable')
    a, b = a[order], b[order]
    starts = np.concatenate(([0], np.cumsum(np.bincount(a, minlength=len(ids)))))
    return [b[starts[v]:starts[v+1]] for v in range(len(ids))]


def matching_inputs(ids, y, degree, ks, ki, k0, salt):
    band = np.floor(np.log2(1+degree)).astype(np.int64)
    signatures = [(int(y[u]), int(band[u]), int(ks[u]), int(ki[u]), int(k0[u])) for u in range(len(ids))]
    groups = {}
    for u, key in enumerate(signatures):
        groups.setdefault(key, []).append(u)
    groups = {key: np.asarray(nodes, dtype=np.int64) for key, nodes in groups.items()}
    order = sorted(range(len(ids)), key=lambda u: hashlib.sha256((salt+'|'+str(int(ids[u]))).encode()).digest())
    rank = np.empty(len(ids), dtype=np.int64)
    rank[np.asarray(order)] = np.arange(len(ids))
    return signatures, groups, rank


def match(v, neighbors, signatures, groups, rank, qs, qi, caliper):
    blocked = np.zeros(len(qs), dtype=bool)
    blocked[v] = True
    blocked[neighbors] = True
    used = np.zeros(len(qs), dtype=bool)
    mn, mc, unmatched = [], [], []
    for u in neighbors[np.argsort(rank[neighbors], kind='stable')]:
        candidate = groups[signatures[int(u)]]
        candidate = candidate[~blocked[candidate]]
        if not len(candidate):
            unmatched.append((int(u), 0))
            continue
        ds, di = np.abs(qs[candidate]-qs[u]), np.abs(qi[candidate]-qi[u])
        keep = (ds <= caliper) & (di <= caliper)
        candidate, distance = candidate[keep], (ds+di)[keep]
        if not len(candidate):
            unmatched.append((int(u), 1))
            continue
        keep = ~used[candidate]
        candidate, distance = candidate[keep], distance[keep]
        if not len(candidate):
            unmatched.append((int(u), 2))
            continue
        a = int(candidate[np.lexsort((rank[candidate], distance))[0]])
        used[a] = True
        mn.append(int(u))
        mc.append(a)
    require(len(mn)+len(unmatched) == len(neighbors) and len(set(mc)) == len(mc),
            'Matching partition/reuse failure')
    return np.asarray(mn, dtype=np.int64), np.asarray(mc, dtype=np.int64), unmatched, blocked


def split_panel(table, edge, ids, y, degree, shared, independent, split, config):
    n, scopes, ne = len(y), 1+len(STRATA), len(EVENTS)
    s, ks, ki, k0 = classify(shared, independent, y)
    events = events_for(shared, independent, y)
    neighbors = same_class_neighbors(edge, ids, y, 24492)
    signatures, groups, rank = matching_inputs(ids, y, degree, ks, ki, k0, config['order_salt'])
    qs, qi = shared['confidence'], independent['confidence']
    classes = [np.flatnonzero(y == c) for c in range(C)]
    class_sums = np.stack([events[x].sum(0) for x in classes])
    num = np.zeros((4, scopes, C, C, ne))
    den = np.zeros((4, scopes, C, C), dtype=np.int64)
    targets = np.zeros_like(den)
    rate_sum = np.zeros_like(num)
    diff_sum = np.zeros((scopes, C, C, ne))
    diff_n = np.zeros((scopes, C, C), dtype=np.int64)
    population = np.zeros((scopes, C), dtype=np.int64)
    support = np.zeros((5, scopes, C), dtype=np.int64)  # targets eligible/matched, eligible/matched/unmatched edges
    reasons = np.zeros((3, scopes, C), dtype=np.int64)
    conf_sum = np.zeros((2, scopes, C))
    conf_max = np.zeros_like(conf_sum)
    uses = np.zeros(n, dtype=np.int64)
    neighbor_used = np.zeros(n, dtype=bool)
    signature_support = {}
    pn, pd, pt, pr = np.zeros((4, ne)), np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), np.zeros((4, ne))
    p_diff, p_match_targets, p_population = np.zeros(ne), 0, 0
    p_unique = np.zeros((4, n), dtype=bool)
    p_wrong_unique = np.zeros((4, ne, n), dtype=bool)
    for v in range(n):
        yv, si = int(y[v]), int(s[v])+1
        ns = neighbors[v]
        mn, mc, unmatched, blocked = match(v, ns, signatures, groups, rank, qs, qi, config['confidence_caliper_each_bank'])
        counts = (len(ns), len(classes[yv])-1-len(ns), len(mn), len(mc))
        require(counts[1] >= 0, 'Raw nonneighbor denominator invalid')
        sums = (events[ns].sum(0), class_sums[yv]-events[v]-events[ns].sum(0),
                events[mn].sum(0), events[mc].sum(0))
        require(np.min(sums[1]) >= -1e-10, 'Raw nonneighbor numerator invalid')
        valid_c = np.arange(C) != yv
        ds, di = np.abs(qs[mn]-qs[mc]), np.abs(qi[mn]-qi[mc])
        for scope in (0, si):
            population[scope, yv] += 1
            support[:, scope, yv] += (int(bool(len(ns))), int(bool(len(mn))), len(ns), len(mn), len(unmatched))
            for _, reason in unmatched:
                reasons[reason, scope, yv] += 1
            conf_sum[:, scope, yv] += (float(ds.sum()), float(di.sum()))
            if len(mn):
                conf_max[:, scope, yv] = np.maximum(conf_max[:, scope, yv], (ds.max(), di.max()))
            for context, (count, total) in enumerate(zip(counts, sums)):
                num[context, scope, yv, valid_c, :] += total[valid_c]
                den[context, scope, yv, valid_c] += count
                if count:
                    targets[context, scope, yv, valid_c] += 1
                    rate_sum[context, scope, yv, valid_c, :] += total[valid_c]/count
            if len(mn):
                diff_sum[scope, yv, valid_c, :] += (sums[2][valid_c]-sums[3][valid_c])/len(mn)
                diff_n[scope, yv, valid_c] += 1
        for u in ns:
            key = (int(s[v]),)+signatures[int(u)]
            signature_support.setdefault(key, np.zeros(5, dtype=np.int64))[0] += 1
        for u in mn:
            key = (int(s[v]),)+signatures[int(u)]
            signature_support[key][1] += 1
        for u, reason in unmatched:
            key = (int(s[v]),)+signatures[u]
            signature_support[key][2+reason] += 1
        np.add.at(uses, mc, 1)
        neighbor_used[mn] = True
        if s[v] == PRIMARY:
            p_population += 1
            c = int(shared['pred'][v])
            require(c == independent['pred'][v] and c != yv and ks[v] == ki[v] == 0,
                    'Primary target definition failure')
            nonneighbors = classes[yv][~blocked[classes[yv]]]
            for context, (count, total, anchors) in enumerate(zip(counts, sums, (ns, nonneighbors, mn, mc))):
                pn[context] += total[c]
                pd[context] += count
                if count:
                    pt[context] += 1
                    pr[context] += total[c]/count
                p_unique[context, anchors] = True
                for event in range(ne):
                    p_wrong_unique[context, event, anchors[events[anchors, c, event] > 0]] = True
            if len(mn):
                p_diff += (sums[2][c]-sums[3][c])/len(mn)
                p_match_targets += 1
    require(int(support[2, 0].sum()) == int(support[3, 0].sum()+support[4, 0].sum()) and
            int(reasons[:, 0].sum()) == int(support[4, 0].sum()), 'Complete unmatched accounting failure')
    for scope in range(scopes):
        stratum = 'all' if scope == 0 else STRATA[scope-1]
        for yv in range(C):
            for c in range(C):
                if c == yv:
                    continue
                dims = dict(split=split, scope='all' if scope == 0 else 'pool_stratum', stratum=stratum,
                            true_class=yv, competitor=c, target_population_n=int(population[scope, yv]))
                for e, event in enumerate(EVENTS):
                    for context, name in enumerate(CONTEXTS):
                        for weight, numerator, denominator in (
                            ('directed_pair', num[context, scope, yv, c, e], den[context, scope, yv, c]),
                            ('equal_target', rate_sum[context, scope, yv, c, e], targets[context, scope, yv, c])):
                            table.add(**dims, event=event, context=name, weighting=weight, metric='wrong_competitor_recurrence',
                                      support_targets=int(targets[context, scope, yv, c]), numerator=numerator, denominator=denominator)
                    table.add(**dims, event=event, context='matched_adjacency_minus_control', weighting='equal_target',
                              metric='matched_excess', support_targets=int(diff_n[scope, yv, c]),
                              numerator=diff_sum[scope, yv, c, e], denominator=diff_n[scope, yv, c])
    primary_dims = dict(split=split, scope='primary', stratum=STRATA[PRIMARY], true_class='all',
                        competitor='native_common_wrong', target_population_n=p_population)
    for e, event in enumerate(EVENTS):
        for context, name in enumerate(CONTEXTS):
            for weight, numerator, denominator in (('directed_pair', pn[context, e], pd[context]),
                                                   ('equal_target', pr[context, e], pt[context])):
                table.add(**primary_dims, event=event, context=name, weighting=weight, metric='wrong_competitor_recurrence',
                          support_targets=int(pt[context]), numerator=numerator, denominator=denominator)
        table.add(**primary_dims, event=event, context='matched_adjacency_minus_control', weighting='equal_target',
                  metric='matched_excess', support_targets=p_match_targets, numerator=p_diff[e], denominator=p_match_targets)
    delta_sum = float(p_diff[1]-p_diff[4])
    table.add(**primary_dims, event='shared_minus_independent_unanimous_wrong', context='matched_adjacency_minus_control',
              weighting='equal_target', metric='primary_Delta', support_targets=p_match_targets,
              numerator=delta_sum, denominator=p_match_targets)
    signature_rows = []
    for key, value in sorted(signature_support.items()):
        require(int(value[0]) == int(value[1]+value[2:].sum()), 'Signature support partition failure')
        signature_rows.append({'target_stratum': STRATA[key[0]], 'true_class': key[1], 'anchor_degree_band': key[2],
            'anchor_shared_k': key[3], 'anchor_independent_k': key[4], 'anchor_single_correct': key[5],
            'eligible_edges': int(value[0]), 'matched_edges': int(value[1]),
            'unmatched_by_reason': dict(zip(REASONS, (int(x) for x in value[2:])))})
    support_result = {'split': split, 'scope_order': ['all']+list(STRATA), 'true_classes': list(range(C)),
        'target_population': population.tolist(),
        'support_axes': ['targets_with_neighbors', 'targets_with_matches', 'eligible_edges', 'matched_edges', 'unmatched_edges'],
        'support_by_scope_and_true_class': support.tolist(), 'unmatched_reason_order': list(REASONS),
        'unmatched_by_reason_scope_class': reasons.tolist(), 'confidence_absolute_distance_sum_shared_then_independent': conf_sum.tolist(),
        'confidence_absolute_distance_mean_shared_then_independent': [[[float(conf_sum[f, scope, c]/support[3, scope, c])
            if support[3, scope, c] else None for c in range(C)] for scope in range(scopes)] for f in range(2)],
        'confidence_absolute_distance_max_shared_then_independent': [[[float(conf_max[f, scope, c])
            if support[3, scope, c] else None for c in range(C)] for scope in range(scopes)] for f in range(2)],
        'confidence_distance_denominator': 'matched_edges in the same scope/class; empty means undefined',
        'unique_matched_neighbor_anchors': int(neighbor_used.sum()), 'unique_matched_controls': int(np.sum(uses > 0)),
        'maximum_control_reuse_across_targets': int(uses.max()) if n else 0,
        'signature_support': signature_rows,
        'primary': {'all_targets': p_population, 'fraction_of_complete_VALID': p_population/n,
            'targets_by_context': dict(zip(CONTEXTS, (int(x) for x in pt))),
            'pairs_by_context': dict(zip(CONTEXTS, (int(x) for x in pd))),
            'unique_anchors_by_context': dict(zip(CONTEXTS, (int(x) for x in p_unique.sum(1)))),
            'distinct_wrong_event_anchors_by_context': {
                context: dict(zip(EVENTS, (int(x) for x in p_wrong_unique[j].sum(1)))) for j, context in enumerate(CONTEXTS)},
            'matched_targets': p_match_targets, 'Delta': delta_sum/p_match_targets if p_match_targets else None,
            'status': 'descriptive_defined' if p_match_targets else 'undefined_empty',
            'adequate_inferential_or_mechanism_support_claim': False}}
    return stratum_counts(shared, independent, y, s, ks, ki, k0), support_result


def run(state, output, custody, config):
    output = Path(output)
    require(config['schema'] == 'amazon-valid-valid-graph-error-diagnostic-v2' and config['splits'] == [0, 1, 2] and
            config['classes'] == list(range(C)) and config['VALID_count_per_split'] == 6123 and
            config['confidence_caliper_each_bank'] == .05 and
            config['order_salt'] == 'common-error-valid-valid-20261005-v2' and
            config['fits'] == 0 and config['FIT_projection_required'] is False and config['execution_admitted'] is False,
            'Exact frozen diagnostic required; root admission is external to settings')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    edge, val_mask = custody.load_visible(state)
    nonself = edge[0] != edge[1]
    degree = np.bincount(edge[0, nonself], minlength=custody.N).astype(np.int64)
    require(int(degree.sum()) == int(nonself.sum()), 'Public degree accounting failure')
    write_json(output/'ENVIRONMENT.json', {'python': sys.version, 'platform': platform.platform(),
        'numpy': np.__version__, 'torch': str(torch.__version__), 'CPU_threads': 1,
        'base_model_execution': False, 'source_import_after_custody': True})
    write_json(output/'INPUTS.json', {'closure': state['closure'], 'protocol': state['protocol'],
        'payload_descriptors': state['payloads'],
        'decode_scope': ['canonical public edge_index', 'official val_mask', 'compact three VALID label packs', '15 selected-logit artifacts'],
        'logit_array_scope': 'Unchanged authenticated V2 loader decodes full saved logits; slice to VALID before all diagnostic arithmetic.',
        'FIT_TRAIN_control_TEST_labels_or_features_decoded': False, 'checkpoint_open_or_replay': False})
    table, strata, supports, identities = Table(), [], [], []
    for split, _, _ in custody.SEEDS:
        ids, y = custody.load_validation(state, val_mask, split)
        panels = {}
        for family_name, label in ((custody.FAMILIES[0], 'shared4'), (custody.FAMILIES[1], 'independent4')):
            found = [f for f in state['registry']['families'] if f['split'] == split and f['family'] == family_name]
            require(len(found) == 1, 'Exactly one complete original family required')
            z, provenance = custody.load_bank(state, found[0])
            z_valid = z[:, ids, :].clone()
            del z
            panels[label] = native(z_valid)
            del z_valid
            identities.append({'split': split, 'classifier': label, 'VALID_count': len(ids),
                'ordered_VALID_ids_sha256': custody.sha_object(ids.tolist()), 'provenance': provenance})
        identities.append({'split': split, 'classifier': 'single', 'VALID_count': len(ids),
            'ordered_VALID_ids_sha256': custody.sha_object(ids.tolist()),
            'alias': 'independent4 member0; no additional fit/load', 'provenance': identities[-1]['provenance'][:1]})
        counts, support = split_panel(table, edge, ids, y, degree[ids], panels['shared4'], panels['independent4'], split, config)
        strata.append({'split': split, **counts})
        supports.append(support)
        with (output/'PROGRESS.jsonl').open('a') as stream:
            stream.write(json.dumps({'event': 'complete_split', 'split': split, 'VALID_count': len(ids),
                'matched_primary_targets': support['primary']['matched_targets']})+'\n')
    require(len(strata) == len(supports) == 3 and len(identities) == 9, 'Complete three-split closure missing')
    for row in state['payloads']:
        custody.verify(state['phase'], row)
    deltas = [x['primary']['Delta'] for x in supports]
    all_defined = all(x is not None for x in deltas)
    summary = {'block_count': 3, 'splitwise_Delta': deltas,
        'mean': sum(deltas)/3 if all_defined else None, 'min': min(deltas) if all_defined else None,
        'max': max(deltas) if all_defined else None,
        'status': 'descriptive_defined' if all_defined else 'undefined_missing_split', 'independent_blocks_claim': False}
    write_json(output/'STRATA.json', {'schema': 'amazon-valid-graph-error-complete-strata-v2', 'splits': strata})
    write_json(output/'SUPPORT.json', {'schema': 'amazon-valid-graph-error-support-v2', 'splits': supports,
        'three_split_primary_summary': summary, 'no_adequate_mechanism_or_power_claim': True})
    write_json(output/'AGGREGATE.json', {'schema': 'amazon-valid-valid-graph-error-recurrence-v2',
        'status': 'complete_retrospective_descriptive_VALID', 'diagnostic': config, 'identities': identities,
        'metric_rows': table.rows, 'three_split_primary_summary': summary,
        'cost': {'descriptive_wall_seconds': time.perf_counter()-started, 'custody': state.get('custody_cost'),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
            'physical_selected_logit_loads': 15, 'fits': 0, 'graph_propagations': 0, 'checkpoint_replays': 0},
        'interpretation': {'VALID_selected_and_label_reused': True, 'same_graph_overlapping_splits': True,
            'iid_inference': False, 'causal_graph_or_sharing': False, 'gradient_interference_identified': False,
            'serving_rule_or_fresh_confirmation': False, 'TRAIN_control_TEST_or_FIT_labels_used': False}})
    with (output/'AGGREGATE.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table.rows)
    write_json(output/'TERMINAL.json', {'status': 'complete', 'split_count': 3, 'native_panels': 9,
        'VALID_count_per_split': 6123, 'metric_row_count': len(table.rows),
        'no_fit_replay_or_extra_label_access': True})
