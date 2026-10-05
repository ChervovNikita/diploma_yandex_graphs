"""Descriptive selected-VALID arithmetic only. Never fits, exports hidden states or solves an oracle."""
import csv
import itertools
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
BINS = 15
TOL = 1e-10
DEGREE_BINS = ('0', '1-2', '3-5', '6-10', '11-20', '>20')
COLUMNS = ('split', 'classifier', 'scope', 'true_class', 'probability_class', 'degree_bin',
           'member', 'pair_left', 'pair_right', 'bin', 'metric', 'support_n',
           'numerator', 'denominator', 'value', 'status', 'unit')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


class Table:
    def __init__(self):
        self.rows = []

    def add(self, metric, numerator, denominator, support, *, value=None,
            status=None, unit='fraction', **dimensions):
        row = {key: None for key in COLUMNS}
        row.update(dimensions)
        if status is None:
            status = 'defined' if denominator is not None and denominator > 0 else 'undefined_empty'
        if value is None and status == 'defined' and numerator is not None:
            value = float(numerator / denominator)
        require(value is None or math.isfinite(float(value)), 'Nonfinite aggregate')
        row.update(metric=metric, support_n=int(support), numerator=numerator,
                   denominator=denominator, value=value, status=status, unit=unit)
        self.rows.append(row)


def native_arrays(z):
    """Exact V2 native arithmetic on already-sliced VALID FP32 logits."""
    require(z.dtype == torch.float32 and z.device.type == 'cpu' and
            z.ndim == 3 and z.shape[2] == C and z.shape[0] in (1, 4), 'VALID logit schema')
    with torch.no_grad():
        p32 = torch.softmax(z, dim=-1)
        q32 = p32.mean(0)
        logp = torch.log_softmax(z.to(torch.float64), dim=-1)
        logq = torch.logsumexp(logp, dim=0) - math.log(z.shape[0])
        p64 = logp.exp()
        p = p64.numpy().transpose(1, 0, 2)
        q = p64.mean(0).numpy().copy()
        member_pred = z.argmax(-1).numpy().T.copy()
        pred = (z[0].argmax(-1) if z.shape[0] == 1 else q32.argmax(-1)).numpy().copy()
        q32 = q32.numpy().copy()
    require(np.isfinite(p).all() and np.isfinite(q).all() and
            np.max(np.abs(p.sum(2)-1)) <= TOL, 'Probability arithmetic invalid')
    return {'p': p, 'q': q, 'logq': logq.numpy().copy(), 'q32': q32,
            'pred': pred, 'member_pred': member_pred,
            'precision_argmax_discrepancies': int(np.sum(pred != q.argmax(1))),
            'single_raw_vs_FP32_probability_argmax_discrepancies':
                int(np.sum(pred != q32.argmax(1))) if z.shape[0] == 1 else 0}


def degree_masks(degree):
    masks = (degree == 0, (degree >= 1) & (degree <= 2),
             (degree >= 3) & (degree <= 5), (degree >= 6) & (degree <= 10),
             (degree >= 11) & (degree <= 20), degree > 20)
    require(np.all(np.sum(np.stack(masks), axis=0) == 1), 'Public degree bins do not partition VALID')
    return tuple(zip(DEGREE_BINS, masks))


def conditions(y, degree):
    yield {'scope': 'all', 'true_class': None, 'degree_bin': 'all'}, np.ones(len(y), dtype=bool)
    for c in range(C):
        yield {'scope': 'true_class', 'true_class': c, 'degree_bin': 'all'}, y == c
    for name, mask in degree_masks(degree):
        yield {'scope': 'degree', 'true_class': None, 'degree_bin': name}, mask


def describe(table, values, prefix, dims):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    table.add(prefix+'_mean', float(values.sum()), n, n, **dims)
    if n:
        center = values-values.mean()
        ss = float(np.dot(center, center))
        table.add(prefix+'_population_SD', ss, n, n, value=math.sqrt(ss/n), unit='score', **dims)
    else:
        table.add(prefix+'_population_SD', None, 0, 0, unit='score', **dims)
    for name, quantile in (('min', 0.0), ('q25', .25), ('median', .5), ('q75', .75), ('max', 1.0)):
        table.add(prefix+'_'+name, None, n, n,
                  value=float(np.quantile(values, quantile, method='linear')) if n else None,
                  unit='score', **dims)


def confidence_bins(table, confidence, correct, dims):
    n = len(confidence)
    idx = np.minimum(np.floor(confidence*BINS).astype(np.int64), BINS-1)
    require(not n or (idx.min() >= 0 and idx.max() < BINS), 'Confidence bin outside [0,1]')
    ece_sum = 0.0
    for b in range(BINS):
        mask = idx == b
        k = int(mask.sum())
        successes = int(correct[mask].sum())
        conf_sum = float(confidence[mask].sum())
        bd = {**dims, 'bin': b}
        table.add('confidence_bin_mass', k, n, n, **bd)
        table.add('confidence_bin_accuracy', successes, k, k, **bd)
        table.add('confidence_bin_mean_probability', conf_sum, k, k, **bd)
        table.add('confidence_bin_abs_gap', abs(successes-conf_sum), k, k, **bd)
        ece_sum += abs(successes-conf_sum)
    table.add('fixed_15bin_ECE', ece_sum, n, n, **dims)


def confusion(table, pred, y, dims):
    n = len(y)
    counts = np.bincount(y*C+pred, minlength=C*C).reshape(C, C)
    require(int(counts.sum()) == n, 'Confusion population lost')
    for c in range(C):
        support = int(counts[c].sum())
        predicted = int(counts[:, c].sum())
        table.add('class_recall_accuracy', int(counts[c, c]), support, support,
                  **{**dims, 'true_class': c})
        table.add('class_precision', int(counts[c, c]), predicted, predicted,
                  **{**dims, 'probability_class': c})
        for d in range(C):
            count = int(counts[c, d])
            cd = {**dims, 'true_class': c, 'probability_class': d}
            table.add('confusion_count', count, n, n, value=float(count), unit='count', **cd)
            table.add('confusion_row_fraction', count, support, support, **cd)


def correlation(table, a, b, prefix, dims):
    n = len(a)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if n:
        ac, bc = aa-aa.mean(), bb-bb.mean()
        a2, b2, ab = float(np.dot(ac, ac)), float(np.dot(bc, bc)), float(np.dot(ac, bc))
    else:
        a2 = b2 = ab = 0.0
    table.add(prefix+'_left_mean', float(aa.sum()), n, n, **dims)
    table.add(prefix+'_right_mean', float(bb.sum()), n, n, **dims)
    table.add(prefix+'_left_population_variance', a2, n, n, **dims)
    table.add(prefix+'_right_population_variance', b2, n, n, **dims)
    table.add(prefix+'_population_covariance', ab, n, n, **dims)
    den = math.sqrt(a2*b2)
    require(not den or abs(ab/den) <= 1+TOL, 'Correlation arithmetic failure')
    table.add(prefix+'_Pearson', ab, den, n,
              value=ab/den if den else None,
              status='defined' if den else ('undefined_empty' if not n else 'undefined_zero_variance'),
              unit='correlation', **dims)


def pair_panel(table, left, right, y, degree, base):
    lc, rc = left['pred'] == y, right['pred'] == y
    for condition, mask in conditions(y, degree):
        dims = {**base, **condition}
        n = int(mask.sum())
        le, re = ~lc[mask], ~rc[mask]
        both_wrong = int(np.sum(le & re))
        only_left_correct = int(np.sum(~le & re))
        only_right_correct = int(np.sum(le & ~re))
        for metric, count in (('both_correct', int(np.sum(~le & ~re))),
                              ('left_only_correct', only_left_correct),
                              ('right_only_correct', only_right_correct),
                              ('double_fault_both_wrong', both_wrong)):
            table.add(metric+'_count', count, n, n, value=float(count), unit='count', **dims)
            table.add(metric+'_fraction', count, n, n, **dims)
        table.add('left_minus_right_accuracy', only_left_correct-only_right_correct, n, n, **dims)
        table.add('left_recovers_right_error', only_left_correct, int(re.sum()), n, **dims)
        table.add('left_loses_right_correct', only_right_correct, int((~re).sum()), n, **dims)
        table.add('error_overlap_Jaccard', both_wrong, int(np.sum(le | re)), n, **dims)
        table.add('left_error_given_right_error', both_wrong, int(re.sum()), n, **dims)
        table.add('right_error_given_left_error', both_wrong, int(le.sum()), n, **dims)
        wrong_agreement = int(np.sum((le & re) & (left['pred'][mask] == right['pred'][mask])))
        table.add('same_wrong_class_given_double_fault', wrong_agreement, both_wrong, n, **dims)
        correlation(table, le, re, 'error_indicator', dims)
        # Conditional probability/residual correlations are frozen by true class.
        # Degree has an error-correlation panel above, without another probability grid.
        if condition['scope'] != 'degree':
            for c in range(C):
                cd = {**dims, 'probability_class': c}
                lp, rp = left['q'][mask, c], right['q'][mask, c]
                target = (y[mask] == c).astype(np.float64)
                correlation(table, lp, rp, 'class_probability', cd)
                correlation(table, lp-target, rp-target, 'class_error_residual', cd)


def group_panel(table, arrays, y, degree, split, name):
    q, p, pred = arrays['q'], arrays['p'], arrays['pred']
    correct = pred == y
    target = np.eye(C, dtype=np.float64)[y]
    brier = np.sum((q-target)**2, axis=1)
    nll = -arrays['logq'][np.arange(len(y)), y]
    conf64 = q[np.arange(len(y)), pred]
    conf32 = arrays['q32'][np.arange(len(y)), pred].astype(np.float64)
    competitors32 = arrays['q32'].copy()
    competitors32[np.arange(len(y)), pred] = -np.inf
    top_margin32 = conf32-competitors32.max(1)
    competitors64 = q.copy()
    competitors64[np.arange(len(y)), y] = -np.inf
    true_margin = q[np.arange(len(y)), y]-competitors64.max(1)
    base = {'split': split, 'classifier': name}
    k_correct = np.sum(arrays['member_pred'] == y[:, None], axis=1)
    for condition, mask in conditions(y, degree):
        dims = {**base, **condition}
        n = int(mask.sum())
        table.add('accuracy', int(correct[mask].sum()), n, n, **dims)
        table.add('error_fraction', int((~correct[mask]).sum()), n, n, **dims)
        member_successes = int(k_correct[mask].sum())
        nm = n*p.shape[1]
        table.add('mean_member_raw_logit_accuracy', member_successes, nm, n, **dims)
        table.add('pooling_gain_over_mean_member_accuracy', int(correct[mask].sum())*p.shape[1]-member_successes, nm, n, **dims)
        table.add('Brier', float(brier[mask].sum()), n, n, unit='Brier', **dims)
        table.add('NLL', float(nll[mask].sum()), n, n, unit='nats', **dims)
        confidence_bins(table, conf64[mask], correct[mask], dims)
        errors = mask & ~correct
        for metric, values in (('erroneous_confidence_FP32', conf32),
                               ('erroneous_confidence_FP64', conf64),
                               ('erroneous_top_vs_runner_up_margin_FP32', top_margin32),
                               ('erroneous_true_vs_best_other_margin_FP64', true_margin)):
            describe(table, values[errors], metric, dims)
        for k in range(p.shape[1]+1):
            kmask = mask & (k_correct == k)
            kn = int(kmask.sum())
            kd = {**dims, 'bin': k}
            table.add('member_correct_histogram_count', kn, n, n, value=float(kn), unit='count', **kd)
            table.add('member_correct_histogram_fraction', kn, n, n, **kd)
            table.add('ensemble_accuracy_given_member_correct_count', int(correct[kmask].sum()), kn, kn, **kd)
        for m in range(p.shape[1]):
            mc = arrays['member_pred'][:, m] == y
            md = {**dims, 'member': m}
            table.add('member_raw_logit_accuracy', int(mc[mask].sum()), n, n, **md)
            table.add('ensemble_minus_member_accuracy', int(correct[mask].sum())-int(mc[mask].sum()), n, n, **md)
            table.add('ensemble_recovers_member_error', int(np.sum(mask & correct & ~mc)), int(np.sum(mask & ~mc)), n, **md)
            table.add('ensemble_loses_member_correct', int(np.sum(mask & ~correct & mc)), int(np.sum(mask & mc)), n, **md)
        if condition['scope'] in ('all', 'degree'):
            confusion(table, pred[mask], y[mask], dims)
    all_dims = {**base, 'scope': 'all', 'degree_bin': 'all'}
    member_brier_sum = float(np.sum((p-target[:, None, :])**2))
    ambiguity_sum = float(np.sum((p-q[:, None, :])**2))
    nm = len(y)*p.shape[1]
    residual = member_brier_sum/nm-float(brier.mean())-ambiguity_sum/nm
    require(abs(residual) <= TOL, 'Probability-pool Brier ambiguity failure')
    table.add('mean_member_Brier', member_brier_sum, nm, len(y), unit='Brier', **all_dims)
    table.add('probability_pool_Brier_ambiguity', ambiguity_sum, nm, len(y), unit='Brier', **all_dims)
    table.add('Brier_ambiguity_identity_residual', None, len(y), len(y), value=residual, unit='Brier', **all_dims)
    for m, k in itertools.combinations(range(p.shape[1]), 2):
        left = {'pred': arrays['member_pred'][:, m], 'q': p[:, m]}
        right = {'pred': arrays['member_pred'][:, k], 'q': p[:, k]}
        pair_panel(table, left, right, y, degree,
                   {**base, 'pair_left': 'member%d' % m, 'pair_right': 'member%d' % k})
    return {'split': split, 'classifier': name, 'VALID_count': len(y),
            'members': p.shape[1], 'correct_count': int(correct.sum()),
            'accuracy': float(correct.mean()), 'Brier': float(brier.mean()), 'NLL': float(nll.mean()),
            'precision_argmax_discrepancies': arrays['precision_argmax_discrepancies'],
            'single_raw_vs_FP32_probability_argmax_discrepancies':
                arrays['single_raw_vs_FP32_probability_argmax_discrepancies']}



def shared_independent_decomposition(table, shared, independent, y, degree, split):
    """Descriptive exact arithmetic decomposition, with no sharing-causality inference."""
    require(shared['p'].shape[1] == independent['p'].shape[1] == 4, 'Paired four-member banks required')
    target = np.eye(C, dtype=np.float64)[y]
    for condition, mask in conditions(y, degree):
        dims = {'split': split, 'classifier': 'shared_minus_independent', **condition}
        n = int(mask.sum())
        sc = int(np.sum(shared['pred'][mask] == y[mask]))
        ic = int(np.sum(independent['pred'][mask] == y[mask]))
        sm = int(np.sum(shared['member_pred'][mask] == y[mask, None]))
        im = int(np.sum(independent['member_pred'][mask] == y[mask, None]))
        table.add('native_accuracy_difference', sc-ic, n, n, **dims)
        table.add('mean_member_accuracy_difference', sm-im, 4*n, n, **dims)
        table.add('pooling_gain_difference', 4*(sc-ic)-(sm-im), 4*n, n, **dims)
        accuracy_residual = ((sc-ic)/n-(sm-im)/(4*n)-(4*(sc-ic)-(sm-im))/(4*n)) if n else None
        require(accuracy_residual is None or abs(accuracy_residual) <= TOL, 'Paired accuracy decomposition failure')
        table.add('accuracy_decomposition_identity_residual', None, n, n, value=accuracy_residual, unit='score', **dims)
        if n:
            sb = float(np.sum((shared['q'][mask]-target[mask])**2))
            ib = float(np.sum((independent['q'][mask]-target[mask])**2))
            smb = float(np.sum((shared['p'][mask]-target[mask, None, :])**2))/4
            imb = float(np.sum((independent['p'][mask]-target[mask, None, :])**2))/4
            sa = float(np.sum((shared['p'][mask]-shared['q'][mask, None, :])**2))/4
            ia = float(np.sum((independent['p'][mask]-independent['q'][mask, None, :])**2))/4
            residual = (sb-ib)-((smb-imb)-(sa-ia))
            require(abs(residual)/n <= TOL, 'Paired Brier ambiguity identity failure')
            for metric, numerator in (('pooled_Brier_difference', sb-ib),
                    ('mean_member_Brier_difference', smb-imb),
                    ('Brier_ambiguity_difference', sa-ia),
                    ('Brier_decomposition_identity_residual', residual)):
                table.add(metric, numerator, n, n, unit='Brier', **dims)
        else:
            for metric in ('pooled_Brier_difference', 'mean_member_Brier_difference',
                           'Brier_ambiguity_difference', 'Brier_decomposition_identity_residual'):
                table.add(metric, None, 0, 0, unit='Brier', **dims)

def run(state, output, custody):
    output = Path(output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    table = Table()
    summaries, identities = [], []
    edge, val_mask = custody.load_visible(state)
    nonself = edge[0] != edge[1]
    public_degree = np.bincount(edge[0, nonself], minlength=custody.N).astype(np.int64)
    require(int(public_degree.sum()) == int(nonself.sum()), 'Public degree accounting failure')
    write_json(output/'ENVIRONMENT.json', {'python': sys.version, 'platform': platform.platform(),
              'numpy': np.__version__, 'torch': str(torch.__version__), 'CPU_threads': 1,
              'base_model_execution': False, 'source_import_after_custody': True})
    write_json(output/'INPUTS.json', {'closure': state['closure'], 'protocol': state['protocol'],
              'payload_descriptors': state['payloads'],
              'decode_scope': ['canonical public edge_index', 'official val_mask',
                               'compact three selected VALID label packs', '15 selected-logit artifacts'],
              'logit_array_scope': 'Full saved tensors are decoded by authenticated V2 loader; slice to VALID before all probability/statistical arithmetic.',
              'TRAIN_control_TEST_labels_or_features_decoded': False,
              'checkpoint_open_or_replay': False})
    for split, _, _ in custody.SEEDS:
        ids, y = custody.load_validation(state, val_mask, split)
        degree = public_degree[ids]
        panels = {}
        independent_alias = None
        for family_name, label in ((custody.FAMILIES[0], 'shared4'), (custody.FAMILIES[1], 'independent4')):
            found = [f for f in state['registry']['families'] if f['split'] == split and f['family'] == family_name]
            require(len(found) == 1, 'Exactly one complete original family required')
            z, provenance = custody.load_bank(state, found[0])
            z_valid = z[:, ids, :].clone()
            require(tuple(z_valid.shape) == (4, 6123, C), 'Complete selected VALID bank missing')
            if label == 'independent4':
                independent_alias = z_valid[:1].clone()
                alias_provenance = provenance[:1]
            del z
            arrays = native_arrays(z_valid)
            del z_valid
            panels[label] = arrays
            summaries.append(group_panel(table, arrays, y, degree, split, label))
            identities.append({'split': split, 'classifier': label, 'VALID_count': len(ids),
                               'ordered_VALID_ids_sha256': custody.sha_object(ids.tolist()),
                               'provenance': provenance})
        require(independent_alias is not None, 'Native member0 alias missing')
        arrays = native_arrays(independent_alias)
        del independent_alias
        panels['single'] = arrays
        summaries.append(group_panel(table, arrays, y, degree, split, 'single'))
        identities.append({'split': split, 'classifier': 'single', 'VALID_count': len(ids),
                           'ordered_VALID_ids_sha256': custody.sha_object(ids.tolist()),
                           'provenance': alias_provenance, 'alias': 'independent4 member0; no extra fit/load'})
        shared_independent_decomposition(table, panels['shared4'], panels['independent4'], y, degree, split)
        for left, right in (('shared4', 'independent4'), ('shared4', 'single'), ('independent4', 'single')):
            pair_panel(table, panels[left], panels[right], y, degree,
                       {'split': split, 'classifier': 'paired_native', 'pair_left': left, 'pair_right': right})
        with (output/'PROGRESS.jsonl').open('a') as stream:
            stream.write(json.dumps({'event': 'complete_split', 'split': split,
                                     'VALID_count_per_classifier': 6123, 'classifier_count': 3})+'\n')
    require(len(summaries) == 9 and all(s['VALID_count'] == 6123 for s in summaries), 'Complete 3-by-3 closure missing')
    for name in ('single', 'shared4', 'independent4'):
        values = [s for s in summaries if s['classifier'] == name]
        for metric in ('accuracy', 'Brier', 'NLL'):
            x = [s[metric] for s in values]
            dims = {'split': 'three_split_summary', 'classifier': name, 'scope': 'split_blocks', 'degree_bin': 'all'}
            table.add(metric+'_three_split_mean', float(sum(x)), 3, 3,
                      unit='Brier' if metric == 'Brier' else ('nats' if metric == 'NLL' else 'fraction'), **dims)
            table.add(metric+'_three_split_min', None, 3, 3, value=float(min(x)), unit='score', **dims)
            table.add(metric+'_three_split_max', None, 3, 3, value=float(max(x)), unit='score', **dims)
    for row in state['payloads']:
        custody.verify(state['phase'], row)
    result = {'schema': 'amazon-selected-VALID-native-error-analysis-v1',
              'status': 'complete_retrospective_descriptive_VALID', 'identities': identities,
              'headline_split_metrics': summaries, 'metric_rows': table.rows,
              'definitions': {'native_single_and_member_class': 'raw FP32 logit argmax, smallest class tie',
                'native_four_member_class': 'mean FP32 softmax probabilities then argmax',
                'probability_metrics': 'stable FP64 member probabilities/arithmetic pool and native stable log-mixture NLL',
                'public_degree': 'unique canonical undirected non-self neighbors; no prediction mask',
                'degree_bins': list(DEGREE_BINS), 'confidence_bins': '15 equal bins, [k/15,(k+1)/15), last closed at 1',
                'correlations': 'population descriptive Pearson; unconditional and conditional true-class panels, null at zero variance/empty support',
                'Brier': 'class-summed mean squared probability error; range 0-2',
                'Brier_decomposition': 'mean member Brier minus mean member squared probability deviation from pool equals pooled FP64 Brier; identity residual reported',
                'member_correct_count': 'count of raw-logit-correct members; neither an attainable-hull indicator nor a served oracle',
                'paired_left_only_correct': 'left recovers a right-native mistake',
                'paired_right_only_correct': 'left loses a right-native correct case',
                'support_n_vs_denominator': 'support_n is the contributing node count; denominator may be conditional error/correct count, class support, N*M, or Pearson scale; both are explicit'},
              'interpretation': {'base_VALID_selected': True, 'descriptive_retrospective_only': True,
                'node_or_split_iid_inference': False, 'causal_sharing_or_information_absence_inferred': False,
                'hull_oracle_executed': False, 'fit_refit_or_promotion': False},
              'cost': {'description_wall_seconds': time.perf_counter()-started,
                'custody': state.get('custody_cost'), 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                'physical_selected_logit_loads': 15, 'probability_calculations': 'nine selected VALID panels only',
                'hidden_exports': 0, 'fits': 0, 'LPs': 0, 'graph_propagations': 0},
              'TEST_or_TRAIN_control_labels_used': False}
    write_json(output/'AGGREGATE.json', result)
    with (output/'AGGREGATE.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table.rows)
    write_json(output/'TERMINAL.json', {'status': 'complete', 'split_count': 3, 'classifier_panels': 9,
              'VALID_count_per_panel': 6123, 'metric_row_count': len(table.rows),
              'no_scientific_fit_or_final_label_access': True})
