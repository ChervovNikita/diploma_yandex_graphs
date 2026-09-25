"""Post hoc member and decision analysis for Roman Empire controls.

All summaries use saved held-out predictions after training has finished. Nothing
in this file changes checkpoints, training, validation selection, or test labels.
For every multiclass node, mean member CE minus pooled-logit CE equals the
mean KL from the pooled softmax to each member softmax; the class label cancels
from the difference. Rescue/harm pairs count member-node decisions corrected
or broken by the pooled prediction.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / 'experiments_iclr' / 'results'
INPUT = RESULT_ROOT / 'projector_controls.csv'
OUT_RUNS = RESULT_ROOT / 'decision_analysis.csv'
OUT_K = RESULT_ROOT / 'decision_by_k.csv'
OUT_SUMMARY = RESULT_ROOT / 'decision_summary.csv'


def softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp(x):
    m = x.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))).squeeze(-1)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with INPUT.open(newline='') as f:
        runs = list(csv.DictReader(f))
    summaries = []
    by_k = []
    for run in runs:
        if not run['prediction_file']:
            continue
        pred_file = ROOT / run['prediction_file']
        if not pred_file.is_file():
            raise FileNotFoundError(f'CSV-selected prediction file is missing: {pred_file}')
        with np.load(pred_file) as data:
            y = data['y_true']
            member_pred = data['member_pred']
            ensemble_pred = data['ensemble_pred']
            if 'member_logits' not in data:
                raise KeyError(f'Selected prediction file has no member_logits: {pred_file}')
            logits = data['member_logits']
        m, n = member_pred.shape
        member_correct = member_pred == y[None, :]
        ensemble_correct = ensemble_pred == y
        k_correct = member_correct.sum(axis=0)
        mean_logits = logits.mean(axis=0)
        member_lse = logsumexp(logits)
        pooled_lse = logsumexp(mean_logits)
        member_ce = member_lse - logits[:, np.arange(n), y]
        pooled_ce = pooled_lse - mean_logits[np.arange(n), y]
        ce_gap = member_ce.mean(axis=0) - pooled_ce
        pooled_logprob = mean_logits - pooled_lse[:, None]
        member_logprob = logits - member_lse[:, :, None]
        pooled_prob = np.exp(pooled_logprob)
        reverse_kl = np.sum(
            pooled_prob[None, :, :] *
            (pooled_logprob[None, :, :] - member_logprob), axis=-1
        ).mean(axis=0)
        identity_error = float(np.max(np.abs(ce_gap - reverse_kl)))
        if identity_error > 1e-5:
            raise AssertionError(f'CE gap / reverse-KL identity failed for '
                                 f'{run["variant"]} split {run["split"]}: '
                                 f'{identity_error}')
        rescue_pairs = int(np.where(ensemble_correct, m - k_correct, 0).sum())
        harm_pairs = int(np.where(~ensemble_correct, k_correct, 0).sum())
        gain_from_pairs_pp = 100 * (rescue_pairs - harm_pairs) / (m * n)
        if abs(gain_from_pairs_pp - 100 * (
                ensemble_correct.mean() - member_correct.mean())) > 1e-8:
            raise AssertionError('Pair-count pooling-gain identity failed')
        mean_prob = softmax(logits).mean(axis=0)
        probability_pred = mean_prob.argmax(axis=-1)
        pairs = list(combinations(range(m), 2))
        double_fault = np.mean([((~member_correct[i]) & (~member_correct[j])).mean()
                                for i, j in pairs]) if pairs else np.nan
        identifier = {key: run[key] for key in ('dataset', 'model', 'variant', 'split')}
        summaries.append({
            **identifier,
            'n_test': n,
            'm': m,
            'mean_member_acc': member_correct.mean(),
            'logit_ensemble_acc': ensemble_correct.mean(),
            'mean_probability_acc': (probability_pred == y).mean(),
            'logit_vs_probability_decisions': (probability_pred != ensemble_pred).sum(),
            'ensemble_gain_vs_mean_member_pp': 100 * (ensemble_correct.mean() - member_correct.mean()),
            'mean_member_ce_nats': float(member_ce.mean()),
            'pooled_ce_nats': float(pooled_ce.mean()),
            'mean_ce_gap_nats': float(ce_gap.mean()),
            'mean_reverse_kl_nats': float(reverse_kl.mean()),
            'ce_kl_identity_max_abs_nats': identity_error,
            'double_fault': double_fault,
            'pool_member_rescue_pairs': rescue_pairs,
            'pool_member_harm_pairs': harm_pairs,
            'pool_net_correct_pairs': rescue_pairs - harm_pairs,
            'pool_gain_from_pairs_pp': gain_from_pairs_pp,
            'k0_ensemble_rescues': int(((k_correct == 0) & ensemble_correct).sum()),
            'pool_correct_with_some_member_wrong_nodes': int(((k_correct < m) & ensemble_correct).sum()),
            'pool_wrong_with_some_member_correct_nodes': int(((k_correct > 0) & ~ensemble_correct).sum()),
            'all_members_correct_ensemble_harms': int(((k_correct == m) & ~ensemble_correct).sum()),
        })
        for k in range(m + 1):
            sel = k_correct == k
            by_k.append({
                **identifier,
                'm': m,
                'k_correct_members': k,
                'nodes': int(sel.sum()),
                'ensemble_correct_nodes': int((sel & ensemble_correct).sum()),
                'ensemble_wrong_nodes': int((sel & ~ensemble_correct).sum()),
            })
    write_csv(OUT_RUNS, summaries)
    write_csv(OUT_K, by_k)
    groups = defaultdict(list)
    for row in summaries:
        groups[(row['dataset'], row['model'], row['variant'])].append(row)
    aggregate = []
    for (dataset, model, variant), runs in sorted(groups.items()):
        runs = sorted(runs, key=lambda row: int(row['split']))
        def mean(name):
            return float(np.mean([float(row[name]) for row in runs]))
        aggregate.append({
            'dataset': dataset, 'model': model, 'variant': variant,
            'n_splits': len(runs),
            'splits': ' '.join(row['split'] for row in runs),
            'mean_reverse_kl_nats': mean('mean_reverse_kl_nats'),
            'mean_member_ce_nats': mean('mean_member_ce_nats'),
            'mean_pooled_ce_nats': mean('pooled_ce_nats'),
            'mean_pooling_gain_pp': mean('ensemble_gain_vs_mean_member_pp'),
            'mean_rescue_pairs_per_split': mean('pool_member_rescue_pairs'),
            'mean_harm_pairs_per_split': mean('pool_member_harm_pairs'),
            'mean_k0_rescues_per_split': mean('k0_ensemble_rescues'),
            'mean_wrong_with_some_correct_member_nodes_per_split':
                mean('pool_wrong_with_some_member_correct_nodes'),
            'mean_logit_probability_decision_changes_per_split':
                mean('logit_vs_probability_decisions'),
        })
    write_csv(OUT_SUMMARY, aggregate)
    print(f'Analyzed {len(summaries)} runs and wrote {OUT_RUNS.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
