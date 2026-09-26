"""Read-only decision accounting for all 40 selected Roman depth-grid runs."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import audit_roman_depth_grid_40 as base

EVIDENCE = base.EVIDENCE


def exact(n: int, d: int) -> Fraction:
    if d == 0:
        raise RuntimeError('Undefined conditional success rate')
    return Fraction(int(n), int(d))


def arm_metrics(depth: int, seed: int, arm: str, part: str) -> dict:
    study = base.study_for(depth, seed)
    run = study / 'results/roman' / f'seed{seed}' / arm
    row = base.load(run / 'result.json')
    official_labels, masks = base.official_split(study)
    expected_ids = np.flatnonzero(masks['val' if part == 'valid' else 'test'])
    with np.load(run / 'selected_predictions.npz', allow_pickle=False) as raw:
        ids = raw[f'{part}_indices'].copy()
        labels = raw[f'{part}_labels'].copy()
        logits = raw[f'{part}_member_logits'].copy()
    if not np.array_equal(ids, expected_ids) or not np.array_equal(labels, official_labels[expected_ids]):
        raise RuntimeError(f'Official labels/IDs differ: {run} {part}')
    if logits.shape != (4, len(ids), 18):
        raise RuntimeError(f'Logit shape differs: {run} {part}')
    member_correct = logits.argmax(axis=2) == labels[None, :]
    k = member_correct.sum(axis=0)
    pool = logits.mean(axis=0).argmax(axis=1) == labels
    n = len(labels)
    covered = int(np.count_nonzero(k >= 1))
    uncovered = n - covered
    if not (0 < covered < n):
        raise RuntimeError(f'Degenerate conditional denominator: {run} {part}')
    utilized = int(np.count_nonzero(pool & (k >= 1)))
    rescued_empty = int(np.count_nonzero(pool & (k == 0)))
    pooled_correct = int(np.count_nonzero(pool))
    member_correct_total = int(k.sum())
    rescued_member_pairs = int(np.sum(np.where(pool, 4-k, 0)))
    harmed_member_pairs = int(np.sum(np.where(~pool, k, 0)))
    C = exact(covered, n)
    U = exact(utilized, covered)
    R = exact(rescued_empty, uncovered)
    A = exact(pooled_correct, n)
    M = exact(member_correct_total, 4*n)
    G = A - M
    if C*U + (1-C)*R != A:
        raise RuntimeError(f'C/U/R identity failed: {run} {part}')
    if G != exact(rescued_member_pairs-harmed_member_pairs, 4*n):
        raise RuntimeError(f'Rescue/harm identity failed: {run} {part}')
    reported = float(row[f'{part}_accuracy'])
    if abs(float(A)-reported) > 1e-6:
        raise RuntimeError(f'Pooled score differs: {run} {part}')
    return {
        'depth': depth, 'seed': seed, 'arm': arm, 'part': part,
        'n': n,
        'counts': {
            'covered': covered, 'uncovered': uncovered,
            'utilized': utilized, 'rescued_when_all_members_wrong': rescued_empty,
            'pooled_correct': pooled_correct,
            'member_correct_node_pairs': member_correct_total,
            'rescued_member_node_pairs': rescued_member_pairs,
            'harmed_member_node_pairs': harmed_member_pairs,
        },
        'C': float(C), 'U': float(U), 'R': float(R),
        'pooled_accuracy': float(A),
        'mean_member_accuracy': float(M),
        'pooling_gain': float(G),
        '_fractions': {'C': C, 'U': U, 'R': R, 'A': A, 'M': M, 'G': G},
        'selected_predictions_sha256': base.sha(run / 'selected_predictions.npz'),
    }


def pair_metrics(tied: dict, untied: dict) -> dict:
    t, u = tied['_fractions'], untied['_fractions']
    dC, dU, dR = t['C']-u['C'], t['U']-u['U'], t['R']-u['R']
    mid_C = (t['C']+u['C'])/2
    mid_U_minus_R = (t['U']-t['R']+u['U']-u['R'])/2
    contribution_C = mid_U_minus_R*dC
    contribution_U = mid_C*dU
    contribution_R = (1-mid_C)*dR
    delta_A = t['A']-u['A']
    if contribution_C+contribution_U+contribution_R != delta_A:
        raise RuntimeError(f'Midpoint identity failed: depth {tied["depth"]} seed {tied["seed"]}')
    delta_M = t['M']-u['M']
    delta_G = t['G']-u['G']
    if delta_M+delta_G != delta_A:
        raise RuntimeError(f'Member/pooling identity failed: depth {tied["depth"]} seed {tied["seed"]}')
    return {
        'depth': tied['depth'], 'seed': tied['seed'], 'part': tied['part'],
        'tied_minus_untied': {k: float(v) for k,v in {
            'C': dC, 'U': dU, 'R': dR, 'pooled_accuracy': delta_A,
            'mean_member_accuracy': delta_M, 'pooling_gain': delta_G,
            'coverage_contribution': contribution_C,
            'utilization_contribution': contribution_U,
            'rescue_contribution': contribution_R,
        }.items()},
    }


def main() -> None:
    audited = base.load(EVIDENCE / 'ROMAN_DEPTH_GRID_40_CELL_AUDIT.json')
    if audited['status'] != 'COMPLETE_40_CELL_READ_ONLY_AUDIT_PASS':
        raise RuntimeError('Complete 40-cell score audit is unavailable')
    arms, pairs, summary = [], [], []
    for part in ('test', 'valid'):
        for depth in (2,3,4,5):
            current = []
            for seed in range(5):
                tied = arm_metrics(depth,seed,'tied',part)
                untied = arm_metrics(depth,seed,'untied_propagation',part)
                pair = pair_metrics(tied,untied)
                arms.extend((tied,untied))
                pairs.append(pair)
                current.append(pair)
            summary.append({
                'part': part, 'depth': depth, 'pairs': 5,
                'mean_tied_minus_untied': {
                    key: float(np.mean([row['tied_minus_untied'][key] for row in current]))
                    for key in current[0]['tied_minus_untied']
                },
                'seedwise_pooled_differences': [row['tied_minus_untied']['pooled_accuracy'] for row in current],
            })
    for row in arms:
        del row['_fractions']
    result = {
        'status': 'COMPLETE_40_CELL_C_U_R_MEMBER_ACCOUNTING_PASS',
        'score_audit_sha256': base.sha(EVIDENCE / 'ROMAN_DEPTH_GRID_40_CELL_AUDIT.json'),
        'definitions': {
            'K': 'number of four member argmaxes equal to official label',
            'P': 'argmax of mean raw float32 member logits equals official label',
            'C': 'Pr(K >= 1)', 'U': 'Pr(P | K >= 1)', 'R': 'Pr(P | K = 0)',
            'identity': 'pooled_accuracy = C*U + (1-C)*R = mean_member_accuracy + pooling_gain',
            'pair_sign': 'tied minus initially matched untied',
            'midpoint_accounting': 'dA = mean(U-R)*dC + mean(C)*dU + (1-mean(C))*dR',
        },
        'arms': arms, 'pairs': pairs, 'depth_summary': summary,
        'scope': 'post hoc selected-checkpoint decisions on one Roman Empire official mask; optimizer seeds are not independent graphs',
    }
    path = EVIDENCE / 'ROMAN_DEPTH_GRID_DECISION_ACCOUNTING.json'
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('COMPLETE_40_CELL_C_U_R_MEMBER_ACCOUNTING_PASS',path)
    for row in summary:
        x=row['mean_tied_minus_untied']
        print(row['part'],row['depth'], 'dA_pp',100*x['pooled_accuracy'], 'dC_pp',100*x['C'],
              'dU_pp',100*x['U'], 'dR_pp',100*x['R'], 'dMember_pp',100*x['mean_member_accuracy'],
              'dPoolGain_pp',100*x['pooling_gain'])


if __name__ == '__main__':
    main()
