"""Verify the anonymous decision-level Roman fixed-mask v4 result."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / 'experiments_iclr/fixed_mask_seed_pair_v4_results'
PROTOCOL = 'roman_sage_fixed_mask0_optimization_seeds_0_1_2_v4'
OMITTED_PINNED = {'pyproject.toml', 'uv.lock', 'data/roman_empire.npz',
                  'reference/roman_dgl_processed_edge_index.npy'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    manifest_path = RESULT / 'source_manifest.json'
    manifest = read(manifest_path)
    assert manifest['protocol'] == PROTOCOL
    manifest_sha = sha(manifest_path)
    assert OMITTED_PINNED == {p for p in manifest['sha256'] if not (ROOT / p).exists()}
    for relative, digest in manifest['sha256'].items():
        if relative not in OMITTED_PINNED:
            assert sha(ROOT / relative) == digest, relative
    calibration = read(ROOT / 'reference/v4_numeric_calibration.json')
    assert calibration['repeats_per_seed_variant'] == 6 and len(calibration['rows']) == 3
    within = max(max(r['tied_repeated_forward_max_abs'], r['untied_repeated_forward_max_abs'])
                 for r in calibration['rows'])
    tol = 2 * within
    assert tol == manifest['configuration']['initial_logit_tolerance']
    assert tol == manifest['configuration']['checkpoint_replay_logit_tolerance']
    assert tol == manifest['numeric_calibration']['derived_tolerance']
    smoke = read(RESULT / 'gpu_initial_smoke.json')
    assert smoke['status'] == 'FULL_GRAPH_ALL_SEED_INITIAL_SMOKE_PASS'
    assert smoke['source_manifest_sha256'] == manifest_sha
    assert smoke['no_validation_or_test_metric_scored'] is True
    assert [r['seed'] for r in smoke['records']] == [0, 1, 2]
    assert all(r['initial_logits_max_abs_difference'] <= tol for r in smoke['records'])
    audit = read(RESULT / 'completion_audit.json')
    assert audit['protocol'] == PROTOCOL and audit['source_manifest_sha256'] == manifest_sha
    assert audit['verified_runs'] == 6 and audit['official_split'] == 0
    assert audit['optimization_seeds'] == [0, 1, 2]
    local = read(RESULT / 'local_audit.json')
    assert local['status'] == 'COMPLETE_6_ARM_LOCAL_AUDIT_PASS'
    assert local['source_manifest_sha256'] == manifest_sha
    derivation = read(ROOT / 'decision_derivation_manifest.json')
    assert derivation['protocol'] == PROTOCOL
    assert derivation['source_manifest_sha256'] == manifest_sha
    assert derivation['original_roman_npz_sha256'] == manifest['sha256']['data/roman_empire.npz']
    assert sha(ROOT / 'official_labels_mask0.npz') == derivation['official_label_anchor_sha256']
    assert len(derivation['records']) == 6
    derived = {(r['seed'], r['variant']): r for r in derivation['records']}
    assert len(derived) == 6
    with np.load(ROOT / 'official_labels_mask0.npz', allow_pickle=False) as data:
        labels = data['node_labels'].copy()
        mask = data['test_mask0'].copy()
    ids = np.flatnonzero(mask)
    assert labels.shape == (22662,) and ids.shape == (5666,)
    scores = {}
    for seed in range(3):
        initial = {}
        for arm in ('gnnm', 'untied_backbone'):
            folder = RESULT / f'seed{seed}' / arm
            row = read(folder / 'result.json')
            assert row['protocol'] == PROTOCOL and row['source_manifest_sha256'] == manifest_sha
            assert row['official_split'] == 0 and row['optimization_seed'] == seed
            assert row['variant'] == arm and row['num_layers'] == 5
            assert row['hidden_dim'] == 512 and row['m'] == 4
            record = derived[seed, arm]
            assert sha(folder / 'result.json') == record['original_result_sha256']
            assert row['artifacts_sha256']['predictions.npz'] == record['original_prediction_sha256']
            for name in ('validation_trace.csv', 'initialization.json'):
                assert sha(folder / name) == row['artifacts_sha256'][name]
            initial[arm] = read(folder / 'initialization.json')
            with (folder / 'validation_trace.csv').open(newline='') as stream:
                trace = list(csv.DictReader(stream))
            expected_steps = [1] + list(range(10, int(row['last_step']) + 1, 10))
            assert [int(r['step']) for r in trace] == expected_steps
            best, winner = -math.inf, 0
            for trace_row in trace:
                value = float(trace_row['val_metric'])
                improved = value > best
                assert int(trace_row['improved']) == int(improved)
                if improved:
                    best, winner = value, int(trace_row['step'])
            assert winner == int(row['best_step'])
            assert abs(best - float(row['best_val_metric'])) < 1e-9
            decision_path = folder / 'selected_decisions.npz'
            assert sha(decision_path) == record['derived_decision_sha256']
            with np.load(decision_path, allow_pickle=False) as pred:
                assert np.array_equal(pred['node_index'], ids)
                assert np.array_equal(pred['y_true'], labels[ids])
                member = pred['member_pred']
                decisions = pred['ensemble_pred']
                assert member.shape == (4, 5666) and decisions.shape == (5666,)
                assert np.issubdtype(member.dtype, np.integer) and np.issubdtype(decisions.dtype, np.integer)
                assert np.all((member >= 0) & (member < 18))
                assert np.all((decisions >= 0) & (decisions < 18))
                score = float(np.mean(decisions == labels[ids]))
                assert abs(score - float(row['test_acc'])) < 1e-6
                assert abs(float(np.mean(member == labels[ids][None, :])) - row['mean_member_acc']) < 1e-6
                any_correct = float(np.mean(np.any(member == labels[ids][None, :], axis=0)))
                assert abs(any_correct - row['at_least_one_member_correct_rate']) < 1e-6
            scores[seed, arm] = score
        for key in ('state_sha256', 'cpu_rng_sha256', 'device_rng_sha256'):
            assert initial['gnnm'][key] == initial['untied_backbone'][key]
    deltas = [100 * (scores[s, 'gnnm'] - scores[s, 'untied_backbone']) for s in range(3)]
    assert np.allclose(deltas, audit['paired_test_accuracy_deltas_percentage_points'], atol=1e-5, rtol=0)
    assert abs(float(np.mean(deltas)) - audit['mean_paired_delta_percentage_points']) < 1e-5
    assert abs(float(np.std(deltas, ddof=1)) - audit['sd_paired_delta_percentage_points']) < 1e-5
    print('COMPACT_6_ARM_SCORE_PASS', deltas)


if __name__ == '__main__':
    main()
