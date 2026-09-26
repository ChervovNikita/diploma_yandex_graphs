"""Verify all five fixed MC Dropout masks from retained class decisions."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / 'mc_dropout_results'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

parser = argparse.ArgumentParser()
parser.add_argument('--public-npz', type=Path)
args = parser.parse_args()
manifest = read(RESULTS / 'compact_derivation.json')
replay = read(RESULTS / 'independent_checkpoint_replay.json')
full_audit = read(RESULTS / 'full_array_audit.json')
freeze = read(HERE / 'mc_dropout_replay_freeze.json')
summary = read(RESULTS / 'summary.json')
probability_freeze = read(HERE / 'MC_PROBABILITY_POOL_ANALYSIS_FREEZE.json')
assert sha(HERE / 'MC_PROBABILITY_POOL_ANALYSIS_FREEZE.json') == manifest['probability_analysis_freeze_sha256']
for name, digest in probability_freeze['source_sha256'].items():
    assert sha(HERE / name) == digest
assert manifest['status'] == 'COMPLETE_FIVE_MASK_DECISION_DERIVATION'
assert replay['status'] == 'COMPLETE_FIVE_MASK_CUDA_REPLAY_PASS'
assert full_audit['status'] == 'COMPLETE_FIVE_MASK_SOURCE_LABEL_LOGIT_ARITHMETIC_PASS'
assert sha(RESULTS / 'independent_checkpoint_replay.json') == manifest['checkpoint_replay_sha256']
assert sha(RESULTS / 'full_array_audit.json') == manifest['full_array_audit_sha256']
assert sha(HERE / 'mc_dropout_replay_freeze.json') == replay['replay_freeze_sha256']
assert sha(HERE / 'audit_mc_dropout_checkpoint.py') == freeze['auditor_sha256']
assert sha(HERE / 'audit_mc_dropout_arrays.py') == full_audit['audit_source_sha256']
assert freeze['logit_max_abs_tolerance'] == 1e-4
assert freeze['require_identical_member_and_pooled_classes'] is True
omitted = set()
for name, digest in freeze['input_sha256'].items():
    path = ROOT / name
    if path.exists():
        assert sha(path) == digest, name
    else:
        assert name == 'data/roman_empire.npz' or (
            name.startswith('experiments_iclr/mc_dropout_results/split') and
            name.endswith('_predictions.npz')) or (
            name.startswith('experiments_iclr/results/predictions/roman-empire_SAGE_base_split')
            and name.endswith('.npz'))
        omitted.add(name)
assert sha(RESULTS / 'summary.json') == full_audit['summary_sha256']
assert len(manifest['records']) == len(replay['records']) == len(full_audit['records']) == 5
assert {r['split'] for r in manifest['records']} == set(range(5))
assert {r['split'] for r in replay['records']} == set(range(5))
assert {r['split'] for r in full_audit['records']} == set(range(5))
anchor = RESULTS / 'official_test_anchor.npz'
assert sha(anchor) == manifest['anchor_sha256']
assert manifest['public_npz_sha256'] == full_audit['public_npz_sha256'] == freeze['input_sha256']['data/roman_empire.npz']
with np.load(anchor, allow_pickle=False) as data:
    labels, masks = data['node_labels'].copy(), data['test_masks'].copy()
assert labels.shape == (22662,) and masks.shape == (5, 22662)
if args.public_npz:
    assert sha(args.public_npz) == manifest['public_npz_sha256']
    with np.load(args.public_npz, allow_pickle=False) as data:
        assert np.array_equal(labels, data['node_labels'])
        assert np.array_equal(masks, data['test_masks'][:5])
scores = []
for split in range(5):
    derived = next(r for r in manifest['records'] if r['split'] == split)
    replay_row = next(r for r in replay['records'] if r['split'] == split)
    audited = next(r for r in full_audit['records'] if r['split'] == split)
    record_path = RESULTS / f'split{split}.json'
    record = read(record_path)
    assert record['split'] == split
    assert sha(record_path) == derived['record_sha256'] == replay_row['record_sha256'] == audited['record_sha256']
    assert derived['original_mc_predictions_sha256'] == record['prediction_sha256'] == replay_row['prediction_sha256'] == audited['predictions_sha256']
    assert record['prediction_sha256'] == probability_freeze['prediction_sha256'][str(split)]
    assert derived['original_base_predictions_sha256'] == replay_row['baseline_prediction_sha256']
    spec = record['input_manifest']
    assert record['input_fingerprint'] == hashlib.sha256(json.dumps(
        spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for field, name in {'script_sha256': 'experiments_iclr/mc_dropout_control.py',
                        'models_sha256': 'models.py', 'datasets_sha256': 'datasets.py',
                        'control_runner_sha256': 'experiments_iclr/projector_controls.py'}.items():
        assert spec[field] == sha(ROOT / name)
    assert spec['data_sha256'] == manifest['public_npz_sha256']
    assert spec['checkpoint_sha256'] == replay_row['checkpoint_sha256'] == audited['checkpoint_sha256']
    assert replay_row['member_decision_mismatches'] == replay_row['pooled_decision_mismatches'] == 0
    assert 0 <= replay_row['max_abs_member_logit_error'] <= 1e-4
    assert replay_row['baseline_decision_mismatches'] == 0
    assert 0 <= replay_row['baseline_max_abs_logit_error'] <= 1e-4
    assert derived['original_base_predictions_sha256'] == freeze['input_sha256'][spec['row']['prediction_file']]
    assert spec['protocol']['passes'] == 4 and spec['protocol']['dropout_probability'] == .2
    assert spec['protocol']['splits'] == list(range(5))
    assert spec['protocol']['pooling'] == 'arithmetic mean of raw logits, then argmax'
    assert int(spec['row']['num_params']) == 6730770
    assert int(spec['row']['split']) == split and int(spec['row']['seed']) == split
    path = RESULTS / f'split{split}_decisions.npz'
    assert sha(path) == derived['compact_sha256']
    with np.load(path, allow_pickle=False) as data:
        ids, y = data['node_index'], data['y_true']
        assert np.array_equal(ids, np.flatnonzero(masks[split]))
        assert np.array_equal(y, labels[ids]) and len(y) == 5666
        assert data['member_pred'].shape == (4, 5666)
        assert data['base_pred'].shape == data['ensemble_pred'].shape == data['probability_pool_pred'].shape == (5666,)
        for key in ('member_pred', 'base_pred', 'ensemble_pred', 'probability_pool_pred'):
            assert np.issubdtype(data[key].dtype, np.integer)
            assert np.all((data[key] >= 0) & (data[key] < 18))
        base = float(np.mean(data['base_pred'] == y))
        mc = float(np.mean(data['ensemble_pred'] == y))
        probability = float(np.mean(data['probability_pool_pred'] == y))
        assert abs(probability - derived['probability_pool_accuracy']) < 1e-12
    assert abs(base - record['deterministic_base_test_metric']) < 1e-7
    assert abs(base - audited['base_accuracy']) < 1e-7
    assert abs(mc - record['mc_test_acc']) < 1e-7
    assert abs(mc - audited['mc_accuracy']) < 1e-7
    assert abs(mc - replay_row['replayed_accuracy']) < 1e-7
    assert abs(100 * (mc - base) - record['delta_vs_same_checkpoint_pp']) < 1e-5
    scores.append({'split': split, 'base_accuracy': base, 'mc_accuracy': mc,
                   'probability_pool_accuracy': probability,
                   'difference_pp': 100 * (mc - base)})
base_mean = 100 * float(np.mean([r['base_accuracy'] for r in scores]))
mc_mean = 100 * float(np.mean([r['mc_accuracy'] for r in scores]))
assert abs(base_mean - summary['mean_base_test_acc_percent']) < 1e-5
assert abs(mc_mean - summary['mean_mc_test_acc_percent']) < 1e-5
assert abs(mc_mean - base_mean - summary['mean_paired_delta_pp']) < 1e-5
print(json.dumps({'status': 'COMPLETE_FIVE_MASK_COMPACT_ACCURACY_PASS',
                  'base_mean_percent': base_mean, 'mc_mean_percent': mc_mean,
                  'difference_pp': mc_mean - base_mean, 'records': scores,
                  'scope': 'Retained hard-class accuracy and provenance checks. Raw pooling, cross-entropy and CUDA checkpoint replay require omitted original artifacts.'}, indent=2))
