"""Independent arithmetic and public-label audit of all five MC Dropout cases."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--public-npz', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
cli = parser.parse_args()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

folder = cli.root / 'experiments_iclr/mc_dropout_results'
assert {p.name for p in folder.glob('split*.json')} == {f'split{i}.json' for i in range(5)}
assert {p.name for p in folder.glob('split*_predictions.npz')} == {f'split{i}_predictions.npz' for i in range(5)}
with np.load(cli.public_npz, allow_pickle=False) as public:
    labels = public['node_labels'].copy()
    masks = public['test_masks'].copy()
assert labels.shape == (22662,) and masks.shape == (10, 22662)
rows = []
source_files = {'script_sha256': 'experiments_iclr/mc_dropout_control.py',
                'models_sha256': 'models.py', 'datasets_sha256': 'datasets.py',
                'control_runner_sha256': 'experiments_iclr/projector_controls.py'}
for split in range(5):
    path = folder / f'split{split}.json'
    record = json.loads(path.read_text())
    spec = record['input_manifest']
    original = spec['row']
    assert record['split'] == split
    assert record['input_fingerprint'] == hashlib.sha256(json.dumps(
        spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for key, source in source_files.items():
        assert spec[key] == sha(cli.root / source)
    assert spec['data_sha256'] == sha(cli.public_npz)
    assert (original['dataset'], original['model'], original['variant']) == ('roman-empire', 'SAGE', 'base')
    for key, value in {'split': split, 'seed': split, 'num_layers': 5,
                       'hidden_dim': 512, 'm': 1, 'num_steps': 5000}.items():
        assert int(original[key]) == value
    assert float(original['lr']) == 3e-5
    assert spec['protocol']['passes'] == 4
    assert spec['protocol']['dropout_probability'] == .2
    assert spec['protocol']['splits'] == list(range(5))
    npz_path = folder / f'split{split}_predictions.npz'
    assert sha(npz_path) == record['prediction_sha256']
    with np.load(npz_path, allow_pickle=False) as data:
        ids, y = data['node_index'], data['y_true']
        logits = data['member_logits'].astype(np.float64)
        assert np.array_equal(ids, np.flatnonzero(masks[split]))
        assert np.array_equal(y, labels[ids])
        assert logits.shape == (4, 5666, 18) and np.isfinite(logits).all()
        assert np.array_equal(logits.argmax(-1), data['member_pred'])
        pool = logits.mean(0)
        pred = pool.argmax(-1)
        assert np.array_equal(pred, data['ensemble_pred'])
        shifted = pool - pool.max(1, keepdims=True)
        prob = np.exp(shifted)
        prob /= prob.sum(1, keepdims=True)
        assert np.max(np.abs(prob - data['ensemble_prob'])) < 1e-5
        acc = float(np.mean(pred == y))
        ce = float(np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y]))
    assert abs(acc - record['mc_test_acc']) < 1e-7
    assert abs(acc - record['mc_test_metric']) < 1e-7
    assert abs(ce - record['mc_test_loss']) < 1e-5
    base = float(original['test_acc'])
    assert abs(base - record['deterministic_base_test_metric']) < 1e-7
    assert abs(100 * (acc - base) - record['delta_vs_same_checkpoint_pp']) < 1e-5
    rows.append({'split': split, 'base_accuracy': base, 'mc_accuracy': acc,
                 'mc_ce': ce, 'difference_pp': 100 * (acc - base),
                 'parameters': int(original['num_params']),
                 'selected_base_step': int(original['best_step']),
                 'record_sha256': sha(path), 'predictions_sha256': sha(npz_path),
                 'checkpoint_sha256': spec['checkpoint_sha256']})
summary_path = folder / 'summary.json'
summary = json.loads(summary_path.read_text())
assert summary['splits'] == list(range(5)) and len(summary['per_split']) == 5
mean_base = float(np.mean([r['base_accuracy'] for r in rows])) * 100
mean_mc = float(np.mean([r['mc_accuracy'] for r in rows])) * 100
assert abs(mean_base - summary['mean_base_test_acc_percent']) < 1e-5
assert abs(mean_mc - summary['mean_mc_test_acc_percent']) < 1e-5
assert abs(mean_mc - mean_base - summary['mean_paired_delta_pp']) < 1e-5
report = {'status': 'COMPLETE_FIVE_MASK_SOURCE_LABEL_LOGIT_ARITHMETIC_PASS',
          'scope': 'Four fixed stochastic dropout passes per already selected BASE checkpoint. Source and saved-output verification, not fresh checkpoint replay.',
          'public_npz_sha256': sha(cli.public_npz), 'audit_source_sha256': sha(Path(__file__)),
          'summary_sha256': sha(summary_path), 'records': rows,
          'mean_base_accuracy_percent': mean_base, 'mean_mc_accuracy_percent': mean_mc,
          'mean_mc_minus_base_pp': mean_mc - mean_base,
          'runtime': json.loads((folder/'split0.json').read_text())['input_manifest']['runtime']}
cli.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('status', 'mean_base_accuracy_percent',
                                      'mean_mc_accuracy_percent', 'mean_mc_minus_base_pp')}, indent=2))
