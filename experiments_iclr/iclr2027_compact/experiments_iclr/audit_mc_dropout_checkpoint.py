"""Independent replay of the fixed five-mask MC Dropout outputs.

Copy to experiments_iclr/ beside the existing immutable prediction records.
This script writes only a new audit JSON. It neither trains nor changes any
selected checkpoint, stochastic draw, or result record.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import mc_dropout_control as mc
from datasets import compute_metrics, load_dataset
from projector_controls import make_model
from types import SimpleNamespace

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument('--device', required=True)
cli = parser.parse_args()
device = torch.device(cli.device)
assert device.type == 'cuda' and device.index is not None
freeze_path = HERE / 'mc_dropout_replay_freeze.json'
freeze = json.loads(freeze_path.read_text())
assert freeze['auditor_sha256'] == sha(Path(__file__))
assert freeze['logit_max_abs_tolerance'] == 1e-4
assert freeze['require_identical_member_and_pooled_classes'] is True
for relative, digest in freeze['input_sha256'].items():
    path = (ROOT / relative).resolve(strict=True)
    assert ROOT.resolve() in path.parents and sha(path) == digest
assert (mc.SPLITS, mc.PASSES, mc.SEED_BASE, mc.SEED_STRIDE) == (tuple(range(5)), 4, 750000, 1000)
torch.set_num_threads(2)
source_rows = mc.source_rows()
graph, train, valid, test, _, classes, binary = load_dataset(
    'roman-empire', add_self_loops=True, device=device, data_dir=str(ROOT / 'data'))
assert not binary and classes == 18
reports = []
for split in range(5):
    train_mask, valid_mask, test_mask = (v[:, split].to(device) for v in (train, valid, test))
    path = HERE / 'mc_dropout_results' / f'split{split}.json'
    archived = json.loads(path.read_text())
    source = archived['input_manifest']
    assert source_rows[split] == source['row']
    assert mc.runtime_manifest(device) == source['runtime']
    checkpoint = mc.checked_checkpoint(source['row']['checkpoint'])
    assert sha(checkpoint) == source['checkpoint_sha256']
    model = make_model(SimpleNamespace(model='SAGE', num_layers=5,
                                       hidden_dim=512, m=1),
                       'base', graph.x.size(1), classes, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert sum(p.numel() for p in model.parameters()) == int(source['row']['num_params'])
    model.eval()
    with torch.inference_mode():
        deterministic = model(graph, graph.x)
        va = compute_metrics(deterministic, graph.y, valid_mask, False, 'roman-empire')
        te = compute_metrics(deterministic, graph.y, test_mask, False, 'roman-empire')
    assert abs(va['metric'] - archived['deterministic_base_val_metric']) < 1e-7
    assert abs(te['metric'] - archived['deterministic_base_test_metric']) < 1e-7
    assert abs(te['loss'] - archived['deterministic_base_test_loss']) < 1e-5
    base_prediction_path = ROOT / source['row']['prediction_file']
    base_replay = deterministic[test_mask].cpu().numpy()
    with np.load(base_prediction_path, allow_pickle=False) as base_saved:
        assert np.array_equal(base_saved['node_index'], torch.where(test_mask)[0].cpu().numpy())
        assert np.array_equal(base_saved['y_true'], graph.y[test_mask].cpu().numpy())
        assert base_saved['member_logits'].shape == (1, 5666, 18)
        base_logit_error = float(np.max(np.abs(base_saved['member_logits'][0] - base_replay)))
        assert base_logit_error <= freeze['logit_max_abs_tolerance']
        assert np.array_equal(base_replay.argmax(-1), base_saved['ensemble_pred'])
        assert np.array_equal(base_replay.argmax(-1), base_saved['member_pred'][0])
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            assert module.p == .2
            module.train()
        else:
            assert module.training is False
    outputs = []
    with torch.inference_mode():
        for j in range(4):
            with torch.random.fork_rng(devices=[device.index]):
                torch.default_generator.manual_seed(750000 + 1000 * split + j)
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(750000 + 1000 * split + j)
                outputs.append(model(graph, graph.x).detach())
        replay = torch.stack(outputs)[:, test_mask].cpu().numpy()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    prediction_path = HERE / 'mc_dropout_results' / f'split{split}_predictions.npz'
    assert sha(prediction_path) == archived['prediction_sha256']
    with np.load(prediction_path, allow_pickle=False) as saved:
        old = saved['member_logits']
        assert old.shape == replay.shape == (4, 5666, 18)
        assert np.array_equal(saved['node_index'], torch.where(test_mask)[0].cpu().numpy())
        assert np.array_equal(saved['y_true'], graph.y[test_mask].cpu().numpy())
        maximum = float(np.max(np.abs(replay - old)))
        assert maximum <= freeze['logit_max_abs_tolerance']
        assert np.array_equal(replay.argmax(-1), saved['member_pred'])
        assert np.array_equal(replay.mean(0).argmax(-1), saved['ensemble_pred'])
        accuracy = float(np.mean(replay.mean(0).argmax(-1) == saved['y_true']))
    assert abs(accuracy - archived['mc_test_acc']) < 1e-7
    reports.append({'split': split, 'checkpoint_sha256': sha(checkpoint),
                    'prediction_sha256': sha(prediction_path),
                    'baseline_prediction_sha256': sha(base_prediction_path),
                    'baseline_max_abs_logit_error': base_logit_error,
                    'baseline_decision_mismatches': 0,
                    'record_sha256': sha(path), 'max_abs_member_logit_error': maximum,
                    'member_decision_mismatches': 0, 'pooled_decision_mismatches': 0,
                    'replayed_accuracy': accuracy})
    print('REPLAY_PASS', split, flush=True)
    del model, before, deterministic, outputs, replay
    torch.cuda.empty_cache()
assert len(reports) == 5
output = HERE / 'mc_dropout_results' / 'independent_checkpoint_replay.json'
assert not output.exists()
output.write_text(json.dumps({'status': 'COMPLETE_FIVE_MASK_CUDA_REPLAY_PASS',
                              'replay_freeze_sha256': sha(freeze_path),
                              'runtime': mc.runtime_manifest(device),
                              'records': reports}, indent=2) + '\n')
print('COMPLETE_FIVE_MASK_CUDA_REPLAY_PASS', flush=True)
