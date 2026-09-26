"""Independent completion and checkpoint gate for the fixed-mask seed pair study."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fixed_mask_seed_pair_v4 import (DEPTH, LOGIT_TOLERANCE, LR, MAX_STEPS,
                                  MEMBERS, PATIENCE_STEPS, REPO, RESULT, SEEDS,
                                  SPLIT, VARIANTS, WIDTH, canonical_state,
                                  check_graph_compatibility,
                                  digest_file, hash_bytes, hash_state,
                                  verify_sources)
from projector_controls import all_logits, make_model
from datasets import compute_metrics, load_dataset
from run_common import set_seed

PROTOCOL = 'roman_sage_fixed_mask0_optimization_seeds_0_1_2_v4'
ARTIFACTS = ('checkpoint.pt', 'predictions.npz', 'validation_trace.csv',
             'initialization.json', 'initial_logits.npy')
REPLAY_LOGIT_TOLERANCE = LOGIT_TOLERANCE


def check_sources() -> str:
    spec_path = RESULT / 'source_manifest.json'
    spec = json.loads(spec_path.read_text())
    assert spec['protocol'] == PROTOCOL
    for relative, expected in spec['sha256'].items():
        path = (REPO / relative).resolve()
        assert path.is_relative_to(REPO) and path.is_file(), relative
        assert digest_file(path) == expected, relative
    return verify_sources()


def trace_winner(path: Path, last_step: int):
    with path.open(newline='') as stream:
        trace = list(csv.DictReader(stream))
    assert trace, path
    expected_steps = [1] + list(range(10, last_step + 1, 10))
    assert [int(r['step']) for r in trace] == expected_steps, path
    assert last_step in (5000, expected_steps[-1]), path
    best = -float('inf')
    best_step = 0
    since = 0
    for row in trace:
        step = int(row['step'])
        val = float(row['val_metric'])
        assert math.isfinite(val) and 0 <= val <= 1, (path, step)
        improved = val > best
        assert int(row['improved']) == int(improved), (path, step)
        if improved:
            best, best_step, since = val, step, 0
        else:
            since += 10
        if since >= PATIENCE_STEPS:
            assert step == last_step, (path, step, last_step)
    assert last_step == MAX_STEPS or since >= PATIENCE_STEPS, path
    return best_step, best


def check_row(seed: int, variant: str, source_sha: str, data, masks,
              output_dim: int, device: torch.device):
    folder = RESULT / f'seed{seed}' / variant
    row = json.loads((folder / 'result.json').read_text())
    assert row['protocol'] == PROTOCOL
    assert row['source_manifest_sha256'] == source_sha
    exact = {'dataset': 'roman-empire', 'model': 'SAGE', 'variant': variant,
             'official_split': SPLIT, 'optimization_seed': seed,
             'num_layers': DEPTH, 'hidden_dim': WIDTH, 'm': MEMBERS,
             'max_steps': MAX_STEPS, 'patience_steps': PATIENCE_STEPS,
             'validation_cadence': 'step 1 and every 10 steps',
             'objective': 'mean of four member cross-entropies',
             'checkpoint_rule': 'strictly greatest pooled validation accuracy; earliest tie'}
    for key, expected in exact.items():
        assert row[key] == expected, (folder, key)
    assert math.isclose(float(row['lr']), LR, rel_tol=0, abs_tol=1e-12)
    for name in ARTIFACTS:
        assert (folder / name).is_file(), (folder, name)
        assert digest_file(folder / name) == row['artifacts_sha256'][name], (folder, name)
    best_step, best_val = trace_winner(folder / 'validation_trace.csv', int(row['last_step']))
    assert int(row['best_step']) == best_step
    assert abs(float(row['best_val_metric']) - best_val) <= 1e-9
    assert 1 <= best_step <= int(row['last_step']) <= MAX_STEPS
    init = json.loads((folder / 'initialization.json').read_text())
    assert digest_file(folder / 'initial_logits.npy') == init['initial_logits_sha256']
    init_logits = np.load(folder / 'initial_logits.npy', allow_pickle=False)
    assert list(init_logits.shape) == init['initial_logits_shape']
    assert init_logits.shape == (MEMBERS, data.x.shape[0], output_dim)
    assert np.isfinite(init_logits).all()
    args = SimpleNamespace(model='SAGE', num_layers=DEPTH, hidden_dim=WIDTH, m=MEMBERS)
    set_seed(seed)
    model = make_model(args, variant, data.x.shape[1], output_dim, device)
    assert row['num_params'] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    cpu_sha = hash_bytes(torch.get_rng_state().numpy().tobytes())
    device_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()
    dev_sha = hash_bytes(device_rng.cpu().numpy().tobytes())
    assert hash_state(canonical_state(model, variant)) == init['state_sha256'], folder
    assert cpu_sha == init['cpu_rng_sha256'] and dev_sha == init['device_rng_sha256'], folder
    model.eval()
    with torch.no_grad():
        fresh_logits = all_logits(model, variant, data, MEMBERS).cpu().numpy()
    assert np.max(np.abs(fresh_logits - init_logits)) <= LOGIT_TOLERANCE, folder
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    with torch.no_grad():
        logits = all_logits(model, variant, data, MEMBERS)
        pooled = logits.mean(0)
        val_metric = compute_metrics(pooled, data.y, data.val_mask, False, 'roman-empire')['metric']
        test_metrics = compute_metrics(pooled, data.y, data.test_mask, False, 'roman-empire')
    assert abs(val_metric - best_val) <= 1e-7, folder
    assert abs(val_metric - float(row['val_metric'])) <= 1e-7, folder
    for field, actual in (('test_metric', test_metrics['metric']),
                          ('test_acc', test_metrics['acc']),
                          ('test_loss', test_metrics['loss'])):
        assert abs(actual - float(row[field])) <= 1e-6, (folder, field)
    with np.load(folder / 'predictions.npz', allow_pickle=False) as pred:
        expected_nodes = data.test_mask.nonzero(as_tuple=False).flatten().cpu().numpy()
        assert np.array_equal(pred['node_index'], expected_nodes), folder
        assert np.array_equal(pred['y_true'], data.y[data.test_mask].cpu().numpy()), folder
        expected_logits = logits[:, data.test_mask].detach().cpu().numpy()
        assert pred['member_logits'].shape == expected_logits.shape
        assert np.max(np.abs(pred['member_logits'] - expected_logits)) <= REPLAY_LOGIT_TOLERANCE, folder
        assert np.array_equal(pred['member_logits'].argmax(axis=-1),
                              expected_logits.argmax(axis=-1)), folder
        expected_pred = pooled[data.test_mask].argmax(dim=-1).cpu().numpy()
        assert np.array_equal(pred['ensemble_pred'], expected_pred), folder
    return row, init, init_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--self-test', action='store_true')
    cli = parser.parse_args()
    if cli.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.csv'
            path.write_text('step,val_metric,improved\n1,0.4,1\n10,0.5,1\n20,0.5,0\n')
            try:
                trace_winner(path, 20)
            except AssertionError:
                pass  # Incomplete before the early-stop threshold is rejected.
            else:
                raise AssertionError('Incomplete trace was accepted')
        print('GATE_SELF_TEST_PASS')
        return
    os.chdir(REPO)
    source_sha = check_sources()
    device = torch.device(cli.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    assert not list(RESULT.glob('seed*/*.inprogress')), 'An unfinished run exists'
    expected = {(seed, variant) for seed in SEEDS for variant in VARIANTS}
    found = set()
    for seed_dir in RESULT.glob('seed*'):
        assert seed_dir.is_dir() and seed_dir.name[4:].isdigit(), seed_dir
        seed = int(seed_dir.name[4:])
        for run_dir in seed_dir.iterdir():
            assert run_dir.is_dir() and (run_dir / 'result.json').is_file(), run_dir
            found.add((seed, run_dir.name))
    assert found == expected, f'Expected exactly six complete runs; found {sorted(found)}'
    data, train, val, test, _, output_dim, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device=device, data_dir='data')
    check_graph_compatibility(data, train, val, test)
    assert not is_binary
    masks = train, val, test
    data.train_mask = train[:, SPLIT].to(device)
    data.val_mask = val[:, SPLIT].to(device)
    data.test_mask = test[:, SPLIT].to(device)
    rows = {}
    for seed in SEEDS:
        pair = {}
        for variant in VARIANTS:
            pair[variant] = check_row(seed, variant, source_sha, data, masks,
                                      output_dim, device)
            rows[seed, variant] = pair[variant][0]
            print('VERIFIED', seed, variant, flush=True)
        left, right = pair['gnnm'], pair['untied_backbone']
        for field in ('state_sha256', 'cpu_rng_sha256', 'device_rng_sha256'):
            assert left[1][field] == right[1][field], (seed, field)
        diff = float(np.max(np.abs(left[2] - right[2])))
        assert diff <= LOGIT_TOLERANCE, (seed, diff)
        assert abs(diff - float(right[0]['pair_initial_logits_max_abs_diff'])) <= 1e-9
    deltas = [100 * (rows[seed, 'gnnm']['test_acc'] - rows[seed, 'untied_backbone']['test_acc'])
              for seed in SEEDS]
    audit = {'protocol': PROTOCOL, 'source_manifest_sha256': source_sha,
             'verified_runs': 6, 'official_split': SPLIT,
             'optimization_seeds': list(SEEDS),
             'paired_test_accuracy_deltas_percentage_points': deltas,
             'mean_paired_delta_percentage_points': float(np.mean(deltas)),
             'sd_paired_delta_percentage_points': float(np.std(deltas, ddof=1)),
             'note': 'Three optimization seeds on one fixed official mask; descriptive, not graph-level replication.'}
    tmp = RESULT / 'completion_audit.json.tmp'
    tmp.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, RESULT / 'completion_audit.json')
    print('COMPLETE_GATE_PASS', json.dumps(audit, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
