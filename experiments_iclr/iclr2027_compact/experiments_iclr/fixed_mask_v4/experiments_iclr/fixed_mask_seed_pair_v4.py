"""Fresh paired optimization seeds on Roman Empire official mask 0.

This is an isolated author-side study. See fixed_mask_seed_pair_v4_protocol.md.
The active projector-controls runner and queue are deliberately untouched.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch_geometric
import sklearn
from torch_geometric.data import Data

from projector_controls import (all_logits, compute_loss, make_model,
                                model_output, test_analysis)
from datasets import compute_metrics, load_dataset
from run_common import set_seed

REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / 'experiments_iclr' / 'fixed_mask_seed_pair_v4_results'
VARIANTS = ('gnnm', 'untied_backbone')
SEEDS = (0, 1, 2)
SPLIT = 0
DEPTH = 5
WIDTH = 512
LR = 3e-5
MEMBERS = 4
MAX_STEPS = 5000
PATIENCE_STEPS = 300
CALIBRATION = json.loads((REPO / 'reference/v4_numeric_calibration.json').read_text())
SAME_STATE_MAX = max(max(float(r['tied_repeated_forward_max_abs']), float(r['untied_repeated_forward_max_abs'])) for r in CALIBRATION['rows'])
LOGIT_TOLERANCE = 2 * SAME_STATE_MAX
ARGS = SimpleNamespace(model='SAGE', num_layers=DEPTH,
                       hidden_dim=WIDTH, m=MEMBERS)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_sources() -> str:
    manifest = RESULT / 'source_manifest.json'
    if not manifest.is_file():
        raise RuntimeError(f'Missing predeclared source manifest: {manifest}')
    spec = json.loads(manifest.read_text())
    if spec['protocol'] != 'roman_sage_fixed_mask0_optimization_seeds_0_1_2_v4':
        raise RuntimeError('Wrong source manifest protocol')
    runtime = {
        'python': '.'.join(map(str, sys.version_info[:3])),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'torch_geometric': torch_geometric.__version__,
        'numpy': np.__version__,
        'sklearn': sklearn.__version__,
    }
    if spec['runtime'] != runtime:
        raise RuntimeError(f'Runtime differs from frozen v4 protocol: {runtime}')
    for relative, expected in spec['sha256'].items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or not path.is_file():
            raise RuntimeError(f'Missing or external pinned file: {relative}')
        actual = digest_file(path)
        if actual != expected:
            raise RuntimeError(f'Pinned source/data changed: {relative}: {actual} != {expected}')
    return digest_file(manifest)


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def check_graph_compatibility(data, train, val, test) -> None:
    """Require exact parity with the original DGL-converted Roman graph."""
    reference = np.load(
        REPO / 'reference' / 'roman_dgl_processed_edge_index.npy',
        allow_pickle=False,
    )
    if reference.shape != (2, 88516) or reference.dtype != np.int64:
        raise RuntimeError('Original-server edge reference has changed shape or dtype')
    if not np.array_equal(data.edge_index.detach().cpu().numpy(), reference):
        raise RuntimeError('PyG compatibility loader differs from original DGL edge order')
    if tuple(data.x.shape) != (22662, 300) or tuple(data.y.shape) != (22662,):
        raise RuntimeError('Unexpected Roman node feature or label shape')
    if any(tuple(mask.shape) != (22662, 10) for mask in (train, val, test)):
        raise RuntimeError('Unexpected Roman official mask shape')
    masks = (train[:, SPLIT].bool(), val[:, SPLIT].bool(), test[:, SPLIT].bool())
    if tuple(int(mask.sum()) for mask in masks) != (11331, 5665, 5666):
        raise RuntimeError('Unexpected Roman mask-0 partition sizes')
    if any(bool((left & right).any()) for left, right in
           ((masks[0], masks[1]), (masks[0], masks[2]), (masks[1], masks[2]))):
        raise RuntimeError('Roman mask-0 partitions overlap')
    print('GRAPH_COMPATIBILITY_PASS exact original DGL edge order and mask 0', flush=True)


def canonical_state(model, variant: str):
    if variant == 'gnnm':
        return model.state_dict()
    if variant != 'untied_backbone':
        raise ValueError(variant)
    raw = model.state_dict()
    canonical = {}
    anchor = 'residual_modules_by_member.0.'
    for key, tensor in raw.items():
        if key.startswith('residual_modules_by_member.'):
            if key.startswith(anchor):
                canonical['residual_modules.' + key[len(anchor):]] = tensor
        else:
            canonical[key] = tensor
    for member in range(1, MEMBERS):
        prefix = f'residual_modules_by_member.{member}.'
        for key, tensor in raw.items():
            if key.startswith(prefix):
                reference = canonical['residual_modules.' + key[len(prefix):]]
                if not torch.equal(tensor, reference):
                    raise RuntimeError(f'Untied member {member} was not initialized identically: {key}')
    return canonical


def hash_state(state) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().contiguous().cpu()
        h.update(key.encode() + b'\0')
        h.update(str(tensor.dtype).encode() + b'\0')
        h.update(str(tuple(tensor.shape)).encode() + b'\0')
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def initial_audit(model, variant: str, data, device, run_dir: Path):
    cpu_rng = torch.get_rng_state().clone()
    device_rng = (torch.cuda.get_rng_state(device).clone()
                  if device.type == 'cuda' else cpu_rng)
    model.eval()
    with torch.no_grad():
        logits = all_logits(model, variant, data, MEMBERS).detach().cpu().numpy()
    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise RuntimeError('Initial forward consumed CPU RNG')
    if device.type == 'cuda' and not torch.equal(device_rng, torch.cuda.get_rng_state(device)):
        raise RuntimeError('Initial forward consumed CUDA RNG')
    path = run_dir / 'initial_logits.npy'
    np.save(path, logits)
    audit = {
        'state_sha256': hash_state(canonical_state(model, variant)),
        'cpu_rng_sha256': hash_bytes(cpu_rng.numpy().tobytes()),
        'device_rng_sha256': hash_bytes(device_rng.cpu().numpy().tobytes()),
        'initial_logits_sha256': digest_file(path),
        'initial_logits_shape': list(logits.shape),
    }
    (run_dir / 'initialization.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    return audit, logits


def require_pair_initialization(seed: int, variant: str, audit, logits):
    if variant == 'gnnm':
        return 0.0
    other = RESULT / f'seed{seed}' / 'gnnm'
    previous = json.loads((other / 'initialization.json').read_text())
    for field in ('state_sha256', 'cpu_rng_sha256', 'device_rng_sha256'):
        if previous[field] != audit[field]:
            raise RuntimeError(f'Paired initialization mismatch for seed {seed}: {field}')
    tied_logits = np.load(other / 'initial_logits.npy', allow_pickle=False)
    if tied_logits.shape != logits.shape:
        raise RuntimeError(f'Paired initial logit shape mismatch for seed {seed}')
    max_diff = float(np.max(np.abs(tied_logits - logits)))
    if max_diff > LOGIT_TOLERANCE:
        raise RuntimeError(f'Paired initial logits mismatch for seed {seed}: {max_diff}')
    return max_diff


def run_one(seed: int, variant: str, data, masks, output_dim: int,
            is_binary: bool, device: torch.device, source_sha: str):
    final = RESULT / f'seed{seed}' / variant
    if final.exists():
        row = json.loads((final / 'result.json').read_text())
        if (row['source_manifest_sha256'] != source_sha
                or row['optimization_seed'] != seed
                or row['official_split'] != SPLIT
                or row['variant'] != variant):
            raise RuntimeError(f'Existing run uses different source manifest: {final}')
        for filename in ('checkpoint.pt', 'predictions.npz', 'validation_trace.csv',
                         'initialization.json', 'initial_logits.npy'):
            artifact = final / filename
            if not artifact.is_file() or digest_file(artifact) != row['artifacts_sha256'][filename]:
                raise RuntimeError(f'Incomplete or altered existing run: {artifact}')
        print('SKIP_COMPLETE', seed, variant, flush=True)
        return
    working = final.with_name(final.name + '.inprogress')
    if working.exists():
        raise RuntimeError(f'Interrupted run needs manual review before rerun: {working}')
    working.mkdir(parents=True)
    data.train_mask = masks[0][:, SPLIT].to(device)
    data.val_mask = masks[1][:, SPLIT].to(device)
    data.test_mask = masks[2][:, SPLIT].to(device)
    set_seed(seed)
    model = make_model(ARGS, variant, data.x.shape[1], output_dim, device)
    audit, initial_logits = initial_audit(model, variant, data, device, working)
    pair_init_max_abs_diff = require_pair_initialization(seed, variant, audit, initial_logits)
    del initial_logits
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_val = -float('inf')
    best_step = 0
    since_improvement = 0
    last_step = 0
    trace_path = working / 'validation_trace.csv'
    start = time.monotonic()
    with trace_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('step', 'val_metric', 'improved'))
        writer.writeheader()
        for step in range(1, MAX_STEPS + 1):
            model.train()
            optimizer.zero_grad()
            for member in range(MEMBERS):
                loss = compute_loss(model_output(model, variant, data, member), data, is_binary)
                (loss / MEMBERS).backward()
            optimizer.step()
            last_step = step
            if step % 10 != 0 and step != 1:
                continue
            model.eval()
            with torch.no_grad():
                mean_val_logits = all_logits(model, variant, data, MEMBERS).mean(0)
                val = compute_metrics(mean_val_logits, data.y, data.val_mask,
                                      is_binary, 'roman-empire')['metric']
            improved = val > best_val
            if improved:
                best_val = val
                best_step = step
                since_improvement = 0
                torch.save(model.state_dict(), working / 'checkpoint.pt')
            else:
                since_improvement += 10
            writer.writerow({'step': step, 'val_metric': repr(val), 'improved': int(improved)})
            f.flush()
            if step % 100 == 0 or step == 1:
                print(f'seed={seed} variant={variant} step={step} val={val:.6f} best={best_val:.6f}', flush=True)
            if since_improvement >= PATIENCE_STEPS:
                break
    train_seconds = time.monotonic() - start
    model.load_state_dict(torch.load(working / 'checkpoint.pt', map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = all_logits(model, variant, data, MEMBERS)
        pooled = logits.mean(0)
        val_metric = compute_metrics(pooled, data.y, data.val_mask, is_binary,
                                     'roman-empire')['metric']
        test_metrics = compute_metrics(pooled, data.y, data.test_mask, is_binary,
                                       'roman-empire')
        descriptive, pred = test_analysis(logits, data.y, data.test_mask, data.edge_index)
    if abs(val_metric - best_val) > 1e-7:
        raise RuntimeError(f'Selected checkpoint val mismatch: {val_metric} != {best_val}')
    np.savez_compressed(working / 'predictions.npz', **pred)
    row = {
        'protocol': 'roman_sage_fixed_mask0_optimization_seeds_0_1_2_v4',
        'source_manifest_sha256': source_sha,
        'dataset': 'roman-empire', 'model': 'SAGE', 'variant': variant,
        'official_split': SPLIT, 'optimization_seed': seed,
        'num_layers': DEPTH, 'hidden_dim': WIDTH, 'lr': LR, 'm': MEMBERS,
        'max_steps': MAX_STEPS, 'patience_steps': PATIENCE_STEPS,
        'validation_cadence': 'step 1 and every 10 steps',
        'objective': 'mean of four member cross-entropies',
        'checkpoint_rule': 'strictly greatest pooled validation accuracy; earliest tie',
        'num_params': num_params, 'best_step': best_step, 'last_step': last_step,
        'best_val_metric': best_val, 'val_metric': val_metric,
        'test_metric': test_metrics['metric'], 'test_acc': test_metrics['acc'],
        'test_loss': test_metrics['loss'], 'train_seconds': train_seconds,
        'pair_initial_logits_max_abs_diff': pair_init_max_abs_diff,
        **descriptive,
        'artifacts_sha256': {name: digest_file(working / name) for name in (
            'checkpoint.pt', 'predictions.npz', 'validation_trace.csv',
            'initialization.json', 'initial_logits.npy')},
    }
    (working / 'result.json').write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
    os.rename(working, final)
    print('RESULT', json.dumps(row, sort_keys=True), flush=True)
    del model, optimizer, logits, pooled
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def cpu_smoke():
    device = torch.device('cpu')
    data = Data(x=torch.randn(8, 5),
                edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                         [1, 2, 3, 4, 5, 6, 7, 0]]),
                y=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
    data.train_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
    small = SimpleNamespace(model='SAGE', num_layers=2, hidden_dim=16, m=4)
    audits = []
    logits = []
    for variant in VARIANTS:
        set_seed(0)
        model = make_model(small, variant, 5, 3, device)
        state_sha = hash_state(canonical_state(model, variant))
        cpu_rng_sha = hash_bytes(torch.get_rng_state().numpy().tobytes())
        model.eval()
        with torch.no_grad():
            initial = all_logits(model, variant, data, 4).clone()
        audits.append((state_sha, cpu_rng_sha))
        logits.append(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        model.train()
        optimizer.zero_grad()
        for member in range(4):
            (compute_loss(model_output(model, variant, data, member), data, False) / 4).backward()
        optimizer.step()
        assert any(p.grad is not None for p in model.parameters())
    assert audits[0] == audits[1], 'paired initial state/RNG differs'
    assert torch.equal(logits[0], logits[1]), 'paired initial functions differ'
    print('CPU_SMOKE_PASS paired initial state, RNG, logits, and one training step', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--cpu-smoke', action='store_true')
    cli = parser.parse_args()
    if tuple(cli.seeds) != SEEDS:
        parser.error('The predeclared study requires exactly --seeds 0 1 2 in order')
    os.chdir(REPO)
    source_sha = verify_sources()
    print('SOURCE_MANIFEST_SHA256', source_sha, flush=True)
    if cli.cpu_smoke:
        cpu_smoke()
        return
    if cli.preflight_only:
        data, train, val, test, _, output_dim, is_binary = load_dataset(
            'roman-empire', add_self_loops=True, device='cpu', data_dir='data')
        check_graph_compatibility(data, train, val, test)
        if is_binary or output_dim != 18:
            raise RuntimeError('Unexpected Roman label configuration')
        print('PREFLIGHT_PASS', flush=True)
        return
    device = torch.device(cli.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('The full study requires a CUDA device; use --cpu-smoke for CPU validation')
    data, train, val, test, _, output_dim, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device=device, data_dir='data')
    check_graph_compatibility(data, train, val, test)
    if is_binary or output_dim <= 2:
        raise RuntimeError('Unexpected Roman Empire label configuration')
    masks = train, val, test
    with (RESULT / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in SEEDS:
            for variant in VARIANTS:
                run_one(seed, variant, data, masks, output_dim, is_binary, device, source_sha)


if __name__ == '__main__':
    main()
