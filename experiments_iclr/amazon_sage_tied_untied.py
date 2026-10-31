"""Initially matched tied versus untied SAGE on Amazon Ratings official masks 0-2.

This is an isolated author-side study. See amazon_sage_tied_untied_protocol.md.
The active projector-controls runner and queue are deliberately untouched.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

from projector_controls import (all_logits, compute_loss, make_model,
                                model_output, test_analysis)
from datasets import compute_metrics, load_dataset
from run_common import set_seed

REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / 'experiments_iclr' / 'amazon_sage_tied_untied_results'
VARIANTS = ('gnnm', 'untied_backbone')
SPLITS = (0, 1, 2)
DATASET = 'amazon-ratings'
DEPTH = 5
WIDTH = 512
LR = 3e-5
MEMBERS = 4
MAX_STEPS = 5000
PATIENCE_STEPS = 300
LOGIT_TOLERANCE = 1e-6
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
    if spec['protocol'] != 'amazon_sage_tied_untied_masks0_1_2_v1':
        raise RuntimeError('Wrong source manifest protocol')
    if set(spec.get('sha256', {})) != set(PINNED):
        raise RuntimeError('Source manifest pin set differs from protocol')
    if spec['sha256']['data/amazon_ratings.npz'] != AMAZON_SHA256:
        raise RuntimeError('Source manifest pins wrong Amazon Ratings data')
    for relative, expected in spec['sha256'].items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or not path.is_file():
            raise RuntimeError(f'Missing or external pinned file: {relative}')
        actual = digest_file(path)
        if actual != expected:
            raise RuntimeError(f'Pinned source/data changed: {relative}: {actual} != {expected}')
    return digest_file(manifest)



PINNED = (
    'data/amazon_ratings.npz',
    'experiments_iclr/data_manifest.json',
    'datasets.py',
    'models.py',
    'run_common.py',
    'ablation/models_ablation.py',
    'experiments_iclr/projector_controls.py',
    'experiments_iclr/amazon_sage_tied_untied.py',
    'experiments_iclr/verify_amazon_sage_tied_untied.py',
    'experiments_iclr/amazon_sage_tied_untied_protocol.md',
    'experiments_iclr/run_amazon_sage_tied_untied_once.sh',
)
AMAZON_SHA256 = '4c3a3e3b9d9f6cba0fede4625a00aad8c5721c1a36ed771367f446763241c7dd'

def dataset_preflight():
    path = REPO / 'data' / 'amazon_ratings.npz'
    if digest_file(path) != AMAZON_SHA256:
        raise RuntimeError('Corrected public Amazon Ratings NPZ hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        expected = {'node_features': (24492, 300), 'node_labels': (24492,),
                    'edges': (93050, 2), 'train_masks': (10, 24492),
                    'val_masks': (10, 24492), 'test_masks': (10, 24492)}
        if set(data.files) != set(expected):
            raise RuntimeError('Unexpected Amazon Ratings NPZ fields')
        for key, shape in expected.items():
            if data[key].shape != shape:
                raise RuntimeError(f'Unexpected Amazon Ratings {key} shape')
        labels = data['node_labels']
        if not np.array_equal(np.unique(labels), np.arange(5)):
            raise RuntimeError('Unexpected Amazon Ratings classes')
        if not np.isfinite(data['node_features']).all():
            raise RuntimeError('Nonfinite Amazon Ratings features')
        edge = data['edges']
        if edge.min() < 0 or edge.max() >= len(labels):
            raise RuntimeError('Amazon Ratings edge endpoint outside graph')
        train, val, test = (data[key] for key in ('train_masks', 'val_masks', 'test_masks'))
        if any(a.dtype != np.bool_ for a in (train, val, test)):
            raise RuntimeError('Amazon Ratings masks are not boolean')
        if np.any((train.astype('i1') + val.astype('i1') + test.astype('i1')) != 1):
            raise RuntimeError('Amazon Ratings official masks do not partition nodes')
        for index in SPLITS:
            if (int(train[index].sum()), int(val[index].sum()), int(test[index].sum())) != (12246, 6123, 6123):
                raise RuntimeError(f'Unexpected Amazon Ratings mask {index} counts')
    return True

def check_untied_storage(model):
    pointers = []
    for stack in model.residual_modules_by_member:
        pointers.extend(parameter.untyped_storage().data_ptr() for parameter in stack.parameters())
    if len(pointers) != len(set(pointers)):
        raise RuntimeError('Untied backbone members share parameter storage')

def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    other = RESULT / f'mask{seed}' / 'gnnm'
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
    final = RESULT / f'mask{seed}' / variant
    if final.exists():
        row = json.loads((final / 'result.json').read_text())
        if (row['source_manifest_sha256'] != source_sha
                or row['optimization_seed'] != seed
                or row['official_split'] != seed
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
    data.train_mask = masks[0][:, seed].to(device)
    data.val_mask = masks[1][:, seed].to(device)
    data.test_mask = masks[2][:, seed].to(device)
    set_seed(seed)
    model = make_model(ARGS, variant, data.x.shape[1], output_dim, device)
    if variant == 'untied_backbone':
        check_untied_storage(model)
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
                                      is_binary, DATASET)['metric']
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
                                     DATASET)['metric']
        test_metrics = compute_metrics(pooled, data.y, data.test_mask, is_binary,
                                       DATASET)
        descriptive, pred = test_analysis(logits, data.y, data.test_mask, data.edge_index)
    if abs(val_metric - best_val) > 1e-7:
        raise RuntimeError(f'Selected checkpoint val mismatch: {val_metric} != {best_val}')
    np.savez_compressed(working / 'predictions.npz', **pred)
    row = {
        'protocol': 'amazon_sage_tied_untied_masks0_1_2_v1',
        'source_manifest_sha256': source_sha,
        'dataset': DATASET, 'model': 'SAGE', 'variant': variant,
        'official_split': seed, 'optimization_seed': seed,
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
    parser.add_argument('--splits', nargs='+', type=int, default=list(SPLITS))
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--cpu-smoke', action='store_true')
    cli = parser.parse_args()
    if tuple(cli.splits) != SPLITS:
        parser.error('The predeclared study requires exactly --splits 0 1 2 in order')
    os.chdir(REPO)
    source_sha = verify_sources()
    dataset_preflight()
    print('SOURCE_MANIFEST_SHA256', source_sha, flush=True)
    if cli.cpu_smoke:
        cpu_smoke()
        return
    if cli.preflight_only:
        print('PREFLIGHT_PASS', flush=True)
        return
    device = torch.device(cli.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('The full study requires a CUDA device; use --cpu-smoke for CPU validation')
    data, train, val, test, _, output_dim, is_binary = load_dataset(
        DATASET, add_self_loops=True, device=device, data_dir='data')
    if is_binary or output_dim != 5 or tuple(data.x.shape) != (24492, 300):
        raise RuntimeError('Unexpected Amazon Ratings dimensions or labels')
    masks = train, val, test
    with (RESULT / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in SPLITS:
            for variant in VARIANTS:
                run_one(seed, variant, data, masks, output_dim, is_binary, device, source_sha)


if __name__ == '__main__':
    main()
