"""Post-selection, label-free hidden-state diversity for Roman Empire GNNM.

The analysis refuses incomplete SAGE/GAT matrices. It never trains, selects a
checkpoint, changes model source, or uses labels in its diversity statistics.
Run --self-check for a small CPU-only hook and metric check.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'experiments_iclr' / 'layerwise_member_diversity_results'
SAGE_ROOT = REPO / 'experiments_iclr' / 'results'
SAGE_RERUN = REPO / 'experiments_iclr' / 'gnnm_split0_verification'
GAT_ROOT = REPO / 'experiments_iclr' / 'gat_failure_pair_results'
SAGE_VARIANTS = (
    'gnnm', 'independent_projectors', 'heads_only', 'input_only',
    'output_only', 'base', 'gnnm_m1', 'untied_backbone',
    'freeze_output_factors', 'ens_pooled',
)
GAT_VARIANTS = ('gnnm', 'untied_backbone')
GAT_DEPTH = {0: 5, 1: 5, 2: 4, 3: 4, 4: 4}
SAMPLE_SIZE = 4096
SAMPLE_SEED = 0
LOGIT_ATOL = 1e-3
LOGIT_RTOL = 1e-4
SOURCE_FILES = (
    'experiments_iclr/layerwise_member_diversity.py',
    'experiments_iclr/layerwise_member_diversity_protocol.md',
    'experiments_iclr/projector_controls.py',
    'ablation/models_ablation.py',
    'models.py',
    'datasets.py',
    'run_common.py',
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def row_digest(row: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def rows_from(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f'Empty CSV: {path}')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'Malformed CSV row: {path}')
    return rows


def artifact_path(row: dict[str, str], field: str, root: Path) -> Path:
    extension = '.pt' if field == 'checkpoint' else '.npz'
    directory = 'checkpoints' if field == 'checkpoint' else 'predictions'
    stem = f"roman-empire_{row['model']}_{row['variant']}_split{row['split']}"
    expected = root / directory / f'{stem}{extension}'
    recorded = Path(row[field])
    if not row[field] or recorded.is_absolute() or '..' in recorded.parts:
        raise ValueError(f'Unsafe selected {field}: {row[field]!r}')
    actual = REPO / recorded
    if actual != expected or not actual.is_file() or actual.stat().st_size == 0:
        raise ValueError(f'Missing or unexpected selected {field}: {actual}')
    if not actual.resolve().is_relative_to(REPO.resolve()):
        raise ValueError(f'Selected {field} resolves outside repository: {actual}')
    return actual


def validate_sage() -> tuple[dict[int, dict[str, str]], list[Path]]:
    csv_path = SAGE_ROOT / 'projector_controls.csv'
    rows = rows_from(csv_path)
    expected = {('roman-empire', 'SAGE', variant, str(split))
                for variant in SAGE_VARIANTS for split in range(5)}
    keys = [(r['dataset'], r['model'], r['variant'], r['split']) for r in rows]
    if len(keys) != 50 or set(keys) != expected:
        raise RuntimeError('SAGE control matrix must contain exactly 50 unique rows')
    by_key = {(r['variant'], int(r['split'])): r for r in rows}
    audit_path = SAGE_ROOT / 'gnnm_split0_verification_audit.json'
    audit = json.loads(audit_path.read_text())
    adopted = audit.get('adopted')
    if adopted not in ('pilot', 'fresh'):
        raise RuntimeError('SAGE split-0 adoption audit is missing a decision')
    selected_zero = by_key['gnnm', 0]
    if row_digest(selected_zero) != audit.get(f'{adopted}_row_sha256'):
        raise RuntimeError('Selected SAGE split-0 row differs from adoption audit')
    verifier = REPO / 'experiments_iclr' / 'verify_gnnm_split0.py'
    runner = REPO / 'experiments_iclr' / 'projector_controls.py'
    if sha256(verifier) != audit.get('verifier_sha256') or sha256(runner) != audit.get('runner_sha256'):
        raise RuntimeError('SAGE runner or verifier differs from adoption audit')
    for name, key in (('runner.sha256', 'runner_sha256'),
                      ('verifier.sha256', 'verifier_sha256'),
                      ('matrix.sha256', 'matrix_sha256')):
        if (SAGE_RERUN / name).read_text().strip() != audit.get(key):
            raise RuntimeError(f'SAGE frozen manifest differs: {name}')
    archive = SAGE_RERUN / 'pilot_archive'
    if sha256(archive / 'projector_controls_before_verification.csv') != audit['matrix_sha256']:
        raise RuntimeError('SAGE archived initial matrix differs from its manifest')
    pilot = json.loads((archive / 'pilot_row.json').read_text())
    fresh_rows = rows_from(SAGE_RERUN / 'projector_controls.csv')
    if len(fresh_rows) != 1 or row_digest(fresh_rows[0]) != audit.get('fresh_row_sha256'):
        raise RuntimeError('SAGE fresh verification row differs from audit')
    if row_digest(pilot) != audit.get('pilot_row_sha256'):
        raise RuntimeError('SAGE archived pilot row differs from audit')
    for label, filename, key in (
        ('pilot checkpoint', 'pilot_checkpoint.pt', 'pilot_checkpoint_sha256'),
        ('pilot prediction', 'pilot_prediction.npz', 'pilot_prediction_sha256'),
    ):
        if sha256(archive / filename) != audit.get(key):
            raise RuntimeError(f'SAGE archived {label} differs from audit')
    fresh_ckpt = artifact_path(fresh_rows[0], 'checkpoint', SAGE_RERUN)
    fresh_pred = artifact_path(fresh_rows[0], 'prediction_file', SAGE_RERUN)
    if sha256(fresh_ckpt) != audit.get('fresh_checkpoint_sha256') or sha256(fresh_pred) != audit.get('fresh_prediction_sha256'):
        raise RuntimeError('SAGE fresh artifacts differ from audit')
    # Recheck the decision, since the original verifier requires a 35-row
    # matrix and cannot be rerun after the remaining 15 controls are appended.
    sys.path.insert(0, str(REPO))
    from experiments_iclr.verify_gnnm_split0 import (
        STATE_TOL, VALIDATION_TOL, max_state_difference,
    )
    difference, same_keys = max_state_difference(archive / 'pilot_checkpoint.pt', fresh_ckpt)
    validation_difference = abs(float(pilot['val_metric']) - float(fresh_rows[0]['val_metric']))
    keep_pilot = (same_keys and pilot['best_step'] == fresh_rows[0]['best_step']
                  and validation_difference <= VALIDATION_TOL
                  and difference <= STATE_TOL)
    if (adopted != ('pilot' if keep_pilot else 'fresh')
            or same_keys != audit.get('same_state_keys')
            or not math.isclose(validation_difference, audit['validation_difference'], rel_tol=0, abs_tol=1e-12)
            or (math.isfinite(difference) and not math.isclose(
                difference, audit['maximum_state_difference'], rel_tol=0, abs_tol=1e-12))):
        raise RuntimeError('SAGE pilot adoption rule disagrees with frozen audit')
    for (variant, split), row in by_key.items():
        expected_m = 1 if variant in ('base', 'gnnm_m1') else 4
        if (int(row['seed']) != split or int(row['num_layers']) != 5
                or int(row['hidden_dim']) != 512 or int(row['m']) != expected_m
                or int(row['num_steps']) != 5000
                or not math.isclose(float(row['lr']), 3e-5, rel_tol=0, abs_tol=1e-12)
                or not 1 <= int(row['best_step']) <= 5000):
            raise RuntimeError(f'Mixed SAGE protocol: {variant} split {split}')
        if not math.isfinite(float(row['val_metric'])):
            raise RuntimeError(f'Nonfinite validation metric: {variant} split {split}')
        root = SAGE_RERUN if (variant, split) == ('gnnm', 0) and adopted == 'fresh' else SAGE_ROOT
        artifact_path(row, 'checkpoint', root)
        artifact_path(row, 'prediction_file', root)
    chosen_root = SAGE_RERUN if adopted == 'fresh' else SAGE_ROOT
    if sha256(artifact_path(selected_zero, 'checkpoint', chosen_root)) != audit.get(f'{adopted}_checkpoint_sha256'):
        raise RuntimeError('Selected SAGE split-0 checkpoint differs from audit')
    if sha256(artifact_path(selected_zero, 'prediction_file', chosen_root)) != audit.get(f'{adopted}_prediction_sha256'):
        raise RuntimeError('Selected SAGE split-0 prediction differs from audit')
    return {split: by_key['gnnm', split] for split in range(5)}, [
        csv_path, audit_path, verifier, SAGE_RERUN / 'matrix.sha256',
        SAGE_RERUN / 'runner.sha256', SAGE_RERUN / 'verifier.sha256',
        archive / 'projector_controls_before_verification.csv',
        archive / 'pilot_row.json', archive / 'pilot_checkpoint.pt',
        archive / 'pilot_prediction.npz', SAGE_RERUN / 'projector_controls.csv',
        fresh_ckpt, fresh_pred,
    ]


def validate_gat() -> tuple[dict[int, dict[str, str]], list[Path]]:
    verifier = REPO / 'experiments_iclr' / 'verify_gat_failure_pair.py'
    checked = subprocess.run(
        [sys.executable, str(verifier), '--complete'], cwd=REPO,
        capture_output=True, text=True,
    )
    if checked.returncode:
        raise RuntimeError('Complete GAT verifier failed: ' + (checked.stderr or checked.stdout).strip())
    csv_path = GAT_ROOT / 'projector_controls.csv'
    rows = rows_from(csv_path)
    expected = {('roman-empire', 'GAT', variant, str(split))
                for variant in GAT_VARIANTS for split in range(5)}
    keys = [(r['dataset'], r['model'], r['variant'], r['split']) for r in rows]
    if len(keys) != 10 or set(keys) != expected:
        raise RuntimeError('GAT pair matrix must contain exactly 10 unique rows')
    by_key = {(r['variant'], int(r['split'])): r for r in rows}
    for (variant, split), row in by_key.items():
        if int(row['num_layers']) != GAT_DEPTH[split]:
            raise RuntimeError(f'Unexpected GAT depth: {variant} split {split}')
        artifact_path(row, 'checkpoint', GAT_ROOT)
        artifact_path(row, 'prediction_file', GAT_ROOT)
    return {split: by_key['gnnm', split] for split in range(5)}, [
        csv_path, verifier, GAT_ROOT / 'protocol.json',
    ]


def make_model(model_name: str, depth: int, input_dim: int, output_dim: int) -> torch.nn.Module:
    sys.path.insert(0, str(REPO))
    from ablation.models_ablation import TABMAblationModel
    return TABMAblationModel(
        model_name=model_name, num_layers=depth, input_dim=input_dim,
        hidden_dim=512, output_dim=output_dim, hidden_dim_multiplier=1,
        num_heads=8, normalization='LayerNorm', dropout=0.2,
        tabm_inits=4, init_scheme='default', device='cpu',
    )


def capture(model, data, sampled: torch.Tensor, tested: torch.Tensor, depth: int):
    """Run four eval paths, preserving only sampled hidden states on CPU."""
    if len(model.residual_modules) != depth:
        raise RuntimeError('Expected one residual wrapper per nominal layer')
    expected_module = 'SAGEModule' if model.model_name == 'SAGE' else 'GATModule'
    expected_conv = 'SAGEConv' if model.model_name == 'SAGE' else 'GATConv'
    stages = ['input_after_gelu']
    for layer, wrapper in enumerate(model.residual_modules, 1):
        if (type(wrapper).__name__ != 'ResidualModuleWrapper'
                or type(wrapper.module).__name__ != expected_module
                or type(wrapper.module.conv).__name__ != expected_conv):
            raise RuntimeError(f'Unexpected graph hook structure at layer {layer}')
        stages.extend((f'conv_{layer}', f'residual_{layer}'))
    values = {stage: [] for stage in stages}
    handles = []
    sampled = sampled.to(data.x.device)
    tested = tested.to(data.x.device)

    def hook(stage):
        def save(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.shape != (data.x.shape[0], 512):
                raise RuntimeError(f'Unexpected {stage} output shape')
            values[stage].append(output.detach().index_select(0, sampled).float().cpu().numpy().copy())
        return save

    handles.append(model.act.register_forward_hook(hook(stages[0])))
    for layer, wrapper in enumerate(model.residual_modules, 1):
        handles.append(wrapper.module.conv.register_forward_hook(hook(f'conv_{layer}')))
        handles.append(wrapper.register_forward_hook(hook(f'residual_{layer}')))
    logits = []
    try:
        model.eval()
        with torch.inference_mode():
            for member in range(4):
                output = model(data, data.x, tabm_seed=member)
                if output.ndim != 2 or output.shape[0] != data.x.shape[0]:
                    raise RuntimeError('Unexpected final logit shape')
                logits.append(output.index_select(0, tested).float().cpu().numpy().copy())
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values[stage]) != 4 for stage in stages):
        raise RuntimeError('A hidden-state hook did not fire once per member')
    return stages, values, np.stack(logits, axis=0)


def stage_metrics(values: list[np.ndarray]) -> dict[str, float | int | None]:
    if len(values) != 4 or len({a.shape for a in values}) != 1:
        raise ValueError('Expected four same-shape hidden-state arrays')
    norms = [np.linalg.norm(x.astype(np.float64), axis=1) for x in values]
    nodewise_cosine = []
    squared_difference = 0.0
    squared_reference = 0.0
    invalid = 0
    for a, b in combinations(range(4), 2):
        va, vb = values[a].astype(np.float64), values[b].astype(np.float64)
        norm_a, norm_b = norms[a], norms[b]
        valid = (norm_a > 1e-10) & (norm_b > 1e-10)
        invalid += int((~valid).sum())
        cos_distance = np.full(len(valid), np.nan)
        cos_distance[valid] = 1.0 - np.clip(
            np.einsum('ij,ij->i', va[valid], vb[valid]) /
            (norm_a[valid] * norm_b[valid]), -1.0, 1.0,
        )
        nodewise_cosine.append(cos_distance)
        squared_difference += float(np.square(va - vb).sum())
        squared_reference += float((np.square(norm_a) + np.square(norm_b)).sum() / 2)
    cosine = np.stack(nodewise_cosine, axis=0)
    valid_nodes = np.isfinite(cosine).all(axis=0)
    per_node = cosine[:, valid_nodes].mean(axis=0) if valid_nodes.any() else np.array([])
    rms = math.sqrt(squared_difference / squared_reference) if squared_reference > 0 else None
    return {
        'mean_pairwise_cosine_distance': float(np.nanmean(cosine)) if np.isfinite(cosine).any() else None,
        'median_node_pairwise_cosine_distance': float(np.median(per_node)) if len(per_node) else None,
        'normalized_rms_pairwise_distance': rms,
        'invalid_cosine_pairs': invalid,
    }


def check_prediction(row: dict[str, str], path: Path, indices: np.ndarray,
                     current: np.ndarray) -> dict[str, float | int]:
    with np.load(path, allow_pickle=False) as saved:
        expected_indices = saved['node_index']
        reference = saved['member_logits']
        member_pred = saved['member_pred']
        pooled_pred = saved['ensemble_pred']
    if not np.array_equal(expected_indices, indices):
        raise RuntimeError('Selected prediction node IDs differ from official test mask')
    if reference.shape != current.shape or not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise RuntimeError('Selected member logits have wrong shape or nonfinite values')
    if not np.array_equal(reference.argmax(axis=-1), member_pred):
        raise RuntimeError('Saved member decisions disagree with saved logits')
    if not np.array_equal(reference.mean(axis=0).argmax(axis=-1), pooled_pred):
        raise RuntimeError('Saved pooled decisions disagree with saved logits')
    difference = np.abs(current.astype(np.float64) - reference.astype(np.float64))
    tolerance = LOGIT_ATOL + LOGIT_RTOL * np.abs(reference.astype(np.float64))
    if (difference > tolerance).any():
        raise RuntimeError(f'Recomputed test logits disagree with selected prediction: max error {difference.max():.6g}')
    if not np.array_equal(current.argmax(axis=-1), member_pred):
        raise RuntimeError('Recomputed member decisions disagree with selected prediction')
    if not np.array_equal(current.mean(axis=0).argmax(axis=-1), pooled_pred):
        raise RuntimeError('Recomputed pooled decisions disagree with selected prediction')
    return {
        'test_nodes_checked': len(indices),
        'maximum_logit_absolute_difference': float(difference.max()),
        'maximum_logit_tolerance_ratio': float((difference / tolerance).max()),
        'member_decision_mismatches': 0,
        'pooled_decision_mismatches': 0,
    }


def self_check() -> None:
    from torch_geometric.data import Data
    identical = stage_metrics([np.ones((2, 3), dtype=np.float32) for _ in range(4)])
    if (abs(identical['mean_pairwise_cosine_distance']) > 1e-12
            or abs(identical['normalized_rms_pairwise_distance']) > 1e-12):
        raise AssertionError('Identical hidden states must have zero distance')
    torch.manual_seed(7)
    x = torch.randn(8, 5)
    node = torch.arange(8)
    edge_index = torch.stack((node, (node + 1) % 8))
    edge_index = torch.cat((edge_index, edge_index.flip(0), torch.stack((node, node))), dim=1)
    data = Data(x=x, edge_index=edge_index)
    for name in ('SAGE', 'GAT'):
        model = make_model(name, 2, 5, 3)
        with torch.no_grad():
            model.input_be_block.R.copy_(torch.tensor([
                [1, 1, 1, 1, 1], [1, -1, 1, -1, 1],
                [-1, 1, -1, 1, -1], [-1, -1, -1, -1, -1],
            ], dtype=torch.float32))
        stages, values, logits = capture(model, data, node, node, 2)
        if len(stages) != 5 or logits.shape != (4, 8, 3):
            raise AssertionError(f'{name}: wrong captured shape')
        for stage in stages:
            metric = stage_metrics(values[stage])
            if (not math.isfinite(metric['mean_pairwise_cosine_distance'])
                    or not math.isfinite(metric['normalized_rms_pairwise_distance'])
                    or metric['normalized_rms_pairwise_distance'] <= 0):
                raise AssertionError(f'{name}: invalid or collapsed {stage}')
        with torch.inference_mode():
            direct = np.stack([model(data, data.x, tabm_seed=m).numpy() for m in range(4)])
        if not np.array_equal(logits, direct):
            raise AssertionError(f'{name}: hooks changed final logits')
    print('CPU hook, shape, nonzero-diversity, and unchanged-logit checks passed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('SAGE', 'GAT'))
    parser.add_argument('--splits', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.model is None:
        parser.error('--model is required unless --self-check is used')
    if not args.splits or len(set(args.splits)) != len(args.splits) or any(s not in range(5) for s in args.splits):
        parser.error('Use distinct official split indices 0 through 4')
    chosen, gate_files = validate_sage() if args.model == 'SAGE' else validate_gat()
    data_file = REPO / 'data' / 'roman_empire.npz'
    data_manifest = REPO / 'experiments_iclr' / 'data_manifest.json'
    manifest = json.loads(data_manifest.read_text())
    if sha256(data_file) != manifest['files']['roman_empire.npz']['sha256']:
        raise RuntimeError('Roman Empire data differs from pinned public NPZ')
    frozen = list(dict.fromkeys(
        [REPO / name for name in SOURCE_FILES] + gate_files + [data_file, data_manifest]
    ))
    for split in args.splits:
        row = chosen[split]
        root = ((SAGE_RERUN if split == 0 and
                 json.loads((SAGE_ROOT / 'gnnm_split0_verification_audit.json').read_text())['adopted'] == 'fresh'
                 else SAGE_ROOT) if args.model == 'SAGE' else GAT_ROOT)
        frozen.extend((artifact_path(row, 'checkpoint', root),
                       artifact_path(row, 'prediction_file', root)))
    frozen = list(dict.fromkeys(frozen))
    before = {str(path.relative_to(REPO)): sha256(path) for path in frozen}
    sys.path.insert(0, str(REPO))
    from datasets import load_dataset
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no GPU work was started')
    data, _, _, test_masks, _, output_dim, is_binary = load_dataset(
        'roman-empire', add_self_loops=True, device=device,
        data_dir=str(REPO / 'data'),
    )
    if is_binary or output_dim != 18:
        raise RuntimeError('Unexpected Roman Empire target shape')
    n = data.x.shape[0]
    sampled_np = np.sort(np.random.default_rng(SAMPLE_SEED).choice(
        n, size=min(SAMPLE_SIZE, n), replace=False,
    ).astype(np.int64))
    sampled = torch.as_tensor(sampled_np)
    analyses = []
    for split in sorted(args.splits):
        row = chosen[split]
        depth = int(row['num_layers'])
        model = make_model(args.model, depth, data.x.shape[1], output_dim)
        state = torch.load(REPO / row['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(device)
        test_index_np = test_masks[:, split].nonzero(as_tuple=False).flatten().numpy()
        test_index = torch.as_tensor(test_index_np)
        stages, values, current = capture(model, data, sampled, test_index, depth)
        verification = check_prediction(
            row, REPO / row['prediction_file'], test_index_np, current,
        )
        analyses.append({
            'split': split,
            'depth': depth,
            'selected_validation_step': int(row['best_step']),
            'prediction_verification': verification,
            'stages': [{'stage': stage, **stage_metrics(values[stage])} for stage in stages],
        })
        del model, state, values, current
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    after = {str(path.relative_to(REPO)): sha256(path) for path in frozen}
    if before != after:
        raise RuntimeError('A frozen source, data, matrix, or selected artifact changed during analysis')
    import torch_geometric
    result = {
        'schema': 1,
        'analysis': 'Post-selection label-free member hidden-state diversity',
        'model': args.model,
        'dataset': 'roman-empire',
        'splits': sorted(args.splits),
        'sample': {'rule': 'Uniform node IDs without labels', 'seed': SAMPLE_SEED,
                   'size': len(sampled_np), 'node_id_sha256': hashlib.sha256(sampled_np.tobytes()).hexdigest()},
        'cosine_rule': 'Mean over six member pairs and sampled nodes; zero-norm pairs omitted',
        'rms_rule': 'sqrt(sum pairwise squared differences / sum pairwise mean squared norms)',
        'logit_tolerances': {'atol': LOGIT_ATOL, 'rtol': LOGIT_RTOL},
        'environment': {'python': sys.version.split()[0], 'torch': torch.__version__,
                        'torch_geometric': torch_geometric.__version__, 'numpy': np.__version__,
                        'device': str(device),
                        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
        'frozen_sha256': before,
        'runs': analyses,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = 'all' if sorted(args.splits) == list(range(5)) else 'splits_' + '_'.join(map(str, sorted(args.splits)))
    output = RESULTS / f'{args.model.lower()}_{suffix}.json'
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if output.exists():
        if output.read_text() != payload:
            raise RuntimeError(f'Existing frozen output differs; refusing overwrite: {output}')
    else:
        temp = output.with_suffix('.tmp')
        with temp.open('w') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, output)
    print(f'Wrote {output.relative_to(REPO)}: {len(analyses)} splits, {len(analyses[0]["stages"])} stages each')


if __name__ == '__main__':
    main()
