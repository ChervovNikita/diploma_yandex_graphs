"""Disabled CPU description of existing complete Wiki15 selected-state geometry.

No model construction, forward, training, probe, selector or accuracy/NLL scorer.
Trusted checkpoint loading is needed solely to read the frozen active head maps.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time


def geometry(np, h, labels):
    """All same-class/different-class node pairs; no graph-edge interpretation."""
    norms = np.linalg.norm(h, axis=-1, keepdims=True)
    unit = h / np.maximum(norms, 1e-12)
    same_sum = np.zeros(4)
    same_count = 0
    residual = h.copy()
    for cls in range(10):
        mask = labels == cls
        count = int(mask.sum())
        assert count >= 2, 'All ten disclosed classes must be retained'
        part = unit[:, mask]
        sums = part.sum(axis=1)
        same_sum += (sums * sums).sum(axis=-1) - (part * part).sum(axis=(1, 2))
        same_count += count * (count - 1)
        residual[:, mask] -= h[:, mask].mean(axis=1, keepdims=True)
    sums = unit.sum(axis=1)
    all_sum = (sums * sums).sum(axis=-1) - (unit * unit).sum(axis=(1, 2))
    different_count = len(labels) * (len(labels) - 1) - same_count
    same, different = same_sum / same_count, (all_sum - same_sum) / different_count
    rn = np.linalg.norm(residual, axis=-1, keepdims=True)
    ru = residual / np.maximum(rn, 1e-12)
    cosines = np.concatenate([(ru[a] * ru[b]).sum(-1) for a, b in itertools.combinations(range(4), 2)])
    return residual, dict(
        class_counts=[int((labels == c).sum()) for c in range(10)],
        same_class_cosine_by_member=same.tolist(), different_class_cosine_by_member=different.tolist(),
        class_cosine_gap_by_member=(same - different).tolist(),
        class_cosine_gap_mean=float((same - different).mean()),
        residual_same_node_member_cosine_mean=float(cosines.mean()),
        residual_same_node_member_cosine_quantiles=np.quantile(cosines, [0, .1, .5, .9, 1]).tolist(),
        hidden_norm_floor_count=int((norms < 1e-12).sum()),
        residual_norm_floor_count=int((rn < 1e-12).sum()),
        ordered_same_class_pairs_per_member=same_count,
        ordered_different_class_pairs_per_member=different_count)


def dispersion(np, values):
    numerator = float(np.mean([np.mean((values[a] - values[b])**2)
                               for a, b in itertools.combinations(range(4), 2)]))
    denominator = float(np.mean(values**2))
    return dict(pair_RMS=numerator**.5, energy_RMS=denominator**.5,
                relative_pair_squared_energy=numerator / denominator if denominator else None,
                zero_energy=denominator == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-adopted', action='store_true')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.root_adopted:
        parser.error('Source preparation only; explicit root adoption is required before numerical imports')
    began = time.monotonic()
    raw_config = args.config.read_bytes()
    assert hashlib.sha256(raw_config).hexdigest() == args.config_sha256
    config = json.loads(raw_config)
    assert socket.gethostname() == config['hostname']
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                        text=True, timeout=10).splitlines()
    assert inventory == config['physical_gpu_inventory']
    root = Path(config['repository']).resolve()
    phase = Path(config['phase']).resolve()
    assert phase.is_relative_to(root)
    assert Path.cwd().resolve() == root
    assert str(Path(sys.executable).absolute()) == config['python']
    assert os.environ.get('PYTHONPATH', '') == ''
    assert args.config.resolve().is_relative_to(phase)
    assert args.output.resolve().is_relative_to(phase) and not args.output.exists()
    gate_path = phase / config['existing_gate']['path']
    assert hashlib.sha256(gate_path.read_bytes()).hexdigest() == config['existing_gate']['sha256']
    spec = importlib.util.spec_from_file_location('_Wiki15_existing_metadata_helpers', gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    inputs = gate.read(gate.bound(config['inputs']))
    for row in config['required_sources']:
        gate.bound(row)
    assert {(r['seed'], r['condition']) for r in inputs['cells']} == {
        (s, c) for s in (6101, 6203, 6307)
        for c in ('plain', 'alignment_only', 'residual_only', 'combined', 'supcon_eq2')}
    assert len(inputs['cells']) == 15
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['NUMEXPR_NUM_THREADS'] = '2'
    import numpy as np
    import torch
    assert np.__version__ == '1.26.4' and str(torch.__version__) == '2.1.2+cu118'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = []
    try:
        for record in inputs['cells']:
            assert time.monotonic() - began < 90, 'Fixed CPU aggregate wall budget exceeded'
            with np.load(gate.bound(record['raw_archive']), allow_pickle=False) as package:
                h = package['member_representations'].astype(np.float64)
                z = package['member_logits'].astype(np.float64)
                labels, ids = package['truth'], package['valid_ids']
            assert h.shape == (4, 5274, 512) and z.shape == (4, 5274, 10)
            assert np.isfinite(h).all() and np.isfinite(z).all()
            assert labels.dtype == ids.dtype == np.dtype('int64')
            for values, key in ((ids, 'valid_ids'), (labels, 'valid_y')):
                assert hashlib.sha256(values.tobytes()).hexdigest() == config['role_hashes'][key]
            checkpoint = gate.bound(record['selected_checkpoint'])
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['epoch'] == record['selected_epoch']
            assert type(saved['global']) is bool and saved['global'] == record['global_mode']
            assert saved['run']['core'] == config['core_sha256']
            prefix = 'models.0.body.' + ('pred_global' if saved['global'] else 'pred_local') + '.'
            state = saved['model']
            w, r, s, bias = [state[prefix + name].detach().cpu().numpy().astype(np.float64)
                             for name in ('weight', 'r', 's', 'bias')]
            assert w.shape == (10, 512) and r.shape == (4, 512) and s.shape == (4, 10) and bias.shape == (10,)
            assert all(np.isfinite(v).all() for v in (w, r, s, bias))
            del saved, state
            # Exact FactorLinear law: bias is added AFTER output scaling.
            folded = s[:, :, None] * w[None] * r[:, None, :]
            reconstructed = np.stack([h[m] @ folded[m].T + bias for m in range(4)])
            error = np.abs(reconstructed - z)
            # Conservative FP32 multiply/accumulate bound, not a score tolerance.
            u = np.finfo(np.float32).eps / 2
            gamma = (2 * 512 + 6) * u / (1 - (2 * 512 + 6) * u)
            magnitude = np.stack([np.abs(h[m] * r[m]) @ np.abs(w).T * np.abs(s[m])
                                 + np.abs(bias) for m in range(4)])
            assert np.all(error <= gamma * magnitude + np.finfo(np.float32).tiny), 'Folded head/cache mismatch'
            residual, descriptive = geometry(np, h, labels)
            # Remove the softmax class-offset gauge. Class residuals also remove bias.
            class_map = folded - folded.mean(axis=1, keepdims=True)
            visible = np.stack([residual[m] @ class_map[m].T for m in range(4)])
            null_fractions, ranks = [], []
            for m in range(4):
                _, singular, right = np.linalg.svd(class_map[m], full_matrices=False)
                tolerance = max(class_map[m].shape) * np.finfo(np.float64).eps * singular[0]
                rank = int((singular > tolerance).sum())
                ranks.append(rank)
                total = float(np.sum(residual[m]**2))
                projection_energy = float(np.sum((residual[m] @ right[:rank].T)**2))
                null_fractions.append(max(0., 1 - projection_energy / total) if total else None)
            margins = z[np.arange(4)[:, None], np.arange(5274)[None, :], labels[None, :]][:, :, None] - z
            mask = np.arange(10)[None, :] != labels[:, None]
            all_margins = margins[:, mask].reshape(4, 5274, 9)
            rows.append(dict(cell=record['cell'], seed=record['seed'], condition=record['condition'],
                             selected_epoch=record['selected_epoch'], global_mode=record['global_mode'],
                             folded_head_max_absolute_cache_difference=float(error.max()),
                             residual_head_visible_dispersion=dispersion(np, visible),
                             all_truth_vs_other_class_margin_dispersion=dispersion(np, all_margins),
                             native_metric_head_null_energy_fraction_by_member=null_fractions,
                             class_contrast_head_rank_by_member=ranks, **descriptive))
            gate.bound(record['selected_checkpoint'])
            del h, z, labels, ids, residual, visible, reconstructed, error, magnitude, margins, all_margins
        deltas = []
        by = {(row['seed'], row['condition']): row for row in rows}
        for seed in (6101, 6203, 6307):
            for name, candidate, base in (
                ('A-P', 'alignment_only', 'plain'), ('R-P', 'residual_only', 'plain'),
                ('C-P', 'combined', 'plain'), ('C-A', 'combined', 'alignment_only'),
                ('S-A', 'supcon_eq2', 'alignment_only'), ('S-P', 'supcon_eq2', 'plain')):
                a, b = by[seed, candidate], by[seed, base]
                deltas.append(dict(seed=seed, contrast=name,
                                   candidate_selected_epoch=a['selected_epoch'], baseline_selected_epoch=b['selected_epoch'],
                                   candidate_global_mode=a['global_mode'], baseline_global_mode=b['global_mode'],
                                   selected_stages_match=a['global_mode'] == b['global_mode'],
                                   class_cosine_gap_delta=a['class_cosine_gap_mean'] - b['class_cosine_gap_mean'],
                                   residual_member_cosine_delta=a['residual_same_node_member_cosine_mean'] - b['residual_same_node_member_cosine_mean'],
                                   head_visible_pair_RMS_delta=a['residual_head_visible_dispersion']['pair_RMS'] - b['residual_head_visible_dispersion']['pair_RMS'],
                                   full_class_margin_pair_RMS_delta=a['all_truth_vs_other_class_margin_dispersion']['pair_RMS'] - b['all_truth_vs_other_class_margin_dispersion']['pair_RMS']))
        result = dict(schema='Wiki15-selected-member-geometry-descriptive-aggregate-v1', status='complete',
                      UTC=datetime.now(timezone.utc).isoformat(), cells=rows, paired_endpoint_deltas=deltas,
                      training=False, models_or_forwards=0, probes_or_fits=0, scores_recomputed=False,
                      raw_arrays_transferred=False, TEST_access=False, config=gate.binding(args.config),
                      source=gate.binding(Path(__file__)), wall_seconds=time.monotonic() - began)
    except Exception as error:
        result = dict(schema='Wiki15-selected-member-geometry-descriptive-aggregate-v1', status='retained_failure',
                      failure=dict(type=type(error).__name__, message=str(error)), completed_cells=rows,
                      comparable_complete_roster=False, automatic_retry=False,
                      wall_seconds=time.monotonic() - began)
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result.update(CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
                      peak_RSS_bytes=usage.ru_maxrss * 1024)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
