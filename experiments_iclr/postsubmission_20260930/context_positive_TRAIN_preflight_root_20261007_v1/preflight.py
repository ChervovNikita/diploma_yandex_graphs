"""Representative TRAIN-only target eligibility; no VALID, TEST, prediction or fit."""
from pathlib import Path
import hashlib
import json
import os
import resource
import socket
import subprocess
import time
import numpy as np
import torch
from context_positive_masks import graph_contexts, positive_masks, sampled_weights, within_class_permuted

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
ROLE_SHA = 'b4769a68123433fc498b6abfafbcc3190f104749eafd72b34384e70dd41ff6aa'
TRAIN_SHA = 'dce6c4a982027604dad198533352d397c5d1b7a0a6d6fa4868468c872457f9e2'
MASK_SOURCE_SHA = '7615d8441a699f297314e7c838b9eb897b9e3dbfe22d1be350742e83f21405a9'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def main():
    assert HERE.is_relative_to(PHASE) and socket.gethostname() == 'anogena-2-0'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert digest(HERE / 'context_positive_masks.py') == MASK_SOURCE_SHA
    started = time.monotonic()
    torch.set_num_threads(2)
    manifest = PHASE / 'learnable_internal_be_safe_role_export_execution_20261007_v1/outputs/wikics/ROLE_MANIFEST.json'
    assert digest(manifest) == ROLE_SHA
    role = json.loads(manifest.read_text())
    row = role['payloads']['train']
    train = (PHASE / row['path']).resolve()
    assert train.is_relative_to(PHASE) and row['sha256'] == TRAIN_SHA and digest(train) == TRAIN_SHA
    with np.load(train, allow_pickle=False) as archive:
        assert set(archive.files) == {'x', 'edge_index', 'ids', 'y'}
        x, edge, ids, labels = (archive[key].copy() for key in ('x', 'edge_index', 'ids', 'y'))
    assert x.shape == (11701, 300) and x.dtype == np.float32
    assert edge.shape == (2, 442907) and edge.dtype == np.int64
    assert ids.shape == labels.shape == (580,) and ids.dtype == labels.dtype == np.int64
    assert len(np.unique(ids)) == 580 and np.isfinite(x).all() and np.isin(labels, np.arange(10)).all()
    rows = torch.linspace(0, 579, steps=512, device='cpu').long().numpy()
    contexts = graph_contexts(x, edge, ids, chunk_edges=1024)
    masks = positive_masks(contexts, labels, neighbors=16)
    permuted, permutations = within_class_permuted(masks, labels, 991327, panel_rows=rows)
    route = sampled_weights(masks, rows, 'route')
    common = sampled_weights(masks, rows, 'common')
    randomized = sampled_weights(permuted, rows, 'route')
    restricted = masks[:, rows[:, None], rows[None, :]]
    restricted_permuted = permuted[:, rows[:, None], rows[None, :]]
    assert np.array_equal(restricted.sum(-1), restricted_permuted.sum(-1))
    assert np.all(np.diagonal(masks, axis1=1, axis2=2)) and np.all(np.diagonal(permuted, axis1=1, axis2=2))
    incompatible = labels[:, None] != labels[None, :]
    assert not np.any(masks & incompatible[None]) and not np.any(permuted & incompatible[None])
    assert np.allclose(route.mean(0), common[0], atol=1e-12, rtol=1e-12)
    target_tv = .5 * np.abs(route - common).sum(-1)
    permutation_tv = .5 * np.abs(route - randomized).sum(-1)
    target_mean, permutation_mean = float(target_tv.mean()), float(permutation_tv.mean())
    eligible = target_mean >= .05 and permutation_mean >= .05
    output = HERE / 'FROZEN_TRAIN_TARGETS.npz'
    assert not output.exists()
    np.savez(output, train_ids=ids, train_labels=labels, panel_rows=rows,
             masks=masks, permuted_masks=permuted, permutations=permutations)
    report = {
        'status': 'ELIGIBLE' if eligible else 'INELIGIBLE_NO_RESCUE',
        'TRAIN_only': True, 'VALID_or_TEST_labels_accessed': False,
        'model_predictions_or_checkpoint_access': False, 'scientific_fits': 0,
        'role_manifest_sha256': ROLE_SHA, 'train_npz_sha256': TRAIN_SHA,
        'mask_source_sha256': MASK_SOURCE_SHA, 'permutation_seed': 991327,
        'neighbor_count': 16, 'panel_count': 512, 'complete_TRAIN_count': 580,
        'target_vs_common_mean_TV': target_mean,
        'target_vs_permuted_mean_TV': permutation_mean,
        'per_route_target_vs_common_mean_TV': target_tv.mean(-1).tolist(),
        'per_route_target_vs_permuted_mean_TV': permutation_tv.mean(-1).tolist(),
        'every_scored_anchor_degree_preserved': True, 'positive_labels_and_self_preserved': True,
        'aggregate_target_mass_preserved': True,
        'restricted_positive_count_range': [int(restricted.sum(-1).min()), int(restricted.sum(-1).max())],
        'original_panel_sha256': hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),
        'target_archive_sha256': digest(output), 'target_archive_bytes': output.stat().st_size,
        'providers': {'numpy': np.__version__, 'torch': str(torch.__version__)},
        'elapsed_seconds': time.monotonic() - started,
        'peak_RSS_KiB_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'interpretation': 'Distinct target construction on complete original TRAIN; no semantic subclass, accuracy or novelty evidence.'}
    with (HERE / 'PREPROCESSING_RESULT.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
