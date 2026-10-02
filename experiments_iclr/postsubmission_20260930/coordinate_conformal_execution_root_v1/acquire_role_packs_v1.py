"""Prepare only the prospectively fixed graph/role inputs; never fit or score."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BLOB = 'a59f6d2e0dc1f11b8e503c426e86f53124c21eb1'
URL = 'https://api.github.com/repos/yandex-research/heterophilous-graphs/git/blobs/' + BLOB


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def confined(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve().is_relative_to(PHASE) and path.resolve() != PHASE,
            'Require a confined absolute phase path')
    part = PHASE
    for name in path.relative_to(PHASE).parts:
        part /= name
        require(not part.is_symlink(), 'Symlink paths are not accepted')
    return path


def binding(item):
    path = confined(item['path'])
    require(digest(path) == item['sha256'], 'Bound input changed: ' + str(path))
    return path


def descriptor(path):
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def canonical_edges(np, values, nodes):
    edges = np.asarray(values, dtype=np.int64)
    if edges.ndim == 2 and edges.shape[1] == 2:
        edges = edges.T
    require(edges.ndim == 2 and edges.shape[0] == 2, 'Edge shape differs')
    require((edges >= 0).all() and (edges < nodes).all(), 'Edge index outside graph')
    original = edges.shape[1]
    loops = int((edges[0] == edges[1]).sum())
    nonself = edges[:, edges[0] != edges[1]]
    pairs = np.stack((np.minimum(nonself[0], nonself[1]), np.maximum(nonself[0], nonself[1])), 1)
    pairs = np.unique(pairs, axis=0)
    return pairs.T.copy(), {'raw_edge_entries': original, 'raw_self_loop_entries': loops,
                            'canonical_undirected_edges': len(pairs),
                            'teacher_directed_edges': 2 * len(pairs),
                            'rule': 'retain all nodes/nonself adjacency; canonical simple undirected graph; no edge subsampling'}


def acquire(request, output, driver):
    import numpy as np
    import torch
    import torch_geometric
    response_start = time.monotonic()
    http = urllib.request.Request(URL, headers={'User-Agent': 'GNNM-provenance-research',
                                                'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(http, timeout=30) as stream:
        response = stream.read()
    blob = json.loads(response)
    require(blob['sha'] == BLOB and blob['size'] == 208215 and blob['encoding'] == 'base64',
            'Pinned Squirrel object differs')
    contents = base64.b64decode(blob['content'])
    require(len(contents) == 208215 and hashlib.sha1(b'blob 208215\0' + contents).hexdigest() == BLOB,
            'Pinned Squirrel Git object integrity differs')
    raw = output / 'raw_sealed'
    raw.mkdir()
    squirrel_path = raw / 'squirrel_filtered.npz'
    squirrel_path.write_bytes(contents)
    write(raw / 'RETRIEVAL.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'url': URL,
          'git_blob_sha': BLOB, 'response_sha256': hashlib.sha256(response).hexdigest(),
          'payload': descriptor(squirrel_path), 'seconds': time.monotonic() - response_start})
    photo_graph = binding(request['photo_graph'])
    photo_raw = binding(request['photo_raw'])
    photo_manifest_path = binding(request['photo_manifest'])
    photo_manifest = json.loads(photo_manifest_path.read_text())
    require(photo_manifest['graph']['sha256'] == request['photo_graph']['sha256']
            and any(item['source_sha256'] == request['photo_raw']['sha256']
                    for item in photo_manifest['author_raw']), 'Photo graph/raw row provenance differs')
    photo_raw_bytes = photo_raw.read_bytes()
    require(hashlib.sha256(photo_raw_bytes).hexdigest() == request['photo_raw']['sha256'],
            'Photo raw archive changed before verified label extraction')
    graph_bytes = photo_graph.read_bytes()
    require(hashlib.sha256(graph_bytes).hexdigest() == request['photo_graph']['sha256'],
            'Photo graph changed before verified tensor loading')
    graph = torch.load(io.BytesIO(graph_bytes), map_location='cpu', weights_only=True)
    require(set(graph) == {'x', 'edge_index'}, 'Photo graph tensor schema differs')
    records = []
    for name, n, f, classes in [('Squirrel', 2223, 2089, 5), ('Photo', 7650, 745, 8)]:
        root = output / name
        root.mkdir()
        published_masks = None
        if name == 'Squirrel':
            with np.load(squirrel_path, allow_pickle=False) as archive:
                features = np.asarray(archive['node_features'], dtype=np.float32)
                raw_edges = np.asarray(archive['edges'], dtype=np.int64)
                published_masks = {target: np.asarray(archive[source], dtype=np.bool_)
                    for target, source in [('train', 'train_masks'), ('validation', 'val_masks'), ('pool', 'test_masks')]}
            provider = {'repository': 'yandex-research/heterophilous-graphs',
                        'release': 'squirrel_filtered.npz', 'git_blob_sha': BLOB,
                        'raw': descriptor(squirrel_path)}
        else:
            features = graph['x'].numpy().copy()
            raw_edges = graph['edge_index'].numpy().copy()
            provider = {'repository': 'shchur/gnn-benchmark', 'release': 'amazon_electronics_photo.npz',
                        'reused_graph_only': True, 'graph': request['photo_graph'], 'raw': request['photo_raw'],
                        'public_manifest': request['photo_manifest'],
                        'executed_pyg_loader': photo_manifest['executed_pyg_loader'],
                        'upstream_preprocessing': photo_manifest['preprocessing'],
                        'upstream_feature_rule': 'Pinned PyG read_npz binarizes positiveCSR attributes; bound public graph values reused unchanged'}
        require(features.dtype == np.float32 and features.shape == (n, f) and np.isfinite(features).all(),
                'Prospective graph dimension/feature integrity differs')
        edges, edge_audit = canonical_edges(np, raw_edges, n)
        np.save(root / 'features.npy', features, allow_pickle=False)
        np.save(root / 'edges.npy', edges, allow_pickle=False)
        metadata = {'graph': name, 'num_nodes': n, 'num_features': f, 'num_classes': classes,
                    'num_edges': edges.shape[1], 'features': descriptor(root / 'features.npy'),
                    'edges': descriptor(root / 'edges.npy'), 'release_provenance': provider,
                    'preprocessing': {'features': ('raw released values cast toFP32; no normalization'
                         if name == 'Squirrel' else 'reuse bound PyG public graph features unchanged; no additional normalization'),
                         'edges': edge_audit}}
        if published_masks is not None:
            for key, values in published_masks.items():
                if values.shape[0] == n:
                    values = values.T
                require(values.ndim == 2 and values.shape[0] >= 3 and values.shape[1] == n,
                        'Published mask dimensions differ')
                published_masks[key] = values
            np.savez(root / 'published_masks.npz', **published_masks)
            metadata['published_masks'] = descriptor(root / 'published_masks.npz')
        graph_manifest = root / 'GRAPH_INPUT.json'
        write(graph_manifest, metadata)
        # Prepare and fingerprint every source/final role before dereferencing
        # raw label arrays. The prepare process receives graph/masks only.
        command = [sys.executable, str(driver), 'prepare', '--input', str(graph_manifest),
                   '--output', str(root / 'prepared_roles')]
        binding(request['driver'])
        with (root / 'PREPARATION.log').open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=100, check=False)
        require(result.returncode == 0, 'Role preparation failed; preserve the attempt')
        binding(request['driver'])
        # This provider may partition public raw labels, but fitting receives
        # only compact source packs. Final-pool labels are separately sealed.
        with np.load(io.BytesIO(contents if name == 'Squirrel' else photo_raw_bytes), allow_pickle=False) as archive:
            labels = np.asarray(archive['node_labels' if name == 'Squirrel' else 'labels'], dtype=np.int64)
        require(labels.shape == (n,) and (labels >= 0).all() and (labels < classes).all(),
                'Prospectively declared label schema differs')
        packs = root / 'labels'
        packs.mkdir()
        source_root, pool_root = packs / 'source', packs / 'sealed_pool'
        source_root.mkdir(); pool_root.mkdir()
        cells = []
        for split, seed in enumerate((17, 29, 43)):
            cell = root / 'prepared_roles' / f'seed{seed}_split{split}'
            role_freeze = cell / 'ROLE_FREEZE.json'
            role_record = json.loads(role_freeze.read_text())
            require(role_record['labels_read'] is False, 'Source roles were not label-blind')
            source = {}
            for role in ('train', 'validation', 'A', 'B', 'D', 'pool'):
                nodes = np.load(cell / (role + '_nodes.npy'), allow_pickle=False)
                target = (pool_root if role == 'pool' else source_root) / f'seed{seed}_split{split}_{role}.npz'
                np.savez(target, nodes=nodes, labels=labels[nodes])
                if role != 'pool':
                    source[role] = descriptor(target)
                else:
                    pool = descriptor(target)
            cells.append({'graph': name, 'seed': seed, 'source_split_index': split,
                          'role_freeze': descriptor(role_freeze), 'source_labels': source,
                          'sealed_pool_labels': pool, 'roles_frozen_before_label_extraction': True,
                          'source_counts': role_record['source_counts']})
        write(root / 'LABEL_PACK_MANIFEST.json', {'graph': descriptor(graph_manifest), 'cells': cells,
              'raw_labels_dereferenced_by_provider': True, 'labels_used_only_for_compact_packaging': True,
              'fits_or_scores_performed': False, 'final_pool_labels_supplied_to_fit': False})
        records.append(descriptor(root / 'LABEL_PACK_MANIFEST.json'))
        del labels, features, edges
    environment_path = output / 'FIT_ENVIRONMENT.json'
    result = subprocess.run([sys.executable, str(driver), 'environment', '--device', 'cuda:0',
                             '--output', str(environment_path)], capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, 'Future fit runtime fingerprint failed: ' + result.stderr)
    binding(request['driver'])
    return {'graphs': records, 'fit_environment': descriptor(environment_path),
            'runtime': {'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__, 'numpy': np.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--supervisor', required=True)
    args = parser.parse_args()
    require(Path.cwd().resolve() == REPO and os.environ.get('GNNM_PHASE_ROOT') == str(PHASE)
            and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Require confined remote wrapper')
    request_path = confined(args.request)
    request_sha = digest(request_path)
    request = json.loads(request_path.read_text())
    require(request['root_admitted'] is True and request['training_admitted'] is False
            and request['heldout_scoring_admitted'] is False and request['squirrel_git_blob_sha'] == BLOB,
            'Only pinned prospective acquisition/role preparation is admitted')
    require(os.environ.get('GNNM_SSH_DESTINATION') ==
            'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru', 'Unauthorized SSH route')
    gpu_rows = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True, timeout=10).stdout.strip().splitlines()
    require(gpu_rows == [GPU], 'Require exactly the authorized oneGPU allocation')
    for item in request['protected_files']:
        binding(item)
    driver = binding(request['driver'])
    output = confined(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write(output / 'START.json', {'UTC': datetime.now(timezone.utc).isoformat(),
          'request_sha256': request_sha, 'source_sha256': digest(Path(__file__)), 'training_steps': 0})
    try:
        payload = acquire(request, output, driver)
        for item in request['protected_files']:
            binding(item)
        require(digest(request_path) == request_sha, 'Admission changed during acquisition')
        write(output / 'TERMINAL.json', {'complete': True, 'elapsed_seconds': time.monotonic()-start,
              'request_sha256': request_sha, 'training_steps': 0, 'heldout_scores': 0, **payload})
    except Exception as error:
        write(output / 'FAILED_ATTEMPT.json', {'type': type(error).__name__, 'reason': str(error),
              'elapsed_seconds': time.monotonic()-start, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
