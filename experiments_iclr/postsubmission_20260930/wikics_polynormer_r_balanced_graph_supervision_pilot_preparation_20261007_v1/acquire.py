#!/usr/bin/env python3
"""Official WikiCS acquisition only; available artifact excludes all TEST labels."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import socket
import subprocess


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    repo = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
    phase = repo / 'experiments_iclr/postsubmission_20260930'
    if Path.cwd() != repo or socket.gethostname() != 'anogena-2-0': raise ValueError('Exact allocation host/cwd required')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact allocation GPU inventory required')
    root = args.root.resolve()
    if root.exists() or not root.is_relative_to(phase.resolve()): raise ValueError('Fresh authorized acquisition root required')
    import torch
    import torch_geometric
    from torch_geometric.datasets import WikiCS
    from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
    if str(torch.__version__) != '2.1.2+cu118' or torch_geometric.__version__ != '2.7.0': raise ValueError('Existing pinned CPU acquisition runtime required')
    root.mkdir(); dataset = WikiCS(root=str(root/'official'), is_undirected=True)
    data = dataset[0]; train_mask = data.train_mask[:, 0]; valid_mask = (data.val_mask | data.stopping_mask)[:, 0]
    if bool((train_mask & valid_mask).any() or (train_mask & data.test_mask).any() or (valid_mask & data.test_mask).any()):
        raise ValueError('Official split0 roles overlap')
    if tuple(data.x.shape) != (11701, 300) or data.x.dtype != torch.float32: raise ValueError('Full raw FP32 WikiCS features required')
    if not bool(torch.isfinite(data.x).all()): raise ValueError('Nonfinite official features')
    edge = to_undirected(data.edge_index, num_nodes=11701)
    edge, _ = remove_self_loops(edge); edge, _ = add_self_loops(edge, num_nodes=11701)
    train_ids = train_mask.nonzero().flatten(); valid_ids = valid_mask.nonzero().flatten()
    permitted_y = torch.cat((data.y[train_ids], data.y[valid_ids]))
    if bool((permitted_y < 0).any() or (permitted_y >= 10).any()): raise ValueError('Declared WikiCS TRAIN/VALID class range differs')
    available = root/'available'; available.mkdir(); payload = available/'wikics_split0.pt'
    torch.save({'x': data.x.contiguous(), 'edge_index': edge.contiguous(), 'train_ids': train_ids,
        'train_y': data.y[train_ids], 'valid_ids': valid_ids, 'valid_y': data.y[valid_ids]}, payload)
    raw = root/'official/raw/data.json'
    manifest = {'dataset': 'WikiCS', 'nodes': 11701, 'features': 300, 'classes': 10, 'official_split': 0,
        'raw_url': 'https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset/data.json',
        'raw_sha256': sha(raw), 'raw_bytes': raw.stat().st_size, 'loader': 'PyG2.7.0 WikiCS(is_undirected=True), no transforms',
        'installed_loader_source_sha256': sha(inspect.getsourcefile(WikiCS)),
        'support': 'to_undirected -> remove_self_loops -> add_self_loops exactly once', 'prepared_directed_edges': edge.shape[1],
        'selfloops': int((edge[0] == edge[1]).sum()), 'TRAIN': len(train_ids), 'VALID': len(valid_ids),
        'VALID_role': '(val_mask | stopping_mask)[:,0]', 'TEST_labels_available_to_trainer': False, 'scientific_training': False,
        'available': {'path': str(payload.relative_to(phase)), 'sha256': sha(payload), 'bytes': payload.stat().st_size},
        'acquisition_runtime': {'torch': str(torch.__version__), 'PyG': torch_geometric.__version__}, 'GPU_tensor_work': False}
    (root/'AVAILABLE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'manifest': str(root/'AVAILABLE_MANIFEST.json'), 'manifest_sha256': sha(root/'AVAILABLE_MANIFEST.json'),
        'nodes': 11701, 'prepared_directed_edges': edge.shape[1], 'TRAIN': len(train_ids), 'VALID': len(valid_ids), 'TEST_labels_available': False, 'training': False}), flush=True)


if __name__ == '__main__': main()
