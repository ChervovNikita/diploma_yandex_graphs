"""Recreate the authorized WikiCS numeric TRAIN/VALID roles on CPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    repo = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
    phase = repo / 'experiments_iclr/postsubmission_20260930'
    assert socket.gethostname() == 'anogena-2-0' and Path.cwd() == repo
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    root = args.root.resolve()
    assert root.is_relative_to(phase) and not root.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    import numpy as np
    import torch
    import torch_geometric
    from torch_geometric.datasets import WikiCS
    from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
    assert torch.__version__ == '2.1.2+cu118' and torch_geometric.__version__ == '2.7.0'
    torch.set_num_threads(2)
    root.mkdir()
    data = WikiCS(root=str(root/'official'), is_undirected=True)[0]
    raw = root/'official/raw/data.json'
    assert sha(raw) == '9bf8cb3ef8eeae81b25e6ccbe0ea195600c205d7edf63ce04f2ec8d9c7dcb3d8'
    train_mask = data.train_mask[:,0]
    valid_mask = (data.val_mask | data.stopping_mask)[:,0]
    assert not (train_mask & valid_mask).any() and not ((train_mask | valid_mask) & data.test_mask).any()
    edge = to_undirected(data.edge_index, num_nodes=11701)
    edge, _ = remove_self_loops(edge)
    edge, _ = add_self_loops(edge, num_nodes=11701)
    assert data.x.shape == (11701,300) and edge.shape == (2,442907)
    train_ids = train_mask.nonzero().flatten()
    valid_ids = valid_mask.nonzero().flatten()
    assert len(train_ids) == 580 and len(valid_ids) == 5274
    arrays = dict(x=data.x.contiguous().numpy(), edge_index=edge.contiguous().numpy(), train_ids=train_ids.numpy(), train_y=data.y[train_ids].numpy(), valid_ids=valid_ids.numpy(), valid_y=data.y[valid_ids].numpy())
    expected = dict(x='6eafa01f0c8c2a3d4cd185b6ee166318aa5fb625cedfe8b66e357e489b71f847', edge_index='0d1e603f6169bc3e11b12d6b035a229960db4139f2294b617745b268f7793a76', train_ids='7586be66453b94a5408fed647a65b90c4a6819afecf891ba518c69221a8361fd', train_y='683746232a70217a0d4d1eda747195022bf08ce9004d69bcb48900bfb1448c6d', valid_ids='48d17843cf300ef7ec3d09e5aaaff26bd81f55a0af73f7ae03d6df8a7a700801', valid_y='2ab8078de1ca949e111b8cfec4a41c8c04ca8ca4be86478daf58d2e683f4cb12')
    fingerprints = {k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}
    assert fingerprints == expected, 'Preserve differing arrays without admitting them to a fit'
    assert not torch.cuda.is_initialized()
    np.savez_compressed(root/'train.npz', x=arrays['x'], edge_index=arrays['edge_index'], ids=arrays['train_ids'], y=arrays['train_y'])
    np.savez_compressed(root/'valid.npz', ids=arrays['valid_ids'], y=arrays['valid_y'])
    report = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), hostname=socket.gethostname(), source_sha256=sha(Path(__file__)), raw_sha256=sha(raw), raw_url='https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset/data.json', torch=torch.__version__, pyg=torch_geometric.__version__, numpy=np.__version__, split=0, VALID_role='(val_mask | stopping_mask)[:,0]', role_tensor_fingerprints=fingerprints, numeric_roles_equal_current_authorized77=True, payloads={k:dict(path=str((root/(k+'.npz')).relative_to(phase)), sha256=sha(root/(k+'.npz')), bytes=(root/(k+'.npz')).stat().st_size) for k in ('train','valid')}, seconds=time.monotonic()-started, scientific_fits=0, TEST_scoring=False, TEST_labels_available_in_trainer_roles=False, public_raw_labels_loaded_only_for_role_projection=True, GPU_tensor_work=False)
    (root/'COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(complete=True, numeric_roles_equal_current_authorized77=True, payloads=report['payloads'], seconds=report['seconds'])))


if __name__ == '__main__':
    main()
