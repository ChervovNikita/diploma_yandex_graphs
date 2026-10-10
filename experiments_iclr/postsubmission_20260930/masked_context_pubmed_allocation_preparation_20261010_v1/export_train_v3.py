"""Public-data custodian. Declared stratified PubMed split, no model or scoring."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SPLIT_SEED = 190111


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    header = json.dumps(dict(dtype=value.dtype.str, shape=list(value.shape)), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header + b'\n' + value.tobytes(order='C')).hexdigest()


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
    assert HERE.is_relative_to(PHASE) and Path.cwd().resolve() == REPO
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    destination = HERE / 'data_v3'
    destination.mkdir(exist_ok=False)
    started = time.monotonic()
    import numpy as np
    import torch
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    torch.set_num_threads(2)
    body = json.loads((PHASE / 'masked_context_be_source_prototype_20261010_v2/SOURCE_BINDINGS.json').read_text())
    bound = body['dependencies']['polyformer_utils']
    utils = PHASE / bound['path']
    assert sha(utils) == bound['sha256']
    env = dict(np=np, torch=torch)
    names = {'index_to_mask', 'take_rest', 'random_splits'}
    definitions = [n for n in ast.parse(utils.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions} == names
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(utils), 'exec'), env)
    # Reuse the exact already acquired official graph, with a new role projection.
    original = HERE / 'data_v2/operator_public_source'
    original_custody = json.loads((HERE/'DATA_V2_CUSTODY.json').read_text())
    for row in original_custody['public_raw_files']:
        assert sha(PHASE/row['path']) == row['sha256']
    dataset = Planetoid(root=str(original), name='PubMed', transform=NormalizeFeatures())
    data = dataset[0]
    assert tuple(data.x.shape) == (19717, 500) and dataset.num_classes == 3
    rng = np.random.RandomState(SPLIT_SEED)
    roles = {'train':[], 'valid':[], 'test':[]}
    for c in range(3):
        ids = rng.permutation(np.flatnonzero(data.y.numpy() == c))
        ntrain, nvalid = int(.6*len(ids)), int(.2*len(ids))
        roles['train'].extend(ids[:ntrain]); roles['valid'].extend(ids[ntrain:ntrain+nvalid]); roles['test'].extend(ids[ntrain+nvalid:])
    for role in roles:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[roles[role]] = True
        setattr(data, {'train':'train_mask', 'valid':'val_mask', 'test':'test_mask'}[role], mask)
    train_ids = data.train_mask.nonzero(as_tuple=True)[0]
    valid_ids = data.val_mask.nonzero(as_tuple=True)[0]
    test_ids = data.test_mask.nonzero(as_tuple=True)[0]
    assert len(train_ids) == 11829 and len(valid_ids) == 3942
    assert torch.all(data.train_mask.to(torch.int8) + data.val_mask.to(torch.int8) + data.test_mask.to(torch.int8) == 1)
    counts = {name:torch.bincount(data.y[ids], minlength=3).tolist() for name, ids in [('TRAIN', train_ids), ('VALID', valid_ids), ('TEST', test_ids)]}
    assert min(counts['VALID']) >= 50 and min(counts['TEST']) >= 50
    arrays = dict(x=np.ascontiguousarray(data.x.numpy(), dtype=np.float32), edge_index=np.ascontiguousarray(data.edge_index.numpy(), dtype=np.int64), train_ids=np.ascontiguousarray(train_ids.numpy(), dtype=np.int64), train_y=np.ascontiguousarray(data.y[train_ids].numpy(), dtype=np.int64))
    train_bundle = destination / 'TRAIN_ONLY.npz'
    np.savez_compressed(train_bundle, **arrays)
    np.savez_compressed(destination / 'VALID_ONLY.npz', valid_ids=valid_ids.numpy(), valid_y=data.y[valid_ids].numpy())
    np.savez_compressed(destination / 'SPLIT_IDS.npz', train_ids=train_ids.numpy(), valid_ids=valid_ids.numpy(), test_ids=test_ids.numpy())
    raw = original_custody['public_raw_files']
    custody = dict(schema='masked-context-PubMed-TRAIN-role-custody-v1', UTC=datetime.now(timezone.utc).isoformat(), split_protocol='class_stratified_floor60_20_20', split_seed=SPLIT_SEED, split_identity='PubMed-class-stratified-floor60-20-20', TRAIN_count=11829, VALID_count=len(valid_ids), TEST_count=len(test_ids), class_role_counts=counts, feature_normalization='PyG.NormalizeFeatures', train_bundle_sha256=sha(train_bundle), train_bundle_bytes=train_bundle.stat().st_size, edge_shape=list(arrays['edge_index'].shape), array_fingerprints={k:fingerprint(v) for k,v in arrays.items()}, exporter_source=dict(path=str(Path(__file__).relative_to(PHASE)), sha256=sha(__file__)), native_split_source=bound, native_split_function_used=False, split_recipe='class-wise RandomState190111 permutation, floor60/20/remainder', previous_native_data_custody_sha256=sha(HERE/'DATA_V2_CUSTODY.json'), public_raw_files=raw, source_urls=original_custody['source_urls'], failed_PyG_acquisition_preserved=True, numerical_fit=False, model_scores_accessed=False, heldout_labels_exposed_to_models=False, inclusive_seconds=time.monotonic()-started)
    (HERE / 'DATA_V3_CUSTODY.json').write_text(json.dumps(custody, indent=2, sort_keys=True) + '\n')
    print(json.dumps(custody, sort_keys=True))


if __name__ == '__main__':
    main()
