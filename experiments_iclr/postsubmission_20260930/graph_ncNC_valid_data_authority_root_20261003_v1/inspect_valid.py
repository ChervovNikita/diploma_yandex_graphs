"""Authenticate only existing TRAIN/raw/VALID files; no model or TEST access."""
from datetime import datetime, timezone
import codecs
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
OUT = PHASE / 'graph_ncNC_valid_data_authority_root_20261003_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    os.chdir(REPO)
    assert subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip() == str(REPO)
    uuids = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert set(uuids) == {'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998',
                          'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
    # Ordinary installed runtime. No namespace, mount, or environment mutation.
    import numpy as np
    import torch
    assert torch.__version__.split('+')[0] == '2.7.1'
    bindings = json.loads((PHASE / 'graph_ncNC_collab_resource_epoch_preparation_20261003_v4/BINDINGS.json').read_text())
    cache = json.loads((PHASE / 'graph_ncNC_collab_resource_epoch_preparation_20261003_v4/custody_metadata/EXISTING_CACHE_MANIFEST_METADATA.json').read_text())
    root = Path(bindings['dataset_root'])
    assert root.resolve().is_relative_to(REPO)
    files = dict(bindings['allowed_data_files'])
    validpath = root / 'split/time/valid.pt'
    files['split/time/valid.pt'] = {'bytes': validpath.stat().st_size,
                                   'sha256': cache['official_split_files']['valid']['sha256']}
    assert set(files) == {'raw/node-feat.csv.gz', 'raw/edge.csv.gz',
                          'split/time/train.pt', 'split/time/valid.pt'}
    for rel, pin in files.items():
        path = root / rel
        assert path.resolve().is_relative_to(root.resolve())
        assert not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(REPO))
        assert path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256']
    loaded = {}
    identities = {}
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
               type(np.dtype(np.int64)), codecs.encode]
    for split, keys in [('train', {'edge', 'year', 'weight'}),
                        ('valid', {'edge', 'edge_neg', 'year', 'weight'})]:
        with torch.serialization.safe_globals(allowed):
            d = torch.load(root / ('split/time/' + split + '.pt'),
                           map_location='cpu', weights_only=True)
        assert type(d) is dict and set(d) == keys
        entries = {}
        for name, a in d.items():
            assert type(a) is np.ndarray and a.dtype == np.dtype(np.int64)
            h = hashlib.sha256(str((a.shape, a.dtype)).encode())
            h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
            entries[name] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                             'sha256': h.hexdigest()}
        edge = d['edge']
        assert edge.ndim == 2 and edge.shape[1] == 2
        assert np.all((edge >= 0) & (edge < bindings['full_node_count']))
        assert d['year'].shape[0] == edge.shape[0] == d['weight'].shape[0]
        if split == 'train':
            assert int(d['year'].max()) == 2017
            assert entries['edge']['sha256'] == bindings['data_tensor_digests']['train_records']
        else:
            assert np.all(d['year'] == 2018)
            neg = d['edge_neg']
            assert neg.ndim == 2 and neg.shape[1] == 2
            assert np.all((neg >= 0) & (neg < bindings['full_node_count']))
            assert entries['edge']['sha256'] == cache['split_identity']['valid']['positive_sha256']
            assert entries['edge_neg']['sha256'] == cache['split_identity']['valid']['negative_sha256']
        loaded[split] = entries
        identities[split] = {'min_year': int(d['year'].min()), 'max_year': int(d['year'].max()),
                             'positive_records': int(len(edge))}
        del d
    dist = importlib.metadata.distribution('ogb')
    evaluator = Path(dist.locate_file('ogb/linkproppred/evaluate.py')).resolve()
    source = evaluator.read_text()
    assert 'def _eval_hits(' in source and 'torch.topk' in source
    result = dict(schema='ncnc-collab-TRAIN-VALID-data-authority-v1',
                  UTC=datetime.now(timezone.utc).isoformat(), dataset_root=str(root),
                  files=files, expected_arrays=loaded, official_year_authority=identities,
                  train_raw_tensor_digests=bindings['data_tensor_digests'],
                  full_node_count=bindings['full_node_count'],
                  ogb_evaluator=dict(path=str(evaluator), version=dist.version,
                                     sha256=sha(evaluator), bytes=evaluator.stat().st_size),
                  runtime=dict(torch=torch.__version__, numpy=np.__version__),
                  physical_GPU_UUIDs=uuids, test_file_opened=False,
                  model_or_checkpoint_loaded=False, predictive_metrics_computed=False,
                  original_paper_scores_changed=False, existing_BUDDY_cache_used_as_graph_features=False)
    OUT.mkdir(exist_ok=False)
    target = OUT / 'DATA_AUTHORITY.json'
    with target.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    # Source text is a small reproducibility binding, not a data or model payload.
    private = OUT / 'private_evidence'
    private.mkdir()
    (private / 'ogb_evaluate.py.txt').write_text(source)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
