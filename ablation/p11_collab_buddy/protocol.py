"""Published BUDDY Collab recipe and frozen source identities."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE = REPOSITORY / 'external_sources/buddy'
UPSTREAM_COMMIT = '3562d94a07d1166faa0949030824bf75ad9bb2c4'
RECIPE = {
    'dataset_name': 'ogbl-collab', 'model': 'BUDDY', 'year': 2007,
    'hidden_channels': 1024, 'lr': 0.02, 'weight_decay': 0.0,
    'epochs': 100, 'patience': None, 'scheduler': None,
    'batch_size': 1024, 'eval_steps': 1, 'feature_dropout': 0.05,
    'label_dropout': 0.1, 'sign_dropout': 0.5, 'sign_k': 0,
    'use_feature': True, 'use_struct_feature': True, 'use_RA': False,
    'add_normed_features': True, 'use_zero_one': False, 'floor_sf': False,
    'max_hash_hops': 2, 'hll_p': 8, 'minhash_num_perm': 128,
    'num_negs': 1, 'train_node_embedding': False, 'propagate_embeddings': False,
    'use_valedges_as_input': True, 'K': 50, 'loss': 'bce',
    'preprocessing_seed': 0, 'prediction_average': 'raw_logits',
    'route_count': 4,
}


def dependency_versions():
    return {name: version(name) for name in
            ('torch', 'torch-geometric', 'torch-scatter', 'torch-sparse',
             'ogb', 'datasketch', 'numpy', 'pandas', 'scipy')}


def namespace():
    return SimpleNamespace(**RECIPE)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    manifest = json.loads((Path(__file__).parent / 'upstream_manifest.json').read_text())
    if manifest['commit'] != UPSTREAM_COMMIT:
        raise RuntimeError('BUDDY source pin differs from the frozen protocol')
    for name, expected in manifest['files'].items():
        if file_hash(REFERENCE / name) != expected:
            raise RuntimeError(f'Pinned BUDDY source changed: {name}')
    sources = [Path(__file__).parent / name for name in
               ('protocol.py', 'models.py', 'data.py', 'cache.py', 'run.py', 'collect.py', 'upstream_manifest.json')]
    sources.extend((Path(__file__).parent / 'historical_contrast').glob('*.py'))
    sources.append(Path(__file__).parent / 'historical_contrast/manifest.json')
    sources.append(REPOSITORY / 'ablation/p09_molecular/normalization.py')
    sources.extend(REPOSITORY / name for name in
                   ('models.py', 'ablation/common/worker.py', 'ablation/common/resumable.py'))
    hashes = {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sources}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def inside_repository(path):
    path = Path(path).resolve()
    if not path.is_relative_to(REPOSITORY):
        raise ValueError(f'BUDDY paths must remain inside {REPOSITORY}: {path}')
    return path
