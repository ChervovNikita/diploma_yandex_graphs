"""Share preprocessing across fit methods and convergence controllers."""

import hashlib
import json
from pathlib import Path

import torch

from ablation.common.resumable import atomic_torch_save, atomic_write_json
from . import data
from .protocol import RECIPE, REFERENCE, file_hash, inside_repository

# This source fingerprint was audited before the transfer extensions. Its data.py
# is byte-identical to commit 41dadeec575cb11ce9d24861a06353ad80c820be.
AUDITED_LEGACY_SOURCE = '59481d87616ddffc722e6e755a33c9256545dfdb9fdbf0124e6667aeb99ba471'


def preprocessing_config(config):
    sources = {
        'data.py': file_hash(Path(data.__file__)),
        'upstream_manifest.json': file_hash(Path(__file__).parent / 'upstream_manifest.json'),
    }
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return dict(config, source_sha256=identity, recipe=dict(RECIPE))


def cache_identity(config):
    adapted = preprocessing_config(config)
    identity = {name: adapted[name] for name in
                ('source_sha256', 'dataset_sha256', 'recipe', 'dependencies')}
    identity['train_edge_cap'] = config['smoke']['train_edges'] if config['smoke'] else 0
    return identity


def prepare_training(config, args, store, stop):
    adapted = preprocessing_config(config)
    identity = cache_identity(config)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = inside_repository(args.cache_dir) / key
    if getattr(args, 'require_cache', False) and not (root / 'train_valid.pt').is_file():
        raise FileNotFoundError('Prepared shared BUDDY cache required: ' + str(root))
    return data.prepare_training(adapted, args, store, stop)


def reuse_audited_cache(config, args, store, stop):
    """Rebind unchanged preprocessing bytes once during online preparation."""
    identity = cache_identity(config)
    root = inside_repository(args.cache_dir)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = root / key
    target.mkdir(parents=True, exist_ok=True)
    with data.cache_lock(target / 'train_valid.lock', store, stop):
        if (target / 'train_valid.pt').exists():
            return target
        expected = dict(identity, source_sha256=AUDITED_LEGACY_SOURCE)
        for path in sorted(root.glob('*/train_valid.pt')):
            payload = torch.load(str(path), map_location='cpu', weights_only=False, mmap=True)
            if payload.get('identity') != expected:
                continue
            payload['identity'] = identity
            atomic_torch_save(payload, target / 'train_valid.pt')
            legacy_test = path.with_name('test.pt')
            if legacy_test.exists():
                # TEST features are prepared without scoring or selecting models.
                test = torch.load(str(legacy_test), map_location='cpu', weights_only=False, mmap=True)
                atomic_torch_save(test, target / 'test.pt')
            atomic_write_json({
                'operation': 'reuse_unchanged_preprocessing',
                'legacy_source_sha256': AUDITED_LEGACY_SOURCE,
                'unchanged_data_module_sha256': file_hash(Path(data.__file__)),
                'legacy_path': str(path.relative_to(root)),
                'legacy_file_sha256': file_hash(path),
                'new_identity': identity,
                'preprocessing_recipe_changed': False,
                'selected_checkpoint_or_prediction_reused': False,
            }, target / 'reuse_receipt.json')
            return target
    raise FileNotFoundError('No audited legacy cache matched the current dataset and dependencies')
