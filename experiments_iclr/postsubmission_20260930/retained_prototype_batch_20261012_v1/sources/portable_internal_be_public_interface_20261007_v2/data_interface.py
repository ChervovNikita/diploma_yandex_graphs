"""Public NPZ TRAIN/VALID input and unchanged V6 batching helpers.

These checks validate supplied arrays. They do not certify that arrays were
obtained from the official dataset. No author approval manifest is accepted or
invented, no provider dataset is instantiated, and no download is performed.
"""
import hashlib
from pathlib import Path
from portable import _module, recipe

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        _DATA = _module('_portable_internal_be_data', 'data.py')
    return _DATA


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def load_train_valid(task, train_npz, valid_npz):
    """Load numeric-only roles; require exact complete benchmark populations."""
    data = _data()
    if task not in data.ROLE_KEYS:
        raise ValueError(task)
    digests = (_sha(train_npz), _sha(valid_npz))
    train = data.load_npz(train_npz, data.ROLE_KEYS[task]['train'])
    valid = data.load_npz(valid_npz, data.ROLE_KEYS[task]['valid'])
    # This is an expected semantic descriptor used by the unchanged validator,
    # not an author role manifest, provenance proof or approval receipt.
    expected = {'split_index': 0} if task == 'wikics' else {'split_kind': 'official_scaffold'}
    if task == 'collab':
        expected = {'valid_negative_count': 100000, 'valid_negative_self_pair_records': 1,
                    'temporal_roles': {'train_max': 2017, 'valid_only': 2018}}
    data.check_projection(task, train, valid, expected)
    if digests != (_sha(train_npz), _sha(valid_npz)):
        raise ValueError('Role file changed during loading/validation')
    provenance = {'schema': 'portable-user-supplied-train-valid-v1', 'task': task,
                  'train_npz_sha256': digests[0], 'valid_npz_sha256': digests[1],
                  'array_shape_domain_role_checks_passed': True,
                  'official_source_verified_by_interface': False,
                  'author_data_approval_claimed': False}
    return train, valid, provenance


def train_batches(task, train, epoch, seed, device='cpu'):
    return _data().batches(task, train, recipe(task)['training'], epoch, seed, device)


def validation_batches(task, train, valid, device='cpu'):
    return _data().valid_batches(task, train, valid, recipe(task)['training'], device)
