"""Typed relocation of unchanged BUDDY GRAPH/TRAIN/VALID receipt identities."""
from pathlib import PurePosixPath

CANONICAL_ROOT = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
SERVER51_ROOT = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
PHASE = 'experiments_iclr/postsubmission_20260930'
ROLES = ('graph', 'train', 'valid')


def absolute(value):
    if not isinstance(value, str) or not value.startswith('/'):
        raise ValueError('Absolute typed provider path required')
    path = PurePosixPath(value)
    if str(path) != value or '..' in path.parts:
        raise ValueError('Normalized provider path required')
    return path


def validate_provider_receipts(actual, canonical, actual_root):
    """Accept only the known root relocation; no file/role/hash substitution.

    The native factory already computed actual file receipts. This function
    neither re-labels those receipts nor touches provider inputs or tensors.
    """
    if actual_root not in (CANONICAL_ROOT, SERVER51_ROOT):
        raise ValueError('Unknown provider execution root')
    root = absolute(actual_root); source = absolute(CANONICAL_ROOT)
    if not isinstance(actual, dict) or not isinstance(canonical, dict) or set(actual) != set(ROLES) or set(canonical) != set(ROLES):
        raise ValueError('Exactly native graph/train/valid receipt roles required')
    relocations = []
    for role in ROLES:
        observed, expected = actual[role], canonical[role]
        if not isinstance(observed, dict) or not isinstance(expected, dict) or set(observed) != {'path', 'bytes', 'sha256'} or set(expected) != {'path', 'bytes', 'sha256'}:
            raise ValueError('Exact native receipt fields required: ' + role)
        canonical_path = absolute(expected['path']); actual_path = absolute(observed['path'])
        if not canonical_path.is_relative_to(source / PHASE):
            raise ValueError('Canonical provider file escapes frozen phase: ' + role)
        relative = canonical_path.relative_to(source)
        if actual_path != root / relative:
            raise ValueError('Provider role path differs from the typed host relocation: ' + role)
        if (type(expected['bytes']) is not int or expected['bytes'] <= 0 or
            type(observed['bytes']) is not int or observed['bytes'] != expected['bytes'] or
            not isinstance(expected['sha256'], str) or len(expected['sha256']) != 64 or
            any(char not in '0123456789abcdef' for char in expected['sha256']) or
            observed['sha256'] != expected['sha256']):
            raise ValueError('Native provider content identity changed: ' + role)
        relocations.append({'role': role, 'canonical_path': str(canonical_path), 'actual_path': str(actual_path),
            'bytes': observed['bytes'], 'sha256': observed['sha256']})
    return {'schema': 'BUDDY-typed-provider-root-binding-v1', 'canonical_root': CANONICAL_ROOT,
        'actual_root': actual_root, 'files': relocations, 'file_bytes_changed': False,
        'native_receipts_relabelled': False, 'TEST_accessed': False}
