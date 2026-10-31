"""Future closed-input verification. Standard-library only; never opens archives."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ARCHIVE_SHA = '90c38ab363dd57ce67ec62e1501532a5189403119c9cf4791a91f127321a3ad6'
PUBLIC = ('info.yaml', 'features.csv', 'edgelist.csv', 'split_masks_RL.csv')
PUBLIC_INPUT_FILES = {'ROLE_FREEZE.json'} | {'tolokers-2/' + name for name in PUBLIC}
INPUT_FILES = {'ROLE_FREEZE.json', 'train_labels.json', 'val_labels.json',
               'LABELPACK_RECEIPT.json', 'SOURCE_LABEL_PARITY.json'} | {
                   'tolokers-2/' + name for name in PUBLIC}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.pending')
    with tmp.open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')
    tmp.replace(path)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.reader(file))


def bool_token(value):
    value = value.strip().lower()
    if value not in {'true', 'false', '0', '1'}:
        raise ValueError('Official mask has an unreviewed boolean token')
    return value in {'true', '1'}


def verify_inputs(root, policy_path, *, public_only=False):
    """Input integrity is separate from operational isolation (namespace_guard)."""
    root = Path(root)
    files = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in input bundle')
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError('Special file in input bundle')
    if files != (PUBLIC_INPUT_FILES if public_only else INPUT_FILES):
        raise ValueError('Worker input allowlist differs; no raw targets/archive admitted')
    roles = load(root / 'ROLE_FREEZE.json')
    freeze_sha = sha(root / 'ROLE_FREEZE.json')
    if (roles.get('schema') != 'tolokers2-public-role-freeze-v1'
            or roles.get('dataset') != 'tolokers-2' or roles.get('split') != 'RL'
            or roles.get('source_archive_sha256') != ARCHIVE_SHA
            or roles.get('role_policy_sha256') != sha(policy_path)
            or roles.get('targets_member_opened') is not False
            or roles.get('test_labels') != 'CLOSED'):
        raise ValueError('Public role freeze identity/policy differs')
    for name in PUBLIC:
        path = root / 'tolokers-2' / name
        descriptor = roles['public_members'][name]
        if sha(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
            raise ValueError('Public member hash/size differs: ' + name)
    feature_rows = csv_rows(root / 'tolokers-2/features.csv')
    mask_rows = csv_rows(root / 'tolokers-2/split_masks_RL.csv')
    node_ids = [row[0] for row in feature_rows[1:]]
    if not node_ids or len(node_ids) != len(set(node_ids)) or node_ids != roles['node_ids']:
        raise ValueError('Frozen public feature row axis differs')
    if [row[0] for row in mask_rows[1:]] != node_ids:
        raise ValueError('Official masks differ from feature row order')
    columns = {role: mask_rows[0].index(role) for role in ('train', 'val', 'test')}
    actual = {role: [] for role in columns}
    unassigned = []
    for node_id, row in zip(node_ids, mask_rows[1:]):
        bits = {role: bool_token(row[index]) for role, index in columns.items()}
        if sum(bits.values()) > 1:
            raise ValueError('Official roles overlap')
        if not any(bits.values()):
            unassigned.append(node_id)
        else:
            actual[next(role for role, bit in bits.items() if bit)].append(node_id)
    if actual != roles['official_role_ids'] or unassigned != roles['unassigned_node_ids']:
        raise ValueError('Official role IDs were altered')
    if public_only:
        descriptors = {name: {'sha256': sha(root / name), 'bytes': (root / name).stat().st_size}
                       for name in sorted(PUBLIC_INPUT_FILES)}
        return roles, None, {'schema': 'public-only-worker-input-identity-v2', 'files': descriptors,
                             'role_freeze_sha256': freeze_sha, 'test_labels': 'CLOSED'}
    label_receipt = load(root / 'LABELPACK_RECEIPT.json')
    parity = load(root / 'SOURCE_LABEL_PARITY.json')
    if (label_receipt.get('role_freeze_sha256') != freeze_sha
            or label_receipt.get('test_values_semantically_decoded') is not False
            or label_receipt.get('test_statistics_emitted') is not False
            or parity.get('passed') is not True or parity.get('role_freeze_sha256') != freeze_sha
            or parity.get('labelpack_receipt_sha256') != sha(root / 'LABELPACK_RECEIPT.json')
            or parity.get('parser') != 'pandas.read_csv(index_col=0).astype(numpy.float32) on isolated source rows'
            or parity.get('test_semantic_access') is not False):
        raise ValueError('Source-label custodian parity receipt missing or mismatched')
    by_role = {item['role']: item for item in label_receipt['source_roles']}
    axis = {node_id: i for i, node_id in enumerate(node_ids)}
    compact = {}
    for role in ('train', 'val'):
        path = root / (role + '_labels.json')
        descriptor = by_role[role]
        pack = load(path)
        if (descriptor['file'] != path.name or descriptor['sha256'] != sha(path)
                or pack.get('role') != role or pack.get('dataset') != 'tolokers-2'
                or pack.get('split') != 'RL' or pack.get('role_freeze_sha256') != freeze_sha
                or pack.get('source_archive_sha256') != ARCHIVE_SHA):
            raise ValueError('Compact labelpack identity differs')
        records = pack['records']
        ids = [record['node_id'] for record in records]
        if (len(ids) != len(set(ids)) or not set(ids) <= set(actual[role])
                or len(records) != descriptor['nonmissing_count']
                or ids != sorted(ids, key=axis.__getitem__)):
            raise ValueError('Compact source rows are duplicated, reordered or outside official role')
        if any(type(row['target']) is not int or row['target'] not in (0, 1) for row in records):
            raise ValueError('Compact labels are not binary integers')
        if {row['target'] for row in records} != {0, 1}:
            raise ValueError('Source role lacks two classes')
        compact[role] = {'rows': [axis[row['node_id']] for row in records],
                         'labels': [row['target'] for row in records]}
    descriptors = {name: {'sha256': sha(root / name), 'bytes': (root / name).stat().st_size}
                   for name in sorted(INPUT_FILES)}
    return roles, compact, {'schema': 'closed-worker-input-identity-v2', 'files': descriptors,
                            'role_freeze_sha256': freeze_sha, 'test_labels': 'CLOSED'}
