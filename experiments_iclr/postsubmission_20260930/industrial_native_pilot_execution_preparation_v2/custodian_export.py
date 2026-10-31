"""UNEXECUTED future custodian, never mounted/executed as the scientific worker.

Separate public and labels invocations ensure official role bytes exist before
any target member opens. Input archive stays on the custodian side. Actual
worker isolation is supplied by launch_isolated.py, not receipt booleans.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
from closed_inputs import ARCHIVE_SHA, PUBLIC, bool_token, csv_rows, sha, write_json
from runtime_fingerprint import verify_source_packet


def archive(path):
    data = Path(path).read_bytes()
    if len(data) != 3361114 or hashlib.sha256(data).hexdigest() != ARCHIVE_SHA:
        raise ValueError('Archive differs from saved acquired identity')
    zip_file = zipfile.ZipFile(io.BytesIO(data))
    if len(zip_file.namelist()) != len(set(zip_file.namelist())):
        raise ValueError('Duplicate ZIP member names')
    return zip_file


def freeze_public(archive_path, output, policy):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    public = output / 'tolokers-2'
    public.mkdir(mode=0o700)
    descriptors = {}
    with archive(archive_path) as zip_file:
        for name in PUBLIC:
            data = zip_file.read('tolokers-2/' + name)
            path = public / name
            path.write_bytes(data)
            os.chmod(path, 0o600)
            descriptors[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        # targets.csv is never opened in the public phase.
    features = csv_rows(public / 'features.csv')
    masks = csv_rows(public / 'split_masks_RL.csv')
    ids = [row[0] for row in features[1:]]
    if not ids or len(ids) != len(set(ids)) or [row[0] for row in masks[1:]] != ids:
        raise ValueError('Feature/official-mask row axis differs')
    columns = {role: masks[0].index(role) for role in ('train', 'val', 'test')}
    official = {role: [] for role in columns}
    unassigned = []
    for node_id, row in zip(ids, masks[1:]):
        bits = {role: bool_token(row[index]) for role, index in columns.items()}
        if sum(bits.values()) > 1:
            raise ValueError('Official roles overlap')
        if not any(bits.values()):
            unassigned.append(node_id)
        else:
            official[next(role for role, bit in bits.items() if bit)].append(node_id)
    write_json(output / 'ROLE_FREEZE.json', {
        'schema': 'tolokers2-public-role-freeze-v1', 'dataset': 'tolokers-2', 'split': 'RL',
        'source_archive_sha256': ARCHIVE_SHA, 'role_policy_sha256': sha(policy),
        'node_ids': ids, 'official_role_ids': official, 'unassigned_node_ids': unassigned,
        'public_members': descriptors, 'targets_member_opened': False, 'test_labels': 'CLOSED',
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    os.chmod(output / 'ROLE_FREEZE.json', 0o600)


def source_labels(archive_path, output, policy, isolation_probe):
    """Parse source-only buffered CSV with exact native pandas float32 semantics."""
    output = Path(output)
    role_path = output / 'ROLE_FREEZE.json'
    roles = json.loads(role_path.read_text())
    freeze_sha = sha(role_path)
    if (roles['role_policy_sha256'] != sha(policy) or roles['source_archive_sha256'] != ARCHIVE_SHA
            or roles['targets_member_opened'] is not False or roles['test_labels'] != 'CLOSED'):
        raise ValueError('Public role freeze must precede source targets')
    probe = json.loads(Path(isolation_probe).read_text())
    if (probe.get('schema') != 'observed-public-only-worker-isolation-probe-v2'
            or probe.get('status') != 'PASSED' or probe.get('scientific_imports_or_updates') is not False
            or probe.get('source_identity') != verify_source_packet(Path(__file__).resolve().parent)
            or probe.get('public_input_identity', {}).get('role_freeze_sha256') != freeze_sha):
        raise ValueError('Actual public-only worker isolation observation must precede source targets')
    for name, descriptor in roles['public_members'].items():
        path = output / 'tolokers-2' / name
        if sha(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
            raise ValueError('Frozen public bytes changed')
        if probe['public_input_identity']['files']['tolokers-2/' + name] != descriptor:
            raise ValueError('Isolation probe used different public bytes')
    for name in ('train_labels.json', 'val_labels.json', 'LABELPACK_RECEIPT.json', 'SOURCE_LABEL_PARITY.json'):
        if (output / name).exists():
            raise ValueError('Do not overwrite an earlier source-label attempt')
    source_role = {node_id: role for role in ('train', 'val')
                   for node_id in roles['official_role_ids'][role]}
    source_rows = {'train': [], 'val': []}
    expected = iter(roles['node_ids'])
    with archive(archive_path) as zip_file, zip_file.open('tolokers-2/targets.csv') as file:
        reader = csv.reader(io.TextIOWrapper(file, encoding='utf-8-sig', newline=''))
        header = next(reader)
        if len(header) != 2:
            raise ValueError('Native binary target file must have index and one target')
        for row in reader:
            node_id = next(expected, None)
            if node_id is None or len(row) != 2 or row[0] != node_id:
                raise ValueError('Target syntax/row axis differs from frozen public rows')
            role = source_role.get(node_id)
            if role is None:
                continue  # Skip test/unassigned before semantic target parsing.
            source_rows[role].append(row)
        if next(expected, None) is not None:
            raise ValueError('Missing target rows')
    import numpy as np
    import pandas as pd
    packs = {}
    summaries = {}
    for role, rows in source_rows.items():
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer)
        writer.writerow(header)
        writer.writerows(rows)
        buffer.seek(0)
        # Exact native read/cast, but the isolated buffer contains source rows only.
        values = pd.read_csv(buffer, index_col=0).astype(np.float32).values.reshape(-1)
        if len(values) != len(rows):
            raise ValueError('Native source-row parser changed row cardinality')
        records = []
        for row, value in zip(rows, values):
            if np.isnan(value):
                continue
            if not np.isfinite(value) or value not in (0., 1.):
                raise ValueError('Source target is outside known binary task')
            records.append({'node_id': row[0], 'target': int(value)})
        if {record['target'] for record in records} != {0, 1}:
            raise ValueError('Source role lacks two classes')
        packs[role] = {'schema': 'compact-source-labelpack-v1', 'role': role,
                       'dataset': 'tolokers-2', 'split': 'RL', 'role_freeze_sha256': freeze_sha,
                       'source_archive_sha256': ARCHIVE_SHA, 'records': records}
        summaries[role] = {'official_count': len(rows), 'nonmissing_count': len(records),
                           'missing_count': len(rows) - len(records)}
    descriptors = []
    for role, pack in packs.items():
        path = output / (role + '_labels.json')
        write_json(path, pack)
        os.chmod(path, 0o600)
        descriptors.append({'role': role, 'file': path.name, 'sha256': sha(path),
                            'nonmissing_count': len(pack['records'])})
    receipt_path = output / 'LABELPACK_RECEIPT.json'
    write_json(receipt_path, {'source_roles': descriptors, 'role_freeze_sha256': freeze_sha,
                             'isolation_probe_receipt_sha256': sha(isolation_probe),
                             'test_values_semantically_decoded': False, 'test_statistics_emitted': False})
    write_json(output / 'SOURCE_LABEL_PARITY.json', {
        'schema': 'native-source-label-parser-receipt-v2', 'passed': True,
        'parser': 'pandas.read_csv(index_col=0).astype(numpy.float32) on isolated source rows',
        'comparison': 'Emitted compact values and nonmissing masks constructed directly from native source-role parsing.',
        'role_freeze_sha256': freeze_sha, 'labelpack_receipt_sha256': sha(receipt_path),
        'source_archive_sha256': ARCHIVE_SHA, 'source_role_summaries': summaries,
        'pandas_version': pd.__version__, 'numpy_version': np.__version__,
        'test_semantic_access': False, 'worker_isolation_claim': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('public', 'labels'))
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--isolation-probe', help='Required observed public-only worker probe receipt for labels phase')
    args = parser.parse_args()
    policy = Path(__file__).resolve().parent / 'ROLE_POLICY.json'
    if args.phase == 'public':
        if args.isolation_probe:
            raise ValueError('Public role freeze precedes the isolation probe')
        freeze_public(args.archive, args.output, policy)
    else:
        if not args.isolation_probe:
            raise ValueError('Labels phase requires the actual public-only worker isolation-probe receipt')
        source_labels(args.archive, args.output, policy, args.isolation_probe)


if __name__ == '__main__':
    main()
