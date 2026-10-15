"""Offline, explicit JSON-field relocation; no data reads or source string replacement."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from host_binding import normalized_absolute, require

RUNTIME_FIELDS = (
    '/callback_module', '/data_root', '/recipe/_repository_root', '/recipe/_redesign_root',
    '/recipe/redesign_root', '/link_policy_module', '/source_rng_emulation/audit',
    '/session_transform/path', '/extension_factory/path', '/session_extension/path',
    '/method_settings/context_manifest', '/recipe/_native_callback',
    '/parent_compiler_descriptor/path',
)
DATA_FIELDS = ('graph', 'train', 'valid', 'test')
PROVENANCE = {'expanded_contract', 'descriptor_preparation', 'source_plan_entry'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def leaves(value, path=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, path + '/' + key.replace('~', '~0').replace('/', '~1'))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, path + '/' + str(i))
    else:
        yield path, value


def is_provenance(pointer):
    return pointer.split('/')[1] in PROVENANCE


def allowed(pointer):
    return pointer in RUNTIME_FIELDS or pointer in tuple('/recipe/data_files/' + key for key in DATA_FIELDS)


def set_field(value, pointer, replacement):
    fields = [v.replace('~1', '/').replace('~0', '~') for v in pointer.lstrip('/').split('/')]
    for key in fields[:-1]:
        value = value[int(key)] if isinstance(value, list) else value[key]
    value[fields[-1]] = replacement



def parent_compiler_authority(job, root):
    """Exact engineering parent JSON receipt; never a whole-subtree exemption."""
    if 'parent_compiler_descriptor' not in job:
        require('parent_runtime_job_hash' not in job, 'Parent canonical hash requires its explicit descriptor receipt')
        return None
    require(job.get('schema') == 'native-full-epoch-fit-v1' and job.get('engineering_only') is True,
            'Parent compiler authority is an explicit engineering descriptor field')
    value = job['parent_compiler_descriptor']
    require(isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'},
            'Exact parent descriptor path/bytes/sha256 receipt required')
    path = normalized_absolute(value['path'])
    require(path.is_relative_to(normalized_absolute(root)) and path.suffix == '.json',
            'Parent authority must be a confined project JSON descriptor')
    require(type(value['bytes']) is int and value['bytes'] > 0, 'Positive parent descriptor byte count required')
    for field, checksum in (('parent descriptor SHA256', value['sha256']),
                            ('parent canonical job hash', job.get('parent_runtime_job_hash'))):
        require(isinstance(checksum, str) and len(checksum) == 64 and all(c in '0123456789abcdef' for c in checksum),
                'Exact ' + field + ' required')
    return value


def audit_runtime_paths(job, root, phase_relative):
    root = normalized_absolute(root)
    phase = root / phase_relative
    parent_compiler_authority(job, str(root))
    rows = []
    for pointer, value in leaves(job):
        if is_provenance(pointer) or not isinstance(value, str) or not value.startswith('/'):
            continue
        require(allowed(pointer), 'Unclassified absolute runtime field: ' + pointer)
        path = normalized_absolute(value)
        require(path.is_relative_to(root), 'Runtime path escapes host root: ' + pointer)
        rows.append({'pointer': pointer, 'path': value})
    require(job['recipe']['_repository_root'] == str(root), 'Recipe repository root must match host')
    require(job['recipe']['_redesign_root'] == str(phase / 'reference_wrapper_redesign_20261013_v1'), 'Recipe redesign root must match declared phase')
    for field in ('callback_module', 'data_root'):
        require(PurePosixPath(job[field]).is_absolute(), 'Absolute explicit runtime field required: ' + field)
    return rows


def rebind_job(original, source_root, source_phase_relative, target_root, target_phase_relative):
    source_root, target_root = normalized_absolute(source_root), normalized_absolute(target_root)
    for phase in (source_phase_relative, target_phase_relative):
        require(not PurePosixPath(phase).is_absolute() and '..' not in PurePosixPath(phase).parts, 'Safe relative phase required')
    source_phase, target_phase = source_root / source_phase_relative, target_root / target_phase_relative
    audit_runtime_paths(original, str(source_root), source_phase_relative)
    bound, changes = copy.deepcopy(original), []
    for pointer, value in leaves(original):
        if is_provenance(pointer) or not allowed(pointer) or not isinstance(value, str):
            continue
        path = normalized_absolute(value)
        # Explicit scope mapping: phase may have a different repo-relative layout.
        replacement = str(target_phase / path.relative_to(source_phase)) if path.is_relative_to(source_phase) else str(target_root / path.relative_to(source_root))
        if replacement != value:
            set_field(bound, pointer, replacement)
            changes.append({'pointer': pointer, 'from': value, 'to': replacement})
    audit_runtime_paths(bound, str(target_root), target_phase_relative)
    for key in PROVENANCE:
        require(bound.get(key) == original.get(key), 'Scientific/provenance contract changed')
    if original.get('expanded_contract'):
        require(digest(original['expanded_contract']) == original['expanded_contract_hash'] == bound['expanded_contract_hash'], 'Expanded scientific identity mismatch')
    return bound, {'schema': 'explicit-runtime-descriptor-path-binding-v1', 'id': original['id'],
        'canonical_runtime_job_hash': digest(original), 'bound_runtime_job_hash': digest(bound),
        'science_contract_hash': original.get('expanded_contract_hash'), 'field_changes': changes,
        'source_hashes_or_data_receipts_rewritten': False, 'scientific_fields_changed': False,
        'destination_source_or_data_bytes_verified': False, 'launch_admitted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'source-root', 'source-phase-relative', 'target-root', 'target-phase-relative', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    bound, receipt = rebind_job(job, args.source_root, args.source_phase_relative, args.target_root, args.target_phase_relative)
    with Path(args.output).open('x') as handle:
        json.dump({'candidate_job': bound, 'binding_receipt': receipt}, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(receipt, sort_keys=True))
