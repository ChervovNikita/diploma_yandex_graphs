"""One guarded, read-only compiler4 JSON scan; no dataset/tensor/model imports."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import resource
import socket
import subprocess
import time


def main():
    started = time.monotonic()
    hostname = socket.gethostname()
    if hostname != 'anogena-2-0':
        raise ValueError('Wrong allocation hostname; no project reads')
    observed = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    if observed != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Wrong allocation sole GPU UUID; no project reads')
    root = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
    expansion = root / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
    def read_pin(path, sha):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('Pinned compiler metadata changed: ' + str(path))
        return json.loads(raw)
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    ready = read_pin(expansion / 'expanded_job_binding_v1/generated_v4/READINESS.json',
                     '80c1084e1d5f07c9df92bbee4933a269910d1b2bb8637fe43fe325e7c34c3fd7')
    index_pin = ready['artifacts']['index']
    if index_pin['sha256'] != '485be18f27a0cdd17e3d6a923e1dc1a46deefe580cf0563373070c4b9fe4bdcb':
        raise ValueError('Wrong compiler4 full index')
    rows = read_pin(index_pin['path'], index_pin['sha256'])['fits']
    if len(rows) != len({r['id'] for r in rows}) or len(rows) != 6460:
        raise ValueError('Exact6460 unique compiler4 fits required')
    allowed = {'/callback_module', '/data_root', '/recipe/_repository_root', '/recipe/_redesign_root',
               '/recipe/redesign_root', '/link_policy_module', '/source_rng_emulation/audit',
               '/session_transform/path', '/extension_factory/path', '/session_extension/path',
               '/method_settings/context_manifest', '/recipe/data_files/graph', '/recipe/data_files/train',
               '/recipe/data_files/valid', '/recipe/data_files/test'}
    provenance = {'expanded_contract', 'descriptor_preparation', 'source_plan_entry'}
    pointers = {}
    violations = []
    def leaves(value, pointer=''):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, pointer + '/' + key.replace('~', '~0').replace('/', '~1'))
        elif isinstance(value, list):
            for number, child in enumerate(value):
                yield from leaves(child, pointer + '/' + str(number))
        else:
            yield pointer, value
    for row in rows:
        descriptor = row['descriptor']
        path = Path(descriptor['path'])
        if not path.is_relative_to(root) or path.suffix != '.json':
            raise ValueError('Only project JSON descriptors may be opened')
        job = read_pin(path, descriptor['sha256'])
        if digest(job) != row['runtime_job_hash'] or job['id'] != row['id']:
            raise ValueError('Descriptor job identity changed')
        if digest(job['expanded_contract']) != row['expanded_contract_hash']:
            raise ValueError('Descriptor science identity changed')
        for pointer, value in leaves(job):
            if not isinstance(value, str) or not value.startswith('/'):
                continue
            role = 'immutable_provenance' if pointer.split('/')[1] in provenance else 'classified_runtime' if pointer in allowed else 'unclassified_runtime'
            record = pointers.setdefault(pointer, {'role': role, 'count': 0, 'examples': []})
            record['count'] += 1
            if len(record['examples']) < 3:
                record['examples'].append({'id': row['id'], 'value': value, 'descriptor': descriptor})
            if role != 'immutable_provenance':
                target = PurePosixPath(value)
                if str(target) != value or '..' in target.parts or not target.is_relative_to(root):
                    violations.append({'id': row['id'], 'pointer': pointer, 'value': value})
    result = {'schema': 'compiler4-full-runtime-path-schema-audit-v1', 'status': 'complete',
              'hostname': hostname, 'GPU_UUIDs': observed, 'fits_scanned': len(rows),
              'compiler_readiness_sha256': '80c1084e1d5f07c9df92bbee4933a269910d1b2bb8637fe43fe325e7c34c3fd7',
              'compiler_index': index_pin, 'all_descriptor_file_job_and_science_hashes_verified': True,
              'absolute_runtime_and_provenance_pointers': dict(sorted(pointers.items())),
              'unclassified_runtime_pointers': {k: v for k, v in sorted(pointers.items()) if v['role'] == 'unclassified_runtime'},
              'runtime_path_normalization_or_confinement_violations': violations,
              'elapsed_seconds': time.monotonic() - started, 'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'descriptor_read_passes': 1, 'projection_writes': 0, 'native_jobs_started': 0,
              'tensor_or_dataset_files_opened': 0, 'server77_accessed': False}
    print('COMPILER4_SCHEMA_AUDIT_JSON=' + json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
