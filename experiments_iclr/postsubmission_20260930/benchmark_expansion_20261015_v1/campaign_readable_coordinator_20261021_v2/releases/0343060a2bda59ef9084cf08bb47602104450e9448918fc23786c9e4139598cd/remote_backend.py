"""Pinned plaintext metadata backend. Only explicit release writes the barrier."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

EXPANSION_REL = 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
PACKAGES = {'port': 'server51_campaign_port_v6', 'execution': 'campaign_execution_20261016_v7',
            'smoke': 'qualification/all_fit_smoke_20261016_v6'}
ROOTS = {'allocation': '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs',
         'server51': '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'}
HOSTS = {'allocation': ('anogena-2-0', ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']),
         'server51': ('olig', ['GPU-9c6c20fa-a620-479d-f72f-163ac74deef2',
                               'GPU-30283973-140d-9e7b-e780-0fcc487dec54'])}
WORKERS = ('allocation/gpu0', 'server51/gpu0', 'server51/gpu1')


def require(value, message):
    if not value:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def receipt(path):
    path = Path(path).resolve()
    return {'path': str(path), **identity(path.read_bytes())}


def verify_file(path, expected):
    actual = receipt(path)
    require(all(actual[k] == expected[k] for k in ('bytes', 'sha256')), 'Changed receipt: ' + str(path))
    return Path(path)


def exact_write(path, raw, mode=0o444):
    """Keep prior evidence byte for byte, including interrupted deliveries."""
    path = Path(path)
    require(not path.is_symlink(), 'Evidence path may not be a symlink')
    if path.exists():
        require(path.read_bytes() == raw, 'Existing evidence differs: ' + str(path))
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part-' + uuid.uuid4().hex)
    with temp.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp, mode)
    created = True
    try:
        os.link(temp, path)
    except FileExistsError:
        require(path.read_bytes() == raw, 'Concurrent evidence differs: ' + str(path))
        created = False
    finally:
        temp.unlink()
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return created


def source_manifest(folder, expected):
    folder = Path(folder).resolve()
    manifest = verify_file(folder / 'MANIFEST.json', expected)
    for row in json.loads(manifest.read_text())['files']:
        member = (folder / row['path']).resolve()
        require(member.is_relative_to(folder), 'Source member escapes package')
        verify_file(member, row)


def validate_bindings(config):
    require(config.get('schema') == 'all-fit-global-smoke-coordinator-bindings-v1', 'Wrong binding schema')
    require(set(config['sources']) == set(PACKAGES), 'Bind all three final source manifests')
    rows = config['workers']
    require(len(rows) == 3 and {row['worker'] for row in rows} == set(WORKERS), 'Bind exactly three workers')
    for row in rows:
        host = row['worker'].split('/')[0]
        expected = Path(ROOTS[host]) / EXPANSION_REL / PACKAGES['execution'] / ('PLAN-' + row['worker'].replace('/', '_') + '.json')
        require(row['plan']['path'] == str(expected), 'Plan is not the frozen execution7 worker plan')
        require(type(row['plan']['bytes']) is int and row['plan']['bytes'] > 0 and
                isinstance(row['plan']['sha256'], str) and len(row['plan']['sha256']) == 64, 'Exact plan receipt required')


def host_guard(host):
    hostname, gpu_uuids = HOSTS[host]
    require(socket.gethostname() == hostname, 'Wrong host: excluded route attempt')
    actual = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    require(actual == gpu_uuids, 'Wrong complete GPU UUID list: excluded route attempt')


class Backend:
    def __init__(self, host, config, root=None, binding_identity=None):
        self.host = host
        self.config = config
        self.binding_identity = binding_identity or identity(json_bytes(config))
        self.root = Path(ROOTS[host] if root is None else root)
        require(self.root.resolve() == self.root, 'Project root must be literal and resolved')
        self.expansion = self.root / EXPANSION_REL
        self.queue = self.expansion / PACKAGES['execution']

    def inside(self, value):
        path = Path(value).resolve()
        require(path.is_relative_to(self.root), 'Path escapes authorized project')
        return path

    def verify(self, path, expected):
        return verify_file(self.inside(path), expected)

    def probe(self):
        for kind, relative in PACKAGES.items():
            folder = self.inside(self.expansion / relative)
            source_manifest(folder, self.config['sources'][kind])
        result = []
        for entry in self.config['workers']:
            worker = entry['worker']
            if worker.split('/')[0] != self.host:
                continue
            plan_path = self.verify(entry['plan']['path'], entry['plan'])
            plan = json.loads(plan_path.read_text())
            require(plan['worker'] == worker and plan['project_root'] == str(self.root), 'Wrong active worker/root')
            require(plan['source_manifest'] == receipt(self.queue / 'MANIFEST.json'), 'Wrong execution source')
            require(plan['smoke_source_manifest'] == receipt(self.expansion / PACKAGES['smoke'] / 'MANIFEST.json'), 'Wrong smoke source')
            require(plan['reporting_entrypoint'] == receipt(self.expansion / PACKAGES['port'] / 'reporting/runtime_entry.py'), 'Wrong port source')
            state = self.queue / 'runs' / worker.replace('/', '_')
            require(plan['state_root'] == str(state) and plan['global_smoke_pass'] == str(self.queue / 'GLOBAL_SMOKE_PASS.json'), 'Wrong state/barrier path')
            owner, summary = self.inside(state / 'OWNER.json'), self.inside(state / 'SMOKE_SUMMARY.json')
            if not owner.exists() or not summary.exists():
                result.append({'worker': worker, 'status': 'pending'})
                continue
            own = json.loads(owner.read_text())
            require(own['plan'] == receipt(plan_path) and own['source_manifest'] == plan['source_manifest'] and
                    own['inventory'] == plan['inventory'], 'Owner source/plan/inventory differs')
            summary_pin = receipt(summary)
            value = json.loads(summary.read_text())
            require(value.get('schema') in ('allfit-worker-smoke-summary-v1', 'all-fit-worker-smoke-summary-v1'), 'Wrong summary schema')
            require(value.get('status') == 'PASS' and value.get('worker') == worker, 'Existing summary is not worker PASS')
            require(value['inventory'] == plan['inventory'], 'Summary/plan inventory differs')
            inventory_pin = receipt(self.verify(plan['inventory']['path'], plan['inventory']))
            result.append({'worker': worker, 'status': 'PASS', 'plan': receipt(plan_path),
                           'inventory': inventory_pin, 'summary': summary_pin})
        return sorted(result, key=lambda row: row['worker'])

    def read_json(self, worker, kind, expected_sha):
        require(kind in ('inventory', 'summary'), 'Only inventory and summary data may be fetched')
        rows = self.probe()
        matches = [row for row in rows if row['worker'] == worker and row['status'] == 'PASS']
        require(len(matches) == 1, 'Worker has no passing bound metadata')
        expected = matches[0][kind]
        require(expected['sha256'] == expected_sha, 'Requested data identity changed')
        raw = self.inside(expected['path']).read_bytes()
        require(identity(raw) == {k: expected[k] for k in ('bytes', 'sha256')}, 'Metadata changed during read')
        json.loads(raw)
        return raw

    def release(self, request_path, request_sha, confirmation):
        """The sole canonical marker write. Its caller must explicitly select release."""
        request_path = self.inside(request_path)
        require(receipt(request_path)['sha256'] == request_sha, 'Release request changed')
        request = json.loads(request_path.read_text())
        require(request.get('schema') == 'all-fit-readable-release-request-v1', 'Wrong release request schema')
        expected = request['marker']
        require(confirmation == expected['sha256'], 'Explicit exact-marker release confirmation required')
        require(request['bindings'] == self.binding_identity, 'Release bindings differ')
        rows = self.probe()
        require(rows == [row for row in request['ready_workers'] if row['worker'].split('/')[0] == self.host],
                'Source, plan, inventory or summary changed before release')
        candidate = self.verify(expected['path'], expected)
        require(candidate.name == 'CANDIDATE_GLOBAL_SMOKE_PASS.json', 'Release requires a separate candidate file')
        raw = candidate.read_bytes()
        require(identity(raw) == {k: expected[k] for k in ('bytes', 'sha256')}, 'Candidate changed during release')
        # Import the source-pinned execution7 checker through its ordinary module path.
        # This function reads metadata only and has exactly the waiting workers' semantics.
        sys.path.insert(0, str(self.queue))
        from common import global_barrier
        for row in rows:
            plan = json.loads(self.verify(row['plan']['path'], row['plan']).read_text())
            require(global_barrier(candidate, plan, row['summary']), 'Execution7 rejected candidate')
        require(self.probe() == rows, 'Bindings changed during release validation')
        marker = self.inside(self.queue / 'GLOBAL_SMOKE_PASS.json')
        created = exact_write(marker, raw)
        return {'marker': receipt(marker), 'existing_identical': not created,
                'full_training_launch_performed': False, 'waiting_workers_may_continue': True}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--host', choices=ROOTS, required=True)
    result.add_argument('--config', required=True)
    result.add_argument('--config-sha256', required=True)
    result.add_argument('--operation', choices=('probe', 'read-json', 'release'), default='probe')
    result.add_argument('--worker', choices=WORKERS)
    result.add_argument('--kind', choices=('inventory', 'summary'))
    result.add_argument('--file-sha256')
    result.add_argument('--release-request')
    result.add_argument('--release-request-sha256')
    result.add_argument('--confirm-release-sha256')
    return result


def main():
    args = parser().parse_args()
    host_guard(args.host)  # Before any project metadata read.
    root = Path(ROOTS[args.host])
    config_path = Path(args.config).resolve()
    require(config_path.is_relative_to(root), 'Configuration escapes authorized project')
    raw = config_path.read_bytes()
    require(identity(raw)['sha256'] == args.config_sha256, 'Binding configuration changed')
    config = json.loads(raw)
    validate_bindings(config)
    backend = Backend(args.host, config, binding_identity=identity(raw))
    if args.operation == 'read-json':
        sys.stdout.buffer.write(backend.read_json(args.worker, args.kind, args.file_sha256))
        return
    if args.operation == 'release':
        require(args.release_request and args.release_request_sha256 and args.confirm_release_sha256,
                'Explicit release request and confirmation required')
        output = backend.release(args.release_request, args.release_request_sha256, args.confirm_release_sha256)
    else:
        marker = backend.inside(backend.queue / 'GLOBAL_SMOKE_PASS.json')
        output = {'workers': backend.probe(), 'existing_marker': receipt(marker) if marker.exists() else None,
                  'markers_deployed': 0, 'full_training_launch_performed': False}
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    main()
