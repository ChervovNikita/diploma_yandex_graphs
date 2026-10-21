"""Verify the 6,460-fit proof by default. Explicit release alone publishes gates."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from remote_backend import (EXPANSION_REL, PACKAGES, ROOTS, WORKERS, exact_write, identity,
                            json_bytes, receipt, require, source_manifest, validate_bindings, verify_file)
from readable_transport import Transport, checked_hash

HERE = Path(__file__).resolve().parent
E = HERE.parent
PACKAGE = HERE.name
BINDINGS = HERE / 'BINDINGS.json'


def load_bindings(path, expected_sha):
    path = Path(path).resolve()
    require(path == BINDINGS, 'Use the separately reviewed frozen BINDINGS.json')
    raw = path.read_bytes()
    require(identity(raw)['sha256'] == expected_sha, 'Binding configuration changed')
    config = json.loads(raw)
    validate_bindings(config)
    for kind, relative in PACKAGES.items():
        source_manifest(E / relative, config['sources'][kind])
    return config, raw


def ready_workers(probes):
    rows = [row for host in ROOTS for row in probes[host]['workers']]
    require(len(rows) == 3 and {row['worker'] for row in rows} == set(WORKERS), 'Wrong probe worker set')
    if any(row['status'] == 'pending' for row in rows):
        return None
    require(all(row['status'] == 'PASS' for row in rows), 'Non-PASS worker')
    rows.sort(key=lambda row: WORKERS.index(row['worker']))
    require(rows[1]['inventory'] == rows[2]['inventory'], 'Server51 workers use different inventories')
    return rows


class Coordinator:
    def __init__(self, config, binding_raw, proof, transport):
        self.config, self.binding_raw = config, binding_raw
        self.proof, self.transport = Path(proof), transport
        self.backend = HERE / 'remote_backend.py'
        self.backend_pin = receipt(self.backend)
        self.binding_pin = identity(binding_raw)

    def deployment(self, host):
        return Path(ROOTS[host]) / EXPANSION_REL / PACKAGE / 'releases' / self.backend_pin['sha256']

    def invoke(self, host, operation='probe', **options):
        folder = self.deployment(host)
        backend, config = folder / 'remote_backend.py', folder / 'BINDINGS.json'
        command = checked_hash(backend, self.backend_pin['sha256']) + checked_hash(config, self.binding_pin['sha256'])
        argv = ['python3', '-B', str(backend), '--host', host, '--config', str(config),
                '--config-sha256', self.binding_pin['sha256'], '--operation', operation]
        for key, value in options.items():
            argv.extend(['--' + key.replace('_', '-'), str(value)])
        command += shlex.join(argv) + '\n'
        raw = self.transport.call(host, command)
        return raw if operation == 'read-json' else json.loads(raw)

    def stage(self):
        staged = {}
        for host in ROOTS:
            folder = self.deployment(host)
            staged[host] = [self.transport.stage_plaintext(host, source, folder / source.name)
                            for source in (self.backend, BINDINGS)]
        return {'status': 'STAGED', 'hosts': staged, 'markers_deployed': 0,
                'full_training_launch_performed': False}

    def probe(self):
        return {host: self.invoke(host) for host in ROOTS}

    def fetch(self, row, kind, destination):
        expected = row[kind]
        if destination.exists():
            verify_file(destination, expected)
            return
        raw = self.invoke(row['worker'].split('/')[0], 'read-json', worker=row['worker'],
                          kind=kind, file_sha256=expected['sha256'])
        require(identity(raw) == {k: expected[k] for k in ('bytes', 'sha256')}, 'Fetched metadata changed')
        json.loads(raw)
        exact_write(destination, raw)

    def verify(self):
        """Only remote read operations. Candidate and validation receipts stay local."""
        exact_write(self.proof / 'BINDINGS.json', self.binding_raw)
        probes = self.probe()
        rows = ready_workers(probes)
        if rows is None:
            return {'status': 'pending', 'workers': probes, 'markers_deployed': 0,
                    'full_training_launch_performed': False}
        exact_write(self.proof / 'READY.json', json_bytes(rows))
        bundle_rows = []
        for row in rows:
            host = row['worker'].split('/')[0]
            inventory = self.proof / ('INVENTORY-' + host + '.json')
            summary = self.proof / ('SMOKE_SUMMARY-' + row['worker'].replace('/', '_') + '.json')
            self.fetch(row, 'inventory', inventory)
            self.fetch(row, 'summary', summary)
            bundle_rows.append({'worker': row['worker'], 'inventory': receipt(inventory), 'summary': receipt(summary)})
        bundle = self.proof / 'BUNDLE.json'
        exact_write(bundle, json_bytes({'workers': bundle_rows}))
        candidate = self.proof / 'CANDIDATE_GLOBAL_SMOKE_PASS.json'
        validation = self.proof / 'VALIDATION.json'
        if validation.exists():
            value = json.loads(validation.read_text())
            require(value['bundle'] == receipt(bundle) and value['validator_source'] == self.config['sources']['smoke'],
                    'Saved validation binding changed')
            verify_file(candidate, value['marker'])
        else:
            output = candidate if not candidate.exists() else self.proof / ('REVALIDATED-' + uuid.uuid4().hex + '.json')
            child = subprocess.run([sys.executable, '-B', str(E / PACKAGES['smoke'] / 'validate.py'),
                                    '--bundle', str(bundle), '--output', str(output)], capture_output=True, text=True)
            require(child.returncode == 0, 'Frozen global validation failed: ' + child.stdout + child.stderr)
            exact_write(candidate, output.read_bytes())
            value = json.loads(candidate.read_text())
            require(value.get('status') == 'PASS' and value.get('fit_count') == 6460, 'Validator did not return exact global PASS')
            require(all(value['source_manifest'][k] == self.config['sources']['smoke'][k] for k in ('bytes', 'sha256')),
                    'Validator source changed during validation')
            exact_write(validation, json_bytes({'bundle': receipt(bundle), 'validator_source': self.config['sources']['smoke'],
                                               'marker': receipt(candidate)}))
        require(ready_workers(self.probe()) == rows, 'Remote metadata changed while collecting proof')
        return {'status': 'VERIFIED', 'fit_count': 6460, 'proof': str(self.proof), 'candidate': receipt(candidate),
                'markers_deployed': 0, 'existing_remote_markers': {host: probes[host]['existing_marker'] for host in ROOTS},
                'full_training_launch_performed': False, 'release_requires_explicit_command': True}

    def release(self, confirmation):
        # Check explicit intent before verification or any remote mutation.
        require(confirmation is not None, 'release requires --confirm-release-sha256 from the verified candidate')
        candidate = self.proof / 'CANDIDATE_GLOBAL_SMOKE_PASS.json'
        require(candidate.exists() and receipt(candidate)['sha256'] == confirmation, 'Release confirmation differs from candidate')
        checked = self.verify()
        require(checked['status'] == 'VERIFIED' and checked['candidate']['sha256'] == confirmation, 'Complete current proof required')
        rows = json.loads((self.proof / 'READY.json').read_text())
        deployed = {}
        for host in ROOTS:
            folder = self.deployment(host) / 'proofs' / confirmation
            marker = self.transport.stage_plaintext(host, candidate, folder / candidate.name)
            request = {'schema': 'all-fit-readable-release-request-v1', 'bindings': self.binding_pin,
                       'ready_workers': rows, 'marker': {k: marker[k] for k in ('path', 'bytes', 'sha256')}}
            request_path = self.proof / ('RELEASE_REQUEST-' + host + '.json')
            exact_write(request_path, json_bytes(request))
            request_pin = self.transport.stage_plaintext(host, request_path, folder / request_path.name)
            result = self.invoke(host, 'release', release_request=request_pin['path'],
                                 release_request_sha256=request_pin['sha256'], confirm_release_sha256=confirmation)
            exact_write(self.proof / ('DELIVERED-' + host + '.json'), json_bytes({'marker': result['marker']}))
            deployed[host] = result
        return {'status': 'RELEASED', 'hosts': deployed, 'proof': str(self.proof),
                'full_training_launch_performed': False, 'waiting_workers_may_continue': True}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('operation', choices=('probe', 'verify', 'stage', 'release'), nargs='?', default='verify')
    result.add_argument('--bindings', default=str(BINDINGS))
    result.add_argument('--bindings-sha256', required=True)
    result.add_argument('--source-sha256', required=True, help='Reviewed coordinator MANIFEST.json SHA256')
    result.add_argument('--confirm-release-sha256')
    return result


def main():
    args = parser().parse_args()
    require(args.operation == 'release' or args.confirm_release_sha256 is None, 'Confirmation is only accepted by explicit release')
    own_manifest = receipt(HERE / 'MANIFEST.json')
    require(own_manifest['sha256'] == args.source_sha256, 'Coordinator source manifest changed')
    source_manifest(HERE, own_manifest)
    config, raw = load_bindings(args.bindings, args.bindings_sha256)
    proof = HERE / 'proofs' / args.bindings_sha256
    proof.mkdir(parents=True, exist_ok=True)
    attempt = proof / ('ATTEMPT-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8] + '.json')
    with (proof / 'COORDINATOR.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coordinator = Coordinator(config, raw, proof, Transport(proof / 'commands'))
        try:
            if args.operation == 'stage':
                result = coordinator.stage()
            elif args.operation == 'probe':
                result = {'status': 'PROBED', 'hosts': coordinator.probe(), 'markers_deployed': 0,
                          'full_training_launch_performed': False}
            elif args.operation == 'release':
                result = coordinator.release(args.confirm_release_sha256)
            else:
                result = coordinator.verify()
        except Exception as exc:
            result = {'status': 'ERROR', 'operation': args.operation, 'error': type(exc).__name__ + ': ' + str(exc),
                      'proof': str(proof), 'prior_outputs_preserved': True, 'full_training_launch_performed': False,
                      'inspect_delivery_receipts_before_retry': args.operation == 'release'}
            exact_write(attempt, json_bytes(result))
            print(json.dumps(result, sort_keys=True))
            return 1
        exact_write(attempt, json_bytes(result))
        print(json.dumps(result, sort_keys=True))
        return 0


if __name__ == '__main__':
    sys.exit(main())
