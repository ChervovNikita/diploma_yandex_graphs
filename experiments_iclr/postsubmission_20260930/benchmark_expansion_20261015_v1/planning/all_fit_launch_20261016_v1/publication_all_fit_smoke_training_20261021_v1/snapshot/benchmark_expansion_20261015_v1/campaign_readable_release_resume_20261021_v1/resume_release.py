"""Resume the exact v2 release with bounded plaintext server51 transfer commands."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'campaign_readable_coordinator_20261021_v2'
V2_SHA = 'e07e6e9fa7f9dc4b13f42bbbbc64cbf93639c404b92f00a21535042ca51033b4'
BINDINGS_SHA = 'b605b36e90fe86b7b58cbb366918ea7c7efdba331a33fd897962696d4f4eda96'
CANDIDATE_SHA = '065b2574147b257ab7aa47163eaaa475f7b8a0f6df1f4a6378b4a43f30d8c4f4'
VALIDATION_SHA = 'fc818330542469277702adba76a3f9e1e66eb4857714fe13942dcd16228b86a7'
PROOF = V2 / 'proofs' / BINDINGS_SHA
CHUNK_BYTES = 8000


def check_package(folder, expected):
    """Check plaintext dependencies before importing them."""
    raw = (folder / 'MANIFEST.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen source manifest changed: ' + str(folder))
    for row in json.loads(raw)['files']:
        path = (folder / row['path']).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Source escapes package')
        content = path.read_bytes()
        if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError('Frozen source changed: ' + str(path))


check_package(V2, V2_SHA)
sys.path.insert(0, str(V2))
import coordinator as frozen
import readable_transport as route
from remote_backend import ROOTS, exact_write, identity, json_bytes, receipt, require


class SmallTransport(route.Transport):
    def stage_plaintext(self, host, source, destination):
        if host == 'allocation':
            return super().stage_plaintext(host, source, destination)
        require(host == 'server51', 'Unknown route')
        source, destination = Path(source), Path(destination)
        raw = source.read_bytes()
        text = raw.decode('utf-8')
        require(raw.endswith(b'\n'), 'Plaintext staging requires a final newline')
        require(destination.is_relative_to(Path(ROOTS[host])), 'Staging escapes authorized project')
        # Every attempt has a new temporary path. Prior incomplete transfers remain.
        temporary = destination.with_name(destination.name + '.incoming-' + uuid.uuid4().hex)
        parent, temp, target = map(lambda p: shlex.quote(str(p)), (destination.parent, temporary, destination))

        def send(command):
            require(len((route.guard_shell(host) + command).encode()) <= 12000,
                    'Plaintext command exceeds the bounded SSH mux argument size')
            return self.call(host, command)

        prepare = ('test "$(realpath -m -- ' + parent + ')" = ' + parent + '\n' +
                   'mkdir -p -- ' + parent + '\n' + 'test ! -e ' + temp + '\n' + 'test ! -L ' + temp + '\n')
        send(prepare)
        blocks, block = [], ''
        for line in text.splitlines(keepends=True):
            require(len(line.encode()) <= CHUNK_BYTES, 'Plaintext line exceeds 8000-byte bound')
            if block and len((block + line).encode()) > CHUNK_BYTES:
                blocks.append(block)
                block = ''
            block += line
        if block:
            blocks.append(block)
        offset = 0
        for block in blocks:
            delimiter = 'READABLE_FILE_' + uuid.uuid4().hex
            require(delimiter not in block.splitlines(), 'Heredoc delimiter collision')
            if offset == 0:
                prefix = 'set -C\ncat > ' + temp
            else:
                prefix = 'test "$(wc -c < ' + temp + ')" -eq ' + str(offset) + '\ncat >> ' + temp
            send(prefix + " <<'" + delimiter + "'\n" + block + delimiter + '\n')
            offset += len(block.encode())
        finalize = (route.checked_hash(temporary, identity(raw)['sha256']) + 'test ! -L ' + target + '\n' +
                    'if test -e ' + target + '; then\n  cmp -- ' + temp + ' ' + target + '\nelse\n' +
                    '  chmod 444 -- ' + temp + '\n  ln -- ' + temp + ' ' + target + '\nfi\n' +
                    'rm -f -- ' + temp + '\n' + route.checked_hash(destination, identity(raw)['sha256']))
        send(finalize)
        staged = send('cat -- ' + target + '\n')
        require(staged == raw, 'Staged plaintext differs from reviewed local source')
        return {'path': str(destination), **identity(raw), 'plaintext_readback_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--confirm-release-sha256', required=True)
    args = parser.parse_args()
    check_package(HERE, args.source_sha256)
    require(args.confirm_release_sha256 == CANDIDATE_SHA, 'Only the existing exact release candidate is permitted')
    require(receipt(PROOF / 'CANDIDATE_GLOBAL_SMOKE_PASS.json')['sha256'] == CANDIDATE_SHA, 'Existing candidate changed')
    require(receipt(PROOF / 'VALIDATION.json')['sha256'] == VALIDATION_SHA, 'Existing completed validation changed')
    delivered = json.loads((PROOF / 'DELIVERED-allocation.json').read_text())
    require(delivered['marker']['sha256'] == CANDIDATE_SHA, 'Previously delivered allocation marker differs')
    config, raw = frozen.load_bindings(frozen.BINDINGS, BINDINGS_SHA)
    attempt = PROOF / ('RESUME_RELEASE-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8] + '.json')
    with (PROOF / 'COORDINATOR.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        instance = frozen.Coordinator(config, raw, PROOF, SmallTransport(PROOF / 'commands'))
        try:
            result = instance.release(args.confirm_release_sha256)
        except Exception as exc:
            result = {'status': 'ERROR', 'error': type(exc).__name__ + ': ' + str(exc),
                      'proof': str(PROOF), 'prior_partial_transfers_preserved': True,
                      'inspect_delivery_receipts_before_retry': True, 'full_training_launch_performed': False}
            exact_write(attempt, json_bytes(result))
            print(json.dumps(result, sort_keys=True))
            return 1
        exact_write(attempt, json_bytes(result))
        print(json.dumps(result, sort_keys=True))
        return 0


if __name__ == '__main__':
    sys.exit(main())
