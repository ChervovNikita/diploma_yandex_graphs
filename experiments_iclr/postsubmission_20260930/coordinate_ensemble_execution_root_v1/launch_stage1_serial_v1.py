"""Run the three frozen validation-only batches serially on the authorized route."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
REMOTE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
REPO = REMOTE.rsplit('/experiments_iclr/', 1)[0]
DESTINATION = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6', DESTINATION]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify(binding):
    remote = Path(binding['path'])
    relative = remote.relative_to(REMOTE)
    path = PHASE / relative
    if path.is_symlink() or not path.resolve().is_relative_to(PHASE) or sha(path) != binding['sha256']:
        raise ValueError('Confined immutable input binding differs')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, output = args.manifest.resolve(), args.output.resolve()
    if not manifest.is_relative_to(PHASE) or not output.is_relative_to(PHASE) or output.exists():
        raise ValueError('Require confined inputs and new queue receipt output')
    if sha(manifest) != args.manifest_sha256:
        raise ValueError('Queue manifest hash differs')
    value = json.loads(manifest.read_text())
    if value['ssh_destination'] != DESTINATION or value['remote_repository'] != REPO or value['remote_phase'] != REMOTE:
        raise ValueError('Only the exact authorized one-GPU route/repository is allowed')
    if not value['root_admitted'] or not value['validation_only'] or value['test_scoring_admitted']:
        raise ValueError('Only fixed validation-only admission is allowed')
    if len(value['batches']) != 3 or sum(b['whole_cap_seconds'] for b in value['batches']) != 84000:
        raise ValueError('Require exactly three fixed bounded invocations')
    bindings = [value[k] for k in ('allocation_amendment', 'forecast_adoption', 'protocol', 'superseded_unexecuted_request')]
    for batch in value['batches']:
        bindings.append(batch['request'])
        request = json.loads(verify(batch['request']).read_text())
        if len(request['operations']) != 18 or request['whole_cap_seconds'] != 28000 or request['per_operation_timeout_seconds'] != 1500:
            raise ValueError('Exact batch operation/cap differs')
        bindings.extend(request['protected_files'])
    for binding in bindings:
        verify(binding)
    output.mkdir(parents=True)
    start = time.monotonic()
    write(output / 'START.json', {'UTC': datetime.now(timezone.utc).isoformat(),
          'manifest_sha256': args.manifest_sha256, 'launcher_sha256': sha(Path(__file__)),
          'ssh_destination': DESTINATION, 'aggregate_reserved_cap_seconds': 84000,
          'shell_local': False, 'validation_only': True, 'test_scoring_admitted': False})
    records, error = [], None
    try:
        for batch in value['batches']:
            if sha(manifest) != args.manifest_sha256:
                raise ValueError('Queue manifest changed')
            for binding in bindings:
                verify(binding)
            number = batch['number']
            command = ('cd ' + shlex.quote(REPO)
                       + ' && source experiments_iclr/postsubmission_20260930/protocols/repo_env.sh'
                       + ' && export GNNM_SSH_DESTINATION=' + shlex.quote(DESTINATION)
                       + ' CUDA_VISIBLE_DEVICES=0'
                       + ' && ' + shlex.join(batch['remote_argv']))
            write(output / f'batch{number:02d}_START.json', {'UTC': datetime.now(timezone.utc).isoformat(),
                  'request': batch['request'], 'remote_argv': batch['remote_argv'],
                  'ssh_argv': SSH, 'remote_shell_command': command})
            print(json.dumps({'event': 'batch_started', 'number': number}), flush=True)
            tick = time.monotonic()
            with (output / f'batch{number:02d}_SSH.log').open('x') as stream:
                child = subprocess.run(SSH + [command], stdout=stream, stderr=subprocess.STDOUT, check=False)
            record = {'number': number, 'exit_code': child.returncode,
                      'launcher_seconds': time.monotonic() - tick,
                      'UTC': datetime.now(timezone.utc).isoformat(),
                      'log_sha256': sha(output / f'batch{number:02d}_SSH.log')}
            records.append(record)
            write(output / f'batch{number:02d}_TERMINAL.json', record)
            print(json.dumps({'event': 'batch_terminal', **record}), flush=True)
            if child.returncode:
                raise RuntimeError('Stop queue on failed SSH/remote invocation; audit retained remote state before retry')
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    complete = error is None and len(records) == 3 and sha(manifest) == args.manifest_sha256
    write(output / 'TERMINAL.json', {'UTC': datetime.now(timezone.utc).isoformat(),
          'complete': complete, 'error': error, 'records': records,
          'launcher_seconds': time.monotonic() - start,
          'manifest_unchanged': sha(manifest) == args.manifest_sha256,
          'remote_completion_requires_retained_evidence_audit': True})
    raise SystemExit(0 if complete else 1)


if __name__ == '__main__':
    main()
