"""Fetch an explicit, bounded inventory from the verified authorized one-GPU phase."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

PHASE = Path(__file__).resolve().parents[1]
REMOTE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    if not receipt.is_relative_to(PHASE) or receipt.exists():
        raise ValueError('Receipt must be new and inside the local phase')
    for value in args.paths:
        p = PurePosixPath(value)
        if p.is_absolute() or '..' in p.parts or not p.parts or str(p) != value:
            raise ValueError('Require explicit normalized phase-relative paths')
    command = 'tar -C ' + shlex.quote(REMOTE) + ' -cf - -- ' + ' '.join(map(shlex.quote, args.paths))
    result = subprocess.run(SSH + [command], capture_output=True, check=False)
    if result.returncode:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open('x') as stream:
            json.dump({'schema': 'gnnm-authorized-explicit-evidence-fetch-v2', 'status': 'failed_before_local_extraction',
                       'UTC': datetime.now(timezone.utc).isoformat(), 'ssh_destination': SSH[-1], 'requested_paths': args.paths,
                       'exit_code': result.returncode, 'stderr': result.stderr.decode(errors='replace')}, stream, indent=2)
            stream.write('\n')
        raise RuntimeError(result.stderr.decode(errors='replace'))
    # Only regular files/directories are admitted. Existing different files fail closed.
    records = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
        for member in archive.getmembers():
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe evidence archive member')
            if not any(p == PurePosixPath(q) or PurePosixPath(q) in p.parents for q in args.paths):
                raise ValueError('Unrequested evidence archive member')
            target = PHASE.joinpath(*p.parts)
            if not target.resolve().is_relative_to(PHASE):
                raise ValueError('Evidence destination escapes local phase')
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            data = archive.extractfile(member).read()
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError('Existing local evidence differs: ' + member.name)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
            records[str(p)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    value = {'schema': 'gnnm-authorized-explicit-evidence-fetch-v2', 'UTC': datetime.now(timezone.utc).isoformat(),
             'ssh_destination': SSH[-1], 'remote_phase': REMOTE, 'requested_paths': args.paths, 'files': records,
             'archive_sha256': hashlib.sha256(result.stdout).hexdigest(),
             'ssh_stderr_sha256': hashlib.sha256(result.stderr).hexdigest()}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'receipt': str(receipt), 'files': len(records), 'bytes': sum(r['bytes'] for r in records.values())}))


if __name__ == '__main__':
    main()
