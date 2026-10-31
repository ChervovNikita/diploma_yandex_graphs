"""Fetch explicit bounded text evidence from the authorized 18.77 Git checkout."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE = '/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.id):
        raise RuntimeError('Fresh simple command identity required')
    for value in args.paths:
        path = PurePosixPath(value)
        if path.is_absolute() or '..' in path.parts or str(path) != value or path.suffix not in {'.json', '.jsonl', '.txt', '.log', '.md'}:
            raise RuntimeError('Only explicit normalized phase text-evidence paths')
    code = r'''
import base64,hashlib,json,pathlib,subprocess
root=pathlib.Path(REMOTE)
repo=root.parent.parent
assert root.resolve()==root and pathlib.Path(subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,check=True,capture_output=True,text=True).stdout.strip()).resolve()==repo
actual=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.splitlines()
assert len(actual)==2 and set(actual)=={'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
rows=[]
for value in PATHS:
 path=root/value
 assert path.resolve().is_relative_to(root) and not path.is_symlink() and path.is_file() and path.stat().st_size<2*1024*1024
 data=path.read_bytes();data.decode('utf-8');assert b'\x00' not in data
 rows.append({'path':value,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()})
print(json.dumps({'files':rows,'actual_git_root':str(repo),'physical_GPU_UUIDs':actual}))
'''.replace('REMOTE', repr(REMOTE)).replace('PATHS', repr(args.paths))
    command_file = HERE / (args.id + '.txt')
    with command_file.open('x') as stream:
        stream.write(shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', code]))
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(HERE / 'run_gpu77_v3.py'),
                             '--id', args.id, '--command-file', str(command_file)],
                            capture_output=True, text=True, timeout=170)
    if result.returncode:
        print(result.stdout); print(result.stderr)
        raise SystemExit(result.returncode)
    value = json.loads((HERE / 'commands' / args.id / 'RECEIPT.json').read_text())
    remote = json.loads(value['stdout'])
    assert [r['path'] for r in remote['files']] == args.paths
    records = []
    for row in remote['files']:
        data = base64.b64decode(row['data'])
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        target = PHASE / row['path']
        assert target.resolve().is_relative_to(PHASE) and not target.is_symlink()
        if target.exists():
            if target.read_bytes() != data:
                raise RuntimeError('Existing evidence differs: ' + row['path'])
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
        records.append({k: row[k] for k in ('path', 'bytes', 'sha256')})
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), target='shmelev@192.168.18.77',
                   files=records, transport_receipt_sha256=hashlib.sha256((HERE / 'commands' / args.id / 'RECEIPT.json').read_bytes()).hexdigest())
    (HERE / 'commands' / args.id / 'FETCH_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'fetched_files':len(records),'bytes':sum(r['bytes'] for r in records),'id':args.id}))


if __name__ == '__main__':
    main()
