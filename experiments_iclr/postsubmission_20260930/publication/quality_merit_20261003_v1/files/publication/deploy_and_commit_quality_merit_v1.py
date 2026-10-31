"""Deploy an exact reviewed publication snapshot and invoke the existing committer."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

PHASE = Path(__file__).resolve().parents[1]
SNAPSHOT = PHASE / 'publication/quality_merit_20261003_v1'
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
CODE = r'''
import base64,hashlib,json,os,pathlib,subprocess,sys
repo=pathlib.Path(sys.argv[1]);login=sys.argv[2];uuid=sys.argv[3]
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
if repo.resolve()!=repo or gpu.stdout.strip().splitlines()!=[uuid]:raise ValueError('Actual authorized repository required')
phase=repo/'experiments_iclr/postsubmission_20260930'
payload=json.load(sys.stdin)
for row in payload['files']:
 rel=pathlib.PurePosixPath(row['path'])
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid reviewed snapshot path')
 path=phase.joinpath(*rel.parts)
 if not path.resolve().is_relative_to(phase):raise ValueError('Escaping snapshot path')
 cursor=phase
 for part in rel.parts:
  cursor/=part
  if cursor.is_symlink():raise ValueError('Symlink target refused')
 data=base64.b64decode(row['data'])
 if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Snapshot transport digest differs')
 if path.exists():
  if path.read_bytes()!=data:raise ValueError('Existing snapshot/source differs')
 else:
  path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(data)
env=os.environ.copy();env['GNNM_SSH_DESTINATION']=login
r=subprocess.run([str(repo/'.venv/bin/python'),str(phase/'publication/commit_reviewed_inventory_v1.py'),
 '--plan','publication/quality_merit_20261003_v1/PLAN_v2.json',
 '--receipt','publication/quality_merit_20261003_v1/COMMIT_RECEIPT_v1.json'],
 cwd=repo,env=env,capture_output=True,text=True,timeout=90)
receipt=phase/'publication/quality_merit_20261003_v1/COMMIT_RECEIPT_v1.json'
print(json.dumps({'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,
 'commit_receipt':json.loads(receipt.read_text()) if receipt.exists() else None}))
raise SystemExit(r.returncode)
'''


def main():
    inventory = json.loads((SNAPSHOT / 'INVENTORY.json').read_text())
    relative = {r['source_snapshot'] for r in inventory['files']}
    relative.update(['publication/quality_merit_20261003_v1/INVENTORY.json',
                     'publication/quality_merit_20261003_v1/PLAN_v2.json',
                     'publication/commit_reviewed_inventory_v1.py'])
    files = []
    for value in sorted(relative):
        path = PHASE / value
        if not path.resolve().is_relative_to(PHASE) or path.is_symlink():
            raise ValueError('Unconfined local snapshot')
        data = path.read_bytes()
        files.append({'path': value, 'sha256': hashlib.sha256(data).hexdigest(),
                      'data': base64.b64encode(data).decode()})
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', CODE, REPO, LOGIN, UUID])
    started = datetime.now(timezone.utc).isoformat()
    r = subprocess.run(SSH + [command], input=json.dumps({'files': files}), capture_output=True, text=True, timeout=120)
    record = {'UTC': started, 'terminal_UTC': datetime.now(timezone.utc).isoformat(),
              'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr,
              'executed_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (SNAPSHOT / 'COMMIT_LAUNCH_v1.json').open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    value = json.loads(r.stdout)
    if value['commit_receipt'] is not None:
        receipt = value['commit_receipt']
        with (SNAPSHOT / 'COMMIT_RECEIPT_v1.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
        print(json.dumps({'commit': receipt['commit'], 'changed_files': len(receipt['changed_paths']), 'push_verified': False}))
    else:
        print(json.dumps({'exit_code': r.returncode, 'stderr': value['stderr']}))
    raise SystemExit(r.returncode)


if __name__ == '__main__':
    main()
