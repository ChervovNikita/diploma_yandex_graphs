"""Launch only official archive acquisition on the verified authorized endpoint."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
REMOTE_CODE = '''
import base64, hashlib, json, os, pathlib, subprocess, sys
repo=pathlib.Path(sys.argv[1]); login=sys.argv[2]; expected=sys.argv[3]
actual=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],check=True,capture_output=True,text=True)
if repo.resolve()!=repo or actual.stdout.strip().splitlines()!=[expected]: raise RuntimeError('Authorized root/GPU required')
git=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,check=True,capture_output=True,text=True)
if pathlib.Path(git.stdout.strip()).resolve()!=repo: raise RuntimeError('Project Git root differs')
payload=json.load(sys.stdin); data=base64.b64decode(payload['data'])
if hashlib.sha256(data).hexdigest()!=payload['sha256']: raise RuntimeError('Source transport differs')
folder=repo/'experiments_iclr/postsubmission_20260930/buddy_official_archive_staging_v1'
if not folder.resolve().is_relative_to(repo): raise RuntimeError('Source folder escapes project')
folder.mkdir(parents=True,exist_ok=True)
path=folder/'stage_official_archive.py'
if path.is_symlink(): raise RuntimeError('Symlink refused')
if path.exists():
 if path.read_bytes()!=data: raise RuntimeError('Existing source differs')
else:
 with path.open('xb') as stream: stream.write(data)
env=os.environ.copy(); env.update(GNNM_SSH_DESTINATION=login,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
result=subprocess.run([str(repo/'.venv/bin/python'),'-B',str(path)],cwd=repo,env=env)
raise SystemExit(result.returncode)
'''


def main():
    receipt = HERE / 'ROOT_LAUNCH.json'
    if receipt.exists():
        raise RuntimeError('Single-use staging identity already attempted')
    data = (HERE / 'stage_official_archive.py').read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE_CODE, REPO, LOGIN, GPU])
    launch = dict(UTC=datetime.now(timezone.utc).isoformat(), ssh_destination=LOGIN,
                  source_sha256=digest, GPU_compute=False, test_members_opened=False,
                  source_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        result = subprocess.run([*SSH, command],
                                input=json.dumps(dict(data=base64.b64encode(data).decode(), sha256=digest)),
                                capture_output=True, text=True, timeout=600)
        launch.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                      status='COMPLETED_ACQUISITION' if result.returncode == 0 else 'FAILED_ACQUISITION')
    except Exception as error:
        launch.update(status='LOCAL_TRANSPORT_FAILED', error_type=type(error).__name__, error=str(error))
    with receipt.open('x') as stream:
        json.dump(launch, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: launch.get(k) for k in ('status', 'exit_code', 'stdout', 'error')}, sort_keys=True))
    return 0 if launch.get('exit_code') == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
