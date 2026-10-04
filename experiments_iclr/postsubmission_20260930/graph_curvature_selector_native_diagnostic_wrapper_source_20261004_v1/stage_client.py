"""Source-only client. Explicit functions only; CLI prints preparation status."""
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SERVER_PHASE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
PYTHON = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python'
ROUTE = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
KEY = '/Users/alex/.ssh/mlspace__private_key_anogena.txt'


def ssh(arguments, data=None):
    command = 'cd '+shlex.quote(SERVER_PHASE)+' && '+shlex.join(['env',
        'GNNM_SSH_DESTINATION='+ROUTE,'PYTHONDONTWRITEBYTECODE=1',*arguments])
    return subprocess.run(['ssh','-T','-o','IdentitiesOnly=yes','-o','UpdateHostKeys=no',
        '-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=20','-i',KEY,
        '-p','2222',ROUTE,command],input=data,capture_output=True,timeout=60,check=False)


def stage():
    records = json.loads((HERE/'STAGING_INPUT.json').read_text())['files']
    manifest = json.loads((HERE/'MANIFEST.json').read_text())
    records += [dict(row,path=HERE.name+'/'+row['path']) for row in manifest['files']]
    for name in ('MANIFEST.json','SEAL.json'):
        data = (HERE/name).read_bytes()
        records.append(dict(path=HERE.name+'/'+name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    payload = []
    for row in records:
        path = PHASE/row['path']
        assert path.suffix in ('.py','.json','.md','.patch'), 'No arrays/binary payloads may be staged'
        data = path.read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        payload.append(dict(row,base64=base64.b64encode(data).decode()))
    script = 'RECORDS = '+repr(payload)+'''\nimport base64,getpass,hashlib,json,os,pathlib,subprocess
assert os.environ.get('GNNM_SSH_DESTINATION') == '''+repr(ROUTE)+'''
rows=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()
assert [row.strip() for row in rows if row.strip()] == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
phase=pathlib.Path('''+repr(SERVER_PHASE)+''')
assert phase.is_dir() and phase.resolve(strict=True) == phase, 'Fixed phase or ancestor is a symlink'
def checked_path(relative):
    relative=pathlib.Path(relative)
    assert not relative.is_absolute() and '..' not in relative.parts
    path=phase/relative
    for ancestor in (path,*path.parents):
        assert not ancestor.is_symlink(), 'Refuse source target/ancestor symlink: '+str(ancestor)
        if ancestor == phase: break
    assert phase in path.parents and path.resolve(strict=False) == path, 'Source path escapes fixed phase'
    return path
for row in RECORDS:
    path=checked_path(row['path'])
    data=base64.b64decode(row['base64'])
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    if path.exists(): assert path.read_bytes()==data, 'Refuse differing source: '+str(path)
for row in RECORDS:
    path=checked_path(row['path'])
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        path=checked_path(row['path'])
        with path.open('xb') as stream: stream.write(base64.b64decode(row['base64']))
print(json.dumps(dict(status='STAGED_SOURCE_ONLY',files=len(RECORDS),GPU_UUID=rows[0].strip(),ssh_destination=os.environ['GNNM_SSH_DESTINATION'],unix_user=getpass.getuser(),arrays_staged=0,launched=False,no_differing_overwrite=True)))
'''
    return ssh([PYTHON,'-B','-'],script.encode())


def launch():
    return ssh([PYTHON,'-B',SERVER_PHASE+'/'+HERE.name+'/launch_once.py'])


if __name__ == '__main__':
    print(json.dumps(dict(status='SOURCE_PREPARATION_ONLY_REVIEW_PENDING',stage_called=False,
        launch_called=False,per_graph_cap_seconds=300,minimum_free_gpu_memory_MiB=32768)))
