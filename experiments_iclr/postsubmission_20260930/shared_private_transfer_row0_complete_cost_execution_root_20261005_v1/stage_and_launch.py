"""Stage exact cost-source/release bytes and detach its bounded owned queue."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SOURCE='shared_private_transfer_row0_complete_cost_queue_released_20261005_v1'
REMOTE_PHASE=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
REMOTE=r'''
import base64,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
payload=json.load(sys.stdin)
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20).strip().splitlines()
assert len(gpu)==1 and gpu[0].split(',')[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac' and int(gpu[0].split(',')[1].strip())*1024**2>=12*1024**3
root=phase/'shared_private_transfer_row0_complete_cost_execution_root_20261005_v1'
source=phase/'shared_private_transfer_row0_complete_cost_queue_released_20261005_v1'
assert not root.exists() and not source.exists()
decoded=[]
for row in payload['files']:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in {root.name,source.name}
 path=phase/rel;assert path.resolve().is_relative_to(phase.resolve())
 raw=base64.b64decode(row['base64'],validate=True);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 decoded.append((path,raw))
for path,raw in decoded:
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source/'SOURCE_MANIFEST.json')==payload['source_manifest_sha256']
for row in json.loads((source/'SOURCE_MANIFEST.json').read_text())['files']:assert sha(source/row['path'])==row['sha256']
release=json.loads((root/'ROOT_RELEASE.json').read_text());assert release['approved'] is True and release['fits_authorized'] is False and release['VALID_values_access'] is False and release['TEST_access'] is False
argv=['/usr/bin/python3','-B',str(source/'queue.py'),'--release',str(root/'ROOT_RELEASE.json')]
with (root/'queue.stdout.log').open('xb') as stdout,(root/'queue.stderr.log').open('xb') as stderr:
 child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,cwd=repo,start_new_session=True)
time.sleep(.2)
p=Path('/proc')/str(child.pid);raw=(p/'stat').read_text();f=raw[raw.rfind(')')+2:].split();actual=[s.decode() for s in (p/'cmdline').read_bytes().split(bytes([0])) if s]
assert actual==argv and int(f[2])==int(f[3])==child.pid and str((p/'cwd').resolve())==str(repo)
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),GPU_metadata=gpu,queue_identity=dict(PID=child.pid,start_ticks=int(f[19]),state=f[0],argv=actual,pgid=int(f[2]),sid=int(f[3]),cwd=str((p/'cwd').resolve())),source_manifest_sha256=sha(source/'SOURCE_MANIFEST.json'),root_release_sha256=sha(root/'ROOT_RELEASE.json'),detached_launches=1,fits=0,VALID_TEST_values_access=False)
(root/'LAUNCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source=PHASE/SOURCE;manifest=source/'SOURCE_MANIFEST.json'
    assert sha(manifest)=='c534e591932230216b3007b986bf3ba6924e941b8a7d5f1452d6ab8281ba43de'
    for row in json.loads(manifest.read_text())['files']:
        p=source/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
        if p.suffix=='.py':ast.parse(p.read_text())
    ast.parse(REMOTE)
    release=json.loads((source/'ROOT_RELEASE_TEMPLATE.json').read_text())
    release.update(schema='root_released_three_F1_complete_TRAIN_cost_cycles_v1',approved=True,
        queue_source_manifest_sha256=sha(manifest),queue_program_sha256=sha(source/'queue.py'),
        root_review_evidence=[{'path':str((HERE/'ROOT_REVIEW.md').relative_to(PHASE)),'sha256':sha(HERE/'ROOT_REVIEW.md')}])
    with (HERE/'ROOT_RELEASE.json').open('x') as f:json.dump(release,f,indent=2,sort_keys=True);f.write('\n')
    paths=[p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[HERE/'ROOT_REVIEW.md',HERE/'ROOT_RELEASE.json',Path(__file__)]
    files=[]
    for p in paths:
        assert p.resolve().is_relative_to(PHASE.resolve()) and not p.is_symlink()
        b=p.read_bytes();files.append({'path':str(p.relative_to(PHASE)),'bytes':len(b),'sha256':sha(p),'base64':base64.b64encode(b).decode()})
    payload={'source_manifest_sha256':sha(manifest),'files':files}
    with (HERE/'STAGING_PAYLOAD.json').open('x') as f:json.dump(payload,f);f.write('\n')
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
      'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -c '+shlex.quote(REMOTE)]
    result=subprocess.run(command,input=json.dumps(payload),capture_output=True,text=True,timeout=50)
    with (HERE/'TRANSPORT.json').open('x') as f:json.dump({'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest()},f,indent=2);f.write('\n')
    if result.returncode:raise RuntimeError(result.stderr)
    receipt=json.loads(result.stdout)
    with (HERE/'LAUNCH_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
