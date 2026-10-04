"""Stage exact graph-view sources and run synthetic CPU correctness checks.

No real graph, labels, checkpoints, predictive values or GPU computation.
"""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess
import zlib

HERE=Path(__file__).resolve().parent
P=HERE.parent
SOURCE=P/'accuracy_first_graph_view_source_preparation_20261004_v1'
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
MANIFEST='135e30466c703b1929eadd680b7152330620e4cde75152ec69317458580efa2a'
PROTOCOL='55f4146b450f2e04c518e07df9a118bb21f7dc65e18dc659eb34e6dbe8edfb46'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def save(name,value):
    with (HERE/name).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,sys,time,zlib
repo=Path(REPO);phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[UUID]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read(),validate=True)))
for row in payload:
 p=phase/row['path'];assert p.resolve().is_relative_to(phase) and not p.is_symlink()
 raw=base64.b64decode(row['data'],validate=True)
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(raw)
root=phase/'graph_view_source_numerical_execution_root_20261004_v1';root.mkdir(exist_ok=True)
source=phase/'accuracy_first_graph_view_source_preparation_20261004_v1'
command=[str(repo/'.venv/bin/python'),'-B',str(source/'test_numerical.py'),'--execute','--manifest-sha256',MANIFEST,'--protocol-sha256',PROTOCOL]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
started=time.perf_counter();r=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True,timeout=120)
value=dict(UTC=datetime.now(timezone.utc).isoformat(),route=dict(login=LOGIN,repository=str(repo),GPU_UUID=UUID),
 command=command,source_manifest_sha256=MANIFEST,protocol_sha256=PROTOCOL,
 exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,wall_seconds=time.perf_counter()-started,
 runtime_device='cpu',CUDA_VISIBLE_DEVICES='',threads=1,real_data_read=False,
 predictive_values_read=False,scientific_training_updates=0,synthetic_implementation_checks_only=True)
with (root/'NUMERICAL_CHECKS.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
'''


def main():
    assert sha((SOURCE/'MANIFEST.json').read_bytes())==MANIFEST
    assert sha((SOURCE/'PROTOCOL.json').read_bytes())==PROTOCOL
    manifest=json.loads((SOURCE/'MANIFEST.json').read_text())
    files=[]
    for row in manifest['payload']:
        path=SOURCE/row['path'];raw=path.read_bytes()
        assert sha(raw)==row['sha256'] and len(raw)==row['bytes']
        files.append(path)
    files += [SOURCE/'MANIFEST.json',SOURCE/'SEAL.json']
    bindings=json.loads((SOURCE/'SOURCE_BINDINGS.json').read_text())
    files += [P/row['path'] for row in bindings['context']]
    payload=[]
    for path in dict.fromkeys(files):
        raw=path.read_bytes()
        payload.append(dict(path=str(path.relative_to(P)),bytes=len(raw),sha256=sha(raw),data=base64.b64encode(raw).decode()))
    save('STAGE_INVENTORY.json',[{k:v for k,v in r.items() if k!='data'} for r in payload])
    code='REPO='+repr(REPO)+'\nLOGIN='+repr(LOGIN)+'\nUUID='+repr(UUID)+'\nMANIFEST='+repr(MANIFEST)+'\nPROTOCOL='+repr(PROTOCOL)+'\n'+REMOTE
    compile(code,'<graph-view-cpu-checks>','exec')
    (HERE/'REMOTE_SOURCE.py.txt').write_text(code)
    command='cd '+shlex.quote(REPO)+' && '+shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code])
    ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
         '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',
         '-o','ConnectTimeout=15',LOGIN,command]
    started=datetime.now(timezone.utc).isoformat()
    r=subprocess.run(ssh,input=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode(),
                     capture_output=True,text=True,timeout=160)
    save('TRANSPORT.json',dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),
         exit_code=r.returncode,stderr=r.stderr,stdout_sha256=sha(r.stdout.encode()),
         remote_source_sha256=sha(code.encode()),private_key_contents_read=False))
    assert r.returncode==0,r.stderr
    value=json.loads(r.stdout);save('NUMERICAL_CHECKS.json',value)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
