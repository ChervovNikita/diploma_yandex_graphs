"""Deploy exact native engineering source and run one synthetic qualification."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PACKETS = {
    'learnability_responsibility_native_amazon_sparse_port_20261005_v1':
        '5401171b2e7cc853b5d15e97ff11ceb8f2cc6466dd4ff08c47f0308e57bb16a1',
    'learnability_responsibility_native_numerical_worker_preparation_20261005_v1':
        'bd4e5c40032ca0bc6181ad054777550d533c4ad23300558974c71bf99844c4e3',
}
WORKER_SHA = '976332545f78f1fe9642b2a4fa9d61127b427739cd85ea9a26301afc92cb61f7'

REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
request=json.load(sys.stdin)
prepared=[]
for row in request['files']:
 relative=Path(row['path']);assert not relative.is_absolute() and '..' not in relative.parts
 p=phase/relative;assert p.resolve().is_relative_to(phase) and not p.is_symlink()
 b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 if p.exists():assert p.is_file() and p.read_bytes()==b
 prepared.append((p,b))
out=phase/request['run_name'];assert not out.exists()
for p,b in prepared:
 if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
for name,h in request['packet_manifests'].items():
 assert hashlib.sha256((phase/name/'MANIFEST.json').read_bytes()).hexdigest()==h
out.mkdir()
(out/'EXECUTION_RELEASE.json').write_text(json.dumps(request['release'],indent=2)+'\n')
worker=phase/'learnability_responsibility_native_numerical_worker_preparation_20261005_v1/qualify.py'
assert hashlib.sha256(worker.read_bytes()).hexdigest()==request['worker_sha256']
runtime=phase/'native_ncn_runtime_20261005_v1/.venv/bin/python'
argv=[str(runtime),'-B',str(worker),'--execute-authorized','--mode','synthetic','--source-root',str(phase),'--output',str(out/'output')]
with (out/'worker.stdout.log').open('xb') as stdout,(out/'worker.stderr.log').open('xb') as stderr:
 started=time.monotonic()
 child=subprocess.Popen(argv,cwd=repo,stdout=stdout,stderr=stderr)
 raw=(Path('/proc')/str(child.pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 launch={'UTC':datetime.now(timezone.utc).isoformat(),'PID':child.pid,'start_ticks':int(fields[19]),'argv':argv,'cwd':str(repo),'worker_sha256':request['worker_sha256'],'scientific_fits':0}
 (out/'LAUNCH_RECEIPT.json').write_text(json.dumps(launch,indent=2)+'\n')
 print(json.dumps({'stage':'launched','owned_process':launch}),flush=True)
 exit_code=child.wait()
 terminal={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':exit_code,'physical_child_seconds':time.monotonic()-started,'owned_process':launch,'artifacts':[]}
 for p in [out/'output/RESULT.json',out/'worker.stdout.log',out/'worker.stderr.log']:
  if p.exists():
   b=p.read_bytes();terminal['artifacts'].append({'path':str(p.relative_to(out)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 (out/'TERMINAL.json').write_text(json.dumps(terminal,indent=2)+'\n')
 result_path=out/'output/RESULT.json';result=json.loads(result_path.read_text()) if result_path.exists() else None
 print(json.dumps({'stage':'terminal','terminal':terminal,'result':result}),flush=True)
raise SystemExit(exit_code)
'''

def main():
    assert not (HERE/'EXECUTION_RELEASE.json').exists(), 'One execution identity only'
    files=[]
    for name,expected in PACKETS.items():
        packet=PHASE/name
        manifest=json.loads((packet/'MANIFEST.json').read_text())
        assert hashlib.sha256((packet/'MANIFEST.json').read_bytes()).hexdigest()==expected
        for row in manifest['files']:
            data=(packet/row['path']).read_bytes()
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        for path in sorted(packet.rglob('*')):
            if path.is_dir():continue
            assert path.is_file() and not path.is_symlink() and path.stat().st_size<2_000_000
            assert path.suffix in ('.py','.json','.md','.txt')
            b=path.read_bytes()
            files.append({'path':str(path.relative_to(PHASE)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
    release={'UTC':datetime.now(timezone.utc).isoformat(),'purpose':'One CPU float64 synthetic full-native engineering qualification',
             'worker_sha256':WORKER_SHA,'packet_manifests':PACKETS,'fixed_deadline_seconds':180,
             'synthetic_nodes':15,'native_local_layers':10,'native_global_layers':1,'native_width':512,
             'scientific_fits_authorized':False,'dataset_label_access':False,'predictive_claim_authorized':False,
             'source_review':'Root read complete worker and fixed pins; independent source reviewer reports no synthetic blocker',
             'remote_wrapper_sha256':hashlib.sha256(REMOTE.encode()).hexdigest()}
    (HERE/'EXECUTION_RELEASE.json').write_text(json.dumps(release,indent=2)+'\n')
    request={'files':files,'packet_manifests':PACKETS,'worker_sha256':WORKER_SHA,'run_name':HERE.name,'release':release}
    command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
             '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
             '-o','UpdateHostKeys=no','-o','ConnectTimeout=20',LOGIN,
             'cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    child.stdin.write(json.dumps(request));child.stdin.close()
    transcript=[]
    for line in child.stdout:
        transcript.append(line)
        item=json.loads(line)
        if item['stage']=='launched':
            (HERE/'LAUNCH_RECEIPT.json').write_text(json.dumps(item['owned_process'],indent=2)+'\n')
            print(json.dumps({'stage':'launched','PID':item['owned_process']['PID'],'start_ticks':item['owned_process']['start_ticks']}),flush=True)
        else:
            (HERE/'TERMINAL.json').write_text(json.dumps(item['terminal'],indent=2)+'\n')
            if item['result'] is not None:
                (HERE/'RESULT.json').write_text(json.dumps(item['result'],indent=2)+'\n')
            print(json.dumps({'stage':'terminal','exit_code':item['terminal']['exit_code'],'status':item['result'].get('status') if item['result'] else None,'error':item['result'].get('error') if item['result'] else None}),flush=True)
    stderr=child.stderr.read();exit_code=child.wait()
    (HERE/'TRANSPORT_RECEIPT.json').write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':exit_code,'stdout':''.join(transcript),'stderr':stderr},indent=2)+'\n')
    raise SystemExit(exit_code)

if __name__=='__main__':main()
