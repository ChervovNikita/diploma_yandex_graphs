"""One paid portable-binding check in the existing runtime, with a finite owner."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H=Path(__file__).resolve().parent
P=H.parent
REMOTE=r'''
import base64,zlib,json,sys,socket,subprocess,pathlib,hashlib,os,datetime,time
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
S=P/'portable_conditional_SAGE_same_runtime_binding_check_source_20261011_v1';H=P/'portable_conditional_SAGE_same_runtime_binding_check_root_20261011_v1'
for row in payload['files']:
    path=P/row['relative'];assert path.resolve().is_relative_to(P) and not path.is_symlink()
    data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
assert not (H/'OWNER_START_V1.json').exists() and not (H/'complete_v1').exists()
admission=json.loads((H/'ADMISSION_V1.json').read_text())
assert admission['permission'] and admission['manual_update_limit']==1 and not admission['CAL_TEST_truth_access'] and not admission['accuracy_NLL_scoring']
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
before=time.monotonic()
with (H/'child_v1.log').open('x') as log:
    child=subprocess.Popen([str(R/'.venv/bin/python'),'-B',str(S/'check.py'),'--package',str(P/'portable_conditional_SAGE_source_20261011_v1'),'--output',str(H/'complete_v1'),'--permit-one-binding-check'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    (H/'OWNER_START_V1.json').write_text(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child_pid=child.pid,external_wall_seconds=600,retry=False),indent=2)+'\n')
    timed_out=False
    try:code=child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        timed_out=True;child.kill();code=child.wait()
owner=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,timed_out=timed_out,seconds=time.monotonic()-before,direct_child_wait=True,child_pid=child.pid,child_absent=not pathlib.Path('/proc',str(child.pid)).exists(),scientific_fits=0,existing_runtime_only=True,clean_install=False,external_portability=False,predictive_evidence=False)
(H/'OWNER_END_V1.json').write_text(json.dumps(owner,indent=2)+'\n')
files={str(q.relative_to(H)):q.read_text() for q in (H/'OWNER_START_V1.json',H/'OWNER_END_V1.json',H/'complete_v1/COMPLETE_BINDING_CHECK.json',H/'complete_v1/FAILURE.json') if q.exists()}
if code!=0:files['child_v1_failure_excerpt.txt']=(H/'child_v1.log').read_text()[-6000:]
print(json.dumps(dict(owner=owner,files=files)))
'''

def main():
    assert not (H/'OWNER_START_V1.json').exists() and not (H/'OWNER_END_V1.json').exists()
    paths=list((P/'portable_conditional_SAGE_same_runtime_binding_check_source_20261011_v1').iterdir())+[H/'ADMISSION_V1.json']
    files=[]
    for path in paths:
        if not path.is_file():continue
        data=path.read_bytes();assert len(data)<200000
        if path.suffix=='.py':compile(data,str(path),'exec')
        files.append(dict(relative=str(path.relative_to(P)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
    spec=importlib.util.spec_from_file_location('binding_transport',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    args=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'one-portable-binding-check','literal-route','no-predictive-score'])]
    r=m.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps(dict(files=files)).encode())).decode(),capture_output=True,text=True,timeout=660)
    (H/'TRANSPORT_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stdout[-2000:]+r.stderr[-2000:]
    value=json.loads(r.stdout)
    for relative,data in value['files'].items():
        path=H/relative;assert path.resolve().is_relative_to(H.resolve()) and not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
    print(json.dumps(dict(owner=value['owner'],complete_check=(H/'complete_v1/COMPLETE_BINDING_CHECK.json').exists(),failure=(H/'complete_v1/FAILURE.json').exists())))

if __name__=='__main__':main()
