"""Deploy sealed source; bind actual runtime; start one detached finite qualifier."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

P=Path(__file__).resolve().parent.parent
H=Path(__file__).resolve().parent
S=P/'native_SAGE_endpoint_value_transport_source_20261011_v1'
REMOTE=r'''
import base64,hashlib,json,socket,subprocess,sys,zlib,os,datetime
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=P/'native_SAGE_endpoint_value_execution_root_20261011_v1'
for row in payload:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 q=P/rel;assert q.resolve().is_relative_to(P);b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256'];q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==b,q
 else:q.write_bytes(b)
assert not (H/'QUALIFICATION_BINDING_V1.json').exists() and not (H/'qualify_owner_v1').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
setup=subprocess.run([str(R/'.venv/bin/python'),'-B',str(H/'bind_qualification_v1.py')],cwd=R,env=env,capture_output=True,text=True,timeout=90)
(H/'binding_stdout.log').write_text(setup.stdout);(H/'binding_stderr.log').write_text(setup.stderr)
assert setup.returncode==0,setup.stderr[-2500:]
with (H/'qualification_owner_stdout.log').open('x') as out,(H/'qualification_owner_stderr.log').open('x') as err:
 child=subprocess.Popen([str(R/'.venv/bin/python'),'-B',str(H/'owner_v1.py'),'--stage','qualify'],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
r=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_PID=child.pid,deployed=len(payload),scientific_fits=0,full_TRAIN_qualification_only=True,binding=json.loads((H/'QUALIFICATION_BINDING_V1.json').read_text()))
(H/'QUALIFICATION_LAUNCH_V1.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
'''

def main():
    sealed=json.loads((S/'SEAL.json').read_text())
    for row in sealed['files']:
        assert hashlib.sha256((S/row['path']).read_bytes()).hexdigest()==row['sha256']
    paths=sorted(S.iterdir())+[H/name for name in ('bind_qualification_v1.py','run_qualification_v1.py','owner_v1.py')]
    rows=[]
    for path in paths:
        if not path.is_file():continue
        b=path.read_bytes()
        if path.suffix=='.py':ast.parse(b)
        rows.append(dict(path=str(path.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
    spec=importlib.util.spec_from_file_location('reviewed_payload_transport',P/'publication/publish_exact_inventory_v19.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'qualification','literal-route','actual-TRAIN'])]
    result=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode(),capture_output=True,text=True,timeout=120)
    (H/'QUALIFICATION_LAUNCH_TRANSPORT_V1.json').write_text(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    assert result.returncode==0,result.stderr[-2500:]
    obj=json.loads(result.stdout)
    (H/'QUALIFICATION_LAUNCH_V1.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(dict(owner_PID=obj['owner_PID'],deployed=obj['deployed'],resources=obj['binding']['resource_observation'],scientific_fits=0)))

if __name__=='__main__':main()
