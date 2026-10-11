"""Deploy exact source files, then launch one bounded full-TRAIN qualification."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H=Path(__file__).resolve().parent;P=H.parent
REMOTE=r'''
import base64,hashlib,json,os,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=P/'native_SAGE_PE4_execution_root_20261011_v1'
for row in payload:
    rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
    p=P/rel;assert p.resolve().is_relative_to(P)
    b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256']
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==b,p
    else:p.write_bytes(b)
assert not (H/'qualify_owner_v1').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
cmd=[str(R/'.venv/bin/python'),'-B',str(H/'owner.py'),'--mode','qualify']
with (H/'qualify_launch_stdout.log').open('x') as out,(H/'qualify_launch_stderr.log').open('x') as err:
    owner=subprocess.Popen(cmd,cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
launch=dict(owner_PID=owner.pid,command=cmd,actual_full_TRAIN_only=True,scientific_fit=False)
(H/'QUALIFICATION_LAUNCH_V1.json').write_text(json.dumps(launch,indent=2)+'\n')
for _ in range(100):
    start=H/'qualify_owner_v1/OWNER_START.json'
    if start.exists():launch['owner_start']=json.loads(start.read_text());break
    if owner.poll() is not None:
        launch['early_owner_exit']=owner.returncode;launch['stderr']=(H/'qualify_launch_stderr.log').read_text()[-2500:];break
    time.sleep(.2)
print(json.dumps(launch))
'''

def main():
    out=H/'QUALIFICATION_LAUNCH_TRANSPORT_V1.json';assert not out.exists()
    folders=['native_SAGE_PE4_full_TRAIN_integration_source_20261011_v1','native_SAGE_PE4_gamma1_source_20261011_v1']
    paths=[p for f in folders for p in sorted((P/f).iterdir()) if p.is_file()]+[H/'owner.py']
    rows=[]
    for p in paths:
        b=p.read_bytes()
        if p.suffix=='.py':ast.parse(b)
        rows.append(dict(path=str(p.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
    spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'qualify','literal-route','one-case'])]
    r=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode(),capture_output=True,text=True,timeout=60)
    out.write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stderr[-2000:]
    result=json.loads(r.stdout);(H/'QUALIFICATION_LAUNCH_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
