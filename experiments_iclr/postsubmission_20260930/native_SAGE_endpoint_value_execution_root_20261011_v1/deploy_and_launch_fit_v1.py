"""Send the resource-only amended source and start its fixed scientific family."""
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
S=P/'native_SAGE_endpoint_value_transport_source_20261011_v2'
REMOTE=r'''
import base64,hashlib,json,socket,subprocess,sys,zlib,os
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
assert not (H/'SCIENTIFIC_LAUNCH_V1.json').exists() and not (H/'fit_owner_v1').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
r=subprocess.run([str(R/'.venv/bin/python'),'-B',str(H/'admit_and_launch_fit_v1.py')],cwd=R,env=env,capture_output=True,text=True,timeout=60)
(H/'admission_launch_stdout.log').write_text(r.stdout);(H/'admission_launch_stderr.log').write_text(r.stderr)
assert r.returncode==0,r.stderr[-2500:]
print(r.stdout.strip())
'''

def main():
    paths=sorted(S.iterdir())+[H/name for name in ('owner_v2.py','admit_and_launch_fit_v1.py')]
    rows=[]
    for path in paths:
        if not path.is_file():continue
        b=path.read_bytes()
        if path.suffix=='.py':ast.parse(b)
        rows.append(dict(path=str(path.relative_to(P)),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
    spec=importlib.util.spec_from_file_location('reviewed_payload_transport',P/'publication/publish_exact_inventory_v19.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'fit','literal-route','fixed-roster'])]
    r=mod.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode(),capture_output=True,text=True,timeout=90)
    (H/'SCIENTIFIC_LAUNCH_TRANSPORT_V1.json').write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stderr[-2500:]
    obj=json.loads(r.stdout);(H/'SCIENTIFIC_LAUNCH_V1.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(obj))

if __name__=='__main__':main()
