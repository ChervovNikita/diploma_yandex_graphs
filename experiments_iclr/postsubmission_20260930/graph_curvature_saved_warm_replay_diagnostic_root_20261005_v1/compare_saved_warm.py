"""Compare two existing engineering warm checkpoints; no fits or score access."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
ROUTE = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REMOTE_PHASE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930'
PYTHON = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/bin/python'
CODE = r'''
from pathlib import Path
import hashlib,json,os,subprocess,time
assert os.environ['GNNM_SSH_DESTINATION']=='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
phase=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
assert Path.cwd()==phase
rows=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout.splitlines()
assert rows==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
started=time.perf_counter()
import torch
paths=[phase/'graph_curvature_selector_native_engineering_execution_20261004_v1/Squirrel/FRESH_NATIVE_WARM.pt',phase/'graph_curvature_selector_native_diagnostic_execution_20261004_v1/Squirrel/FRESH_NATIVE_WARM.pt']
for p in paths: assert p.resolve()==p and p.is_file() and p.stat().st_size<100000000
values=[torch.load(p,map_location='cpu') for p in paths]
differences=[]
def compare(a,b,path):
 if type(a) is not type(b): differences.append(dict(path=path,kind='type'));return
 if torch.is_tensor(a):
  if a.shape!=b.shape or a.dtype!=b.dtype: differences.append(dict(path=path,kind='tensor_metadata'));return
  if not torch.equal(a,b):
   x=a!=b
   row=dict(path=path,kind='tensor_values',dtype=str(a.dtype),shape=list(a.shape),different_elements=int(x.sum()),elements=a.numel())
   if a.is_floating_point():row.update(max_abs_difference=float((a.double()-b.double()).abs().max()),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
   differences.append(row)
 elif isinstance(a,dict):
  if a.keys()!=b.keys():differences.append(dict(path=path,kind='keys'));return
  for k in a:compare(a[k],b[k],path+'['+repr(k)+']')
 elif isinstance(a,(list,tuple)):
  if len(a)!=len(b):differences.append(dict(path=path,kind='length'));return
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'['+str(i)+']')
 elif a!=b:differences.append(dict(path=path,kind='scalar'))
for k in values[0]:compare(values[0][k],values[1][k],k)
groups={}
for row in differences:
 group=row['path'].split('[')[0];groups[group]=groups.get(group,0)+1
print(json.dumps(dict(schema='saved-native-warm-comparison-v1',equal=not differences,difference_count=len(differences),difference_groups=groups,first_differences=differences[:30],files=[dict(path=str(p.relative_to(phase)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],torch=torch.__version__,wall_seconds=time.perf_counter()-started,new_training=False,new_forward=False,predictive_values_exposed=False,signals_sent=False)))
'''

def main():
    destination = HERE/'RESULT.json'
    assert not destination.exists()
    command = 'cd '+shlex.quote(REMOTE_PHASE)+' && '+shlex.join(['env','GNNM_SSH_DESTINATION='+ROUTE,'PYTHONDONTWRITEBYTECODE=1',PYTHON,'-B','-'])
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',ROUTE,command],input=CODE,capture_output=True,text=True,timeout=60)
    receipt = dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stderr=result.stderr,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),route=ROUTE)
    if result.returncode == 0:
        receipt['result'] = json.loads(result.stdout)
    else:
        receipt['stdout'] = result.stdout
    with destination.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2))
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
