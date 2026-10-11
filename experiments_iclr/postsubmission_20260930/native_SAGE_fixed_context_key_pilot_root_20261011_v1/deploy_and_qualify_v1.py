"""Deploy one reviewed immutable packet and bound its TRAIN-only qualifier."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

P=Path('postsubmission_research_20260930').resolve()
H=P/'native_SAGE_fixed_context_key_pilot_root_20261011_v1'
S=P/'native_SAGE_fixed_context_key_source_20261011_v1'
REMOTE=r"""import base64,hashlib,json,os,socket,subprocess,sys,time,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
H=P/'native_SAGE_fixed_context_key_pilot_root_20261011_v1'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='a4449c7f6915d6318bb2c44e784d7fdcea911342'
for row in payload:
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 q=P/rel;assert q.resolve().is_relative_to(P)
 data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
 q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==data,q
 else:q.write_bytes(data)
f=json.loads((H/'QUALIFICATION_FREEZE.json').read_text())
assert f['independent_source_review_passed'] and f['TRAIN_only']
for row in f['bound_files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
assert not (H/'QUALIFICATION_EXECUTION_V1.json').exists() and not (H/'actual_family_v1').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
started=time.monotonic()
try:
 r=subprocess.run([str(R/'.venv/bin/python'),'-B',str(H/'qualify.py')],cwd=R,env=env,capture_output=True,text=True,timeout=210)
 v={'exit_code':r.returncode,'seconds':time.monotonic()-started,'stdout':r.stdout,'stderr':r.stderr,'hard_timeout_seconds':210,'TRAIN_only':True,'TEST_access':False}
except subprocess.TimeoutExpired as error:
 v={'exit_code':-9,'seconds':time.monotonic()-started,'hard_timeout_seconds':210,'TRAIN_only':True,'TEST_access':False,'failure':'Hard qualifier timeout; subprocess.run killed and waited for direct qualifier child','stdout':str(error.stdout),'stderr':str(error.stderr)}
(H/'QUALIFICATION_EXECUTION_V1.json').write_text(json.dumps(v,indent=2)+'\n')
q=H/'ACTUAL_QUALIFICATION_V1.json'
print(json.dumps({'execution':v,'qualification':json.loads(q.read_text()) if q.exists() else None}))
"""

def main():
    frozen=json.loads((H/'QUALIFICATION_FREEZE.json').read_text())
    assert frozen['independent_source_review_passed']
    paths=[f for f in S.iterdir() if f.is_file()]
    paths += [H/name for name in ['CONFIG.json','qualify.py','QUALIFICATION_FREEZE.json']]
    paths += [P/'native_SAGE_fixed_context_key_prospective_root_20261011_v1'/name for name in ['PROSPECTIVE_PROTOCOL.json','DECISION.md']]
    paths += [f for f in (P/'native_SAGE_fixed_context_key_independent_source_review_20261011_v1').iterdir() if f.is_file()]
    rows=[]
    for path in paths:
        data=path.read_bytes()
        rows.append({'path':str(path.relative_to(P)),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(data).decode()})
    spec=importlib.util.spec_from_file_location('reviewed_literal_transport',P/'publication/publish_exact_inventory_v19.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE,'qualify','literal-route','TRAIN-only'])]
    result=transport.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode(),capture_output=True,text=True,timeout=240)
    (H/'QUALIFICATION_TRANSPORT_V1.json').write_text(json.dumps({'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2)+'\n')
    assert result.returncode==0,result.stderr[-2000:]
    value=json.loads(result.stdout)
    (H/'QUALIFICATION_EXECUTION_V1.json').write_text(json.dumps(value['execution'],indent=2)+'\n')
    if value['qualification'] is not None:
        (H/'ACTUAL_QUALIFICATION_V1.json').write_text(json.dumps(value['qualification'],indent=2)+'\n')
    assert value['execution']['exit_code']==0 and value['qualification']['qualified']
    q=value['qualification']
    print(json.dumps({'qualified':True,'units':len(q['units']),'TRAIN_trajectories':q['native_TRAIN_trajectories'],'seconds':q['elapsed_seconds'],'CUDA_peak_allocated_bytes':q['root_CUDA_peak_allocated_bytes'],'scientific_fits':0,'VALID_quality_scored':False,'TEST_access':False}))

if __name__=='__main__':main()
