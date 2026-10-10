"""Execute the bound nine-unit TRAIN qualifier on the sole authorized GPU."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
P = HERE.parent
folders = ['native_SAGE_GNCL_SupCon_2x2_source_20261010_v1',
           'native_SAGE_GNCL_SupCon_2x2_independent_source_review_20261010_v1']
paths = []
for name in folders:
    paths.extend(q for q in (P / name).iterdir() if q.is_file() and q.suffix in {'.py', '.json', '.md'})
paths.extend(HERE / name for name in ['qualify.py', 'CONFIG.json', 'DECISION.md', 'PROSPECTIVE_PROTOCOL.json', 'QUALIFICATION_FREEZE.json'])
rows = []
for q in paths:
    data = q.read_bytes()
    rows.append(dict(path=str(q.relative_to(P)), sha256=hashlib.sha256(data).hexdigest(), data=base64.b64encode(data).decode()))
encoded = base64.b64encode(zlib.compress(json.dumps(rows).encode())).decode()
remote = r'''
import base64,hashlib,json,os,socket,subprocess,time,zlib
from datetime import datetime,timezone
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
u='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[u]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930';H=P/'SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='763553fed7770190525b84ae9bb052789c7d459d'
for row in json.loads(zlib.decompress(base64.b64decode(ENCODED))):
    rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
    q=P/rel;assert q.resolve().is_relative_to(P)
    data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
    q.parent.mkdir(parents=True,exist_ok=True)
    if q.exists():assert q.read_bytes()==data
    else:
        with q.open('xb') as f:f.write(data)
receipt=H/'QUALIFICATION_EXECUTION_V1.json'
if not receipt.exists():
    assert not (H/'actual_qualification_v1').exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=u,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    start=time.monotonic();utc=datetime.now(timezone.utc).isoformat()
    proc=subprocess.run([str(R/'.venv/bin/python'),'-B',str(H/'qualify.py')],cwd=R,env=env,text=True,capture_output=True,timeout=180)
    value=dict(start_UTC=utc,terminal_UTC=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,
               qualification_freeze_sha256=hashlib.sha256((H/'QUALIFICATION_FREEZE.json').read_bytes()).hexdigest(),TEST_access=False)
    receipt.write_text(json.dumps(value,indent=2)+'\n')
value=json.loads(receipt.read_text())
q=H/'actual_qualification_v1/QUALIFICATION.json'
qualification=json.loads(q.read_text()) if q.exists() else None
prior={'already_verified_head':'763553fed7770190525b84ae9bb052789c7d459d'}
print('QUALIFICATION_JSON='+json.dumps(dict(execution=value,qualification=qualification,prior_push_receipt=prior)))
'''.replace('ENCODED', repr(encoded))
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
      '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
      '-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
      'python3 -c '+shlex.quote(remote)]
proc=subprocess.run(args,text=True,capture_output=True)
assert proc.returncode==0,proc.stderr[-2000:]
lines=[s for s in proc.stdout.splitlines() if s.startswith('QUALIFICATION_JSON=')]
assert len(lines)==1
value=json.loads(lines[0].split('=',1)[1])
(HERE/'QUALIFICATION_EXECUTION_V1.json').write_text(json.dumps(value['execution'],indent=2)+'\n')
if value['qualification'] is not None:
    (HERE/'ACTUAL_QUALIFICATION_V1.json').write_text(json.dumps(value['qualification'],indent=2)+'\n')

print(json.dumps(dict(execution=value['execution'],qualification=value['qualification']),indent=2))
