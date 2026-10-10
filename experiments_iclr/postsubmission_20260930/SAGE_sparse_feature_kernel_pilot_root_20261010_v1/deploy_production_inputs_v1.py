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
folders = ['native_SAGE_sparse_feature_kernel_source_20261010_v1',
           'native_SAGE_sparse_feature_kernel_source_review_20261010_v1']
paths = []
for name in folders:
    paths.extend(q for q in (P / name).iterdir() if q.is_file() and q.suffix in {'.py', '.json', '.md', '.patch'})
paths.extend(HERE / name for name in ['qualify.py', 'CONFIG.json', 'DECISION.md', 'QUALIFICATION_FREEZE.json','CALIBRATION_POLICY.json','PROSPECTIVE_PROTOCOL.json','run.py','owner.py','admit_references_and_calibration_v1.py'])
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
P=R/'experiments_iclr/postsubmission_20260930';H=P/'SAGE_sparse_feature_kernel_pilot_root_20261010_v1'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='ab8419011cebd09beedc35f3e756baab95995b7e'
for row in json.loads(zlib.decompress(base64.b64decode(ENCODED))):
    rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
    q=P/rel;assert q.resolve().is_relative_to(P)
    data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
    q.parent.mkdir(parents=True,exist_ok=True)
    if q.exists():assert q.read_bytes()==data
    else:
        with q.open('xb') as f:f.write(data)
print('DEPLOY_OK')
'''.replace('ENCODED', repr(encoded))
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
      '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
      '-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
      'python3 -c '+shlex.quote(remote)]
proc=subprocess.run(args,text=True,capture_output=True)
assert proc.returncode==0,proc.stderr[-2000:]
print(proc.stdout.strip())
