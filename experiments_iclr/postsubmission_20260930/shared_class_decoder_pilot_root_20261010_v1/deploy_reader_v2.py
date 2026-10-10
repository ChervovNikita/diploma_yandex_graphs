"""Copy reviewed reader source; read only closure metadata from the live fit."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import zlib

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'shared_class_decoder_complete_reader_20261010_v2'
names = ['analysis.py', 'README.md', 'STATIC_VERIFICATION.json', 'SOURCE.json']
payload = []
for name in names:
    data = (SOURCE / name).read_bytes()
    payload.append(dict(name=name, sha256=hashlib.sha256(data).hexdigest(), data=base64.b64encode(data).decode()))
encoded = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
remote = r'''
import base64,hashlib,json,socket,subprocess,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
u='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[u]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
S=P/'shared_class_decoder_complete_reader_20261010_v2'
rows=json.loads(zlib.decompress(base64.b64decode(ENCODED)))
assert [r['name'] for r in rows]==['analysis.py','README.md','STATIC_VERIFICATION.json','SOURCE.json']
S.mkdir(exist_ok=True)
for row in rows:
    data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==row['sha256']
    target=S/row['name']
    if target.exists():
        assert target.read_bytes()==data
    else:
        with target.open('xb') as f:f.write(data)
H=P/'shared_class_decoder_pilot_root_20261010_v1'
end=H/'OWNER_END.json'
f=H/'actual_family_v1/PROGRESS.json'
print('DEPLOY_JSON='+json.dumps(dict(files=[{k:r[k] for k in ['name','sha256']} for r in rows],
    owner_end=json.loads(end.read_text()) if end.exists() else None,
    progress=json.loads(f.read_text()) if f.exists() else None)))
'''.replace('ENCODED', repr(encoded))
args=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
      '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes',
      '-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
      'python3 -c '+shlex.quote(remote)]
p=subprocess.run(args,text=True,capture_output=True)
assert p.returncode==0,p.stderr[-2000:]
lines=[s for s in p.stdout.splitlines() if s.startswith('DEPLOY_JSON=')]
assert len(lines)==1
value=json.loads(lines[0].split('=',1)[1])
(HERE/'READER_DEPLOY_V2.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
