import base64,hashlib,importlib.util,json,shlex,zlib
from pathlib import Path
P=Path('postsubmission_research_20260930').resolve();H=P/'SAGE_continuous_context_affinity_pilot_root_20261011_v2'
rows=[]
for name in ['REFERENCE_ADMISSION.json','CALIBRATION_REUSE_ADMISSION.json','PROSPECTIVE_PROTOCOL.json','SCIENTIFIC_ADMISSION.json','QUALIFICATION_FREEZE.json','CALIBRATION_POLICY.json','ACTUAL_QUALIFICATION_V1.json','FREEZE.json','run.py','owner.py']:
 p=H/name;b=p.read_bytes();rows.append({'path':str(p.relative_to(P)),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
REMOTE="""import base64,hashlib,json,socket,subprocess,sys,zlib
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';H=P/'SAGE_continuous_context_affinity_pilot_root_20261011_v2'
for row in payload:
 q=P/row['path'];assert q.resolve().is_relative_to(P);b=base64.b64decode(row['data']);assert hashlib.sha256(b).hexdigest()==row['sha256']
 if q.exists():assert q.read_bytes()==b,q
 else:q.write_bytes(b)
f=json.loads((H/'FREEZE.json').read_text())
for row in f['bound_files']:assert hashlib.sha256((R/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
assert not (H/'actual_family_v1').exists() and not (H/'OWNER_START.json').exists()
print(json.dumps({'deployed':len(payload),'freeze_bound_files':len(f['bound_files']),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'gpu_resources':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines(),'scientific_fit_started':False}))
"""
spec=importlib.util.spec_from_file_location('reviewed_stream_transport',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
args=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
r=m.terminal_transport(args,input=base64.b64encode(zlib.compress(json.dumps(rows).encode(),9)).decode(),capture_output=True,text=True,timeout=70)
v={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(H/'DEPLOYMENT_V3.json').write_text(json.dumps(v,indent=2)+'\n');print(r.stdout[-1500:]);assert r.returncode==0,r.stderr[-1000:]
