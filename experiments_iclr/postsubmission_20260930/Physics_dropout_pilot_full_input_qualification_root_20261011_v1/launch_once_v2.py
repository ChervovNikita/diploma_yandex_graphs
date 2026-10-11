"""Deploy exact qualified sources and run the two finite execution stages once."""
import base64,hashlib,importlib.util,json,shlex,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=HERE.parent
SOURCE=P/'Physics_dropout_pilot_execution_source_20261011_v1'
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_P=R+'/experiments_iclr/postsubmission_20260930'
manifest=json.loads((SOURCE/'QUALIFICATION_MANIFEST.json').read_text())
files=[]
for path in [*(SOURCE/rel for rel in manifest['files']),SOURCE/'QUALIFICATION_MANIFEST.json',HERE/'ROOT_DECISION.json']:
 b=path.read_bytes();files.append(dict(path=str(path.relative_to(P)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
payload=dict(files=files,source_dir=SOURCE.name,execution_dir=HERE.name)
(HERE/'DEPLOY_PAYLOAD_V2.json').write_text(json.dumps(payload,indent=2)+'\n')
remote=r'''
import base64,hashlib,json,pathlib,socket,subprocess,sys,time,zlib,os
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';S=P/payload['source_dir'];E=P/payload['execution_dir']
assert not S.exists() and not E.exists()
for row in payload['files']:
 p=P/row['path'];assert p.resolve().is_relative_to(P)
 b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
os.chdir(R)
def receipt(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,a):p.write_text(json.dumps(a,indent=2)+'\n')
root=receipt(E/'ROOT_DECISION.json');source=receipt(S/'QUALIFICATION_MANIFEST.json')
base=dict(execution_enabled=True,root_decision=root,finite_external_owner=True,source_manifest=source)
raw=P/'native_SAGE_Physics_confirmation_root_20261011_v1/raw/ms_academic_phy.npz'
limits=dict(wall_seconds=600,cuda_bytes=48*1024**3,rss_bytes=64*1024**3)
a=dict(base,stage='custody',output=str(E/'fresh_roles_v1'),owner_output=str(E/'custody_owner_v1'),limits=limits,raw_archive=dict(path=str(raw),bytes=16374005,sha256='e4d68468eba5fb8f6b280ffd406bd53e6e4295ba1dd2967d6fe99b978e4b3375'))
write(E/'CUSTODY_CONFIG.json',a)
start=time.monotonic();stage='custody';complete=False
try:
 for stage in ('custody','qualify'):
  if stage=='qualify':
   c=dict(base,stage='qualify',output=str(E/'qualification_v1'),owner_output=str(E/'qualification_owner_v1'),limits=dict(limits,wall_seconds=900),custody=receipt(E/'fresh_roles_v1/CUSTODY.json'))
   write(E/'QUALIFY_CONFIG.json',c)
  cfg=E/('CUSTODY_CONFIG.json' if stage=='custody' else 'QUALIFY_CONFIG.json')
  process=subprocess.run([str(R/'.venv/bin/python'),'-B','-u',str(S/'owner.py'),str(cfg)],cwd=R,capture_output=True,text=True,timeout=660 if stage=='custody' else 960)
  write(E/(stage+'_OWNER_TRANSPORT.json'),dict(exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr))
  assert process.returncode==0,stage+' finite owner failed: '+process.stderr
  end=json.loads((pathlib.Path(json.loads(cfg.read_text())['owner_output'])/'OWNER_END.json').read_text());assert end['status']=='COMPLETE_EXIT_0' and end['child_pid_absent'] and end['owned_cuda_pid_absent']
 complete=True
finally:
 status=dict(complete=complete,last_stage=stage,seconds=time.monotonic()-start,scientific_predictive_fits_admitted=False,no_SELECT_CAL_DEV_quality=True)
 write(E/'PIPELINE_END.json',status)
paths=['ROOT_DECISION.json','CUSTODY_CONFIG.json','QUALIFY_CONFIG.json','custody_owner_v1/OWNER_START.json','custody_owner_v1/OWNER_END.json','custody_OWNER_TRANSPORT.json','qualification_owner_v1/OWNER_START.json','qualification_owner_v1/OWNER_END.json','qualify_OWNER_TRANSPORT.json','fresh_roles_v1/CUSTODY.json','qualification_v1/QUALIFICATION.json','PIPELINE_END.json']
rows=[]
for name in paths:
 p=E/name;r=receipt(p);r.update(name=name,data=base64.b64encode(p.read_bytes()).decode());rows.append(r)
print(json.dumps(dict(status=status,files=rows)))
'''
(HERE/'REMOTE_CODE_V2.py.txt').write_text(remote)
spec=importlib.util.spec_from_file_location('terminal_transport',P/'publication/publish_exact_inventory_v19.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,R,'GPU-44039938-fd82-41d2-fefd-de71514e2fac','execute'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',command]
result=module.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode(),capture_output=True,text=True,timeout=1650)
(HERE/'TRANSPORT_V2.json').write_text(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
if result.returncode:
 print(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout[-3000:],stderr=result.stderr[-3000:])));raise SystemExit(result.returncode)
a=json.loads(result.stdout);out=HERE/'actual_receipts';out.mkdir()
for row in a['files']:
 b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 path=out/row['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
print(json.dumps(a['status']))
