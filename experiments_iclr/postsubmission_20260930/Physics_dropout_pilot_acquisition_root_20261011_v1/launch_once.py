"""Launch one complete fixed scientific panel; no partial quality reads."""
import base64,hashlib,importlib.util,json,shlex,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;P=HERE.parent;SOURCE=P/'Physics_dropout_pilot_execution_source_20261011_v2'
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
manifest=json.loads((SOURCE/'ACQUISITION_MANIFEST.json').read_text());files=[]
for path in [*(SOURCE/rel for rel in manifest['files']),SOURCE/'ACQUISITION_MANIFEST.json',HERE/'ROOT_DECISION.json']:
 b=path.read_bytes();files.append(dict(path=str(path.relative_to(P)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
Q=P/'Physics_dropout_pilot_full_input_qualification_root_20261011_v1/actual_receipts'
expected={rel:hashlib.sha256((Q/rel).read_bytes()).hexdigest() for rel in ['fresh_roles_v1/CUSTODY.json','qualification_v1/QUALIFICATION.json','qualification_owner_v1/OWNER_END.json']}
payload=dict(files=files,source_dir=SOURCE.name,execution_dir=HERE.name,qualified_receipt_hashes=expected)
(HERE/'DEPLOY_PAYLOAD.json').write_text(json.dumps(payload,indent=2)+'\n')
remote=r'''
import base64,hashlib,json,pathlib,socket,subprocess,sys,time,zlib,os
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';S=P/payload['source_dir'];E=P/payload['execution_dir'];Q=P/'Physics_dropout_pilot_full_input_qualification_root_20261011_v1'
assert not S.exists() and not E.exists()
def receipt(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
for rel,sha in payload['qualified_receipt_hashes'].items():assert receipt(Q/rel)['sha256']==sha
for row in payload['files']:
 p=P/row['path'];assert p.resolve().is_relative_to(P)
 b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
os.chdir(R)
def write(p,a):p.write_text(json.dumps(a,indent=2)+'\n')
config=dict(execution_enabled=True,stage='acquire',root_decision=receipt(E/'ROOT_DECISION.json'),finite_external_owner=True,source_manifest=receipt(S/'ACQUISITION_MANIFEST.json'),custody=receipt(Q/'fresh_roles_v1/CUSTODY.json'),output=str(E/'acquisition_v1'),owner_output=str(E/'owner_v1'),limits=dict(wall_seconds=43200,cuda_bytes=48*1024**3,rss_bytes=64*1024**3),qualification=dict(result=receipt(Q/'qualification_v1/QUALIFICATION.json'),owner_end=receipt(Q/'qualification_owner_v1/OWNER_END.json')),qualification_manifest=receipt(P/'Physics_dropout_pilot_execution_source_20261011_v1/QUALIFICATION_MANIFEST.json'))
write(E/'CONFIG.json',config)
command=[str(R/'.venv/bin/python'),'-B','-u',str(S/'owner.py'),str(E/'CONFIG.json')]
with (E/'owner_launcher.log').open('xb') as log:
 owner=subprocess.Popen(command,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 deadline=time.monotonic()+15
 while not (E/'owner_v1/OWNER_START.json').exists() and owner.poll() is None and time.monotonic()<deadline:time.sleep(.05)
 assert owner.poll() is None and (E/'owner_v1/OWNER_START.json').exists(),'Owner failed before start; see retained project log'
 start=json.loads((E/'owner_v1/OWNER_START.json').read_text());assert start['direct_child_wait'] and start['owner']['pid']==owner.pid
launch=dict(status='RUNNING',owner_pid=owner.pid,child_pid=start['child']['pid'],banks=42,bodies=69,CAL=42,source_manifest=config['source_manifest'],config=receipt(E/'CONFIG.json'),partial_quality_opened=False,CAL_DEV_truth_read=False,command=command)
write(E/'LAUNCH.json',launch)
rows=[]
for rel in ['ROOT_DECISION.json','CONFIG.json','LAUNCH.json','owner_v1/OWNER_START.json']:
 p=E/rel;r=receipt(p);r.update(name=rel,data=base64.b64encode(p.read_bytes()).decode());rows.append(r)
print(json.dumps(dict(launch=launch,files=rows)))
'''
(HERE/'REMOTE_CODE.py.txt').write_text(remote)
spec=importlib.util.spec_from_file_location('terminal_transport',P/'publication/publish_exact_inventory_v19.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,R,'GPU-44039938-fd82-41d2-fefd-de71514e2fac','execute'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',command]
result=module.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode(),capture_output=True,text=True,timeout=90)
(HERE/'TRANSPORT.json').write_text(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
if result.returncode:
 print(json.dumps(dict(exit_code=result.returncode,stdout=result.stdout[-3000:],stderr=result.stderr[-3000:])));raise SystemExit(result.returncode)
a=json.loads(result.stdout);out=HERE/'actual_receipts';out.mkdir()
for row in a['files']:
 b=base64.b64decode(row['data']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 path=out/row['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
print(json.dumps(a['launch']))
