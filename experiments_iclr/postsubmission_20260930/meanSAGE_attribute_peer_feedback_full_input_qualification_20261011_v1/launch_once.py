"""One root-authorized custody and one full-input SX4 qualification; no retry."""
import base64,hashlib,importlib.util,json,shlex,zlib,datetime
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parent;S=P/'meanSAGE_attribute_peer_feedback_execution_source_20261011_v2'
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
def receipt(p):
 d=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(d),sha256=hashlib.sha256(d).hexdigest())
probe=json.loads((W/'RESOURCE_RUNTIME_PROBE.json').read_text());assert probe['route_verified'] and probe['GPU_free_bytes']>=20*1024**3 and probe['host_MemAvailable_bytes']>=20*1024**3
assert not (W/'ROOT_ADMISSION.json').exists() and not (W/'TRANSPORT.json').exists()
admission=dict(status='ROOT_AUTHORIZED_CUSTODY_AND_ONE_SX4_QUALIFICATION_ONLY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authority='Explicit delegated root instruction: preserve V1, seal replay-tolerance repair V2, deploy fresh roles and exactly one full-input SX4 qualification on literal authorized allocation. Root admits full study separately.',permitted_stages=['custody','qualify'],scientific_fits=False,CAL_or_DEV_scoring=False,automatic_retry=False,extra_qualification_or_numeric_grid=False,source_manifest=receipt(S/'MANIFEST.json'),proposal=receipt(P/'shared_attribute_incidence_feedback_architecture_scout_20261011_v1/STUDY_PROPOSAL.json'),source_assessment=receipt(P/'native_meanSAGE_attribute_peer_feedback_source_assessment_20261011_v1/SOURCE_ASSESSMENT.json'),runtime_binding=receipt(P/'meanSAGE_attribute_peer_feedback_runtime_binding_20261011_v1/BINDING.json'),resource_runtime_probe=receipt(W/'RESOURCE_RUNTIME_PROBE.json'),runtime_versions=probe['runtime_versions'],cuda_cap_bytes=16*1024**3,rss_cap_bytes=16*1024**3,custody_wall_seconds=600,qualification_wall_seconds=900,protected_pids=[635864,635866,636243],timing_context='Overlapping scheduling with sign-dropout family/watcher by root instruction; protected PID snapshot contains no entries. No isolated speed claim.',qualification_scope='Frozen qualifier constructor ownership/count inspections plus ONE SX4 full-input TRAIN update/strict restore/label-free SELECT; no extra update or predictive metric.')
(W/'ROOT_ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
manifest=json.loads((S/'MANIFEST.json').read_text());files=[]
for p in [*(S/rel for rel in manifest['files']),S/'MANIFEST.json',S/'SEAL.json',W/'ROOT_ADMISSION.json',W/'RESOURCE_RUNTIME_PROBE.json']:
 d=p.read_bytes();files.append(dict(path=str(p.relative_to(P)),bytes=len(d),sha256=hashlib.sha256(d).hexdigest(),data=base64.b64encode(d).decode()))
payload=dict(files=files,source_dir=S.name,execution_dir=W.name,runtime_versions=probe['runtime_versions'])
(W/'DEPLOY_PAYLOAD.json').write_text(json.dumps(payload,indent=2)+'\n')
remote=r'''
import base64,hashlib,json,pathlib,socket,subprocess,sys,time,zlib,os,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
assert sys.argv[1:4]==['/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs','GPU-44039938-fd82-41d2-fefd-de71514e2fac','execute']
R=pathlib.Path(sys.argv[1]);P=R/'experiments_iclr/postsubmission_20260930';S=P/payload['source_dir'];E=P/payload['execution_dir']
assert not S.exists() and not E.exists(),'Preserve staging collision, never replace/retry'
for row in payload['files']:
 p=P/row['path'];assert p.resolve().is_relative_to(P)
 d=base64.b64decode(row['data']);assert len(d)==row['bytes'] and hashlib.sha256(d).hexdigest()==row['sha256'];p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(d)
os.chdir(R)
def receipt(p):
 d=p.read_bytes();return dict(path=str(p),bytes=len(d),sha256=hashlib.sha256(d).hexdigest())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def resources(stage):
 assert socket.gethostname()=='anogena-2-0'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[sys.argv[2]]
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip())*1024**2
 available=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 assert free>=20*1024**3 and available>=20*1024**3,'Insufficient root delegated headroom'
 return dict(stage=stage,GPU_free_bytes=free,host_MemAvailable_bytes=available,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
base=dict(execution_enabled=True,root_decision=receipt(E/'ROOT_ADMISSION.json'),finite_external_owner=True,source_manifest=receipt(S/'MANIFEST.json'),expected_runtime_versions=payload['runtime_versions'],blocks=[10103,10427,10949],arms=['P1','F1','I4','S4','PA1','IA4','SA4','SM4','SX4','UX4'])
raw=P/'native_SAGE_Physics_confirmation_root_20261011_v1/raw/ms_academic_phy.npz'
limits=dict(wall_seconds=600,cuda_bytes=16*1024**3,rss_bytes=16*1024**3)
config=dict(base,stage='custody',output=str(E/'fresh_roles_v1'),owner_output=str(E/'custody_owner_v1'),limits=limits,raw_archive=dict(path=str(raw),bytes=16374005,sha256='e4d68468eba5fb8f6b280ffd406bd53e6e4295ba1dd2967d6fe99b978e4b3375'))
write(E/'CUSTODY_CONFIG.json',config)
start=time.monotonic();stage='custody';complete=False;failure=None;observations=[]
try:
 for stage in ('custody','qualify'):
  observations.append(resources(stage));write(E/'PRE_STAGE_RESOURCES.json',observations)
  if stage=='qualify':
   config=dict(base,stage='qualify',output=str(E/'qualification_v1'),owner_output=str(E/'qualification_owner_v1'),limits=dict(limits,wall_seconds=900),custody=receipt(E/'fresh_roles_v1/CUSTODY.json'))
   write(E/'QUALIFY_CONFIG.json',config)
  path=E/('CUSTODY_CONFIG.json' if stage=='custody' else 'QUALIFY_CONFIG.json')
  process=subprocess.run([str(R/'.venv/bin/python'),'-B','-u',str(S/'owner.py'),str(path)],cwd=R,capture_output=True,text=True,timeout=660 if stage=='custody' else 960)
  write(E/(stage+'_OWNER_TRANSPORT.json'),dict(exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr))
  assert process.returncode==0,stage+' finite owner failed'
  end=json.loads((pathlib.Path(config['owner_output'])/'OWNER_END.json').read_text());assert end['status']=='COMPLETE_EXIT_0' and end['child_pid_absent'] and end['owned_cuda_pid_absent']
 complete=True
except BaseException as error:failure=dict(type=type(error).__name__,message=str(error))
finally:
 status=dict(complete=complete,last_stage=stage,seconds=time.monotonic()-start,failure=failure,automatic_retry=False,scientific_fits_admitted=False,CAL_DEV_scoring=False,timing_context='Overlapping scheduling context; no isolated speed claim',protected_pids_untouched=[635864,635866,636243]);write(E/'PIPELINE_END.json',status)
paths=['ROOT_ADMISSION.json','RESOURCE_RUNTIME_PROBE.json','CUSTODY_CONFIG.json','QUALIFY_CONFIG.json','PRE_STAGE_RESOURCES.json','custody_owner_v1/OWNER_START.json','custody_owner_v1/OWNER_END.json','custody_OWNER_TRANSPORT.json','qualification_owner_v1/OWNER_START.json','qualification_owner_v1/OWNER_END.json','qualify_OWNER_TRANSPORT.json','fresh_roles_v1/CUSTODY.json','qualification_v1/QUALIFICATION.json','PIPELINE_END.json']
rows=[]
for name in paths:
 p=E/name
 if p.exists():r=receipt(p);r.update(name=name,data=base64.b64encode(p.read_bytes()).decode());rows.append(r)
print(json.dumps(dict(status=status,files=rows,whole_source_identity=receipt(S/'MANIFEST.json'))))
'''
(W/'REMOTE_CODE.py.txt').write_text(remote)
spec=importlib.util.spec_from_file_location('reviewed_peer_pty',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,R,'GPU-44039938-fd82-41d2-fefd-de71514e2fac','execute'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',command]
r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode(),capture_output=True,text=True,timeout=1650)
(W/'TRANSPORT.json').write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
if r.returncode:print(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout[-3000:],stderr=r.stderr[-3000:])));raise SystemExit(r.returncode)
v=json.loads(r.stdout);out=W/'actual_receipts';out.mkdir()
for row in v['files']:
 d=base64.b64decode(row['data']);assert len(d)==row['bytes'] and hashlib.sha256(d).hexdigest()==row['sha256'];p=out/row['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(d)
(W/'REMOTE_SOURCE_IDENTITY.json').write_text(json.dumps(v['whole_source_identity'],indent=2)+'\n');print(json.dumps(v['status'],indent=2))
raise SystemExit(0 if v['status']['complete'] else 1)
