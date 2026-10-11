"""Exactly one root-authorized Torch2.1 compatibility qualification retry."""
import base64,hashlib,importlib.util,json,shlex,zlib,datetime
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parent;S=P/'meanSAGE_attribute_peer_feedback_execution_source_20261011_v3';OLD=P/'meanSAGE_attribute_peer_feedback_full_input_qualification_20261011_v1'
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs';RP=R+'/experiments_iclr/postsubmission_20260930'
def receipt(p):
 d=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(d),sha256=hashlib.sha256(d).hexdigest())
assert not (W/'ROOT_ADMISSION.json').exists() and not (W/'TRANSPORT.json').exists()
custody=json.loads((OLD/'actual_receipts/qualification_v1/QUALIFICATION.json').read_text())['role_custody'];assert custody['sha256']=='331680470f36e99b27d58160f271237d1ce5e22090feeac6fd77975f46af1cb4'
admission=dict(status='ROOT_AUTHORIZED_ONE_CORRECTED_SX4_QUALIFICATION_RETRY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authority='Explicit delegated root follow-up authorizes exactly one compatibility retry with same closed fresh-role custody, sourceV3 CPUuniform stream priming only,16GiB CUDA/RSS,900s direct owner. No user approval required; full study remains separate.',permitted_stages=['qualify'],custody_recreation=False,scientific_fits=False,CAL_DEV_scoring=False,automatic_retry=False,extra_qualification_numeric_grid=False,source_manifest=receipt(S/'MANIFEST.json'),failed_qualification=receipt(OLD/'actual_receipts/qualification_v1/QUALIFICATION.json'),failed_owner=receipt(OLD/'actual_receipts/qualification_owner_v1/OWNER_END.json'),closed_custody=custody,closed_custody_owner=receipt(OLD/'actual_receipts/custody_owner_v1/OWNER_END.json'),runtime_versions=dict(numpy='1.26.4',torch='2.1.2+cu118'),cuda_cap_bytes=16*1024**3,rss_cap_bytes=16*1024**3,wall_seconds=900,protected_pids=[635864,635866,636243],timing_context='Overlapping scheduling context; no isolated speed claim')
(W/'ROOT_ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n');manifest=json.loads((S/'MANIFEST.json').read_text());files=[]
for p in [*(S/rel for rel in manifest['files']),S/'MANIFEST.json',S/'SEAL.json',W/'ROOT_ADMISSION.json']:
 d=p.read_bytes();files.append(dict(path=str(p.relative_to(P)),bytes=len(d),sha256=hashlib.sha256(d).hexdigest(),data=base64.b64encode(d).decode()))
payload=dict(files=files,source_dir=S.name,execution_dir=W.name,custody=custody,runtime_versions=admission['runtime_versions'])
(W/'DEPLOY_PAYLOAD.json').write_text(json.dumps(payload,indent=2)+'\n')
remote=r'''
import base64,hashlib,json,pathlib,socket,subprocess,sys,time,zlib,os,datetime
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
assert sys.argv[1:4]==['/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs','GPU-44039938-fd82-41d2-fefd-de71514e2fac','execute']
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=15).strip())*1024**2
available=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert free>=20*1024**3 and available>=20*1024**3,'Insufficient delegated headroom'
R=pathlib.Path(sys.argv[1]);P=R/'experiments_iclr/postsubmission_20260930';S=P/payload['source_dir'];E=P/payload['execution_dir'];assert not S.exists() and not E.exists()
def receipt(p):
 d=p.read_bytes();return dict(path=str(p),bytes=len(d),sha256=hashlib.sha256(d).hexdigest())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
assert receipt(pathlib.Path(payload['custody']['path']))==payload['custody'],'Exact previously closed fresh roles required'
for row in payload['files']:
 p=P/row['path'];assert p.resolve().is_relative_to(P);d=base64.b64decode(row['data']);assert len(d)==row['bytes'] and hashlib.sha256(d).hexdigest()==row['sha256'];p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(d)
os.chdir(R);write(E/'PRE_STAGE_RESOURCES.json',dict(hostname=socket.gethostname(),gpu_uuid=sys.argv[2],GPU_free_bytes=free,host_MemAvailable_bytes=available,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
config=dict(execution_enabled=True,stage='qualify',root_decision=receipt(E/'ROOT_ADMISSION.json'),finite_external_owner=True,source_manifest=receipt(S/'MANIFEST.json'),expected_runtime_versions=payload['runtime_versions'],blocks=[10103,10427,10949],arms=['P1','F1','I4','S4','PA1','IA4','SA4','SM4','SX4','UX4'],custody=payload['custody'],output=str(E/'qualification_v1'),owner_output=str(E/'qualification_owner_v1'),limits=dict(wall_seconds=900,cuda_bytes=16*1024**3,rss_bytes=16*1024**3))
write(E/'QUALIFY_CONFIG.json',config);start=time.monotonic();complete=False;failure=None
try:
 process=subprocess.run([str(R/'.venv/bin/python'),'-B','-u',str(S/'owner.py'),str(E/'QUALIFY_CONFIG.json')],cwd=R,capture_output=True,text=True,timeout=960)
 write(E/'qualify_OWNER_TRANSPORT.json',dict(exit_code=process.returncode,stdout=process.stdout,stderr=process.stderr));assert process.returncode==0,'Corrected qualifier finite owner failed'
 end=json.loads((E/'qualification_owner_v1/OWNER_END.json').read_text());assert end['status']=='COMPLETE_EXIT_0' and end['child_pid_absent'] and end['owned_cuda_pid_absent'];complete=True
except BaseException as error:failure=dict(type=type(error).__name__,message=str(error))
finally:
 status=dict(complete=complete,seconds=time.monotonic()-start,failure=failure,automatic_retry=False,custody_recreated=False,scientific_fits_admitted=False,CAL_DEV_scoring=False,protected_pids_untouched=[635864,635866,636243],timing_context='Overlapping scheduling context; no isolated speed claim');write(E/'PIPELINE_END.json',status)
rows=[]
for name in ['ROOT_ADMISSION.json','PRE_STAGE_RESOURCES.json','QUALIFY_CONFIG.json','qualification_owner_v1/OWNER_START.json','qualification_owner_v1/OWNER_END.json','qualify_OWNER_TRANSPORT.json','qualification_v1/QUALIFICATION.json','PIPELINE_END.json']:
 p=E/name
 if p.exists():r=receipt(p);r.update(name=name,data=base64.b64encode(p.read_bytes()).decode());rows.append(r)
print(json.dumps(dict(status=status,files=rows,whole_source_identity=receipt(S/'MANIFEST.json'),closed_custody=payload['custody'])))
'''
(W/'REMOTE_CODE.py.txt').write_text(remote)
spec=importlib.util.spec_from_file_location('reviewed_peer_pty',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
command=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,R,'GPU-44039938-fd82-41d2-fefd-de71514e2fac','execute'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',command]
r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode(),capture_output=True,text=True,timeout=1050)
(W/'TRANSPORT.json').write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
if r.returncode:print(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout[-3000:],stderr=r.stderr[-3000:])));raise SystemExit(r.returncode)
v=json.loads(r.stdout);out=W/'actual_receipts';out.mkdir()
for row in v['files']:
 d=base64.b64decode(row['data']);assert len(d)==row['bytes'] and hashlib.sha256(d).hexdigest()==row['sha256'];p=out/row['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(d)
(W/'REMOTE_SOURCE_IDENTITY.json').write_text(json.dumps(v['whole_source_identity'],indent=2)+'\n');(W/'REUSED_CUSTODY.json').write_text(json.dumps(v['closed_custody'],indent=2)+'\n');print(json.dumps(v['status'],indent=2));raise SystemExit(0 if v['status']['complete'] else 1)
