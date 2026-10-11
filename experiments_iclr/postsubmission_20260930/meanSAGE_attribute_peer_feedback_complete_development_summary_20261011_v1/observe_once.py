"""Read only fixed peer owner/watch identities, lifecycle receipts and counts."""
import base64,datetime,importlib.util,json,pathlib,shlex,zlib
HERE=pathlib.Path(__file__).resolve().parent
P=HERE.parent
R='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
remote=r"""
import base64,datetime,hashlib,json,pathlib,socket,subprocess,sys,zlib
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
E=P/'meanSAGE_attribute_peer_feedback_acquisition_root_20261011_v1';W=P/'meanSAGE_attribute_peer_feedback_readout_after_whole_acquisition_root_20261011_v1'/'watch_v1'
def receipt(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def metadata(p):
 if not p.exists():return dict(path=str(p),exists=False)
 d=json.loads(p.read_text());return dict(exists=True,receipt=receipt(p),value=d)
def proc(pid):
 p=pathlib.Path('/proc')/str(pid)/'stat'
 if not p.exists():return dict(pid=pid,observed_present=False)
 text=p.read_text();fields=text.rsplit(')',1)[1].split();return dict(pid=pid,observed_present=True,state=fields[0],start_ticks=int(fields[19]),pgid=int(fields[2]))
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),route_verified=True,host='anogena-2-0',sole_UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac',specific_processes=[proc(p) for p in [636808,636810,637442]],predictive_metrics_opened=False,observation_only=True)
paths=dict(acquisition_progress=E/'acquisition_v1/ACQUISITION_PROGRESS.json',acquisition_owner_end=E/'owner_v1/OWNER_END.json',watch_start=W/'WATCH_START.json',watch_progress=W/'WATCH_PROGRESS.json',whole_family_closure=W/'WHOLE_FAMILY_CLOSURE.json',all_CAL_closure=W/'ALL_CAL_CLOSURE.json',CAL_owner_end=W/'calibrate_owner_v1/OWNER_END.json',score_owner_end=W/'score_owner_v1/OWNER_END.json',watch_end=W/'WATCH_END.json')
out['metadata']={name:metadata(p) for name,p in paths.items()}
print(json.dumps(out))
"""
spec=importlib.util.spec_from_file_location('peer_transport_v19',P/'publication/publish_exact_inventory_v19.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cmd=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,R,UUID,'observe'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',cmd]
name=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
try:
 r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(b'{}')).decode(),capture_output=True,text=True,timeout=45)
 record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr,transport_only=True)
 if r.returncode:record['stdout']=r.stdout
 else:
  result=json.loads(r.stdout);record['observation']=result
except BaseException as error:
 record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),observation_timeout_or_transport_error=dict(type=type(error).__name__,message=str(error)),scientific_failure_inferred=False)
with (HERE/'observations'/(name+'.json')).open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
if 'observation' not in record:
 print(json.dumps({k:v for k,v in record.items() if k not in ['stderr','stdout']}));raise SystemExit(1)
a=record['observation'];minimal=dict(UTC=a['UTC'],processes=a['specific_processes'],observation_file=str(HERE/'observations'/(name+'.json')))
for name,row in a['metadata'].items():
 if row['exists']:
  value=row['value'];minimal[name]={k:v for k,v in value.items() if k in ['closed_banks','optimizer_trajectories','physical_backbones','expected_banks','complete','last_stage','stage','status','exit_code','child_pid_absent','owned_cuda_pid_absent','failure','all30banks_48states_57bodies_before30CAL_then_onewholeDEV','all_CAL_owners_closed_before_DEV','endpoints','scalar_updates','banks','exports']}
print(json.dumps(minimal,indent=2))
