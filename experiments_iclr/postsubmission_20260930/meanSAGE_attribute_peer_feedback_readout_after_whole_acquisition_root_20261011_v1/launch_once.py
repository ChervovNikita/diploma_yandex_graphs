"""Install and start the fixed peer study's own complete-family readout."""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import zlib

H=Path(__file__).resolve().parent
P=H.parent
S=P/'peer_readout_source_20261011_v1'
manifest=json.loads((S/'MANIFEST.json').read_text())
manifest_sha=hashlib.sha256((S/'MANIFEST.json').read_bytes()).hexdigest()
assert manifest_sha=='dc67704999a131d6bcb7ca16ef4a7ce2565f38876618f037f04d8dac54d48e95'
decision=dict(UTC=datetime.now(timezone.utc).isoformat(),automatic_whole_family_readout_admitted=True,watcher_manifest_sha256=manifest_sha,peer_source_manifest_sha256='f83cb34bae29bda7bd542075bb3009d973fc7ab3bf494bd2cdfcf0801e704cd4',acquisition='meanSAGE_attribute_peer_feedback_acquisition_root_20261011_v1',full_banks=30,selected_optimizer_states=48,physical_bodies=57,CAL_endpoints=30,scalar_updates=15000,all_acquisition_owner_closure_before_CAL=True,all_CAL_owner_closure_before_whole_DEV=True,raw_serving='mean member class probabilities',CAL='CPU float64; common inside-member T;Adam .05;500updates;logTclamp[-5,5];latest tied minimum0..500',no_partial_metrics=True,no_retries_or_rescue=True,existing_source_semantics_unchanged=True,qualification_repeated=False,scientific_outcome_does_not_choose_handoff=True,whole_watch_wall_seconds=47000)
with (H/'ROOT_DECISION.json').open('x') as f:f.write(json.dumps(decision,indent=2)+'\n')
rows=[]
for path in [*sorted(S.rglob('*')),H/'ROOT_DECISION.json']:
    if not path.is_file() or '__pycache__' in path.parts:continue
    data=path.read_bytes()
    rows.append(dict(path=str(path.relative_to(P)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode()))
payload=dict(files=rows,source_manifest_sha256=manifest_sha)
remote=r'''
import base64,hashlib,json,os,pathlib,socket,subprocess,sys,time,zlib
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))
R=pathlib.Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';S=P/'peer_readout_source_20261011_v1';H=P/'meanSAGE_attribute_peer_feedback_readout_after_whole_acquisition_root_20261011_v1'
assert not S.exists() and not H.exists(),'Retain existing launch; never duplicate watcher'
def receipt(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
for row in payload['files']:
 p=P/row['path'];assert p.resolve().is_relative_to(P)
 data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
assert receipt(S/'MANIFEST.json')['sha256']==payload['source_manifest_sha256']
for rel,row in json.loads((S/'MANIFEST.json').read_text())['files'].items():
 got=receipt(S/rel);assert got['bytes']==row['bytes'] and got['sha256']==row['sha256']
config=json.loads((S/'CONFIG.disabled.json').read_text());config.update(execution_enabled=True,root_decision=receipt(H/'ROOT_DECISION.json'),watcher_manifest=receipt(S/'MANIFEST.json'))
for key in ('acquisition_config','acquisition_owner_start'):assert receipt(pathlib.Path(config[key]['path']))==config[key]
with (H/'CONFIG.json').open('x') as f:f.write(json.dumps(config,indent=2)+'\n')
os.chdir(R)
cmd=[str(R/'.venv/bin/python'),'-B','-u',str(S/'watcher.py'),str(H/'CONFIG.json')]
with (H/'watcher_launcher.log').open('xb') as log:
 watcher=subprocess.Popen(cmd,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 deadline=time.monotonic()+15;startfile=H/'watch_v1/WATCH_START.json'
 while not startfile.exists() and watcher.poll() is None and time.monotonic()<deadline:time.sleep(.05)
 assert watcher.poll() is None and startfile.exists(),'Watcher failed before waiting; retained project log'
start=json.loads(startfile.read_text());assert start['pid']==watcher.pid and start['stage']=='waiting_for_whole_acquisition'
launch=dict(status='FINITE_WHOLE_FAMILY_WAIT',watcher_pid=watcher.pid,config=receipt(H/'CONFIG.json'),watch_start=receipt(startfile),source_manifest=receipt(S/'MANIFEST.json'),command=cmd,partial_quality_opened=False,no_model_retry=True)
with (H/'LAUNCH.json').open('x') as f:f.write(json.dumps(launch,indent=2)+'\n')
rows=[]
for rel in ['ROOT_DECISION.json','CONFIG.json','LAUNCH.json','watch_v1/WATCH_START.json']:
 p=H/rel;r=receipt(p);r.update(name=rel,data=base64.b64encode(p.read_bytes()).decode());rows.append(r)
print(json.dumps(dict(launch=launch,files=rows)))
'''
(H/'DEPLOY_PAYLOAD.json').write_text(json.dumps(payload,indent=2)+'\n')
(H/'REMOTE_CODE.py.txt').write_text(remote)
spec=importlib.util.spec_from_file_location('transport',P/'publication/publish_exact_inventory_v19.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cmd=shlex.join(['/usr/bin/python3','-I','-S','-B','-c',remote,'repo','GPU','readout'])
argv=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',cmd]
r=m.terminal_transport(argv,input=base64.b64encode(zlib.compress(json.dumps(payload).encode(),9)).decode(),capture_output=True,text=True,timeout=60)
(H/'TRANSPORT.json').write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
if r.returncode:
    print(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout[-3000:],stderr=r.stderr[-3000:])))
    raise SystemExit(r.returncode)
a=json.loads(r.stdout);out=H/'actual_receipts';out.mkdir()
for row in a['files']:
    data=base64.b64decode(row['data']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    path=out/row['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
print(json.dumps(a['launch']))
