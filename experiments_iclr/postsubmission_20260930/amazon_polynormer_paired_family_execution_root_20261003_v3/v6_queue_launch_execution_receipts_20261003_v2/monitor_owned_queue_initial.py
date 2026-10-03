"""Read only exact owned process handles and registered fit progress metadata."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--sequence',type=int,required=True)
args=parser.parse_args()
assert 1<=args.sequence<=100000
prefix='MONITOR_%04d'%args.sequence
CODE='''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930';root=phase/'amazon_polynormer_paired_family_execution_root_20261003_v3'
launch=root/'v6_queue_detached_launch_20261003_v1';queue=root/'v6_full_schedule_v1'
os.chdir(repo);g=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15);assert Path.cwd()==repo and g.returncode==0 and g.stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
files=[]
def metadata(p):
 assert p.suffix in ('.json','.txt') and p.stat().st_size<131072
 b=p.read_bytes();b.decode('utf8');row=dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));files.append(dict(descriptor=row,base64=base64.b64encode(b).decode()));return json.loads(b) if p.suffix=='.json' else b.decode()
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split();argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
  rss=next((int(line.split()[1])*1024 for line in (p/'status').read_text().splitlines() if line.startswith('VmRSS:')),0)
  return dict(pid=pid,start_ticks=int(v[19]),ppid=int(v[1]),pgid=int(v[2]),sid=int(v[3]),state=v[0],argv=argv,VmRSS_bytes=rss)
 except FileNotFoundError:return None
detached=json.loads((launch/'DETACHED_LAUNCH.json').read_text());supervisor_launch=metadata(launch/'SUPERVISOR_LAUNCH.json')
handles=[]
for role,expected in [('detached_runner',detached['runner_identity']),('queue_supervisor',supervisor_launch['supervisor_identity'])]:
 current=identity(expected['pid'])
 if current is not None:
  assert current['start_ticks']==expected['start_ticks'] and (current['argv']==expected['argv'] or current['state']=='Z')
 handles.append(dict(role=role,expected_identity=expected,current_identity=current))
queue_launch=metadata(queue/'LAUNCH.json') if (queue/'LAUNCH.json').is_file() else None
if queue_launch:
 current=identity(queue_launch['worker_pid'])
 if current is not None:assert current['argv']==queue_launch['command'] or current['state']=='Z'
 handles.append(dict(role='queue_worker_and_sequential_child_supervisor',admitted_launch=queue_launch,current_identity=current))
registry=json.loads((root/'registry/REGISTRY.json').read_text());fits=[];failures=[]
for registered in registry['physical_fits']:
 fit=phase/registered['output'];row=dict(fit_id=registered['id'],registered_output=registered['output'],output_exists=fit.exists(),claim_exists=(phase/registered['claim_path']).exists(),trace_progress=None,launch=None,physical_terminal=None)
 if (fit/'LAUNCH.json').is_file():
  row['launch']=metadata(fit/'LAUNCH.json');current=identity(row['launch']['worker_pid'])
  if current is not None:assert current['argv']==row['launch']['command'] or current['state']=='Z'
  handles.append(dict(role='registered_fit_worker',fit_id=registered['id'],admitted_launch=row['launch'],current_identity=current))
 trace=fit/'TRACE.jsonl'
 if trace.is_file() and trace.stat().st_size:
  with trace.open('rb') as s:
   size=trace.stat().st_size;offset=max(0,size-65536);s.seek(offset);tail=s.read()
  lines=tail.splitlines()
  if offset and lines:lines=lines[1:]
  if not tail.endswith(bytes([10])) and lines:lines=lines[:-1]
  if lines:
   event=json.loads(lines[-1]);row['trace_progress']={k:event[k] for k in ('actual_update','author_display_epoch','stage','stage_epoch','actual_local_updates','actual_global_updates','training_plus_VAL_seconds','memory') if k in event}
   row['trace_progress'].update(registered_total_updates=2700,trace_bytes_observed=size,quality_fields_not_reported=True)
 if (fit/'TERMINAL.json').is_file():row['physical_terminal']=metadata(fit/'TERMINAL.json')
 if (fit/'FAILURE.json').is_file():failures.append(dict(fit_id=registered['id'],failure=metadata(fit/'FAILURE.json')))
 fits.append(row)
for p in [queue/'FAILURE.json',queue/'TERMINAL.json',launch/'SUPERVISOR_TERMINAL.json']:
 if p.is_file():
  value=metadata(p)
  if value.get('status')=='failed' or value.get('physical_supervisor_exit_code',0)!=0:failures.append(dict(path=str(p.relative_to(phase)),failure=value))
status='failed' if failures else 'complete' if (queue/'FREEZE.json').is_file() else 'running'
if status=='failed':
 for p in [queue/'OUTER_TRACEBACK.txt',queue/'stderr.txt',launch/'SUPERVISOR_STDERR.txt',launch/'RUNNER_STDERR.txt']:
  if p.is_file() and p.stat().st_size<131072:metadata(p)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),route=dict(repository=str(repo),GPU_UUID=g.stdout.strip()),status=status,published_queue=detached['published_queue'],handles=handles,registered_fit_progress=fits,failures=failures,metadata_only=True,exact_owned_handles_only=True,signals_sent=False,retries_or_new_launches=False,quality_ranking_or_TEST_control_scoring=False,files=files)))
'''
with (HERE/(prefix+'_REMOTE_CODE.py.txt')).open('x') as h:h.write(CODE)
ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
started=datetime.now(timezone.utc).isoformat()
r=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',CODE])],capture_output=True,text=True)
transport=dict(start_UTC=started,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stderr=r.stderr,stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest(),client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),private_key_contents_read=False,read_only=True)
with (HERE/(prefix+'_TRANSPORT.json')).open('x') as h:json.dump(transport,h,indent=2);h.write('\n')
assert r.returncode==0,r.stderr
value=json.loads(r.stdout);files=value.pop('files');value['fetched_descriptors']=[]
for entry in files:
    row=entry['descriptor'];raw=base64.b64decode(entry['base64'],validate=True)
    assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    p=PHASE/row['path'];assert p.resolve().is_relative_to(PHASE)
    if p.exists():assert p.read_bytes()==raw
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as h:h.write(raw)
    value['fetched_descriptors'].append(row)
with (HERE/(prefix+'_RESULT.json')).open('x') as h:json.dump(value,h,indent=2);h.write('\n')
print(json.dumps(dict(UTC=value['UTC'],status=value['status'],failures=value['failures'],handles=[dict(role=h['role'],fit_id=h.get('fit_id'),identity=h.get('current_identity')) for h in value['handles']],fit_progress=[r for r in value['registered_fit_progress'] if r['output_exists']],metadata_only=True),indent=2))
