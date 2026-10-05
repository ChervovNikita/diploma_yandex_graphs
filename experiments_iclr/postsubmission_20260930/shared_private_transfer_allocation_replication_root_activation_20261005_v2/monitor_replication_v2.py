"""Observe only fixed replication queues, their known process handles and progress counters."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE=r"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
p=phase/'shared_private_transfer_allocation_replication_preparation_20261005_v2/PROSPECTIVE_AMENDMENT.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='d965a699bc9d150470ea2b4b0fc0c07a4ffd90139230db065e0473b0311d7e2d';a=json.loads(p.read_text())
value={'UTC':datetime.now(timezone.utc).isoformat(),'files':[],'owned_process_observations':[],'scores_read':False,'TEST_access':False}
def metadata(p):
 if p.exists():
  assert p.resolve().is_relative_to(phase) and not p.is_symlink() and p.stat().st_size<2_000_000
  b=p.read_bytes();value['files'].append({'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'utf8':b.decode()})
def handle(owner,expected_argv):
 p=Path('/proc')/str(owner['PID'])
 try:raw=(p/'stat').read_text()
 except FileNotFoundError:return {'PID':owner['PID'],'start_ticks':owner['start_ticks'],'known_handle_absent':True}
 f=raw[raw.rfind(')')+2:].split()
 if int(f[19])!=owner['start_ticks']:return {'PID':owner['PID'],'start_ticks':owner['start_ticks'],'known_handle_absent':True,'foreign_process_not_inspected':True}
 argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x];cwd=str((p/'cwd').resolve())
 return {'PID':owner['PID'],'start_ticks':int(f[19]),'known_handle_absent':False,'state':f[0],'owned_argv_matches':argv==expected_argv,'owned_cwd_matches':cwd==str(repo)}
for q in a['queues']:
 root=phase/q['execution_directory_relative']
 if not root.exists():continue
 for name in ['LAUNCH_RECEIPT.json','QUEUE_START.json','QUEUE_PROGRESS.json','QUEUE_FAILURE.json','BLOCK_FREEZE.json']:metadata(root/name)
 if (root/'LAUNCH_RECEIPT.json').exists():
  launch=json.loads((root/'LAUNCH_RECEIPT.json').read_text());value['owned_process_observations'].append({'queue_id':q['queue_id'],**handle(launch['queue_identity'],launch['argv'])})
 for cell in q['ordered_cell_ids']:
  for name in [cell+'.CHILD_STARTED.json',cell+'.EXIT.json']:metadata(root/'logs'/name)
  for name in ['PROGRESS.json','FAILURE.json']:metadata(root/'runs'/cell/name)
print(json.dumps(value))
"""
command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=20','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
r=subprocess.run(command,capture_output=True,text=True,timeout=50)
snapshot=HERE/('monitor_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));snapshot.mkdir()
receipt={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stderr':r.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest(),'scientific_fit_or_score_or_TEST_access':False}
if r.returncode==0:
 v=json.loads(r.stdout);receipt['result']=v
 for row in v['files']:
  b=row['utf8'].encode();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];p=snapshot/row['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
else:receipt['stdout']=r.stdout
(snapshot/'MONITOR_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
progress=[]
partial=[]
for row in receipt.get('result',{}).get('files',[]):
 if not row['path'].endswith(('/PROGRESS.json','/QUEUE_PROGRESS.json','/QUEUE_FAILURE.json')):continue
 try:metadata=json.loads(row['utf8'])
 except json.JSONDecodeError:
  partial.append({'path':row['path'],'bytes':row['bytes'],'sha256':row['sha256'],'status':'unparseable_observation_retained_not_terminal_evidence'})
 else:progress.append({'path':row['path'],'metadata':metadata})
print(json.dumps({'exit_code':r.returncode,'snapshot':str(snapshot),'owned_process_observations':receipt.get('result',{}).get('owned_process_observations'),'progress':progress,'partial_metadata_observations':partial}))
