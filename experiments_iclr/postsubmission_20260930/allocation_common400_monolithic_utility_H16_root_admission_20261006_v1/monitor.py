"""Read only the known utility engineering handles and metadata."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,shlex
D=Path(__file__).resolve().parent
P=D.parent
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess,sys
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
owner=json.load(sys.stdin);result={'UTC':datetime.now(timezone.utc).isoformat(),'files':[],'held_scoring':False}
proc=Path('/proc')/str(owner['PID'])
try:
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 if int(f[19])==owner['start_ticks']:result['owned_process']={'PID':owner['PID'],'start_ticks':owner['start_ticks'],'state':f[0],'cwd_matches':str((proc/'cwd').resolve())==str(repo),'argv_matches':[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]==owner['argv']}
 else:result['owned_process']={'known_handle_absent':True,'foreign_not_inspected':True}
except FileNotFoundError:result['owned_process']={'known_handle_absent':True}
base=phase/'allocation_common400_monolithic_utility_H16_execution_root_20261006_v1'
for rel in ('supervisor/CHILD_LAUNCH.json','supervisor/TERMINAL.json','fit/WORKER_TERMINAL.json','fit/RUN.json'):
 q=base/rel
 if q.exists():
  assert q.resolve().is_relative_to(base) and not q.is_symlink() and q.stat().st_size<2000000
  b=q.read_bytes();result['files'].append({'path':str(q.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'utf8':b.decode()})
if result['owned_process'].get('known_handle_absent') and not (base/'supervisor/TERMINAL.json').exists():
 for rel in ('stdout.log','stderr.log'):
  q=phase/'allocation_common400_monolithic_utility_H16_launch_root_20261006_v1'/rel
  assert q.is_file() and q.stat().st_size<131072
  b=q.read_bytes();result['files'].append({'path':str(q.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'utf8':b.decode()})
events=base/'fit/ATTEMPTED_OPERATIONS.jsonl'
if events.exists():
 assert events.is_file() and not events.is_symlink() and events.resolve().is_relative_to(base) and events.stat().st_size<2000000
 complete=[];partial_lines=0
 for line in events.read_text().splitlines():
  try:e=json.loads(line)
  except json.JSONDecodeError:partial_lines+=1;continue
  if e.get('kind')=='episode_completed':complete.append({'episode':e['episode'],'elapsed_seconds':e['elapsed_seconds']})
 result['episode_progress']={'completed':len(complete),'latest_complete':complete[-1] if complete else None,'partial_append_lines':partial_lines}
print(json.dumps(result))
'''
def main():
 owner=json.loads((D/'LAUNCH_RECEIPT.json').read_text())
 command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','cd '+shlex.quote(REPO)+' && /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)]
 r=subprocess.run(command,input=json.dumps(owner),capture_output=True,text=True,timeout=45)
 snapshot=D/('MONITOR_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');v={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stderr':r.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest()}
 if r.returncode:v['stdout']=r.stdout;print(json.dumps(v))
 else:
  v['result']=json.loads(r.stdout);result=v['result'];rows=[]
  for row in result['files']:
   try:q=json.loads(row['utf8'])
   except json.JSONDecodeError:q={}
   rows.append({'path':row['path'],'bytes':row['bytes'],'sha256':row['sha256'],'status':q.get('status'),'checks_completed':[c['name'] for c in q.get('checks',[])],'error':q.get('error',q.get('primary_error')),'completed_episodes':q.get('completed_episodes')})
  print(json.dumps({'UTC':result['UTC'],'owned_process':result['owned_process'],'files':rows,'episode_progress':result.get('episode_progress')}))
  if any(row['path'].endswith('/TERMINAL.json') for row in result['files']):
   handoff=P/'allocation_common400_monolithic_utility_H16_terminal_handoff_root_20261006_v1';handoff.mkdir(exist_ok=True)
   for row in result['files']:
    b=row['utf8'].encode();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];target=handoff/Path(row['path']).name
    if target.exists():assert target.read_bytes()==b
    else:target.write_bytes(b);target.chmod(0o444)
 snapshot.write_text(json.dumps(v,indent=2)+'\n')
if __name__=='__main__':main()
