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
owner=json.load(sys.stdin);result={'UTC':datetime.now(timezone.utc).isoformat(),'files':[],'native_fit_or_held_scoring':False}
proc=Path('/proc')/str(owner['PID'])
try:
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 if int(f[19])==owner['start_ticks']:result['owned_process']={'PID':owner['PID'],'start_ticks':owner['start_ticks'],'state':f[0],'cwd_matches':str((proc/'cwd').resolve())==str(repo),'argv_matches':[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]==owner['argv']}
 else:result['owned_process']={'known_handle_absent':True,'foreign_not_inspected':True}
except FileNotFoundError:result['owned_process']={'known_handle_absent':True}
base=phase/'allocation_monolithic_utility_native_execution_root_20261006_v1/owned_run01'
for rel in ('CHILD_LAUNCH.json','TERMINAL.json','child/RESULT.json'):
 q=base/rel
 if q.exists():
  assert q.resolve().is_relative_to(base) and not q.is_symlink() and q.stat().st_size<2000000
  b=q.read_bytes();result['files'].append({'path':str(q.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'utf8':b.decode()})
if result['owned_process'].get('known_handle_absent') and not (base/'TERMINAL.json').exists():
 for rel in ('stdout.log','stderr.log'):
  q=phase/'allocation_monolithic_utility_native_launch_root_20261006_v1'/rel
  assert q.is_file() and q.stat().st_size<131072
  b=q.read_bytes();result['files'].append({'path':str(q.relative_to(phase)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'utf8':b.decode()})
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
   rows.append({'path':row['path'],'bytes':row['bytes'],'sha256':row['sha256'],'status':q.get('status'),'checks_completed':[c['name'] for c in q.get('checks',[])],'error':q.get('error')})
  print(json.dumps({'UTC':result['UTC'],'owned_process':result['owned_process'],'files':rows}))
  if any(row['path'].endswith('/TERMINAL.json') for row in result['files']):
   handoff=P/'allocation_monolithic_utility_native_terminal_handoff_root_20261006_v1';handoff.mkdir(exist_ok=True)
   for row in result['files']:
    b=row['utf8'].encode();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'];target=handoff/('WORKER_RESULT.json' if '/child/' in row['path'] else Path(row['path']).name)
    if target.exists():assert target.read_bytes()==b
    else:target.write_bytes(b);target.chmod(0o444)
 snapshot.write_text(json.dumps(v,indent=2)+'\n')
if __name__=='__main__':main()
