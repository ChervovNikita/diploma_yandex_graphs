"""Read only the registered singleton pilot's owned handles and fit progress."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REMOTE=r'''
import json,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
expected=json.load(sys.stdin)
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'shared_private_transfer_paired_pilot_execution_root_20261005_v2'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def read(p):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<262144
 return json.loads(p.read_text())
def physical(owner):
 if owner is None:return None
 p=Path('/proc')/str(owner['PID'])
 try:
  raw=(p/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
  if int(v[19])!=owner['start_ticks']:return {'PID':owner['PID'],'fresh_identity_matches':False,'PID_reused':True}
  argv=[s.decode() for s in (p/'cmdline').read_bytes().split(bytes([0])) if s]
  return {'PID':owner['PID'],'start_ticks':int(v[19]),'state':v[0],'fresh_identity_matches':argv==owner['argv'] or v[0]=='Z','cwd':str((p/'cwd').resolve()) if v[0]!='Z' else None}
 except FileNotFoundError:return None
value={'UTC':datetime.now(timezone.utc).isoformat(),'queue_physical':physical(expected['queue_identity']),
 'scores_read':False,'TEST_access':False,'signals_sent':False,'new_launches':0}
for name in ('QUEUE_START.json','QUEUE_PROGRESS.json','QUEUE_FAILURE.json','CURRENT_RESOURCES.json'):
 p=root/name
 if p.is_file():value[name]=read(p)
p=root/'CURRENT_PROCESS.json'
if p.is_file():
 current=read(p);value['current_cell']=current['cell_id'];value['child_physical']=physical(current['identity'])
 progress=root/'runs'/current['cell_id']/'PROGRESS.json'
 if progress.is_file():value['fit_progress']=read(progress)
 failed=root/'runs'/current['cell_id']/'FAILURE.json'
 if failed.is_file():value['fit_failure']=read(failed)
value['whole_block_freeze_exists']=(root/'BLOCK_FREEZE.json').is_file()
print(json.dumps(value))
'''
launch=json.loads((HERE/'LAUNCH_RECEIPT.json').read_text())
command=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
 '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20',
 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
 'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec python3 -I -B -c '+shlex.quote(REMOTE)]
r=subprocess.run(command,input=json.dumps(launch),capture_output=True,text=True,timeout=45)
out=HERE/('observation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
receipt={'exit_code':r.returncode,'stderr':r.stderr,'remote_source_sha256':hashlib.sha256(REMOTE.encode()).hexdigest()}
if r.returncode:receipt['stdout']=r.stdout
with (out/'TRANSPORT.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
if r.returncode:raise RuntimeError(r.stderr)
value=json.loads(r.stdout)
with (out/'OBSERVATION.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
