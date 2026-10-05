"""Observe exact owned queue progress, without opening comparative scores."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',type=int,required=True)
    args=parser.parse_args()
    dest=ROOT/('QUEUE_MONITOR_%03d.json'%args.sequence)
    assert args.sequence>0 and not dest.exists()
    code='''from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=phase/'citeseer_endpoint_frame_paired_development_20261005_v1'
owned=json.loads((root/'OWNED_QUEUE_PROCESS.json').read_text());proc=Path('/proc')/str(owned['pid']);identity=None
if proc.exists():
 raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 assert int(fields[19])==owned['start_ticks']
 argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(bytes([0])) if v]
 assert argv==owned['command'] or fields[0]=='Z'
 identity=dict(pid=owned['pid'],state=fields[0],start_ticks=int(fields[19]))
value=dict(UTC=datetime.now(timezone.utc).isoformat(),queue_identity=identity,status='pending',TEST_access=False,comparative_outcomes_read=False,signals_sent=False)
p=root/'QUEUE_PROGRESS.json'
if p.exists():
 progress=json.loads(p.read_text());value['progress']=progress;value['status']='running'
 if progress.get('current'):
  current=root/'fits'/progress['current'];cp=current/'PROGRESS.json'
  if cp.exists():value['current_fit_progress']=json.loads(cp.read_text())
  child=Path('/proc')/str(progress['child_pid'])
  if child.exists():
   raw=(child/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
   argv=[v.decode() for v in (child/'cmdline').read_bytes().split(bytes([0])) if v]
   assert argv==progress['command'] or fields[0]=='Z'
   value['child_identity']=dict(pid=progress['child_pid'],state=fields[0],start_ticks=int(fields[19]))
for name,status in [('QUEUE_FAILURE.json','failed'),('COHORT_FREEZE.json','complete')]:
 f=root/name
 if f.exists():
  terminal=json.loads(f.read_text());value['status']=status
  value['terminal']={k:v for k,v in terminal.items() if k!='completed_physical_fits'}
  if status=='complete':value['completed_fits']=len(terminal['completed_physical_fits'])
if identity is None and value['status'] not in ['failed','complete']:value['status']='process_missing_without_terminal'
if value['status'] in ['failed','process_missing_without_terminal']:
 f=root/'queue.stderr.log'
 if f.exists():value['stderr_tail']=f.read_text()[-12000:]
print(json.dumps(value))
'''
    ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
         '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
         '-o','UpdateHostKeys=no','-o','ConnectTimeout=15',
         'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    r=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code])],capture_output=True,text=True,timeout=35)
    value=dict(UTC=datetime.now(timezone.utc).isoformat(),transport_exit_code=r.returncode,stderr=r.stderr)
    if r.returncode==0:value['observation']=json.loads(r.stdout)
    else:value['stdout']=r.stdout
    with dest.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))


if __name__=='__main__':
    main()
