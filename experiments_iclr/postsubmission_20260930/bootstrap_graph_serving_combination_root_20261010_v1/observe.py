"""Read progress and owner closure without opening partial scientific quality."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
r=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
p=r/'experiments_iclr/postsubmission_20260930';h=p/'bootstrap_graph_serving_combination_root_20261010_v1'
read=lambda q:json.loads(q.read_text()) if q.exists() else None
launch=read(h/'LAUNCH.json');start=read(h/'OWNER_START.json');end=read(h/'OWNER_END.json')
owner=None
if launch:
 q=Path('/proc',str(launch['PID']),'stat')
 if q.exists():
  f=q.read_text().rsplit(')',1)[1].split();assert int(f[19])==launch['start_ticks'];owner=dict(PID=launch['PID'],start_ticks=int(f[19]),state=f[0])
progress=dict(study=read(h/'actual_study_v1/PROGRESS.json'))
failure=owner is None and (not end or not end['scientific_success'])
tails={}
if failure:
 for name in ('owner.stderr.log','worker.stderr.log'):
  q=h/name
  if q.exists():tails[name]=q.read_text()[-12000:]
result=dict(UTC=datetime.now(timezone.utc).isoformat(),owner=owner,start=start,end=end,progress=progress,failure=failure,error_tails=tails,partial_quality_opened=False,
            push_receipt=read(p/'publication/graph_reliability_complete855_and_method_status_20261010_v1/PUSH_RECEIPT.json'))
print(json.dumps(result))
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',required=True)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    q=here/args.receipt
    assert q.resolve().is_relative_to(here) and not q.exists()
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
    r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    data=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
    if r.returncode==0:data['result']=json.loads(r.stdout)
    q.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data.get('result',data)))
    return r.returncode

if __name__=='__main__':
    raise SystemExit(main())
