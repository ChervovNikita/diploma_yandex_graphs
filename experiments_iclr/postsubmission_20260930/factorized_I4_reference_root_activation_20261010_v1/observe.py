"""Observe one finite reference stage and fetch evidence only after closure."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REMOTE=r'''
from datetime import datetime,timezone
from pathlib import Path
import json,socket,subprocess,sys
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
H=P/'pubmed_factorized_I4_reference_source_20261010_v1';A=P/'factorized_I4_reference_root_activation_20261010_v1';stage=sys.argv[1]
starter=json.loads((A/stage/'STARTER.json').read_text());pid=starter['PID'];q=Path('/proc',str(pid),'stat');owner=None
if q.is_file():
 f=q.read_text().rsplit(')',1)[1].split();assert int(f[19])==starter['start_ticks'];owner=dict(PID=pid,start_ticks=int(f[19]),state=f[0])
family=H/stage;done=family/'FAMILY_COMPLETE.json';failure=family/'FAMILY_FAILURE.json'
complete=done.is_file() and owner is None
retained={}
if complete or failure.is_file():
 plan=json.loads((H/(stage.upper()+'_OWNER_PLAN.json')).read_text())
 names=[family/'FAMILY_START.json',done,failure]
 for r in plan['records']:
  O=H/'owners'/r['owner_id']
  names.extend(O/n for n in ('RAW_OWNER_TERMINAL.json','TERMINAL_CUSTODY.json','ABSENCE.json'))
  if complete:names.append(H/stage/'cells'/r['record_id']/'COMPLETE.json')
 for q in names:
  if q.is_file():retained[str(q.relative_to(P))]=json.loads(q.read_text())
errors={}
if failure.is_file():
 for q in (A/stage/'owner.stderr',):
  if q.is_file():errors[str(q.relative_to(P))]=q.read_text()[-10000:]
 for q in (H/'owners').glob(stage+'__*/WORKER.log'):
  errors[str(q.relative_to(P))]=q.read_text()[-10000:]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,owner=owner,complete=complete,failure=failure.is_file(),retained=retained,error_tails=errors,scientific_partial_quality_opened=False)))
'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=('admission','qualification','science','assembly','comparison'));args=p.parse_args()
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(args.stage)]
    r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (HERE/(args.stage+'_OBSERVATION_'+tag+'.json')).write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    assert r.returncode==0,r.stderr
    v=json.loads(r.stdout)
    for name,data in v['retained'].items():
        q=HERE/'fetched'/name;assert q.resolve().is_relative_to((HERE/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:value for k,value in v.items() if k!='retained'}))

if __name__=='__main__':main()
