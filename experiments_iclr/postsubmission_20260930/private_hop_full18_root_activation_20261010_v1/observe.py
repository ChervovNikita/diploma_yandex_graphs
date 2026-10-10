"""Read private-hop progress without opening partial predictive scores."""
from datetime import datetime,timezone
from pathlib import Path
import json,shlex,subprocess
HERE=Path(__file__).resolve().parent
REMOTE=r'''
from datetime import datetime,timezone
from pathlib import Path
import json,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
H=P/'private_hop_credit_pubmed_fullfit_source_20261010_v1';A=P/'private_hop_full18_root_activation_20261010_v1'
start=json.loads((A/'STARTER.json').read_text());pid=start['PID'];q=Path('/proc',str(pid),'stat');owner=None
if q.exists():
 f=q.read_text().rsplit(')',1)[1].split();assert int(f[19])==start['start_ticks'];owner=dict(PID=pid,state=f[0],start_ticks=int(f[19]))
family=H/'science';progress=family/'FAMILY_PROGRESS.json';p=json.loads(progress.read_text()) if progress.exists() else {}
epoch=p.get('epoch')
if epoch is None and p.get('current_record'):
 q=family/'cells'/p['current_record']/'PROGRESS.json'
 if q.exists():epoch=json.loads(q.read_text()).get('epoch')
done=family/'FAMILY_COMPLETE.json';failed=family/'FAMILY_FAILURE.json';closed=done.exists() and owner is None
retained={};errors={}
if closed or failed.exists():
 for q in (done,failed):
  if q.exists():retained[str(q.relative_to(P))]=json.loads(q.read_text())
 if failed.exists():
  r=json.loads(failed.read_text())['failed_record']
  for q in (A/'owner.stderr',H/'owners'/('science__'+r)/'WORKER.log'):
   if q.exists():errors[str(q.relative_to(P))]=q.read_text()[-10000:]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),owner=owner,closed=closed,failure=failed.exists(),current_record=p.get('current_record'),completed_records=len(p.get('completed',[])),epoch=epoch,quality_opened=False,retained=retained,error_tails=errors)))
'''
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
(HERE/('OBSERVATION_'+tag+'.json')).write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
assert r.returncode==0,r.stderr
v=json.loads(r.stdout)
for name,row in v['retained'].items():
 q=HERE/'fetched'/name;assert q.resolve().is_relative_to((HERE/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(row,indent=2)+'\n')
print(json.dumps({k:row for k,row in v.items() if k!='retained'}))
