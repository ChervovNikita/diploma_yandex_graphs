"""Observe finite full18 comparison and fetch admitted compact JSON evidence."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
REMOTE=r'''
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
H=P/'private_hop_credit_pubmed_fullfit_source_20261010_v1';A=P/'closed_pubmed_readouts_root_20261010_v1'
start=json.loads((A/'FULL18_STARTER.json').read_text());live=Path('/proc',str(start['PID'])).exists()
family=H/'comparison';complete=not live and (family/'FAMILY_COMPLETE.json').is_file();failed=(family/'FAMILY_FAILURE.json').is_file()
files=[]
if complete or failed:
 O=H/'owners/comparison__full18__exact_paired_errors'
 files.extend(q for q in [family/'FAMILY_COMPLETE.json',family/'FAMILY_FAILURE.json',O/'RAW_OWNER_TERMINAL.json',O/'TERMINAL_CUSTODY.json',O/'ABSENCE.json'] if q.is_file())
if complete:
 assert json.loads((O/'TERMINAL_CUSTODY.json').read_text())['complete']
 out=family/'exact_paired_errors';C=out/'COMPLETE.json'
 assert hashlib.sha256(C.read_bytes()).hexdigest()==json.loads((O/'TERMINAL_CUSTODY.json').read_text())['complete_sha256']
 assert json.loads(C.read_text())['complete'];files.extend([C,out/'EXACT_PAIRED_ERROR_CHANGES.json',out/'ACTUAL_COSTS.json'])
 release=json.loads((H/'COMPARISON_RELEASE.json').read_text())
 for row in release['records']:
  q=Path(row['complete']['path']);assert hashlib.sha256(q.read_bytes()).hexdigest()==row['complete']['sha256'];files.append(q)
errors={}
if failed:
 for q in [A/'full18_owner.stderr',O/'WORKER.log']:
  if q.is_file():errors[str(q.relative_to(P))]=q.read_text()[-10000:]
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),owner_live=live,complete=complete,failure=failed,error_tails=errors,files=[dict(path=str(q.relative_to(P)),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),text=q.read_text()) for q in files])))
'''
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
(HERE/('FULL18_OBSERVATION_'+tag+'.json')).write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
assert r.returncode==0,r.stderr
v=json.loads(r.stdout)
for row in v['files']:
 q=HERE/'fetched'/row['path'];assert q.resolve().is_relative_to((HERE/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_text(row['text']);assert hashlib.sha256(q.read_bytes()).hexdigest()==row['sha256']
print(json.dumps({k:x for k,x in v.items() if k!='files'}))
