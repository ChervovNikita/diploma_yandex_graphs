"""Stage two confined scripts and run the finite stored-prediction diagnostic."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
H=Path(__file__).resolve().parent
files={q.name:base64.b64encode(q.read_bytes()).decode() for q in [H/'run_owned.py',H/'render.py']}
remote=r'''
from pathlib import Path
import base64,json,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
H=P/'pubmed_strong_reference_complete12_root_20261010_v1'
H.mkdir(exist_ok=False)
FILES=__FILES__
for name,data in FILES.items():
 q=H/name
 assert q.parent==H
 q.write_bytes(base64.b64decode(data))
subprocess.run(['python3','-B',str(H/'render.py')],cwd=R,check=True,timeout=20)
res=subprocess.run(['python3','-B',str(H/'run_owned.py')],cwd=R,check=False,timeout=325)
retained={}
for q in H.glob('*.json'):retained[str(q.relative_to(P))]=json.loads(q.read_text())
E=P/'pubmed_strong_reference_comparison_execution_20261010_v1'
for q in E.glob('*.json'):retained[str(q.relative_to(P))]=json.loads(q.read_text())
retained['worker_log_tail']=(H/'WORKER.log').read_text()[-5000:]
print(json.dumps(dict(exit_code=res.returncode,retained=retained)),flush=True)
raise SystemExit(res.returncode)
'''.replace('__FILES__',repr(files))
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(remote)]
result=subprocess.run(argv,capture_output=True,text=True,timeout=370)
(H/'TRANSPORT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
lines=result.stdout.splitlines();last=json.loads(lines[-1]) if lines else {}
for name,data in last.get('retained',{}).items():
 if name=='worker_log_tail': (H/'WORKER_TAIL.txt').write_text(data);continue
 q=H/'fetched'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(exit_code=result.returncode,files=list(last.get('retained',{})))))
raise SystemExit(result.returncode)
