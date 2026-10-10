"""Read current qualification and publication receipts on the literal authorized route."""
from pathlib import Path
import json
import shlex
import subprocess
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import json, socket, subprocess
from datetime import datetime, timezone
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
A=P/'typed_context_allocation_qualification_preparation_20261010_v2'
E=P/'typed_context_allocation_qualification_execution_20261010_v2'
files={}
for q in [A/'LAUNCH.json', A/'TERMINAL.json', A/'MONITOR_FAILURE.json', E/'COHORT_REPORT.json',
 P/'publication/typed_context_output_cap_correction_and_stored_diagnostic_20261010_v1/PUSH_RECEIPT.json',
 P/'publication/typed_context_output_cap_correction_and_stored_diagnostic_20261010_v1/PUSH_RECEIPT_retry_after_input_timeout.json',
 P/'publication/typed_context_V3_qualification_and_distinct_credit_source_20261010_v1/PUSH_RECEIPT.json']:
 if q.is_file(): files[str(q.relative_to(P))]=json.loads(q.read_text())
def identity(pid):
 q=Path('/proc',str(pid),'stat')
 if not q.is_file(): return None
 text=q.read_text();f=text[text.rfind(')')+2:].split()
 return dict(pid=pid,state=f[0],start_ticks=int(f[19]),ppid=int(f[1]),pgid=int(f[2]),sid=int(f[3]))
owner=identity(591233)
assert owner is None or owner['start_ticks']==6040925766
launch=files.get(str((A/'LAUNCH.json').relative_to(P)))
child=identity(launch['child']['pid']) if launch else None
if child: assert child['start_ticks']==launch['child']['start_ticks']
reports=[]
for q in sorted(E.glob('**/REPORT.json')):
 if q.stat().st_size < 1024*1024: files[str(q.relative_to(P))]=json.loads(q.read_text())
stderr=(A/'worker.stderr').read_text()[-5000:] if (A/'worker.stderr').is_file() else None
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),owner=owner,child=child,files=files,stderr_tail=stderr,scientific=False,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))
'''
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
record=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
(HERE/'ALLOCATION_TRANSPORT.json').write_text(json.dumps(record,indent=2)+'\n')
if result.returncode: raise SystemExit(result.returncode)
data=json.loads(result.stdout)
(HERE/'ALLOCATION_OBSERVATION.json').write_text(json.dumps(data,indent=2)+'\n')
for name,value in data['files'].items():
 q=HERE/'fetched'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(owner=data['owner'],child=data['child'],files=list(data['files']),stderr_tail=data['stderr_tail'])))
