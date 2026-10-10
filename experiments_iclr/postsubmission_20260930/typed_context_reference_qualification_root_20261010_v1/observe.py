"""Observe the exact reference qualifier and preserve its terminal evidence."""
from pathlib import Path
from datetime import datetime, timezone
import base64, hashlib, json, shlex, subprocess

H=Path(__file__).resolve().parent
remote=r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930');H=P/'typed_context_reference_qualification_root_20261010_v1';E=P/'typed_context_reference_qualification_execution_20261010_v1'
starter=json.loads((H/'STARTER.json').read_text());assert starter['PID']==600076 and starter['start_ticks']==6041810622
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==starter['boot_id']
def identity(pid):
    q=Path('/proc',str(pid),'stat')
    if not q.is_file():return None
    f=q.read_text().rsplit(')',1)[1].split();return dict(PID=pid,start_ticks=int(f[19]),state=f[0])
owner=identity(starter['PID']);assert owner is None or owner['start_ticks']==starter['start_ticks']
files=[]
def retain(q):
    data=q.read_bytes();files.append(dict(path=str(q.relative_to(P)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode()))
terminal=None
for q in H.glob('*.json'):
    retain(q)
    if q.name=='TERMINAL.json':terminal=json.loads(q.read_text())
cohort=None
q=E/'COHORT_REPORT.json'
if q.is_file():
    v=json.loads(q.read_text());cohort=dict(status=v['status'],complete=v['complete'],qualification_passed=v['qualification_passed'],body_records=len(v['runs']),assembled_banks=len(v['ensembles']),error=v.get('error'),quality_fields_omitted=True)
    if terminal is not None:retain(q)
for q in H.glob('*.stderr'):
    if terminal is not None:retain(q)
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),owner=owner,terminal=terminal,cohort=cohort,files=files,scientific_result=False)),flush=True)
'''
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(remote)]
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
res=subprocess.run(argv,capture_output=True,text=True,timeout=60)
(H/('OBSERVE_TRANSPORT_'+tag+'.json')).write_text(json.dumps(dict(exit_code=res.returncode,stdout=res.stdout,stderr=res.stderr),indent=2)+'\n')
if res.returncode:print(res.stderr);raise SystemExit(res.returncode)
value=json.loads(res.stdout)
for row in value.pop('files'):
    data=base64.b64decode(row['base64']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    q=H/'fetched'/row['path'];assert q.resolve().is_relative_to((H/'fetched').resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
(H/('OBSERVATION_'+tag+'.json')).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value))
