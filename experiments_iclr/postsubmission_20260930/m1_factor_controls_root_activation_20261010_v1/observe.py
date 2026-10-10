"""Observe fixed scientific owners without reading incomplete quality."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='24c315a7-3c08-471f-b550-b9a3e1faf75d'
def identity(pid):
 q=Path('/proc',str(pid),'stat')
 if not q.is_file():return None
 t=q.read_text();f=t[t.rfind(')')+2:].split()
 return dict(PID=pid,state=f[0],start_ticks=int(f[19]),pgid=int(f[2]),sid=int(f[3]))
families={}
for name,pid,ticks,purpose in [
 ('private_cmcl_polyformer_root_preparation_20261010_v2',594139,6041323646,'science'),
 ('pubmed_factor1_controls_source_20261010_v1',596470,6041554776,'science'),
 ('pubmed_strong_reference_root_execution_20261010_v1',591965,6041124203,'science')]:
 d=P/name/purpose;owner=identity(pid);assert owner is None or owner['start_ticks']==ticks
 q=d/'FAMILY_PROGRESS.json';progress=json.loads(q.read_text()) if q.is_file() else None
 child=None
 if progress and progress.get('active_child'):
  birth=progress['active_child'];child=identity(birth['PID'])
  assert child is None or child['start_ticks']==birth['start_ticks']
 files={}
 for label in ['FAMILY_START.json','FAMILY_COMPLETE.json','FAMILY_FAILURE.json']:
  q=d/label
  if q.is_file():files[label]=json.loads(q.read_text())
 families[name]=dict(owner=owner,child=child,progress=progress,terminal_metadata=files,
                    family_complete=(d/'FAMILY_COMPLETE.json').is_file(),family_failure=(d/'FAMILY_FAILURE.json').is_file())
hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['RESEARCH_STATE.md','PUBLIC_STATUS.md','research_ledger.json','RESEARCH_WORKFLOW.md']}
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),families=families,partial_quality_opened=False,
 canonical_sha256=hashes,allocation_HEAD=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip())))
'''
argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
      '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
      'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
(HERE/('TRANSPORT_'+tag+'.json')).write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
if r.returncode:raise SystemExit(r.returncode)
data=json.loads(r.stdout)
(HERE/('OBSERVATION_'+tag+'.json')).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
