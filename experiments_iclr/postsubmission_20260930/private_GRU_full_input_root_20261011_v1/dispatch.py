"""Run one separately admitted finite derivative stage and preserve actual receipts."""
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
REMOTE=r"""
import json,os,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
os.chdir(repo)
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'private_GRU_full_input_root_20261011_v1'
config=json.loads((root/'CONFIG.json').read_text())
owner=phase/'native_meanSAGE_private_GRU_qualification_source_20261011_v1/owner.py'
r=subprocess.run([str(repo/'.venv/bin/python'),'-B','-u',str(owner),str(root/'CONFIG.json')],cwd=repo,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=config['limits']['wall_seconds']+70)
value=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
for name,path in [('OWNER_END.json',Path(config['owner_output'])/'OWNER_END.json'),('QUALIFICATION.json',Path(config['output'])/'QUALIFICATION.json'),('COSTS.json',Path(config['output'])/'COSTS.json'),('FAILURE.json',Path(config['output'])/'FAILURE.json')]:
 if path.exists():value[name]=json.loads(path.read_text())
if r.returncode:
 log=Path(config['owner_output'])/'child.log'
 if log.exists():value['child_failure_log']=log.read_text()[-14000:]
print(json.dumps(value,separators=(',',':')))
"""
a=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
r=subprocess.run(a,capture_output=True,text=True,timeout=1890)
(HERE/'TRANSPORT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
assert r.returncode==0,'Transport observation failed; inspect same owner before any retry'
v=json.loads(r.stdout);(HERE/'ACTUAL_RESULT.json').write_text(json.dumps(v,indent=2)+'\n')
for name in ['OWNER_END.json','QUALIFICATION.json','COSTS.json','FAILURE.json']:
 if name in v:(HERE/name).write_text(json.dumps(v[name],indent=2)+'\n')
qualification=v.get('QUALIFICATION.json',{});print(json.dumps(dict(exit_code=v['exit_code'],owner_status=v.get('OWNER_END.json',{}).get('status'),complete=qualification.get('complete'),status=qualification.get('status'),TRAIN_seconds=qualification.get('TRAIN_seconds'),SELECT0_seconds=qualification.get('checkpoint0_SELECT_seconds'),restored_SELECT_seconds=qualification.get('restored_SELECT_seconds'),costs=qualification.get('costs'),backend_operations=qualification.get('backend_operations'),failure=qualification.get('failure'),failure_log=v.get('child_failure_log'))))
raise SystemExit(v['exit_code'])
