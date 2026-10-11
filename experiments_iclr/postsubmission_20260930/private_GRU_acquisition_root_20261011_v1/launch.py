"""Launch one detached finite acquisition owner; preserve actual process identities."""
import json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
REMOTE=r"""
import json,os,socket,subprocess,time
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');os.chdir(repo)
phase=repo/'experiments_iclr/postsubmission_20260930';root=phase/'private_GRU_acquisition_root_20261011_v1';config=json.loads((root/'CONFIG.json').read_text())
assert not Path(config['owner_output']).exists() and not Path(config['output']).exists()
argv=[str(repo/'.venv/bin/python'),'-B','-u',str(phase/'native_meanSAGE_private_GRU_execution_source_20261011_v1/owner.py'),str(root/'CONFIG.json')]
with (root/'launch.log').open('xb') as log:
 child=subprocess.Popen(argv,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
def identity(pid):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 a=p.read_text().rsplit(')',1)[1].split();return dict(pid=pid,start_ticks=int(a[19]),pgid=int(a[2]),state=a[0])
for _ in range(75):
 start=Path(config['owner_output'])/'OWNER_START.json'
 if start.exists():break
 if child.poll() is not None:break
 time.sleep(.2)
v=dict(owner_spawn_pid=child.pid,owner_observed=identity(child.pid),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),owner_start=json.loads(start.read_text()) if start.exists() else None)
if v['owner_start']:v['scientific_child_observed']=identity(v['owner_start']['child']['pid'])
end=Path(config['owner_output'])/'OWNER_END.json'
if end.exists():v['owner_end']=json.loads(end.read_text())
print(json.dumps(v))
"""
a=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
r=subprocess.run(a,capture_output=True,text=True,timeout=45)
(HERE/'LAUNCH_TRANSPORT.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
assert r.returncode==0,'Transport failure: inspect same owner before any retry'
v=json.loads(r.stdout);(HERE/'ACTUAL_LAUNCH.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v))
