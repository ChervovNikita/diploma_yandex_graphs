"""Observe the exact finite scientific owner; progress only, no prediction scoring."""
import argparse,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
REMOTE=r"""
import json,socket,subprocess
from pathlib import Path
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
p=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/private_GRU_acquisition_root_20261011_v1')
v={}
for name in ('OWNER_START.json','OWNER_END.json'):
 f=p/'owner_v1'/name
 if f.exists():v[name]=json.loads(f.read_text())
start=v.get('OWNER_START.json',{})
for key in ('owner','child'):
 ident=start.get(key)
 if ident:
  f=Path('/proc')/str(ident['pid'])/'stat'
  if f.exists():
   a=f.read_text().rsplit(')',1)[1].split();observed=dict(pid=ident['pid'],start_ticks=int(a[19]),pgid=int(a[2]),state=a[0]);v[key+'_observed']=observed
   assert all(observed[k]==ident[k] for k in ('pid','start_ticks','pgid'))
  else:v[key+'_observed']=None
out=p/'acquisition_v1'
f=out/'ACQUISITION_PROGRESS.json'
if f.exists():v['acquisition_progress']=json.loads(f.read_text())
v['fits_progress']=[dict(relative=str(f.relative_to(out)),progress=json.loads(f.read_text())) for f in sorted(out.glob('block_*/fit_*/PROGRESS.json'))]
v['complete_panel_exists']=(out/'COMPLETE_PANEL.json').exists()
if 'OWNER_END.json' in v and v['OWNER_END.json']['status']!='COMPLETE_EXIT_0':
 f=p/'owner_v1/child.log'
 if f.exists():v['failure_log']=f.read_text()[-14000:]
print(json.dumps(v))
"""
p=argparse.ArgumentParser();p.add_argument('label');args=p.parse_args();assert args.label.replace('_','').isalnum()
a=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
r=subprocess.run(a,capture_output=True,text=True,timeout=45)
f=HERE/('OBSERVATION_'+args.label+'.json');assert not f.exists();v=dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
f.write_text(json.dumps(v,indent=2)+'\n');assert r.returncode==0;print(r.stdout.strip())
