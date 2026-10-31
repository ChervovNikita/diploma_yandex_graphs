"""Progress/identity only for the finite no-update resource owner."""
import argparse,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
REMOTE=r"""
import json,socket,subprocess
from pathlib import Path
from collections import Counter
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/massive_native_packet_20261012_v2')
owner=json.loads((root/'resource_owner_v1/ACTUAL_LAUNCH.json').read_text())['child']
f=Path('/proc')/str(owner['pid'])/'stat'
observed=None
if f.exists():
 fields=f.read_text().rsplit(')',1)[1].split();observed={'pid':owner['pid'],'start_ticks':int(fields[19]),'pgid':int(fields[2]),'state':fields[0]}
 assert all(observed[k]==owner[k] for k in ('pid','start_ticks','pgid'))
f=root/'resources_v1/MANIFEST.json'
v=json.loads(f.read_text()) if f.exists() else None
value={'owner':owner,'observed':observed,'status':v.get('status') if v else None,'cases_closed':len(v['cases']) if v else 0,'case_states':dict(Counter(r['status'] for r in v['cases'])) if v else {},'failed_cases':[{'cell':r['cell'],'case':r['case'],'oom':r['oom']} for r in v['cases'] if r['status']!='complete'] if v else []}
value['log_tail']=(root/'resource_owner_v1/child.log').read_text()[-4500:]
for name in ('RESOURCES.json','COST_PROJECTION.json'):
 f=root/'resources_v1'/name
 if f.exists():
  item=json.loads(f.read_text());value[name]={k:x for k,x in item.items() if k not in ('profiles','cells')}
print(json.dumps(value))
"""
p=argparse.ArgumentParser();p.add_argument('label');args=p.parse_args();assert args.label.replace('_','').isalnum()
f=HERE/('NATIVE_RESOURCE_OBSERVATION_'+args.label+'.json');assert not f.exists()
a=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',shlex.join(['/usr/bin/python3','-I','-S','-B','-c',REMOTE])]
r=subprocess.run(a,capture_output=True,text=True,timeout=45)
f.write_text(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
assert r.returncode==0,r.stdout+r.stderr
print(r.stdout)
