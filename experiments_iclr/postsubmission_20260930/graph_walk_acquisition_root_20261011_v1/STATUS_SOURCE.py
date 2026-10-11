import json,socket,subprocess
from pathlib import Path
h=socket.gethostname();g=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines();assert h=='anogena-2-0' and g==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
base=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/graph_walk_acquisition_root_20261011_v1');value=dict(route_verified=True)
if (base/'owner_v1/OWNER_END.json').exists():
 value['owner_end']=json.loads((base/'owner_v1/OWNER_END.json').read_text())
 if value['owner_end']['status']!='COMPLETE_EXIT_0':value['failure_log']=(base/'owner_v1/child.log').read_text()[-12000:]
else:
 start=json.loads((base/'owner_v1/OWNER_START.json').read_text());pid=start['child']['pid'];value['child_alive']=Path('/proc/'+str(pid)).exists();value['owner_alive']=Path('/proc/'+str(start['owner']['pid'])).exists();value['child_pid']=pid
for p in sorted((base/'acquisition_v1').glob('block_*/fit_*/PROGRESS.json')):
 value['latest_training_progress']=dict(path=str(p),**json.loads(p.read_text()))
if (base/'acquisition_v1/ACQUISITION_PROGRESS.json').exists():value['whole_panel_progress']=json.loads((base/'acquisition_v1/ACQUISITION_PROGRESS.json').read_text())
print(json.dumps(value),flush=True)
