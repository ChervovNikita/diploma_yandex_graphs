import json,socket,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
A=P/'learnable_internal_be_resource_activation_root_20261007_v1';E=P/'learnable_internal_be_WikiCS_resource_execution_root_20261007_v1'
proc=Path('/proc/505130/stat');owner=None
if proc.exists():
 stat=proc.read_text();f=stat[stat.rfind(')')+2:].split();assert int(f[19])==6015035338;owner={'PID':505130,'start_ticks':int(f[19]),'state':f[0]}
rows=[]
for f in sorted((E/'receipts').glob('*_RESOURCE.json')):
 a=json.loads(f.read_text());rows.append({'path':str(f.relative_to(P)),'passed':a['passed'],'status':a['status'],'inclusive_seconds':a['inclusive_seconds'],'peak_GPU_bytes':a['peak_GPU_bytes'],'seed':a['seed'],'work':a['work'] if not a['passed'] else {k:a['work'].get(k) for k in ('members','own_views','modes','committed_TRAIN_updates')},'failure_type':a['failure_type']})
result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owner':owner,'resources':rows,'complete':(E/'COMPLETE.json').exists(),'failed':(E/'FAILURE.json').exists(),'predictive_scores_read':False}
if result['failed']:
 result['driver_log_tail']=(A/'DRIVER.log').read_text()[-4000:]
 logs=sorted((E/'receipts').glob('*_CHILD.log'))
 result['child_log_tails']={f.name:f.read_text()[-2500:] for f in logs}
print(json.dumps(result))
