import json,socket,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def who(pid):
 p=Path('/proc')/str(pid)
 try:
  s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split();return {'PID':pid,'start_ticks':int(f[19]),'state':f[0],'ppid':int(f[1])}
 except FileNotFoundError:return None
D=P/'wikics_independent17_successor_execution_root_20261007_v2';r={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller':who(502012),'admitted_dependency':(D/'DEPENDENCY_ADMITTED.json').exists(),'cells':{},'failure_present':(D/'FAILURE.json').exists(),'complete_present':(D/'COMPLETE.json').exists(),'quality_values_read':False}
if r['controller']:assert r['controller']['start_ticks']==6014762157
for arm in ['I_native','U_stage']:
 d=D/'runs'/arm;item={'output_exists':d.exists(),'freeze_present':(d/'FREEZE.json').exists(),'failure_present':(d/'FAILURE.json').exists()};f=d/'PROGRESS.json'
 if f.exists():item['progress']={k:v for k,v in json.loads(f.read_text()).items() if k in ('arm','native_seed','epoch','complete_target','correction_epoch','target_epochs')}
 q=D/'logs'/(arm+'_17.CHILD_STARTED.json')
 if q.exists():
  a=json.loads(q.read_text());i=a['identity'];item['child']=who(i['PID']) if i else None
 item['ordinary_TRAIN_smokes']=[str(p.relative_to(d)) for p in d.glob('**/TRAIN_SMOKE*.json')]
 r['cells'][arm]=item
print(json.dumps(r))
