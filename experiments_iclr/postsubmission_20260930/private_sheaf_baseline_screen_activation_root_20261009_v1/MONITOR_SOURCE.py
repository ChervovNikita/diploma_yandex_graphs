from pathlib import Path
import json,socket,subprocess,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'private_sheaf_baseline_screen_activation_root_20261009_v1';D=P/'private_sheaf_baseline_screen_execution_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def ident(pid):
 try:
  s=Path('/proc',str(pid),'stat').read_text();v=s[s.rfind(')')+2:].split();return {'pid':pid,'start_ticks':int(v[19]),'state':v[0]}
 except FileNotFoundError:return None
launch=json.loads((A/'LAUNCH.json').read_text());parent=ident(launch['parent']['pid']);out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent':parent,'scientific_scores_read':False,'cells':{}}
if parent:assert parent['start_ticks']==launch['parent']['start_ticks']
if (A/'WORKER_OWNER.json').exists():
 owner=json.loads((A/'WORKER_OWNER.json').read_text());out['child']=ident(owner['child']['pid']);out['saved_child']=owner['child']
 if out['child']:assert out['child']['start_ticks']==owner['child']['start_ticks']
for f in D.glob('*/RESULT.json'):
 x=json.loads(f.read_text());out['cells'][f.parent.name]={'status':x['status']}
 history=f.parent/'HISTORY.jsonl'
 if history.exists():
  lines=history.read_text().splitlines();out['cells'][f.parent.name]['completed_epochs']=json.loads(lines[-1])['epoch'] if lines else 0
for name in ('RESOURCE_ADMISSION.json','TERMINAL.json'):
 f=A/name
 if f.exists():out[name]=json.loads(f.read_text())
if parent is None:out['owner_log']=(A/'OWNER.log').read_text()[-4000:];out['worker_log']=(A/'WORKER.log').read_text()[-4000:] if (A/'WORKER.log').exists() else None
print(json.dumps(out))
