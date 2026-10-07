import os,socket,json,subprocess,datetime
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R);assert socket.gethostname()=='anogena-2-0'
D=P/'learnable_internal_be_WikiCS_scientific_family_execution_root_20261007_v1'
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scores_read':False}
for name in ['OWNER.json','RUNNING_CELL.json','MEMORY_WAIT.json','LEDGER.json','DRIVER_FAILURE.json','FAMILY_CLOSURE.json']:
 q=D/name
 if not q.is_file():continue
 a=json.loads(q.read_text())
 if name in ('LEDGER.json','FAMILY_CLOSURE.json'):
  rows=a if name=='LEDGER.json' else a['cells'];out[name]=[{'cell':r['cell'],'status':r['status'],'error_type':r.get('error_type'),'elapsed':r.get('inclusive_cell_driver_seconds')} for r in rows]
 else:out[name]=a
running=out.get('RUNNING_CELL.json',{}).get('cell')
if running:
 q=D/'fits/outputs'/running/'PROGRESS.json'
 if q.is_file():out['native_progress']=json.loads(q.read_text())
q=P/'learnable_internal_be_WikiCS_scientific_family_activation_20261007_v1/DRIVER.log'
if q.stat().st_size:out['controller_error_log']=q.read_text()[-2500:]
out['gpu']=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();assert out['gpu'].split(',')[0].strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
print(json.dumps(out))
