"""Read progress metadata of already running owners; no quality or launches."""
import json
from pathlib import Path
from remote_control import call,REMOTE_EXPANSION
code = r'''import json,datetime,subprocess
from pathlib import Path
E=Path(@EXP@)
def live(pid):
 p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 v=p.read_text().rsplit(')',1)[1].split();return {'state':v[0],'start_ticks':v[19]}
root=E/'results/structural_private_risk_v1'
out=json.loads((root/'OUTCOMES.json').read_text());rows=[]
for item in out:
 row={k:item.get(k) for k in ('seed','variant','status','completed_epochs','worker_pid','wall_seconds')}
 if item.get('status')=='started':
  row['worker']=live(item['worker_pid']);cmd=item['command']
  path=Path(cmd[cmd.index('--output')+1])/'PROGRESS.json'
  if path.exists():
   progress=json.loads(path.read_text())
   row['completed_epochs']=progress['train']['epoch'];row['optimizer_calls']=progress['optimizer_calls']
 rows.append(row)
q=json.loads((E/'qualification/pubmed_cuda_queue_24/QUEUE.json').read_text())
r={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'structural_fits':rows,'END_exists':(root/'END.json').exists(),'structural_owner':live(848485),'GRU_worker':live(640428),'PubMed_owner':live(849490),'PubMed_queue_status':q['status'],'GPU_utilization_used_MiB_total_MiB':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip(),'partial_quality_read':False,'new_jobs':0}
print(json.dumps(r))
'''
r=json.loads(call(code.replace('@EXP@',repr(REMOTE_EXPANSION)),timeout=55))
p=Path(__file__).parent/'CURRENT_EXECUTION_28.json'
with p.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
