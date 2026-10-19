"""Compact live progress and failure logs for the three authorized queue owners."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
sys.path.insert(0, str(E))
sys.path.insert(0, str(E/'planning/three_gpu_sync_20261015_v1'))
import remote_control
import remote51

CODE = r'''
from pathlib import Path
import json,subprocess,datetime,hashlib
def tail(path,limit=4000):
 if not path.exists():return None
 with path.open('rb') as stream:
  stream.seek(max(0,path.stat().st_size-limit));return stream.read(limit).decode(errors='replace')
root=Path(ROOT_VALUE);host=HOST_VALUE;e=root/'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
queue=next((e/('campaign_execution_20261016_v'+str(v)) for v in (7,6,5,4,3,2,1) if (e/('campaign_execution_20261016_v'+str(v))/'runs').exists()),e/'campaign_execution_20261016_v1');result={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':host,'queue_version':queue.name,'workers':[]}
for worker in ([host+'/gpu0'] if host=='allocation' else [host+'/gpu0',host+'/gpu1']):
 p=queue/'runs'/worker.replace('/','_');row={'worker':worker}
 if (p/'STATUS.json').exists():row['status']=json.loads((p/'STATUS.json').read_text())
 row['tmux']=subprocess.run(['tmux','list-panes','-t','gnnm-allfit-'+worker.replace('/','-')+'-20261016','-F','#{pane_dead}'],capture_output=True,text=True).stdout.strip()
 if row.get('status',{}).get('phase') in ('smoke_failed','supervisor_error','full_failed') or row.get('status',{}).get('counts',{}).get('failed',0):
  logs=sorted(p.glob('OWNER-*.log'));row['owner_log_tail']=tail(logs[-1]) if logs else ''
  failures=[]
  ids=row.get('status',{}).get('failed_fit_ids',[])
  states=[p/'smokes'/hashlib.sha256(i.encode()).hexdigest()/'STATE.json' for i in ids[:4]] if ids else (p/'smokes').glob('*/STATE.json')
  for state in states:
   value=json.loads(state.read_text())
   if value['status']=='failed':
    directory=state.parent/'attempts'/('%04d'%value['attempt']);log=directory/'native.log'
    failures.append({'state':value,'native_log_tail':tail(log)})
    if len(failures)>=4:break
  row['failures']=failures
  row['failure_examples_truncated']=row.get('status',{}).get('counts',{}).get('failed',0)>len(failures)
 if (p/'CAPACITY_WAIT.json').exists():
  capacity=json.loads((p/'CAPACITY_WAIT.json').read_text());row['last_capacity_decision']={'id':capacity['id'],'decision':capacity['decision'],'sampled_at':capacity.get('live',{}).get('sampled_at'),'historical_record_not_current_worker_state':True}
 result['workers'].append(row)
result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
if host=='allocation':
 p=e/'planning/all_fit_launch_20261016_v1/ALLOCATION_PREREQUISITES.json'
 result['prerequisites']=json.loads(p.read_text()) if p.exists() else None
print(json.dumps(result))
'''


def observe(host):
    remote = remote_control if host == 'allocation' else remote51
    root = remote.REMOTE_ROOT if host == 'allocation' else remote.ROOT
    code = CODE.replace('ROOT_VALUE', repr(root)).replace('HOST_VALUE', repr(host))
    result = json.loads(remote.call(code, timeout=45))
    folder = HERE/'observations'; folder.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    (folder/(host+'_'+stamp+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(observe, ('allocation', 'server51')):
            print(json.dumps(result), flush=True)
