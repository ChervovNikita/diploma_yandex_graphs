from pathlib import Path
from datetime import datetime,timezone
import argparse,subprocess,json,os,time,traceback
ROOT=Path(__file__).resolve().parent
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
DRIVER=REPO/'experiments_iclr/postsubmission_20260930/graph_ncNC_collab_predictive_driver_preparation_20261003_v1/pilot_run.py'
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
p=argparse.ArgumentParser();p.add_argument('--gpu-index',type=int,choices=[0,1],required=True);args=p.parse_args()
queue=ROOT/('queue_GPU'+str(args.gpu_index));queue.mkdir(exist_ok=False)
release_path=ROOT/('ROOT_FIT_RELEASE_GPU'+str(args.gpu_index)+'.json');release=json.loads(release_path.read_text())
started=time.monotonic();records=[];failure=None
write(queue/'STARTED.json',{'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'release':str(release_path),'GPU_UUID':release['cuda_visible_devices'],'planned_units':len(release['authorized_invocations']),'automatic_retry_or_restart':False})
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'],PYTHONPATH=str(REPO/'.gnnm_runtime/buddy_extra_v1/site'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
try:
 for index,inv in enumerate(release['authorized_invocations']):
  assert inv['stage']=='fit'
  argv=[PYTHON,'-B',str(DRIVER),'--root-release',str(release_path),'--stage','fit','--unit',inv['unit'],'--base-seed',str(inv['base_seed']),'--output',inv['output_directory']]
  begin=time.monotonic();log=queue/(str(index)+'_'+inv['unit']+'_seed'+str(inv['base_seed'])+'.log')
  with log.open('x') as f:
   child=subprocess.Popen(argv,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
   write(queue/('CHILD_'+str(index)+'_STARTED.json'),{'UTC':datetime.now(timezone.utc).isoformat(),'PID':child.pid,'parent_PID':os.getpid(),'argv':argv,'log':str(log),'unit':inv['unit'],'base_seed':inv['base_seed']})
   code=child.wait()
  row={'UTC':datetime.now(timezone.utc).isoformat(),'index':index,'unit':inv['unit'],'base_seed':inv['base_seed'],'PID':child.pid,'exit_code':code,'wall_seconds':time.monotonic()-begin,'output_directory':inv['output_directory']}
  records.append(row);write(queue/('CHILD_'+str(index)+'_TERMINAL.json'),row)
  if code:break
except Exception as e:
 failure={'error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
finally:
 complete=not failure and len(records)==len(release['authorized_invocations']) and all(v['exit_code']==0 for v in records)
 write(queue/'TERMINAL.json',{'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'status':'COMPLETE' if complete else 'FAILED','records':records,'failure':failure,'elapsed_seconds':time.monotonic()-started,'unattempted_invocations':release['authorized_invocations'][len(records):],'partial_outcome_comparison':False,'automatic_retry_or_restart':False,'other_jobs_mutated':False})
