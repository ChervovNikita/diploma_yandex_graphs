from pathlib import Path
from datetime import datetime,timezone
import subprocess,os,json,hashlib,time
ROOT=Path(__file__).resolve().parent
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
DRIVER=PHASE/'graph_ncNC_collab_predictive_driver_preparation_20261003_v1/pilot_run.py'
RELEASE=ROOT/'ROOT_RELEASE.json'
release=json.loads(RELEASE.read_text())
started=time.monotonic();records=[]
def write(name,value):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
write('SUPERVISOR_STARTED.json',{'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'release_sha256':hashlib.sha256(RELEASE.read_bytes()).hexdigest(),'ordinary_host_execution':True,'scientific_fits_authorized':False})
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'],PYTHONPATH=str(REPO/'.gnnm_runtime/buddy_extra_v1/site'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
for i,inv in enumerate(release['authorized_invocations']):
 assert inv['stage'] in ('synthetic','resource_engineering')
 argv=[PYTHON,'-B',str(DRIVER),'--root-release',str(RELEASE),'--stage',inv['stage'],'--unit',inv['unit'],'--base-seed',str(inv['base_seed']),'--output',inv['output_directory']]
 log=ROOT/('child_'+str(i)+'_'+inv['stage']+'_'+inv['unit']+'.log')
 begin=time.monotonic()
 with log.open('x') as f:
  child=subprocess.Popen(argv,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
  write('CHILD_'+str(i)+'_STARTED.json',{'UTC':datetime.now(timezone.utc).isoformat(),'PID':child.pid,'parent_PID':os.getpid(),'argv':argv,'log':str(log)})
  code=child.wait()
 row={'index':i,'stage':inv['stage'],'unit':inv['unit'],'PID':child.pid,'exit_code':code,'elapsed_seconds':time.monotonic()-begin,'output_directory':inv['output_directory']}
 records.append(row);write('CHILD_'+str(i)+'_TERMINAL.json',row)
 if code:break
write('SUPERVISOR_TERMINAL.json',{'UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'status':'COMPLETE' if len(records)==5 and all(x['exit_code']==0 for x in records) else 'FAILED','records':records,'elapsed_seconds':time.monotonic()-started,'unattempted_invocations':release['authorized_invocations'][len(records):],'scientific_fits_authorized':False,'automatic_retry_or_restart':False,'other_jobs_mutated':False})
