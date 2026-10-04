from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
ROOT=PHASE/'graph_view_gpu77_runtime_qualification_root_20261004_v1'
PYTHON='/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python3.12'
UUID='GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
UUIDS=['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
assert Path.cwd()==REPO and os.uname().nodename=='peptide'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==UUIDS
env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT),CUBLAS_WORKSPACE_CONFIG=':4096:8')
def proc(pid):
 raw=(Path('/proc')/str(pid)/'stat').read_text();f=raw[raw.rfind(')')+2:].split()
 return dict(PID=pid,state=f[0],parent=int(f[1]),group=int(f[2]),session=int(f[3]),start_ticks=int(f[19]))
def save(name,value):
 with (ROOT/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
rows=[];start=time.perf_counter()
for stage in ('local','global'):
 free=int(subprocess.run(['nvidia-smi','--id='+UUID,'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip())
 if free<35000:
  rows.append(dict(stage=stage,status='NOT_STARTED_INSUFFICIENT_FREE_MEMORY',free_MiB=free));break
 command=[PYTHON,'-B',str(ROOT/'check_one.py'),'--family','bank','--condition','tied_persistent','--stage',stage]
 started=time.perf_counter()
 with (ROOT/(stage+'.stdout')).open('xb') as out,(ROOT/(stage+'.stderr')).open('xb') as err:
  child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
  identity=proc(child.pid)
  save(stage+'_CHILD_STARTED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),identity=identity,command=command,selected_GPU_UUID=UUID,free_MiB_at_start=free))
  exit_code=child.wait()
 row=dict(stage=stage,exit_code=exit_code,wall_seconds=time.perf_counter()-started,identity=identity,command=command,child_closed_and_reaped=True,stdout_sha256=hashlib.sha256((ROOT/(stage+'.stdout')).read_bytes()).hexdigest(),stderr_sha256=hashlib.sha256((ROOT/(stage+'.stderr')).read_bytes()).hexdigest())
 result=ROOT/('bank_tied_persistent_'+stage+'.json')
 if exit_code==0:
  row['result']=json.loads(result.read_text());row['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
 else:row['stderr']=(ROOT/(stage+'.stderr')).read_text()
 save(stage+'_PHYSICAL_TERMINAL.json',row);rows.append(row)
 if exit_code!=0:break
save('QUALIFICATION.json',dict(UTC=datetime.now(timezone.utc).isoformat(),cases=rows,expected_cases=2,wall_seconds=time.perf_counter()-start,all_component_checks_complete=len(rows)==2 and all(r.get('exit_code')==0 for r in rows),route=dict(login='shmelev@192.168.18.77',repository=str(REPO),physical_UUIDs=UUIDS,selected_GPU_UUID=UUID),predictive_values_read=False,VALIDATION_or_TEST_targets_read=False,scientific_training_updates=0,checkpoint_or_logits_saved=False,eligible_as_donor=False,automatic_retry=False,job_signals_sent=False,co_resident_with_DDI_queue=True))
save('SUPERVISOR_TERMINAL.json',dict(UTC=datetime.now(timezone.utc).isoformat(),identity=proc(os.getpid()),completed=True,wall_seconds=time.perf_counter()-start,job_signals_sent=False))
