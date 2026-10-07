"""One detached resource parent with compact CPU-cost accounting; no retry."""
import json,os,resource,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PHASE=ROOT.parent
REPO=PHASE.parents[1]
PYTHON=PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
QUALIFIER=PHASE/'learnable_internal_be_resource_qualifier_source_20261007_v1'
def ticks(pid):return int(Path('/proc/'+str(pid)+'/stat').read_text().rsplit(') ',1)[1].split()[19])
started=time.monotonic();os.umask(0o077)
assert not (ROOT/'OUTER_TERMINAL.json').exists()
old=resource.getrusage(resource.RUSAGE_CHILDREN)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-44039938-fd82-41d2-fefd-de71514e2fac',PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages'));env.pop('PYTHONHOME',None)
with (ROOT/'supervisor.log').open('x') as log:
 child=subprocess.Popen([str(PYTHON),'-B',str(QUALIFIER/'supervise.py'),'--job',str(ROOT/'job.json')],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  owner={'outer_PID':os.getpid(),'outer_start_ticks':ticks(os.getpid()),'resource_supervisor_PID':child.pid,'resource_supervisor_start_ticks':ticks(child.pid)}
  (ROOT/'SUPERVISOR_OWNER.json').write_text(json.dumps(owner,indent=2,sort_keys=True)+'\n')
 finally:code=child.wait()
new=resource.getrusage(resource.RUSAGE_CHILDREN)
result={'exit_code':code,'wall_seconds':time.monotonic()-started,'user_CPU_seconds':new.ru_utime-old.ru_utime,'system_CPU_seconds':new.ru_stime-old.ru_stime,'peak_RSS_bytes':int(new.ru_maxrss)*1024,'input_blocks':new.ru_inblock-old.ru_inblock,'output_blocks':new.ru_oublock-old.ru_oublock,'supervisor_owner':owner,'resource_only':True,'predictive_scores_opened':False,'automatic_retry':False,'resource_weights_used_as_fit_start':False}
(ROOT/'OUTER_TERMINAL.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
