import json,os,signal,subprocess,time
from pathlib import Path
E=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1')
command=json.loads((E/'analysis/shared_gradient_shadow_audit_v1/COMMAND.json').read_text())
control=E/'analysis/shared_gradient_shadow_execution_v1'
result={'started_unix':time.time(),'command':command,'automatic_retries':0}
environment=os.environ.copy();environment.update(command['environment'])
with (control/'WORKER.log').open('xb') as log:
 child=subprocess.Popen(command['argv'],cwd='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs',env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result['worker_pid']=child.pid
 result['worker_start_ticks']=int(Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
 (control/'START.json').write_text(json.dumps(result,indent=2)+'\n')
 try:result['returncode']=child.wait(timeout=1900)
 except subprocess.TimeoutExpired:
  result['wall_limit_exceeded']=True
  os.killpg(child.pid,signal.SIGTERM)
  try:result['returncode']=child.wait(timeout=20)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGKILL);result['returncode']=child.wait(timeout=10)
 result['finished_unix']=time.time()
 (control/'END.json').write_text(json.dumps(result,indent=2)+'\n')
