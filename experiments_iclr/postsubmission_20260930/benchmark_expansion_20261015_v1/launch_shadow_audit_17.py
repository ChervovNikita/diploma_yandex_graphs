"""Root's one bounded launch of the reviewed TRAIN-only diagnostic."""
import json
from pathlib import Path
from remote_control import call, REMOTE_ROOT, REMOTE_EXPANSION, HERE


def main():
    supervisor = '''import json,os,signal,subprocess,time
from pathlib import Path
E=Path(EXPANSION)
command=json.loads((E/'analysis/shared_gradient_shadow_audit_v1/COMMAND.json').read_text())
control=E/'analysis/shared_gradient_shadow_execution_v1'
result={'started_unix':time.time(),'command':command,'automatic_retries':0}
environment=os.environ.copy();environment.update(command['environment'])
with (control/'WORKER.log').open('xb') as log:
 child=subprocess.Popen(command['argv'],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result['worker_pid']=child.pid
 result['worker_start_ticks']=int(Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
 (control/'START.json').write_text(json.dumps(result,indent=2)+'\\n')
 try:result['returncode']=child.wait(timeout=1900)
 except subprocess.TimeoutExpired:
  result['wall_limit_exceeded']=True
  os.killpg(child.pid,signal.SIGTERM)
  try:result['returncode']=child.wait(timeout=20)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGKILL);result['returncode']=child.wait(timeout=10)
 result['finished_unix']=time.time()
 (control/'END.json').write_text(json.dumps(result,indent=2)+'\\n')
'''.replace('EXPANSION', repr(REMOTE_EXPANSION)).replace('ROOT', repr(REMOTE_ROOT))
    code = '''import json,os,subprocess,sys,hashlib,time
from pathlib import Path
E=Path(EXPANSION)
assert hashlib.sha256((E/'analysis/shared_gradient_shadow_audit_v1/MANIFEST.json').read_bytes()).hexdigest()=='ebccdc3ebe44c31cf9cacd205096e6b44b396f552656d02ddf5a4e16b8bdba30'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
assert set(gpu)<= {'640428'},'Unexpected CUDA peer; root must review'
if '640428' in gpu:assert int(Path('/proc/640428/stat').read_text().rsplit(')',1)[1].split()[19])==6055039165
assert not (E/'results/shared_gradient_shadow_audit_v1').exists()
control=E/'analysis/shared_gradient_shadow_execution_v1';control.mkdir()
(control/'supervisor.py').write_text(SUPERVISOR_LITERAL)
with (control/'SUPERVISOR.log').open('xb') as log:
 owner=subprocess.Popen([sys.executable,'-B',str(control/'supervisor.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'owner_pid':owner.pid,'owner_start_ticks':int(Path('/proc/'+str(owner.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]),'launch_unix':time.time(),'control':str(control),'work':'three disposable complete TRAIN epochs; no VALID/TEST','whole_process_limit_seconds':1900,'prelaunch_GPU_peers':gpu,'review':'Original train_update/Adam retained; diagnostic replay restores parameters, buffers and RNG. Sole GRU peer; 15.3GB hard cgroup headroom plus 14.9GB inactive file. No host settings or caches changed.'}
(control/'LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''.replace('EXPANSION', repr(REMOTE_EXPANSION)).replace('SUPERVISOR_LITERAL', repr(supervisor)).replace('ROOT', repr(REMOTE_ROOT))
    result = json.loads(call(code))
    (HERE/'SHADOW_LAUNCH_17.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
