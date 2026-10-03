from pathlib import Path
from datetime import datetime,timezone
import json,os,resource,signal,subprocess,sys
root=Path(__file__).resolve().parent;phase=root.parent;repo=phase.parents[1]
assert str(repo)=='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs' and Path.cwd().resolve()==repo
packet=phase/'graph_mixed_block_training_preparation_20261003_v2';run=root/'transport/root_full40_run01';release=root/'TRAINING_RELEASE_run01.json';r=json.loads(release.read_text())
assert r['execution_authorized'] is True and r['root_observed_qualification'] is True
resource.setrlimit(resource.RLIMIT_AS,(r['address_space_limit_bytes'],r['address_space_limit_bytes']))
resource.setrlimit(resource.RLIMIT_CPU,(86400,87000))
command=[sys.executable,'-B',str(packet/'train_policies.py'),'--freeze',str(root/'FROZEN_STUDY.json'),'--admission',str(release),'--run-name','root_full40_run01']
start=datetime.now(timezone.utc).isoformat();timed_out=False
with (run/'child.stdout.log').open('x') as out,(run/'child.stderr.log').open('x') as err:
 child=subprocess.Popen(command,cwd=repo,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 with (run/'CHILD.json').open('x') as f:json.dump(dict(PID=child.pid,command=command),f,indent=2);f.write('\n')
 try:code=child.wait(timeout=86400)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
result=dict(start_UTC=start,terminal_UTC=datetime.now(timezone.utc).isoformat(),exit_code=code,timed_out=timed_out,unrelated_processes_touched=False,automatic_restart=False)
with (run/'SUPERVISOR_TERMINAL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
raise SystemExit(code)
