"""Release the fixed fresh-all40 study only after all eight focused checks pass."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parent
LOGIN='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
MANIFEST='ee2cc8623c8752e1fc5a56dde5968c624d2e9e9b735f9534672fae6549c592d4'
REMOTE=r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys
repo=Path(sys.argv[1]);manifest_sha=sys.argv[2];phase=repo/'experiments_iclr/postsubmission_20260930';root=phase/'graph_mixed_block_execution_root_20261003_v1';packet=phase/'graph_mixed_block_training_preparation_20261003_v2'
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo)
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def descriptor(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert descriptor(packet/'MANIFEST.json')['sha256']==manifest_sha
sys.path.insert(0,str(packet));import common as c
c.packet_guard()
qpath=packet/'qualification/root_focused_qualification_run02/QUALIFICATION.json'
qualified=json.loads(qpath.read_text());freeze=root/'FROZEN_STUDY.json';freeze_sha=descriptor(freeze)['sha256']
assert qualified['status']=='qualified' and qualified['synthetic_same_state_witness_passed'] is True
assert qualified['prepared_manifest_sha256']==manifest_sha and qualified['study_freeze_sha256']==freeze_sha
assert qualified['scientific_design_sha256']=='739160acd4ffb0380348f79496e88624db6163f4a3a93902b043d58eaacc18c5'
assert qualified['originals_preserved'] is True and qualified['validation_or_test_scored'] is False
assert [(r['dataset'],r['policy']) for r in qualified['rows']]==[(d,p) for d in c.DATASETS for p in c.POLICIES]
assert all(r['status']=='qualified' and r['consecutive_updates']==6 and r['next_step_complete_state_replay'] is True for r in qualified['rows'])
assert json.loads(freeze.read_text())['baseline_reuse']==dict(mode='fresh_all40',audit=None)
assert not list((packet/'runs').glob('*/STUDY_STARTED.json'))
release=json.loads((packet/'TRAINING_RELEASE_TEMPLATE.json').read_text())
release.update(device='cpu',threads=1,execution_authorized=True,root_observed_qualification=True,prepared_manifest_sha256=manifest_sha,study_freeze_sha256=freeze_sha,qualification=descriptor(qpath),run_name='root_full40_run01',status='root_released_fixed_independent_full40',root_release_UTC=datetime.now(timezone.utc).isoformat(),address_space_limit_bytes=16*1024**3,resource_limits_installed_preimport=True)
release_path=root/'TRAINING_RELEASE_run01.json'
with release_path.open('x') as f:json.dump(release,f,indent=2,sort_keys=True);f.write('\n')
c.admission(str(freeze),str(release_path),'training')
runner=root/'training_supervisor_run01.py';run=root/'transport/root_full40_run01';run.mkdir(parents=True,exist_ok=False)
with runner.open('x') as f:f.write(SUPERVISOR_SOURCE)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (run/'supervisor.stdout.log').open('x') as out,(run/'supervisor.stderr.log').open('x') as err:
 child=subprocess.Popen([str(repo/'.venv/bin/python'),'-B',str(runner)],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
start=dict(UTC=datetime.now(timezone.utc).isoformat(),PID=child.pid,status='fixed_full40_training_dispatched',source_manifest_sha256=manifest_sha,root_release=descriptor(release_path),qualification=descriptor(qpath),source_freeze=descriptor(freeze),runner=descriptor(runner),paired_cases=40,baseline_mode='fresh_all40',test_labels_closed=True,original_scores_changed=False,CP_confirmation_started=False)
with (run/'STARTED.json').open('x') as f:json.dump(start,f,indent=2);f.write('\n')
print(json.dumps(start))
'''
SUPERVISOR=r'''from pathlib import Path
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
'''


def main():
    code=REMOTE.replace('SUPERVISOR_SOURCE',repr(SUPERVISOR))
    (ROOT/'FULL40_REMOTE_DISPATCH_CODE.txt').write_text(code)
    ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes',LOGIN]
    result=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',code,REPO,MANIFEST])],capture_output=True,text=True,timeout=60)
    record=dict(UTC=datetime.now(timezone.utc).isoformat(),destination=LOGIN,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,executed_code_sha256=hashlib.sha256(code.encode()).hexdigest())
    with (ROOT/'FULL40_TRANSPORT_RECEIPT.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
