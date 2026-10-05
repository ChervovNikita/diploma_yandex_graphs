"""Observe only the already launched singleton PENCIL cohort's exact handles."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import json,socket,subprocess,sys
launch=json.load(sys.stdin)
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'pencil_citeseer_native300_train_valid_execution_20261005_v1'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def read(p):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<65536
 return json.loads(p.read_text())
def physical(role,owner,argv):
 pid=owner.get('pid',owner.get('PID'));ticks=owner['start_ticks'];p=Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text();v=stat[stat.rfind(')')+2:].split()
  if int(v[19])!=ticks:return dict(role=role,PID=pid,start_ticks=ticks,identity_matches=False,PID_reused=True)
  current=[s.decode() for s in (p/'cmdline').read_bytes().split(bytes([0])) if s]
  cwd=str((p/'cwd').resolve()) if v[0]!='Z' else None
  return dict(role=role,PID=pid,start_ticks=ticks,state=v[0],physically_live=v[0]!='Z',identity_matches=v[0]=='Z' or (current==argv and cwd==str(repo)),cwd=cwd)
 except FileNotFoundError:return dict(role=role,PID=pid,start_ticks=ticks,missing=True)
owner=dict(pid=launch['supervisor_pid'],start_ticks=launch['supervisor_start_ticks'])
rows=[physical('supervisor',owner,launch['command'])]
progress=read(root/'COHORT_PROGRESS.json')
seed=progress.get('active_seed')
value=dict(UTC=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),known_handles=rows,active_seed=seed,completed_fits=len(progress.get('completed_fits',[])),signals_sent=False,new_launches=0,scores_opened=False)
if seed is not None:
 assert type(seed) is int and seed in (0,1,2)
 receipt_path=root/('seed_'+str(seed))/'SUPERVISOR_FIT_RECEIPT.json'
 if receipt_path.is_file():
  receipt=read(receipt_path)
  command=list(launch['command'])
  assert command[1].endswith('/supervisor.py')
  command[1]=command[1][:-len('supervisor.py')]+'worker.py'
  command=command[:-2]+['--seed',str(seed),'--output',str(root/('seed_'+str(seed))/'run01')]
  rows.append(physical('active_scientific_child',receipt['child_identity'],command))
  value['active_child_status']=receipt['status']
  value['active_child_exit_code']=receipt.get('child_exit_code')
if (root/'COHORT_FAILURE.json').is_file():value['cohort_failure']=read(root/'COHORT_FAILURE.json')
print(json.dumps(value))
'''


def main():
    launch = json.loads((HERE / 'LAUNCH_RECEIPT.json').read_text())
    command = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
               '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ConnectTimeout=20', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
               'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python -I -B -']
    submitted = REMOTE.replace("launch=json.load(sys.stdin)", "launch=" + repr(launch))
    result = subprocess.run(command, input=submitted, capture_output=True, text=True, timeout=45)
    output = HERE / ('physical_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir()
    receipt = dict(exit_code=result.returncode, stderr=result.stderr, stdout=result.stdout,
                   remote_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(), submitted_sha256=hashlib.sha256(submitted.encode()).hexdigest(), read_only=True)
    (output / 'REMOTE_CODE.py.txt').write_text(submitted)
    (output / 'TRANSPORT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(result.stderr)
    value = json.loads(result.stdout)
    (output / 'OBSERVATION.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
