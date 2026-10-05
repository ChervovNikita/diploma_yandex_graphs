"""Execute the reviewed complete-cohort analysis once; retain small receipts only."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REMOTE = r'''
from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
os.chdir(repo)
assert socket.gethostname()=='anogena-2-0'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
analysis=phase/'citeseer_frame_complete_analysis_20261005_v2'
cohort=phase/'citeseer_endpoint_frame_paired_development_20261005_v1'
terminal=cohort/'COHORT_FREEZE.json'
assert terminal.is_file(),'Wait for exact completed cohort; no outcomes opened'
freeze=json.loads(terminal.read_text())
assert freeze['complete'] is True and len(freeze['completed_physical_fits'])==36 and freeze['TEST_access'] is False
assert sha(analysis/'analyze_valid.py')=='4006920c39200f9eb93ecffaaf55718be3917fe579295e9bbd4ba7045d9f8cee'
assert not any((analysis/name).exists() for name in ('START.json','RESULTS.json','FAILURE.json'))
out=phase/'citeseer_frame_complete_analysis_execution_root_20261005_v1'
assert not out.exists();out.mkdir()
env=dict(os.environ)
env.update(PYTHONPATH=str(phase/'native_ncn_dependency_overlay_20261005_v1')+':'+str(repo/'.venv/lib/python3.11/site-packages'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
command=[str(phase/'native_ncn_runtime_20261005_v1/.venv/bin/python'),'-B',str(analysis/'analyze_valid.py')]
start=time.monotonic()
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),command=command,analysis_source_sha256=sha(analysis/'analyze_valid.py'),cohort_freeze_sha256=sha(terminal),child_environment={k:env[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_VISIBLE_DEVICES')},fits=0,optimizer_updates=0,TEST_access=False)
try:
 child=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True,timeout=120)
 receipt.update(exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr)
except subprocess.TimeoutExpired as error:
 receipt.update(exit_code=None,status='timeout_preserved_no_retry',stdout=(error.stdout or b'').decode() if isinstance(error.stdout,bytes) else error.stdout,stderr=(error.stderr or b'').decode() if isinstance(error.stderr,bytes) else error.stderr)
receipt['inclusive_seconds']=time.monotonic()-start
(out/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
files=[]
for p in [out/'EXECUTION_RECEIPT.json',terminal,analysis/'START.json',analysis/'RESULTS.json',analysis/'FAILURE.json']:
 if p.is_file():
  b=p.read_bytes();assert len(b)<1000000
  files.append(dict(path=str(p.relative_to(phase)),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),base64=base64.b64encode(b).decode()))
print(json.dumps(dict(receipt=receipt,files=files)))
'''


def main():
    assert not (HERE / 'TRANSPORT.json').exists()
    ssh = ['ssh', '-T', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=15',
           'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    result = subprocess.run([*ssh, shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE])],
                            capture_output=True, text=True, timeout=150)
    receipt = dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=result.returncode,
                   stderr=result.stderr, stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if result.returncode:
        receipt['stdout'] = result.stdout
    with (HERE / 'TRANSPORT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    for row in payload['files']:
        path = PHASE / row['path']
        assert path.resolve().is_relative_to(PHASE.resolve())
        data = base64.b64decode(row['base64'], validate=True)
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        if path.exists():
            assert path.read_bytes() == data
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream: stream.write(data)
    print(json.dumps(payload['receipt']))


if __name__ == '__main__':
    main()
