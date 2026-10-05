"""One CPU fixture on the authorized allocation after 18.77 network failure."""
from pathlib import Path
from datetime import datetime, timezone
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def save(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    prior = PHASE/'endpoint_frame_component_execution_20261005_v1/PROTOCOL.json'
    protocol = json.loads(prior.read_text())
    rows=[]
    for row in protocol['files']:
        raw=(PHASE/row['path']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        rows.append(dict(row,data=base64.b64encode(raw).decode()))
    protocol.update(UTC=datetime.now(timezone.utc).isoformat(),expected_host_route=LOGIN,
        expected_gpu_uuid=UUID,reason='18.77 failed before authentication; network diagnostic timed out. No numerical execution observed there.',
        prior_failure_reference='endpoint_frame_component_execution_20261005_v1/SSH_ROUTE_DIAGNOSTIC.json',
        source_protocol_sha256=hashlib.sha256(prior.read_bytes()).hexdigest())
    protocol.pop('expected_host',None)
    save('PROTOCOL.json',protocol)
    code='''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,subprocess,time
repo=Path(REPO);phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd()==repo
g=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True)
assert g.stdout.splitlines()==[UUID]
for row in ROWS:
 p=phase/row['path'];assert p.resolve().is_relative_to(phase)
 raw=base64.b64decode(row['data'],validate=True)
 assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==raw
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as stream:stream.write(raw)
root=phase/'endpoint_frame_onegpu_component_20261005_v1'
root.mkdir(parents=True,exist_ok=True)
assert not (root/'ATTEMPT.json').exists()
with (root/'ATTEMPT.json').open('x') as stream:json.dump(dict(UTC=datetime.now(timezone.utc).isoformat(),sources=[{k:v for k,v in r.items() if k!='data'} for r in ROWS]),stream)
command=[str(repo/'.venv/bin/python'),'-B',str(phase/'native_endpoint_private_factor_mechanism_assessment_20261005_v1/check_cpu.py')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic()
try:
 child=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True,timeout=60)
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='PASS' if child.returncode==0 else 'FAIL',exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr,elapsed_seconds=time.monotonic()-start,command=command,CUDA_VISIBLE_DEVICES='',gpu_route=UUID,fit_count=0,optimizer_steps=0,data_checkpoint_outcome_access=False)
except subprocess.TimeoutExpired as error:
 result=dict(UTC=datetime.now(timezone.utc).isoformat(),status='TIMEOUT',elapsed_seconds=time.monotonic()-start,stdout=str(error.stdout),stderr=str(error.stderr),fit_count=0,optimizer_steps=0,data_checkpoint_outcome_access=False)
with (root/'RECEIPT.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write(chr(10))
print(json.dumps(result))
'''
    code='REPO='+repr(REPO)+'\nUUID='+repr(UUID)+'\nROWS='+repr(rows)+'\n'+code
    command='cd '+shlex.quote(REPO)+" && /usr/bin/python3 -I -S -B - <<'COMPONENTPY'\n"+code+'\nCOMPONENTPY\n'
    assert len(command.encode())<100000
    ssh=['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
         '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
         '-o','UpdateHostKeys=no','-o','ConnectTimeout=15',LOGIN]
    result=subprocess.run([*ssh,command],capture_output=True,text=True,timeout=90)
    save('TRANSPORT.json',dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr,command_sha256=hashlib.sha256(command.encode()).hexdigest()))
    assert result.returncode==0,result.stderr
    receipt=json.loads(result.stdout)
    save('RECEIPT.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
