"""Stdlib transport/supervisor for the root-authorized ONE discarded CPU gate."""
from pathlib import Path
import ast
import hashlib
import json
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REMOTE_BODY = r'''
from pathlib import Path
import base64
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

payload = json.load(sys.stdin)
root = Path(payload['remote_directory'])
expected = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/shared_core_private_learning_native_derivative_execution_20261005_v1')
assert root == expected and not root.exists()
assert socket.gethostname() == 'anogena-2-0'
uuids = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
assert uuids == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
decoded = []
for row in payload['files']:
    relative = Path(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    assert relative.parts[0] in {'source','native_source','JOB.json'}
    content = base64.b64decode(row['base64'],validate=True)
    assert len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256']
    decoded.append((relative,content))
root.mkdir()
for relative,content in decoded:
    path = root/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(root/'source/SOURCE_MANIFEST.json') == 'd69b8551984914c65bc47059a2414fe2f93be837429bc9aafbfa57ee2d5e0075'
assert sha(root/'source/qualify_train_only_sgd.py') == '1c0a2017b772972b4f5bb0cfa267c1b6afdd2b631d1d5d19ed6557194c38af27'
assert sha(root/'source/PARAMETER_PARTITION.json') == 'b374fbcaa0c70331be1a5731107c675990384b3eb7c39a3ffb93ba9d29823dd2'
for directory in ['source','native_source']:
    for row in json.loads((root/directory/'SOURCE_MANIFEST.json').read_text())['files']:
        assert sha(root/directory/row['path']) == row['sha256']
job = json.loads((root/'JOB.json').read_text())
assert job['source_review_approved'] is True and job['operator_qualification_only'] is True and job['fits_authorized'] is False
assert job['seed'] == job['factor_seed'] == 20261005
assert job['expected_hostname'] == socket.gethostname()
assert job['output_directory'] == str(root/'result') and job['native_source_directory'] == str(root/'native_source')
assert job['external_360_second_hard_bound_confirmed'] is True
assert set(job['train_only_inputs']) == {'train_pos.txt','gnn_feature'}
authority = job['train_only_inputs']['gnn_feature']
assert sha(Path(authority['authority_reference'])) == authority['authority_sha256']
stage = dict(time_utc=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),GPU_UUIDs=uuids,
             staged_files=[dict(path=row['path'],bytes=row['bytes'],sha256=row['sha256']) for row in payload['files']],
             sealed_manifest_sha256=sha(root/'source/SOURCE_MANIFEST.json'),job_sha256=sha(root/'JOB.json'),
             fits_authorized=False,numerical_launches=0)
(root/'STAGING_RECEIPT.json').write_text(json.dumps(stage,indent=2)+'\n')
interpreter = Path(payload['interpreter'])
phase = root.parent
assert interpreter == phase/'native_ncn_runtime_20261005_v1/.venv/bin/python' and interpreter.is_file()
assert payload['PYTHONPATH'] == str(phase/'native_ncn_dependency_overlay_20261005_v1')+':/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/.venv/lib/python3.11/site-packages'
env = dict(os.environ)
env.update(PYTHONPATH=payload['PYTHONPATH'],CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
           OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
env.pop('PYTHONHOME',None)
argv = [str(interpreter),str(root/'source/qualify_train_only_sgd.py'),'--job',str(root/'JOB.json'),'--output',str(root/'result')]
started = time.monotonic()
(root/'RUN_STARTED.json').write_text(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),argv=argv,
    hard_timeout_seconds=360,attempt=1,CUDA_VISIBLE_DEVICES='',cpu_threads=2,fits=0),indent=2)+'\n')
timed_out = False
with (root/'child.stdout.log').open('wb') as stdout, (root/'child.stderr.log').open('wb') as stderr:
    process = subprocess.Popen(argv,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,env=env,
                               cwd=root,start_new_session=True)
    try:
        process.wait(timeout=360)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid,signal.SIGKILL)
        process.wait()
elapsed = time.monotonic()-started
diagnostics = {}
for name in ['RESULT.json','FAILURE.json']:
    path = root/'result'/name
    if path.exists(): diagnostics[name] = json.loads(path.read_text())
source_unchanged = all(sha(root/row['path']) == row['sha256'] for row in payload['files'])
receipt = dict(time_utc=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),GPU_UUIDs=uuids,
    attempt=1,numerical_launches=1,exit_code=process.returncode,hard_timeout_seconds=360,timed_out=timed_out,
    inclusive_supervised_seconds=elapsed,interpreter=str(interpreter),PYTHONPATH=payload['PYTHONPATH'],
    CUDA_VISIBLE_DEVICES='',source_bytes_unchanged=source_unchanged,job_sha256=sha(root/'JOB.json'),
    fits=0,VALID_TEST_payloads_opened=False,checkpoints=False,metrics=False,
    stdout_sha256=sha(root/'child.stdout.log'),stderr_sha256=sha(root/'child.stderr.log'))
(root/'EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(staging=stage,execution=receipt,diagnostics=diagnostics,
                     child_stdout=(root/'child.stdout.log').read_text(),child_stderr=(root/'child.stderr.log').read_text()),indent=2))
'''


def main():
    ast.parse(REMOTE_BODY)
    ssh = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=20',
           'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    command = 'python3 -I -c ' + shlex.quote(REMOTE_BODY)
    started = time.monotonic()
    with (ROOT/'STAGING_PAYLOAD.json').open('rb') as payload:
        result = subprocess.run([*ssh,command],stdin=payload,capture_output=True,text=True,timeout=400)
    receipt = dict(exit_code=result.returncode,inclusive_transport_seconds=time.monotonic()-started,
                   remote_supervisor_source_sha256=hashlib.sha256(REMOTE_BODY.encode()).hexdigest(),
                   stdout=result.stdout,stderr=result.stderr)
    (ROOT/'TRANSPORT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode != 0:
        print(json.dumps(receipt,indent=2))
        raise SystemExit(result.returncode)
    remote_result = json.loads(result.stdout)
    for field,name in [('staging','STAGING_RECEIPT.json'),('execution','EXECUTION_RECEIPT.json')]:
        (ROOT/name).write_text(json.dumps(remote_result[field],indent=2)+'\n')
    for name,value in remote_result['diagnostics'].items():
        (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
    (ROOT/'child.stdout.log').write_text(remote_result['child_stdout'])
    (ROOT/'child.stderr.log').write_text(remote_result['child_stderr'])
    print(json.dumps(dict(execution=remote_result['execution'],diagnostics=remote_result['diagnostics']),indent=2))


if __name__ == '__main__':
    main()
