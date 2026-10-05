"""Stage one reviewed, fixed CPU score analysis on the authorized allocation.

Only compact results/receipts return to the Mac. Existing evidence is untouched.
"""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
EXPECTED_SOURCE = '2809880f33e31228375436c36cb7443286ce4d6874ffcb619d5c07832a1208cc'
EXPECTED_PROTOCOL = '2b50dfe915978671e313853b993eca97dca745e6adb04cbc54f08ebe02e3d7e0'
REMOTE = r'''
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,os,socket,subprocess,sys,time
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
phase=repo/'experiments_iclr/postsubmission_20260930'
root=phase/'citeseer_saved_independent_bank_pooling_screen_20261005_v3'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
payload=json.load(sys.stdin)
assert phase.resolve(strict=True)==phase and phase.is_dir() and not phase.is_symlink()
assert root.parent==phase
assert payload['root_source_review_approved'] is True and not root.exists()
files=[]
for row in payload['files']:
 assert row['name'] in ('analyze.py','PROTOCOL.json')
 data=base64.b64decode(row['base64'],validate=True)
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 files.append((row['name'],data))
assert {n for n,b in files}=={'analyze.py','PROTOCOL.json'}
root.mkdir()
for name,data in files:
 with (root/name).open('xb') as stream:stream.write(data)
env=dict(os.environ)
env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',
 PYTHONPATH=str(phase/'native_ncn_dependency_overlay_20261005_v1')+':'+str(repo/'.venv/lib/python3.11/site-packages'))
command=[str(phase/'native_ncn_runtime_20261005_v1/.venv/bin/python'),'-B',str(root/'analyze.py')]
started=time.monotonic()
receipt=dict(start_UTC=datetime.now(timezone.utc).isoformat(),command=command,
 source_review_sha256=payload['source_review_sha256'],root_source_review_approved=True,
 TEST_access=False,model_forwards=0,fit_count=0,timeout_seconds=120)
try:
 child=subprocess.run(command,cwd=repo,env=env,capture_output=True,text=True,timeout=120)
 receipt.update(exit_code=child.returncode,stdout=child.stdout,stderr=child.stderr)
except subprocess.TimeoutExpired as error:
 receipt.update(exit_code=None,status='OWN_CHILD_TIMEOUT_NO_RETRY',
  stdout=(error.stdout or b'').decode() if isinstance(error.stdout,bytes) else error.stdout,
  stderr=(error.stderr or b'').decode() if isinstance(error.stderr,bytes) else error.stderr)
receipt.update(terminal_UTC=datetime.now(timezone.utc).isoformat(),inclusive_seconds=time.monotonic()-started)
with (root/'EXECUTION_RECEIPT.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
fetched=[]
for name in ('START.json','RESULTS.json','EXECUTION_RECEIPT.json'):
 p=root/name
 if p.is_file():
  assert not p.is_symlink() and p.stat().st_size<65536
  raw=p.read_bytes()
  fetched.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),base64=base64.b64encode(raw).decode()))
print(json.dumps(dict(receipt=receipt,files=fetched)))
'''

def main():
    review = HERE/'ROOT_EXECUTION_ADMISSION.json'
    assert review.is_file() and not review.is_symlink()
    admission = json.loads(review.read_text())
    assert admission['source_review_resolved'] is True
    assert admission['study_directory'] == HERE.name
    assert set(admission['approved_files']) == {'analyze.py','PROTOCOL.json','execute.py'}
    for name, expected in admission['approved_files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected
    independent = HERE.parent/admission['independent_review_project_relative']
    assert independent.resolve().is_relative_to(HERE.parent.resolve()) and not independent.is_symlink()
    assert hashlib.sha256(independent.read_bytes()).hexdigest() == admission['independent_review_sha256']
    assert not (HERE/'TRANSPORT.json').exists() and not (HERE/'DISPATCH_INTENT.json').exists()
    payload = dict(root_source_review_approved=True,
                   source_review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(), files=[])
    for name, expected in [('analyze.py',EXPECTED_SOURCE),('PROTOCOL.json',EXPECTED_PROTOCOL)]:
        data = (HERE/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        payload['files'].append(dict(name=name,bytes=len(data),sha256=expected,
                                     base64=base64.b64encode(data).decode()))
    ssh = ['ssh','-T','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
           '-o','UpdateHostKeys=no','-o','ConnectTimeout=20',
           'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    command = 'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)
    intent = dict(UTC=datetime.now(timezone.utc).isoformat(),
        ssh_destination='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222',
        source_review_sha256=payload['source_review_sha256'],approved_files=admission['approved_files'],
        remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
        no_automatic_redispatch=True,uncertain_transport_requires_readonly_recovery=True)
    with (HERE/'DISPATCH_INTENT.json').open('x') as stream:
        json.dump(intent,stream,indent=2);stream.write('\n')
    try:
        child = subprocess.run([*ssh,command],input=json.dumps(payload),capture_output=True,text=True,timeout=180)
    except (Exception,KeyboardInterrupt) as error:
        failed = dict(UTC=datetime.now(timezone.utc).isoformat(),transport_exception=type(error).__name__,
            remote_completion_unknown=True,no_automatic_redispatch=True,
            dispatch_intent_sha256=hashlib.sha256((HERE/'DISPATCH_INTENT.json').read_bytes()).hexdigest(),
            remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),raw_predictions_transferred=False)
        with (HERE/'TRANSPORT.json').open('x') as stream:
            json.dump(failed,stream,indent=2);stream.write('\n')
        raise
    transport = dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=child.returncode,stderr=child.stderr,
        remote_source_sha256=hashlib.sha256(REMOTE.encode()).hexdigest(),
        client_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stdout_sha256=hashlib.sha256(child.stdout.encode()).hexdigest(),raw_predictions_transferred=False)
    if child.returncode:
        transport['stdout'] = child.stdout
    with (HERE/'TRANSPORT.json').open('x') as stream:
        json.dump(transport,stream,indent=2);stream.write('\n')
    assert child.returncode == 0, child.stderr
    received = json.loads(child.stdout)
    for row in received['files']:
        assert row['name'] in ('START.json','RESULTS.json','EXECUTION_RECEIPT.json')
        raw = base64.b64decode(row['base64'],validate=True)
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        with (HERE/row['name']).open('xb') as stream:
            stream.write(raw)
    print(json.dumps(received['receipt']))

if __name__ == '__main__':
    main()
