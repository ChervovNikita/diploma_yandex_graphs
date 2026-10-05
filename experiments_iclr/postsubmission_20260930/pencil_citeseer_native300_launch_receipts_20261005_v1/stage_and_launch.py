#!/usr/bin/env python3
"""Stage the exact root-reviewed packet and launch its single fresh cohort."""
from datetime import datetime, timezone
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PHASE_LOCAL = HERE.parent
SOURCE = PHASE_LOCAL / 'pencil_citeseer_native300_train_valid_preparation_20261005_v2'
ROOT_RELEASE = PHASE_LOCAL / 'pencil_citeseer_native300_root_release_20261005_v1'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(SOURCE / 'MANIFEST.json') == 'af9feab8d6c84b527de5ea203ce6a0b97e832c0e1246f652eddf0fd34ee8f40b'
assert sha(ROOT_RELEASE / 'ROOT_RELEASE.json') == '170470763283069a4d19bfe7c93373f3317e6971879879a57fd062dd094fe7c7'
manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
for row in manifest['files']:
    assert sha(SOURCE / row['path']) == row['sha256']
payload_files = []
for folder, names in ((SOURCE, [row['path'] for row in manifest['files']] + ['MANIFEST.json', 'SEAL.json']),
                      (ROOT_RELEASE, ['ROOT_RELEASE.json', 'REVIEW.md', 'RECEIPT.json'])):
    for name in names:
        path = folder / name
        data = path.read_bytes()
        payload_files.append(dict(phase_relative=str(path.relative_to(PHASE_LOCAL)),
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                base64=base64.b64encode(data).decode()))
payload = dict(files=payload_files,
               source_manifest_sha256=sha(SOURCE / 'MANIFEST.json'),
               release_sha256=sha(ROOT_RELEASE / 'ROOT_RELEASE.json'))
(HERE / 'STAGING_PAYLOAD.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
remote = r'''
from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json,os,socket,subprocess,sys,time,traceback
REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
SOURCE=PHASE/'pencil_citeseer_native300_train_valid_preparation_20261005_v2'
RELEASE=PHASE/'pencil_citeseer_native300_root_release_20261005_v1'
RECEIPTS=PHASE/'pencil_citeseer_native300_launch_receipts_20261005_v1'
OUTPUT=PHASE/'pencil_citeseer_native300_train_valid_execution_20261005_v1'
GPU_UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload=PAYLOAD_LITERAL
receipt=dict(UTC=datetime.now(timezone.utc).isoformat(),expected_login='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru:2222',
 host=socket.gethostname(),cwd=str(Path.cwd()),status='GATING_AND_STAGING',files=[],other_jobs_signalled=False,
 TEST_access=False,scientific_outcomes_opened=False,automatic_retry=False,cohort_launch_count=0)
assert Path.cwd().resolve()==REPO and socket.gethostname()=='anogena-2-0'
assert not OUTPUT.exists() and not RECEIPTS.exists()
RECEIPTS.mkdir()
try:
 plan_bytes=next(base64.b64decode(row['base64']) for row in payload['files'] if row['phase_relative'].endswith('/PLAN.json'))
 plan=json.loads(plan_bytes)
 memory_info={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  key,value=line.split(':',1);memory_info[key]=int(value.strip().split()[0])*1024
 query=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=30)
 rows=[line.split(',') for line in query.stdout.strip().splitlines()]
 assert len(rows)==1 and rows[0][0].strip()==GPU_UUID
 free=int(rows[0][1].strip())*1024**2
 receipt['fresh_preflight_gate']=dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_UUID=GPU_UUID,
  GPU_free_bytes=free,minimum_GPU_free_bytes=plan['dispatch_eligibility']['minimum_free_GPU_bytes'],
  CPU_available_bytes=memory_info['MemAvailable'],minimum_CPU_available_bytes=plan['dispatch_eligibility']['minimum_CPU_available_bytes'],
  utilization_percent=int(rows[0][3].strip()))
 assert free>=plan['dispatch_eligibility']['minimum_free_GPU_bytes'] and memory_info['MemAvailable']>=plan['dispatch_eligibility']['minimum_CPU_available_bytes']
 for row in payload['files']:
  relative=Path(row['phase_relative']);assert not relative.is_absolute() and '..' not in relative.parts
  target=PHASE/relative;assert target.resolve().is_relative_to(PHASE)
  data=base64.b64decode(row['base64']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  if target.exists():
   assert target.is_file() and sha(target)==row['sha256'];action='REUSED_EXACT_EXISTING_BYTES'
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(data)
   action='STAGED_EXACT_REVIEWED_BYTES'
  receipt['files'].append(dict(path=str(target),sha256=row['sha256'],bytes=row['bytes'],action=action))
 assert sha(SOURCE/'MANIFEST.json')==payload['source_manifest_sha256']=='af9feab8d6c84b527de5ea203ce6a0b97e832c0e1246f652eddf0fd34ee8f40b'
 assert sha(RELEASE/'ROOT_RELEASE.json')==payload['release_sha256']=='170470763283069a4d19bfe7c93373f3317e6971879879a57fd062dd094fe7c7'
 release=json.loads((RELEASE/'ROOT_RELEASE.json').read_text())
 assert release['root_source_review_approved'] is True and release['scientific_comparator_authorized'] is True and release['TEST_authorized'] is False
 assert release['execution_output_phase_relative']==str(OUTPUT.relative_to(PHASE)) and not OUTPUT.exists()
 env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1';env.pop('PYTHONPATH',None)
 command=[sys.executable,str(SOURCE/'supervisor.py'),'--release',str(RELEASE/'ROOT_RELEASE.json'),'--release-sha256',payload['release_sha256'],'--output',str(OUTPUT)]
 log=(RECEIPTS/'supervisor_stdout_stderr.log').open('xb')
 child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 log.close()
 raw=(Path('/proc')/str(child.pid)/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
 receipt.update(status='DETACHED_COHORT_SUPERVISOR_LAUNCHED',UTC=datetime.now(timezone.utc).isoformat(),command=command,
  supervisor_pid=child.pid,supervisor_pgid=int(fields[2]),supervisor_sid=int(fields[3]),supervisor_start_ticks=int(fields[19]),
  cohort_launch_count=1,release_sha256=payload['release_sha256'],source_manifest_sha256=payload['source_manifest_sha256'])
 (RECEIPTS/'LAUNCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps(receipt,sort_keys=True))
except BaseException as error:
 receipt.update(status='FAIL_GATE_OR_STAGING_PRESERVED_NO_RETRY',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
 (RECEIPTS/'LAUNCH_FAILURE.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps(receipt,sort_keys=True))
 raise
'''
remote = remote.replace('PAYLOAD_LITERAL', repr(payload))
ast.parse(remote)
(HERE / 'REMOTE_STAGE_AND_LAUNCH_SOURCE.py').write_text(remote)
(HERE / 'LOCAL_START.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),
        remote_source_sha256=hashlib.sha256(remote.encode()).hexdigest(),
        reviewed_manifest_sha256=payload['source_manifest_sha256'],release_sha256=payload['release_sha256']), indent=2) + '\n')
command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20',
        '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222',
        'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
        'cd /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs && '
        'exec /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/'
        'experiments_iclr/postsubmission_20260930/native_ncn_runtime_20261005_v1/.venv/bin/python -']
result = subprocess.run(command, input=remote, capture_output=True, text=True, timeout=90)
transport = dict(UTC=datetime.now(timezone.utc).isoformat(),command=command,
                 exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,automatic_retry=False)
(HERE / 'LAUNCH_TRANSPORT.json').write_text(json.dumps(transport, indent=2) + '\n')
if result.stdout:
    receipt = json.loads(result.stdout)
    (HERE / ('LAUNCH_RECEIPT.json' if result.returncode == 0 else 'LAUNCH_FAILURE.json')).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, sort_keys=True))
else:
    print(json.dumps(transport, sort_keys=True))
if result.returncode:
    sys.exit(result.returncode)
