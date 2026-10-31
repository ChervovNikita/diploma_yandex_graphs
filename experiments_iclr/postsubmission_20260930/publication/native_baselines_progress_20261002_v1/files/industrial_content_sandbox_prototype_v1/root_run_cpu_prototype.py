"""UNEXECUTED root-invocable fixed-route CPU content-sandbox diagnostic.

Author prepares source only. Root reviews and invokes; fresh synthetic fixtures
only, bounded public GPU-UUID inventory preflight, 30-second remote child cap,
60-second transport cap, no automatic retry. No GPU scientific computation.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
REMOTE_REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_PHASE = REMOTE_REPO + '/experiments_iclr/postsubmission_20260930'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]
WORKER_SHA256 = '2d0c1c18ae17432c20ded3cc509a90db5c5d3de9471b43f6a766e16e6b1973c4'

OUTER = r'''
import datetime,hashlib,json,os,pathlib,re,signal,subprocess,sys,time
phase=pathlib.Path(sys.argv[1]); run_id=sys.argv[2]; source=sys.argv[3]; expected=sys.argv[4]
if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}',run_id): raise ValueError('Simple fresh run identity required')
if hashlib.sha256(source.encode()).hexdigest()!=expected: raise ValueError('Transmitted worker source pin mismatch')
if not phase.is_absolute() or phase.resolve(strict=True)!=phase: raise ValueError('Actual fixed project phase required')
inventory_argv=['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits']
inventory_started=time.monotonic()
inventory=subprocess.run(inventory_argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                         timeout=10,check=False,env={'PATH':'/usr/bin'})
inventory_uuids=[line.strip() for line in inventory.stdout.splitlines() if line.strip()]
authorized_uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
if inventory.returncode!=0 or inventory_uuids!=[authorized_uuid]:
    raise RuntimeError('Authorized single-GPU inventory mismatch before fixture writes: '+
                       json.dumps({'exit_code':inventory.returncode,'uuids':inventory_uuids,'stderr':inventory.stderr}))
inventory_seconds=time.monotonic()-inventory_started
root=phase/'industrial_content_sandbox_prototype_v1/root_runs'/run_id
cursor=phase
for part in root.relative_to(phase).parts:
    cursor/=part
    if cursor.is_symlink(): raise ValueError('Symlink forbidden in diagnostic destination')
if root.exists(): raise ValueError('Fresh single-use run identity required')
root.parent.mkdir(parents=True,exist_ok=True); root.mkdir(mode=0o700)
for name in ('public_inputs','excluded_fixture','outputs','cache','tmp','trusted_source'):
    (root/name).mkdir(mode=0o700)
(root/'public_inputs/public.txt').write_bytes(b'public synthetic fixture\n')
(root/'excluded_fixture/blocked.txt').write_bytes(b'excluded synthetic fixture\n')
worker=root/'trusted_source/cpu_content_diagnostic.py'
worker.write_text(source)
host={name:os.readlink('/proc/self/ns/'+name) for name in ('user','mnt','pid','net','ipc','uts')}
mountinfo=pathlib.Path('/proc/self/mountinfo').read_text()
argv=['/usr/bin/unshare','--user','--map-root-user','--mount','--propagation','unchanged',
      '--pid','--fork','--net','--ipc','--uts','--kill-child',
      '/usr/bin/python3','-I','-S','-B',str(worker),str(root),json.dumps(host),expected]
environment={'PATH':'/usr/bin','HOME':str(root/'tmp'),'TMPDIR':str(root/'tmp'),
             'XDG_CACHE_HOME':str(root/'cache'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'}
receipt={'schema':'root-cpu-content-sandbox-prototype-v1','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'run_root':str(root),'worker_sha256':expected,'argv':argv,'host_namespaces':host,
         'host_mountinfo_before':mountinfo,'CPU_worker_only':True,'GPU_compute_action':False,
         'inventory_preflight':{'argv':inventory_argv,'uuids':inventory_uuids,'authorized_uuid':authorized_uuid,
                                'exit_code':inventory.returncode,'seconds':inventory_seconds,
                                'stderr':inventory.stderr,'before_fixture_writes':True,'timeout_seconds':10},
         'real_data_archive_label_model_access':False,'scientific_execution':False,
         'automatic_retry':False,'remote_wall_cap_seconds':30}
started=time.monotonic()
child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       text=True,close_fds=True,start_new_session=True,env=environment,cwd=root)
try:
    stdout,stderr=child.communicate(timeout=30); receipt['timed_out']=False
except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL)
    stdout,stderr=child.communicate(); receipt['timed_out']=True
receipt.update(exit_code=child.returncode,seconds=time.monotonic()-started,stdout=stdout,stderr=stderr)
try: receipt['worker_result']=json.loads(stdout)
except ValueError: receipt['worker_result']=None
receipt['host_namespaces_after']={name:os.readlink('/proc/self/ns/'+name) for name in host}
receipt['host_mountinfo_after']=pathlib.Path('/proc/self/mountinfo').read_text()
receipt['host_namespaces_and_mountinfo_unchanged']=(receipt['host_namespaces_after']==host and receipt['host_mountinfo_after']==mountinfo)
receipt['worker_source_unchanged']=(hashlib.sha256(worker.read_bytes()).hexdigest()==expected)
result=receipt['worker_result']
receipt['status']=('PASSED_CPU_DIAGNOSTIC_ONLY' if child.returncode==0 and not receipt['timed_out']
                   and receipt['host_namespaces_and_mountinfo_unchanged'] and receipt['worker_source_unchanged']
                   and isinstance(result,dict) and result.get('status')=='PASSED_CPU_FIXTURE_ONLY'
                   and result.get('source_sha256')==expected else 'REFUSED_OR_FAILED_DIAGNOSTIC')
with (root/'CONTENT_SANDBOX_CAPABILITY.json').open('x') as stream:
    json.dump(receipt,stream,indent=2,allow_nan=False); stream.write('\n')
print(json.dumps(receipt,sort_keys=True,allow_nan=False))
raise SystemExit(0 if receipt['status']=='PASSED_CPU_DIAGNOSTIC_ONLY' else 1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='Fresh synthetic diagnostic identity; defaults to UTC/UUID.')
    args = parser.parse_args()
    run_id = args.run_id or ('cpu_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12])
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', run_id):
        raise ValueError('Simple fresh run identity required')
    source = (HERE / 'cpu_content_diagnostic.py').read_text()
    if hashlib.sha256(source.encode()).hexdigest() != WORKER_SHA256:
        raise RuntimeError('Pinned worker source changed; review and reseal before execution')
    output = HERE / 'root_runs' / run_id
    if output.parent.is_symlink():
        raise RuntimeError('Local root_runs must be an actual repo directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', OUTER,
                          REMOTE_PHASE, run_id, source, WORKER_SHA256])
    launch = {'schema': 'local-root-cpu-content-sandbox-launch-v1',
              'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'run_id': run_id, 'ssh_destination': LOGIN, 'worker_sha256': WORKER_SHA256,
              'CPU_worker_only': True, 'GPU_compute_action': False, 'scientific_execution': False,
              'GPU_UUID_inventory_preflight': True,
              'authorized_GPU_UUID': 'GPU-44039938-fd82-41d2-fefd-de71514e2fac',
              'inventory_timeout_seconds': 10,
              'real_data_archive_label_model_access': False, 'automatic_retry': False,
              'remote_wall_cap_seconds': 30, 'transport_timeout_seconds': 60}
    started = time.monotonic()
    proc = None
    try:
        proc = subprocess.run([*SSH, command], stdin=subprocess.DEVNULL, text=True,
                              capture_output=True, timeout=60, check=False)
        (output / 'stdout.txt').write_text(proc.stdout)
        (output / 'stderr.txt').write_text(proc.stderr)
        try:
            receipt = json.loads(proc.stdout)
        except ValueError:
            receipt = None
        if receipt is not None:
            with (output / 'CONTENT_SANDBOX_CAPABILITY.json').open('x') as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
                stream.write('\n')
        launch.update(exit_code=proc.returncode,
                      status='COMPLETED_CPU_DIAGNOSTIC_ONLY' if proc.returncode == 0 else 'REFUSED_OR_FAILED_DIAGNOSTIC')
    except BaseException as error:
        launch.update(status='LOCAL_TRANSPORT_FAILED', error_type=type(error).__name__, error=str(error))
        if isinstance(error, subprocess.TimeoutExpired):
            for name, value in [('stdout.txt', error.stdout), ('stderr.txt', error.stderr)]:
                if value is not None:
                    (output / name).write_bytes(value if isinstance(value, bytes) else value.encode())
    launch['seconds'] = time.monotonic() - started
    with (output / 'ROOT_LAUNCH.json').open('x') as stream:
        json.dump(launch, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'receipt': str(output / 'ROOT_LAUNCH.json'), 'status': launch['status']}, sort_keys=True))
    return 0 if proc is not None and proc.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
