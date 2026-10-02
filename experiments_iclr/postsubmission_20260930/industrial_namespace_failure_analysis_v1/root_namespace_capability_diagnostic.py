"""Root-invocable bounded namespace diagnostic; not executed by its author.

Standard library only. Fixed authorized SSH route. No data, label, model or GPU
access. All mount mutations occur in verified new user/mount/PID namespaces,
under a fresh repo-owned directory, after rejecting shared hosting mounts.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
REMOTE_REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_PHASE = REMOTE_REPO + '/experiments_iclr/postsubmission_20260930'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', LOGIN]

WORKER = r'''
import ctypes, json, os, pathlib, re, sys, time
root = pathlib.Path(sys.argv[1]); image = pathlib.Path(sys.argv[2])
expected_host = json.loads(sys.argv[3])
result = {'schema':'industrial-private-subtree-capability-worker-v1',
          'scientific_execution':False, 'data_access':False, 'label_access':False,
          'GPU_access':False, 'automatic_retry':False, 'tests':[],
          'mount_mutations_attempted':0}

def namespaces():
    return {name:os.readlink('/proc/self/ns/'+name) for name in ('user','mnt','pid','net','ipc','uts')}

def unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda match:chr(int(match.group(1),8)), value)

def mounts():
    rows=[]
    for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines():
        before,after=line.split(' - ',1); fields=before.split(); suffix=after.split()
        optional=fields[6:]
        rows.append({'mount_id':int(fields[0]), 'parent_id':int(fields[1]),
                     'root':unescape(fields[3]), 'target':unescape(fields[4]),
                     'options':fields[5].split(','), 'optional_fields':optional,
                     'shared':[value for value in optional if value.startswith('shared:')],
                     'master':[value for value in optional if value.startswith('master:')],
                     'filesystem':suffix[0]})
    return rows

def hosting(path, exact=False):
    target=str(path)
    rows=[row for row in mounts() if target==row['target'] or
          (not exact and (row['target']=='/' or target.startswith(row['target'].rstrip('/')+'/')))]
    if not rows: raise RuntimeError('No hosting mount for '+target)
    longest=max(len(row['target']) for row in rows)
    chosen=[row for row in rows if len(row['target'])==longest]
    if len(chosen)!=1: raise RuntimeError('Ambiguous stacked hosting mounts for '+target)
    return chosen[0]

def safe_destination(path):
    row=hosting(path)
    if row['shared']: raise RuntimeError('Shared destination hosting mount; no mutation allowed: '+str(path))
    return row

libc=ctypes.CDLL(None,use_errno=True)
libc.mount.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_char_p]
libc.mount.restype=ctypes.c_int
MS_RDONLY=1; MS_REMOUNT=32; MS_BIND=4096; MS_REC=16384; MS_PRIVATE=262144

def mount_test(name, source, target, filesystem, flags, data=None):
    before=safe_destination(target)
    ctypes.set_errno(0); started=time.monotonic()
    encoded=lambda value:None if value is None else str(value).encode()
    result['mount_mutations_attempted']+=1
    returned=libc.mount(encoded(source),encoded(target),encoded(filesystem),flags,encoded(data))
    err=ctypes.get_errno() if returned else 0
    row={'name':name, 'source':None if source is None else str(source), 'target':str(target),
         'filesystem':filesystem, 'flags':flags, 'data':data,
         'hosting_mount_before':before, 'returned':returned, 'errno':err,
         'error':os.strerror(err) if err else None,
         'seconds':time.monotonic()-started, 'passed':returned==0}
    if returned==0: row['mount_after']=hosting(target,exact=True)
    result['tests'].append(row)
    return returned==0

try:
    observed=namespaces()
    result.update(host_namespaces=expected_host, worker_namespaces=observed,
                  worker_pid=os.getpid(), worker_uid=os.getuid())
    if os.getuid()!=0 or os.getpid()!=1 or any(observed[name]==expected_host[name] for name in expected_host):
        raise RuntimeError('Distinct user/mount/PID/net/IPC/UTS namespace and mapped-root PID1 required')
    if not root.is_absolute() or not image.is_absolute() or '..' in root.parts or '..' in image.parts:
        raise RuntimeError('Absolute confined paths required')
    for path in (root,image):
        if path.is_symlink() or not path.is_dir(): raise RuntimeError('Actual directory required: '+str(path))
    if root==image or root in image.parents or image in root.parents:
        raise RuntimeError('Diagnostic and runtime-image trees must be disjoint')
    targets=[root/'tmpfs',root/'runtime_bind',root/'proc']
    result['inherited_hosting_mounts_before_any_mutation']={str(path):hosting(path) for path in targets}
    result['runtime_source_hosting_mount_before_any_mutation']=hosting(image)
    # Preflight all targets before the first mount. Slave/master is allowed;
    # shared propagation is refused. Check again immediately before each mount.
    if any(row['shared'] for row in result['inherited_hosting_mounts_before_any_mutation'].values()):
        raise RuntimeError('Shared inherited destination hosting mount; zero mutations required')
    if result['runtime_source_hosting_mount_before_any_mutation']['shared']:
        raise RuntimeError('Shared runtime source could give its bind shared propagation; refusing it')
    for target in targets:
        if target.is_symlink() or not target.is_dir() or any(target.iterdir()):
            raise RuntimeError('Fresh empty actual diagnostic mount target required: '+str(target))
    mount_test('minimal_tmpfs', 'tmpfs', root/'tmpfs', 'tmpfs', 0, 'size=1048576,mode=0700')
    bound=mount_test('runtime_image_bind', image, root/'runtime_bind', None, MS_BIND)
    if bound:
        # This operates on the new bind, never inherited `/`. It tests the
        # precise subtree alternative to v2's rejected root propagation change.
        mount_test('private_runtime_subtree', None, root/'runtime_bind', None, MS_PRIVATE|MS_REC)
        readonly=mount_test('runtime_image_readonly_remount', None, root/'runtime_bind', None,
                            MS_BIND|MS_REMOUNT|MS_RDONLY)
        if readonly:
            row=hosting(root/'runtime_bind',exact=True)
            result['readonly_bind_observed']='ro' in row['options']
            result['private_bind_observed']=not row['shared'] and not row['master']
            if not result['readonly_bind_observed']:
                raise RuntimeError('Readonly remount returned success without observed ro')
        # No writes or directory traversal inside runtime image are attempted.
    if mount_test('new_pid_namespace_proc', 'proc', root/'proc', 'proc', 0):
        result['new_proc_self_pid']=os.readlink(root/'proc/self')
        if result['new_proc_self_pid']!='1': raise RuntimeError('New proc does not expose worker as PID1')
    result['mountinfo_after']=mounts()
    required={'minimal_tmpfs','runtime_image_bind','runtime_image_readonly_remount','new_pid_namespace_proc'}
    result['required_mount_operations_passed']=(required=={row['name'] for row in result['tests'] if row['name'] in required}
        and all(row['passed'] for row in result['tests'] if row['name'] in required))
    result['all_mount_calls_passed']=len(result['tests'])==5 and all(row['passed'] for row in result['tests'])
    passed=(result['required_mount_operations_passed'] and result.get('readonly_bind_observed') is True
            and result.get('private_bind_observed') is True)
    result['status']=('PASSED_DIAGNOSTIC_ONLY' if passed and result['all_mount_calls_passed'] else
                      'PASSED_REQUIRED_MOUNTS_ALREADY_PRIVATE' if passed else 'MOUNT_CAPABILITY_FAILURE')
except BaseException as error:
    result['status']='REFUSED_OR_FAILED_DIAGNOSTIC'
    result['error_type']=type(error).__name__; result['error']=str(error)
print(json.dumps(result,sort_keys=True,allow_nan=False))
raise SystemExit(0 if result['status'] in ('PASSED_DIAGNOSTIC_ONLY','PASSED_REQUIRED_MOUNTS_ALREADY_PRIVATE') else 1)
'''

OUTER = r'''
import datetime,json,os,pathlib,shutil,signal,subprocess,sys,time
phase=pathlib.Path(sys.argv[1]); run_id=sys.argv[2]; worker=sys.argv[3]
root=phase/'industrial_namespace_failure_analysis_v1/root_capability_runs'/run_id
image=phase/'industrial_runtime_image_root_v1/runtime_image_v1'
def checked(path):
    if not path.is_relative_to(phase) or '..' in path.parts: raise ValueError('Confined project path required')
    cursor=phase
    for part in path.relative_to(phase).parts:
        cursor/=part
        if cursor.is_symlink(): raise ValueError('Symlink forbidden: '+str(cursor))
checked(root); checked(image)
if root.exists(): raise ValueError('Fresh single-use diagnostic directory required')
if not image.is_dir(): raise ValueError('Existing reviewed runtime image directory required')
root.parent.mkdir(parents=True,exist_ok=True)
root.mkdir(mode=0o700)
for name in ('tmpfs','runtime_bind','proc'): (root/name).mkdir(mode=0o700)
host={name:os.readlink('/proc/self/ns/'+name) for name in ('user','mnt','pid','net','ipc','uts')}
host_mountinfo=pathlib.Path('/proc/self/mountinfo').read_text()
unshare=shutil.which('unshare')
if not unshare: raise RuntimeError('unshare unavailable')
argv=[unshare,'--user','--map-root-user','--mount','--propagation','unchanged',
      '--pid','--fork','--net','--ipc','--uts','--kill-child',sys.executable,
      '-I','-S','-B','-c',worker,str(root),str(image),json.dumps(host)]
receipt={'schema':'root-bounded-namespace-capability-diagnostic-v1',
         'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'run_root':str(root),'runtime_image':str(image), 'argv':argv,
         'host_namespaces':host,'host_mountinfo_before':host_mountinfo,
         'scientific_execution':False,'data_access':False,'label_access':False,
         'GPU_access':False,'automatic_retry':False,'wall_cap_seconds':30}
started=time.monotonic()
child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                       text=True,close_fds=True,start_new_session=True)
try:
    stdout,stderr=child.communicate(timeout=30); receipt['timed_out']=False
except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL)
    stdout,stderr=child.communicate(); receipt['timed_out']=True
receipt.update(exit_code=child.returncode,seconds=time.monotonic()-started,stdout=stdout,stderr=stderr)
try: receipt['worker_result']=json.loads(stdout)
except ValueError: receipt['worker_result']=None
receipt['host_mountinfo_after']=pathlib.Path('/proc/self/mountinfo').read_text()
receipt['host_namespaces_after']={name:os.readlink('/proc/self/ns/'+name) for name in host}
receipt['host_mount_namespace_unchanged']=(receipt['host_namespaces_after']==host and
                                         receipt['host_mountinfo_after']==host_mountinfo)
receipt['status']=('PASSED_DIAGNOSTIC_ONLY' if child.returncode==0 and
                   not receipt['timed_out'] and receipt['host_mount_namespace_unchanged'] else
                   'REFUSED_OR_FAILED_DIAGNOSTIC')
with (root/'NAMESPACE_CAPABILITY.json').open('x') as stream:
    json.dump(receipt,stream,indent=2,allow_nan=False); stream.write('\n')
print(json.dumps(receipt,sort_keys=True,allow_nan=False))
raise SystemExit(0 if receipt['status']=='PASSED_DIAGNOSTIC_ONLY' else 1)
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', help='Optional fresh metadata identity; defaults to a UTC/UUID name.')
    args = parser.parse_args()
    run_id = args.run_id or ('capability_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12])
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}', run_id):
        raise ValueError('Simple fresh run identity required')
    output = HERE/'root_capability_runs'/run_id
    output.parent.mkdir(parents=True,exist_ok=True)
    output.mkdir(mode=0o700)
    command = shlex.join(['/usr/bin/python3','-I','-S','-B','-c',OUTER,REMOTE_PHASE,run_id,WORKER])
    launch = {'schema':'local-root-namespace-capability-launch-v1',
              'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'run_id':run_id,'ssh_destination':LOGIN,'automatic_retry':False,
              'scientific_execution':False,'data_access':False,'label_access':False,'GPU_access':False,
              'remote_wall_cap_seconds':30,'local_transport_timeout_seconds':60}
    started = time.monotonic()
    try:
        proc = subprocess.run([*SSH,command], stdin=subprocess.DEVNULL,text=True,capture_output=True,
                              timeout=60,check=False)
        launch.update(exit_code=proc.returncode,seconds=time.monotonic()-started)
        (output/'stdout.txt').write_text(proc.stdout)
        (output/'stderr.txt').write_text(proc.stderr)
        try:
            receipt = json.loads(proc.stdout)
        except ValueError:
            receipt = None
        if receipt is not None:
            with (output/'NAMESPACE_CAPABILITY.json').open('x') as stream:
                json.dump(receipt,stream,indent=2,allow_nan=False); stream.write('\n')
        launch['status'] = 'COMPLETED_DIAGNOSTIC_ONLY' if proc.returncode==0 else 'REFUSED_OR_FAILED_DIAGNOSTIC'
    except BaseException as error:
        launch.update(status='LOCAL_TRANSPORT_FAILED',error_type=type(error).__name__,error=str(error),
                      seconds=time.monotonic()-started)
        proc = None
    with (output/'ROOT_LAUNCH.json').open('x') as stream:
        json.dump(launch,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({'receipt':str(output/'ROOT_LAUNCH.json'),'status':launch['status']},sort_keys=True))
    return 0 if proc is not None and proc.returncode==0 else 1

if __name__ == '__main__':
    raise SystemExit(main())
