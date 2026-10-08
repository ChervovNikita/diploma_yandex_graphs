"""Disabled normal-host bounded qualification/evaluation process owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SEEDS=(6101,6203,6307)


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):value.update(block)
    return value.hexdigest()


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def bound(row):
    path=(PHASE/row['path']).resolve(strict=True)
    require(path.is_relative_to(PHASE) and sha(path)==row['sha256'],'Exact project binding')
    return path


def identity(pid):
    try:
        value=Path('/proc',str(pid),'stat').read_text();parts=value[value.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(parts[19]),group=int(parts[2]),session=int(parts[3]))
    except FileNotFoundError:return None


def validate(release_path,release_sha256,authorized):
    if authorized is not True:raise PermissionError('Disabled wrapper; exact root release required')
    release_path=Path(release_path).resolve(strict=True)
    require(release_path.is_relative_to(PHASE) and sha(release_path)==release_sha256,'Bound root release')
    cfg=read(release_path)
    require(cfg['enabled'] is True and cfg['source_review_approved'] is True
        and cfg['all3_native_and_all12_correction_records_complete'] is True
        and cfg['root_comparative_opening_authorized'] is True
        and cfg['operation'] in ('qualify','evaluate'),'Complete-family root opening before state access')
    require(sha(HERE/'MANIFEST.json')==cfg['wrapper_manifest_sha256'],'Exact adopted wrapper packet')
    for row in read(HERE/'MANIFEST.json')['files']:
        path=bound(row);require(path.is_relative_to(HERE),'Wrapper payload scope')
    pins=read(HERE/'SOURCE_BINDINGS.json')
    for row in pins['source_files']:bound(row)
    owner=read(bound(pins['screen_owner_bindings']));runtime=owner['runtime']
    require(PHASE==Path(runtime['phase']) and Path.cwd()==Path(runtime['repository'])
        and socket.gethostname()==runtime['hostname'],'Original one-GPU allocation route')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
        text=True,timeout=5).splitlines()==[runtime['GPU_uuid']],'Sole authorized physical GPU')
    bound(cfg['whole_family_opening'])  # Hash only; never interpret candidate outcomes.
    require(set(cfg['native_state_paths'])==set(cfg['native_state_sha256'])=={str(s) for s in SEEDS},
        'All three authentic own-best file bindings')
    require(type(cfg['active_seconds']) is int and 1<=cfg['active_seconds']<=
        (3600 if cfg['operation']=='qualify' else 21600),'Finite prospective worker budget')
    output=(PHASE/cfg['output_relative']).resolve()
    require(output.is_relative_to(PHASE),'Owned output inside repository phase')
    return cfg,pins,owner,runtime,output


def stop(child,saved,actions):
    if child is None or child.poll() is not None:return
    require(saved is not None and identity(child.pid)==saved
        and saved['group']==saved['session']==child.pid,'Stop only original detached owned group')
    os.killpg(child.pid,signal.SIGTERM);actions.append('SIGTERM_owned_group')
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        require(identity(child.pid)==saved,'Original owned group before SIGKILL')
        os.killpg(child.pid,signal.SIGKILL);actions.append('SIGKILL_owned_group');child.wait(timeout=5)


def supervise(release_path,release_sha256,authorized):
    validation_started=time.monotonic()
    cfg,pins,owner,runtime,output=validate(release_path,release_sha256,authorized)
    child=saved=None;actions=[];started=time.monotonic();terminal=dict(complete=False,
        validation_seconds=started-validation_started)
    def interrupted(number,frame):raise RuntimeError('Owned supervisor interrupted '+str(number))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
            CUDA_VISIBLE_DEVICES=runtime['GPU_uuid'],PYTHONPATH=os.pathsep.join(runtime['PYTHONPATH']))
        environment.pop('PYTHONHOME',None)
        argv=[runtime['python'],'-B',str(HERE/'worker.py'),'--release',str(Path(release_path).resolve()),
            '--release-sha256',release_sha256,'--authorized']
        with (output/'WORKER.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=runtime['repository'],env=environment,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        saved=identity(child.pid)
        require(saved is not None or child.poll() is not None,'Owned worker identity or terminal launch')
        if saved:require(saved['group']==saved['session']==child.pid,'Detached owned worker group')
        write(output/'WORKER_STARTED.json',dict(worker=saved,argv=argv,active_seconds=cfg['active_seconds'],
            release_sha256=release_sha256,automatic_retry=False))
        code=child.wait(timeout=cfg['active_seconds'])
        terminal.update(complete=code==0,worker_exit_code=code)
    except BaseException as error:
        terminal['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        try:stop(child,saved,actions)
        except BaseException as error:terminal['cleanup_error']=dict(type=type(error).__name__,message=str(error))
        terminal.update(worker=saved,cleanup_actions=actions,seconds=time.monotonic()-started,
            release_sha256=release_sha256,operation=cfg['operation'],automatic_retry=False,
            other_process_cancellation=False,filesystem_namespace_changes=False)
        write(output/'TERMINAL.json',terminal)
    return 0 if terminal['complete'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',required=True,type=Path);parser.add_argument('--release-sha256',required=True)
    parser.add_argument('--authorized',action='store_true');parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    if args.supervise:sys.exit(supervise(args.release,args.release_sha256,args.authorized))
    launch_started=time.monotonic()
    cfg,pins,owner,runtime,output=validate(args.release,args.release_sha256,args.authorized)
    require(not output.exists(),'Fresh once-only wrapper output; no retry/resume')
    output.mkdir(parents=True,exist_ok=False)
    argv=['/usr/bin/python3','-I','-S','-B',str(HERE/'owned.py'),'--supervise','--release',
        str(args.release.resolve()),'--release-sha256',args.release_sha256,'--authorized']
    try:
        with (output/'OWNER.log').open('xb') as log:
            parent=subprocess.Popen(argv,cwd=runtime['repository'],stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(output/'LAUNCH.json',dict(supervisor=identity(parent.pid),argv=argv,
            release_sha256=args.release_sha256,operation=cfg['operation'],automatic_retry=False,
            root_detached=True,worker_budget_seconds=cfg['active_seconds'],
            launch_preparation_seconds=time.monotonic()-launch_started))
    except BaseException as error:
        write(output/'LAUNCH_FAILURE.json',dict(type=type(error).__name__,message=str(error),automatic_retry=False))
        raise
    print(str(output/'LAUNCH.json'))


if __name__=='__main__':main()
