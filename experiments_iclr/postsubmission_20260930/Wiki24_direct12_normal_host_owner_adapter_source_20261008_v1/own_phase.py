"""One normal-host stdlib owner for an exact reviewed direct12 phase."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
SCIENCE = PHASE / 'Wiki24_direct12_head_refit_source_20261008_v2'
SCIENCE_SHA = '41e4c5b689d85e9a6c5d382a15858c594bd7108899eac807d4d507e8c8b704eb'
PYTHON = PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PYTHONPATH = str(PHASE/'native_ncn_dependency_overlay_20261005_v1') + ':' + str(REPO/'.venv/lib/python3.11/site-packages')
STAGES = dict(collect=('collect_features.py',3590,10,3600),
              fit=('run_refits.py',46790,10,46800), read=('read_family.py',590,10,600))
CONDITIONS = ('BE_factor_refit','BE_full_refit','ordinary_full_refit','single_full_refit')
SEEDS = (6101,6203,6307)


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def inside(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Phase-relative path required')
    path = (PHASE/relative).resolve()
    require(path.is_relative_to(PHASE.resolve()), 'Path leaves phase')
    return path


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE.resolve())),sha256=sha(path),bytes=path.stat().st_size)


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path)==row['sha256'], 'Bound file changed')
    if 'bytes' in row: require(path.stat().st_size==row['bytes'], 'Bound size changed')
    return path


def seal(directory, expected):
    require(sha(directory/'MANIFEST.json')==expected, 'Exact reviewed source manifest')
    for row in read(directory/'MANIFEST.json')['files']:
        path=(directory/row['path']).resolve()
        require(path.is_relative_to(directory.resolve()) and sha(path)==row['sha256']
                and path.stat().st_size==row['bytes'], 'Source bytes changed')


def proc(pid):
    try:
        f=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),ppid=int(f[1]),state=f[0])
    except (FileNotFoundError,ProcessLookupError): return None


def identity(row): return {k:row[k] for k in ('pid','start_ticks','group','session')}


def live(handle):
    actual=proc(handle['pid'])
    return actual if actual is not None and identity(actual)==handle and actual['state']!='Z' else None


def phase_environment(stage):
    threads='2' if stage=='collect' else '1'
    return dict(os.environ,PYTHONPATH=PYTHONPATH,CUDA_VISIBLE_DEVICES=GPU if stage=='collect' else '',
        OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,NUMEXPR_NUM_THREADS=threads,
        PYTHONDONTWRITEBYTECODE='1')


def expected_argv(stage, release_path, digest):
    return [str(PYTHON),'-B',str(SCIENCE/STAGES[stage][0]),'--release',str(release_path),'--release-sha256',digest]


def worker_jobs(science_output):
    return {str(science_output/(c+'_'+str(s)+'_JOB.json')) for s in SEEDS for c in CONDITIONS}


def record_workers(child_handle, science_output, science_release, output, records, unknown):
    # The only PID inventory is the DIRECT children of this exact owned process.
    parent=live(child_handle)
    if parent is None: return
    path=Path('/proc',str(parent['pid']),'task',str(parent['pid']),'children')
    try: children=[int(v) for v in path.read_text().split()]
    except FileNotFoundError: return
    allowed=worker_jobs(science_output)
    for pid in children:
        actual=proc(pid)
        if actual is None or actual['ppid']!=parent['pid']: continue
        handle=identity(actual)
        if handle['pid']!=handle['group'] or handle['pid']!=handle['session']: continue
        key=(handle['pid'],handle['start_ticks'])
        try:
            argv=[part.decode() for part in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0') if part]
            require(len(argv)==7 and argv[:4]==[str(PYTHON),'-B',str((SCIENCE/'run_refits.py').resolve()),'--worker-job']
                    and argv[4] in allowed and argv[5]=='--worker-sha256', 'Not an exact reviewed fit-worker argv')
            job_path=Path(argv[4]); require(sha(job_path)==argv[6], 'Exact worker job bytes')
            job=read(job_path)
            require(job['owner']==child_handle and job['root_release']==science_release
                    and job['source_manifest_sha256']==SCIENCE_SHA, 'Exact worker parent/release/source')
            require(job['row']['bundle']+'_JOB.json'==job_path.name and job['row']['condition'] in CONDITIONS
                    and job['row']['seed'] in SEEDS, 'Fixed worker roster')
            if key not in records:
                record=dict(handle=handle,verified_parent=child_handle,worker_job=binding(job_path),argv=argv,
                            relationship='direct child observed while exact admitted fit-family parent was live')
                records[key]=record
                write(output/('WORKER_'+str(pid)+'_'+str(handle['start_ticks'])+'.json'),record)
        except (ValueError,KeyError,FileNotFoundError,ProcessLookupError,UnicodeDecodeError) as error:
            # Never signal an unknown child. A transient pre-exec child can be
            # qualified on a later pass; unresolved entries block a terminal claim.
            unknown[key]=dict(handle=handle,verified_parent=child_handle,error=str(error))
        else:
            unknown.pop(key,None)


def verified_worker(record):
    handle=record['handle']
    actual=live(handle)
    if actual is None: return None
    try:
        argv=[p.decode() for p in Path('/proc',str(handle['pid']),'cmdline').read_bytes().split(b'\0') if p]
        if argv!=record['argv']: return None
        bound(record['worker_job'])
    except (FileNotFoundError,ProcessLookupError,ValueError,UnicodeDecodeError): return None
    return actual


def signal_group(handle, signum):
    actual=live(handle)
    if actual is None: return False
    require(handle['pid']==handle['group']==handle['session'], 'Only exact owned session leader')
    try: os.killpg(handle['pid'],signum)
    except ProcessLookupError: return False
    return True


def signal_parent(handle, signum):
    if live(handle) is None: return False
    try: os.kill(handle['pid'],signum)
    except ProcessLookupError: return False
    return True


def cleanup(child, handle, stage, records, deadline, capture):
    result=[]
    # Prevent any new worker admission before capturing and terminating sessions.
    frozen=signal_parent(handle,signal.SIGSTOP) if stage=='fit' else False
    parent_actions=dict(handle=handle,SIGSTOP_sent=frozen,SIGTERM_sent=False,SIGCONT_sent=False,SIGKILL_sent=False)
    if frozen:
        until=min(deadline,time.monotonic()+.2)
        while time.monotonic()<until:
            actual=live(handle)
            if actual is None or actual['state'] in ('T','t'): break
            time.sleep(.005)
        capture()
    parent_actions['SIGTERM_sent']=signal_group(handle,signal.SIGTERM)
    for record in records.values():
        valid=verified_worker(record)
        sent=signal_group(record['handle'],signal.SIGTERM) if valid is not None else False
        result.append(dict(handle=record['handle'],SIGTERM_sent=sent,SIGKILL_sent=False))
    if frozen: parent_actions['SIGCONT_sent']=signal_parent(handle,signal.SIGCONT)  # Deliver pending TERM.
    try: child.wait(timeout=max(0,min(5,deadline-time.monotonic())))
    except subprocess.TimeoutExpired: pass
    if child.poll() is None: parent_actions['SIGTERM_sent']=signal_group(handle,signal.SIGTERM) or parent_actions['SIGTERM_sent']
    for record in records.values():
        if verified_worker(record) is not None:
            sent=signal_group(record['handle'],signal.SIGKILL)
            next(row for row in result if row['handle']==record['handle'])['SIGKILL_sent']=sent
    if child.poll() is None: parent_actions['SIGKILL_sent']=signal_group(handle,signal.SIGKILL)
    try:
        code=child.wait(timeout=max(0,deadline-time.monotonic()))
        reaped=True
    except subprocess.TimeoutExpired:
        code=child.poll(); reaped=code is not None
    for row in result:
        current=proc(row['handle']['pid'])
        row.update(original_worker_live=live(row['handle']) is not None,
                   original_worker_terminal_observed=current is None or identity(current)!=row['handle'] or current['state']=='Z',
                   descendant_wait_reap_claimed=False)
    return code,reaped,dict(phase_parent=parent_actions,workers=result)


def interrupted(signum, frame): raise TimeoutError('Owner interrupted by signal '+str(signum))


def main():
    began=time.monotonic()
    began_utc=datetime.now(timezone.utc).isoformat()
    parser=argparse.ArgumentParser()
    parser.add_argument('--activation',type=Path,required=True); parser.add_argument('--activation-sha256',required=True)
    args=parser.parse_args(); os.umask(0o077)
    require(socket.gethostname()=='anogena-2-0' and Path.cwd().resolve()==REPO.resolve(), 'Normal authorized host/repository')
    require(sha(args.activation)==args.activation_sha256, 'Exact root owner activation')
    cfg=read(args.activation)
    require(cfg.get('schema')=='Wiki24-direct12-owner-activation-v1', 'Fixed adapter activation')
    for key in ('enabled','root_execution_authorized','owner_source_review_approved','resource_admission_confirmed'):
        require(cfg.get(key) is True, 'Disabled pending root admission: '+key)
    require(cfg.get('TEST_access') is False and cfg.get('automatic_retry') is False, 'No TEST/retry')
    seal(ROOT,cfg['owner_manifest_sha256']); seal(SCIENCE,SCIENCE_SHA)
    review=read(bound(cfg['owner_review']))
    require(review.get('approved') is True and review.get('source_manifest_sha256')==cfg['owner_manifest_sha256'], 'Exact once-reviewed owner source')
    stage=cfg['stage']; require(stage in STAGES, 'Only collect/fit/read')
    release_path=bound(cfg['science_release']); release=read(release_path)
    require(release.get('stage')==stage and release.get('source_manifest_sha256')==SCIENCE_SHA
            and release.get('TEST_access') is False and release.get('automatic_retry') is False, 'Exact reviewed phase release')
    for key in ('enabled','root_execution_authorized','source_review_approved','finite_cost_charged','external_hard_bound_confirmed'):
        require(release.get(key) is True, 'Disabled scientific phase: '+key)
    source_review=read(bound(release['source_review']))
    require(source_review.get('approved') is True and source_review.get('source_manifest_sha256')==SCIENCE_SHA, 'Exact scientific source approval')
    argv=expected_argv(stage,release_path,sha(release_path))
    require(cfg['argv']==argv, 'Exact reviewed argv; no shell or extra options')
    output=inside(cfg['receipt_directory']); science_output=inside(release['output_directory'])
    require(not output.exists() and output.parent.is_dir() and not science_output.exists()
            and output!=science_output and not output.is_relative_to(science_output)
            and not science_output.is_relative_to(output), 'Fresh distinct owner/science outputs')
    output.mkdir(mode=0o700)
    _,active,grace,hard=STAGES[stage]; deadline=began+hard
    require(time.monotonic()<began+active, 'Prelaunch active cap')
    env=phase_environment(stage)
    owner=proc(os.getpid())
    receipt=dict(schema='Wiki24-direct12-phase-owner-live-v1',stage=stage,adapter_owner=owner,
        activation=binding(args.activation),science_release=binding(release_path),argv=argv,cwd=str(REPO),
        environment={k:env[k] for k in ('PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        active_seconds=active,cleanup_seconds=grace,hard_seconds=hard,started_monotonic=began,
        active_deadline_monotonic=began+active,hard_deadline_monotonic=deadline,
        science_manifest_sha256=SCIENCE_SHA,owner_manifest_sha256=cfg['owner_manifest_sha256'],
        started_UTC=began_utc)
    write(output/'LIVE.json',receipt)
    for signum in (signal.SIGTERM,signal.SIGINT): signal.signal(signum,interrupted)
    child=None; handle=None; code=None; reaped=False; timed_out=False; error=None; phase_wait_time=None
    records={}; unknown={}; cleanup_records=[]
    capture=lambda:record_workers(handle,science_output,receipt['science_release'],output,records,unknown) if stage=='fit' and handle is not None else None
    try:
        with (output/'phase.log').open('x') as log:
            child=subprocess.Popen(argv,cwd=REPO,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
            actual=proc(child.pid)
            require(actual is not None and actual['group']==actual['session']==child.pid, 'Actual owned phase session')
            handle=identity(actual); write(output/'CHILD_OWNER.json',handle)
            while True:
                if stage=='fit': record_workers(handle,science_output,receipt['science_release'],output,records,unknown)
                remaining=began+active-time.monotonic()
                if remaining<=0: raise subprocess.TimeoutExpired(argv,active)
                try:
                    code=child.wait(timeout=min(.25,remaining)); reaped=True; phase_wait_time=time.monotonic(); break
                except subprocess.TimeoutExpired: continue
    except (Exception,KeyboardInterrupt) as failure:
        timed_out=isinstance(failure,subprocess.TimeoutExpired)
        error=type(failure).__name__+': '+str(failure)
        if child is not None and handle is not None:
            if stage=='fit': record_workers(handle,science_output,receipt['science_release'],output,records,unknown)
            code,reaped,cleanup_records=cleanup(child,handle,stage,records,min(deadline,time.monotonic()+grace),capture)
            if reaped: phase_wait_time=time.monotonic()
    finally:
        if child is not None and handle is not None and reaped and any(verified_worker(r) is not None for r in records.values()):
            error=error or 'A verified recorded worker survived phase exit'
            code,reaped,cleanup_records=cleanup(child,handle,stage,records,min(deadline,time.monotonic()+grace),capture)
        worker_states=[dict(handle=r['handle'],original_worker_live=live(r['handle']) is not None,
                            descendant_wait_reap_claimed=False) for r in records.values()]
        phase_terminal=reaped and code is not None
        fit_seal=None; closure_terminal=stage!='fit'; resolved_unknown=[]
        if stage=='fit' and (science_output/'SEAL.json').is_file():
            try:
                fit_seal=binding(science_output/'SEAL.json'); saved=read(science_output/'SEAL.json')
                closure=read(bound(saved['closure']))
                closure_terminal=(saved.get('closed') is True and saved.get('whole12_accounted') is True
                    and saved.get('source_manifest_sha256')==SCIENCE_SHA and closure.get('all_owned_workers_reaped') is True)
                if closure_terminal:
                    for row in closure['rows']:
                        if 'owner' not in row or 'worker_job' not in row: continue
                        job=read(bound(row['worker_job']))
                        require(job['owner']==handle and job['root_release']==receipt['science_release']
                                and job['source_manifest_sha256']==SCIENCE_SHA, 'Closed fitter worker ownership')
                        key=(row['owner']['pid'],row['owner']['start_ticks'])
                        if key in unknown and unknown[key]['handle']==row['owner']:
                            resolved_unknown.append(unknown.pop(key))
            except (KeyError,ValueError,FileNotFoundError): closure_terminal=False
        # Unknown observed child PIDs are never queried after losing the parent.
        # Without exact recorded custody, do not assert all descendants terminal.
        all_terminal=phase_terminal and closure_terminal and not any(r['original_worker_live'] for r in worker_states) and not unknown
        elapsed=time.monotonic()-began
        phase_elapsed=None if phase_wait_time is None else phase_wait_time-began
        active_exceeded=phase_elapsed is not None and phase_elapsed>active
        hard_exceeded=elapsed>hard
        within_bounds=code==0 and not timed_out and error is None and all_terminal and not active_exceeded and not hard_exceeded
        terminal=dict(schema='Wiki24-direct12-phase-owner-terminal-v1',stage=stage,adapter_owner=owner,
            phase_owner=handle,exit_code=code,actual_phase_exit_and_reap_observed=phase_terminal,
            owner_and_children_terminal=all_terminal,fit_seal=fit_seal,
            active_timeout=timed_out,error=error,recorded_workers=worker_states,
            unqualified_direct_children=list(unknown.values()),transient_children_resolved_by_reviewed_fitter_reap=resolved_unknown,
            cleanup=cleanup_records,log=binding(output/'phase.log') if (output/'phase.log').is_file() else None,
            live_receipt=binding(output/'LIVE.json'),inclusive_wall_seconds=elapsed,
            phase_exit_reap_observation_elapsed_seconds=phase_elapsed,
            active_cap_exceeded=active_exceeded,hard_seconds=hard,hard_cap_exceeded=hard_exceeded,
            within_bound_terminal_phase=within_bounds,finished_UTC=datetime.now(timezone.utc).isoformat(),
            automatic_retry=False,TEST_access=False,unrelated_processes_scanned=False,
            descendant_reap_evidence='Only reviewed fitter closure claims its own workers reaped; adapter claims wait/reap only for its direct phase child.')
        write(output/'TERMINAL.json',terminal)
    return 0 if within_bounds else 1


if __name__=='__main__': sys.exit(main())
