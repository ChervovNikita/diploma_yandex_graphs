"""Three fresh full1100epoch SupCon fits, two fixed normal77 GPU lanes.

Only source/setup/completion/cost metadata is read. No prediction, trace,
checkpoint or quality reader exists. No retries or original-score replacement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (6101,6203,6307)


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''): digest.update(chunk)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value, fresh=False):
    payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if fresh:
        with Path(path).open('x') as stream: stream.write(payload)
    else:
        temporary=Path(path).with_suffix(Path(path).suffix+'.tmp')
        temporary.write_text(payload); os.replace(temporary,path)


def inside(relative):
    relative=Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts,'Phase-relative custody required')
    path=(PHASE/relative).resolve()
    require(path!=PHASE and path.is_relative_to(PHASE),'Path leaves project phase')
    return path


def bound(row):
    path=inside(row['path'])
    require(path.is_file() and sha(path)==row['sha256'],'Exact bound input changed: '+row['path'])
    if 'bytes' in row: require(path.stat().st_size==row['bytes'],'Bound size changed')
    return path


def seal(root, expected):
    require(sha(root/'MANIFEST.json')==expected,'Exact reviewed manifest required')
    for row in read(root/'MANIFEST.json')['files']:
        file=(root/row['path']).resolve()
        require(file.is_relative_to(root) and file.stat().st_size==row['bytes']
                and sha(file)==row['sha256'],'Source payload changed')


def proc(pid):
    # Same /proc identity fields as the reviewed normal-host direct12 owner.
    try:
        f=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,start_ticks=int(f[19]),group=int(f[2]),session=int(f[3]),ppid=int(f[1]),state=f[0])
    except (FileNotFoundError,ProcessLookupError): return None


def identity(row): return {k:row[k] for k in ('pid','start_ticks','group','session')}


def live(handle):
    actual=proc(handle['pid'])
    return actual if actual and identity(actual)==handle and actual['state']!='Z' else None


def signal_owned(handle, signum):
    if live(handle) is None: return False
    require(handle['pid']==handle['group']==handle['session'],'Only the exact owned session')
    try: os.killpg(handle['pid'],signum)
    except ProcessLookupError: return False
    return True


def stop(child, handle, deadline):
    actions=dict(SIGTERM=False,SIGCONT=False,SIGKILL=False)
    if child.poll() is None:
        require(handle is not None,'No unverified child may be signalled')
        actions['SIGTERM']=signal_owned(handle,signal.SIGTERM)
        actions['SIGCONT']=signal_owned(handle,signal.SIGCONT)  # Deliver TERM to a stopped owned child.
    try: return child.wait(timeout=max(0,min(5,deadline-time.monotonic()))),True,actions
    except subprocess.TimeoutExpired: pass
    actions['SIGKILL']=signal_owned(handle,signal.SIGKILL)
    try: return child.wait(timeout=max(0,deadline-time.monotonic())),True,actions
    except subprocess.TimeoutExpired: return child.poll(),False,actions


def providers():
    return {name:importlib.metadata.version(name) for name in
            ('torch','numpy','torch-geometric','torch-scatter','torch-sparse','ogb')}


def free_bytes(gpu):
    value=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free',
                                   '--format=csv,noheader,nounits'],text=True,timeout=5).strip()
    return int(value)*1024**2


def verify_setup(pins):
    runtime=pins['runtime']
    require(socket.gethostname()==runtime['hostname'] and Path.cwd()==Path(runtime['repository']),
            'Registered normal77 host/repository only')
    require(str(Path(sys.executable).absolute())==runtime['python']
            and os.environ.get('PYTHONPATH','')==runtime['PYTHONPATH'],'Normal existing interpreter/providers')
    actual=providers(); require(all(actual[k]==runtime[k] for k in actual),'Original numeric provider versions')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()
    require(inventory==runtime['physical_gpu_inventory'],'Original physical GPU inventory')
    seal(inside(pins['supcon_source']['path']),pins['supcon_source']['manifest_sha256'])
    seal(inside(pins['public_source']['path']),pins['public_source']['manifest_sha256'])
    for row in pins['inputs'].values(): bound(row)
    protocol=read(bound(pins['protocol']))
    require(protocol['root_adopted'] is True and protocol['seeds']==list(SEEDS)
            and protocol['public_source_manifest']==pins['supcon_source']['manifest_sha256'],'Exact frozen full3 protocol')
    return actual


def completion(output, seed, pins):
    # COMPLETE contains non-quality completion/source/cost metadata in the pinned CLI.
    path=output/'COMPLETE.json'; require(path.is_file(),'Missing complete-horizon metadata')
    record=read(path); specification=record['public_loss_comparison']
    require(record['complete'] is True and record['seed']==seed and record['epochs']==record['steps']==1100
            and record['TEST_scoring'] is False,'Full original horizon and role policy')
    require(specification['condition']=='supcon_eq2' and specification['epochs']==1100
            and specification['public_interface_manifest_sha256']==pins['public_source']['manifest_sha256'],
            'Exact candidate/source identity')
    source=inside(pins['supcon_source']['path'])
    require(specification['public_wrapper_sha256']==sha(source/'train.py')
            and specification['loss_module_sha256']==sha(source/'losses.py')
            and specification['recompute_sha256']==sha(source/'recompute.py'),'Completed source identities')
    require(record['active_loss_calls']==dict(alignment=1100,residual=0),'SupCon-only loss calls')
    require(record['execution_accounting']==dict(shadow_member_forwards=8800,replay_member_forwards=8800,
        output_cotangent_collections=1100,member_reverse_collections=8800,optimizer_bank_updates=1100,
        exact_member_RNG_endpoint_checks=1100),'Complete actual replay work')
    entry=read(output.parent.parent/'entry'/('seed'+str(seed)+'_TERMINAL.json'))
    require(entry['exit_code']==0 and entry['allocator_cap_bytes']==pins['limits']['owned_GPU_allocator_cap_bytes']
            and type(entry['peak_CUDA_reserved_bytes']) is int
            and entry['peak_CUDA_reserved_bytes']<=entry['allocator_cap_bytes'],'Actual bounded allocator entry')
    return dict(completion_sha256=sha(path),selected_sha256=record['selected_sha256'],
                entry_terminal_sha256=sha(output.parent.parent/'entry'/('seed'+str(seed)+'_TERMINAL.json')),
                worker_CPU_user_seconds=entry['CPU_user_seconds'],worker_CPU_system_seconds=entry['CPU_system_seconds'],
                worker_peak_RSS_bytes=entry['peak_RSS_bytes'],peak_CUDA_allocated_bytes=entry['peak_CUDA_allocated_bytes'],
                peak_CUDA_reserved_bytes=entry['peak_CUDA_reserved_bytes'])


def interrupted(signum, frame): raise InterruptedError('Controller signal '+str(signum))


def main():
    began=time.monotonic(); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',type=Path,required=True); parser.add_argument('--admission-sha256',required=True)
    args=parser.parse_args(); os.umask(0o077)
    require(sha(args.admission)==args.admission_sha256,'Exact separate root launch admission')
    cfg=read(args.admission); pins=read(HERE/'SOURCE_BINDINGS.json'); limits=pins['limits']
    require(cfg.get('schema')=='canonical-SupCon-full3-launch-admission-v1','Fixed root admission schema')
    for flag in ('enabled','root_scientific_launch_authorized','controller_source_review_approved','finite_resource_admission_confirmed'):
        require(cfg.get(flag) is True,'Root launch admission missing: '+flag)
    require(cfg.get('TEST_access') is False and cfg.get('automatic_retry') is False,'No TEST or retry')
    require(cfg['protocol_sha256']==pins['protocol']['sha256'],'Root admission binds the frozen full3 protocol')
    seal(HERE,cfg['controller_manifest_sha256']); actual_providers=verify_setup(pins)
    output=inside(cfg['output_directory']); require(not output.exists(),'Fresh family output required')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('cells','logs','jobs','entry'): (output/name).mkdir()
    owner=identity(proc(os.getpid())); family_hard=began+limits['family_hard_seconds']
    family_active=family_hard-limits['fit_cleanup_seconds']
    write(output/'OWNER.json',dict(owner=owner,admission_sha256=args.admission_sha256,
        controller_manifest_sha256=cfg['controller_manifest_sha256'],protocol=pins['protocol'],limits=limits,
        runtime_providers=actual_providers,started_UTC=datetime.now(timezone.utc).isoformat(),
        source_identity=pins['supcon_source'],original_quality_opened=False),fresh=True)
    queues={int(k):list(v) for k,v in pins['lanes'].items()}; waiting={}; active={}
    rows={s:dict(seed=s,condition='supcon_eq2',physical_gpu_uuid=pins['seed_GPU_map'][str(s)],
        status='unlaunched',automatic_retry=False) for s in SEEDS}; fatal=None
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,interrupted)

    def finish(lane, code, reaped, cleanup=None, reason=None):
        item=active.pop(lane); row=item['row']; elapsed=time.monotonic()-item['started']
        item['log'].close(); row.update(exit_code=code,actual_exit_and_reap=reaped,
            inclusive_fit_seconds=elapsed,owned_child_after=proc(item['child'].pid),cleanup=cleanup,
            log_sha256=sha(item['log_path']),hard_cap_exceeded=time.monotonic()>item['hard'])
        if reason: row['status']=reason
        elif code==0 and reaped:
            try: row.update(completion(item['output'],row['seed'],pins),status='complete')
            except (ValueError,KeyError,FileNotFoundError) as error:
                row.update(status='retained_completion_failure',error=str(error))
        else: row['status']='retained_fit_timeout' if elapsed>=limits['fit_active_seconds'] else 'retained_fit_failure'
        if row['hard_cap_exceeded']: row['status']='retained_hard_cap_overrun'
        if not reaped: row['status']='retained_unreaped_child'
        write(output/'CELL_STATUS.json',[rows[s] for s in SEEDS])
        print(json.dumps(dict(event='fit_terminal',seed=row['seed'],status=row['status'],
            exit_code=code,reaped=reaped,inclusive_seconds=elapsed)),flush=True)
        return reaped

    try:
        while any(queues.values()) or active:
            now=time.monotonic()
            if now>=family_active: raise TimeoutError('Whole-family active deadline; cleanup reserved')
            for lane,item in list(active.items()):
                try: code=item['child'].wait(timeout=0)
                except subprocess.TimeoutExpired:
                    if time.monotonic()<item['active_deadline']: continue
                    code,reaped,actions=stop(item['child'],item['handle'],min(item['hard'],family_hard))
                    if not finish(lane,code,reaped,actions,'retained_fit_timeout'):
                        raise RuntimeError('Owned fit was not reaped within its hard deadline')
                else: finish(lane,code,True)
            for lane,queue in queues.items():
                if lane in active or not queue: continue
                seed=queue[0]; row=rows[seed]; waiting.setdefault(lane,time.monotonic())
                gpu=row['physical_gpu_uuid']; available=free_bytes(gpu)
                row.update(memory_wait_seconds=time.monotonic()-waiting[lane],last_free_GPU_bytes=available)
                if row['memory_wait_seconds']>=limits['memory_wait_seconds_per_cell']:
                    row['status']='retained_memory_wait_timeout'; queue.pop(0); waiting.pop(lane)
                    write(output/'CELL_STATUS.json',[rows[s] for s in SEEDS])
                    continue
                if available<limits['minimum_fresh_free_GPU_bytes']:
                    continue
                verify_setup(pins)  # Fresh source/provider/data pins before each Popen.
                require(time.monotonic()+limits['fit_hard_seconds']<=family_hard,'Full per-cell envelope must fit family deadline')
                cell_output=output/'cells'/('supcon_eq2_'+str(seed)); require(not cell_output.exists(),'Fresh cell output')
                log_path=output/'logs'/('seed'+str(seed)+'.log'); job_path=output/'jobs'/('seed'+str(seed)+'.json')
                started=time.monotonic(); job=dict(schema='canonical-SupCon-full3-owned-worker-v1',seed=seed,
                    parent_owner=owner,controller_manifest_sha256=cfg['controller_manifest_sha256'],
                    admission_path=str(args.admission.resolve()),admission_sha256=args.admission_sha256,
                    protocol_sha256=pins['protocol']['sha256'],output=str(cell_output),
                    entry_directory=str(output/'entry'),physical_gpu_uuid=gpu,
                    fit_started_monotonic=started,active_deadline_monotonic=started+limits['fit_active_seconds'],
                    hard_deadline_monotonic=started+limits['fit_hard_seconds'])
                write(job_path,job,fresh=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1')
                argv=[sys.executable,'-B',str(HERE/'entry.py'),'--job',str(job_path),'--job-sha256',sha(job_path)]
                log=log_path.open('xb')
                try:
                    child=subprocess.Popen(argv,cwd=pins['runtime']['repository'],env=env,
                                           start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
                except BaseException as error:
                    log.close(); row.update(status='retained_launch_failure',error_type=type(error).__name__,error=str(error))
                    raise
                item=dict(child=child,handle=None,row=row,started=started,hard=job['hard_deadline_monotonic'],
                    active_deadline=job['active_deadline_monotonic'],log=log,log_path=log_path,output=cell_output)
                active[lane]=item; actual=proc(child.pid)
                require(actual is not None and actual['ppid']==os.getpid()
                        and actual['group']==actual['session']==child.pid,'Actual Popen PID/start/session custody')
                item['handle']=identity(actual); row.update(status='running',lane=lane,
                    owner=item['handle'],argv=argv,job_sha256=sha(job_path),launched_UTC=datetime.now(timezone.utc).isoformat())
                queue.pop(0); waiting.pop(lane)
                write(output/'CELL_STATUS.json',[rows[s] for s in SEEDS])
                print(json.dumps(dict(event='fit_launched',seed=seed,lane=lane,
                    physical_gpu_uuid=gpu,pid=child.pid,start_ticks=actual['start_ticks'])),flush=True)
            deadlines=[family_active]+[x['active_deadline'] for x in active.values()]
            deadlines += [v+limits['memory_wait_seconds_per_cell'] for v in waiting.values()]
            time.sleep(max(0,min(2,min(deadlines)-time.monotonic())))
    except (Exception,KeyboardInterrupt) as error:
        fatal=type(error).__name__+': '+str(error)
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,signal.SIG_IGN)
        # TERM every exact owned child first; then each bounded wait/reap shares the family deadline.
        for item in active.values():
            if item['handle'] is not None:
                signal_owned(item['handle'],signal.SIGTERM); signal_owned(item['handle'],signal.SIGCONT)
        cleanup_deadline=min(family_hard,time.monotonic()+limits['fit_cleanup_seconds'])
        for lane,item in list(active.items()):
            try:
                code,reaped,actions=stop(item['child'],item['handle'],min(item['hard'],cleanup_deadline))
                finish(lane,code,reaped,actions,'retained_controller_stop')
            except (ValueError,RuntimeError) as error:
                item['log'].close(); rows[item['row']['seed']].update(status='retained_unverified_or_unreaped_child',error=str(error))
        for row in rows.values():
            if row['status']=='unlaunched': row['status']='unlaunched_after_controller_stop'
        closure=dict(schema='canonical-SupCon-full3-owned-family-closure-v1',family_accounted=True,
            all_new_fits_complete=all(rows[s]['status']=='complete' for s in SEEDS),rows=[rows[s] for s in SEEDS],
            owner=owner,fatal_error=fatal,inclusive_family_seconds=time.monotonic()-began,
            family_hard_cap_exceeded=time.monotonic()>family_hard,limits=limits,
            controller_manifest_sha256=cfg['controller_manifest_sha256'],protocol_sha256=pins['protocol']['sha256'],
            automatic_retry=False,quality_or_prediction_opening_performed=False,
            comparative_opening_authorized=False,original12_closure_still_required=True,
            original_paper_scores_changed=False,unrelated_processes_signalled=False)
        write(output/'CLOSURE.json',closure,fresh=True)
        print(json.dumps(dict(event='family_terminal',family_accounted=True,
            all_new_fits_complete=closure['all_new_fits_complete'],quality_opened=False,
            inclusive_seconds=closure['inclusive_family_seconds'])),flush=True)
    return 0 if closure['all_new_fits_complete'] and fatal is None and not closure['family_hard_cap_exceeded'] else 1


if __name__=='__main__': sys.exit(main())
