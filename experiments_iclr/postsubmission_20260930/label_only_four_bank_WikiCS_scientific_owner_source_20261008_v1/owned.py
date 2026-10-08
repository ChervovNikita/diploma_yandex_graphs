"""Disabled normal-host serial owner; exactly three full four-bank seed blocks."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
SEEDS=(6101,6203,6307)
ARMS=('C4','S_joint4head','U4_sharedB','S_one_path')
ACTIVE=43200; CLEANUP=15; STARTUP=120; ADMISSION=60; TERMINAL=30
FAMILY=STARTUP+3*(ADMISSION+ACTIVE+CLEANUP+TERMINAL)


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):value.update(block)
    return value.hexdigest()


def write(path,value,exclusive=False):
    path=Path(path)
    if exclusive:
        with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    else:
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(temporary,path)


def identity(pid):
    try:
        value=Path('/proc',str(pid),'stat').read_text();parts=value[value.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(parts[19]),state=parts[0],group=int(parts[2]),session=int(parts[3]))
    except FileNotFoundError:return None


def bound(phase,row):
    path=phase/row['path']
    require(sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'Changed exact binding: '+row['path'])
    return path


def seal(root,digest):
    require(sha(root/'MANIFEST.json')==digest,'Changed sealed manifest')
    for row in read(root/'MANIFEST.json')['files']:
        path=root/row['path'];require(sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'Changed source payload')


def gpu_rows():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()


def validate(release_path,release_sha,authorized):
    require(authorized is True and sha(release_path)==release_sha,'Separate exact enabled root release required')
    cfg=read(release_path);fixed=read(HERE/'ROOT_RELEASE_TEMPLATE_DISABLED.json')
    variable={'enabled','scientific_execution_authorized','source_review_approved','owner_manifest_sha256','execution_source_commit'}
    require(set(cfg)==set(fixed) and all(cfg[k]==fixed[k] for k in fixed if k not in variable)
        and all(cfg[k] is True for k in ('enabled','scientific_execution_authorized','source_review_approved')),'Disabled/exact frozen scientific release')
    require(isinstance(cfg['execution_source_commit'],str) and len(cfg['execution_source_commit'])==40
        and all(c in '0123456789abcdef' for c in cfg['execution_source_commit']),'Actual committed execution source required')
    seal(HERE,cfg['owner_manifest_sha256']);pins=read(HERE/'SOURCE_BINDINGS.json');runtime=pins['runtime'];phase=Path(runtime['phase'])
    require(HERE.parent==phase and socket.gethostname()==runtime['hostname'],'Original normal allocation host/phase')
    for row in pins['source_files']:bound(phase,row)
    for item in pins['source_manifests']:seal(phase/item['directory'],item['sha256'])
    projection=read(bound(phase,pins['projection_manifest']))
    require(projection['schema']=='internal-be-official-role-projection-v2' and projection['official_split_preserved'] is True
        and projection['train_count']==580 and projection['valid_count']==5274 and projection['TEST_values_in_payload'] is False,'Official whole split0/noTEST')
    for row in [pins['safe_payload'],pins['official_manifest'],*pins['roles'].values(),pins['polynormer']]:bound(phase,row)
    require(projection['payloads']==pins['roles'] and projection['source_custody']['safe_payload']==pins['safe_payload'],'Projection/numeric role custody')
    result=read(bound(phase,pins['engineering_result']));terminal=read(bound(phase,pins['engineering_terminal']))
    witness=read(bound(phase,pins['engineering_absence']));review=read(bound(phase,pins['engineering_review']))
    require(result['engineering_passed'] is True and result['error'] is None and len(result['checks'])==8
        and all(x is True for x in result['checks'].values()) and result['work']['native_complete_updates']==4
        and result['work']['all_four_bank_complete_updates']==4 and result['VALID_truth_scoring'] is False
        and result['selector_performed'] is False and result['complete_1100_schedule_qualified'] is False,'Exact successful discarded CUDA engineering receipt')
    require(terminal['outcome']=='engineering_passed' and terminal['return_code']==0 and terminal['worker_reaped'] is True
        and review['engineering_only_adopted'] is True and witness['observed']==dict(parent=None,worker=None)
        and witness['source_commit']==pins['qualified_source_commit'],'Engineering terminal/root adoption/absence custody')
    require(result['source_binding']['manifest_sha256']==pins['screen_manifest_sha256']
        and result['source_binding']['program_sha256']==pins['screen_program_sha256']
        and result['observed_metadata_sha256']['projection_manifest']==pins['projection_manifest']['sha256']
        and result['observed_metadata_sha256']['available_manifest']==pins['official_manifest']['sha256'],'Qualification/source/data join')
    require(max(result['own_peak_cuda_allocated_bytes'],result['own_peak_cuda_reserved_bytes'])<=cfg['max_owned_GPU_bytes'],'Qualified peak within own safety cap')
    require(result['runtime']['torch']=='2.1.2+cu118' and result['runtime']['numpy']=='1.26.4','Qualified original providers')
    rows=gpu_rows();pids={row.split(',')[0].strip() for row in rows}
    for key in ('recorded_parent','recorded_worker'):
        old=witness[key];actual=identity(old['pid'])
        require((actual is None or actual['start_ticks']!=old['start_ticks']) and str(old['pid']) not in pids,'Engineering identity freshly terminal; no requalification')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=5).splitlines()==[runtime['GPU_uuid']],'Sole original physical GPU')
    repo=runtime['repository'];commit=cfg['execution_source_commit']
    subprocess.run(['git','merge-base','--is-ancestor',pins['qualified_source_commit'],commit],cwd=repo,check=True,timeout=10)
    committed=subprocess.check_output(['git','show',commit+':experiments_iclr/postsubmission_20260930/'+HERE.name+'/MANIFEST.json'],cwd=repo,timeout=10)
    require(hashlib.sha256(committed).hexdigest()==cfg['owner_manifest_sha256'],'Owner manifest bound to actual execution commit')
    require(Path(runtime['python']).is_file() and all(Path(x).is_dir() for x in runtime['PYTHONPATH']),'Existing qualified runtime only')
    return cfg,pins,phase


def stop(child,saved,actions,deadline=None):
    deadline=min(time.monotonic()+CLEANUP,deadline) if deadline is not None else time.monotonic()+CLEANUP
    def send(number):
        current=identity(child.pid)
        if current is None:return
        require(all(current[k]==saved[k] for k in ('pid','start_ticks','group','session'))
            and saved['group']==saved['session']==saved['pid'],'Only exact newly owned child group may be signaled')
        try:os.killpg(saved['group'],number);actions.append(int(number))
        except ProcessLookupError:pass
    if child.poll() is None:send(signal.SIGTERM)
    try:child.wait(timeout=max(.01,min(5,deadline-time.monotonic())))
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL);child.wait(timeout=max(.01,deadline-time.monotonic()))


def run_owned(*,release_path,release_sha256,later_execution_authorized=False,launch_started=None):
    require(later_execution_authorized is True,'Disabled serial scientific owner')
    began=time.monotonic() if launch_started is None else launch_started
    def interrupted(number,frame):raise RuntimeError('Owned family stop/deadline '+str(number))
    for number in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(number,interrupted)
    signal.setitimer(signal.ITIMER_REAL,max(.01,FAMILY-CLEANUP-(time.monotonic()-began)))
    cfg,pins,phase=validate(release_path,release_sha256,True)
    parent=identity(os.getpid());require(parent['group']==parent['session']==parent['pid'],'Detached scientific parent required')
    require(time.monotonic()-began<STARTUP,'Finite startup reserve')
    output=phase/cfg['output_relative'];require(not output.exists(),'Fresh family only; no restart/resume/overwrite')
    output.mkdir(parents=True);(output/'handles').mkdir();(output/'logs').mkdir()
    write(output/'PARENT_OWNER.json',dict(identity=parent,release_sha256=release_sha256,execution_source_commit=cfg['execution_source_commit'],
        qualified_source_commit=pins['qualified_source_commit'],family_seconds=FAMILY,serial_one_active_seed=True,coexecution='Mol18',scores_read=False),True)
    completed=[];child=saved=None;active_seed=None;closed=False;error=None
    try:
        for seed in SEEDS:
            active_seed=seed;admitted=time.monotonic()
            free=int(subprocess.check_output(['nvidia-smi','--id='+pins['runtime']['GPU_uuid'],'--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True,timeout=5).strip())*1024**2
            require(free>=cfg['minimum_fresh_GPU_bytes'] and time.monotonic()-admitted<ADMISSION,'Fresh GPU headroom/finite admission; no arbitrary waiting')
            cell=output/('seed'+str(seed));require(not cell.exists(),'Never restart an old seed')
            handle=output/'handles'/('seed'+str(seed)+'.json')
            value=dict(parent=parent,seed=seed,release_path=str(Path(release_path).resolve()),release_sha256=release_sha256,
                output=str(cell),child=None,owner_manifest_sha256=cfg['owner_manifest_sha256'])
            write(handle,value,True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=pins['runtime']['GPU_uuid'],PYTHONPATH=os.pathsep.join(pins['runtime']['PYTHONPATH']),
                PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
            env.pop('PYTHONHOME',None);start=time.monotonic();reason=None;actions=[];max_gpu=max_rss=0
            with (output/'logs'/('seed'+str(seed)+'.log')).open('xb') as log:
                child=subprocess.Popen([pins['runtime']['python'],'-B',str(HERE/'owned.py'),'--worker',str(handle)],
                    cwd=pins['runtime']['repository'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                saved=identity(child.pid);require(saved and saved['group']==saved['session']==child.pid,'Exact newly owned seed child')
                value['child']=saved;write(handle,value)
                while child.poll() is None:
                    if time.monotonic()-start>=ACTIVE:reason='fixed12hour_active_safety_bound';break
                    try:
                        status=Path('/proc',str(child.pid),'status').read_text()
                        rss=next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0)
                    except FileNotFoundError:
                        if child.poll() is not None:break
                        raise
                    rows=gpu_rows();memory=sum(int(parts[1].strip())*1024**2 for parts in (x.split(',') for x in rows)
                        if len(parts)==2 and parts[0].strip()==str(child.pid) and parts[1].strip().isdigit())
                    max_gpu=max(max_gpu,memory);max_rss=max(max_rss,rss)
                    if memory>cfg['max_owned_GPU_bytes'] or rss>cfg['max_owned_RSS_bytes']:reason='owned_resource_safety_cap';break
                    try:child.wait(timeout=min(2,max(.01,ACTIVE-(time.monotonic()-start))))
                    except subprocess.TimeoutExpired:pass
                if reason:stop(child,saved,actions,deadline=start+ACTIVE+CLEANUP)
                else:child.wait(timeout=.01)
            elapsed=time.monotonic()-start;terminal_start=time.monotonic()
            require(child.poll() is not None and identity(child.pid) is None and str(child.pid) not in
                {x.split(',')[0].strip() for x in gpu_rows()},'Owned child wait/reap/absence/noCUDA terminal custody')
            receipt=dict(seed=seed,child=saved,exit_code=child.returncode,reaped=True,reason=reason,signals=actions,
                active_and_cleanup_seconds=elapsed,resource_admission_seconds=start-admitted,
                max_sampled_owned_GPU_bytes=max_gpu,max_sampled_RSS_bytes=max_rss,scores_read=False,automatic_retry=False)
            write(output/'handles'/('seed'+str(seed)+'.EXIT.json'),receipt,True)
            require(child.returncode==0 and reason is None and elapsed<=ACTIVE+CLEANUP,'Failed/incomplete seed retained; no retry')
            done=read(cell/'COMPLETE.json');require(done['complete'] is True and done['epochs']==1100
                and done['native_trajectories']==1 and done['run']['seed']==seed
                and done['run']['source']['manifest_sha256']==pins['screen_manifest_sha256']
                and [x['arm'] for x in done['required_bank_records']]==list(ARMS),'Exact full frozen seed/block completion')
            require(done['work']['native_update_completions']==done['work']['all_bank_update_completions']==1100
                and done['work']['complete_VALID_events']==1100 and done['work']['native_local_restorations']==1,'Full1100 native/four-bank work')
            for row in done['required_bank_records']:
                require(row['complete'] is True and row['epochs']==1100 and sha(cell/row['arm']/'selected.pt')==row['selected_sha256'],
                    'Every required correction record and selected-state bytes retained')
            require(time.monotonic()-terminal_start<TERMINAL,'Finite cell terminal reserve')
            completed.append(dict(seed=seed,completion_sha256=sha(cell/'COMPLETE.json'),worker_cost_sha256=sha(cell/'WORKER_COST.json'),
                required_arm_count=4,exit_receipt=receipt));child=saved=None;active_seed=None
            write(output/'PROGRESS.json',dict(completed=completed,fixed_seeds=list(SEEDS),scores_read=False))
        require(len(completed)==3 and time.monotonic()-began<FAMILY,'All three fixed seed blocks within family safety envelope')
        write(output/'FAMILY_CLOSURE.json',dict(complete=True,completed=completed,fixed_seeds=list(SEEDS),required_correction_records=12,
            full_epochs_per_block=1100,qualified_source_commit=pins['qualified_source_commit'],execution_source_commit=cfg['execution_source_commit'],
            protocol=pins['protocol'],engineering_result=pins['engineering_result'],dataset_projection=pins['projection_manifest'],
            source_manifest_sha256=cfg['owner_manifest_sha256'],scores_read=False,comparative_opening_authorized=False,automatic_retry=False),True)
        closed=True
    except BaseException as caught:
        error=dict(type=type(caught).__name__,message=str(caught));cleanup=None
        signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.setitimer(signal.ITIMER_REAL,0)
        if child is not None:
            actions=[]
            try:stop(child,saved,actions);cleanup=dict(reaped=child.poll() is not None,exit_code=child.returncode,signals=actions)
            except BaseException as failed:cleanup=dict(reaped=False,error=type(failed).__name__+': '+str(failed),signals=actions)
        write(output/'FAMILY_FAILURE.json',dict(error=error,completed=completed,active_seed=active_seed,owned_cleanup=cleanup,
            all_partial_files_and_costs_retained=True,automatic_retry=False,scores_read=False),True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        write(output/'PARENT_TERMINAL.json',dict(family_complete=closed,error=error,completed_seed_count=len(completed),
            seconds=time.monotonic()-began,owner=parent,scores_read=False,automatic_retry=False),True)
    return dict(family_complete=True,seeds=list(SEEDS),output=str(output),scores_read=False)


def worker(handle_path):
    deadline=time.monotonic()+5
    while True:
        handle=read(handle_path)
        if handle['child'] and handle['child']['pid']==os.getpid():break
        require(time.monotonic()<deadline,'Finite parent registration');time.sleep(.01)
    actual=identity(os.getpid());parent=identity(os.getppid())
    require(actual==handle['child'] or all(actual[k]==handle['child'][k] for k in ('pid','start_ticks','group','session')),'Exact owned child')
    require(parent and all(parent[k]==handle['parent'][k] for k in ('pid','start_ticks','group','session')),'Exact live detached parent')
    require(ctypes.CDLL(None,use_errno=True).prctl(1,int(signal.SIGTERM),0,0,0)==0,'Owned parent-death stop signal')
    require(os.getppid()==parent['pid'] and identity(parent['pid'])['start_ticks']==parent['start_ticks'],'Parent remained live during registration')
    def interrupted(number,frame):raise RuntimeError('Owned child stop '+str(number))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    started=time.monotonic();output=Path(handle['output']);torch=None;success=False;error=None;cfg=None
    try:
        cfg,pins,phase=validate(handle['release_path'],handle['release_sha256'],True)
        require(handle['seed'] in SEEDS and sys.executable==pins['runtime']['python'],'Original pinned child interpreter/seed')
        import torch
        require(str(torch.__version__)=='2.1.2+cu118' and torch.cuda.device_count()==1,'Qualified single-GPU provider')
        torch.cuda.set_per_process_memory_fraction(cfg['max_owned_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        path=bound(phase,pins['screen_program']);spec=importlib.util.spec_from_file_location('_owned_frozen_four_bank_screen',path)
        screen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=screen;spec.loader.exec_module(screen)
        screen.run_complete(train=phase/pins['roles']['train']['path'],valid=phase/pins['roles']['valid']['path'],output=output,
            polynormer=phase/pins['polynormer']['path'],seed=handle['seed'],device='cuda:0',later_execution_authorized=True)
        success=True
    except BaseException as caught:error=dict(type=type(caught).__name__,message=str(caught));raise
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        cost_path=output/'WORKER_COST.json' if output.is_dir() else Path(handle_path).parent/('seed'+str(handle['seed'])+'.WORKER_COST.json')
        try:
            allocated=torch.cuda.max_memory_allocated(0) if torch and torch.cuda.is_initialized() else None
            reserved=torch.cuda.max_memory_reserved(0) if torch and torch.cuda.is_initialized() else None
        except Exception:allocated=reserved=None
        write(cost_path,dict(success=success,error=error,seed=handle['seed'],seconds=time.monotonic()-started,
                CPU_user_seconds=usage.ru_utime,CPU_system_seconds=usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,
                peak_cuda_allocated_bytes=allocated,peak_cuda_reserved_bytes=reserved,
                execution_source_commit=cfg['execution_source_commit'] if cfg else None,release_sha256=handle['release_sha256'],
                partial_driver_outputs_preserved=True,automatic_retry=False),True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=Path)
    parser.add_argument('--release',type=Path);parser.add_argument('--release-sha256');parser.add_argument('--authorized',action='store_true')
    parser.add_argument('--launch-started',type=float);args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:print(json.dumps(run_owned(release_path=args.release,release_sha256=args.release_sha256,
        later_execution_authorized=args.authorized,launch_started=args.launch_started)))


if __name__=='__main__':main()
