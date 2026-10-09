"""Inactive serial36 normal-host owner; only witnessed owned groups may be stopped."""
import argparse
import ctypes
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from common import HERE, PHASE, CLEANUP, bound, module, queue, read, release_config, require, sealed, sha, source_identity, write

_HELPERS = None


def helpers(pins=None):
    global _HELPERS
    if _HELPERS is None:
        if pins is None:
            pins = read(HERE / 'SOURCE_BINDINGS.json')
        _HELPERS = module(bound(PHASE, pins['process_helpers']), '_qk36_audited_process_custody')
    return _HELPERS


def identity(pid): return helpers().identity(pid)
def same(current, saved): return helpers().same(current, saved)
def members(saved): return helpers().members(saved)
def gpu_rows(): return helpers().gpu_rows()


def validate(path, digest, authorized=False):
    cfg, pins, phase = release_config(path, digest, authorized)
    runtime = pins['runtime']
    require(read(bound(phase, pins['runtime_metadata'])) == runtime, 'Exact original native runtime metadata')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
            text=True, timeout=5).splitlines() == [runtime['GPU_uuid']], 'Sole declared physical GPU')
    for row in pins['roles'].values(): bound(phase, row)
    commit = cfg['execution_source_commit']
    for directory, expected in ((HERE.name, cfg['owner_manifest_sha256']),
                                (pins['adapter_directory'], pins['adapter_manifest_sha256']),
                                (pins['public_directory'], pins['public_manifest_sha256'])):
        data = subprocess.check_output(['git', 'show', commit + ':' + pins['repository_phase_relative']
                + '/' + directory + '/MANIFEST.json'], cwd=runtime['repository'], timeout=10)
        import hashlib
        require(hashlib.sha256(data).hexdigest() == expected, 'Exact source manifest in execution commit')
    require(Path(runtime['python']).is_file() and all(Path(p).is_dir() for p in runtime['PYTHONPATH']), 'Existing normal runtime')
    return cfg, pins, phase


def verify_closed_family(root, cfg, pins, manifest):
    """Mechanical all36 endpoint barrier; no metrics, traces or weights deserialize."""
    records = {}
    parent = read(root / 'PARENT_OWNER.json')
    require(parent['required_group_endpoints'] == 36 and parent['queue'] == queue(cfg)
            and parent['source_manifest_sha256'] == manifest
            and parent['execution_source_commit'] == cfg['execution_source_commit'], 'Exact prospective whole queue')
    for spec in queue(cfg):
        cell = root / spec['key']
        require(not (cell / 'FAILURE.json').exists(), 'No failed endpoint in complete36 family')
        value = read(cell / 'COMPLETE.json')
        context = value['owner_context']
        members_count = 1 if spec['kind'] == 'single' else 4
        adam = 4 if spec['kind'] == 'independent4' else 1
        require(value['complete'] is True and value['task'] == 'wikics' and value['arm'] == spec['kind']
                and value['seed'] == spec['seed'] and value['epochs'] == value['steps'] == 1100
                and value['all1100_and100_schedule'] is True and value['TEST_scoring'] is False
                and value['comparative_opening_performed'] is False and context['cell'] == spec
                and context['source'] == source_identity(pins, manifest) and context['roles'] == pins['roles']
                and context['runtime'] == pins['runtime'] and context['release_sha256'] == parent['release_sha256']
                and context['qualification_adoption'] == cfg['qualification_adoption']
                and context['execution_source_commit'] == cfg['execution_source_commit']
                and value['operator_binding']['operator'] == spec['operator']
                and value['operator_binding']['kind'] == spec['kind']
                and value['operator_binding']['seed'] == spec['seed']
                and value['operator_binding']['adapter_manifest_sha256'] == pins['adapter_manifest_sha256']
                and value['operator_work'] == dict(update_attempts=1100, completed_updates=1100,
                    member_view_forwards=2*members_count*1100, member_view_backwards=2*members_count*1100, Adam_steps=adam*1100),
                'Exact complete ordinary/shared F cell, source and work')
        expected = {'RUN.json', 'PRIVATE_VALID_TRACE.json', 'selected_local.pt', 'selected.pt'}
        if spec['kind'] in ('single', 'independent4'):
            expected |= {'own_' + which + '_' + str(m) + '.pt'
                         for which in ('local', 'best') for m in range(members_count)}
        if spec['kind'] == 'independent4': expected.add('PRIVATE_OWN_BEST_BANK.json')
        require(set(value['bound_files']) == expected and sha(cell / 'selected.pt') == value['selected_sha256'],
                'Complete own/local/selected file inventory')
        for relative, digest in value['bound_files'].items():
            file = (cell / relative).resolve(strict=True)
            require(file.is_relative_to(cell.resolve()) and sha(file) == digest, 'Exact endpoint file bytes')
        records[spec['key']] = dict(cell=spec, complete_sha256=sha(cell / 'COMPLETE.json'))
    require(len(records) == 36, 'Exactly36 completed groups; never a smaller substituted panel')
    return records


def run_owned(*, release_path, release_sha256, authorized=False, launch_started=None):
    began = time.monotonic() if launch_started is None else launch_started
    cfg, pins, phase = release_config(release_path, release_sha256, authorized)
    output, activation = phase / cfg['output_relative'], phase / cfg['activation_relative']
    require(not output.exists(), 'Fresh complete36 family only; no retry/resume/overwrite')
    output.mkdir()
    (output / 'handles').mkdir()
    (output / 'logs').mkdir()
    parent, completed = identity(os.getpid()), []
    child = saved = None
    known, active, error, closed = [], None, None, False
    family_deadline = began + cfg['family_seconds']

    def interrupted(number, frame): raise RuntimeError('Owned complete36 stop/deadline ' + str(number))
    def timer(deadline):
        require(time.monotonic() < deadline, 'Finite complete36 owner reserve')
        signal.setitimer(signal.ITIMER_REAL, deadline-time.monotonic())
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM): signal.signal(number, interrupted)
    try:
        timer(began + cfg['startup_seconds'])
        require(parent and parent['group'] == parent['session'] == parent['pid'], 'Detached exact scientific parent')
        deadline = min(began+cfg['startup_seconds'], time.monotonic()+5)
        while not same(read(activation / 'LAUNCH.json').get('parent'), parent):
            require(time.monotonic() < deadline, 'Finite exact parent launch registration')
            time.sleep(.01)
        write(output / 'PARENT_OWNER.json', dict(identity=parent, release_sha256=release_sha256,
            execution_source_commit=cfg['execution_source_commit'], source_manifest_sha256=cfg['owner_manifest_sha256'],
            queue=queue(cfg), required_group_endpoints=36, serial_one_active_cell=True, scores_read=False), True)
        cfg, pins, phase = validate(release_path, release_sha256, True)
        for index, spec in enumerate(queue(cfg)):
            active = spec
            admitted = time.monotonic()
            timer(min(family_deadline-CLEANUP, admitted+cfg['admission_seconds']))
            free = int(subprocess.check_output(['nvidia-smi', '--id=' + pins['runtime']['GPU_uuid'],
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 1024**2
            require(free >= cfg['minimum_fresh_GPU_bytes'], 'Root-admitted fresh GPU headroom; no other-process action')
            cell = output / spec['key']
            cell.parent.mkdir(parents=True, exist_ok=True)
            require(not cell.exists(), 'Fresh per-cell output')
            handle = output / 'handles' / ('cell' + str(index) + '.json')
            value = dict(parent=parent, cell=spec, release_path=str(Path(release_path).resolve()),
                         release_sha256=release_sha256, output=str(cell), child=None)
            write(handle, value, True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=pins['runtime']['GPU_uuid'],
                PYTHONPATH=os.pathsep.join(pins['runtime']['PYTHONPATH']), PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
            env.pop('PYTHONHOME', None)
            start, reason, actions, max_gpu, max_rss = time.monotonic(), None, [], 0, 0
            child_usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            timer(min(family_deadline-CLEANUP, start+cfg['cell_active_seconds']))
            with (output / 'logs' / ('cell' + str(index) + '.log')).open('xb') as log:
                child = subprocess.Popen([pins['runtime']['python'], '-B', str(HERE / 'owned.py'), '--worker', str(handle)],
                    cwd=pins['runtime']['repository'], env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                saved = identity(child.pid)
                require(saved and saved['group'] == saved['session'] == child.pid, 'New child PID/birth/group custody')
                known = [saved]
                value['child'] = saved
                write(handle, value)
                timer(min(family_deadline-CLEANUP, start+cfg['cell_active_seconds']+CLEANUP))
                while child.poll() is None:
                    if time.monotonic()-start >= cfg['cell_active_seconds']:
                        reason = 'root_admitted_cell_active_limit'
                        break
                    require(same(identity(child.pid), saved), 'Live own child birth custody')
                    current = members(saved)
                    known = list({(item['pid'],item['start_ticks']):item for item in known+current}.values())
                    rss = 0
                    for item in current:
                        try:
                            status = Path('/proc', str(item['pid']), 'status').read_text()
                            require(same(identity(item['pid']), item), 'Owned resource-sample birth custody')
                            rss += next((int(row.split()[1])*1024 for row in status.splitlines() if row.startswith('VmRSS:')), 0)
                        except FileNotFoundError: pass
                    gpu = sum(used for pid,used in gpu_rows() if pid in {item['pid'] for item in current})
                    max_gpu, max_rss = max(max_gpu,gpu), max(max_rss,rss)
                    if gpu > cfg['max_owned_GPU_bytes'] or rss > cfg['max_owned_RSS_bytes']:
                        reason = 'root_admitted_owned_resource_limit'
                        break
                    try: child.wait(timeout=min(2,max(.01,cfg['cell_active_seconds']-(time.monotonic()-start))))
                    except subprocess.TimeoutExpired: pass
                if reason: helpers().stop(child, saved, known, actions, min(family_deadline,start+cfg['cell_active_seconds']+CLEANUP))
                else: child.wait(timeout=.01)
            elapsed, terminal = time.monotonic()-start, time.monotonic()
            timer(min(family_deadline,terminal+cfg['terminal_seconds']))
            absent = identity(saved['pid']) is None and not members(saved)
            no_cuda = not {pid for pid,_ in gpu_rows()} & {item['pid'] for item in known}
            child_usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            receipt = dict(cell=spec, child=saved, witnessed_owned_members=known, reaped=child.poll() is not None,
                exit_code=child.returncode, reason=reason, group_absent=absent, no_owned_CUDA=no_cuda, signals=actions,
                active_and_cleanup_seconds=elapsed, admission_seconds=start-admitted,
                max_sampled_owned_GPU_bytes=max_gpu, max_sampled_owned_RSS_bytes=max_rss,
                owner_subprocess_CPU_user_seconds=child_usage_after.ru_utime-child_usage_before.ru_utime,
                owner_subprocess_CPU_system_seconds=child_usage_after.ru_stime-child_usage_before.ru_stime,
                CPU_scope='Worker plus owner metadata-query subprocesses during this cell interval',
                partial_work_and_costs_retained=True, scores_read=False, automatic_retry=False)
            write(handle.with_suffix('.EXIT.json'), receipt, True)
            require(child.returncode == 0 and reason is None and absent and no_cuda
                    and elapsed <= cfg['cell_active_seconds']+CLEANUP, 'Clean successful own cell; no retry')
            cost = read(cell / 'WORKER_COST.json')
            require(cost['success'] is True and cost['peak_RSS_bytes'] <= cfg['max_owned_RSS_bytes']
                    and max(cost['peak_cuda_allocated_bytes'],cost['peak_cuda_reserved_bytes']) <= cfg['max_owned_GPU_bytes'],
                    'Successful inclusive worker cost within admitted caps')
            done = read(cell / 'COMPLETE.json')
            require(done['complete'] is True and done['owner_context']['cell'] == spec, 'Completed exact cell receipt')
            completed.append(dict(cell=spec, handle_relative=str(handle.relative_to(output)), exit=receipt,
                                  complete_sha256=sha(cell/'COMPLETE.json'), cost_sha256=sha(cell/'WORKER_COST.json')))
            child = saved = None
            known, active = [], None
            write(output/'PROGRESS.json',dict(completed=completed,required_group_endpoints=36,scores_read=False))
        records = verify_closed_family(output,cfg,pins,cfg['owner_manifest_sha256'])
        write(output/'FAMILY_CLOSURE.json',dict(complete=True,required_group_endpoints=36,queue=queue(cfg),
            fixed_seeds=cfg['seeds'],completed=completed,records=records,source=source_identity(pins,cfg['owner_manifest_sha256']),
            release_sha256=release_sha256,execution_source_commit=cfg['execution_source_commit'],
            scores_read=False,comparative_opening_authorized=False,automatic_retry=False),True)
        closed = True
    except BaseException as caught:
        error = dict(type=type(caught).__name__,message=str(caught))
        for number in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM): signal.signal(number,signal.SIG_IGN)
        signal.setitimer(signal.ITIMER_REAL,0)
        cleanup = None
        if child is not None:
            actions=[]
            try:
                helpers().stop(child,saved,known,actions,time.monotonic()+CLEANUP)
                cleanup=dict(reaped=True,exit_code=child.returncode,signals=actions,group_absent=True)
            except BaseException as failed: cleanup=dict(error=type(failed).__name__+': '+str(failed),signals=actions)
        write(output/'FAMILY_FAILURE.json',dict(error=error,completed=completed,active_cell=active,owned_cleanup=cleanup,
            all_partial_files_and_costs_retained=True,scores_read=False,automatic_retry=False),True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        write(output/'PARENT_TERMINAL.json',dict(family_complete=closed,error=error,owner=parent,
            completed_group_count=len(completed),inclusive_seconds=time.monotonic()-began,
            owner_CPU_user_seconds=usage.ru_utime,owner_CPU_system_seconds=usage.ru_stime,
            subprocess_CPU_user_seconds=children.ru_utime,subprocess_CPU_system_seconds=children.ru_stime,
            owner_peak_RSS_bytes=usage.ru_maxrss*1024,subprocess_peak_RSS_bytes=children.ru_maxrss*1024,
            scores_read=False,automatic_retry=False),True)
    return dict(family_complete=closed,completed_groups=len(completed),output=str(output),scores_read=False)


def worker(handle_path):
    began, torch, cfg, success, error = time.monotonic(), None, None, False, None
    handle = read(handle_path)
    output = Path(handle['output'])
    try:
        deadline = began+5
        while not handle.get('child'):
            require(time.monotonic()<deadline,'Finite worker registration')
            time.sleep(.01)
            handle=read(handle_path)
        require(same(identity(os.getpid()),handle['child']) and same(identity(os.getppid()),handle['parent']),
                'Exact parent/child birth custody')
        require(ctypes.CDLL(None,use_errno=True).prctl(1,int(signal.SIGTERM),0,0,0)==0,'Owned parent-death stop')
        require(same(identity(os.getppid()),handle['parent']),'Parent live after death-signal registration')
        cfg,pins,phase=validate(handle['release_path'],handle['release_sha256'],True)
        require(handle['cell'] in queue(cfg) and output == phase/cfg['output_relative']/handle['cell']['key']
                and Path(sys.executable).resolve()==Path(pins['runtime']['python']).resolve(),'Exact cell/runtime/output')
        def interrupted(number,frame): raise RuntimeError('Owned worker deadline '+str(number))
        for number in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM): signal.signal(number,interrupted)
        signal.setitimer(signal.ITIMER_REAL,max(.01,cfg['cell_active_seconds']-(time.monotonic()-began)))
        import torch
        require(str(torch.__version__)==pins['runtime']['torch'] and torch.cuda.device_count()==1,'Qualified one-GPU runtime')
        torch.cuda.set_per_process_memory_fraction(cfg['max_owned_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
        from cell import run_cell
        run_cell(spec=handle['cell'],output=output,cfg=cfg,pins=pins,manifest=cfg['owner_manifest_sha256'],
                 release_sha256=handle['release_sha256'],authorized=True)
        success=True
    except BaseException as caught:
        error=dict(type=type(caught).__name__,message=str(caught))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        usage=resource.getrusage(resource.RUSAGE_SELF)
        allocated=reserved=None
        if torch is not None and torch.cuda.is_initialized():
            allocated,reserved=torch.cuda.max_memory_allocated(0),torch.cuda.max_memory_reserved(0)
        cost=output/'WORKER_COST.json' if output.is_dir() else Path(handle_path).with_suffix('.WORKER_COST.json')
        write(cost,dict(success=success,error=error,cell=handle['cell'],inclusive_seconds=time.monotonic()-began,
            CPU_user_seconds=usage.ru_utime,CPU_system_seconds=usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024,
            peak_cuda_allocated_bytes=allocated,peak_cuda_reserved_bytes=reserved,
            release_sha256=handle['release_sha256'],execution_source_commit=cfg['execution_source_commit'] if cfg else None,
            partial_work_and_costs_retained=True,automatic_retry=False),True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--release-sha256')
    parser.add_argument('--authorized',action='store_true')
    parser.add_argument('--launch-started',type=float)
    args=parser.parse_args()
    if args.worker: worker(args.worker)
    else: run_owned(release_path=args.release,release_sha256=args.release_sha256,
                    authorized=args.authorized,launch_started=args.launch_started)


if __name__=='__main__': main()
