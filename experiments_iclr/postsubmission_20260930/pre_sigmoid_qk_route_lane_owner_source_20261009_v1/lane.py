"""Disabled bounded normal-host lane: one qualification or twelve serial cells."""
import argparse
import ctypes
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from common import HERE, PHASE, ASSIGNMENTS, OPERATORS, KINDS, config, module, qualification, read, require, sha, sources, write


def interrupted(number, frame):
    raise RuntimeError('Owned lane stop/deadline ' + str(number))


def timer(deadline):
    require(time.monotonic() < deadline, 'Finite phase reserve')
    signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())


def environment(route):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=route['GPU_uuid'], PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    env.pop('PYTHONHOME', None)
    if route['PYTHONPATH']: env['PYTHONPATH'] = os.pathsep.join(route['PYTHONPATH'])
    else: env.pop('PYTHONPATH', None)
    return env


def resources(custody, current):
    rss = 0
    for item in current:
        try:
            status = Path('/proc', str(item['pid']), 'status').read_text()
            require(custody.same(custody.identity(item['pid']), item), 'PID/birth resource sample')
            rss += next((int(row.split()[1]) * 1024 for row in status.splitlines() if row.startswith('VmRSS:')), 0)
        except FileNotFoundError:
            pass
    return rss, sum(used for pid, used in custody.gpu_rows() if pid in {item['pid'] for item in current})


def work_receipt(spec, cell, cfg, route, route_pins):
    done = read(cell / 'COMPLETE.json')  # This endpoint contains no metric values.
    m = 1 if spec['kind'] == 'single' else 4
    expected = dict(update_attempts=1100, completed_updates=1100, member_view_forwards=2200*m,
                    member_view_backwards=2200*m, Adam_steps=4400 if spec['kind'] == 'independent4' else 1100)
    require(done['complete'] is True and done['seed'] == spec['seed'] and done['arm'] == spec['kind']
            and done['steps'] == done['epochs'] == 1100 and done['all1100_and100_schedule'] is True
            and done['operator_work'] == expected and done['owner_context']['cell'] == spec
            and done['owner_context']['execution_source_commit'] == cfg['execution_source_commit']
            and done['owner_context']['runtime'] == route and done['owner_context']['roles'] == route_pins['roles']
            and done['owner_context']['source']['adapter_manifest_sha256'] == route_pins['math_manifest_sha256']
            and done['owner_context']['source']['original_driver_sha256'] == 'ea486cd59b33c1a5dcf9b0cf19672690e469e011f2b2db4781913d102625729a'
            and done['operator_binding']['physical_GPU_uuid'] == route['GPU_uuid']
            and done['operator_binding']['route_adapter_manifest_sha256'] == read(PHASE / 'pre_sigmoid_qk_host_routes_source_20261009_v1' / 'SEAL.json')['manifest_sha256']
            and done['operator_binding']['operator'] == spec['operator']
            and done['comparative_opening_performed'] is False, 'Exact original full1100/F/100/route completion')
    return dict(cell=spec, complete=True, steps=1100, local_epochs=100, operator_work=expected,
                route_id=cfg['route_id'], physical_GPU_uuid=route['GPU_uuid'],
                role_hashes={k: v['sha256'] for k, v in route_pins['roles'].items()},
                completion_sha256=sha(cell / 'COMPLETE.json'), bound_files=done['bound_files'], scores_read=False)


def run(*, release, release_sha256, authorized=False, launch_started=None):
    began = time.monotonic() if launch_started is None else launch_started
    cfg, pins, custody, routing, route, route_pins, cells = config(release, release_sha256, authorized)
    limit = cfg['resources']
    deadline = began + limit['owner_seconds']
    output = PHASE / cfg['output_relative']
    output.mkdir(exist_ok=False)
    for directory in ('handles', 'logs', 'costs'): (output / directory).mkdir()
    parent = custody.identity(os.getpid())
    child = saved = None
    known, exits, records = [], [], {}
    error = None
    success = False
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM): signal.signal(number, interrupted)
    try:
        timer(min(deadline - limit['cleanup_seconds'], began + limit['startup_seconds']))
        require(parent and parent['group'] == parent['session'] == parent['pid'], 'Detached own parent group')
        registered = min(time.monotonic() + 5, began + limit['startup_seconds'])
        while not custody.same(read(PHASE / cfg['activation_relative'] / 'LAUNCH.json').get('parent'), parent):
            require(time.monotonic() < registered, 'Finite parent launch registration')
            time.sleep(.01)
        if cfg['mode'] == 'seed_block': qualification(cfg, pins, custody, route)
        write(output / 'PARENT_OWNER.json', dict(parent=parent, route_id=cfg['route_id'], mode=cfg['mode'],
              release_sha256=release_sha256, execution_source_commit=cfg['execution_source_commit'],
              plan_sha256=cfg['plan']['sha256'] if cfg['plan'] else None, resources=limit, scores_read=False), True)
        jobs = cells if cells else [None]
        for index, spec in enumerate(jobs):
            name = 'qualification' if spec is None else '%02d__%s__%s' % (index, spec['operator'], spec['kind'])
            admitted = time.monotonic()
            timer(min(deadline - limit['cleanup_seconds'], admitted + limit['admission_seconds']))
            free = int(subprocess.check_output(['nvidia-smi', '--id=' + route['GPU_uuid'], '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 1024**2
            require(free >= limit['minimum_fresh_GPU_bytes'], 'Current free memory permits own bounded cell and coexecution')
            cell = output / ('qualification' if spec is None else spec['key'])
            cell.parent.mkdir(parents=True, exist_ok=True)
            require(not cell.exists(), 'Fresh no-retry cell output')
            handle = output / 'handles' / (name + '.json')
            value = dict(parent=parent, child=None, spec=spec, output=str(cell), cost=str(output / 'costs' / (name + '.json')),
                         release_path=str(Path(release).resolve()), release_sha256=release_sha256)
            write(handle, value, True)
            start = time.monotonic()
            reason, actions, known = None, [], []
            max_gpu = max_rss = 0
            active_end = min(start + limit['active_seconds'], deadline - limit['cleanup_seconds'] - limit['terminal_seconds'])
            timer(active_end + limit['cleanup_seconds'])
            with (output / 'logs' / (name + '.log')).open('xb') as log:
                child = subprocess.Popen([route['python'], '-B', str(HERE / 'lane.py'), '--worker', str(handle)],
                    cwd=route['repository'], env=environment(route), stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                saved = custody.identity(child.pid)
                require(saved and saved['group'] == saved['session'] == child.pid, 'Actual child PID/birth/group/session/boot')
                known = [saved]
                value['child'] = saved
                write(handle, value)
                try:
                    while child.poll() is None:
                        if time.monotonic() >= active_end:
                            reason = 'active_time_bound'
                            break
                        require(custody.same(custody.identity(child.pid), saved), 'Live actual owned child')
                        current = custody.members(saved)
                        known = list({(p['pid'], p['start_ticks']): p for p in known + current}.values())
                        rss, gpu = resources(custody, current)
                        max_rss, max_gpu = max(max_rss, rss), max(max_gpu, gpu)
                        if rss > limit['max_owned_RSS_bytes'] or gpu > limit['max_owned_GPU_bytes']:
                            reason = 'owned_resource_bound'
                            break
                        try: child.wait(timeout=min(2, max(.01, active_end-time.monotonic())))
                        except subprocess.TimeoutExpired: pass
                    if custody.members(saved):
                        reason = reason or 'owned_descendants_after_leader_exit'
                        custody.stop(child, saved, known, actions, min(deadline, time.monotonic()+limit['cleanup_seconds']))
                    else: child.wait(timeout=.01)
                except BaseException:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    if custody.members(saved): custody.stop(child, saved, known, actions, min(deadline, time.monotonic()+limit['cleanup_seconds']))
                    raise
            timer(min(deadline, time.monotonic() + limit['terminal_seconds']))
            absent = not custody.members(saved)
            no_cuda = not ({p['pid'] for p in known} & {pid for pid, _ in custody.gpu_rows()})
            terminal = dict(child=saved, witnessed_owned_members=known, exit_code=child.returncode,
                reaped=child.poll() is not None, group_absent=absent, no_owned_CUDA=no_cuda, reason=reason, signals=actions,
                active_and_cleanup_seconds=time.monotonic()-start, resource_admission_seconds=start-admitted,
                max_sampled_owned_GPU_bytes=max_gpu, max_sampled_owned_RSS_bytes=max_rss,
                costs_sha256=sha(value['cost']) if Path(value['cost']).exists() else None,
                partial_work_and_costs_retained=True, automatic_retry=False, scores_read=False)
            write(output / 'handles' / (name + '.EXIT.json'), terminal, True)
            exits.append(terminal)
            require(child.returncode == 0 and reason is None and absent and no_cuda, 'Successful clean owned endpoint')
            cost = read(value['cost'])
            require(cost['success'] is True and cost['peak_RSS_bytes'] <= limit['max_owned_RSS_bytes']
                    and cost['peak_cuda_reserved_bytes'] <= limit['max_owned_GPU_bytes'], 'Actual worker cost bounds')
            if spec is not None: records[spec['key']] = work_receipt(spec, cell, cfg, route, route_pins)
            else:
                report = read(cell / 'REPORT.json')
                require(report['complete'] is True and report['completed_updates'] == 24, 'Complete unscored V2 qualification')
            child = saved = None
            known = []
        success = True
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        cleanup_error = None
        if child is not None and saved is not None:
            try:
                if custody.members(saved): custody.stop(child, saved, known, actions, min(deadline, time.monotonic()+limit['cleanup_seconds']))
                else: child.wait(timeout=.01)
            except BaseException as caught:
                cleanup_error = dict(type=type(caught).__name__, message=str(caught))
                success = False
            if not any(custody.same(exit['child'], saved) for exit in exits):
                terminal = dict(child=saved, witnessed_owned_members=known, exit_code=child.poll(),
                    reaped=child.poll() is not None, group_absent=not custody.members(saved),
                    no_owned_CUDA=None, reason=error, signals=actions, cleanup_error=cleanup_error,
                    active_and_cleanup_seconds=time.monotonic()-start,
                    max_sampled_owned_GPU_bytes=max_gpu, max_sampled_owned_RSS_bytes=max_rss,
                    partial_work_and_costs_retained=True, automatic_retry=False, scores_read=False)
                try: terminal['no_owned_CUDA'] = not ({p['pid'] for p in known} & {pid for pid, _ in custody.gpu_rows()})
                except Exception as caught: terminal['CUDA_accounting_error'] = str(caught)
                write(output / 'handles' / (name + '.EXIT.json'), terminal, True)
                exits.append(terminal)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        write(output / 'CLOSURE.json', dict(complete=success, clean_owned_terminal=success,
            actual_owned_absence_attested=False, parent=parent, exits=exits, incomplete_child=saved,
            incomplete_witnesses=known, error=error, cleanup_error=cleanup_error, cells=records,
            mode=cfg['mode'], route_id=cfg['route_id'], physical_GPU_uuid=route['GPU_uuid'],
            plan_sha256=cfg['plan']['sha256'] if cfg['plan'] else None, execution_source_commit=cfg['execution_source_commit'],
            route_adapter_manifest_sha256=pins['routing']['manifest_sha256'], release_sha256=release_sha256,
            elapsed_seconds=time.monotonic()-began, CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
            peak_parent_RSS_bytes=usage.ru_maxrss*1024, scores_read=False, comparative_opening_authorized=False,
            automatic_retry=False, partial_work_and_costs_retained=True, lane_endpoint_count=len(records)), True)
    require(success, 'Lane failure retained; no automatic retry or partial opening')
    return dict(complete=True, closure=str(output / 'CLOSURE.json'), parent_absence_check_pending=True)


def worker(handle_path):
    started = time.monotonic()
    handle = read(handle_path)
    cfg = torch = None
    success, error = False, None
    try:
        pins, custody, routing = sources()
        register_end = time.monotonic() + 5
        while not handle['child']:
            require(time.monotonic() < register_end, 'Finite child registration')
            time.sleep(.01)
            handle = read(handle_path)
        actual, parent = custody.identity(os.getpid()), custody.identity(os.getppid())
        require(custody.same(actual, handle['child']) and custody.same(parent, handle['parent']), 'Exact child/parent custody')
        require(ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGTERM), 0, 0, 0) == 0, 'Own parent-death stop signal')
        require(os.getppid() == parent['pid'] and custody.same(custody.identity(parent['pid']), parent), 'Parent live through registration')
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM): signal.signal(number, interrupted)
        cfg, pins, custody, routing, route, route_pins, cells = config(handle['release_path'], handle['release_sha256'], True)
        timer(started + cfg['resources']['active_seconds'])
        routing.verify_route(route)
        routing.verify_roles(route, route_pins)
        require(Path(handle_path).resolve().parent == PHASE / cfg['output_relative'] / 'handles', 'Own registered handle path')
        import torch
        torch.cuda.set_per_process_memory_fraction(cfg['resources']['max_owned_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats(0)
        report = None
        if cfg['mode'] == 'qualify':
            require(handle['spec'] is None, 'One engineering child only')
            report = module(HERE / 'qualifier.py', '_lane_host_qualifier').run(cfg, pins, routing, route, route_pins, handle['output'])
        else:
            require(handle['spec'] in cells, 'Fixed assigned seed cell')
            qualification(cfg, pins, custody, route)
            cell_cfg = dict(cfg, seeds=[ASSIGNMENTS[cfg['route_id']]], operators=list(OPERATORS), kinds=list(KINDS), assigned_cells=cells)
            module(PHASE / pins['routing']['directory'] / 'route_cell.py', '_lane_final_route_cell').run_cell(
                route_id=cfg['route_id'], spec=handle['spec'], output=handle['output'], cfg=cell_cfg,
                release_sha256=handle['release_sha256'], authorized=True)
        success = True
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        allocated = reserved = None
        if torch is not None and torch.cuda.is_initialized():
            allocated, reserved = torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)
            # V2 resets allocator peaks per disposable cell; retain the maximum of all windows.
            if cfg['mode'] == 'qualify' and (Path(handle['output']) / 'REPORT.json').exists():
                def peaks(value):
                    if isinstance(value, dict):
                        yield (value.get('CUDA_peak_allocated_bytes', 0), value.get('CUDA_peak_reserved_bytes', 0))
                        for item in value.values(): yield from peaks(item)
                    elif isinstance(value, list):
                        for item in value: yield from peaks(item)
                windows = list(peaks(read(Path(handle['output']) / 'REPORT.json')))
                allocated = max([allocated] + [a for a, _ in windows])
                reserved = max([reserved] + [r for _, r in windows])
        write(handle['cost'], dict(success=success, error=error, spec=handle['spec'], seconds=time.monotonic()-started,
            CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime, peak_RSS_bytes=usage.ru_maxrss*1024,
            peak_cuda_allocated_bytes=allocated, peak_cuda_reserved_bytes=reserved,
            execution_source_commit=cfg['execution_source_commit'] if cfg else None,
            release_sha256=handle['release_sha256'], partial_work_retained=True, automatic_retry=False), True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--release-sha256')
    parser.add_argument('--authorized', action='store_true')
    parser.add_argument('--launch-started', type=float)
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    else: print(run(release=args.release, release_sha256=args.release_sha256, authorized=args.authorized, launch_started=args.launch_started))


if __name__ == '__main__': main()
