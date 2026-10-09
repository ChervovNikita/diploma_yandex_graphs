"""Fixed once-only phase1 invocation of the reviewed original-route collector."""
import argparse
import importlib.util
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def argv(pid):
    return Path('/proc', str(pid), 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')


def context(args):
    # All imports below are the existing stdlib metadata/ownership sources.
    import json
    pins = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    gate_path = PHASE / pins['gate_program']['path']
    import hashlib
    assert gate_path.stat().st_size == pins['gate_program']['bytes']
    assert hashlib.sha256(gate_path.read_bytes()).hexdigest() == pins['gate_program']['sha256']
    qk = load(gate_path, '_phase1_exact_original_gate')
    scientific = qk.read(gate_path.parent / 'SOURCE_BINDINGS.json')
    g = qk.helpers(scientific)
    g.verify(pins['readout_manifest'])
    g.verify(g.binding(HERE / 'MANIFEST.json'))
    release = Path(args.release).resolve(strict=True)
    assert release.is_relative_to(PHASE) and g.sha(release) == args.release_sha256
    cfg = g.read(release)
    assert all(cfg[k] is True for k in pins['existing_release_flags'])
    assert cfg['collection_phase'] == 'native_baselines' and cfg['maximum_member_forwards'] == 9
    assert cfg['maximum_route_member_forwards'] == 36
    assert cfg['execution_source_commit'] == scientific['source_execution_commit']
    assert cfg['readout_manifest_sha256'] == pins['readout_manifest']['sha256']
    assert all(cfg[k] is False for k in ('training', 'TEST_access', 'automatic_retry', 'reselection', 'calibration'))
    route = scientific['routes'][cfg['readout_route_id']]
    assert socket.gethostname() == route['hostname'] and Path.cwd().resolve() == Path(route['repository'])
    assert PHASE == Path(route['phase'])
    supervision = g.read(g.bound(cfg['external_supervision']))
    entry = g.bound(supervision['entry_program'])
    assert supervision['entry_program'] == pins['collector_program']
    assert supervision['enabled'] is True and supervision['finite_owned_bound_confirmed'] is True
    assert supervision['parent_invocation_program'] == g.binding(Path(__file__))
    assert supervision['custody_program'] == pins['custody_program']
    assert supervision['original_resource_helper'] == pins['lane_program']
    assert supervision['original_environment_helper'] == pins['lane_program']
    assert supervision['physical_GPU_uuid'] == route['GPU_uuid'] and supervision['route_id'] == cfg['readout_route_id']
    assert supervision['argv_prefix'] == [route['python'], '-B', str(entry)]
    assert supervision['release_argument_path'] == str(release)
    assert (supervision['active_seconds'], supervision['cleanup_seconds'], supervision['hard_seconds']) == (3600, 15, 3615)
    assert (supervision['owned_GPU_bytes'], supervision['owned_RSS_bytes'], supervision['maximum_output_bytes']) == (25769803776, 8589934592, 4294967296)
    custody = g.module(g.bound(pins['custody_program']), '_phase1_existing_owned')
    lane = qk.source_lane(g, scientific)
    root = g.inside(pins['supervision_root'] + '/' + cfg['readout_route_id'])
    output = g.inside(cfg['output_directory'])
    command = supervision['argv_prefix'] + ['--release', str(release), '--release-sha256', args.release_sha256]
    return pins, qk, scientific, g, cfg, route, custody, lane, root, output, release, supervision, command


def launch(args, ctx):
    pins, qk, scientific, g, cfg, route, custody, lane, root, output, release, limits, command = ctx
    assert not output.exists() and not root.exists()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    custody.write(root / 'LAUNCH.json', dict(parent=None, reserved=True, release=g.binding(release), automatic_retry=False), True)
    parent_argv = ['/usr/bin/python3', '-I', '-S', '-B', str(Path(__file__).resolve()),
        '--mode', 'supervise', '--release', str(release), '--release-sha256', args.release_sha256]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    environment.pop('PYTHONHOME', None); environment.pop('PYTHONPATH', None)
    with (root / 'PARENT.log').open('xb') as log:
        parent = subprocess.Popen(parent_argv, cwd=route['repository'], env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    saved = custody.identity(parent.pid)
    assert saved and saved['pid'] == saved['group'] == saved['session']
    custody.write(root / 'LAUNCH.json', dict(parent=saved, actual_parent_argv=argv(parent.pid), requested_parent_argv=parent_argv,
        root_launcher_identity=custody.identity(os.getpid()), actual_root_launcher_argv=argv(os.getpid()),
        root_launcher_entry_program=g.binding(Path(__file__)),
        parent_entry_program=g.binding(Path(__file__)), collector_program=pins['collector_program'],
        release=g.binding(release), detached_parent_direct_wait_observed=False, detached_parent_OS_exit=None,
        reserved=True, automatic_retry=False, numerical_execution_by_launcher=False))
    print(str(root / 'LAUNCH.json'))


def supervise(args, ctx):
    pins, qk, scientific, g, cfg, route, custody, lane, root, output, release, limits, command = ctx
    began = args.started; usage = args.usage_start
    parent = custody.identity(os.getpid())
    assert parent and parent['pid'] == parent['group'] == parent['session']
    registration_end = time.monotonic() + 5
    while not custody.same(g.read(root / 'LAUNCH.json').get('parent'), parent):
        assert time.monotonic() < registration_end
        time.sleep(.01)
    assert not output.exists()
    common = dict(route_id=cfg['readout_route_id'], collection_phase='native_baselines',
        source_program=pins['collector_program'], readout_manifest=pins['readout_manifest'],
        release=g.binding(release), custody_program=pins['custody_program'])
    custody.write(root / 'PARENT_OWNER.json', dict(**common, parent_identity=parent, actual_parent_argv=argv(os.getpid()),
        actual_parent_executable=str(Path(sys.executable).resolve()),
        parent_entry_program=g.binding(Path(__file__)), parent_source_manifest=g.binding(HERE / 'MANIFEST.json'),
        original_resource_helper=pins['lane_program'], original_environment_helper=pins['lane_program'],
        invocation_Git_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=route['repository'], text=True, timeout=10).strip(),
        scientific_execution_source_commit=cfg['execution_source_commit'], detached_parent_direct_wait_observed=False,
        detached_parent_OS_exit=None, numerical_execution_by_parent=False), True)
    child = saved = None; known = []; actions = []; error = cleanup_error = reason = None
    max_gpu = max_rss = 0; direct_wait = False; start = None; elapsed = admission = None
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM): signal.signal(number, lane.interrupted)
    try:
        admitted = time.monotonic()
        free = int(subprocess.check_output(['nvidia-smi', '--id=' + route['GPU_uuid'], '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 1024**2
        assert free >= limits['owned_GPU_bytes']
        start = time.monotonic(); admission = start - admitted
        lane.timer(start + limits['active_seconds'])
        with (root / 'COLLECTOR.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=route['repository'], env=lane.environment(route), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            saved = custody.identity(child.pid)
            assert saved and saved['pid'] == saved['group'] == saved['session'] and saved['boot_id'] == parent['boot_id']
            known = [saved]
            actual = argv(child.pid)
            assert actual == command and int(Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()[1]) == parent['pid']
            executable = str(Path('/proc', str(child.pid), 'exe').resolve(strict=True))
            assert executable == str(Path(route['python']).resolve(strict=True))
            assert custody.same(custody.identity(parent['pid']), parent) and custody.same(custody.identity(child.pid), saved)
            custody.write(root / 'PROCESS_WITNESS.json', dict(**common, schema='qk36-phase-collector-process-witness-v1',
                root_observed=True, observation_scope='Actual /proc observations by this root-invoked metadata parent',
                observation_program=g.binding(Path(__file__)), parent_identity=parent, child_identity=saved,
                actual_parent_argv=argv(parent['pid']), actual_child_argv=actual, actual_child_ppid=parent['pid'],
                actual_child_executable=executable, parent_entry_program=g.binding(Path(__file__)),
                original_resource_helper=pins['lane_program']), True)
            while child.poll() is None:
                if time.monotonic() >= start + limits['active_seconds']:
                    reason = 'active_time_bound'; break
                if not custody.same(custody.identity(child.pid), saved):
                    if child.poll() is not None: break
                    raise RuntimeError('Live actual collector PID/birth custody')
                current = custody.members(saved)
                known = list({tuple(p[k] for k in custody.CUSTODY): p for p in known + current}.values())
                rss, gpu = lane.resources(custody, [parent, *current])
                max_rss, max_gpu = max(max_rss, rss), max(max_gpu, gpu)
                stored = 0
                for directory in (root, output):
                    if directory.is_dir():
                        for path in directory.rglob('*'):
                            try:
                                if path.is_file(): stored += path.stat().st_size
                            except FileNotFoundError: pass
                if rss > limits['owned_RSS_bytes'] or gpu > limits['owned_GPU_bytes'] or stored > limits['maximum_output_bytes']:
                    reason = 'owned_resource_or_output_bound'; break
                try: child.wait(timeout=min(2, max(.01, start + limits['active_seconds'] - time.monotonic())))
                except subprocess.TimeoutExpired: pass
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught)); reason = reason or 'owned_parent_stop_or_failure'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM): signal.signal(number, signal.SIG_IGN)
        if child is not None and saved is not None:
            try:
                if custody.members(saved):
                    reason = reason or 'owned_descendants_after_leader_exit'
                    current = custody.members(saved)
                    known = list({tuple(p[k] for k in custody.CUSTODY): p for p in known + current}.values())
                    custody.stop(child, saved, known, actions, min(start + limits['hard_seconds'], time.monotonic() + limits['cleanup_seconds']))
                child.wait(timeout=.01); direct_wait = True
            except BaseException as caught: cleanup_error = dict(type=type(caught).__name__, message=str(caught))
            elapsed = time.monotonic() - start
            terminal = dict(child=saved, witnessed_owned_members=known, exit_code=child.returncode,
                reaped=direct_wait, actual_direct_wait_observed=direct_wait, group_absent=not custody.members(saved),
                no_owned_CUDA=None, reason=reason, signals=actions, error=error, cleanup_error=cleanup_error,
                active_and_cleanup_seconds=elapsed, resource_admission_seconds=admission,
                max_sampled_owned_GPU_bytes=max_gpu, max_sampled_owned_RSS_bytes=max_rss,
                partial_work_and_costs_retained=True, automatic_retry=False, scores_read=False,
                source_program=pins['collector_program'], parent_invocation_program=g.binding(Path(__file__)),
                release=g.binding(release), original_resource_helper=pins['lane_program'])
            try: terminal['no_owned_CUDA'] = not ({p['pid'] for p in known} & {pid for pid, _ in custody.gpu_rows()})
            except BaseException as caught: terminal['CUDA_accounting_error'] = dict(type=type(caught).__name__, message=str(caught))
            custody.write(root / 'CHILD.EXIT.json', terminal, True)
        end = resource.getrusage(resource.RUSAGE_SELF)
        custody.write(root / 'PARENT_FINISH.json', dict(**common, parent_identity=parent, child_identity=saved,
            error=error, cleanup_error=cleanup_error, actual_child_exit_receipt=g.binding(root / 'CHILD.EXIT.json') if (root / 'CHILD.EXIT.json').is_file() else None,
            inclusive_parent_seconds=time.monotonic() - began, CPU_user_seconds=end.ru_utime - usage.ru_utime,
            CPU_system_seconds=end.ru_stime - usage.ru_stime, peak_parent_RSS_bytes=end.ru_maxrss * 1024,
            parent_cost_scope='Main entry, source/config metadata, registration, child active/cleanup and retained receipts; interpreter/module startup unavailable separately',
            detached_parent_direct_wait_observed=False, detached_parent_OS_exit=None,
            separated_cleanup_seconds=None, separated_cleanup_duration_unavailable=True,
            partial_work_and_costs_retained=True, automatic_retry=False, scores_read=False), True)
    assert error is None and cleanup_error is None and direct_wait and child.returncode == 0 and reason is None


def attest(args, ctx):
    pins, qk, scientific, g, cfg, route, custody, lane, root, output, release, limits, command = ctx
    began = args.started
    result = dict(complete=False, failure=None, detached_parent_direct_wait_observed=False, detached_parent_OS_exit=None,
        numerical_imports=False, collector_launch=False, automatic_retry=False)
    try:
        owner = g.read(root / 'PARENT_OWNER.json'); actual = g.read(root / 'CHILD.EXIT.json')
        witness = g.read(root / 'PROCESS_WITNESS.json'); parent, child = owner['parent_identity'], actual['child']
        assert witness['actual_child_argv'] == command and qk.same_process(witness['parent_identity'], parent) and qk.same_process(witness['child_identity'], child)
        known = [parent, *actual['witnessed_owned_members']]
        absence = dict(schema='qk36-phase-collector-owned-absence-v1', root_observed=True, route_id=cfg['readout_route_id'],
            collection_phase='native_baselines', source_program=pins['collector_program'], release=g.binding(release),
            custody_program=pins['custody_program'], parent_identity=parent, child_identity=child,
            observation_program=g.binding(Path(__file__)), actual_observer_identity=custody.identity(os.getpid()), actual_observer_argv=argv(os.getpid()),
            identity_observations=[dict(saved=p, current=custody.identity(p['pid'])) for p in known],
            group_observations=[dict(saved=p, current_members=custody.members(p)) for p in (parent, child)],
            current_CUDA_pids=[pid for pid, _ in custody.gpu_rows()])
        custody.write(root / 'OWNED_ABSENCE_OBSERVATION.json', absence, True)
        assert all(not custody.same(row['current'], row['saved']) for row in absence['identity_observations'])
        assert all(row['current_members'] == [] for row in absence['group_observations'])
        assert not ({p['pid'] for p in known} & set(absence['current_CUDA_pids']))
        collection_row, cost_row = g.binding(output / 'compact/COLLECTION.json'), g.binding(output / 'compact/COST.json')
        exit_row = g.binding(root / 'CHILD.EXIT.json')
        terminal = dict(schema='qk36-original-route-reader-terminal-v2', source_program=pins['collector_program'],
            route_id=cfg['readout_route_id'], collection_phase='native_baselines', release=g.binding(release),
            parent_identity=parent, child_identity=child, collection=collection_row, cost=cost_row,
            actual_wait_and_reap=actual['reaped'], actual_exit_code=actual['exit_code'],
            actual_child_exit_receipt=exit_row, actual_process_witness=g.binding(root / 'PROCESS_WITNESS.json'),
            actual_owned_absence_receipt=g.binding(root / 'OWNED_ABSENCE_OBSERVATION.json'),
            cleanup_cost_and_failures=exit_row, separated_cleanup_seconds=None, separated_cleanup_duration_unavailable=True,
            owned_group_absent=True, no_owned_CUDA=True, parent_direct_wait_and_reap_observed=False,
            parent_actual_exit_code=None, actual_parent_exit_receipt=None, unknown_fields_preserved=True)
        custody.write(root / 'TERMINAL.json', terminal, True)
        # This unchanged stdlib validator checks the complete nine-call phase1,
        # exact release/source/process/EXIT/cost/absence contract. It decodes no arrays.
        proof = qk.phase_terminal(g, scientific, g.binding(root / 'TERMINAL.json'), cfg['readout_route_id'], 'native_baselines',
            collection_row, cost_row, g.read(g.bound(collection_row)), g.read(g.bound(cost_row)), cfg['readout_manifest_sha256'])
        result.update(complete=True, actual_phase1_owner_terminal_custody=g.binding(root / 'TERMINAL.json'),
            phase_owner_succeeded=proof['phase_owner_succeeded'])
        print(str(root / 'TERMINAL.json'))
    except BaseException as caught:
        result['failure'] = dict(type=type(caught).__name__, message=str(caught))
        raise
    finally:
        result['metadata_seconds'] = time.monotonic() - began
        custody.write(root / 'ATTESTATION_RESULT.json', result, True)


def main():
    began = time.monotonic(); usage = resource.getrusage(resource.RUSAGE_SELF)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('launch', 'supervise', 'attest'), default='launch')
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    args.started = began; args.usage_start = usage
    ctx = context(args)
    {'launch': launch, 'supervise': supervise, 'attest': attest}[args.mode](args, ctx)


if __name__ == '__main__': main()
