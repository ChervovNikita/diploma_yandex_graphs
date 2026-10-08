"""Detached normal77 full12 owner; reviewed run_fit is the only child supervisor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
POLICIES = ('alphaF', 'allJ', 'phiJ', 'relationJ')
STOP = threading.Event()
LOCK = threading.Lock()
ADMITTED = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(relative):
    value = Path(relative)
    require(not value.is_absolute() and '..' not in value.parts, 'Phase-relative custody required')
    path = (PHASE / value).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Path leaves project phase')
    return path


def phase_file(relative):
    path = inside(relative); require(path.is_file(), 'Required phase file absent')
    return path


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path), bytes=path.stat().st_size)


def bound(row):
    path = phase_file(row['path'])
    require(sha(path) == row['sha256'] and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Bound bytes changed: ' + row['path'])
    return path


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def module(row, name):
    spec = importlib.util.spec_from_file_location(name, bound(row))
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


def seal(row):
    path = bound(row)
    for item in read(path)['files']:
        file = (path.parent / item['path']).resolve()
        require(file.is_relative_to(path.parent), 'Source payload leaves sealed directory')
        bound(dict(item, path=str(file.relative_to(PHASE))))


def physical(pins):
    runtime = pins['runtime']
    require(platform.system() == 'Linux' and socket.gethostname() == runtime['hostname']
        and Path.cwd().resolve() == Path(runtime['repository']).resolve()
        and str(Path(sys.executable).absolute()) == runtime['python']
        and os.environ.get('PYTHONPATH', '') == '', 'Ordinary original77 host/repository/interpreter')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    require(inventory == runtime['physical_gpu_inventory'], 'Original two-GPU physical inventory')
    actual = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')}
    require(actual == {name: runtime[name] for name in actual}, 'Original numeric provider metadata')
    return actual


def source_check(pins):
    seal(pins['scientific_manifest'])
    for row in pins['scientific_bindings'].values():
        # Hash role archives only during actual admission; never interpret arrays or old anchors.
        bound(row)
    for key in ('public_manifest', 'constructor_manifest', 'attention_manifest', 'pool_adapter_manifest'):
        seal(pins['scientific_bindings'][key])
    bound(pins['ownership_helper']); bound(pins['run_fit_helper'])


def no_cuda(helper, pids):
    rows = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], 10)
    return not any(len(parts) > 1 and parts[1].strip().isdigit() and int(parts[1]) in pids
                   for parts in (row.split(',') for row in rows))


def not_live(helper, pid, ticks):
    observed = helper.identity(pid)
    return (observed is None or observed['start_ticks'] != ticks or observed['state'] == 'Z'), observed


def dependency_terminal(pins, helper):
    dependency = pins['GPU0_dependency']; expected = dependency['owner']
    gone, observed = not_live(helper, expected['pid'], expected['start_ticks'])
    if not gone:
        return None
    root = inside(dependency['output_directory'])
    if not (root / 'OWNER.json').is_file() or not (root / 'UNION_CLOSURE.json').is_file():
        return None
    owner = read(root / 'OWNER.json')['owner']
    require(owner['PID'] == expected['pid'] and owner['start_ticks'] == expected['start_ticks'], 'Exact unused2 owner custody')
    closure = read(root / 'UNION_CLOSURE.json')
    require(closure['schema'] == 'Wiki12-old10-plus-never-started2-union-closure-v1'
            and closure['automatic_retry'] is False and closure['quality_scores_read'] is False,
            'Original unused2 terminal family metadata')
    pids = {expected['pid']}; children = []
    for cell in dependency['cells']:
        start_path = root / 'logs' / (cell + '.CHILD_STARTED.json')
        exit_path = root / 'logs' / (cell + '.EXIT.json')
        if not start_path.exists():
            children.append(dict(cell_id=cell, launched=False)); continue
        if not exit_path.exists():
            return None
        started, receipt = read(start_path), read(exit_path)
        child = receipt['raw_identity_observation']
        require(child is not None and started['raw_identity_observation'] == child
                and receipt['terminal_wait_observed'] is True, 'Exact unused2 launched child wait/reap')
        gone, state = not_live(helper, child['PID'], child['start_ticks'])
        if not gone:
            return None
        pids.add(child['PID']); children.append(dict(cell_id=cell, launched=True, identity=child,
            exit_receipt=binding(exit_path), direct_wait_observed=True, current_identity=state))
    if not no_cuda(helper, pids):
        return None
    return dict(owner_receipt=binding(root / 'OWNER.json'), owner_identity=owner, current_owner_identity=observed,
        union_closure=binding(root / 'UNION_CLOSURE.json'), dependency_complete=closure['complete'],
        children=children, all_launched_children_terminal=True, exact_owner_not_live=True, owned_PIDs_no_CUDA_rows=True,
        failed_dependency_not_reclassified_as_success=True)


def lane_admission(index, cfg, pins, family_hash):
    # Root may add an unadmitted lane once. Every consumed hash binding stays immutable.
    with LOCK:
        path = phase_file(cfg['lane_admission_index'])
        cursor = read(path)
        require(cursor['schema'] == 'graph-relation-full12-root-lane-index-v1'
                and cursor['root_lane_binding_authorized'] is True
                and cursor['family_release_sha256'] == family_hash
                and cursor['controller_manifest_sha256'] == cfg['controller_manifest_sha256']
                and set(cursor['lanes']) == {'0', '1'}, 'Exact root lane-binding index')
        for key, consumed in ADMITTED.items():
            require(cursor['lanes'][key] == consumed, 'Consumed lane admission changed')
        lane_row = cursor['lanes'][str(index)]
        if lane_row is None:
            return None
        path = bound(lane_row); admission = read(path); lane = pins['lanes'][str(index)]
        require(admission['schema'] == 'graph-relation-full12-root-lane-admission-v1'
                and all(admission.get(flag) is True for flag in ('enabled', 'root_fit_launch_authorized',
                    'source_review_approved', 'native_qualification_approved'))
                and admission['controller_manifest_sha256'] == cfg['controller_manifest_sha256']
                and admission['source_manifest_sha256'] == pins['scientific_manifest']['sha256']
                and admission['lane'] == index and admission['physical_gpu_uuid'] == lane['physical_gpu_uuid']
                and admission['TEST_access'] is False and admission['automatic_retry'] is False,
                'Exact separately enabled root lane admission')
        qualification = read(bound(admission['native_qualification']))
        require(qualification['complete'] is True and qualification['source_static_only'] is False
                and qualification['source_manifest_sha256'] == pins['scientific_manifest']['sha256']
                and qualification['physical_gpu_uuid'] == lane['physical_gpu_uuid']
                and qualification['policies'] == list(POLICIES)
                and qualification['real_complete_TRAIN_updates'] == 8
                and qualification['VALID_scores_read'] is False and qualification['TEST_access'] is False,
                'Actual same-GPU native qualification; no inherited fixture credit')
        cells = admission['cells']
        require([row['cell_id'] for row in cells] == lane['cells'], 'Frozen lane roster/order')
        mutable = {'enabled', 'root_execution_authorized', 'source_review_approved', 'native_qualification_approved',
                   'source_manifest_sha256', 'native_qualification', 'physical_gpu_uuid', 'GPU_assignment_requires_root_freeze'}
        for cell_row in cells:
            released = read(bound(cell_row)); preview = read(bound(pins['previews'][cell_row['cell_id']]))
            require(all(released[key] == value for key, value in preview.items() if key not in mutable),
                    'Root release changed fixed science/initializer/selector/limits or pending-union scope')
            require(all(released.get(flag) is True for flag in ('enabled', 'root_execution_authorized', 'source_review_approved', 'native_qualification_approved'))
                    and released['source_manifest_sha256'] == pins['scientific_manifest']['sha256']
                    and released['native_qualification'] == admission['native_qualification']
                    and released['physical_gpu_uuid'] == lane['physical_gpu_uuid']
                    and released['GPU_assignment_requires_root_freeze'] is False
                    and released['anchor_reuse_authorized'] is False and released['comparative_opening_authorized'] is False
                    and released['original12_union_closure'] is None and released['original12_union_pending'] is True,
                    'Training-only root release; no original union/anchor numerical opening')
        ADMITTED[str(index)] = lane_row
        return dict(binding=lane_row, value=admission, index_binding=binding(phase_file(cfg['lane_admission_index'])))


def lane(index, cfg, pins, family_hash, output, deadline):
    helper = module(pins['ownership_helper'], '_relation12_owned_' + str(index))
    supervisor = module(pins['run_fit_helper'], '_relation12_runfit_' + str(index))
    plan = pins['lanes'][str(index)]; gpu = plan['physical_gpu_uuid']; helper.GPU = gpu
    source = bound(pins['scientific_manifest']).parent
    context = SimpleNamespace(REPO=Path(pins['runtime']['repository']), SOURCE=source,
        SOURCE_SHA=pins['scientific_manifest']['sha256'], GPU_UUID=gpu,
        GPU_UUIDS=tuple(pins['runtime']['physical_gpu_inventory']), phase_file=phase_file,
        physical_host=lambda: physical(pins), sha=sha, write=write)
    records = [dict(cell_id=cell, status='unlaunched', physical_gpu_uuid=gpu, automatic_retry=False) for cell in plan['cells']]
    schedule_used = 0.; completed = []; fatal = None; admitted = None
    scheduling_started = None; fit_call_started = None; current = None
    try:
        for record in records:
            current = record; began = time.monotonic(); scheduling_started = began; ready = None
            while ready is None:
                require(not STOP.is_set(), 'Owner stop requested; no next fit')
                remaining = min(pins['scheduling_seconds_per_lane'] - schedule_used - (time.monotonic() - began),
                    deadline - time.monotonic() - pins['reserved_next_cell_seconds'])
                require(remaining > 0, 'Finite lane scheduling/family window exhausted before next whole fit')
                physical(pins)
                candidate = lane_admission(index, cfg, pins, family_hash)
                dependency = dependency_terminal(pins, helper) if index == 0 else dict(required=False)
                rows = helper.query(['--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], 10)
                available = [row.split(',')[1].strip() for row in rows if row.split(',')[0].strip() == gpu]
                require(len(available) == 1 and available[0].isdigit(), 'Exact lane GPU telemetry')
                free = int(available[0]) * 1024**2
                write(output / ('LANE_' + str(index) + '_SCHEDULING.json'), dict(UTC=helper.now(), next_cell=record['cell_id'],
                    root_lane_admitted=candidate is not None, dependency_terminal=dependency,
                    fresh_free_GPU_bytes=free, required_free_GPU_bytes=pins['resource_limits']['minimum_fresh_GPU_free_bytes'],
                    cumulative_scheduling_seconds=schedule_used + time.monotonic() - began,
                    scheduling_seconds_remaining=max(0, remaining), scientific_child_started=False, scores_read=False))
                if candidate is not None and dependency is not None and free >= pins['resource_limits']['minimum_fresh_GPU_free_bytes']:
                    ready = candidate; admitted = candidate; break
                STOP.wait(min(15, max(.01, remaining)))
            schedule_used += time.monotonic() - began
            scheduling_started = None
            record.update(scheduling_seconds=time.monotonic() - began, status='admitted', lane_admission=ready['binding'])
            write(output / ('LANE_' + str(index) + '_ADMISSION.json'), ready)
            row = next(row for row in ready['value']['cells'] if row['cell_id'] == record['cell_id'])
            release_path = bound(row); release = read(release_path); target = inside(release['output'])
            require(not target.exists(), 'Fresh fit output; no completed-cell retry')
            source_check(pins); physical(pins)
            require(not STOP.is_set() and deadline - time.monotonic() >= pins['reserved_next_cell_seconds'], 'Family envelope cannot cover next full fit/admission')
            entry = dict(cell_id=record['cell_id'], job_relative=row['path'], job_sha256=row['sha256'],
                hard_seconds=32390, argv=[pins['runtime']['python'], '-B', str(bound(pins['train_program'])),
                    '--release', str(release_path), '--release-sha256', row['sha256']])
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
            environment.pop('PYTHONHOME', None)
            record.update(status='running', release=binding(release_path), argv=entry['argv'], native_qualification=ready['value']['native_qualification'])
            write(output / ('LANE_' + str(index) + '_PROGRESS.json'), dict(rows=records, scores_read=False))
            fit_call_started = time.monotonic()
            receipt = supervisor.run_fit(helper, output, entry, {'resource_limits': pins['resource_limits']}, environment, target, context)
            record['inclusive_run_fit_seconds'] = time.monotonic() - fit_call_started; fit_call_started = None
            preflight_path = output / 'logs' / (record['cell_id'] + '.PREFLIGHT.json')
            record.update(original_resource_admission=binding(preflight_path),
                          original_resource_wait_seconds=read(preflight_path)['elapsed_seconds'])
            child = receipt['raw_identity_observation']
            absent = child is not None and helper.identity(child['PID']) is None
            clean = child is not None and no_cuda(helper, {child['PID']})
            record.update(exit_receipt=binding(output / 'logs' / (record['cell_id'] + '.EXIT.json')),
                actual_exit_receipt=receipt, child_identity=child, child_absent=absent, child_no_CUDA_rows=clean,
                direct_wait_and_reap_observed=receipt['terminal_wait_observed'], fit_and_cleanup_seconds=receipt['elapsed_seconds'])
            require(receipt['exit_code'] == 0 and receipt['reason'] is None and receipt['signals_sent'] == []
                    and receipt['terminal_wait_observed'] is True and absent and clean and receipt['elapsed_seconds'] <= 32400,
                    'Original owned fit failed terminal/resource/time custody; preserve, no retry')
            require(not (target / 'FAILURE.json').exists(), 'Scientific failure retained')
            endpoint = read(target / 'COMPLETE.json'); descriptor = endpoint['graph_relation_credit']
            # Whitelist completion/source/work fields; no validation trace, accuracy or TRAIN-loss readout.
            work = dict(shadow_member_forwards=8800, replay_member_forwards=8800, output_cotangent_collections=2200,
                member_reverse_collections=17600, optimizer_bank_updates=1100, exact_member_RNG_endpoint_checks=1100)
            require(endpoint['complete'] is True and endpoint['epochs'] == endpoint['steps'] == 1100
                and endpoint['seed'] == release['seed'] and endpoint['arm'] == endpoint['method_identity'] == 'graph_relation_credit__' + release['policy']
                and endpoint['TEST_scoring'] is False and endpoint['risk_beta'] == .5
                and endpoint['source_manifest_sha256'] == pins['scientific_manifest']['sha256']
                and descriptor['new_method_source_manifest_sha256'] == pins['scientific_manifest']['sha256']
                and descriptor['train_program_sha256'] == pins['train_program']['sha256']
                and descriptor['partition']['policy'] == release['policy'] and descriptor['risk_beta'] == .5
                and descriptor['auxiliary_weight'] == 0. and descriptor['effective_model_contrastive_flag'] is False
                and descriptor['work'] == work and descriptor['validation_work']['evaluations'] == 1100
                and descriptor['validation_work']['member_forwards'] == 4400, 'Exact complete policy/source/two-block work')
            selected = target / 'selected.pt'; require(sha(selected) == endpoint['selected_sha256'], 'Original selected byte custody; no deserialization')
            record.update(status='complete', complete=True, completion=binding(target / 'COMPLETE.json'),
                selected_checkpoint=binding(selected), execution_accounting=work, development_member_forwards=4400,
                source_manifest=pins['scientific_manifest'], train_program=pins['train_program'], scores_read=False)
            completed.append(record['cell_id'])
            write(output / ('LANE_' + str(index) + '_PROGRESS.json'), dict(rows=records, completed=completed, scores_read=False))
    except (Exception, KeyboardInterrupt) as error:
        fatal = dict(error_type=type(error).__name__, error=str(error), automatic_retry=False)
        if scheduling_started is not None:
            elapsed = time.monotonic() - scheduling_started; schedule_used += elapsed
            if current is not None:
                current.update(status='retained_scheduling_failure', scheduling_seconds=elapsed, failure=fatal)
        if current is not None:
            if fit_call_started is not None:
                current['inclusive_run_fit_seconds'] = time.monotonic() - fit_call_started
            for suffix in ('PREFLIGHT', 'PROCESS', 'CHILD_STARTED', 'EXIT'):
                artifact = output / 'logs' / (current['cell_id'] + '.' + suffix + '.json')
                if artifact.is_file():
                    current.setdefault('retained_failure_custody', {})[suffix] = binding(artifact)
        for record in records:
            if record['status'] in ('running', 'admitted'):
                record.update(status='retained_failure', failure=fatal)
    finally:
        complete = all(row['status'] == 'complete' for row in records) and fatal is None
        closure = dict(complete=complete, lane=index, physical_gpu_uuid=gpu, rows=records, completed=completed,
            cumulative_scheduling_seconds=schedule_used, failure=fatal, lane_admission=admitted,
            automatic_retry=False, scores_read=False, TEST_access=False)
        write(output / ('LANE_' + str(index) + '_CLOSURE.json'), closure)
    return closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); os.umask(0o077)
    began = time.monotonic(); usage = resource.getrusage(resource.RUSAGE_SELF)
    require(sha(args.release) == args.release_sha256, 'Exact separate root family release')
    cfg = read(args.release)
    require(cfg['schema'] == 'graph-relation-full12-root-family-release-v1'
        and all(cfg.get(key) is True for key in ('enabled', 'root_full12_launch_authorized', 'source_review_approved',
            'finite_family_resource_admission_confirmed', 'future_root_lane_admissions_authorized')),
        'Source preparation is inactive pending root release')
    require(cfg['TEST_access'] is False and cfg['automatic_retry'] is False
        and cfg['anchor_reuse_authorized'] is False and cfg['comparative_opening_authorized'] is False, 'Training-only family scope')
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    seal(dict(path=str((HERE / 'MANIFEST.json').relative_to(PHASE)), sha256=cfg['controller_manifest_sha256']))
    providers = physical(pins); source_check(pins)
    output = inside(cfg['owner_output_directory']); require(not output.exists() and output.parent.is_dir(), 'Fresh owner family; no resume/retry')
    for row in pins['previews'].values():
        require(not inside(read(bound(row))['output']).exists(), 'No previously trained/completed cell admitted')
    helper = module(pins['ownership_helper'], '_relation12_parent_identity')
    identity = helper.identity(os.getpid())
    require(identity is not None and identity['PID'] == identity['pgid'] == identity['sid'], 'Fresh detached root owner session')
    output.mkdir(mode=0o700); (output / 'logs').mkdir(mode=0o700)
    write(output / 'PARENT_OWNER.json', dict(identity=identity, release=binding(args.release), providers=providers,
        source_manifest=pins['scientific_manifest'], controller_manifest_sha256=cfg['controller_manifest_sha256'],
        family_hard_seconds=pins['family_hard_seconds'], lane_scheduling_seconds=pins['scheduling_seconds_per_lane'],
        anchor_reuse_authorized=False, comparative_opening_authorized=False, scores_read=False))
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: STOP.set())
    # No interruption/signaling of active run_fit calls; each retains its original finite owned bounds.
    timer = threading.Timer(max(.01, pins['family_hard_seconds'] - (time.monotonic() - began)), STOP.set)
    timer.daemon = True; timer.start(); results = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, index, cfg, pins, args.release_sha256, output,
                                   began + pins['family_hard_seconds']) for index in (0, 1)]
            results = [future.result() for future in futures]
    finally:
        timer.cancel(); now = resource.getrusage(resource.RUSAGE_SELF)
        closure = dict(schema='graph-relation-full12-normal77-family-closure-v1', complete=len(results) == 2 and all(row['complete'] for row in results),
            fixed_scientific_cells=12, lane_results=results, parent_identity=identity, root_release=binding(args.release),
            source_manifest=pins['scientific_manifest'], controller_manifest_sha256=cfg['controller_manifest_sha256'],
            inclusive_family_seconds=time.monotonic() - began, family_hard_seconds=pins['family_hard_seconds'],
            family_hard_cap_exceeded=time.monotonic() - began > pins['family_hard_seconds'],
            CPU_user_seconds=now.ru_utime - usage.ru_utime, CPU_system_seconds=now.ru_stime - usage.ru_stime,
            peak_RSS_bytes=int(now.ru_maxrss * 1024), stopped=STOP.is_set(), automatic_retry=False,
            original_anchors_numerically_read=False, comparative_outcomes_opened=False,
            anchor_reuse_authorized=False, comparative_opening_authorized=False, TEST_access=False,
            namespace_or_mount_changes=False, UTC=datetime.now(timezone.utc).isoformat())
        if closure['family_hard_cap_exceeded']:
            closure['complete'] = False
        write(output / 'FAMILY_CLOSURE.json', closure)
    print(json.dumps(dict(family_terminal=True, complete=closure['complete'], fixed_cells=12, scores_read=False)), flush=True)
    return 0 if closure['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
