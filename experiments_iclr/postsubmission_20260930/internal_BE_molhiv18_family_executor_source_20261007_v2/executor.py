"""Disabled serial Mol18 executor and ordinary outer reap; stdlib only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
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
REPO = PHASE.parents[1]
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CONDITIONS = ('single', 'independent4', 'O', 'I', 'P', 'G')
SEEDS = (7101, 7203, 7307)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def inside(relative):
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Phase-relative path required')
    path = (PHASE / relative).resolve()
    require(path != PHASE.resolve() and path.is_relative_to(PHASE.resolve()), 'Project phase path required')
    return path


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE.resolve())), sha256=sha(path), bytes=path.stat().st_size)


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Exact bound file changed')
    return path


def proc(pid):
    try:
        fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), group=int(fields[2]), session=int(fields[3]), start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_live(group):
    result = []
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            try:
                row = proc(int(path.name))
            except PermissionError:
                continue
            if row and row['group'] == group and row['state'] != 'Z':
                result.append(row['pid'])
    return result


def owned_leader(handle):
    actual = proc(handle['pid'])
    return bool(actual and actual['pid'] == handle['pid'] and actual['start_ticks'] == handle['start_ticks']
        and actual['group'] == handle['group'] == handle['pid']
        and actual['session'] == handle['session'] == handle['pid'])


def stop_owned(child, handle, deadline):
    # Numeric PGID alone is never custody after the saved leader disappears.
    if group_live(child.pid):
        if handle['pid'] != child.pid or not owned_leader(handle):
            return False
        os.killpg(child.pid, signal.SIGTERM)
        grace_deadline = min(deadline, time.monotonic() + 5)
        while group_live(child.pid) and time.monotonic() < grace_deadline:
            child.poll(); time.sleep(.1)
        if group_live(child.pid):
            if handle['pid'] != child.pid or not owned_leader(handle):
                return False
            os.killpg(child.pid, signal.SIGKILL)
    if child.poll() is None:
        try:
            child.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    return child.poll() is not None and not group_live(child.pid)


def interrupted(signum, frame):
    raise TimeoutError('Owned family process received signal ' + str(signum))


def prepare(path, digest):
    require(sys.platform == 'linux' and socket.gethostname() == 'anogena-2-0'
            and Path.cwd().resolve() == REPO.resolve(), 'Authorized normal Linux project host required')
    rows = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=5).splitlines()
    require(rows == [GPU], 'Sole authorized physical GPU required')
    require(Path(path).resolve().is_relative_to(PHASE.resolve()), 'Root release inside the authorized phase required')
    require(sha(path) == digest, 'Exact root release required'); cfg = read(path)
    require(cfg.get('schema') == 'internal-be-molhiv18-family-release-v1' and cfg.get('enabled') is True
            and cfg.get('root_scientific_launch_authorized') is True and cfg.get('source_review_approved') is True
            and cfg.get('same_host_GPU_readiness_confirmed') is True and cfg.get('allocation_exclusive_for_family') is True
            and cfg.get('TEST_access') is False and cfg.get('automatic_retry') is False, 'Disabled pending exact root family admission')
    require(cfg['executor_manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Reviewed executor source differs')
    for row in read(HERE / 'MANIFEST.json')['files']:
        item = HERE / row['path']; require(sha(item) == row['sha256'] and item.stat().st_size == row['bytes'], 'Executor source seal changed')
    pins = read(HERE / 'SOURCE_BINDINGS.json'); adoption = read(bound(pins['adoption']))
    require(adoption['protocol_adopted'] is True and adoption['conditions'] == list(CONDITIONS)
            and adoption['paired_development_seeds'] == list(SEEDS) and adoption['constructor'] == 'be_init'
            and adoption['lambda'] == .5 and adoption['horizon']['epochs'] == 100
            and adoption['horizon']['updates_per_fit'] == 25800, 'Exact frozen18 adoption required')
    require(cfg['adoption'] == pins['adoption'] and cfg['budget_source'] == pins['budget_source']
            and adoption['sources'] == pins['source_manifests'], 'Unchanged adoption/budget/source binding')
    budget = read(bound(pins['budget_source']))['budget']
    limits = (budget['cell_active_compute_seconds'], budget['cell_cleanup_grace_seconds'], budget['cell_hard_seconds'])
    require(limits == (32390, 10, 32400), 'Reuse exact existing frozen per-cell safety caps')
    for name, digest in adoption['sources'].items():
        base = PHASE / name; require(sha(base / 'MANIFEST.json') == digest, 'Adopted training source changed')
        for row in read(base / 'MANIFEST.json')['files']:
            item = base / row['path']; require(sha(item) == row['sha256'] and item.stat().st_size == row['bytes'], 'Adopted source bytes changed')
    dispatcher = bound(pins['dispatcher'])
    require(pins['dispatcher']['path'] == adoption['ready_dispatcher']['path']
            and pins['dispatcher']['sha256'] == adoption['ready_dispatcher']['sha256'], 'Exact adopted callable dispatcher')
    require(sha(bound(pins['handoff_manifest'])) == adoption['ready_dispatcher']['handoff_manifest_sha256'], 'Exact dispatcher handoff seal')
    output = inside(cfg['execution_directory'])
    protected = [HERE, inside(pins['adoption']['path']).parent,
                 inside(pins['budget_source']['path']).parent.parent, dispatcher.parent,
                 inside(adoption['frozen_role_projection']['root_relative_directory'])]
    protected.extend(PHASE / name for name in adoption['sources'])
    require(not any(output == root or output.is_relative_to(root) for root in protected), 'Fresh output outside source/control/data directories required')
    return cfg, pins, adoption, limits, dispatcher, output


def completion(row, adoption):
    directory = inside(row['fit_output']); artifacts = {}
    for name in ('RUN.json', 'COMPLETE.json', 'PROGRESS.json', 'VALID_TRACE.json', 'selected.pt'):
        require((directory / name).is_file(), 'Full fit artifact missing: ' + name)
        artifacts[name] = binding(directory / name)
    if row['condition'] == 'independent4':
        for name in ('OWN_BEST_BANK.json', 'own_best_0.pt', 'own_best_1.pt', 'own_best_2.pt', 'own_best_3.pt'):
            require((directory / name).is_file(), 'Original own-bank artifact missing: ' + name)
            artifacts[name] = binding(directory / name)
    require(not (directory / 'FAILURE.json').exists(), 'Complete endpoint also contains failure')
    done, progress, run = (read(directory / name) for name in ('COMPLETE.json', 'PROGRESS.json', 'RUN.json'))
    arm = row['condition'] if row['condition'] in ('single', 'independent4') else 'be_init__allocation_' + row['condition']
    require(done.get('complete') is True and done['task'] == 'molhiv' and done['arm'] == arm
            and done['seed'] == row['seed'] and done['epochs'] == 100 and done['steps'] == 25800
            and done.get('TEST_scoring') is False and done['selected_sha256'] == artifacts['selected.pt']['sha256']
            and progress['epoch'] == 100 and progress['steps'] == 25800, 'Actual complete100-epoch selected-state custody required')
    require(run['task'] == 'molhiv' and run['arm'] == arm and run['seed'] == row['seed']
            and run.get('TEST_scoring') is False and run['native'] == {}, 'Actual adopted public run identity')
    public = PHASE / 'portable_internal_be_public_interface_20261007_v2'
    expected_core = {Path(item['path']).name: item['sha256'] for item in read(public / 'MANIFEST.json')['files']
                     if item['path'] in ('core/factors.py', 'core/models.py', 'core/objectives.py', 'core/selection.py')}
    require(run.get('core') == expected_core and run.get('driver_sha256') == sha(public / 'train.py'), 'Actual adopted public core/driver identity')
    roles = adoption['frozen_role_projection']['roles']
    require(run['data']['train_npz_sha256'] == roles['train']['sha256']
            and run['data']['valid_npz_sha256'] == roles['valid']['sha256']
            and run['data']['schema'] == 'portable-user-supplied-train-valid-v1'
            and run['data']['array_shape_domain_role_checks_passed'] is True, 'Actual frozen full role identity')
    if row['condition'] in ('O', 'I', 'P', 'G'):
        meta = done['allocation_control']; counts = meta['adapter_runtime_metadata']['counters']
        require(meta['policy'] == row['condition'] and meta['base_constructor_arm'] == 'be_init' and meta['lambda'] == .5
                and meta['completed_source_steps'] == 25800
                and all(counts[key] == 25800 for key in ('two_view_updates', 'Adam_calls', 'Adam_steps'))
                and counts['Session_forward_calls'] == 51600 and counts['member_forwards'] == 206400
                and counts['autograd_grad_calls'] == 25800 * {'O': 2, 'I': 2, 'P': 3, 'G': 2}[row['condition']]
                and meta['actual_external_work']['complete_VALID_evaluations'] == 100
                and meta['actual_external_work']['complete_VALID_forward_calls'] == 3300
                and meta['actual_external_work']['complete_VALID_member_forwards'] == 13200,
                'Full adopted policy work required')
    return artifacts


def roster(output):
    family = output / 'fits'
    return [dict(condition=c, seed=s, cell=c + '_' + str(s), status='unlaunched', owner_handle_id=None,
                 fit_output=str((family / (c + '_' + str(s))).relative_to(PHASE))) for s in SEEDS for c in CONDITIONS]


def close_family(output, cells, pins, adoption, owner, started, hard, fatal):
    for row in cells:
        if row['status'] == 'unlaunched':
            row['unavailable_reason'] = fatal or 'Not reached before finite closure'
        else:
            directory = inside(row['fit_output'])
            if not directory.exists():
                directory.mkdir()
                row['empty_output_directory_retained_by_closure'] = True
    closure = dict(schema='internal-be-molhiv-family-engineering-closure-v1', closed=True, task='molhiv',
        adoption=pins['adoption'], source_manifests=adoption['sources'], family_root=str((output / 'fits').relative_to(PHASE)),
        cells=cells, owner=owner, executor_manifest_sha256=sha(HERE / 'MANIFEST.json'), fatal=fatal,
        all18_accounted=True, complete_cells=sum(row['status'] == 'complete' for row in cells),
        inclusive_family_seconds=time.monotonic() - started, finite_family_hard_seconds=hard,
        quality_values_opened=False, TEST_access=False, automatic_retry=False)
    write(output / 'LEDGER.json', cells); write(output / 'FAMILY_CLOSURE.json', closure)
    return closure


def run_cells(cfg, pins, adoption, limits, dispatcher, output, parent):
    active, cleanup, hard = limits; started = parent['family_started_monotonic']
    deadline = parent['family_hard_deadline_monotonic'] - cleanup
    family = output / 'fits'; family.mkdir(); receipts = output / 'receipts'; receipts.mkdir()
    owner = proc(os.getpid()); write(output / 'FAMILY_OWNER.json', dict(owner, release=cfg, source_manifest_sha256=sha(HERE / 'MANIFEST.json')))
    cells = roster(output)
    handles = {}; fatal = None; current = None; child = None; handle = None
    write(output / 'LEDGER.json', cells); write(output / 'HANDLES.json', handles)
    roles = adoption['frozen_role_projection']; role_paths = {}
    try:
        for role in ('train', 'valid'):
            path = inside(roles['root_relative_directory']) / roles['roles'][role]['path']
            require(path.is_file() and sha(path) == roles['roles'][role]['sha256'], 'Frozen numeric role file changed')
            role_paths[role] = path
        for current in cells:
            if deadline - time.monotonic() < hard:
                fatal = 'Finite family bound cannot admit another full cell'; break
            began = time.monotonic(); cell_deadline = began + hard; before = resource.getrusage(resource.RUSAGE_CHILDREN)
            child = None; handle = None; timed_out = False
            env = dict(os.environ, INTERNAL_BE_ROOT_ADOPTED='1', TASK='molhiv', CONDITION=current['condition'],
                SEED=str(current['seed']), TRAIN=str(role_paths['train']), VALID=str(role_paths['valid']), OUTPUT=str(inside(current['fit_output'])),
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
            command = ['/bin/bash', str(dispatcher)]
            write(receipts / (current['cell'] + '_JOB.json'), dict(current, command=command,
                environment={key: env[key] for key in ('TASK', 'CONDITION', 'SEED', 'TRAIN', 'VALID', 'OUTPUT')},
                active_seconds=active, cleanup_seconds=cleanup, hard_seconds=hard, TEST_access=False, automatic_retry=False))
            try:
                # No fit/model/epoch loop is copied: the adopted dispatcher owns all training.
                with (receipts / (current['cell'] + '.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=REPO, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                    current['owner_handle_id'] = current['cell']; current['status'] = 'failed'
                    write(output / 'LEDGER.json', cells)
                    handle = proc(child.pid); require(handle is not None and handle['group'] == child.pid
                        and handle['session'] == child.pid, 'Actual fit process session required')
                    write(receipts / (current['cell'] + '_OWNER.json'), handle)
                    write(output / 'LEDGER.json', cells)
                    try:
                        code = child.wait(timeout=max(0, active - (time.monotonic() - began)))
                        current['exit_code'] = code
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        current['error'] = 'Retained active timeout; no shortened fit or retry'
                        current['cleanup_custody_unavailable'] = not stop_owned(child, handle, cell_deadline)
                        current['exit_code'] = child.poll()
                terminal = dict(handle, exit_code=current['exit_code'], reaped=current['exit_code'] is not None,
                    children_terminal=not group_live(child.pid), inclusive_seconds=time.monotonic() - began)
                if not terminal['children_terminal']:
                    stopped = stop_owned(child, handle, cell_deadline)
                    current['cleanup_custody_unavailable'] = current.get('cleanup_custody_unavailable', False) or not stopped
                    terminal['children_terminal'] = not group_live(child.pid)
                handles[current['cell']] = terminal
                if current.get('cleanup_custody_unavailable') or not terminal['reaped'] or not terminal['children_terminal']:
                    fatal = 'Owned fit exit/reap or saved-leader cleanup custody unavailable'; break
                if not timed_out and current['exit_code'] == 0 and time.monotonic() <= cell_deadline:
                    try:
                        current['artifacts'] = completion(current, adoption)
                        require(time.monotonic() <= cell_deadline, 'Per-cell hard cap expired during artifact custody')
                        current['status'] = 'complete'
                    except Exception as error:
                        current.update(status='invalid', error=type(error).__name__ + ': ' + str(error)); fatal = 'Selected-state/source custody invalid'
                else:
                    current['status'] = 'failed'
                directory = inside(current['fit_output']); directory.mkdir(exist_ok=True)
                if (directory / 'FAILURE.json').is_file():
                    current['failure'] = binding(directory / 'FAILURE.json')
            except BaseException as error:
                current.update(status='failed' if child is not None else 'unlaunched', error=type(error).__name__ + ': ' + str(error))
                if child is not None and handle is not None:
                    stopped = stop_owned(child, handle, cell_deadline)
                    current['cleanup_custody_unavailable'] = current.get('cleanup_custody_unavailable', False) or not stopped
                    code = child.poll()
                    current['exit_code'] = code
                    handles[current['cell']] = dict(handle, exit_code=code, reaped=code is not None, children_terminal=not group_live(child.pid))
                fatal = current['error']
            finally:
                if child is not None:
                    inside(current['fit_output']).mkdir(exist_ok=True)
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                current.update(inclusive_seconds=time.monotonic() - began, CPU_user_seconds=after.ru_utime - before.ru_utime,
                    CPU_system_seconds=after.ru_stime - before.ru_stime, cumulative_child_peak_RSS_bytes=after.ru_maxrss * 1024)
                write(receipts / (current['cell'] + '_CELL.json'), current)
                write(output / 'HANDLES.json', handles); write(output / 'LEDGER.json', cells)
            if fatal:
                break
    except BaseException as error:
        fatal = type(error).__name__ + ': ' + str(error)
    finally:
        write(output / 'HANDLES.json', handles); write(output / 'LEDGER.json', cells)
        close_family(output, cells, pins, adoption, owner, started, 18 * hard, fatal)
    return 0 if len(cells) == 18 and all(row['status'] == 'complete' for row in cells) else 1


def supervise(args, prepared):
    cfg, pins, adoption, limits, dispatcher, output = prepared
    require(not output.exists() and output.parent.is_dir(), 'Fresh family execution directory required')
    output.mkdir(mode=0o700); started = time.monotonic(); hard = 18 * limits[2]; cleanup = limits[1]
    child = None; handle = None; code = None; error = None
    active_timeout_triggered = False; cleanup_custody_unavailable = False
    write(output / 'PARENT_OWNER.json', dict(proc(os.getpid()), release_sha256=args.release_sha256,
        family_started_monotonic=started, family_hard_deadline_monotonic=started + hard))
    try:
        with (output / 'family.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-I', '-S', '-B', str(HERE / 'executor.py'), '--release', str(args.release),
                '--release-sha256', args.release_sha256, '--worker'], cwd=REPO, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
            handle = proc(child.pid); require(handle is not None and handle['group'] == child.pid
                and handle['session'] == child.pid, 'Actual family owner session required')
            write(output / 'FAMILY_OWNER_LIVE.json', handle)
            try:
                code = child.wait(timeout=max(0, hard - cleanup - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                active_timeout_triggered = True; error = 'Finite whole-family active bound'
                cleanup_custody_unavailable = not stop_owned(child, handle, started + hard); code = child.poll()
    except BaseException as failure:
        error = type(failure).__name__ + ': ' + str(failure)
        if child is not None and handle is not None:
            stopped = stop_owned(child, handle, started + hard)
            cleanup_custody_unavailable = cleanup_custody_unavailable or not stopped; code = child.poll()
    handles = read(output / 'HANDLES.json') if (output / 'HANDLES.json').is_file() else {}
    # Emergency owner failure: clean only the actual fit groups saved by this
    # owner. Orphan exit/reap is never fabricated; incomplete custody blocks readout.
    cleanup_records = []
    for path in (output / 'receipts').glob('*_OWNER.json') if (output / 'receipts').is_dir() else []:
        owned = read(path)
        if group_live(owned['pid']):
            record = dict(owned, emergency_owned_group_cleanup=False, actual_exit_and_reap_unavailable=True,
                cleanup_custody_unavailable=False)
            if owned_leader(owned):
                os.killpg(owned['pid'], signal.SIGTERM)
                record['emergency_owned_group_cleanup'] = True
            else:
                record['cleanup_custody_unavailable'] = True
            cleanup_deadline = min(started + hard, time.monotonic() + 5)
            while not record['cleanup_custody_unavailable'] and group_live(owned['pid']) and time.monotonic() < cleanup_deadline:
                time.sleep(.1)
            if group_live(owned['pid']):
                if owned_leader(owned):
                    os.killpg(owned['pid'], signal.SIGKILL)
                    record['emergency_owned_group_cleanup'] = True
                else:
                    record['cleanup_custody_unavailable'] = True
            record['children_terminal_observed'] = not group_live(owned['pid'])
            cleanup_records.append(record)
    if cleanup_records:
        write(output / 'EMERGENCY_CLEANUP.json', cleanup_records)
        error = error or 'Emergency owned-group cleanup; terminal exit/reap remains unavailable'
    if cleanup_custody_unavailable:
        error = error or 'Saved owner custody unavailable; no signal sent to a numeric group alone'
    closure_path = output / 'FAMILY_CLOSURE.json'
    if not closure_path.is_file():
        (output / 'fits').mkdir(exist_ok=True)
        cells = read(output / 'LEDGER.json') if (output / 'LEDGER.json').is_file() else roster(output)
        close_family(output, cells, pins, adoption, handle, started, hard,
            error or 'Owner exited without closure; unavailable cells retained')
    closure = read(closure_path)
    if error:
        closure['fatal'] = closure.get('fatal') or error
        write(closure_path, closure)
    cell_custody = all(row.get('owner_handle_id') in handles and not row.get('cleanup_custody_unavailable')
        for row in closure['cells'] if row['status'] != 'unlaunched')
    if handle is not None:
        handles['family_executor'] = dict(handle, exit_code=code, reaped=code is not None,
            children_terminal=cell_custody and not group_live(child.pid)
                and all(row.get('reaped') is True and row.get('children_terminal') is True for row in handles.values()))
    elapsed = time.monotonic() - started; hard_cap_exceeded = elapsed > hard
    if hard_cap_exceeded:
        error = error or 'Finite whole-family hard cap exceeded'
    terminal = dict(schema='internal-be-molhiv-terminal-evidence-v1', observed_at_UTC=datetime.now(timezone.utc).isoformat(),
        engineering_closure=binding(closure_path) if closure_path.is_file() else None, handles=handles,
        owner_and_children_terminal=bool(not hard_cap_exceeded and not active_timeout_triggered
            and not cleanup_custody_unavailable and not cleanup_records
            and cell_custody and handle is not None and code is not None
            and all(row.get('reaped') is True and row.get('children_terminal') is True for row in handles.values())),
        inclusive_parent_seconds=elapsed, finite_family_hard_seconds=hard,
        family_hard_cap_exceeded=hard_cap_exceeded, within_family_hard_cap=not hard_cap_exceeded,
        finite_family_active_seconds=hard - cleanup, family_active_timeout_triggered=active_timeout_triggered,
        cleanup_custody_unavailable=cleanup_custody_unavailable or any(row['cleanup_custody_unavailable'] for row in cleanup_records), error=error,
        quality_values_opened=False, TEST_access=False, automatic_retry=False)
    write(output / 'TERMINAL_EVIDENCE.json', terminal)
    print(json.dumps(dict(complete=code == 0 and terminal['owner_and_children_terminal'],
        closed=closure_path.is_file(), owner_and_children_terminal=terminal['owner_and_children_terminal'], scores_opened=False)))
    return 0 if code == 0 and terminal['owner_and_children_terminal'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); os.umask(0o077)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    prepared = prepare(args.release, args.release_sha256)
    if args.worker:
        parent = read(prepared[-1] / 'PARENT_OWNER.json'); actual = proc(os.getppid())
        require(actual is not None and actual['pid'] == parent['pid'] and actual['start_ticks'] == parent['start_ticks'], 'Actual owning outer parent required')
        return run_cells(*prepared, parent)
    return supervise(args, prepared)


if __name__ == '__main__':
    raise SystemExit(main())
