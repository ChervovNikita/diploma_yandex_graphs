"""One ordinary waiting owner: at most24h total, one180+10s owned GPU child."""
import argparse
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
WIKI = PHASE / 'learnable_internal_be_WikiCS_scientific_family_execution_root_20261007_v1'
SUITE = PHASE / 'learnable_internal_be_contrastive_multitask_suite_20261007_v4'
QUALIFIER = PHASE / 'learnable_internal_be_resource_qualifier_source_20261007_v1'
DRIVER = PHASE / 'learnable_internal_be_WikiCS_scientific_family_driver_source_20261007_v1'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SCIENCE_PID, SCIENCE_TICKS = 510850, 6015502511
SCIENCE_SHA = '76de82781e7fd496a5a3382b3a71a5781ea023dfe3777469cc005a2b19afbfce'
DRIVER_SHA = '9aeb3cd2f37e64fc082de800cd83af55b1c53fc45f313f7d084b133c0598ec9d'
WAIT_HARD, ACTIVE, CLEANUP, HARD = 86400, 180, 10, 190
MINIMUM_FREE = 8 * 1024**3
WINDOWS = {'single': (100, 500, 1), 'single_contrastive': (100, 500, 1), 'independent4': (200, 800, 4)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


def proc(pid):
    try:
        fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(') ', 1)[1].split()
        args = Path('/proc/' + str(pid) + '/cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        return dict(PID=pid, state=fields[0], PPID=int(fields[1]), group=int(fields[2]), start_ticks=int(fields[19]), argv=args)
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(pid, ticks):
    item = proc(pid)
    return item if item and item['start_ticks'] == ticks and item['state'] != 'Z' else None


def gpu():
    require(socket.gethostname() == 'anogena-2-0', 'Authorized normal host only')
    lines = subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=3).strip().splitlines()
    require(len(lines) == 1 and lines[0].split(',')[0].strip() == GPU, 'Sole authorized GPU only')
    return int(lines[0].split(',')[1].strip()) * 1024**2


def original_science_processes():
    # Concrete existing script identities only; no process namespace or lock.
    scripts = {str(SUITE / name) for name in ('run.py', 'supervise.py')}
    scripts |= {str(QUALIFIER / name) for name in ('worker.py', 'supervise.py')}
    scripts.add(str(DRIVER / 'driver.py'))
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            item = proc(int(path.name))
        except PermissionError:
            continue
        if item and item['state'] != 'Z' and any(arg in scripts for arg in item['argv']):
            found.append({key: item[key] for key in ('PID', 'start_ticks', 'state')})
    return found


def closed_family():
    path = WIKI / 'FAMILY_CLOSURE.json'
    if not path.is_file() or live(SCIENCE_PID, SCIENCE_TICKS):
        return None
    closed = read(path); owner = closed['owner']
    require(owner['PID'] == SCIENCE_PID and owner['start_ticks'] == SCIENCE_TICKS
            and owner['driver_manifest_sha256'] == DRIVER_SHA, 'Exact saved science owner')
    require(closed.get('schema') == 'internal-be-WikiCS-family-closure-v1' and closed.get('closed') is True
            and len(closed['cells']) == 24 and len({row['cell'] for row in closed['cells']}) == 24
            and closed.get('all_full_endpoints_or_retained_failure') is True
            and closed.get('TEST_access') is False and closed.get('predictive_values_opened') is False,
            'Authoritative whole24 engineering closure required')
    active = original_science_processes()
    if active:
        return None
    return dict(mode='whole_Wiki24_closed_all_science_terminal', closure_path=str(path),
        closure_sha256=sha(path), science_owner_PID=SCIENCE_PID, science_owner_start_ticks=SCIENCE_TICKS,
        original_science_processes_live=[], scores_read=False)


def coexisting_science(window=True):
    owner = read(WIKI / 'OWNER.json')
    require(owner['PID'] == SCIENCE_PID and owner['start_ticks'] == SCIENCE_TICKS
            and owner['driver_manifest_sha256'] == DRIVER_SHA, 'Exact recorded science owner')
    actual_owner = live(SCIENCE_PID, SCIENCE_TICKS)
    if actual_owner is None:
        return None
    running = read(WIKI / 'RUNNING_CELL.json'); arm = running['arm']; seed = running['seed']; cell = running['cell']
    if arm not in WINDOWS:
        return None
    first_epoch, last_epoch, members = WINDOWS[arm]
    require(seed in (6101, 6203, 6307) and cell == arm + '_' + str(seed), 'Exact eligible source cell')
    job_path = WIKI / 'fit_jobs' / (cell + '.json'); job = read(job_path)
    require(job['task'] == 'wikics' and job['arm'] == arm and job['seed'] == seed
            and job['source_manifest_sha256'] == SCIENCE_SHA and job.get('TEST_access') is False,
            'Unchanged exact native scientific job')
    progress_path = WIKI / 'fits/outputs' / cell / 'PROGRESS.json'; progress = read(progress_path)
    epoch = progress['epoch']
    if type(epoch) is not int or progress['complete_epochs'] != 1100 or (window and not first_epoch <= epoch <= last_epoch):
        return None
    receipt_path = WIKI / 'fits/receipts' / (cell + '_LIVE.json'); receipt = read(receipt_path)
    require(receipt['job_sha256'] == sha(job_path) and receipt['cell_identity']['task'] == 'wikics'
            and receipt['cell_identity']['arm'] == arm and receipt['cell_identity']['members'] == members,
            'Actual native live receipt/job custody')
    supervisor = live(receipt['supervisor_pid'], receipt['supervisor_start_ticks'])
    if supervisor is None or supervisor['PPID'] != SCIENCE_PID:
        return None
    require(str(SUITE / 'supervise.py') in supervisor['argv'] and str(job_path) in supervisor['argv'],
            'Actual native scientific supervisor command')
    if receipt_path.with_name(receipt_path.stem + '_TERMINAL.json').exists():
        return None
    children = Path('/proc/' + str(supervisor['PID']) + '/task/' + str(supervisor['PID']) + '/children').read_text().split()
    actual = []
    for child in children:
        item = proc(int(child))
        if item and item['state'] != 'Z' and item['PPID'] == supervisor['PID'] and str(SUITE / 'run.py') in item['argv'] and str(job_path) in item['argv']:
            actual.append(item)
    if len(actual) != 1:
        return None
    mode = 'already_active_warmed_ordinary_independent4_science' if arm == 'independent4' else 'already_active_native_single_science'
    return dict(mode=mode, cell=cell, arm=arm, seed=seed, epoch=epoch, members=members,
        admission_epoch_window=[first_epoch, last_epoch], ordinary_no_auxiliary=arm == 'independent4',
        complete_epochs=1100, science_owner_PID=SCIENCE_PID, science_owner_start_ticks=SCIENCE_TICKS,
        supervisor={key: supervisor[key] for key in ('PID', 'start_ticks', 'PPID')},
        native_child={key: actual[0][key] for key in ('PID', 'start_ticks', 'PPID')},
        job_path=str(job_path), job_sha256=sha(job_path), live_receipt_sha256=sha(receipt_path),
        progress_path=str(progress_path), progress_sha256=sha(progress_path), scores_read=False)


def admission():
    free = gpu()
    value = coexisting_science() if live(SCIENCE_PID, SCIENCE_TICKS) else closed_family()
    if value is None or free < MINIMUM_FREE:
        running = read(WIKI / 'RUNNING_CELL.json') if (WIKI / 'RUNNING_CELL.json').is_file() else {}
        return dict(eligible=False, free_GPU_bytes=free, minimum_free_GPU_bytes=MINIMUM_FREE,
                    current_science_cell=running.get('cell'), current_science_arm=running.get('arm'),
                    expected_science_owner_live=live(SCIENCE_PID, SCIENCE_TICKS) is not None,
                    reason='Required active single/ordinary-own window or terminal whole24 closure and memory are not yet available', scores_read=False)
    return dict(value, eligible=True, free_GPU_bytes=free, minimum_free_GPU_bytes=MINIMUM_FREE, physical_GPU_uuid=GPU)


def coexistence(admitted):
    free = gpu()
    if admitted['mode'] == 'whole_Wiki24_closed_all_science_terminal':
        current = closed_family()
    else:
        current = coexisting_science(window=False)
        if current is not None and (current['cell'] != admitted['cell'] or current['native_child'] != admitted['native_child'] or current['supervisor'] != admitted['supervisor']):
            current = None
        if current is None:
            current = closed_family()
    return dict(safe=current is not None, free_GPU_bytes=free, science=current, scores_read=False)


def interrupted(signum, frame):
    raise TimeoutError('Owned GPU waiting parent received signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approval', type=Path, required=True); parser.add_argument('--approval-sha256', required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--dependency-path', type=Path, action='append', default=[])
    for name in ('public-root', 'adapter-root', 'interface-root', 'allocation-root', 'train', 'valid', 'geometry', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic(); os.umask(0o077)
    require(sys.platform == 'linux' and Path.cwd().resolve() == REPO.resolve(), 'Normal Linux project host execution')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Waiting owner keeps CUDA hidden')
    require(sha(args.approval) == args.approval_sha256, 'Exact root approval bytes')
    approval = read(args.approval)
    require(approval.get('approved') is True and approval.get('GPU_wait_and_representative_execution_authorized') is True
            and approval['source_manifest_sha256'] == sha(HERE / 'MANIFEST.json'), 'Source-review-first root GPU authority')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = HERE / row['path']; require(sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Sealed GPU source changed')
    require(args.python.is_file(), 'Existing interpreter required')
    output = args.output.resolve(); require(output.is_relative_to(PHASE.resolve()) and not output.exists(), 'Fresh project phase output')
    gpu(); output.mkdir(parents=True, mode=0o700)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    child = None; identity = None; code = None; status = 'waiting'; work_started = None; admitted = None
    peak = samples = 0; monitor_errors = []; wait_observations = 0
    before = resource.getrusage(resource.RUSAGE_CHILDREN); own_before = resource.getrusage(resource.RUSAGE_SELF)
    write(output / 'OWNER.json', dict(parent_PID=os.getpid(), parent_start_ticks=proc(os.getpid())['start_ticks'],
        state='waiting_no_GPU_child', total_owner_hard_seconds=WAIT_HARD, GPU_active_seconds=ACTIVE, GPU_cleanup_seconds=CLEANUP))

    def stop_owned():
        if child is None or child.poll() is not None:
            return
        actual = live(child.pid, identity['worker_start_ticks'])
        require(actual is not None and actual['group'] == child.pid, 'Own GPU child identity/group differs; no signal sent')
        os.killpg(child.pid, signal.SIGTERM)
        remaining = max(0, min(HARD - (time.monotonic() - work_started), WAIT_HARD - (time.monotonic() - started)))
        try:
            child.wait(timeout=min(5, remaining))
        except subprocess.TimeoutExpired:
            actual = live(child.pid, identity['worker_start_ticks'])
            require(actual is not None and actual['group'] == child.pid, 'Own GPU child identity/group changed; no further signal')
            os.killpg(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=max(0, min(HARD - (time.monotonic() - work_started), WAIT_HARD - (time.monotonic() - started))))
            except subprocess.TimeoutExpired:
                pass

    try:
        while time.monotonic() - started < WAIT_HARD - HARD:
            try:
                observed = admission()
            except (FileNotFoundError, ProcessLookupError):
                observed = dict(eligible=False, reason='Source-owned admission metadata is between lifecycle events', scores_read=False)
            wait_observations += 1
            wait_record = dict(observed, waiting_seconds=time.monotonic() - started, observations=wait_observations)
            write(output / 'WAIT.json', wait_record); append(output / 'WAIT_HISTORY.jsonl', wait_record)
            if observed['eligible']:
                # Recheck immediately before the only numerical child starts.
                confirmed = admission()
                if confirmed['eligible'] and confirmed['mode'] == observed['mode'] and confirmed.get('native_child') == observed.get('native_child'):
                    admitted = confirmed; break
            time.sleep(min(20, max(0, WAIT_HARD - HARD - (time.monotonic() - started))))
        if admitted is None:
            status = 'finite_wait_expired_no_GPU_child'
        else:
            write(output / 'ADMISSION.json', dict(admitted, wait_seconds=time.monotonic() - started))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONPATH=os.pathsep.join(str(path.resolve()) for path in args.dependency_path),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYTHONHOME', None)
            command = [str(args.python.absolute()), '-B', str(HERE / 'check.py')]
            for name in ('public_root', 'adapter_root', 'interface_root', 'allocation_root', 'train', 'valid', 'geometry'):
                command.extend(['--' + name.replace('_', '-'), str(getattr(args, name).resolve())])
            command.extend(['--output', str(output / 'work')])
            write(output / 'INVOCATION.json', dict(command=command, source_manifest_sha256=sha(HERE / 'MANIFEST.json'),
                approval_sha256=args.approval_sha256, CUDA_VISIBLE_DEVICES=GPU, threads=2, automatic_retry=False))
            work_started = time.monotonic()
            with (output / 'check.log').open('x') as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                identity = dict(parent_PID=os.getpid(), parent_start_ticks=proc(os.getpid())['start_ticks'],
                    worker_PID=child.pid, worker_start_ticks=proc(child.pid)['start_ticks'], GPU_active_seconds=ACTIVE, GPU_cleanup_seconds=CLEANUP, GPU_hard_seconds=HARD)
                write(output / 'OWNER.json', dict(identity, state='owned_GPU_child_running')); last = 0
                while child.poll() is None:
                    now = time.monotonic()
                    if now - work_started >= ACTIVE or now - started >= WAIT_HARD - CLEANUP:
                        status = 'active_timeout'; stop_owned(); break
                    if now - last >= 1 and ACTIVE - (now - work_started) > 5:
                        last = now; current = coexistence(admitted)
                        coexist_record = dict(current, GPU_work_seconds=now - work_started)
                        write(output / 'COEXISTENCE.json', coexist_record); append(output / 'COEXISTENCE_HISTORY.jsonl', coexist_record)
                        if not current['safe']:
                            status = 'science_coexistence_changed_own_child_stopped'; stop_owned(); break
                        try:
                            raw = subprocess.check_output(['/usr/bin/nvidia-smi', '-i', GPU, '--query-compute-apps=pid,used_gpu_memory',
                                '--format=csv,noheader,nounits'], text=True, timeout=2)
                            values = [int(row.split(',')[1].strip()) * 1024**2 for row in raw.splitlines() if row.split(',')[0].strip() == str(child.pid)]
                            peak = max([peak] + values); samples += 1
                        except Exception as error:
                            monitor_errors.append(type(error).__name__)
                    time.sleep(.2)
                code = child.poll()
                if status == 'waiting':
                    status = 'complete' if code == 0 else 'worker_failed'
    except BaseException as error:
        status = 'parent_failure'
        write(output / 'PARENT_FAILURE.json', dict(error_type=type(error).__name__, error=str(error), GPU_child_started=child is not None, automatic_retry=False))
        try:
            stop_owned()
        except BaseException as cleanup_error:
            write(output / 'CLEANUP_FAILURE.json', dict(error_type=type(cleanup_error).__name__, error=str(cleanup_error), automatic_retry=False))
        code = child.poll() if child else None
    after = resource.getrusage(resource.RUSAGE_CHILDREN); own_after = resource.getrusage(resource.RUSAGE_SELF); elapsed = time.monotonic() - started
    work_seconds = time.monotonic() - work_started if work_started is not None else 0
    candidate_path = output / 'work/CANDIDATE.json'
    candidate = read(candidate_path) if status == 'complete' and candidate_path.is_file() else None
    complete = status == 'complete' and code == 0 and elapsed <= WAIT_HARD and work_seconds <= ACTIVE and candidate is not None and candidate['complete'] and not monitor_errors
    terminal = dict(complete=complete, status=status, exit_code=code, reaped=child is not None and code is not None,
        identity=identity, inclusive_owner_seconds=elapsed, admitted_GPU_work_seconds=work_seconds, total_owner_hard_seconds=WAIT_HARD,
        GPU_active_seconds=ACTIVE, GPU_cleanup_seconds=CLEANUP, GPU_hard_seconds=HARD,
        cap_exceeded=elapsed > WAIT_HARD or work_seconds > HARD, sampled_worker_GPU_peak_bytes=peak,
        driver_samples=samples, monitor_errors=monitor_errors, wait_observations=wait_observations,
        CPU_user_seconds=after.ru_utime - before.ru_utime, CPU_system_seconds=after.ru_stime - before.ru_stime,
        parent_CPU_user_seconds=own_after.ru_utime - own_before.ru_utime, parent_CPU_system_seconds=own_after.ru_stime - own_before.ru_stime,
        cumulative_owned_process_peak_RSS_bytes=after.ru_maxrss * 1024, GPU_child_started=child is not None,
        quality_values_closed=True, TEST_scoring=False, automatic_retry=False, other_jobs_modified=False, other_processes_signalled=False)
    write(output / 'TERMINAL.json', terminal)
    if complete:
        write(output / 'WORK_RECEIPT.json', dict(candidate, schema='allocation-OIPG-MolHIV-GPU-representative-work-v1',
            exit_code=0, reaped=True, identity=identity, admission=admitted, parent_costs=terminal,
            peak_GPU_bytes=max(peak, candidate['peak_GPU_bytes']), parent_terminal_sha256=sha(output / 'TERMINAL.json'),
            worker_source_manifest_sha256=sha(HERE / 'MANIFEST.json')))
    print(json.dumps(dict(complete=complete, status=status, GPU_child_started=child is not None, quality_values_closed=True)))
    raise SystemExit(0 if complete else 1)


if __name__ == '__main__':
    main()
