"""Disabled successor: finite owners for three engineering interfaces or nine fixed references."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def identity(pid):
    s = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(PID=pid, start_ticks=int(s[19]), pgid=int(s[2]), sid=int(s[3]),
                parent=int(s[1]), RSS_bytes=int(s[21])*os.sysconf('SC_PAGE_SIZE'))


def gpu_processes(timeout=5):
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], text=True, timeout=timeout).splitlines()
    return {int(line.split(',')[0].strip()):int(line.split(',')[1].strip())*1024**2
            for line in lines if line.strip() and line.split(',')[1].strip().isdigit()}


def size(folder):
    total = 0
    if folder.exists():
        for path in folder.rglob('*'):
            try:
                if path.is_file(): total += path.stat().st_size
            except FileNotFoundError:
                pass  # A worker can atomically replace its selected checkpoint.
    return total


def own_one(row, plan):
    release_path = PHASE/row['release']
    assert release_path.resolve().is_relative_to(HERE) and sha(release_path) == row['release_sha256']
    release = json.loads(release_path.read_text())
    limits = release['limits']
    assert limits == row['limits'] and release['record_id'] == row['record_id']
    assert release['source_manifest_sha256'] == plan['source_manifest_sha256']
    out = Path(release['output'])
    assert out.resolve().is_relative_to(HERE) and not out.exists()
    own = HERE/'owners'/row['owner_id']
    own.mkdir(parents=True, exist_ok=False)
    entrypoint = PHASE/row['entrypoint']['path']
    assert entrypoint.resolve().is_relative_to(PHASE) and sha(entrypoint) == row['entrypoint']['sha256']
    argv = [release['runtime']['python']['path'], '-B', str(entrypoint), *row['entry_args'],
            '--release', str(release_path), '--release-sha256', row['release_sha256']]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONPATH=release['runtime']['PYTHONPATH'],
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    utc = datetime.now(timezone.utc).isoformat()
    peak_rss = peak_gpu = peak_output = 0
    failure = None
    observation_failures = 0
    directly_waited = False
    cleanup_errors = []
    log = own/'WORKER.log'
    with log.open('xb') as stream:
        child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        birth = identity(child.pid)
        assert birth['pgid'] == child.pid and birth['sid'] == child.pid
        write(own/'START.json', dict(UTC=utc, child_identity=birth, argv=argv,
                                    release_sha256=row['release_sha256'], automatic_retry=False))
        try:
            while child.poll() is None:
                if time.monotonic()-started >= limits['external_active_seconds']:
                    failure = 'active_time_cap'
                    break
                try:
                    current = identity(child.pid)
                except FileNotFoundError:
                    if child.poll() is not None: break
                    raise
                assert current['start_ticks'] == birth['start_ticks'] and current['pgid'] == child.pid
                peak_rss = max(peak_rss, current['RSS_bytes'])
                peak_output = max(peak_output, size(out))
                try:
                    own_gpu = gpu_processes().get(child.pid, 0)
                    peak_gpu = max(peak_gpu, own_gpu)
                    observation_failures = 0
                except (subprocess.SubprocessError, ValueError):
                    observation_failures += 1
                    if observation_failures >= 3: raise RuntimeError('Repeated GPU resource-observation failure')
                for value, cap, reason in ((peak_rss,limits['RSS_bytes'],'RSS_cap'),
                                          (peak_gpu,limits['GPU_bytes'],'GPU_cap'),
                                          (peak_output,limits['output_bytes'],'output_cap'),
                                          (log.stat().st_size,limits['log_bytes'],'log_cap')):
                    if value > cap: failure = reason
                progress = out/'PROGRESS.json'
                epoch = None
                if progress.exists():
                    try: epoch = json.loads(progress.read_text()).get('private_epoch')
                    except (json.JSONDecodeError, FileNotFoundError): pass
                write(HERE/plan['purpose']/'FAMILY_PROGRESS.json', dict(current_record=row['record_id'], epoch=epoch,
                    active_child=birth, no_quality_fields_opened=True, completed=plan['_completed']))
                if failure: break
                time.sleep(2)
        except BaseException as error:
            failure = type(error).__name__+': '+str(error)
        finally:
            cleanup_started = time.monotonic()
            cleanup_deadline = cleanup_started + limits['external_cleanup_seconds']
            try:
                if child.poll() is None:
                    live = identity(child.pid)
                    assert live['start_ticks'] == birth['start_ticks'] and live['pgid'] == child.pid
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=min(5, max(.001,cleanup_deadline-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        live = identity(child.pid)
                        assert live['start_ticks'] == birth['start_ticks'] and live['pgid'] == child.pid
                        os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=max(.001,cleanup_deadline-time.monotonic()))
                directly_waited = True
            except BaseException as error:
                code = child.returncode
                cleanup_errors.append(type(error).__name__+': '+str(error))
                failure = failure or 'cleanup_wait_unconfirmed'
    peak_output = max(peak_output, size(out))
    if peak_output > limits['output_bytes']: failure = 'output_cap'
    if log.stat().st_size > limits['log_bytes']: failure = 'log_cap'
    absent = False
    process_absent = cuda_absent = None
    while time.monotonic() < cleanup_deadline:
        try:
            try: process_absent = identity(child.pid)['start_ticks'] != birth['start_ticks']
            except FileNotFoundError: process_absent = True
            remaining = cleanup_deadline-time.monotonic()
            if remaining <= 0: break
            cuda_absent = child.pid not in gpu_processes(timeout=min(2,remaining))
        except BaseException as error:
            cleanup_errors.append(type(error).__name__+': '+str(error))
            cuda_absent = None
        if process_absent and cuda_absent:
            absent = True
            break
        time.sleep(min(.5,max(0,cleanup_deadline-time.monotonic())))
    if not absent: failure = failure or 'owned_absence_unconfirmed_within_cleanup_cap'
    absence = dict(owned_process_absence_verified=process_absent,
                   owned_CUDA_absence_verified=cuda_absent, child_identity=birth,
                   observation_UTC=datetime.now(timezone.utc).isoformat())
    write(own/'ABSENCE.json', absence)
    complete_file = out/'COMPLETE.json'
    good = directly_waited and code == 0 and failure is None and absent and complete_file.is_file()
    raw = dict(directly_waited=directly_waited, child_exit_code=code, cap_or_owner_failure=failure,
               argv=argv, child_identity=birth, peak_sampled_RSS_bytes=peak_rss,
               peak_sampled_GPU_bytes=peak_gpu, peak_output_bytes=peak_output,
               inclusive_seconds=time.monotonic()-started, start_UTC=utc,
               terminal_UTC=datetime.now(timezone.utc).isoformat(), complete=good,
               automatic_retry=False, owned_absence=absence, cleanup_errors=cleanup_errors,
               cleanup_seconds=time.monotonic()-cleanup_started, cleanup_cap_seconds=limits['external_cleanup_seconds'])
    write(own/'RAW_OWNER_TERMINAL.json', raw)
    descriptor = lambda f:dict(path=str(f), sha256=sha(f))
    terminal = dict(schema='PubMed-strong-reference-owned-terminal-custody-v1', actual=True,
        complete=good, argv=argv, owner_id=row['owner_id'], purpose=plan['purpose'],
        record_id=row['record_id'], directly_waited=directly_waited, child_exit_code=code,
        cap_or_owner_failure=failure, release_sha256=row['release_sha256'],
        source_manifest_sha256=release['source_manifest_sha256'],
        complete_sha256=sha(complete_file) if complete_file.is_file() else None,
        raw_owner_terminal=descriptor(own/'RAW_OWNER_TERMINAL.json'),
        owned_absence_evidence=descriptor(own/'ABSENCE.json'),
        owned_process_absence_verified=process_absent, owned_CUDA_absence_verified=cuda_absent)
    write(own/'TERMINAL_CUSTODY.json', terminal)
    return good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines() == [GPU]
    assert HERE.is_relative_to(PHASE) and Path.cwd().resolve() == REPO
    assert args.plan.resolve().is_relative_to(HERE) and sha(args.plan) == args.plan_sha256
    plan = json.loads(args.plan.read_text())
    assert plan['enabled'] and not plan['automatic_retry'] and sha(__file__) == plan['owner_sha256']
    assert plan['purpose'] in ('engineering','science')
    conditions=('single_native','single_mean4_dropout','independent4_own')
    seeds=(9101,) if plan['purpose']=='engineering' else (9101,9203,9307)
    expected=[f'seed{s}__{c}' for s in seeds for c in conditions]
    assert [r['record_id'] for r in plan['records']] == expected
    for row in plan['records']:
        assert row['owner_id'] == plan['purpose']+'__'+row['record_id']
        assert row['entry_args'] == ([] if plan['purpose']=='engineering' else ['--mode','run'])
        expected_name='qualify.py' if plan['purpose']=='engineering' else 'train.py'
        assert Path(row['entrypoint']['path']).name == expected_name
    family=HERE/plan['purpose']
    family.mkdir(exist_ok=True)
    assert not (family/'FAMILY_START.json').exists()
    write(family/'FAMILY_START.json', dict(owner=identity(os.getpid()), plan_sha256=args.plan_sha256,
                                        UTC=datetime.now(timezone.utc).isoformat(), automatic_retry=False))
    plan['_completed'] = []
    for row in plan['records']:
        if not own_one(row, plan):
            write(family/'FAMILY_FAILURE.json', dict(failed_record=row['record_id'],
                    completed=plan['_completed'], automatic_retry=False, no_partial_comparison=True))
            raise SystemExit(1)
        plan['_completed'].append(row['record_id'])
    write(family/'FAMILY_COMPLETE.json', dict(complete=True, records=plan['_completed'],
                all_records_directly_waited=True, purpose=plan['purpose'], quality_fields_opened=False, comparison_pending=True))


if __name__ == '__main__':
    main()
