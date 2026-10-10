"""Full-input typed-context qualification, using only its exact owned child."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
A = P / 'typed_context_allocation_qualification_preparation_20261010_v1'
N = P / 'typed_label_context_factor_source_prototype_20261010_v3'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2)
        output.write(chr(10))

def identity(pid):
    text = Path('/proc', str(pid), 'stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), group=int(fields[2]),
                session=int(fields[3]), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def check_owned(saved):
    assert identity(saved['pid']) == saved, 'Never signal an unowned or reused process'

def output_size(folder):
    total = 0
    for path in Path(folder).rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # A selected checkpoint may be atomically replaced.
    return total


def owned_memory(pid):
    rss = 0
    try:
        for row in Path('/proc', str(pid), 'status').read_text().splitlines():
            if row.startswith('VmRSS:'):
                rss = int(row.split()[1]) * 1024
    except FileNotFoundError:
        return None
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True, timeout=5)
    device = 0
    for row in raw.splitlines():
        parts = [part.strip() for part in row.split(',')]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            device += int(parts[1].split()[0]) * 1024**2
    return dict(RSS_bytes=rss, device_resident_bytes=device)

assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
assert not (A / 'LAUNCH.json').exists() and not (A / 'TERMINAL.json').exists()
cfg = json.loads((A / 'RELEASE.json').read_text())
assert cfg['enabled'] and cfg['resource_budget']['root_owns_external_resource_monitor']
assert cfg['cuda_device_uuid'] == GPU and not Path(cfg['output_directory']).exists()
assert hashlib.sha256((N / 'SEAL.json').read_bytes()).hexdigest() == cfg['source_seal_sha256']
env = dict(os.environ, **cfg['runtime_environment'], PYTHONDONTWRITEBYTECODE='1', DGLBACKEND='pytorch')
argv = [cfg['python_executable'], '-B', str(A / 'ENTRY.py'),
        '--release-sha256', hashlib.sha256((A / 'RELEASE.json').read_bytes()).hexdigest()]
source_commit = subprocess.check_output(['git', '-C', str(R), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
launcher_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
release_sha256 = hashlib.sha256((A/'RELEASE.json').read_bytes()).hexdigest()
started = time.monotonic()
child = saved = None
peak = dict(RSS_bytes=0, device_resident_bytes=0)
stop_reason = None
cleanup_errors = []
directly_waited = False
with (A/'worker.stdout').open('x') as stdout, (A/'worker.stderr').open('x') as stderr:
    try:
        child = subprocess.Popen(argv, cwd=R, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        saved = identity(child.pid)
        assert saved['group'] == child.pid and saved['session'] == child.pid
        launch = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), child=saved,
                      parent=identity(os.getpid()), argv=argv, source_commit=source_commit,
                      launcher_sha256=launcher_sha256, release_sha256=release_sha256,
                      wall_limit_seconds=3600, automatic_retry=False,
                      model_work='One full paired update and selected replay for all six conditions',
                      filesystem_namespace_or_mount_changes=False)
        write(A/'LAUNCH.json', launch)
        print(json.dumps(dict(launched=launch)), flush=True)
        while child.poll() is None:
            check_owned(saved)
            memory = owned_memory(child.pid)
            if memory is not None:
                for key in peak:
                    peak[key] = max(peak[key], memory[key])
                if memory['RSS_bytes'] > cfg['resource_budget']['host_RSS_bytes']:
                    stop_reason = 'owned qualifier exceeded its declared host RSS budget'
                if memory['device_resident_bytes'] > cfg['resource_budget']['device_bytes']:
                    stop_reason = 'owned qualifier exceeded its declared device budget'
            if (A/'worker.stdout').stat().st_size + (A/'worker.stderr').stat().st_size > 8*1024**2:
                stop_reason = 'owned qualifier exceeded its declared log budget'
            if output_size(cfg['output_directory']) > 12*1024**3:
                stop_reason = 'owned qualifier exceeded its declared output budget'
            if time.monotonic()-started > 3600:
                stop_reason = 'owned qualifier reached its declared one-hour bound'
            if stop_reason:
                break
            time.sleep(2)
    except BaseException as error:
        stop_reason = type(error).__name__+': '+str(error)
    finally:
        # Every operation after Popen, including launch bookkeeping, reaches here.
        if child is not None:
            try:
                if child.poll() is None:
                    if saved is None:
                        saved = identity(child.pid)
                    check_owned(saved)
                    os.killpg(saved['group'], signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        check_owned(saved)
                        os.killpg(saved['group'], signal.SIGKILL)
                        child.wait(timeout=5)
                child.wait(timeout=1)
                directly_waited = True
            except BaseException as error:
                cleanup_errors.append(type(error).__name__+': '+str(error))
                stop_reason = stop_reason or 'owned cleanup or direct wait unconfirmed'
# Failure logging cannot bypass the cleanup block above.
if stop_reason is not None:
    write(A/'MONITOR_FAILURE.json', dict(reason=stop_reason, child=saved,
          child_PID=None if child is None else child.pid, cleanup_errors=cleanup_errors))
if output_size(cfg['output_directory']) > 12*1024**3:
    stop_reason = stop_reason or 'owned qualifier exceeded output budget at completion'
if (A/'worker.stdout').stat().st_size + (A/'worker.stderr').stat().st_size > 8*1024**2:
    stop_reason = stop_reason or 'owned qualifier exceeded log budget at completion'
cuda_absent = None
try:
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=2)
    cuda_absent = child is not None and str(child.pid) not in [q.strip() for q in raw.splitlines()]
except BaseException as error:
    cleanup_errors.append(type(error).__name__+': '+str(error))
process_absent = child is not None and not Path('/proc',str(child.pid)).exists()
code = None if child is None else child.returncode
terminal = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=code,
                seconds=time.monotonic()-started, sampled_peak=peak, stop_reason=stop_reason,
                child_reaped=directly_waited, original_pid_absent=process_absent,
                owned_CUDA_absence_verified=cuda_absent, cleanup_errors=cleanup_errors,
                other_processes_or_jobs_changed=False, comparative_scores_opened=False)
write(A/'TERMINAL.json', terminal)
print(json.dumps(dict(terminal=terminal)), flush=True)
raise SystemExit(0 if code == 0 and stop_reason is None and directly_waited and process_absent and cuda_absent else 1)
