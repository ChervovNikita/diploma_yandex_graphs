"""One normal-host native qualification with monitoring of its exact owned child."""
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
A = P / 'sehgnn_full_input_M4_source_supply_integration_qualification_activation_root_20261009_v1'
N = P / 'sehgnn_full_input_M4_source_supply_integration_qualification_source_20261009_v1'
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

def owned_memory(pid):
    rss = 0
    try:
        for row in Path('/proc', str(pid), 'status').read_text().splitlines():
            if row.startswith('VmRSS:'):
                rss = int(row.split()[1]) * 1024
    except FileNotFoundError:
        return None
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
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
assert cfg['enabled'] and cfg['root_external_resource_monitor']
native_cfg = json.loads(Path(cfg['native_runtime_release']['path']).read_text())
assert native_cfg['cuda_device_uuid'] == GPU and not Path(cfg['output_directory']).exists()
assert hashlib.sha256((N / 'SEAL.json').read_bytes()).hexdigest() == cfg['source_seal_sha256']
env = dict(os.environ, **native_cfg['runtime_environment'], PYTHONDONTWRITEBYTECODE='1', DGLBACKEND='pytorch')
argv = [native_cfg['python_executable'], '-B', str(N / 'qualify_integration.py'), '--qualify',
        '--release', str(A / 'RELEASE.json'), '--input-root', cfg['input_root'],
        '--output', cfg['output_directory']]
started = time.monotonic()
with (A / 'worker.stdout').open('x') as stdout, (A / 'worker.stderr').open('x') as stderr:
    child = subprocess.Popen(argv, cwd=R, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    saved = identity(child.pid)
    launch = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), child=saved,
                  parent=identity(os.getpid()), argv=argv,
                  source_commit=subprocess.check_output(['git', '-C', str(R), 'rev-parse', 'HEAD'], text=True).strip(),
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  release_sha256=hashlib.sha256((A / 'RELEASE.json').read_bytes()).hexdigest(),
                  wall_limit_seconds=3600, model_work='one full native M4 own update, source opportunity and fixed-state reconstruction',
                  automatic_retry=False, filesystem_namespace_or_mount_changes=False)
    write(A / 'LAUNCH.json', launch)
    print(json.dumps(dict(launched=launch)), flush=True)
    peak = dict(RSS_bytes=0, device_resident_bytes=0)
    stop_reason = None
    try:
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
            if time.monotonic() - started > 3600:
                stop_reason = 'owned qualifier reached its declared one-hour bound'
            if stop_reason:
                check_owned(saved)
                os.killpg(saved['group'], signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    check_owned(saved)
                    os.killpg(saved['group'], signal.SIGKILL)
                    child.wait(timeout=10)
                break
            time.sleep(2)
        child.wait()
    except BaseException as error:
        # Preserve the exact handle on an observation failure; do not restart or
        # signal other work. Root inspects the same child before any next action.
        write(A / 'MONITOR_FAILURE.json', dict(type=type(error).__name__, message=str(error), child=saved))
        raise
terminal = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=child.returncode,
                seconds=time.monotonic() - started, sampled_peak=peak, stop_reason=stop_reason,
                child_reaped=True, original_pid_absent=not Path('/proc', str(child.pid)).exists(),
                other_processes_or_jobs_changed=False, comparative_scores_opened=False)
write(A / 'TERMINAL.json', terminal)
print(json.dumps(dict(terminal=terminal)), flush=True)
raise SystemExit(child.returncode if stop_reason is None else 1)
