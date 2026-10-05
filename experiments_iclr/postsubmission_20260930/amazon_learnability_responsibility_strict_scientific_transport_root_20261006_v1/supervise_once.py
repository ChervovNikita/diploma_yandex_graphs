"""Supervise one already admitted scientific child; never retry it."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import socket
import subprocess
import sys
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
OUT = PHASE / 'amazon_learnability_responsibility_strict_scientific_execution_root_20261006_v1'
RUNNER = PHASE / 'amazon_learnability_responsibility_strict_process_scientific_runner_preparation_20261006_v2/run_scientific.py'
RUNNER_SHA = '7398bad90c91132b19ed31d0e04763a16c838fb80c3d8c3ddcde223757963891'
WORKER = PHASE / 'amazon_learnability_responsibility_sequential_train_only_release_root_20261006_v1/six_arm_worker.py'
WORKER_SHA = '0ec5bb6e4d391a990c15657e9fedffd7f02006c75d375bba0a873235ada2ba76'
RUNTIME = PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
PUBLIC_B = PHASE / 'learnability_responsibility_native_full_execution_root_20261005_v1/roles/public_b'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
WATCHDOG_SECONDS = 14520


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    assert not path.exists()
    temp = OUT / (name + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temp.replace(path)
    path.chmod(0o444)


def identity(pid, argv):
    proc = Path('/proc') / str(pid)
    raw = (proc / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    actual = [x.decode() for x in (proc / 'cmdline').read_bytes().split(bytes([0])) if x]
    assert actual == argv and (proc / 'cwd').resolve() == REPO
    return {'PID': pid, 'start_ticks': int(fields[19]), 'argv': argv, 'cwd': str(REPO)}


def main():
    started = time.monotonic()
    assert Path.cwd().resolve() == REPO and socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).split() == [UUID]
    assert sha(RUNNER) == RUNNER_SHA and sha(WORKER) == WORKER_SHA
    release = json.loads((OUT / 'EXECUTION_RELEASE.json').read_text())
    assert release['external_watchdog_seconds'] == WATCHDOG_SECONDS
    assert release['runner_sha256'] == RUNNER_SHA and release['scientific_worker_sha256'] == WORKER_SHA
    assert sha(Path(__file__)) == release['supervisor_sha256']
    assert not (OUT / 'output').exists() and not (OUT / 'CHILD_LAUNCH.json').exists()
    argv = [str(RUNTIME), '-B', str(RUNNER), '--execute-authorized', '--source-root', str(PHASE),
            '--repository', str(REPO), '--released-worker', str(WORKER),
            '--public-b-dir', str(PUBLIC_B), '--output', str(OUT / 'output')]
    terminal = {'UTC_start': datetime.now(timezone.utc).isoformat(), 'external_watchdog_seconds': WATCHDOG_SECONDS,
                'watchdog_fired': False, 'retry': False, 'A_scoring': False, 'VALID_TEST_access': False}
    try:
        with (OUT / 'worker.stdout.log').open('xb') as stdout, (OUT / 'worker.stderr.log').open('xb') as stderr:
            child_started = time.monotonic()
            child = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            owner = identity(child.pid, argv)
            owner.update(UTC=datetime.now(timezone.utc).isoformat(), runner_sha256=RUNNER_SHA, worker_sha256=WORKER_SHA)
            write('CHILD_LAUNCH.json', owner)
            terminal['owned_child'] = owner
            try:
                terminal['exit_code'] = child.wait(timeout=max(0.0, WATCHDOG_SECONDS - (time.monotonic() - child_started)))
            except subprocess.TimeoutExpired:
                terminal['watchdog_fired'] = True
                child.terminate()
                try:
                    terminal['exit_code'] = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    terminal['exit_code'] = child.wait()
            terminal['physical_child_seconds'] = time.monotonic() - child_started
        terminal['artifacts'] = []
        for relative in ('output/RUNNER_RESULT.json', 'output/scientific_run/COMPLETE.json',
                         'output/scientific_run/FAILURE.json', 'output/scientific_run/ATTEMPTED_OPERATIONS.jsonl',
                         'worker.stdout.log', 'worker.stderr.log'):
            path = OUT / relative
            if path.exists():
                terminal['artifacts'].append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        terminal['status'] = 'TERMINAL_OWNED_CHILD'
    except BaseException as error:
        terminal.update(status='SUPERVISOR_FAILURE', error_type=type(error).__name__, error=str(error))
        # Do not leave an admitted child running without its watchdog if metadata fails.
        if 'child' in locals() and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            terminal['owned_child_stopped_on_supervisor_failure'] = True
    finally:
        terminal.update(UTC_end=datetime.now(timezone.utc).isoformat(), supervisor_seconds=time.monotonic() - started)
        write('TERMINAL.json', terminal)
    return 0 if terminal.get('exit_code') == 0 and terminal['status'] == 'TERMINAL_OWNED_CHILD' else 1


if __name__ == '__main__':
    sys.exit(main())
