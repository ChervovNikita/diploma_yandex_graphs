"""Root-owned bounded replay custody; stdlib only, exact direct children only."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = PHASE / 'graph_init_serialized_checkpoint_replay_supervision_root_20261003_v1'
OUT = HERE / 'run_v1'
PACKET = PHASE / 'graph_init_serialized_checkpoint_replay_preparation_20261003_v1'
SCRIPT = PACKET / 'replay_selected_checkpoints.py'
PYTHON = REPO / '.venv/bin/python'
ADMISSION = PHASE / 'graph_init_serialized_checkpoint_replay_release_20261003_v1/ROOT_REPLAY_ADMISSION.json'
ADMISSION_SHA = '4d41ee8a3af40a5a31d900f1c643571101b6dc5d1dc3ad8b53e644f71ad970fe'
MANIFEST_SHA = '966ddaa6def6bd6a1f9c195e257b652b851e47460984d3166e8a61e5ba077fd8'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
WALL = 1800
HOST = 8589934592


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def desc(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def bound(row):
    p = Path(row['path'])
    require(p.is_absolute() and p.resolve() == p and p.is_relative_to(PHASE), 'Exact phase path required')
    require(sha(p) == row['sha256'], 'Bound file changed: ' + str(p))
    if 'bytes' in row:
        require(p.stat().st_size == row['bytes'], 'Bound byte count changed')
    return p


def sources():
    require(sha(ADMISSION) == ADMISSION_SHA, 'Exact released admission changed')
    a = read(ADMISSION)
    require(a['root_release'] is True and a['authorized_checkpoint_inference'] is True,
            'Replay authority absent')
    require(a['wall_cap_seconds'] == WALL and a['host_memory_cap_bytes'] == HOST and
            a['gpu_memory_cap_bytes'] == 80530636800, 'Released caps changed')
    require(a['interpreter'] == str(PYTHON) and a['gpu_uuid'] == GPU and a['device'] == 'cuda:0', 'Route changed')
    require(a['source_manifest']['sha256'] == MANIFEST_SHA and
            bound(a['source_manifest']) == PACKET / 'MANIFEST.json', 'Source manifest changed')
    m = read(PACKET / 'MANIFEST.json')
    for row in m['payload']:
        bound(dict(row, path=str(PACKET / row['path'])))
    require(len(m['payload']) == 8, 'Source payload coverage changed')
    review = read(bound(a['independent_source_review']))
    require(review['approved_for_checkpoint_replay_execution'] is True and
            review['source_manifest'] == a['source_manifest'], 'Exact source review absent')
    require(bound(a['cohort']) == PACKET / 'COHORT.json', 'Cohort changed')
    return a


def probe(label, deadline):
    so, se = OUT / (label + '_gpu_uuid.stdout'), OUT / (label + '_gpu_uuid.stderr')
    with so.open('xb') as f, se.open('xb') as g:
        p = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                           stdout=f, stderr=g, timeout=min(30, max(0.001, deadline - time.monotonic())))
    require(p.returncode == 0 and so.read_text().strip().splitlines() == [GPU] and se.read_text() == '',
            'Observed sole-GPU route differs')
    return dict(repository=str(REPO), ssh_destination=LOGIN, ssh_port=2222, gpu_uuid=GPU,
                stdout=desc(so), stderr=desc(se))


def identity(pid):
    fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, parent_pid=int(fields[1]), start_ticks=int(fields[19]))


def rss(pid):
    try:
        lines = Path('/proc/' + str(pid) + '/status').read_text().splitlines()
    except FileNotFoundError:
        return 0
    return next((int(s.split()[1]) * 1024 for s in lines if s.startswith('VmRSS:')), 0)


def reap(child, flags):
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    try:
        pid, status, use = os.wait4(child.pid, flags)
        if not pid:
            return None
        require(pid == child.pid, 'Unexpected owned child')
        child.returncode = os.waitstatus_to_exitcode(status)
        child.owned_usage = dict(user_seconds=use.ru_utime, system_seconds=use.ru_stime,
                                peak_resident_bytes=int(use.ru_maxrss) * 1024)
        return child.owned_usage
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, mask)


def owned(argv, label, env, deadline):
    """Never poll/wait through Popen: only this exact unreaped child and pidfd."""
    child = fd = None
    usage = ident = None
    dispatch = utc()
    begin = time.monotonic()
    so, se = OUT / (label + '.stdout.log'), OUT / (label + '.stderr.log')
    try:
        with so.open('xb') as f, se.open('xb') as g:
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            try:
                child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=g,
                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, mask))
                fd = os.pidfd_open(child.pid, 0)
                ident = identity(child.pid)
                require(ident['parent_pid'] == os.getpid(), 'Direct child parent differs')
                write(OUT / (label + '_CHILD.json'), dict(UTC=utc(), command=argv, child_identity=ident,
                                                       pidfd_owned_until_wait4_reap=True))
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Root whole-process wall cap reached')
                ready, _, _ = select.select([fd], [], [], min(0.25, remaining))
                if ready:
                    usage = reap(child, 0)
                    break
                require(rss(child.pid) <= HOST, 'Owned child RSS exceeds released cap')
        reaped = utc()
        require(usage['peak_resident_bytes'] <= HOST, 'Final own-wait4 RSS exceeds released cap')
        value = dict(command=argv, child_identity=ident, child_exit_code=child.returncode,
                     own_wait4_reaped=True, dispatch_UTC=dispatch, reap_UTC=reaped,
                     whole_process_seconds=time.monotonic()-begin, child_usage=usage,
                     stdout=desc(so), stderr=desc(se), automatic_retry=False, timed_out=False)
        write(OUT / (label + '_PROCESS.json'), value)
        return value
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None and child.returncode is None:
            try:
                if fd is not None:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                else:
                    # Exact direct child remains unreaped; PID cannot be reused.
                    os.kill(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            end = time.monotonic() + 5
            while time.monotonic() < end:
                usage = reap(child, os.WNOHANG)
                if usage is not None:
                    break
                time.sleep(0.02)
        write(OUT / (label + '_FAILURE.json'), dict(error_type=type(error).__name__, error=str(error),
            child_identity=ident, child_exit_code=None if child is None else child.returncode,
            own_wait4_reaped=usage is not None, child_usage=usage, exact_direct_child_only=True,
            timed_out=isinstance(error, TimeoutError), automatic_retry=False, UTC=utc()))
        raise
    finally:
        if fd is not None:
            os.close(fd)


def alarm(signum, frame):
    raise TimeoutError('Root whole-process wall cap reached')


def main():
    require(sys.executable == '/usr/bin/python3' and sys.platform.startswith('linux') and
            Path.cwd() == REPO and Path(__file__).resolve() == HERE / 'supervise_replay.py',
            'Use exact system supervisor interpreter and remote repository')
    require(hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal') and hasattr(os, 'wait4'),
            'pidfd/own wait4 APIs required')
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + WALL
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, WALL)
    completed, error, replay_receipt = False, None, None
    try:
        a = sources()
        output = Path(a['output'])
        require(not output.exists(), 'Replay output occupied; no retry')
        write(OUT / 'START.json', dict(UTC=utc(), admission=desc(ADMISSION), supervisor=desc(Path(__file__)),
                                     supervisor_interpreter=sys.executable, wall_cap_seconds=WALL,
                                     host_memory_cap_bytes=HOST, automatic_retry=False))
        before = probe('before', deadline)
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
        argv = [str(PYTHON), str(SCRIPT), 'run', '--admission', str(ADMISSION)]
        child = owned(argv, 'inference', env, deadline)
        after = probe('after', deadline)
        require(child['child_exit_code'] == 0, 'Replay child exited unsuccessfully')
        require(sources() == a, 'Release/source custody changed')
        result_path, start_path = output / 'INFERENCE_RESULT.json', output / 'STARTED.json'
        result, start = read(result_path), read(start_path)
        require(start['child_identity'] == result['child_identity'] == child['child_identity'] and
                start['command'] == result['command'] == argv, 'Physical dispatch identity differs from result')
        physical = dict(schema='graph-init-checkpoint-replay-physical-closure-v1',
            source_manifest=a['source_manifest'], admission=desc(ADMISSION), started=desc(start_path),
            result=desc(result_path), output=str(output), route_before=before, route_after=after, **child)
        write(OUT / 'PHYSICAL_CLOSURE.json', physical)
        closure = [str(PYTHON), str(SCRIPT), 'close', '--admission', str(ADMISSION),
                   '--physical-receipt', str(OUT / 'PHYSICAL_CLOSURE.json'),
                   '--physical-sha256', sha(OUT / 'PHYSICAL_CLOSURE.json')]
        closed = owned(closure, 'closure', env, deadline)
        require(closed['child_exit_code'] == 0, 'Stdlib closure exited unsuccessfully')
        replay_receipt = desc(output / 'SERIALIZED_CHECKPOINT_REPLAY_RECEIPT.json')
        require(read(Path(replay_receipt['path']))['physical_closure'] is True, 'Closure receipt absent')
        completed = True
    except BaseException as exception:
        error = dict(type=type(exception).__name__, message=str(exception), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    write(OUT / 'TERMINAL.json', dict(UTC=utc(), completed=completed, error=error,
        whole_supervision_seconds=time.monotonic()-started, automatic_retry=False,
        serialized_replay_receipt=replay_receipt))
    print(json.dumps(dict(completed=completed, error=error, output=str(OUT), replay_receipt=replay_receipt)), flush=True)
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
