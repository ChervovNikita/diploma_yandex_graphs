"""One finite owned public-context engineering child; no retries or scientific fit."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    value = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = value[value.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0])


def inside(relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Phase-relative custody required')
    value = (PHASE / rel).resolve()
    require(value != PHASE and value.is_relative_to(PHASE), 'Output leaves the research phase')
    return value


def write(root, name, value):
    with (root / name).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def stop_owned(child, saved, signals, deadline):
    """At most5 seconds TERM plus5 seconds KILL; only our original process group."""
    if child.poll() is not None:
        return child.wait(timeout=min(5., max(0., deadline-time.monotonic())))
    require(identity(child.pid)['start_ticks'] == saved['start_ticks'], 'Owned child identity changed')
    try:
        os.killpg(child.pid, signal.SIGTERM)
        signals.append('SIGTERM_OWNED_CHILD_GROUP')
    except ProcessLookupError:
        return child.wait(timeout=min(5., max(0., deadline-time.monotonic())))
    try:
        return child.wait(timeout=min(5., max(0., deadline-time.monotonic())))
    except subprocess.TimeoutExpired:
        if child.poll() is not None:
            return child.wait(timeout=min(5., max(0., deadline-time.monotonic())))
        require(identity(child.pid)['start_ticks'] == saved['start_ticks'], 'Owned child identity changed')
        try:
            os.killpg(child.pid, signal.SIGKILL)
            signals.append('SIGKILL_OWNED_CHILD_GROUP')
        except ProcessLookupError:
            pass
        return child.wait(timeout=min(5., max(0., deadline-time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    require(Path.cwd() == REPO and HERE.is_relative_to(PHASE), 'Authorized allocation repository only')
    require(socket.gethostname() == 'anogena-2-0', 'Authorized allocation host only')
    require(sha(args.release) == args.release_sha256, 'Exact engineering release required')
    cfg = json.loads(args.release.read_text())
    require(cfg.get('schema') == 'public-context-full-native-qualifier-release-v1'
        and cfg.get('enabled') is True and cfg.get('root_engineering_execution_authorized') is True
        and cfg.get('external_supervision_confirmed') is True
        and cfg.get('scientific_fit') is False and cfg.get('automatic_retry') is False
        and cfg.get('VALID_labels_access') is False and cfg.get('TEST_access') is False,
        'Engineering only, separately authorized and finite')
    require(cfg['external_active_seconds'] == 1800 and cfg['external_cleanup_seconds'] == 10
        and cfg['external_hard_seconds'] == 1810, 'Measured finite lifetime envelope')
    require(sha(HERE/'MANIFEST.json') == cfg['source_manifest_sha256'], 'Exact reviewed source required')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        require((HERE/row['path']).stat().st_size == row['bytes'] and sha(HERE/row['path']) == row['sha256'],
                'Qualifier source changed')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=10).splitlines() == [GPU], 'Actual sole physical allocation GPU')
    custody = inside(cfg['custody_output'])
    output = inside(cfg['output'])
    require(custody != output and not custody.is_relative_to(output) and not output.is_relative_to(custody)
        and not custody.is_relative_to(HERE) and not output.is_relative_to(HERE), 'Fresh separate custody/child outputs')
    custody.mkdir(parents=True, exist_ok=False)
    write(custody, 'PARENT_OWNER.json', identity(os.getpid()))
    telemetry = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    write(custody, 'FRESH_RESOURCE.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        columns=['uuid','memory.total_MiB','memory.free_MiB','utilization.gpu_percent'], actual=telemetry))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1') + ':' + str(REPO/'.venv/lib/python3.11/site-packages'))
    argv = [str(PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'), '-B', str(HERE/'qualify.py'),
        '--release', str(args.release.resolve()), '--release-sha256', args.release_sha256]
    child = saved = None
    signals = []
    started = cleanup_deadline = None
    def interrupted(signum, frame):
        raise RuntimeError('Controller interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (custody/'stdout.log').open('x') as out, (custody/'stderr.log').open('x') as err:
            # Protect the spawn/capture window; qualify.py unblocks the inherited mask before work.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                started = time.monotonic()
                child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=err, start_new_session=True)
                saved = identity(child.pid)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            write(custody, 'CHILD_OWNER.json', dict(saved, argv=argv))
            timeout = False
            try:
                exit_code = child.wait(timeout=max(0., 1800-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                timeout = True
                cleanup_deadline = min(started+1810, time.monotonic()+10)
                exit_code = stop_owned(child, saved, signals, cleanup_deadline)
        receipt = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), child=saved,
            exit_code=exit_code, exit_authority='subprocess.Popen.wait', child_reaped=True,
            timeout=timeout, signals=signals, elapsed_seconds=time.monotonic()-started,
            scientific_fits=0, VALID_TEST_labels_access=False, release_sha256=args.release_sha256)
        write(custody, 'EXIT.json', receipt)
        child = None
        require(exit_code == 0 and not timeout, 'Engineering child failed; no retry')
        qualified = output/'QUALIFIED.json'
        result = json.loads(qualified.read_text())
        require(result.get('complete') is True and result.get('discarded_native_updates') == 10
            and result.get('scientific_fit') is False and result.get('VALID_labels_access') is False
            and result.get('TEST_access') is False, 'Whole bounded engineering coverage required')
        write(custody, 'VERIFIED_COMPLETION.json', dict(qualification_sha256=sha(qualified),
            complete=True, scientific_fit=False, scores_read=False))
    except BaseException as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cleanup = None
        if child is not None:
            cleanup_signals = []
            try:
                if child.poll() is None:
                    cleanup_deadline = cleanup_deadline or min(started+1810, time.monotonic()+10)
                    cleanup_exit = stop_owned(child, saved or identity(child.pid), cleanup_signals, cleanup_deadline)
                else:
                    cleanup_exit = child.wait(timeout=5)
                cleanup = dict(child_reaped=True, exit_code=cleanup_exit, signals=cleanup_signals)
            except BaseException as cleanup_error:
                cleanup = dict(child_reaped=False, signals=cleanup_signals,
                    error_type=type(cleanup_error).__name__, error=str(cleanup_error))
        write(custody, 'CONTROLLER_FAILURE.json', dict(error_type=type(error).__name__, error=str(error),
            active_child_cleanup=cleanup, automatic_retry=False, scientific_fits=0, scores_read=False))
        raise


if __name__ == '__main__':
    main()
