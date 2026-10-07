"""Finite CPU-only reader of discarded engineering states; no model forward."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import signal
import socket
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent


def ident(pid):
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    f = text[text.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(f[19]), state=f[0])


def write(name, value):
    with (HERE / name).open('x') as out:
        json.dump(value, out, indent=2)
        out.write('\n')


def main():
    assert Path.cwd() == REPO and HERE.is_relative_to(PHASE) and socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    write('PARENT_OWNER.json', ident(os.getpid()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONPATH=str(PHASE / 'native_ncn_dependency_overlay_20261005_v1') + ':' + str(REPO / '.venv/lib/python3.11/site-packages'))
    release = HERE / 'RELEASE.json'
    argv = [str(PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'), '-B',
        str(PHASE / 'contrastive_BE_native_first_update_CPU_diagnostic_source_20261008_v1/compare.py'),
        '--release', str(release), '--release-sha256', hashlib.sha256(release.read_bytes()).hexdigest()]
    child = saved = None
    signals = []
    started = time.monotonic()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    reason = None
    peak_rss = 0
    def interrupted(signum, frame):
        raise RuntimeError('CPU owner interrupted: ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (HERE / 'stdout.log').open('x') as out, (HERE / 'stderr.log').open('x') as err:
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            saved = ident(child.pid)
            write('CHILD_OWNER.json', dict(saved, argv=argv))
            while child.poll() is None:
                try:
                    text = (Path('/proc') / str(child.pid) / 'status').read_text()
                    rss = next(int(line.split()[1])*1024 for line in text.splitlines() if line.startswith('VmRSS:'))
                    peak_rss = max(peak_rss, rss)
                except (FileNotFoundError, StopIteration):
                    pass
                if peak_rss > 4*1024**3:
                    reason = 'owned_RSS_cap'
                    break
                if time.monotonic()-started > 300:
                    reason = 'active_timeout'
                    break
                time.sleep(.2)
    except BaseException as error:
        reason = type(error).__name__ + ': ' + str(error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            expected = saved or ident(child.pid)
            assert ident(child.pid)['start_ticks'] == expected['start_ticks']
            os.killpg(child.pid, signal.SIGTERM)
            signals.append('SIGTERM_OWNED_CHILD_GROUP')
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                assert child.poll() is None and ident(child.pid)['start_ticks'] == expected['start_ticks']
                os.killpg(child.pid, signal.SIGKILL)
                signals.append('SIGKILL_OWNED_CHILD_GROUP')
                child.wait(timeout=5)
        exit_code = child.wait(timeout=5) if child is not None else None
        terminal_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        write('EXIT.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            exit_code=exit_code, exit_authority='subprocess.Popen.wait', child_reaped=child is not None,
            stop_reason=reason, signals=signals, elapsed_seconds=time.monotonic()-started,
            child_CPU_seconds=terminal_usage.ru_utime+terminal_usage.ru_stime-usage.ru_utime-usage.ru_stime,
            maximum_sampled_child_RSS_bytes=peak_rss, scientific_fits=0, model_forward_calls=0, GPU_used=False))


if __name__ == '__main__':
    main()
