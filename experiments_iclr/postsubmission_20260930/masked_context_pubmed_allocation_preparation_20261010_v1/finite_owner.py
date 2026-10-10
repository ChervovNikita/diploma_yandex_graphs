"""Finite supervisor for one project child. No retries or foreign signals."""
import argparse
from datetime import datetime, timezone
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


def identity(pid):
    value = Path('/proc') / str(pid)
    stat = (value / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(PID=pid, start_ticks=int(stat[19]), pgid=int(stat[2]), rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
    plan_path = args.plan.resolve(strict=True)
    assert plan_path.is_relative_to(HERE) and Path.cwd().resolve() == REPO
    spec = json.loads(plan_path.read_text())
    assert spec['enabled'] is True and spec['automatic_retry'] is False
    assert spec['seconds'] in (600, 1800) and spec['RSS_bytes'] == 16*1024**3
    assert spec['GPU_bytes'] in (0, 32*1024**3)
    log = HERE / spec['log']
    terminal = HERE / spec['terminal']
    assert log.resolve().is_relative_to(HERE) and terminal.resolve().is_relative_to(HERE)
    assert not log.exists() and not terminal.exists()
    argv = spec['argv']
    assert Path(argv[0]).absolute() == PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
    assert argv[1] == '-B' and Path(argv[2]).resolve().is_relative_to(PHASE)
    env = dict(os.environ, PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages'), CUDA_VISIBLE_DEVICES=GPU if spec['GPU_bytes'] else '', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    started = time.monotonic()
    utc = datetime.now(timezone.utc).isoformat()
    peak_rss = peak_gpu = 0
    cause = None
    with log.open('xb') as handle:
        child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        birth = identity(child.pid)
        assert birth['pgid'] == child.pid
        try:
            while child.poll() is None:
                if time.monotonic()-started > spec['seconds']:
                    cause = 'active_time_cap'
                    break
                current = identity(child.pid)
                assert current['start_ticks'] == birth['start_ticks'] and current['pgid'] == child.pid
                peak_rss = max(peak_rss, current['rss_bytes'])
                if peak_rss > spec['RSS_bytes']:
                    cause = 'RSS_cap'
                    break
                if log.stat().st_size > 8*1024**2:
                    cause = 'log_cap'
                    break
                if spec['GPU_bytes']:
                    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
                    own = [int(line.split(',')[1].strip())*1024**2 for line in lines if line.split(',')[0].strip() == str(child.pid)]
                    peak_gpu = max([peak_gpu, *own])
                    if peak_gpu > spec['GPU_bytes']:
                        cause = 'GPU_cap'
                        break
                time.sleep(1)
        except BaseException as error:
            cause = type(error).__name__+': '+str(error)
        finally:
            if child.poll() is None:
                current = identity(child.pid)
                assert current['start_ticks'] == birth['start_ticks'] and current['pgid'] == child.pid
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    current = identity(child.pid)
                    assert current['start_ticks'] == birth['start_ticks'] and current['pgid'] == child.pid
                    os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    result = dict(schema='masked-context-single-child-finite-owner-v1', start_UTC=utc, terminal_UTC=datetime.now(timezone.utc).isoformat(), inclusive_seconds=time.monotonic()-started, child_identity=birth, directly_waited=True, child_exit_code=code, cap_or_owner_failure=cause, peak_sampled_RSS_bytes=peak_rss, peak_sampled_GPU_bytes=peak_gpu, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), automatic_retry=False, argv=argv, numerical_scope=spec['scope'])
    terminal.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if code == 0 and cause is None else 1)


if __name__ == '__main__':
    main()
