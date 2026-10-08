"""One finite native qualification, normal host, exact owned child only."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import time

R = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
P = R / 'experiments_iclr/postsubmission_20260930'
A = None  # Bound by the reviewed queue to a separate fresh root activation.
ACTIVATION_NAME = None
S = P / 'graph_relation_private_credit_source_20261008_v2'
GPU = 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998'
UUIDS = ['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
RELEASE_SHA = None  # Exact separately root-enabled GPU0 release hash.
SOURCE_SHA = 'bd4031d735aedf6c7f2e555bc3768239d3bec4dde18e1a3485be30209749c61f'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(name, value):
    target = A / name
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def ident(pid):
    try:
        raw = Path('/proc', str(pid), 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0],
                    group=int(fields[2]), session=int(fields[3]))
    except FileNotFoundError:
        return None

def query(option):
    return subprocess.check_output(['nvidia-smi', option,
        '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()

def main():
    assert Path.cwd() == R and socket.gethostname() == 'peptide'
    assert query('--query-gpu=uuid') == UUIDS
    assert A.is_relative_to(P) and A.name == ACTIVATION_NAME
    assert hashlib.sha256((A / 'RELEASE.json').read_bytes()).hexdigest() == RELEASE_SHA
    assert hashlib.sha256((S / 'SOURCE_MANIFEST.json').read_bytes()).hexdigest() == SOURCE_SHA
    for row in json.loads((S / 'SOURCE_MANIFEST.json').read_text())['files']:
        file = S / row['path']
        assert file.resolve().is_relative_to(S) and file.stat().st_size == row['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row['sha256']
    release = json.loads((A / 'RELEASE.json').read_text())
    free = {row.split(',')[0].strip(): int(row.split(',')[1].strip()) * 1024 ** 2
            for row in query('--query-gpu=uuid,memory.free')}
    assert free[GPU] >= 36 * 1024 ** 3
    owner = ident(os.getpid())
    assert owner['group'] == owner['session'] == owner['pid']
    write('OWNER.json', dict(UTC=now(), owner=owner, source_manifest_sha256=SOURCE_SHA,
        release_sha256=RELEASE_SHA, external_active_seconds=300, cleanup_seconds=10,
        external_hard_seconds=330, terminal_telemetry_reserve_seconds=20,
        scientific_fits=0, quality_scores_read=False))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONPATH='',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    environment.pop('PYTHONHOME', None)
    argv = [release['runtime_python'], '-B', str(S / 'qualify.py'),
            '--release', str(A / 'RELEASE.json'), '--release-sha256', RELEASE_SHA]
    started = time.monotonic()
    child = None
    identity = None
    reason = None
    signals = []
    peak_RSS = 0
    peak_GPU = 0
    interrupted = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: interrupted.__setitem__(0, True))
    with (A / 'stdout.log').open('xb') as stdout, (A / 'stderr.log').open('xb') as stderr:
        child = subprocess.Popen(argv, cwd=R, env=environment, stdout=stdout,
            stderr=stderr, start_new_session=True)
        identity = ident(child.pid)
        assert identity and identity['group'] == identity['session'] == child.pid
        write('CHILD_OWNER.json', dict(identity=identity, argv=argv, UTC=now()))
        while child.poll() is None:
            try:
                actual = ident(child.pid)
                assert actual is not None and actual['start_ticks'] == identity['start_ticks']
                status = Path('/proc', str(child.pid), 'status').read_text().splitlines()
                rss = next((int(row.split()[1]) * 1024 for row in status if row.startswith('VmRSS:')), 0)
                peak_RSS = max(peak_RSS, rss)
                used = 0
                for row in query('--query-compute-apps=gpu_uuid,pid,used_memory'):
                    parts = [value.strip() for value in row.split(',')]
                    if len(parts) == 3 and parts[0] == GPU and parts[1] == str(child.pid):
                        assert parts[2].isdigit()
                        used += int(parts[2]) * 1024 ** 2
                peak_GPU = max(peak_GPU, used)
                assert rss <= 32 * 1024 ** 3 and used <= 32 * 1024 ** 3
                assert sum((A / name).stat().st_size for name in ('stdout.log', 'stderr.log')) <= 8 * 1024 ** 2
                if interrupted[0]:
                    reason = 'Supervisor stop requested'
                    break
                if time.monotonic() - started >= 300:
                    reason = 'Qualification active deadline'
                    break
            except Exception as error:
                reason = type(error).__name__ + ': ' + str(error)
                break
            time.sleep(2)
        if child.poll() is None:
            actual = ident(child.pid)
            if actual and actual['start_ticks'] == identity['start_ticks']:
                os.killpg(child.pid, signal.SIGTERM)
                signals.append('TERM')
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                actual = ident(child.pid)
                if actual and actual['start_ticks'] == identity['start_ticks']:
                    os.killpg(child.pid, signal.SIGKILL)
                    signals.append('KILL')
                child.wait(timeout=5)
        code = child.wait()
    remaining = [row for row in query('--query-compute-apps=gpu_uuid,pid,used_memory')
                 if len(row.split(',')) > 1 and row.split(',')[1].strip() == str(child.pid)]
    write('EXIT.json', dict(UTC=now(), child_identity=identity, exit_code=code,
        actual_exit_and_reap=True, child_after=ident(child.pid), child_no_CUDA_rows=not remaining,
        reason=reason, signals=signals, inclusive_seconds=time.monotonic() - started,
        peak_observed_child_RSS_bytes=peak_RSS, peak_observed_child_GPU_bytes=peak_GPU,
        source_manifest_sha256=SOURCE_SHA, release_sha256=RELEASE_SHA,
        scientific_fits=0, quality_scores_read=False, automatic_retry=False))
    return 0 if code == 0 and reason is None and not remaining else 1

if __name__ == '__main__':
    raise SystemExit(main())
