"""Run one finite, detached native engineering child in the authorized repo."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def identity(pid):
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return dict(pid=pid, start_ticks=int(fields[19]), state=fields[0])


def write(name, value):
    with (HERE / name).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    assert Path.cwd() == REPO and HERE.is_relative_to(PHASE)
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
    release_path = HERE / 'RELEASE.json'
    release_sha = hashlib.sha256(release_path.read_bytes()).hexdigest()
    release = json.loads(release_path.read_text())
    assert release['external_active_seconds'] == 1800 and release['external_cleanup_seconds'] == 10
    write('PARENT_OWNER.json', identity(os.getpid()))
    write('FRESH_RESOURCE.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        columns=['uuid', 'memory.total_MiB', 'memory.free_MiB', 'utilization.gpu_percent'],
        actual=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        PYTHONPATH=str(PHASE / 'native_ncn_dependency_overlay_20261005_v1') + ':' + str(REPO / '.venv/lib/python3.11/site-packages'))
    argv = [str(PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'), '-B',
        str(PHASE / 'contrastive_BE_graph_context_full_native_qualification_source_20261008_v1/qualify.py'),
        '--release', str(release_path), '--release-sha256', release_sha]
    started = time.monotonic()
    signals = []
    with (HERE / 'stdout.log').open('x') as out, (HERE / 'stderr.log').open('x') as err:
        child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, start_new_session=True)
        saved = identity(child.pid)
        write('CHILD_OWNER.json', dict(saved, argv=argv))
        timeout = False
        try:
            exit_code = child.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            timeout = True
            assert child.poll() is None and identity(child.pid)['start_ticks'] == saved['start_ticks']
            os.killpg(child.pid, signal.SIGTERM)
            signals.append('SIGTERM_OWNED_CHILD_GROUP')
            try:
                exit_code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                assert child.poll() is None and identity(child.pid)['start_ticks'] == saved['start_ticks']
                os.killpg(child.pid, signal.SIGKILL)
                signals.append('SIGKILL_OWNED_CHILD_GROUP')
                exit_code = child.wait(timeout=5)
    write('EXIT.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        exit_code=exit_code, exit_authority='subprocess.Popen.wait', child_reaped=True,
        timeout=timeout, signals=signals, elapsed_seconds=time.monotonic()-started,
        scientific_fits=0, VALID_TEST_labels_access=False, release_sha256=release_sha))


if __name__ == '__main__':
    main()
