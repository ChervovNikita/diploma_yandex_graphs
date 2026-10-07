"""Detached TRAIN-only preprocessing controller; no validation, checkpoint or fit."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import socket
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
PYTHON = PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
LIBRARIES = str(PHASE / 'native_ncn_dependency_overlay_20261005_v1') + ':' + str(REPO / '.venv/lib/python3.11/site-packages')


def identity(pid):
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0]}


def write(name, value):
    with (HERE / name).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def main():
    assert Path.cwd() == REPO and HERE.is_relative_to(PHASE)
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    seal = json.loads((HERE / 'HELPER_SOURCE_SEAL.json').read_text())
    for row in seal['files']:
        file = HERE / row['path']
        assert file.stat().st_size == row['bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row['sha256']
    write('PARENT_OWNER.json', identity(os.getpid()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONPATH=LIBRARIES)
    started = time.monotonic()
    with (HERE / 'stdout.log').open('x') as out, (HERE / 'stderr.log').open('x') as err:
        child = subprocess.Popen([str(PYTHON), '-B', str(HERE / 'preflight.py')],
                                 cwd=HERE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, start_new_session=True)
        write('CHILD_OWNER.json', identity(child.pid))
        timed_out = False
        try:
            exit_code = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()
            exit_code = child.wait()
    write('EXIT.json', {'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'exit_code': exit_code, 'exit_authority': 'Popen.wait',
                       'child_reaped': True, 'timeout': timed_out,
                       'elapsed_seconds': time.monotonic() - started,
                       'scientific_TRAIN_data_access': True, 'VALID_TEST_data_access': False, 'scientific_fits': 0,
                       'GPU_used': False})


if __name__ == '__main__':
    main()
