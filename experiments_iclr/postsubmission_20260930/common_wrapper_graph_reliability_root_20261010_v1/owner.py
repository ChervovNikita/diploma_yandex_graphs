"""One finite native-backbone-family subprocess, direct wait, and closure receipt."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def stamp():
    return datetime.now(timezone.utc).isoformat()

def identity(pid):
    p = Path('/proc', str(pid), 'stat')
    if not p.exists():
        return None
    fields = p.read_text().rsplit(')', 1)[1].split()
    return dict(PID=pid, start_ticks=int(fields[19]), state=fields[0])

def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [UUID]
    here = Path(__file__).resolve().parent
    phase = REPO / 'experiments_iclr/postsubmission_20260930'
    assert here.is_relative_to(phase) and Path.cwd() == REPO
    frozen = json.loads((here / 'FREEZE.json').read_text())
    for row in frozen['bound_files']:
        path = phase / row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
    output = here / 'actual_study_v1'
    assert not output.exists() and not (here / 'OWNER_START.json').exists()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    cmd = [str(REPO / '.venv/bin/python'), '-B', str(here / 'run.py')]
    start = time.monotonic()
    with (here / 'worker.stdout.log').open('x') as out, (here / 'worker.stderr.log').open('x') as err:
        child = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out, stderr=err, start_new_session=True)
        child_id = identity(child.pid)
        write(here / 'OWNER_START.json', dict(UTC=stamp(), owner=identity(os.getpid()), child=child_id,
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), hostname=socket.gethostname(),
              gpu_uuid=UUID, freeze_sha256=hashlib.sha256((here / 'FREEZE.json').read_bytes()).hexdigest(),
              finite_seconds=21600, scientific_command=cmd))
        timed_out = False
        try:
            code = child.wait(timeout=21600)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=15)
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines()
    own_cuda = str(child.pid) in [x.strip() for x in rows]
    complete_path = output / 'COMPLETE_STUDY.json'
    complete = None
    if code == 0 and complete_path.is_file():
        report = json.loads(complete_path.read_text())
        complete = report.get('complete_roster') is True and len(report.get('banks', [])) == 45 and report.get('fit_calls') == 855 and report.get('TEST_access') is False
    closed = identity(child.pid) is None and not own_cuda
    receipt = dict(UTC=stamp(), exit_code=code, timed_out=timed_out,
              direct_child_wait=True, child=child_id, child_pid_absent=identity(child.pid) is None,
              owned_cuda_pid_absent=not own_cuda, seconds=time.monotonic()-start,
              complete_family=complete is True, scientific_success=code == 0 and complete is True and closed,
              complete_sha256=hashlib.sha256(complete_path.read_bytes()).hexdigest() if complete_path.exists() else None,
              TEST_access=False, overlap='Normal host execution; timings are observed costs, not isolated benchmarks.')
    write(here / 'OWNER_END.json', receipt)
    return 0 if receipt['scientific_success'] else 1

if __name__ == '__main__':
    sys.exit(main())
