"""Finite direct-child ownership for the admitted fixed-feature diagnostic."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import socket
import subprocess
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert socket.gethostname() == 'anogena-2-0'
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
P = R / 'experiments_iclr/postsubmission_20260930'
H = P / 'WikiCS_E_source_context_association_diagnostic_root_20261011_v1'
S = P / 'WikiCS_E_source_context_association_diagnostic_source_20261011_v1'
admission = json.loads((H / 'ADMISSION_V1.json').read_text())
assert hashlib.sha256((S / 'diagnose.py').read_bytes()).hexdigest() == admission['source_sha256']
assert hashlib.sha256((S / 'PLAN.json').read_bytes()).hexdigest() == admission['plan_sha256']
assert not (H / 'OWNER_START_V1.json').exists() and not (H / 'complete_v1').exists()
started = time.monotonic()
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
with (H / 'child_v1.log').open('x') as log:
    child = subprocess.Popen([str(R / '.venv/bin/python'), '-B', str(S / 'diagnose.py'), '--admission', str(H / 'ADMISSION_V1.json'), '--output', str(H / 'complete_v1')], cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (H / 'OWNER_START_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), child_pid=child.pid, parent_pid=os.getpid(), wall_seconds=630, complete_roster_required=True, retry=False), indent=2)+'\n')
    timed_out = False
    try:
        code = child.wait(timeout=630)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            code = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    terminal = H / 'complete_v1/COMPLETE_ASSOCIATION_DIAGNOSTIC.json'
    complete = terminal.exists() and json.loads(terminal.read_text()).get('complete') is True
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=code, timed_out=timed_out, seconds=time.monotonic()-started, direct_child_wait=True, child_pid=child.pid, child_absent=not Path('/proc', str(child.pid)).exists(), complete_roster=complete, TEST_access=False, current_Physics_Penn_access=False, model_fits=0, retry=False)
    (H / 'OWNER_END_V1.json').write_text(json.dumps(result, indent=2)+'\n')
