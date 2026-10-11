"""Finite direct-child ownership for the admitted TRAIN-gradient diagnostic."""
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
H = P / 'WikiCS_E_TRAIN_shared_gradient_cancellation_diagnostic_root_20261011_v1'
S = P / 'WikiCS_E_TRAIN_shared_gradient_cancellation_diagnostic_source_20261011_v1'
admission = json.loads((H / 'ADMISSION_V1.json').read_text())
assert hashlib.sha256((S / 'diagnose.py').read_bytes()).hexdigest() == admission['source_sha256']
assert hashlib.sha256((S / 'PLAN.json').read_bytes()).hexdigest() == admission['plan_sha256']
runtime = admission['runtime_fingerprint']
runtime_path = Path(runtime['path']); assert runtime_path.resolve().is_relative_to(R)
assert runtime_path.stat().st_size == runtime['bytes'] and hashlib.sha256(runtime_path.read_bytes()).hexdigest() == runtime['sha256']
runtime_value = json.loads(runtime_path.read_text())
assert runtime_value['driver'] == subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip()
for item in runtime_value['python_sources'].values():
    path = Path(item['path']); assert path.resolve().is_relative_to(R)
    assert path.stat().st_size == item['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
assert not (H / 'OWNER_START_V1.json').exists() and not (H / 'complete_v1').exists()
started = time.monotonic()
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH', None)
env.pop('PYTHONHOME', None)
with (H / 'child_v1.log').open('x') as log:
    child = subprocess.Popen([str(R / '.venv/bin/python'), '-B', str(S / 'diagnose.py'), '--admission', str(H / 'ADMISSION_V1.json'), '--output', str(H / 'complete_v1')], cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (H / 'OWNER_START_V1.json').write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(), child_pid=child.pid, parent_pid=os.getpid(), wall_seconds=240, complete_roster_required=True, retry=False), indent=2)+'\n')
    timed_out = False
    try:
        code = child.wait(timeout=240)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
    terminal = H / 'complete_v1/COMPLETE_TRAIN_GRADIENT_DIAGNOSTIC.json'
    complete = terminal.exists() and json.loads(terminal.read_text()).get('complete') is True
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), exit_code=code, timed_out=timed_out, seconds=time.monotonic()-started, direct_child_wait=True, child_pid=child.pid, child_absent=not Path('/proc', str(child.pid)).exists(), complete_roster=complete, owned_cuda_pid_absent=str(child.pid) not in [line.split(',')[0].strip() for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).splitlines()], TEST_access=False, current_Physics_Penn_access=False, model_fits=0, retry=False)
    (H / 'OWNER_END_V1.json').write_text(json.dumps(result, indent=2)+'\n')
