"""Finite external owner of the fixed two-child scientific cohort."""
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

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
H = Path(__file__).resolve().parent
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def main():
    assert socket.gethostname() == 'anogena-2-0' and Path.cwd() == R
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID]
    spec_path = H / 'ROOT_INTEGRATION_V1.json'; raw = spec_path.read_bytes(); spec = json.loads(raw)
    source = Path(spec['core_port_directory']).parent / 'native_SAGE_Physics_coupled_fit_integration_source_20261011_v1/run_coupled.py'
    owner = H / 'external_cohort_owner_v1'; owner.mkdir(exist_ok=False)
    start = time.monotonic(); result = dict(UTC=datetime.now(timezone.utc).isoformat(), status='RUNNING', direct_child_wait=True,
                                          external_wall_seconds=172920, spec_sha256=hashlib.sha256(raw).hexdigest(), no_retry=True)
    with (owner / 'cohort.log').open('xb') as log:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=UUID, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        child = subprocess.Popen([sys.executable, '-B', '-u', str(source), '--spec', str(spec_path), '--spec-sha256', result['spec_sha256']],
                                 cwd=R, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        result['child_pid'] = child.pid
        (owner / 'OWNER_START.json').write_text(json.dumps(result, indent=2) + '\n')
        try:
            result['exit_code'] = child.wait(timeout=172920)
            result['status'] = 'COMPLETE_EXIT_0' if child.returncode == 0 else 'FAILED_RETAINED'
        except subprocess.TimeoutExpired:
            result.update(status='FAILED_RETAINED', timed_out=True)
            start_path = H / 'cohort_owner_v1/OWNER_START.json'
            if start_path.exists():
                for row in json.loads(start_path.read_text())['children'].values():
                    pid = row['pid']; stat = Path(f'/proc/{pid}/stat')
                    if stat.exists() and int(stat.read_text().split(') ')[1].split()[1]) == child.pid:
                        try: os.killpg(pid, signal.SIGTERM)
                        except ProcessLookupError: pass
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=15)
            result['exit_code'] = child.returncode
        result['seconds'] = time.monotonic() - start
        result['child_pid_absent'] = not Path(f'/proc/{child.pid}').exists()
        (owner / 'OWNER_END.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__': main()
