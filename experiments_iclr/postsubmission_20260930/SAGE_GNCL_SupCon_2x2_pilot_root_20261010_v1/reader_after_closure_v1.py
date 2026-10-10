"""Run the sealed complete analysis once, after the existing fit owner closes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [UUID]
    H = Path(__file__).resolve().parent
    assert H == R / 'experiments_iclr/postsubmission_20260930/SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
    assert Path.cwd() == R
    source = H.parent / 'SAGE_GNCL_SupCon_2x2_complete_reader_source_20261010_v2/analysis.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'd5aec472b43c195200e06d4480449cd74efa1e0031b4a1c6bdf27c5b69649ade'
    expected = json.loads((H / 'OWNER_START.json').read_text())
    assert expected['owner']['PID'] == 622591 and expected['child']['PID'] == 622593
    assert expected['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert not (H / 'complete_analysis_v1').exists()
    began = time.monotonic()
    write(H / 'READER_JOB_START_V1.json', dict(UTC=now(), PID=os.getpid(), boot_id=expected['boot_id'], waiting_for_owner=622591, wait_timeout_seconds=21600, analysis_timeout_seconds=10800, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), TEST_access=False))
    child = None
    result = dict(success=False, TEST_access=False)
    try:
        while not (H / 'OWNER_END.json').is_file():
            assert time.monotonic() - began < 21600, 'Fit closure wait timed out'
            time.sleep(30)
        closed = json.loads((H / 'OWNER_END.json').read_text())
        assert closed['scientific_success'] and closed['complete_family'] and closed['exit_code'] == 0, 'Existing fixed fit family did not close successfully'
        assert closed['direct_child_wait'] and closed['child_pid_absent'] and closed['owned_cuda_pid_absent']
        assert not Path('/proc/622593').exists(), 'Fit child still present'
        env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        command = [str(R / '.venv/bin/python'), '-B', str(source), '--report', str(H / 'complete_analysis_v1/COMPLETE_REPORT.json')]
        with (H / 'reader.stdout.log').open('x') as out, (H / 'reader.stderr.log').open('x') as err:
            child = subprocess.Popen(command, cwd=R, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            write(H / 'READER_EXECUTION_START_V1.json', dict(UTC=now(), child_PID=child.pid, command=command, fit_owner_end_sha256=hashlib.sha256((H / 'OWNER_END.json').read_bytes()).hexdigest(), full_fit_closure_before_execution=True, TEST_access=False))
            code = child.wait(timeout=10800)
        result.update(exit_code=code, success=code == 0 and (H / 'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json').is_file(), direct_child_wait=True)
    except Exception as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        result.update(error_type=type(exc).__name__, error=str(exc))
    result.update(UTC=now(), seconds=time.monotonic()-began, analysis_child_PID=child.pid if child is not None else None)
    write(H / 'READER_JOB_END_V1.json', result)
    return 0 if result['success'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
