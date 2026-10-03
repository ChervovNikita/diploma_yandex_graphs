"""Supervise the finite synthetic CPU check; leave research workers untouched."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')


def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def main():
    started = time.monotonic()
    terminal = {'UTC_started': datetime.now(timezone.utc).isoformat(), 'status': 'FAILED',
                'supervisor_PID': os.getpid(), 'other_jobs_changed': False, 'predictive_fit': False}
    child = None
    try:
        assert HERE == REPO / 'experiments_iclr/postsubmission_20260930/graph_conditional_response_native_qualification_execution_root_20261003_v1'
        admission = json.loads((HERE / 'ROOT_CPU_ADMISSION.json').read_text())
        assert admission['explicit_native_cpu_engineering_checks_authorized'] is True
        assert admission['benchmark_or_predictive_execution'] is False
        assert hashlib.sha256(Path(admission['interpreter_path']).read_bytes()).hexdigest() == admission['interpreter_sha256']
        (HERE / 'run01').mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONPATH=os.pathsep.join(str(REPO / '.gnnm_runtime' / name / 'site')
                       for name in ['conditional_xxhash_v1', 'conditional_pyg27_v1', 'buddy_extra_v1']))
        argv = [admission['interpreter_path'], '-B', str(HERE / 'run_native.py')]
        with (HERE / 'CHILD.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(HERE / 'CHILD_START.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'child_PID': child.pid,
                  'supervisor_PID': os.getpid(), 'argv': argv, 'threads': 1, 'CUDA_visible_devices': '',
                  'root_admission_sha256': hashlib.sha256((HERE / 'ROOT_CPU_ADMISSION.json').read_bytes()).hexdigest()})
            try:
                exit_code = child.wait(timeout=300)
            except subprocess.TimeoutExpired:
                # The cap applies solely to this recorded direct child.
                assert Path(os.readlink(Path('/proc') / str(child.pid) / 'cwd')).is_relative_to(REPO)
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise RuntimeError('Prospective 300-second synthetic CPU cap reached')
        terminal.update(child_PID=child.pid, child_exit_code=exit_code)
        receipt_path = HERE / 'run01/CHECK_RECEIPT.json'
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            terminal.update(check_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                            check_status=receipt['status'], completed_families=len(receipt['family_records']))
            if (exit_code == 0 and receipt['status'] == 'passed' and len(receipt['family_records']) == 8
                    and all(row['status'] == 'passed' for row in receipt['family_records'])
                    and (HERE / 'run01/RESULT.json').is_file()):
                terminal['status'] = 'ALL_EIGHT_NATIVE_CPU_FAMILIES_PASSED_AND_REAPED'
    except Exception as error:
        terminal.update(error_type=type(error).__name__, reason=str(error))
        if child is not None:
            terminal.update(child_PID=child.pid, child_poll_at_exception=child.poll())
    finally:
        terminal.update(UTC_finished=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started)
        write(HERE / 'TERMINAL.json', terminal)
    return 0 if terminal['status'] == 'ALL_EIGHT_NATIVE_CPU_FAMILIES_PASSED_AND_REAPED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
