"""Supervise one explicitly admitted complete native17-batch two-twin resource engineering process on 18.77."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
PREPARATION = PHASE / 'graph_ncNC_collab_resource_epoch_preparation_20261003_v4'
ADMISSION = HERE / 'ROOT_ADMISSION.json'
AUTHORIZED = {'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def gpu_check(selected):
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free,memory.total',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = [line.split(',') for line in result.stdout.strip().splitlines()]
    inventory = {row[0].strip(): {'free_MiB': int(row[1]), 'total_MiB': int(row[2])} for row in rows}
    if set(inventory) != AUTHORIZED or selected not in inventory:
        raise RuntimeError('Unexpected authorized-host GPU identity')
    if inventory[selected]['free_MiB'] < 66560:
        raise RuntimeError('Scheduling deferred: selected GPU has less than 65 GiB free')
    return inventory


def main():
    started = time.monotonic()
    record = {'UTC': datetime.now(timezone.utc).isoformat(), 'supervisor_PID': os.getpid(),
              'status': 'FAILED', 'data_or_predictive_metrics_opened_by_supervisor': False,
              'other_jobs_changed': False}
    child = None
    try:
        if not HERE.is_relative_to(PHASE) or HERE != PHASE / 'graph_ncNC_resource_epoch_execution_root_20261003_v1':
            raise RuntimeError('Wrong project supervisor path')
        repo = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=REPO,
                              check=True, capture_output=True, text=True).stdout.strip()
        if repo != str(REPO):
            raise RuntimeError('Wrong repository root')
        admission = json.loads(ADMISSION.read_text())
        if admission['stage'] != 'resource_epoch' or admission['status'] != 'ROOT_ADMITTED':
            raise RuntimeError('Only the explicitly admitted GPU parity stage may run')
        if sha(PREPARATION / 'MANIFEST.json') != admission['preparation_manifest_sha256']:
            raise RuntimeError('Preparation manifest changed')
        if sha(admission['interpreter_path']) != admission['interpreter_sha256']:
            raise RuntimeError('Interpreter binary changed')
        inventory = gpu_check(admission['cuda_visible_devices'])
        runtime = HERE / 'runtime'
        (runtime / 'tmp').mkdir(parents=True, exist_ok=False)
        (runtime / 'cache').mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': admission['cuda_visible_devices'],
                    'PYTHONPATH': str(REPO / '.gnnm_runtime/buddy_extra_v1/site'),
                    'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4',
                    'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                    'TMPDIR': str(runtime / 'tmp'), 'XDG_CACHE_HOME': str(runtime / 'cache')})
        argv = [admission['interpreter_path'], '-B', str(PREPARATION / 'run.py'),
                '--stage', 'resource_epoch', '--root-admission', str(ADMISSION),
                '--output-directory', admission['output_directory']]
        with (HERE / 'CHILD.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=PREPARATION, env=env, stdout=log, stderr=subprocess.STDOUT)
            write(HERE / 'CHILD_START.json', {'UTC': datetime.now(timezone.utc).isoformat(),
                  'supervisor_PID': os.getpid(), 'child_PID': child.pid, 'argv': argv,
                  'admission_sha256': sha(ADMISSION), 'supervisor_sha256': sha(__file__),
                  'GPU_inventory_at_start': inventory, 'predictive_fit_admitted': False})
            exit_code = child.wait()
        record.update({'child_PID': child.pid, 'child_exit_code': exit_code})
        qualification = Path(admission['output_directory']) / 'QUALIFICATION.json'
        if qualification.is_file():
            result = json.loads(qualification.read_text())
            record.update({'qualification_path': str(qualification), 'qualification_sha256': sha(qualification),
                           'qualification_status': result.get('status')})
            if (exit_code == 0 and result.get('status') == 'BOTH_COMPLETE_NATIVE_RESOURCE_EPOCHS_FINITE'
                    and result.get('all_required_checks_passed') is True
                    and result.get('accounting_complete') is True and not result.get('failures')):
                record['status'] = 'BOTH_RESOURCE_EPOCH_PROCESSES_COMPLETED_AND_PASSED'
        else:
            record['qualification_file_present'] = False
    except Exception as error:
        record.update({'error_type': type(error).__name__, 'reason': str(error)[:2000]})
        if child is not None:
            record['child_PID'] = child.pid
            record['child_poll_at_exception'] = child.poll()
    finally:
        record['UTC_finished'] = datetime.now(timezone.utc).isoformat()
        record['supervisor_wall_seconds'] = time.monotonic() - started
        write(HERE / 'TERMINAL.json', record)
    return 0 if record['status'] == 'BOTH_RESOURCE_EPOCH_PROCESSES_COMPLETED_AND_PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
