"""Run fixed WikiCS resource measurements only, with scores closed."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
HERE = Path(__file__).resolve().parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PYTHON = PHASE/'native_ncn_runtime_20261005_v1/.venv/bin/python'
QUALIFIER = PHASE/'learnable_internal_be_resource_qualifier_source_20261007_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    if socket.gethostname() != 'anogena-2-0' or Path.cwd().resolve() != REPO:
        raise ValueError('Exact authorized singleton repository required')
    if subprocess.check_output(['/usr/bin/nvidia-smi', '--query-gpu=uuid',
            '--format=csv,noheader'], text=True).splitlines() != [GPU]:
        raise ValueError('Exact one-GPU allocation required')
    config = json.loads((HERE/'CONFIG_ADOPTED.json').read_text())
    if config.get('root_resource_execution_authorized') is not True:
        raise ValueError('Disabled root resource authority')
    if config.get('scientific_fit_authorized') is not False or config.get('TEST_access') is not False:
        raise ValueError('Resource-only scope required')
    for row in config['bound_files']:
        path = PHASE/row['path']
        if not path.resolve().is_relative_to(PHASE) or sha(path) != row['sha256']:
            raise ValueError('Fixed source/review/data/job changed')
    execution = PHASE/config['execution_directory']
    execution.mkdir(exist_ok=False)
    (execution/'outputs').mkdir()
    (execution/'receipts').mkdir()
    stat = Path('/proc/self/stat').read_text()
    owner = {'PID': os.getpid(), 'start_ticks': int(stat[stat.rfind(')')+2:].split()[19]),
        'config_sha256': sha(HERE/'CONFIG_ADOPTED.json'), 'scientific_fit': False}
    write(execution/'OWNER.json', owner)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
        PYTHONPATH=str(PHASE/'native_ncn_dependency_overlay_20261005_v1')+':'+str(REPO/'.venv/lib/python3.11/site-packages'))
    env.pop('PYTHONHOME', None)
    rows = []
    for item in config['jobs']:
        job_path = PHASE/item['path']
        started = time.monotonic()
        with (execution/'receipts'/(item['cell']+'_PARENT.log')).open('x') as log:
            result = subprocess.run([str(PYTHON), '-B', str(QUALIFIER/'supervise.py'),
                '--job', str(job_path)], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt_path = PHASE/item['resource_receipt_path']
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        passed = result.returncode == 0 and receipt is not None and receipt.get('passed') is True
        if passed and receipt.get('seed') != item['seed']:
            raise ValueError('Measured seed differs from fixed job')
        row = {'cell': item['cell'], 'seed': item['seed'], 'arm': item['arm'],
            'exit_code': result.returncode, 'passed': passed,
            'inclusive_driver_seconds': time.monotonic()-started,
            'resource_receipt_path': item['resource_receipt_path'],
            'resource_receipt_sha256': sha(receipt_path) if receipt_path.is_file() else None,
            'predictive_scores_read': False}
        rows.append(row)
        write(execution/'receipts'/(item['cell']+'_DRIVER.json'), row)
        if not passed:
            write(execution/'FAILURE.json', {'owner': owner, 'attempted': rows,
                'remaining_jobs_not_launched': len(config['jobs'])-len(rows),
                'automatic_retry': False, 'scientific_fits': 0})
            raise SystemExit(1)
    write(execution/'COMPLETE.json', {'owner': owner, 'measurements': rows,
        'scientific_fits': 0, 'predictive_scores_read': False,
        'scope': 'registered seed/arm resource trajectories, not full scientific horizons'})


if __name__ == '__main__':
    main()
