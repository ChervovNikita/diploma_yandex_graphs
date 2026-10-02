"""Seven sealed BUDDY checks on the actual 18.77 CPU runtime; no data or GPU compute."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE / 'buddy_shared_cache_execution_v4'
RUN = PHASE / 'gpu77_buddy_numerical_qualification_v1/root_run_v1'
UUIDS = {'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}
SOURCE_SHA = 'ba528e95d90be0ec32e5aacb0fb3d5445eca9a68d38ee215138e72521ea67ff8'
HEAD = '617876e5f16b76d22dfa31f5869db8f13b44d932'
PYTHON = '/disk/10tb/home/shmelev/miniconda3/envs/rapids-25.06/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def main():
    if Path.cwd().resolve() != REPO or Path(git('rev-parse', '--show-toplevel')).resolve() != REPO:
        raise RuntimeError('Actual authorized Git checkout required')
    if git('rev-parse', 'HEAD') != HEAD:
        raise RuntimeError('Expected published source commit differs')
    actual = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True).stdout.splitlines()
    if len(actual) != 2 or set(actual) != UUIDS:
        raise RuntimeError('Authorized two-GPU host differs')
    if sha(SOURCE / 'SOURCE_MANIFEST.json') != SOURCE_SHA:
        raise RuntimeError('Sealed v4 source manifest differs')
    for row in json.loads((SOURCE / 'SOURCE_MANIFEST.json').read_text())['files']:
        if sha(SOURCE / row['path']) != row['sha256']:
            raise RuntimeError('V4 source byte differs')
    extra = REPO / '.gnnm_runtime/buddy_extra_v1/site'
    setup_path = extra.parent / 'SETUP_RECEIPT.json'
    setup = json.loads(setup_path.read_text())
    if setup['status'] != 'DEPENDENCIES_INSTALLED_UNQUALIFIED' or setup['exit_code'] != 0:
        raise RuntimeError('Project-local runtime setup failed')
    for name, digest in setup['installed_RECORD_sha256'].items():
        if sha(extra / name) != digest:
            raise RuntimeError('Installed dependency RECORD differs')
    RUN.mkdir(parents=True, exist_ok=False)
    for name in ('tmp', 'cache'):
        (RUN / name).mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(extra), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TMPDIR=str(RUN / 'tmp'), XDG_CACHE_HOME=str(RUN / 'cache'), TORCH_HOME=str(RUN / 'cache/torch'))
    started = time.monotonic()
    result = subprocess.run([PYTHON, '-B', str(SOURCE / 'test_cpu.py'), '--output', str(RUN / 'CPU_QUALIFICATION.json')],
                            cwd=SOURCE, env=env, capture_output=True, text=True, timeout=110)
    (RUN / 'stdout.txt').write_text(result.stdout)
    (RUN / 'stderr.txt').write_text(result.stderr)
    path = RUN / 'CPU_QUALIFICATION.json'
    certificate = json.loads(path.read_text()) if path.exists() else None
    passed = result.returncode == 0 and certificate and certificate['status'] == 'synthetic_cpu_pass' and certificate['test_count'] == 7
    record = dict(UTC=datetime.now(timezone.utc).isoformat(), schema='buddy77-seven-CPU-qualification-v1',
                  actual_git_root=str(REPO), git_head=HEAD, physical_GPU_UUIDs=sorted(UUIDS),
                  GPU_compute=False, dataset_access=False, source_manifest_sha256=SOURCE_SHA,
                  executed_wrapper_sha256=sha(Path(__file__)), setup_receipt_sha256=sha(setup_path),
                  status='SEVEN_CPU_CHECKS_PASSED' if passed else 'NUMERICAL_QUALIFICATION_FAILED',
                  exit_code=result.returncode, seconds=time.monotonic() - started,
                  certificate=certificate, stdout=result.stdout, stderr=result.stderr,
                  other_jobs_stopped=False)
    (RUN / 'CPU_RUN_RECEIPT.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, sort_keys=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
