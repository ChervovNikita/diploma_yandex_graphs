"""Isolated dependency setup for the verified two-GPU checkout; no GPU work."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
OUT = REPO / '.gnnm_runtime/buddy_extra_v1'
UUIDS = {'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced'}


def main():
    if Path.cwd().resolve() != REPO or REPO.resolve() != REPO:
        raise RuntimeError('Actual two-GPU project checkout required')
    actual = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True).stdout.splitlines()
    if len(actual) != 2 or set(actual) != UUIDS:
        raise RuntimeError('Physical two-GPU identities differ')
    receipt_path = OUT / 'SETUP_RECEIPT.json'
    if receipt_path.exists() or (OUT / 'site').exists():
        raise RuntimeError('Single-use setup identity already attempted')
    for name in ('tmp', 'cache'):
        (OUT / name).mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', TMPDIR=str(OUT / 'tmp'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               XDG_CACHE_HOME=str(OUT / 'cache'))
    packages = ['datasketch==1.6.5', 'torch-sparse==0.6.18+pt27cu126',
                'torch-scatter==2.1.2+pt27cu126', 'ogb==1.3.6',
                'outdated==0.2.2', 'littleutils==0.2.4']
    started = time.monotonic()
    record = dict(schema='buddy-77-isolated-dependency-setup-v1', UTC=datetime.now(timezone.utc).isoformat(),
                  Python=sys.version, GPU_UUIDs=sorted(UUIDS), actual_git_root=str(REPO),
                  base_runtime_modified=False, GPU_compute=False, datasets_or_models_opened=False,
                  packages_requested=packages, expected_Torch='2.7.1', expected_CUDA_build='12.6',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    argv = [sys.executable, '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check',
            'install', '--no-input', '--no-cache-dir', '--no-deps', '--only-binary=:all:',
            '--target', str(OUT / 'site'), '--index-url', 'https://pypi.org/simple',
            '--find-links', 'https://data.pyg.org/whl/torch-2.7.0+cu126.html', *packages]
    try:
        result = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        (OUT / 'pip.stdout.txt').write_text(result.stdout)
        (OUT / 'pip.stderr.txt').write_text(result.stderr)
        record.update(exit_code=result.returncode,
                      status='DEPENDENCIES_INSTALLED_UNQUALIFIED' if result.returncode == 0 else 'DEPENDENCY_SETUP_FAILED')
        if result.returncode == 0:
            distributions = list(importlib.metadata.distributions(path=[str(OUT / 'site')]))
            record['installed_distributions'] = {d.metadata['Name']: d.version for d in distributions}
            record['installed_RECORD_sha256'] = {p.relative_to(OUT / 'site').as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (OUT / 'site').glob('*.dist-info/RECORD')}
    except Exception as error:
        record.update(status='DEPENDENCY_SETUP_FAILED', error_type=type(error).__name__, error=str(error))
    record['seconds'] = time.monotonic() - started
    with receipt_path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record, sort_keys=True))
    return 0 if record.get('exit_code') == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
