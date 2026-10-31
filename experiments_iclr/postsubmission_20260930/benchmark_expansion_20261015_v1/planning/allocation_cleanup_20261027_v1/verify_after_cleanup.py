"""Verify the retained environment, live queue, and repository usage after cleanup."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
EXPANSION = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
AUDIT = EXPANSION / 'planning/allocation_cleanup_20261027_v1'

if '--imports' in sys.argv:
    import torch
    import torch_geometric
    import torch_sparse
    import torch_scatter
    import ogb
    import numpy
    import scipy
    import sklearn
    print(json.dumps({module.__name__: getattr(module, '__version__', 'imported')
                      for module in (torch, torch_geometric, torch_sparse,
                                     torch_scatter, ogb, numpy, scipy, sklearn)}))
    sys.exit(0)

assert ROOT.resolve() == ROOT and AUDIT.resolve().is_relative_to(ROOT)
os.chdir(ROOT)
monitor = EXPANSION / 'planning/both_hosts_audit_20261024_v1/queue_health.py'
assert hashlib.sha256(monitor.read_bytes()).hexdigest() == '25cb77f3d9c63dddb9545aeab5914c9130f88b1066a3b989414162280fb86f02'
result = {'utc_started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'repository': str(ROOT), 'training_modified': False}
env = dict(os.environ)
temporary = AUDIT / 'runtime-temp'
temporary.mkdir(exist_ok=True)
env.update(PYTHONDONTWRITEBYTECODE='1', XDG_CACHE_HOME=str(ROOT / '.cache'),
           TORCH_HOME=str(ROOT / '.cache/torch'), TMPDIR=str(temporary))
check = subprocess.run([str(ROOT / '.venv/bin/python'), '-B', str(Path(__file__)), '--imports'],
                       cwd=ROOT, env=env, capture_output=True, text=True)
result['environment'] = {'returncode': check.returncode, 'stdout': check.stdout,
                         'stderr': check.stderr}
print('Environment import verification:', check.returncode, flush=True)
queue = subprocess.run(['python3', '-B', str(monitor), '--host', 'allocation'],
                       cwd=ROOT, env=env, capture_output=True, text=True)
result['queue_check'] = {'returncode': queue.returncode, 'stderr': queue.stderr}
if queue.returncode == 0:
    result['queue_check']['result'] = json.loads(queue.stdout)
else:
    result['queue_check']['stdout'] = queue.stdout
print('Queue health:', json.dumps(result['queue_check']), flush=True)
print('Measuring repository usage; no files outside the repository are traversed.', flush=True)
usage = subprocess.run(['du', '-s', '-B1', str(ROOT)], capture_output=True, text=True)
result['repository_usage'] = {'returncode': usage.returncode, 'stdout': usage.stdout,
                              'stderr': usage.stderr}
if usage.returncode == 0:
    result['repository_allocated_bytes'] = int(usage.stdout.split()[0])
result['filesystem_free_bytes'] = shutil.disk_usage(ROOT).free
deletion = json.loads((AUDIT / 'REDUNDANT_FILE_DELETION_RECEIPT.json').read_text())
result['removed_files_absent'] = all(not (ROOT / row['path']).exists() for row in deletion['deleted_files'])
result['utc_finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (AUDIT / 'POST_CLEANUP_VERIFICATION.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result, indent=2), flush=True)
sys.exit(0 if check.returncode == queue.returncode == usage.returncode == 0 and result['removed_files_absent'] else 1)
