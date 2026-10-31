"""Finite metadata-only census child bound to the exact declared final master."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

if socket.gethostname() != 'anogena-2-0' or subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
    raise ValueError('Allocation route failed verification')
root = Path(__file__).resolve().parents[2]
packet = root / 'final_sync_master_v29_20261013_v1'
bindings = json.loads((packet / 'EXACT_FINAL_BINDINGS.json').read_text())
for name in ('PLAN', 'SUMMARY', 'INPUT_INDEX', 'CONTROLLER_MANIFEST'):
    row = bindings[name]
    raw = (root / row['path']).read_bytes()
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Exact root final binding differs: ' + name)
owner = root / 'final_master_sync_owner_20261013_v3/runtime/finite_owner.py'
if hashlib.sha256(owner.read_bytes()).hexdigest() != 'd05c4139cf086a340509001ff9450aa88b8be1c16badad32602f19edeadc91fc':
    raise ValueError('Sealed finite owner changed')
sys.path.insert(0, str(owner.parent))
from finite_owner import identity, run_child, write
folder = packet / 'census_owner_attempt1'
if folder.exists():
    raise FileExistsError('Prior census attempt retained')
folder.mkdir()
write(folder / 'OWNER_START.json', {'owner': identity(os.getpid()), 'bindings': bindings, 'maximum_child_wall_seconds': 1800, 'scientific_fits': 0, 'GO': False})
generator = root / 'final_run_census_preparation_20261013_v1/runtime/build_run_census.py'
manifest = json.loads((generator.parent.parent / 'MANIFEST.json').read_text())
bound = next(row for row in manifest['files'] if row['path'].endswith('/runtime/build_run_census.py'))
if len(generator.read_bytes()) != bound['bytes'] or hashlib.sha256(generator.read_bytes()).hexdigest() != bound['sha256']:
    raise ValueError('Sealed census generator changed')
command = [sys.executable, '-B', str(generator), '--project-root', str(root), '--master', bindings['PLAN']['path'], '--master-hash', bindings['PLAN']['canonical_sha256'], '--summary', bindings['SUMMARY']['path'], '--summary-sha256', bindings['SUMMARY']['sha256'], '--output', 'final_census_master_v29_20261013_v1']
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
result = run_child(command, folder, 'EXACT_FINAL_CENSUS', env, 1800)
write(folder / 'OWNER_RESULT.json', {'status': 'complete' if result['exit_code'] == 0 else 'failed', 'record': result, 'bindings': bindings, 'scientific_fits': 0, 'GO': False})
print(json.dumps({'status': 'complete' if result['exit_code'] == 0 else 'failed', 'result': result, 'scientific_fits': 0, 'GO': False}, sort_keys=True))
sys.exit(0 if result['exit_code'] == 0 else 1)
