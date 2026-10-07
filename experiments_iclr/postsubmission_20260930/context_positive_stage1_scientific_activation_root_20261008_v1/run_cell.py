"""Apply the measured runtime envelope, then run the sealed scientific driver."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import socket
import subprocess
import sys

GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
assert socket.gethostname() == 'anogena-2-0' and os.environ.get('CUDA_VISIBLE_DEVICES') == GPU
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [GPU]
release = json.loads(Path(sys.argv[1]).read_text())
qualifier = PHASE / release['native_qualification']['path']
assert hashlib.sha256(qualifier.read_bytes()).hexdigest() == release['native_qualification']['sha256']
assert json.loads(qualifier.read_text())['complete'] is True
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(release['owned_GPU_memory_cap_bytes'] / torch.cuda.get_device_properties(0).total_memory, 0)
driver = PHASE / 'contrastive_BE_steering_continuing_research_20261007_v1/integration_successor_v2/train_context.py'
sys.path.insert(0, str(driver.parent))
sys.argv = [str(driver), '--task', 'wikics', '--arm', release['method'],
    '--release', str(Path(sys.argv[1]).resolve()), '--seed', str(release['seed']), '--device', 'cuda:0',
    '--train', str(PHASE / release['inputs']['train']['path']),
    '--valid', str(PHASE / release['inputs']['development']['path']),
    '--polynormer', str(PHASE / release['inputs']['polynormer']['path']),
    '--targets', str(PHASE / release['inputs']['target_archive']['path']),
    '--target-preflight', str(PHASE / release['inputs']['target_preflight']['path']),
    '--role-manifest', str(PHASE / release['inputs']['role_manifest']['path']),
    '--output', str(PHASE / release['output'])]
runpy.run_path(str(driver), run_name='__main__')
