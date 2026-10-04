"""Ordinary source-only wrapper helpers; no operation occurs on import."""
from pathlib import Path
import getpass
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SERVER_PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
REPO = SERVER_PHASE.parents[1]
PYTHON = REPO/'.venv/bin/python'
ROUTE = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
RUNNER = 'graph_curvature_selector_exact_native_qualification_source_20261004_v2/runner.py'
EXECUTION = PHASE/'graph_curvature_selector_native_diagnostic_execution_20261004_v1'
CAP_SECONDS = 300
GRACE_SECONDS = 5
MINIMUM_FREE_MIB = 32768  # Prospective 32GiB headroom; no measured method claim.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gpu_snapshot():
    require(PHASE == SERVER_PHASE and Path.cwd().resolve() == SERVER_PHASE
        and os.environ.get('GNNM_SSH_DESTINATION') == ROUTE, 'Authorized login destination/project required')
    result = subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    require(len(rows) == 1, 'Exactly one physical GPU required')
    uuid,total,free = [part.strip() for part in rows[0].split(',')]
    require(uuid == UUID, 'Authorized sole GPU UUID required')
    return dict(ssh_destination=ROUTE,unix_user=getpass.getuser(),uid=os.getuid(),
        uuid=uuid,total_MiB=int(total),free_MiB=int(free),observed_unix=time.time())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def process_identity(pid, argv):
    """Metadata for this wrapper or a just-created owned child only."""
    stat = Path('/proc')/str(pid)/'stat'
    body = stat.read_text()
    fields = body.rsplit(')',1)[1].split()
    require(int(body.split(' ',1)[0]) == pid and len(fields) >= 20,
            'Owned process identity unavailable')
    return dict(pid=pid,pgid=os.getpgid(pid),start_ticks=int(fields[19]),argv=list(argv),
                identity_source='owned-process /proc stat field22')


def verify_sources():
    manifest = json.loads((HERE/'MANIFEST.json').read_text())
    seal = json.loads((HERE/'SEAL.json').read_text())
    require(hashlib.sha256((HERE/'MANIFEST.json').read_bytes()).hexdigest() == seal['manifest_sha256'],
            'Wrapper seal differs')
    records = [dict(row,path=str(HERE.relative_to(PHASE)/row['path'])) for row in manifest['files']]
    records += json.loads((HERE/'STAGING_INPUT.json').read_text())['files']
    for row in records:
        path = PHASE/row['path']
        data = path.read_bytes()
        require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                'Staged source differs: '+str(path))
    return dict(verified_files=len(records),wrapper_manifest_sha256=seal['manifest_sha256'])


def child_environment():
    env = os.environ.copy()
    env.update(GNNM_SSH_DESTINATION=ROUTE,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        NVIDIA_TF32_OVERRIDE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
        TMPDIR=str(EXECUTION/'tmp'),XDG_CACHE_HOME=str(EXECUTION/'cache'),
        TORCH_HOME=str(EXECUTION/'cache/torch'),TORCH_EXTENSIONS_DIR=str(EXECUTION/'cache/torch_extensions'),
        CUDA_CACHE_PATH=str(EXECUTION/'cache/cuda'))
    return env
