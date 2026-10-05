"""Stdlib-only pilot custody; no model, tensor, score or GPU work at import."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE/'shared_backbone_private_transfer_training_source_20261005_v2'
SOURCE_SHA = 'db7102df30491be8809ea4295b9ce0d5c48f3608129f09dcfef74b8f7aa2233f'
GATE_RELATIVE = 'shared_backbone_private_transfer_fp32_execution_root_20261005_v1/result/RESULT.json'
GATE_SHA = '8601b1138c3f7ab65cb1525d5d8274a8e14e22731cd060801eec437f6169806d'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SUPERVISOR_RELATIVE = 'shared_backbone_private_transfer_complete_cost_queue_preparation_20261005_v3/queue.py'
SUPERVISOR_SHA = '7b195b600ad3cc57c4cf51a1da927164173ca2b0e80e96cf2b6c6449ce7ad468'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def relative_path(value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Require a relative path in the authorized phase')
    return path


def phase_file(value):
    path=(PHASE/relative_path(value)).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve(strict=True)) or not path.is_file():
        raise ValueError('Phase input leaves the project')
    return path


def verify_packet(expected):
    manifest=ROOT.parent/'MANIFEST.json'
    if sha(manifest)!=expected: raise ValueError('Reviewed explicit replication source manifest changed')
    for row in read(manifest)['files']:
        path=(ROOT.parent/relative_path(row['path'])).resolve(strict=True)
        if not path.is_relative_to(ROOT.parent) or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Replication source bytes changed: '+row['path'])


def verify_training():
    if sha(SOURCE/'SOURCE_MANIFEST.json')!=SOURCE_SHA or sha(phase_file(GATE_RELATIVE))!=GATE_SHA:
        raise ValueError('Exact already-qualified v2 source/gate changed')
    for row in read(SOURCE/'SOURCE_MANIFEST.json')['files']:
        if sha(SOURCE/row['path'])!=row['sha256']: raise ValueError('Qualified training file changed')


def physical_host():
    if platform.system()!='Linux' or socket.gethostname()!='anogena-2-0':
        raise ValueError('Only the authorized one-GPU scientific host may run this queue')
    result=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=30)
    if result.stdout.split()!=[GPU_UUID]: raise ValueError('Authorized singleton physical GPU differs')


def fresh_phase_directory(path):
    path=Path(path).resolve()
    if path.exists() or not path.is_relative_to(PHASE.resolve(strict=True)) or not path.parent.is_dir():
        raise ValueError('Require a fresh directory in the authorized phase')
    return path


def check_environment(overrides):
    allowed={'CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'PYTHONPATH','PYTHONDONTWRITEBYTECODE'}
    if set(overrides)-allowed or any(not isinstance(v,str) for v in overrides.values()):
        raise ValueError('Only explicit reviewed runtime environment overrides are allowed')
    if overrides.get('CUDA_VISIBLE_DEVICES')!='0': raise ValueError('Only authorized CUDA device0')
    environment=dict(os.environ,**overrides)
    environment.pop('PYTHONHOME',None)
    return environment


def supervisor():
    path=phase_file(SUPERVISOR_RELATIVE)
    if sha(path)!=SUPERVISOR_SHA:raise ValueError('Already reviewed v3 stdlib process supervisor changed')
    spec=importlib.util.spec_from_file_location('private_transfer_owned_supervisor_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
