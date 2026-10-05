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
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO/'experiments_iclr/postsubmission_20260930'
SOURCE = PHASE/'shared_backbone_private_transfer_training_source_gpu77_20261005_v1'
SOURCE_SHA = '7f274c09bb317e6976d0c8b8636e779dc3ffcd2f2b76493d00f008b8bd08cebc'
GATE_RELATIVE = 'shared_private_transfer_gpu77_fp32_execution_root_20261005_v2/result/RESULT.json'
GATE_SHA = 'b7eae293ec549de0746bb951733f8df6e350808669bcfb53a938c28d0768e476'
GPU_UUIDS = ('GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998', 'GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced')
GPU_UUID = None
CANONICAL_PLAN_SHA = 'ae4b0f5c77cf48a86ccdbe51179fc95881157ac225f593c3992a5a5014d5346b'
SUPERVISOR_RELATIVE = 'shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py'
SUPERVISOR_SHA = 'e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd'


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
    manifest=ROOT/'SOURCE_MANIFEST.json'
    if sha(manifest)!=expected: raise ValueError('Reviewed pilot source manifest changed')
    for row in read(manifest)['files']:
        path=(ROOT/relative_path(row['path'])).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Pilot source bytes changed: '+row['path'])


def verify_training():
    if sha(SOURCE/'SOURCE_MANIFEST.json')!=SOURCE_SHA or sha(phase_file(GATE_RELATIVE))!=GATE_SHA:
        raise ValueError('Exact already-qualified v2 source/gate changed')
    for row in read(SOURCE/'SOURCE_MANIFEST.json')['files']:
        if sha(SOURCE/row['path'])!=row['sha256']: raise ValueError('Qualified training file changed')


def physical_host():
    if platform.system()!='Linux' or socket.gethostname()!='peptide':
        raise ValueError('Only the authorized peptide repository may run this queue')
    result=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=30)
    if tuple(result.stdout.split())!=GPU_UUIDS or GPU_UUID not in GPU_UUIDS:
        raise ValueError('Authorized two-GPU inventory or selected physical UUID differs')


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
    if GPU_UUID not in GPU_UUIDS or overrides.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:
        raise ValueError('Only the frozen block physical GPU UUID is visible')
    environment=dict(os.environ,**overrides)
    environment.pop('PYTHONHOME',None)
    return environment


def supervisor():
    path=phase_file(SUPERVISOR_RELATIVE)
    if sha(path)!=SUPERVISOR_SHA:raise ValueError('Already reviewed v3 stdlib process supervisor changed')
    spec=importlib.util.spec_from_file_location('private_transfer_owned_supervisor_v3',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.GPU=GPU_UUID
    return module


def bind_block(blocks):
    """Host-only binding to one complete prespecified block and physical GPU."""
    global GPU_UUID
    if blocks not in (['b1'],['b2']):
        raise ValueError('Each peptide queue must contain exactly b1 or b2')
    GPU_UUID=GPU_UUIDS[int(blocks[0][1])-1]
