"""Metadata custody for a disabled CPU-only allocation input inspection."""
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
ROOT = Path(__file__).resolve().parent
EXECUTION = PHASE / 'private_transfer_pubmed_allocation_available_execution_root_20261005_v2'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
ARCHIVE = PHASE / 'citeseer_heart_official_acquisition_server_20261005_v1/withheld_inputs/HeaRT.tar.gz'
ARCHIVE_SHA = '7b7042476319a353bdb6c50b5f402b89b9006a2fde2d1258b7adcbd1d22629ba'
ARCHIVE_BYTES = 880240878
EXPECTED = {
    'train_pos.txt': 'c6de89d86371909f738d620846540168d4b6256fed88dc9d8ab3609cb5357fb4',
    'valid_pos.txt': '31f55d457ce8ea1a75b0b501e0707814e7968be1e9e0b01f0bbff2b8ef9cdc0b',
    'heart_valid_samples.npy': '3a2f9ea0c11dd5221d9a36771bcd0e2d273a4d059aaa2dd3dff03fc46e9e66e4',
    'gnn_feature': 'c895f9e8e2d96eae8d610e7be8740f77fe1cb6b40e02a73046b2f02aa63a8dd5',
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while part := stream.read(1024 * 1024):
            value.update(part)
    return value.hexdigest()


def confined(path, *, exists):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Explicit absolute allocation phase path required')
    if not path.is_relative_to(PHASE):
        raise ValueError('Path leaves the authorized project phase')
    current = PHASE
    for part in path.relative_to(PHASE).parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Symlink in deliberate file path')
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(PHASE.resolve(strict=True)):
        raise ValueError('Resolved file leaves the project phase')
    return resolved


def authorize(program, recipe_path, purpose):
    if platform.system() != 'Linux' or socket.gethostname() != 'anogena-2-0':
        raise ValueError('Authorized one-GPU allocation host only')
    if Path.cwd().resolve() != REPO or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Project cwd and explicitly CPU-only process required')
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=20).split()
    if uuids != [UUID]:
        raise ValueError('Exact singleton allocation UUID differs')
    confined(ROOT, exists=True)
    recipe_path = confined(recipe_path, exists=True)
    recipe = json.loads(recipe_path.read_text())
    if recipe.get('purpose') != purpose or purpose not in ('extract_available', 'inspect_available'):
        raise ValueError('Wrong input-inspection purpose')
    if recipe.get('source_review_approved') is not True or recipe.get('available_inspection_authorized') is not True:
        raise ValueError('Source and input inspection require root admission')
    if recipe.get('external_owned_bounds_confirmed') is not True:
        raise ValueError('Reviewed owned supervision required')
    if any(recipe.get(key) is not False for key in ('TEST_access', 'model_access', 'fit_admission', 'network_access', 'retry')):
        raise ValueError('Input inspection only, no model/TEST/network/retry')
    if recipe.get('members') != ['dataset/pubmed/' + name for name in EXPECTED]:
        raise ValueError('Exactly four original TRAIN/VALID input members required')
    if recipe.get('archive') != {'path': str(ARCHIVE), 'sha256': ARCHIVE_SHA, 'bytes': ARCHIVE_BYTES}:
        raise ValueError('Authenticated resident archive identity differs')
    if recipe.get('bounds') != {'soft_seconds': 600, 'hard_seconds': 900, 'RSS_bytes': 2147483648, 'output_bytes': 134217728}:
        raise ValueError('Fixed CPU inspection bounds differ')
    manifest = confined(ROOT / 'SOURCE_MANIFEST.json', exists=True)
    if sha(manifest) != recipe.get('source_manifest_sha256'):
        raise ValueError('Reviewed source manifest changed')
    for row in json.loads(manifest.read_text())['files']:
        path = confined(ROOT / row['path'], exists=True)
        if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Reviewed source closure changed')
    if sha(program) != recipe.get('program_sha256'):
        raise ValueError('Reviewed executable changed')
    for key in ('root_review', 'owned_supervision'):
        record = recipe.get(key, {})
        if record.get('approved') is not True or not record.get('evidence'):
            raise ValueError('Concrete root review/owned bounds absent')
        for row in record['evidence']:
            path = confined(PHASE / row['path'], exists=True)
            if not path.is_file() or sha(path) != row['sha256']:
                raise ValueError('Root admission evidence changed')
    if recipe.get('execution_directory') != str(EXECUTION):
        raise ValueError('Fresh input execution identity differs')
    return recipe
