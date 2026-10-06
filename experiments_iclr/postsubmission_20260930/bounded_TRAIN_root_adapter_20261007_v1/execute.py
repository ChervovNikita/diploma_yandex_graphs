"""Bind explicit TRAIN jobs to the existing reviewed child supervisor once."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
OLD = PHASE / 'citeseer_known_ranking_control_matched_reference_owned_preparation_20261006_v2/owner.py'
OLD_SHA = '1b1edf43a895600515bbf340fbc813b5dbc48e325ef73603fc5118146d4d5fdf'
HELPER = PHASE / 'shared_private_transfer_gpu77_qualification_preparation_20261005_v3/ownership_helpers.py'
HELPER_SHA = 'e71503c87865546319cddbbf7a4f9f15d13cdf9e4875e406d65de21e64e047fd'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def confined(relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Expected phase relative file')
    path = (PHASE / rel).resolve(strict=True)
    if not path.is_relative_to(PHASE.resolve()) or not path.is_file():
        raise ValueError('Input is outside authorized phase')
    return path

def physical():
    if Path.cwd().resolve() != REPO or socket.gethostname() != 'anogena-2-0':
        raise ValueError('Wrong authorized host/repository')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).split() != [GPU]:
        raise ValueError('Wrong allocation GPU')

def load(name, path, expected):
    if sha(path) != expected:
        raise ValueError('Existing supervisor changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    physical()
    if sha(args.config) != args.config_sha256:
        raise ValueError('Root bound config changed')
    cfg = read(args.config)
    if cfg['execution_enabled'] is not True or sha(__file__) != cfg['adapter_sha256']:
        raise ValueError('Explicit root adapter release required')
    for row in cfg['bindings']:
        if sha(confined(row['path'])) != row['sha256']:
            raise ValueError('Root bound source or evidence changed')
    root = PHASE / cfg['execution_root_relative']
    if not root.resolve().is_relative_to(PHASE) or (root / 'owner').exists():
        raise ValueError('Fresh one shot owner only')
    root = root / 'owner'
    root.mkdir()
    (root / 'logs').mkdir()
    helper = load('bounded_existing_helper', HELPER, HELPER_SHA)
    helper.GPU = GPU
    old = load('bounded_existing_supervisor', OLD, OLD_SHA)
    owner = helper.identity(os.getpid())
    if owner is None or owner['sid'] != os.getpid() or owner['pgid'] != os.getpid():
        raise ValueError('Detached owner session required')
    write(root / 'START.json', {'UTC': helper.now(), 'identity': owner, 'config_sha256': args.config_sha256, 'TRAIN_only': True})
    environment = dict(os.environ, **cfg['environment'])
    environment.pop('PYTHONHOME', None)
    completed = []
    try:
        for entry in cfg['entries']:
            physical()
            job_path = confined(entry['job_relative'])
            if sha(job_path) != entry['job_sha256']:
                raise ValueError('Root bound job changed')
            job = read(job_path)
            if job['TEST_access'] is not False or job['VALID_values_access'] is not False or job['fits_authorized'] is not False or job['retry'] is not False:
                raise ValueError('TRAIN only qualification or discarded cost required')
            output = Path(entry['output_directory'])
            if output.exists() or not output.resolve().is_relative_to(PHASE) or not output.parent.is_dir():
                raise ValueError('Fresh output required')
            context = SimpleNamespace(REPO=REPO, SOURCE=PHASE/cfg['source_relative'], SOURCE_SHA=cfg['source_sha256'], GPU_UUID=GPU, GPU_UUIDS=(GPU,), phase_file=confined, physical_host=physical, sha=sha, write=write)
            terminal = old.run_fit(helper, root, entry, {'resource_limits': cfg['resource_limits']}, environment, output, context)
            write(root / (entry['cell_id'] + '.TERMINAL.json'), terminal)
            ident = terminal['raw_identity_observation']
            absent = ident is not None and helper.identity(ident['PID']) is None
            rows = helper.query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], 10)
            no_cuda = ident is not None and not any(len(parts)>1 and parts[1].strip()==str(ident['PID']) for parts in (row.split(',') for row in rows))
            write(root / (entry['cell_id'] + '.PHYSICAL_TERMINAL.json'), {'owned_PID_absent': absent, 'owned_PID_no_CUDA_rows': no_cuda, 'PID': ident['PID'] if ident else None})
            if terminal['exit_code'] != 0 or terminal['reason'] is not None or terminal['signals_sent'] or not terminal['terminal_wait_observed'] or not absent or not no_cuda or (output/'FAILURE.json').exists():
                raise ValueError('TRAIN job failed clean terminal closure. Preserve all outputs')
            freeze_path = output / entry['completion_filename']
            freeze = read(freeze_path)
            for key, value in entry['completion_requires'].items():
                if freeze.get(key) != value:
                    raise ValueError('TRAIN job completion differs: ' + key)
            completed.append({'cell_id': entry['cell_id'], 'completion_relative': str(freeze_path.relative_to(PHASE)), 'sha256': sha(freeze_path), 'terminal': terminal})
            write(root / 'PROGRESS.json', {'UTC': helper.now(), 'completed': completed, 'total': len(cfg['entries']), 'scores_read': False})
        write(root / 'COMPLETE.json', {'UTC': helper.now(), 'completed': completed, 'TRAIN_only': True, 'fits': 0, 'retry': False})
    except (Exception, KeyboardInterrupt) as error:
        write(root / 'FAILURE.json', {'UTC': helper.now(), 'error': type(error).__name__ + ': ' + str(error), 'completed': completed, 'preserve_outputs': True, 'retry': False})
        raise

if __name__ == '__main__':
    main()
