"""Admitted CPU construction/storage inventory only: no graph or training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--supervisor', required=True, type=Path)
    args = parser.parse_args()
    phase = Path(__file__).resolve().parents[1]
    request = args.request.resolve()
    supervisor = args.supervisor.resolve()
    if not request.is_relative_to(phase) or not supervisor.is_relative_to(phase):
        raise ValueError('Escaping request or supervisor')
    bound = Path(os.environ['GNNM_BOUND_START_JSON']).resolve()
    if not bound.is_relative_to(phase):
        raise ValueError('Escaping bound provenance')
    provenance = json.loads(bound.read_text())
    if not bound.is_relative_to(phase) or provenance['root_request']['sha256'] != sha(request):
        raise ValueError('Unbound request')
    if not (supervisor / 'environment.json').is_file():
        raise ValueError('Missing live supervisor')
    r = json.loads(request.read_text())
    if r['operation'] != 'fixed_CPU_storage_inventory' or r['data_read_or_training_admitted']:
        raise ValueError('Wrong operation')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU only')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if gpu != 'GPU-44039938-fd82-41d2-fefd-de71514e2fac':
        raise ValueError('Wrong physical allocation')
    for row in r['sources']:
        path = (phase / row['path']).resolve()
        if not path.is_relative_to(phase) or sha(path) != row['sha256']:
            raise ValueError('Source mismatch: ' + row['path'])
    output = (phase / r['output_relative']).resolve()
    if not output.is_relative_to(phase) or output.exists():
        raise ValueError('Require new confined output')
    output.mkdir(parents=True)
    import torch
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    path = phase / 'coordinate_ensemble_source_v1/models.py'
    spec = importlib.util.spec_from_file_location('fixed_storage_models', path)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    started = time.monotonic()
    records = []
    for cell in r['models']:
        model = source.build_model(cell['arm'], cell['features'], cell['width'], cell['classes'],
                                   seed=17, permutation_seed=1017, members=4,
                                   num_layers=cell['depth'], dropout=0.2,
                                   hidden_dim_multiplier=1, normalization='layer', num_heads=8)
        report = source.storage_report(model)
        if any(t.device.type != 'cpu' for t in list(model.parameters()) + list(model.buffers())):
            raise ValueError('Non-CPU model')
        records.append({'cell': cell, 'storage': report})
        del model
    result = {'operation': r['operation'], 'source_hashes_verified': True,
              'request_sha256': sha(request), 'torch_version': torch.__version__,
              'CUDA_hidden': True, 'data_or_training_used': False,
              'seconds': time.monotonic() - started, 'models': records}
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'models': len(records), 'seconds': result['seconds'], 'complete': True}), flush=True)


if __name__ == '__main__':
    main()
