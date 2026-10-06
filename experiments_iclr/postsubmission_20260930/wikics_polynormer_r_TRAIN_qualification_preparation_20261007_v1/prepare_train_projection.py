#!/usr/bin/env python3
"""Acquisition-only role projection; not a model or qualification invocation."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    repo = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
    phase = repo/'experiments_iclr/postsubmission_20260930'
    if Path.cwd() != repo or socket.gethostname() != 'anogena-2-0': raise ValueError('Exact authorized allocation required')
    if subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact singleton allocation GPU inventory required')
    dataset = phase/'wikics_official_acquisition_root_20261007_v1'
    source_manifest = dataset/'AVAILABLE_MANIFEST.json'
    if sha(source_manifest) != '90eb221f523f892cdc9418ab5fb6a199dc8614e62a73774a2cd471a95d61f348': raise ValueError('Official source authority changed')
    record = json.loads(source_manifest.read_text())['available']; source = phase/record['path']
    if sha(source) != record['sha256']: raise ValueError('Original safe acquisition artifact changed')
    import torch
    if str(torch.__version__) != '2.1.2+cu118': raise ValueError('Existing CPU acquisition runtime required')
    data = torch.load(source, map_location='cpu', weights_only=True)
    # This separate acquisition projection deserializes the authorized safe
    # TRAIN/VALID container, but uses only TRAIN fields; it performs no scoring.
    selected = {key: data[key] for key in ('x', 'edge_index', 'train_ids', 'train_y')}
    output = dataset/'TRAIN_only_projection_20261007_v1'; output.mkdir()
    file = output/'TRAIN_ONLY.pt'; torch.save(selected, file); file.chmod(0o444)
    result = {'scope': 'CPU_acquisition_only_TRAIN_role_projection', 'source_manifest': {'path': str(source_manifest.relative_to(phase)), 'sha256': sha(source_manifest)},
        'source_available_sha256': record['sha256'], 'projection_program_sha256': sha(__file__),
        'available': {'path': str(file.relative_to(phase)), 'sha256': sha(file), 'bytes': file.stat().st_size},
        'fields': list(selected), 'nodes': 11701, 'features': 300, 'classes': 10, 'directed_edges': selected['edge_index'].shape[1], 'TRAIN': len(selected['train_ids']),
        'VALID_values_used': False, 'TEST_values_used': False, 'qualified_reader_deserializes_VALID_TEST_values': False,
        'model_forward_or_training': False, 'GPU_tensor_work': False, 'source_preserved_sha256': sha(source),
        'inclusive_CPU_seconds': time.monotonic()-started}
    manifest = output/'TRAIN_ONLY_MANIFEST.json'; manifest.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n'); manifest.chmod(0o444)
    print(json.dumps({'manifest_path': str(manifest.relative_to(phase)), 'manifest_sha256': sha(manifest), 'projection': result}), flush=True)


if __name__ == '__main__': main()
