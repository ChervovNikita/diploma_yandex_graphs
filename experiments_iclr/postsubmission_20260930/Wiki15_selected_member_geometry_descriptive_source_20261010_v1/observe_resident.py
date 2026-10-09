"""Read only finite Wiki15 file custody, NPY headers and checkpoint key names.

Stdlib only. No ndarray/tensor/checkpoint deserialization or model import.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickletools
import resource
import socket
import struct
import subprocess
import time
import zipfile


def observe(inputs):
    began = time.monotonic()
    assert socket.gethostname() == inputs['hostname'], 'Wrong host'
    inventory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=10).splitlines()
    assert inventory == inputs['physical_gpu_inventory'], 'Wrong physical GPU inventory'
    root = Path(inputs['repository'])
    os.chdir(root)
    assert Path.cwd().resolve() == root.resolve()
    phase = Path(inputs['phase'])
    assert phase.is_relative_to(root)
    bytes_hashed = 0

    def bound(row):
        nonlocal bytes_hashed
        assert time.monotonic() - began < 50, 'Metadata wall budget exceeded'
        path = (phase / row['path']).resolve()
        assert path.is_relative_to(phase) and path.is_file(), 'Absent or invalid file: ' + row['path']
        assert path.stat().st_size == row['bytes'], 'Recorded size changed'
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                digest.update(chunk)
                bytes_hashed += len(chunk)
        assert digest.hexdigest() == row['sha256'], 'Recorded hash changed: ' + row['path']
        return path

    def header(stream):
        assert stream.read(6) == b'\x93NUMPY'
        version = tuple(stream.read(2))
        assert version in ((1, 0), (2, 0), (3, 0))
        width = 2 if version == (1, 0) else 4
        count = struct.unpack('<H' if width == 2 else '<I', stream.read(width))[0]
        assert count <= 65536
        result = ast.literal_eval(stream.read(count).decode('utf-8' if version == (3, 0) else 'latin1'))
        assert isinstance(result, dict)
        return result

    source_files = []
    for row in inputs['source_files']:
        if (phase / row['path']).is_file():
            bound(row)
            source_files.append(dict(row, verified=True, resident=True))
        else:
            source_files.append(dict(row, verified=False, resident=False))
    results = []
    for record in inputs['cells']:
        archive = bound(record['raw_archive'])
        headers = {}
        with zipfile.ZipFile(archive) as package:
            for name in ('member_representations', 'member_logits', 'valid_ids', 'truth'):
                with package.open(name + '.npy') as stream:
                    headers[name] = header(stream)
            assert tuple(headers['member_representations']['shape']) == (4, 5274, 512)
            assert tuple(headers['member_logits']['shape']) == (4, 5274, 10)
            assert tuple(headers['valid_ids']['shape']) == tuple(headers['truth']['shape']) == (5274,)
            assert headers['member_representations']['descr'] == headers['member_logits']['descr'] == '<f4'
        checkpoint = bound(record['selected_checkpoint'])
        with zipfile.ZipFile(checkpoint) as package:
            names = [name for name in package.namelist() if name.endswith('/data.pkl')]
            assert len(names) == 1 and package.getinfo(names[0]).file_size <= 2 * 1024**2
            key_names = {arg for op, arg, _ in pickletools.genops(package.read(names[0]))
                         if isinstance(arg, str) and arg.startswith('models.0.body.pred_')}
        active = 'pred_global' if record['global_mode'] else 'pred_local'
        required = {'models.0.body.' + active + '.' + name for name in ('weight', 'r', 's', 'bias')}
        assert required <= key_names, 'Active exact head keys absent'
        results.append(dict(record, exact_file_bindings_verified=True, headers=headers,
                            active_head=active, checkpoint_head_key_names=sorted(key_names),
                            head_tensor_values_deserialized=False))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return dict(schema='Wiki15-resident-geometry-source-metadata-v1',
                UTC=datetime.now(timezone.utc).isoformat(), status='PASS',
                hostname=socket.gethostname(), physical_gpu_inventory=inventory,
                phase=str(phase),
                source_files=source_files, cells=results, bytes_hashed=bytes_hashed,
                wall_seconds=time.monotonic() - began,
                CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
                peak_RSS_bytes=usage.ru_maxrss * 1024,
                numerical_imports=False, tensor_or_ndarray_deserialization=False,
                model_calls=0, scores_recomputed=False, raw_arrays_transferred=False)


if __name__ == '__main__':
    # The transport prepends the finite immutable metadata dictionary.
    print(json.dumps(observe(INPUTS), sort_keys=True))
