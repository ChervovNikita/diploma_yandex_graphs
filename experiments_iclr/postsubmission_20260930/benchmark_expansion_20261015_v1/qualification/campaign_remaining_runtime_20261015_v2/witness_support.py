"""Stdlib-only source/admission helpers for two finite allocation checks."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
E = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
HERE = Path(__file__).resolve().parent
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
GiB = 1024**3


def require(value, message):
    if not value:
        raise ValueError(message)


def guard():
    require(socket.gethostname() == 'anogena-2-0', 'Excluded route: exact allocation hostname required')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
            text=True, timeout=10).splitlines() == [GPU], 'Excluded route: exact sole physical GPU required')


def inside(path, root=ROOT):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), 'Path escapes the admitted repository')
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def verify(pin, root=ROOT):
    path = inside(pin['path'], root)
    require(receipt(path) == pin, 'Pinned bytes changed: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def module(path, name):
    path = Path(path).resolve()
    require(name not in sys.modules, 'Fresh source module required: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def manifest(pin, phase=None):
    path = verify(pin)
    value = read(path)
    for row in value['files']:
        member = (path.parent / row['path']).resolve()
        require(member.is_relative_to(path.parent), 'Manifest member escapes package')
        actual = receipt(member)
        require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Manifest member changed: ' + str(member))
    bindings = path.parent / 'SOURCE_BINDINGS.json'
    if phase is not None and bindings.exists():
        require(any(row['path'] == bindings.name for row in value['files']), 'Unpinned dependency list')
        for row in read(bindings)['files']:
            p = inside(phase / row['path_relative_to_phase'])
            actual = receipt(p)
            require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Pinned dependency changed: ' + str(p))
    return path


def own_source(expected):
    require(receipt(HERE / 'MANIFEST.json')['sha256'] == expected, 'Exact root-reviewed witness source required')
    manifest(receipt(HERE / 'MANIFEST.json'))
    inputs = read(HERE / 'INPUTS.json')
    for pin in inputs['source_packages']:
        manifest(pin, E.parent)
    for pin in inputs['source_files']:
        verify(pin)
    return inputs


def limits(case):
    require(case in ('bound_observer_molbace', 'buddy_selected_VALID'), 'Unknown finite check')
    return {'GPU_reservation_bytes': (4 if case == 'bound_observer_molbace' else 40)*GiB,
            'host_RSS_reservation_bytes': (4 if case == 'bound_observer_molbace' else 16)*GiB,
            'disk_reservation_bytes': (1 if case == 'bound_observer_molbace' else 8)*GiB,
            'wall_seconds': 180 if case == 'bound_observer_molbace' else 600,
            'supervisor_wall_seconds': 190 if case == 'bound_observer_molbace' else 610}
