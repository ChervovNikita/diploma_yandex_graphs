"""Small shared metadata helpers; no numerical imports or remote operations."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
PORT = E / 'server51_campaign_port_v1'
PORT_SHA = '5107ebd694b1333629ec306808f64eb3e63c2b9fc24e4f4e3cbab7f0649b9209'
CANONICAL_SHA = 'be7bfe7c3beb5e5c9e639b53404455ee8b92b20f4dc038b05224fe141782ff64'
WORKERS = ('allocation/gpu0', 'server51/gpu0', 'server51/gpu1')
COUNTS = {'allocation/gpu0': 2154, 'server51/gpu0': 2153, 'server51/gpu1': 2153}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def inside(path, root):
    path = Path(path).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Path outside project: ' + str(path))
    return path


def verify(value, root=None):
    path = inside(value['path'], root) if root is not None else Path(value['path']).resolve()
    actual = pin(path)
    require(all(actual[k] == value[k] for k in ('bytes', 'sha256') if k in value), 'Changed receipt: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def read_pinned(path, sha256, root=None):
    value = pin(inside(path, root) if root else path)
    require(value['sha256'] == sha256, 'Input hash changed: ' + str(path))
    return read(path), value


def write(path, value, *, exclusive=False):
    path = Path(path)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
        with temp.open('w') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def guard(root, worker=None):
    host = module(PORT / 'host_binding.py', '_all_fit_smoke_host')
    config, config_pin = host.load_config()
    host.guard(config)
    require(Path(root).resolve() == Path(config['project_root']), 'Wrong project root')
    if worker is not None:
        require(config['worker'] == worker, 'Wrong physical worker')
    return config, config_pin


def own_source():
    value = read(HERE / 'MANIFEST.json')
    require(value['schema'] == 'all-fit-two-update-smoke-source-v1', 'Wrong smoke source')
    for row in value['files']:
        actual = pin(inside(HERE / row['path'], HERE))
        require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Smoke source changed')
    return pin(HERE / 'MANIFEST.json')


def fit_directory(fit_id):
    return hashlib.sha256(fit_id.encode()).hexdigest()
