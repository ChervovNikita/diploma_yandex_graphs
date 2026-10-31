"""Durable metadata and Linux process/lock custody; Python standard library only."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
E = HERE.parent
PORT = E / 'server51_campaign_port_v5'
SMOKE = E / 'qualification/all_fit_smoke_20261016_v5'
RESOURCES = E / 'qualification/all_fit_resources_20261016_v1'
WORKERS = ('allocation/gpu0', 'server51/gpu0', 'server51/gpu1')
COUNTS = {'allocation/gpu0': 2154, 'server51/gpu0': 2153, 'server51/gpu1': 2153}
GiB = 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify(receipt, root=None):
    path = Path(receipt['path']).resolve()
    if root is not None:
        require(path.is_relative_to(Path(root).resolve()), 'Receipt escapes project')
    actual = pin(path)
    require(all(actual[key] == receipt[key] for key in ('bytes', 'sha256') if key in receipt), 'Changed receipt: ' + str(path))
    return path


def token(value):
    return hashlib.sha256(value.encode()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def source_manifest():
    manifest = read(HERE / 'MANIFEST.json')
    for row in manifest['files']:
        actual = pin(HERE / row['path'])
        require(all(actual[key] == row[key] for key in ('bytes', 'sha256')), 'Execution source changed')
    return pin(HERE / 'MANIFEST.json')


def identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    path = Path('/proc') / str(pid)
    raw = (path / 'stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return {'pid': pid, 'uid': path.stat().st_uid, 'start_ticks': int(fields[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def same_process(value):
    try:
        return value == identity(value['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return False


def lock(path, blocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:
        handle.close(); return None
    return handle


def locked(path):
    handle = lock(path)
    if handle is None:
        return True
    handle.close(); return False


def environment(worker):
    host = module(PORT / 'host_binding.py', '_campaign_host_binding')
    path = host.config_path(worker)
    config, receipt = host.load_config(path)
    os.environ.update(BENCHMARK_HOST_CONFIG=str(path), BENCHMARK_HOST_CONFIG_SHA256=receipt['sha256'],
                      CUDA_VISIBLE_DEVICES=config['physical_gpu_uuid'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    host.guard(config)
    return config, receipt


def global_barrier(path, plan, summary_pin):
    """A count is insufficient: exact IDs and both local inventory/summary hashes bind GO."""
    if not Path(path).exists():
        return False
    value = read(path)
    require(value.get('status') == 'PASS' and value.get('fit_count') == 6460, 'Global smoke has not passed')
    require(value.get('inventory_sha256s', {}).get(plan['worker']) == plan['inventory']['sha256'], 'Global smoke inventory differs')
    require(value.get('worker_summary_sha256s', {}).get(plan['worker']) == summary_pin['sha256'], 'Global smoke summary differs')
    require(value.get('schema') == 'all-fit-global-smoke-pass-v1', 'Wrong global smoke schema')
    require(value.get('canonical_fit_ids_sha256') == plan['canonical_fit_ids_sha256'], 'Global canonical fit identity differs')
    inventory = read(verify(plan['inventory'], plan['project_root']))
    require(value.get('canonical_fit_ids') == inventory['canonical_fit_ids'], 'Global exact fit IDs differ')
    require(len(value['canonical_fit_ids']) == len(set(value['canonical_fit_ids'])) == 6460, 'Missing/duplicate global fit')
    require(sorted(row['id'] for row in value.get('coverage', [])) == value['canonical_fit_ids'], 'Global receipt coverage differs')
    require(value.get('TEST_accessed') is False and value.get('class_substitution') is False, 'Global scope differs')
    require(set(value['inventory_sha256s']) == set(WORKERS) and set(value['worker_summary_sha256s']) == set(WORKERS), 'Missing global worker')
    require(all(value['source_manifest'][key] == plan['smoke_source_manifest'][key] for key in ('bytes', 'sha256')), 'Global smoke validator source differs')
    verify(plan['smoke_source_manifest'], plan['project_root'])
    return True
