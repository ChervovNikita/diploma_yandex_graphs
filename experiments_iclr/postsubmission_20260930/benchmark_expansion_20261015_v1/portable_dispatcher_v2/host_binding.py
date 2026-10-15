"""Explicit per-worker host identity; no scientific imports or implicit relocation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import subprocess

HERE = Path(__file__).resolve().parent
WORKERS = ('allocation/gpu0', 'server77/gpu0', 'server77/gpu1')
ALLOCATION_ROOT = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
ALLOCATION_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PHASE_RELATIVE = 'experiments_iclr/postsubmission_20260930'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def normalized_absolute(value):
    require(isinstance(value, str) and value.startswith('/'), 'Absolute POSIX path required')
    path = PurePosixPath(value)
    require(str(path) == value and '..' not in path.parts, 'Normalized path without traversal required')
    return path


def confined(path, root):
    path = Path(path).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Path escapes configured project root')
    return path


def pin(path):
    data = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def verify(receipt, root, parse_json=True):
    path = confined(receipt['path'], root)
    actual = pin(path)
    require(all(actual[k] == receipt[k] for k in ('bytes', 'sha256') if k in receipt), 'Host authority bytes changed')
    return json.loads(path.read_text()) if parse_json else actual


def validate_config(config):
    require(config['schema'] == 'portable-campaign-host-v1', 'Typed host configuration required')
    require(config['worker'] in WORKERS, 'Unknown physical worker')
    root = normalized_absolute(config['project_root'])
    phase = PurePosixPath(config['phase_relative'])
    require(not phase.is_absolute() and '..' not in phase.parts, 'Root-relative phase required')
    require(config['maximum_children_per_physical_GPU'] == 2, 'Keep the two-compatible-child ceiling')
    require(type(config['host_enabled']) is bool, 'Explicit host admission flag required')
    require(config['logical_device'] == 'cuda:0', 'Each UUID-masked worker uses logical cuda:0')
    require(config['physical_gpu_index'] == int(config['worker'][-1]), 'Worker/index mismatch')
    if config['worker'] == 'allocation/gpu0':
        require(str(root) == ALLOCATION_ROOT and str(phase) == PHASE_RELATIVE, 'Allocation roots are fixed')
        require(config['expected_hostname'] == 'anogena-2-0' and config['physical_gpu_uuid'] == ALLOCATION_UUID,
                'Preserve literal allocation identity')
        require(config['expected_gpu_uuids_by_index'] == [ALLOCATION_UUID], 'Allocation must have its sole GPU')
    if config['host_enabled']:
        require(bool(config['expected_hostname']) and bool(config['physical_gpu_uuid']), 'Enabled host requires actual identity')
        uuids = config['expected_gpu_uuids_by_index']
        require(isinstance(uuids, list) and len(uuids) > config['physical_gpu_index']
                and uuids[config['physical_gpu_index']] == config['physical_gpu_uuid']
                and len(set(uuids)) == len(uuids), 'Explicit complete device identity table required')
    return config


def config_path(worker):
    require(worker in WORKERS, 'Unknown worker')
    return HERE / 'hosts' / (worker.replace('/', '_') + '.json')


def load_config(path=None, expected_sha256=None):
    path = Path(path or os.environ.get('BENCHMARK_HOST_CONFIG', config_path('allocation/gpu0')))
    # This small bootstrap configuration is the only file read before identity.
    require(path.resolve().parent == HERE / 'hosts', 'Use the bundled reviewed host configuration')
    data = path.read_bytes()
    expected = expected_sha256 or os.environ.get('BENCHMARK_HOST_CONFIG_SHA256')
    if expected:
        require(hashlib.sha256(data).hexdigest() == expected, 'Host configuration hash changed')
    config = validate_config(json.loads(data))
    require(path.resolve() == config_path(config['worker']), 'Worker/config filename mismatch')
    return config, pin(path)


def check_observed_identity(config, hostname, gpu_uuids):
    """Pure comparison, separately exercised without probes."""
    require(config['host_enabled'], 'Host disabled; no identity/resource probe or native work authorized')
    require(hostname == config['expected_hostname'], 'Unexpected scientific hostname')
    require(gpu_uuids == config['expected_gpu_uuids_by_index'], 'Unexpected physical GPU identity/index order')


def guard(config=None):
    config, receipt = load_config() if config is None else (validate_config(config), None)
    # Disabled77 exits before hostname lookup, nvidia-smi, or authority reads.
    require(config['host_enabled'], 'Host disabled; explicit reauthorization and qualification required')
    hostname = socket.gethostname()
    require(hostname == config['expected_hostname'], 'Unexpected scientific hostname; no further project access')
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                   text=True, timeout=10).splitlines()
    check_observed_identity(config, hostname, uuids)
    root = Path(config['project_root'])
    expected_here = root / config['phase_relative'] / 'benchmark_expansion_20261015_v1/portable_dispatcher_v2'
    require(HERE == expected_here, 'Guard source is outside the declared deployment')
    if config['worker'].startswith('server77/'):
        for field in ('human_host_reauthorization', 'identity_receipt', 'resource_receipt'):
            require(config.get(field), 'Future host lacks ' + field)
            value = verify(config[field], root)
            require(value.get('status') == 'PASS' and value.get('worker') == config['worker'], 'Wrong host authority scope')
            if field == 'human_host_reauthorization':
                require(value.get('user_explicit_host_reauthorization') is True and value.get('human_instruction_evidence'),
                        'A configuration boolean is not user reauthorization')
                verify(value['human_instruction_evidence'], root, parse_json=False)
            else:
                require(value.get('hostname') == hostname and value.get('GPU_UUID') == config['physical_gpu_uuid'],
                        'Identity/resource receipt is for another physical device')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    require(mask == config['physical_gpu_uuid'] or (config['worker'] == 'allocation/gpu0' and mask == '0'),
            'Use the exact UUID mask (or existing sole-allocation index0 mask) for logical cuda:0')
    return config


def bound_project_root(path):
    config, _ = load_config()
    root = Path(config['project_root']).resolve()
    require(Path(path).resolve() == root, 'Runtime project root differs from guarded host configuration')
    return root


def validate_host_job(job, root):
    """Check runtime field confinement; provenance remains immutable history."""
    from descriptor_binding import audit_runtime_paths
    config, _ = load_config()
    require(Path(root).resolve() == Path(config['project_root']).resolve(), 'Wrong descriptor root')
    require(job.get('TEST_access') is False and job.get('device') == 'cuda:0', 'Preserve TEST closure/logical device')
    for row in audit_runtime_paths(job, config['project_root'], config['phase_relative']):
        confined(row['path'], root)  # Resolve real links on the executing host; never create path aliases.
