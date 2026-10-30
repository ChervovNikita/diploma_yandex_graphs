"""Explicit, additive identity authority for the three authorized allocation GPUs.

The frozen native host module remains on disk unchanged. This module installs
only the actual new identity/configuration functions into its ordinary loaded
module object. Its descriptor validation and all scientific source stay intact.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PORT = E / 'server51_campaign_port_v6'
UUIDS = [
    'GPU-ff06ca1f-403a-f2d0-b584-71e177452667',
    'GPU-898f1f43-286a-0614-7317-ceb12eebaaf8',
    'GPU-976a5cd3-df12-2535-fa51-e3ea063688ba',
    'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',
    'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',
    'GPU-3db85505-d061-e45d-d154-e7e8c1cad497',
    'GPU-6c281e4f-b62c-f866-f7ea-fd3786643f31',
]
ONE_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SHARDS = ('allocation/gpu0', 'anogena/gpu1', 'anogena/gpu6')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def config(shard):
    require(shard in SHARDS, 'Only the three explicitly authorized devices are allowed')
    index = int(shard[-1])
    many = shard.startswith('anogena/')
    table = UUIDS if many else [ONE_UUID]
    return {'schema': 'allocation-three-device-host-v1', 'worker': shard,
            'project_root': str(ROOT), 'phase_relative': 'experiments_iclr/postsubmission_20260930',
            'expected_hostname': 'anogena-0' if many else 'anogena-2-0',
            'expected_gpu_uuids_by_index': table, 'physical_gpu_index': index,
            'physical_gpu_uuid': table[index], 'logical_device': 'cuda:0',
            'maximum_children_per_physical_GPU': 4, 'host_enabled': True,
            'campaign_GO': False}


def config_path(shard):
    config(shard)
    return HERE / 'hosts' / (shard.replace('/', '_') + '.json')


def validate_config(value):
    require(value == config(value.get('worker')), 'Exact authorized host configuration required')
    return value


def load_config(path=None, expected_sha256=None):
    path = Path(path or os.environ['BENCHMARK_HOST_CONFIG']).resolve()
    value = validate_config(json.loads(path.read_text()))
    require(path == config_path(value['worker']), 'Host configuration path differs')
    receipt = pin(path)
    expected = expected_sha256 or os.environ.get('BENCHMARK_HOST_CONFIG_SHA256')
    require(expected and receipt['sha256'] == expected, 'Pinned host configuration required')
    return value, receipt


def check_identity(value, hostname, table):
    validate_config(value)
    require(hostname == value['expected_hostname'], 'Wrong scientific hostname')
    require(table == value['expected_gpu_uuids_by_index'], 'Wrong complete indexed GPU inventory')


def guard(value=None):
    value = load_config()[0] if value is None else validate_config(value)
    require(socket.gethostname() == value['expected_hostname'], 'Wrong host before project access')
    table = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                    text=True, timeout=15).splitlines()
    check_identity(value, socket.gethostname(), table)
    require(HERE == ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/allocation_gpu1_gpu6_20261030_v1',
            'Host extension is outside its authorized repository deployment')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == value['physical_gpu_uuid'], 'Exact authorized UUID mask required')
    return value


def bind(shard):
    path = config_path(shard)
    os.environ.update(BENCHMARK_HOST_CONFIG=str(path), BENCHMARK_HOST_CONFIG_SHA256=pin(path)['sha256'],
                      CUDA_VISIBLE_DEVICES=config(shard)['physical_gpu_uuid'], OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                      PYTHONDONTWRITEBYTECODE='1')
    return guard()


def bound_project_root(path):
    guard()
    require(Path(path).resolve() == ROOT, 'Native root differs from the shared repository')
    return ROOT


def install_native_authority(composed):
    """Replace identity policy, never model/loss/data/checkpoint functions."""
    guard()
    path = PORT / 'host_binding.py'
    spec = importlib.util.spec_from_file_location('_allocation_extension_native_authority', path)
    authority = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(authority)
    for name in ('guard', 'load_config', 'validate_config', 'config_path', 'bound_project_root'):
        setattr(authority, name, globals()[name])
    sys.modules['_server51_composed_host_authority'] = authority
    if not composed:
        require('host_binding' not in sys.modules, 'Fresh native interpreter required')
        sys.modules['host_binding'] = authority
    return {'actual_identity_authority': pin(__file__), 'frozen_descriptor_validator': pin(path),
            'config': load_config()[1], 'composed': bool(composed),
            'scientific_source_modified': False}
