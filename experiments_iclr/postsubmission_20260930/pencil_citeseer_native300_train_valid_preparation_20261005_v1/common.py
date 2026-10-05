"""Standard-library gates and custody for an unlaunched native300 comparator."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
PREP = PHASE / 'pencil_collab_paired_predictive_preparation_20261004_v3'
ADAPTER = PHASE / 'pencil_citeseer_heart_competitor_source_plan_20261005_v1'
NCN = PHASE / 'citeseer_heart_ncn_trainval_runner_source_20261005_v1'
RESOURCE = PHASE / 'pencil_citeseer_heart_resource_plan_audit_20261005_v1'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate(release_path, release_sha256, seed, output):
    require(platform.system() == 'Linux' and Path.cwd().resolve() == REPO
            and socket.gethostname() == 'anogena-2-0', 'Authorized Linux repository/host required')
    require(HERE.resolve().is_relative_to(PHASE) and release_path.resolve().is_relative_to(PHASE),
            'Prepared source/release leaves authorized phase')
    require(sha(release_path) == release_sha256, 'Actual reviewed release differs')
    release = json.loads(release_path.read_text())
    require(release.get('root_source_review_approved') is True
            and release.get('scientific_comparator_authorized') is True
            and release.get('TEST_authorized') is False, 'Scientific root release absent')
    require(sha(HERE / 'MANIFEST.json') == release['source_manifest_sha256'],
            'Reviewed substantive source manifest differs')
    manifest = json.loads((HERE / 'MANIFEST.json').read_text())
    for row in manifest['files']:
        require(sha(HERE / row['path']) == row['sha256'], 'Prepared source packet changed')
    plan = json.loads((HERE / 'PLAN.json').read_text())
    require(plan['seeds'] == [0, 1, 2] and seed in plan['seeds'], 'Frozen seeds0/1/2 required')
    require(release['plan_sha256'] == sha(HERE / 'PLAN.json')
            and release['caps'] == plan['caps_proposed_for_root_review'], 'Frozen plan/caps differ')
    expected_output = PHASE / release['execution_output_phase_relative'] / ('seed_' + str(seed)) / 'run01'
    require(output.resolve() == expected_output.resolve(), 'Worker output differs from the uniquely released cohort')
    devices = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=30)
    require(devices.stdout.split() == [GPU_UUID], 'Authorized singleton GPU UUID differs')
    require(not output.exists() and output.resolve().is_relative_to(PHASE)
            and output.parent.is_dir(), 'Require fresh output inside owned fit parent')
    require(sha(RESOURCE / 'MANIFEST.json') ==
            'f1a5cae24d746f32360bafe7dcb1eb7557356a05cbb53b28d41a28dcd93894a3',
            'Previously reviewed native closure changed')
    native_bindings = json.loads((RESOURCE / 'MANIFEST.json').read_text())
    for row in native_bindings['external_source_files']:
        path = PHASE / row['phase_relative']
        require(path.resolve().is_relative_to(PHASE) and sha(path) == row['sha256'],
                'Pinned native closure/adapter/config bytes changed')
    require(sha(ADAPTER / 'FEATURE_ENABLED_TRAIN_VALID_PROPOSAL.yaml') == plan['config_sha256'],
            'Frozen recipe configuration changed')
    resource_result = PHASE / 'pencil_citeseer_zero_update_resource_execution_20261005_v1/run01/RESULT.json'
    require(sha(resource_result) == plan['resource_RESULT_sha256'], 'Measured resource/readiness basis differs')
    require(json.loads(resource_result.read_text())['status'] == 'PASS_ZERO_UPDATE_RESOURCE_ONLY',
            'Complete measured resource PASS required; no checkpoint is read or reused')
    available = PHASE / plan['available_manifest_phase_relative']
    feature = PHASE / 'citeseer_feature_qualification_20261005_v2/RESULT.json'
    require(sha(available) == plan['available_manifest_sha256']
            and sha(feature) == plan['feature_qualification_sha256'], 'Authenticated data authority changed')
    dependency_path = PHASE / plan['dependency_binding_phase_relative']
    require(sha(dependency_path) == plan['dependency_binding_sha256'], 'One-GPU dependency admission changed')
    dependency = json.loads(dependency_path.read_text())
    require(dependency['host'] == 'anogena-2-0'
            and dependency['one_gpu_native_pencil_import_pass'] is True
            and dependency['installed_file_inventory_complete'] is True
            and dependency['PYTHONPATH'] == os.environ.get('PYTHONPATH'), 'Dependency route/path differs')
    for row in dependency['dependency_files']:
        path = Path(row['path'])
        require(path.resolve().is_relative_to(REPO) and sha(path) == row['sha256'],
                'Admitted ancillary dependency file changed')
    require(os.environ.get('RANK') == os.environ.get('LOCAL_RANK') == '0'
            and os.environ.get('WORLD_SIZE') == '1', 'Exactly one directly supervised rank')
    return plan, release, dependency, native_bindings, available


def file_row(path):
    return dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path))


def output_bytes(output):
    return sum(path.stat().st_size for path in output.rglob('*') if path.is_file())


def binary(path, writer, check_resources):
    temporary = path.with_suffix(path.suffix + '.tmp')
    require(not temporary.exists(), 'Do not overwrite a partial scientific artifact')
    with temporary.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    check_resources()
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    check_resources()


def utc():
    return datetime.now(timezone.utc).isoformat()
