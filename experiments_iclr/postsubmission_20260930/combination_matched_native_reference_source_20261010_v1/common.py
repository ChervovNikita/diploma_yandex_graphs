"""Metadata admission only; numerical providers load after enabled root admission."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SEEDS = (6101, 6203, 6307)
FAMILIES = ('native_single', 'native_independent4')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def inside(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Canonical phase-relative path')
    path = (PHASE/relative).resolve()
    require(path != PHASE and path.is_relative_to(PHASE), 'Inside authorized project phase')
    return path


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(PHASE)), bytes=path.stat().st_size, sha256=sha(path))


def bound(row):
    path = inside(row['path'])
    require(path.is_file() and sha(path) == row['sha256']
            and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Exact bound bytes')
    return path


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def spec_check(spec):
    require(spec['family'] in FAMILIES and spec['paired_seed'] in SEEDS
            and type(spec['member_index']) is int
            and spec['member_index'] in range(1 if spec['family'] == 'native_single' else 4)
            and type(spec['body_seed']) is int
            and spec['body_seed'] == spec['paired_seed']+1009*spec['member_index'],
            'Fixed condition/paired block/private body, no new optimizer block')


def source_gate():
    pins = read(HERE/'SOURCE_BINDINGS.json')
    for row in read(HERE/'SOURCE_MANIFEST.json')['files']:
        path = (HERE/row['path']).resolve()
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Sealed new reference source unchanged')
    for row in pins['source_files']:
        bound(row)
    root = bound(pins['public_manifest']).parent
    for row in read(root/'MANIFEST.json')['files']:
        path = root/row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Original native public source unchanged')
    return pins


def admission(release_path, release_sha, assembly=False):
    require(sha(release_path) == release_sha, 'Exact prospective root release')
    release = read(release_path)
    require(all(release.get(key) is True for key in ('enabled', 'root_execution_authorized',
                'source_review_approved', 'reference_runtime_qualified')),
            'Disabled source: no model import until separately reviewed/qualified/root-admitted')
    pins = source_gate(); runtime = pins['runtime']
    require(release['source_manifest_sha256'] == sha(HERE/'SOURCE_MANIFEST.json')
            and socket.gethostname() == runtime['hostname'] and Path.cwd().resolve() == Path(runtime['repository'])
            and str(Path(sys.executable).absolute()) == runtime['python']
            and os.environ.get('PYTHONPATH', '') == '', 'Exact normal77 source/cwd/interpreter')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                text=True, timeout=10).splitlines() == runtime['physical_gpu_inventory'], 'Exact physical GPU inventory')
    require(all(importlib.metadata.version(name) == runtime[name] for name in
                ('torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')), 'Pinned numeric providers')
    require(all(release.get(key) is False for key in ('TEST_access', 'automatic_retry', 'HPO',
                'calibration', 'reselection', 'reference_scores_open_before_whole_closure')), 'No outcome rescue')
    adoption = read(bound(release['conditional_root_adoption']))
    require(adoption['root_adopted'] is True and adoption['all_six_candidate_endpoints_closed'] is True
            and adoption['candidate_screen_merits_references'] is True
            and adoption['reference_source_manifest_sha256'] == release['source_manifest_sha256'],
            'Current candidate whole6 closure and separate conditional reference admission')
    qualification = read(bound(release['reference_qualification']))
    require(qualification['complete'] is True
            and qualification['source_manifest_sha256'] == release['source_manifest_sha256']
            and qualification['strategy'] == 'separate-native-M1-bodies'
            and qualification['local_and_global_full_TRAIN_updates_checked'] is True
            and qualification['full_TRAIN_updates'] > 0
            and qualification['VALID_scores_read'] is False and qualification['TEST_access'] is False,
            'Actual reference full-TRAIN qualification, no inherited four-row-bank credit')
    gpu = release['physical_gpu_uuid']
    require(gpu in runtime['physical_gpu_inventory'] and qualification['physical_gpu_uuid'] == gpu,
            'Actual same-GPU reference qualification')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ('' if assembly else gpu), 'Admitted device visibility')
    supervision = read(bound(release['external_supervision']))
    entry = HERE/('assemble_bank.py' if assembly else 'train.py')
    require(supervision['enabled'] is True and supervision['root_owns_finite_launch_and_actual_costs'] is True
            and supervision['entry_program'] == binding(entry)
            and supervision['release_argument_path'] == str(Path(release_path).resolve())
            and supervision['argv_prefix'] == [runtime['python'], '-B', str(entry)]
            and supervision['existing_run_fit_helper'] == pins['run_fit_helper']
            and supervision['existing_ownership_helper'] == pins['ownership_helper']
            and supervision['cleanup_seconds'] == 10
            and supervision['hard_seconds'] == supervision['active_seconds']+10
            and 0 < supervision['active_seconds'] <= (890 if assembly else 32390)
            and supervision['owned_GPU_bytes'] == (0 if assembly else 32*1024**3)
            and 0 < supervision['owned_RSS_bytes'] <= (8 if assembly else 32)*1024**3
            and 0 < supervision['maximum_output_bytes'] <= 4*1024**3
            and 0 < supervision['combined_log_bytes'] <= 8*1024**2, 'Exact finite existing-owner custody')
    return release, pins
