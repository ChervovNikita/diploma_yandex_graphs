"""Root-disabled engineering admission, ordinary supervision and deferred reuse."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
EXECUTION = PHASE / 'pubmed_shared4_bridge_qualification_execution_root_20261004_v1'
NATIVE = PHASE / 'pubmed_native_predictive_program_source_20261004_v3'
BRIDGE = PHASE / 'pubmed_shared4_extension_source_preparation_20261004_v1'
PROTOTYPE = PHASE / 'graph_ncNC_member_completion_qualification_preparation_20261003_v2'
LOCK = threading.RLock()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with LOCK:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def gate(release_path, pin, stage):
    require(stage == 'qualification', 'One engineering qualification stage only')
    require(Path.cwd().resolve() == REPO and os.environ.get('GNNM_SSH_DESTINATION') == 'shmelev@192.168.18.77', 'Exact ordinary repository/route required')
    require(HERE.resolve().is_relative_to(PHASE), 'Source leaves phase')
    require(release_path.resolve() == EXECUTION / 'ROOT_RELEASE_qualification.json' and not release_path.is_symlink() and sha(release_path) == pin, 'Exact external root release required')
    release = json.loads(release_path.read_text())
    plan = json.loads((HERE / 'PLAN.json').read_text())
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
    require(release.get('schema') == 'pubmed-shared4-engineering-root-release-v1' and release.get('status') == 'APPROVED'
            and release.get('root_authorization_reference') and release.get('root_source_review_approved') is True, 'Root source approval/release absent')
    require(release.get('authorized_stages') == [stage] and release.get('automatic_retry') is False
            and release.get('scientific_fit_admitted') is False and release.get('TEST_supported') is False, 'One non-scientific TRAIN/VALID stage required')
    require(sha(HERE / 'MANIFEST.json') == release.get('source_manifest_sha256') and sha(HERE / 'PLAN.json') == release.get('plan_sha256'), 'Source/plan identity differs')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE) and not path.is_symlink() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source payload differs')
    for row in binding['external_source_pins']:
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE) and not path.is_symlink() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Dependency differs: ' + row['path'])
    review_path = Path(release['independent_source_review_path']).resolve()
    require(review_path.is_relative_to(PHASE) and not review_path.is_symlink() and sha(review_path) == release['independent_source_review_sha256'], 'Exact independent source review absent')
    review = json.loads(review_path.read_text())
    require(review.get('status') == 'PASS' and review.get('candidate_manifest_sha256') == release['source_manifest_sha256'] and review.get('execution_authorized') is False, 'Independent review is not an exact source PASS')
    require(release.get('caps') == plan['stages'][stage]['caps'] and release.get('invocation') == plan['stages'][stage]['invocation'], 'Bounded invocation/caps differ')
    require(release.get('input_authority') == plan['input_authority'] and release.get('existing_qualification_reused') is True, 'Existing input/native qualification authority absent')
    feature_receipt = plan['input_authority']['feature_equivalence_receipt']
    feature_path = PHASE / feature_receipt['path']
    require(not feature_path.is_symlink() and sha(feature_path) == feature_receipt['sha256'], 'Raw feature equivalence receipt differs')
    for name, specification in binding['prerequisite_receipts'].items():
        row = release.get('prerequisites', {}).get(name, {})
        require(row.get('verified_by_root') is True and row.get('path') == specification['path'] and row.get('sha256'), 'Prerequisite absent: ' + name)
        if specification.get('sha256'):
            require(row['sha256'] == specification['sha256'], 'Prerequisite pin differs: ' + name)
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE) and not path.is_symlink() and sha(path) == row['sha256'], 'Prerequisite custody differs: ' + name)
        require(json.loads(path.read_text()).get('status') == specification['required_status'], 'Prerequisite incomplete: ' + name)
    profile = plan['execution_profile']
    require(sys.platform.startswith('linux') and socket.gethostname() == profile['host'], 'Exact normal host required')
    require(Path(sys.executable).resolve() == Path(profile['interpreter_path']).resolve() and sha(profile['interpreter_path']) == profile['interpreter_sha256'], 'Interpreter differs')
    for key in ('PYTHONPATH', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES'):
        require(os.environ.get(key) == profile['environment'][key], 'Process environment differs: ' + key)
    for name, version in profile['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Distribution differs: ' + name)
    authority = json.loads((PHASE / binding['runtime_authority_path']).read_text())
    for row in authority['runtime_source_pins'] + authority['runtime_binary_files'] + [authority['negative_sampler']]:
        require(sha(row['path']) == row['sha256'], 'Runtime source/binary differs')
    adapter = PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1'
    snapshot = json.loads((adapter / 'PINNED_AUTHOR_SOURCE_AND_FUNCTIONS.json').read_text())
    for row in snapshot['snapshot_files']:
        require(sha(adapter / 'public_author_code/HeaRT' / row['path']) == row['sha256'], 'Author snapshot differs')
    return release, plan


def runtime(plan):
    # Only called after the stdlib gate. Reuse qualified available-only loading,
    # numerical bodies, complete-state helpers and VALID without a native refit.
    native = load_module('pubmed_shared4_reused_native_common', NATIVE / 'common.py')
    result = native.setup_runtime(plan)
    torch, np, bodies = result[:3]
    sys.path.insert(0, str(PROTOTYPE))
    import graph_ops
    import prototype
    bridge = load_module('pubmed_shared4_source_bridge', BRIDGE / 'bridge.py')
    bridge.authenticate_modules(PHASE, bodies, prototype, graph_ops)
    authority = json.loads((PHASE / json.loads((HERE / 'SOURCE_BINDING.json').read_text())['runtime_authority_path']).read_text())
    for row in authority['runtime_source_pins']:
        module = __import__(row['module'], fromlist=['*'])
        require(Path(module.__file__).resolve() == Path(row['path']).resolve(), 'Runtime import shadowed: ' + row['module'])
    source_dirs = {Path(row['path']).parent for row in authority['runtime_source_pins'] if row['module'] in ('torch_sparse', 'torch_scatter')}
    expected_binaries = {Path(row['path']).resolve() for row in authority['runtime_binary_files']}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries if Path(path).resolve().parent in source_dirs}
    require(loaded and loaded <= expected_binaries and all(any(path.parent == directory for path in loaded) for directory in source_dirs), 'Sparse/scatter binary admission differs')
    import inspect
    sampler_pin = authority['negative_sampler']
    require(Path(inspect.getsourcefile(bodies.negative_sampling)).resolve() == Path(sampler_pin['path']).resolve()
            and hashlib.sha256(inspect.getsource(bodies.negative_sampling).encode()).hexdigest() == sampler_pin['function_sha256'], 'Native sampler function/path differs')
    signature = inspect.signature(bodies.negative_sampling)
    require(signature.parameters['num_neg_samples'].default is None and signature.parameters['method'].default == 'sparse'
            and signature.parameters['force_undirected'].default is False, 'Original sampler defaults differ')
    require(torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1 and not torch.are_deterministic_algorithms_enabled(), 'Qualified thread/kernel profile differs')
    return native, result, bridge, prototype, graph_ops


class Progress:
    def __init__(self, output):
        self.output = output
        self.value = {'phase': 'setup', 'current_arm': None, 'current_epoch': 0,
                      'Adam_started': 0, 'Adam_completed': 0, 'native_epochs_started': 0,
                      'native_epochs_completed': 0, 'VALID_started': 0, 'VALID_completed': 0,
                      'serializations': 0, 'weights_only_loads': 0, 'factory_calls': 0,
                      'native_reference_factory_calls': 0, 'gradient_backward_calls': 0,
                      'encoder_started': 0, 'encoder_completed': 0, 'root_decoder_started': 0,
                      'root_decoder_completed': 0, 'root_query_rows_started': 0,
                      'root_query_rows_completed': 0, 'neighbor_calls_started': 0,
                      'neighbor_calls_completed': 0, 'neighbor_query_rows': 0,
                      'common_rows': 0, 'left_residual_rows': 0, 'right_residual_rows': 0}

    def update(self, **values):
        with LOCK:
            self.value.update(values)
            write(self.output / 'PROGRESS.json', self.value)

    def add(self, **values):
        with LOCK:
            for key, count in values.items():
                self.value[key] += count
            write(self.output / 'PROGRESS.json', self.value)

    def snapshot(self):
        with LOCK:
            return dict(self.value)


def monitor(output, caps, progress, torch):
    stopped = threading.Event()
    peaks = {'CUDA_observed': False, 'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0}
    def sample():
        with LOCK:
            if torch.cuda.is_initialized():
                peaks.update(CUDA_observed=True, cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                             cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
            write(output / 'CUDA_PEAKS.json', peaks)
            write(output / 'PROGRESS.json', progress.snapshot())
            require(all(peaks[key] <= caps[key] for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')), 'CUDA peak cap exceeded')
    def watch():
        while not stopped.wait(.25):
            try:
                sample()
            except BaseException as error:
                write(output / 'MONITOR_FAILURE.json', {'status': 'FAILED', 'type': type(error).__name__, 'condition': str(error), 'progress': progress.snapshot()})
                os._exit(88)
    observer = threading.Thread(target=watch, daemon=True)
    observer.start()
    return stopped, observer, sample


def inventory(root):
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'FINAL_CUSTODY.json' and not path.name.endswith('.tmp'):
            require(not path.is_symlink(), 'Owned artifact symlink forbidden')
            rows.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':sha(path)})
    return rows
