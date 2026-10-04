"""Disabled TRAIN-only native engineering control; numerical imports follow gate."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
EXECUTION = PHASE / 'pubmed_native_only_continuation_control_execution_root_20261004_v1'
NATIVE = PHASE / 'pubmed_native_predictive_program_source_20261004_v3'
ADAPTER = PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1'
QUALIFIED = PHASE / 'pubmed_heart_native_numerical_qualification_source_20261004_v2'
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


def check_input_files(plan):
    authority = plan['input_authority']
    require(set(authority['file_sha256']) == {'train_pos.txt', 'gnn_feature'}, 'Only TRAIN and raw features admitted')
    available = PHASE / authority['available_directory']
    require(available.resolve().is_relative_to(PHASE), 'Inputs leave phase')
    rows = {}
    for name, pin in authority['file_sha256'].items():
        path = available / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(available)
                and sha(path) == pin, 'TRAIN/feature identity differs')
        rows[name] = dict(sha256=pin, bytes=path.stat().st_size)
    return rows


def gate(release_path, pin, stage):
    require(stage == 'native_continuation_control', 'One fresh native control only')
    require(Path.cwd().resolve() == REPO and os.environ.get('GNNM_SSH_DESTINATION') == 'shmelev@192.168.18.77', 'Exact ordinary repository/route required')
    require(HERE.resolve().is_relative_to(PHASE), 'Source leaves phase')
    require(release_path.resolve() == EXECUTION / 'ROOT_RELEASE_native_continuation_control.json'
            and not release_path.is_symlink() and sha(release_path) == pin, 'Exact external root release required')
    release = json.loads(release_path.read_text())
    plan = json.loads((HERE / 'PLAN.json').read_text())
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
    require(release.get('schema') == 'pubmed-native-only-control-root-release-v1'
            and release.get('status') == 'APPROVED' and release.get('root_authorization_reference')
            and release.get('root_source_review_approved') is True, 'Root exact source approval absent')
    require(release.get('authorized_stages') == [stage] and release.get('automatic_retry') is False
            and release.get('diagnostic_only') is True and release.get('scientific_fit_admitted') is False
            and release.get('state_donor_allowed') is False
            and all(release.get(key) is True for key in ('no_VALID_reads', 'no_TEST', 'no_scores')), 'Engineering TRAIN-only scope differs')
    require(sha(HERE / 'MANIFEST.json') == release.get('source_manifest_sha256')
            and sha(HERE / 'PLAN.json') == release.get('plan_sha256'), 'Source/plan differs')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = HERE / row['path']
        require(path.resolve().is_relative_to(HERE) and not path.is_symlink()
                and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source payload differs')
    for row in binding['external_source_pins']:
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE) and not path.is_symlink()
                and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Dependency source differs')
    review_path = Path(release['independent_source_review_path'])
    require(review_path.resolve().is_relative_to(PHASE) and not review_path.is_symlink()
            and sha(review_path) == release['independent_source_review_sha256'], 'Exact independent review absent')
    review = json.loads(review_path.read_text())
    require(review.get('status') == 'PASS' and review.get('candidate_manifest_sha256') == release['source_manifest_sha256']
            and review.get('execution_authorized') is False, 'Exact independent control source PASS absent')
    require(release.get('caps') == plan['stages'][stage]['caps'] == plan['caps']
            and release.get('invocation') == plan['stages'][stage]['invocation']
            and release.get('input_authority') == plan['input_authority']
            and release.get('existing_qualification_reused') is True, 'Frozen control authority/caps/invocation differ')
    for name, spec in binding['prerequisite_receipts'].items():
        row = release.get('prerequisites', {}).get(name, {})
        require(row.get('verified_by_root') is True and row.get('path') == spec['path']
                and row.get('sha256') == spec['sha256'], 'Native prerequisite absent')
        path = PHASE / row['path']
        require(not path.is_symlink() and path.resolve().is_relative_to(PHASE)
                and sha(path) == row['sha256'] and json.loads(path.read_text()).get('status') == spec['required_status'], 'Native prerequisite differs')
    feature = plan['input_authority']['feature_equivalence_receipt']
    require(sha(PHASE / feature['path']) == feature['sha256'], 'Raw feature authority differs')
    require(sha(HERE / 'PRESERVED_SHARED4_DIAGNOSTIC_SUMMARY.json') == binding['preserved_compact_summary_sha256'], 'Preserved failure evidence differs')
    profile = plan['execution_profile']
    require(sys.platform == 'linux' and socket.gethostname() == profile['host'], 'Exact ordinary host required')
    require(Path(sys.executable).resolve() == Path(profile['interpreter_path']).resolve()
            and sha(profile['interpreter_path']) == profile['interpreter_sha256'], 'Interpreter differs')
    for key, value in profile['environment'].items():
        require(os.environ.get(key) == value, 'Exact ordinary environment differs')
    for name, version in profile['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Runtime distribution differs')
    authority = json.loads((PHASE / binding['runtime_authority_path']).read_text())
    for row in authority['runtime_source_pins'] + authority['runtime_binary_files'] + [authority['negative_sampler']]:
        require(sha(row['path']) == row['sha256'] and ('bytes' not in row or Path(row['path']).stat().st_size == row['bytes']), 'Runtime file differs')
    snapshot = json.loads((ADAPTER / 'PINNED_AUTHOR_SOURCE_AND_FUNCTIONS.json').read_text())
    for row in snapshot['snapshot_files']:
        require(sha(ADAPTER / 'public_author_code/HeaRT' / row['path']) == row['sha256'], 'Native author source differs')
    check_input_files(plan)
    return release, plan


def runtime(plan):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    profile = plan['execution_profile']
    require(torch.__version__ == profile['torch_version'] and torch.version.cuda == profile['cuda_version']
            and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Qualified single CUDA runtime required')
    props = torch.cuda.get_device_properties(0)
    require(props.name == profile['gpu_name'] and props.total_memory == profile['gpu_total_memory_bytes'], 'Physical GPU differs')
    require(str(torch.get_default_dtype()) == 'torch.float32' and torch.get_float32_matmul_precision() == 'highest'
            and not torch.are_deterministic_algorithms_enabled() and not torch.is_autocast_enabled(), 'Ordinary False precision profile differs')
    sys.path.insert(0, str(ADAPTER / 'public_author_code/HeaRT/benchmarking'))
    bodies = load_module('pubmed_native_control_qualified_bodies', QUALIFIED / 'native_bodies.py')
    native = load_module('pubmed_native_control_state_helpers', NATIVE / 'common.py')
    # Never call native.setup_runtime or either all-available loader/inspector.
    inspector = load_module('pubmed_native_control_train_row_reader', ADAPTER / 'inspect_available.py')
    inspector.server_guard()
    identities = check_input_files(plan)
    available = PHASE / plan['input_authority']['available_directory']
    require('weights_only' in inspect.signature(torch.load).parameters, 'weights_only feature loading required')
    supplied = torch.load(available / 'gnn_feature', map_location='cpu', weights_only=True)
    require(isinstance(supplied, dict) and 'entity_embedding' in supplied, 'Native feature key absent')
    x = supplied['entity_embedding']
    require(isinstance(x, torch.Tensor) and x.layout == torch.strided and list(x.shape) == [19717, 500]
            and x.dtype == torch.float32 and bool(torch.isfinite(x).all()), 'Native raw features differ; no cast/normalization')
    train = torch.tensor(inspector.read_rows('train_pos.txt', 37676), dtype=torch.long)
    authority = json.loads((PHASE / json.loads((HERE / 'SOURCE_BINDING.json').read_text())['runtime_authority_path']).read_text())
    for row in authority['runtime_source_pins']:
        module = __import__(row['module'], fromlist=['*'])
        require(Path(module.__file__).resolve() == Path(row['path']).resolve(), 'Runtime import shadowed')
    source_dirs = {Path(row['path']).parent for row in authority['runtime_source_pins'] if row['module'] in ('torch_sparse', 'torch_scatter')}
    expected_binaries = {Path(row['path']).resolve() for row in authority['runtime_binary_files']}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries if Path(path).resolve().parent in source_dirs}
    require(loaded and loaded <= expected_binaries and all(any(path.parent == directory for path in loaded) for directory in source_dirs), 'Sparse/scatter binary admission differs')
    pin = authority['negative_sampler']
    require(Path(inspect.getsourcefile(bodies.negative_sampling)).resolve() == Path(pin['path']).resolve()
            and hashlib.sha256(inspect.getsource(bodies.negative_sampling).encode()).hexdigest() == pin['function_sha256'], 'Native sampler differs')
    signature = inspect.signature(bodies.negative_sampling)
    require(signature.parameters['num_neg_samples'].default is None and signature.parameters['method'].default == 'sparse'
            and signature.parameters['force_undirected'].default is False, 'Native sampler defaults differ')
    # The unchanged observer accepts this alias and executes the reference's
    # original code object, without importing any shared4 facade/prototype.
    observed_bodies = SimpleNamespace(candidate_ncnc_train=bodies.reference_ncnc_train,
                                     negative_sampling=bodies.negative_sampling, PermIterator=bodies.PermIterator)
    inspection = dict(files=identities, TRAIN_rows=len(train), node_population=19717,
                      feature=dict(key='entity_embedding', shape=list(x.shape), dtype=str(x.dtype), finite=True),
                      VALID_files_opened=False, TEST_files_opened=False, score_files_opened=False, state_files_opened=False,
                      no_VALID_or_TEST_or_score_computation=True)
    return native, (torch, np, bodies, observed_bodies, x, train, inspection, identities)


def native_work(unit, progress, maximum_updates=252):
    encoder, _, optimizer = unit[:3]
    def before(optimizer, args, kwargs):
        require(progress.snapshot()['Adam_started'] < maximum_updates, 'Explicit total native update cap exhausted')
        progress.add(Adam_started=1)
    handles = [encoder.register_forward_pre_hook(lambda module, args: progress.add(encoder_started=1)),
               encoder.register_forward_hook(lambda module, args, result: progress.add(encoder_completed=1)),
               optimizer.register_step_pre_hook(before),
               optimizer.register_step_post_hook(lambda optimizer, args, kwargs: progress.add(Adam_completed=1))]
    def remove():
        for handle in handles:
            handle.remove()
    return remove


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
