"""Shared stdlib gate and serialized state helpers; imports native code only on release."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
EXECUTION = PHASE / 'pubmed_shared4_predictive_program_execution_root_20261004_v1'
BRIDGE = PHASE / 'pubmed_shared4_extension_source_preparation_20261004_v1'
PROTOTYPE = PHASE / 'graph_ncNC_member_completion_qualification_preparation_20261003_v2'
BRIDGE_QUALIFIED = PHASE / 'pubmed_shared4_bridge_qualification_source_20261004_v2'
ADAPTER = PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1'
QUALIFIED = PHASE / 'pubmed_heart_native_numerical_qualification_source_20261004_v2'
AUTHOR = ADAPTER / 'public_author_code/HeaRT/benchmarking'
_WRITE_LOCK = threading.RLock()


def require(value, message):
    if not value:
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
    with _WRITE_LOCK:
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate(release_path, pin, stage):
    require(stage in ('fit_cohort', 'replay_selected'), 'One known scientific stage required')
    require(Path.cwd().resolve() == REPO and os.environ.get('GNNM_SSH_DESTINATION') == 'shmelev@192.168.18.77', 'Exact repository/route required')
    require(HERE.resolve().is_relative_to(PHASE), 'Source leaves phase')
    require(release_path.resolve() == EXECUTION / ('ROOT_RELEASE_' + stage + '.json') and not release_path.is_symlink() and sha(release_path) == pin, 'Exact external root release required')
    release = json.loads(release_path.read_text())
    cohort = json.loads((HERE / 'COHORT.json').read_text())
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
    require(release.get('schema') == 'pubmed-shared4-predictive-root-release-v1' and release.get('status') == 'APPROVED' and release.get('root_authorization_reference') and release.get('root_source_review_approved') is True, 'Root scientific source approval/release absent')
    require(release.get('authorized_stages') == [stage] and release.get('automatic_retry') is False and release.get('TEST_supported') is False, 'One TRAIN/VALID-only stage required')
    require(release.get('scientific_fit_admitted') is (stage == 'fit_cohort'), 'Fit/replay admission differs')
    require(sha(HERE / 'MANIFEST.json') == release.get('source_manifest_sha256'), 'Source manifest differs')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = HERE / row['path']
        require(path.resolve().is_relative_to(HERE) and not path.is_symlink() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source payload differs')
    for row in binding['external_source_pins']:
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE) and not path.is_symlink() and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Qualified dependency differs: ' + row['path'])
    review_path = Path(release['independent_source_review_path']).resolve()
    require(review_path.is_relative_to(PHASE) and not review_path.is_symlink() and sha(review_path) == release['independent_source_review_sha256'], 'Exact independent source review absent')
    review = json.loads(review_path.read_text())
    require(review.get('status') == 'PASS' and review.get('candidate_manifest_sha256') == release['source_manifest_sha256'] and review.get('execution_authorized') is False, 'Independent source review is not exact PASS')
    require(release.get('cohort_sha256') == sha(HERE / 'COHORT.json') and release.get('caps') == cohort['stages'][stage]['caps'] and release.get('invocation') == cohort['stages'][stage]['invocation'], 'Frozen cohort/caps/invocation differ')
    for name, specification in binding['prerequisite_receipts'].items():
        row = release.get('prerequisites', {}).get(name, {})
        require(row.get('verified_by_root') is True and row.get('sha256') and row.get('path') == specification['path'], 'Root prerequisite absent: ' + name)
        if specification.get('sha256'):
            require(row['sha256'] == specification['sha256'], 'Prerequisite pin differs: ' + name)
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE) and not path.is_symlink() and sha(path) == row['sha256'], 'Prerequisite custody differs: ' + name)
        if specification.get('bytes'):
            require(path.stat().st_size == specification['bytes'], 'Prerequisite size differs: ' + name)
        require(json.loads(path.read_text()).get('status') == specification['required_status'], 'Prerequisite incomplete: ' + name)
    qrow = release['prerequisites']['bridge_qualification']
    trow = release['prerequisites']['bridge_qualification_supervisor']
    qpath, tpath = PHASE / qrow['path'], PHASE / trow['path']
    qualification, terminal = json.loads(qpath.read_text()), json.loads(tpath.read_text())
    require(qualification.get('source_manifest_sha256') == binding['bridge_qualification_source_manifest_sha256']
            and terminal.get('source_manifest_sha256') == binding['bridge_qualification_source_manifest_sha256'], 'Passed bridge qualifier source differs')
    require(qualification.get('schema') == 'pubmed-shared4-bridge-engineering-qualification-v1'
            and qualification.get('scientific_fit_admitted') is False and qualification.get('state_donor_allowed') is False
            and qualification.get('TEST_supported') is False
            and qualification['progress']['Adam_started'] == qualification['progress']['Adam_completed'] == 504
            and qualification['progress']['native_epochs_started'] == qualification['progress']['native_epochs_completed'] == 14
            and qualification['progress']['VALID_started'] == qualification['progress']['VALID_completed'] == 4,
            'Complete engineering-only504-update bridge qualification required')
    require(terminal['result']['sha256'] == sha(qpath) and terminal.get('ordinary_host_execution') is True
            and terminal['final_custody']['sha256'] == sha(qpath.parent / 'FINAL_CUSTODY.json')
            and json.loads((qpath.parent / 'FINAL_CUSTODY.json').read_text()).get('completed') is True, 'Passed qualifier physical/custody completion absent')
    require(release.get('feature_authority', {}).get('verified') is True and release['feature_authority'].get('evidence') and release['feature_authority'].get('origin'), 'Feature authority absent')
    require(release.get('negative_pool_authority', {}).get('verified') is True and release['negative_pool_authority'].get('semantics') and release['negative_pool_authority'].get('evidence'), 'Released pool semantics absent')
    require(release.get('feature_equivalence_receipt_sha256') == binding['feature_equivalence_receipt_sha256'], 'Raw feature equivalence prerequisite differs')
    require(sha(PHASE / binding['feature_equivalence_receipt_path']) == binding['feature_equivalence_receipt_sha256'], 'Raw feature receipt custody differs')
    profile = cohort['execution_profile']
    require(sys.platform.startswith('linux') and socket.gethostname() == profile['host'], 'Exact ordinary host required')
    require(Path(sys.executable).resolve() == Path(profile['interpreter_path']).resolve() and sha(profile['interpreter_path']) == profile['interpreter_sha256'], 'Interpreter differs')
    for key in ('PYTHONPATH', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES'):
        require(os.environ.get(key) == profile['environment'][key], 'Declared process environment differs: ' + key)
    for name, version in profile['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Distribution differs: ' + name)
    authority = json.loads((PHASE / binding['runtime_authority_path']).read_text())
    for row in authority['runtime_source_pins'] + authority['runtime_binary_files'] + [authority['negative_sampler']]:
        require(sha(row['path']) == row['sha256'], 'Qualified runtime source/binary differs')
    snapshot = json.loads((ADAPTER / 'PINNED_AUTHOR_SOURCE_AND_FUNCTIONS.json').read_text())
    for row in snapshot['snapshot_files']:
        require(sha(ADAPTER / 'public_author_code/HeaRT' / row['path']) == row['sha256'], 'Pinned author source differs')
    if stage == 'replay_selected':
        freeze = EXECUTION / 'fit_cohort/run01/COHORT_FREEZE.json'
        terminal_path = EXECUTION / 'supervision/fit_cohort/run01/SUPERVISOR_TERMINAL.json'
        require(release.get('fit_cohort_freeze_sha256') == sha(freeze) and release.get('fit_supervisor_terminal_sha256') == sha(terminal_path), 'Completed own scientific fit custody absent')
        terminal_value = json.loads(terminal_path.read_text())
        freeze_value = json.loads(freeze.read_text())
        require(freeze_value.get('schema') == 'pubmed-shared4-cohort-freeze-v1' and freeze_value.get('status') == 'COMPLETE'
                and freeze_value.get('source_manifest_sha256') == release['source_manifest_sha256']
                and terminal_value.get('status') == 'COMPLETE' and terminal_value.get('stage') == 'fit_cohort'
                and terminal_value.get('source_manifest_sha256') == release['source_manifest_sha256'], 'Own scientific fit did not complete within cap')
        require(terminal_value['result']['sha256'] == sha(freeze) and terminal_value['final_custody']['sha256'] == sha(freeze.parent / 'FINAL_CUSTODY.json'), 'Own scientific fit terminal custody differs')
        require(json.loads((freeze.parent / 'FINAL_CUSTODY.json').read_text()).get('completed') is True,
                'Own scientific fit final custody incomplete')
    return release, cohort


def clone(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone(child, torch) for key, child in value.items()}
    if isinstance(value, list):
        return [clone(child, torch) for child in value]
    if isinstance(value, tuple):
        return tuple(clone(child, torch) for child in value)
    return copy.deepcopy(value)


def rng(torch, np, gpu):
    nr = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (nr[0], torch.tensor(nr[1].astype('int64')), nr[2], nr[3], nr[4]),
            'torch_cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if gpu else []}


def restore_rng(state, torch, np, gpu):
    random.setstate(state['python'])
    nr = state['numpy']
    np.random.set_state((nr[0], nr[1].numpy().astype('uint32'), nr[2], nr[3], nr[4]))
    torch.set_rng_state(state['torch_cpu'])
    if gpu:
        torch.cuda.set_rng_state_all(state['cuda'])


def capture(unit, torch, np):
    model, predictor, optimizer, _, _, _ = unit
    return clone({'encoder': model.state_dict(), 'predictor': predictor.state_dict(), 'Adam': optimizer.state_dict(),
                  'gradients': [{name: parameter.grad for name, parameter in root.named_parameters()} for root in (model, predictor)],
                  'flags': [{name: module.training for name, module in root.named_modules()} for root in (model, predictor)],
                  'invest': [getattr(module, 'invest', None) for module in (model, predictor)], 'RNG': rng(torch, np, True)}, torch)


def restore(unit, state, torch, np):
    model, predictor, optimizer, _, _, _ = unit
    model.load_state_dict(state['encoder'], strict=True)
    predictor.load_state_dict(state['predictor'], strict=True)
    optimizer.load_state_dict(state['Adam'])
    for index, root in enumerate((model, predictor)):
        require(set(dict(root.named_parameters())) == set(state['gradients'][index]), 'Serialized gradient names differ')
        for name, parameter in root.named_parameters():
            grad = state['gradients'][index][name]
            parameter.grad = None if grad is None else grad.to(parameter.device).clone()
        require(set(dict(root.named_modules())) == set(state['flags'][index]), 'Serialized module flags differ')
        for name, module in root.named_modules():
            module.training = state['flags'][index][name]
        if state['invest'][index] is not None:
            root.invest = state['invest'][index]
    restore_rng(state['RNG'], torch, np, True)


def validation(bodies, model_name, unit, valid, negative, torch):
    model, predictor, _, x, data, _ = unit
    scope = dict(vars(bodies))
    scope.update(model=model, predictor=predictor, model_name=model_name, x=x, data=data, device=torch.device('cuda:0'),
                 valid=valid, valid_neg=negative, split_edge={'valid': {'edge': valid, 'edge_neg': negative}})
    serve = bodies.bound_validation('candidate', model_name, scope)
    pos, neg = clone(serve(), torch)
    require(pos.shape == (2216,) and neg.shape == (2216, 500) and bool(torch.isfinite(pos).all()) and bool(torch.isfinite(neg).all()), 'Complete finite VALID scores required')
    return pos, neg


def setup_runtime(cohort):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    profile = cohort['execution_profile']
    require(torch.__version__ == profile['torch_version'] and torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.version.cuda == '12.6', 'Qualified single CUDA runtime required')
    props = torch.cuda.get_device_properties(0)
    require(props.name == profile['gpu_name'] and props.total_memory == profile['gpu_total_memory_bytes'], 'Physical GPU profile differs')
    require(str(torch.get_default_dtype()) == 'torch.float32' and torch.get_float32_matmul_precision() == 'highest' and not torch.are_deterministic_algorithms_enabled() and not torch.is_autocast_enabled(), 'Native precision profile differs')
    sys.path.insert(0, str(AUTHOR))
    sys.path.insert(0, str(ADAPTER))
    bodies = load_module('qualified_pubmed_native_predictive_bodies', QUALIFIED / 'native_bodies.py')
    from evalutors import evaluate_mrr, eval_mrr
    from inspect_available import inspect_available, load_available
    inspection = inspect_available()
    require(inspection['available_geometry_checks_pass'], 'Complete available-input geometry failed')
    x, train_rows, valid_rows, pool, identities = load_available()
    require(list(x.shape) == [19717, 500], 'Root raw feature geometry differs')
    return (torch, np, bodies, evaluate_mrr, eval_mrr, inspection, x,
            torch.tensor(train_rows, dtype=torch.long), torch.tensor(valid_rows, dtype=torch.long),
            torch.from_numpy(np.array(pool, copy=True)), identities)


def save_tensor(path, value, torch):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def inventory(root):
    rows = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'FINAL_CUSTODY.json' and not path.name.endswith('.tmp'):
            require(not path.is_symlink(), 'Owned output symlink forbidden')
            rows.append({'path': str(path.relative_to(root)), 'sha256': sha(path), 'bytes': path.stat().st_size})
    return rows


def start_monitor(output, caps, progress, torch):
    stopped = threading.Event()
    peaks = {'CUDA_observed': False, 'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0}
    def sample():
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
    return stopped, observer, peaks, sample


def equal(left, right, torch, label):
    require(type(left) is type(right), label + ': type differs')
    if isinstance(left, torch.Tensor):
        require(left.shape == right.shape and left.dtype == right.dtype and left.layout == right.layout and torch.equal(left, right), label + ': tensor differs')
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), label + ': keys differ')
        for key in left:
            equal(left[key], right[key], torch, label + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), label + ': length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, torch, label + '/' + str(index))
    else:
        require(left == right, label + ': value differs')


def extension(bodies, torch):
    sys.path.insert(0, str(PROTOTYPE))
    import graph_ops
    import prototype
    bridge = load_module('pubmed_shared4_scientific_bridge', BRIDGE / 'bridge.py')
    observed = load_module('pubmed_shared4_scientific_native_observer', BRIDGE_QUALIFIED / 'native_observer.py')
    bridge.authenticate_modules(PHASE, bodies, prototype, graph_ops)
    authority = json.loads((PHASE / json.loads((HERE / 'SOURCE_BINDING.json').read_text())['runtime_authority_path']).read_text())
    for row in authority['runtime_source_pins']:
        module = __import__(row['module'], fromlist=['*'])
        require(Path(module.__file__).resolve() == Path(row['path']).resolve(), 'Runtime source import shadowed')
    directories = {Path(row['path']).parent for row in authority['runtime_source_pins'] if row['module'] in ('torch_sparse','torch_scatter')}
    binaries = {Path(row['path']).resolve() for row in authority['runtime_binary_files']}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries if Path(path).resolve().parent in directories}
    require(loaded and loaded <= binaries and all(any(path.parent == directory for path in loaded) for directory in directories), 'Loaded sparse/scatter binary admission differs')
    import inspect
    pin = authority['negative_sampler']
    require(Path(inspect.getsourcefile(bodies.negative_sampling)).resolve() == Path(pin['path']).resolve()
            and hashlib.sha256(inspect.getsource(bodies.negative_sampling).encode()).hexdigest() == pin['function_sha256'], 'Native negative sampler function/path differs')
    signature = inspect.signature(bodies.negative_sampling)
    require(signature.parameters['num_neg_samples'].default is None and signature.parameters['method'].default == 'sparse'
            and signature.parameters['force_undirected'].default is False, 'Native sampler defaults differ')
    return bridge, prototype, graph_ops, observed


def fresh_unit(bridge, prototype, graph_ops, bodies, fit, x, train):
    require(fit['model'] == 'NCNC_shared4' and fit['member_count'] == 4, 'Fixed shared4 model identity required')
    return bridge.make_shared4(PHASE, bodies, prototype, graph_ops, fit['seed'], fit['mode'], x, train)


class Progress:
    """RNG-neutral scalar counters; native quarter-second hard-exit lower bounds."""
    def __init__(self, output, stage):
        self.output = output
        self.value = {'stage':stage,'native_epochs_started':0,'native_epochs_completed':0,
                      'Adam_started':0,'Adam_completed':0,'complete_VALID_serves':0,
                      'current_fit':None,'current_epoch':0,'current_phase':'setup',
                      'completed_fits':[],'completed_replays':[]}

    def update(self, **values):
        with _WRITE_LOCK:
            self.value.update(values)
            write(self.output / 'PROGRESS.json', self.value)

    def add(self, **values):
        with _WRITE_LOCK:
            for key, value in values.items():
                self.value[key] += value

    def completed(self, kind, identity):
        with _WRITE_LOCK:
            self.value[kind].append(identity)
            write(self.output / 'PROGRESS.json', self.value)

    def snapshot(self):
        with _WRITE_LOCK:
            return copy.deepcopy(self.value)


STREAM_KEYS = ('negative_calls','iterator_calls','batches','start_RNG_sha256','end_RNG_sha256')


def stream_identity(row):
    require(set(STREAM_KEYS) <= set(row), 'Incomplete native TRAIN stream receipt')
    return {key:row[key] for key in STREAM_KEYS}
