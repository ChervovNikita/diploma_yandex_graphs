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
EXECUTION = PHASE / 'pubmed_native_predictive_program_execution_root_20261004_v1'
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
    require(stage in ('fit_cohort', 'replay_selected'), 'Unknown stage')
    require(Path.cwd().resolve() == REPO and os.environ.get('GNNM_SSH_DESTINATION') == 'shmelev@192.168.18.77', 'Exact repository/route required')
    require(HERE.resolve().is_relative_to(PHASE), 'Source leaves phase')
    require(release_path.resolve() == EXECUTION / ('ROOT_RELEASE_' + stage + '.json') and not release_path.is_symlink() and sha(release_path) == pin, 'Exact external root release required')
    release = json.loads(release_path.read_text())
    cohort = json.loads((HERE / 'COHORT.json').read_text())
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
    require(release.get('schema') == 'pubmed-native-predictive-root-release-v1' and release.get('status') == 'APPROVED' and release.get('root_authorization_reference') and release.get('root_source_review_approved') is True, 'Root source review/release absent')
    require(release.get('authorized_stages') == [stage] and release.get('automatic_retry') is False and release.get('TEST_supported') is False, 'One TRAIN/VALID-only stage required')
    require(release.get('scientific_fit_admitted') is (stage == 'fit_cohort'), 'Fit/replay admission differs')
    require(sha(HERE / 'MANIFEST.json') == release.get('source_manifest_sha256'), 'Source manifest differs')
    for row in json.loads((HERE / 'MANIFEST.json').read_text())['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE) and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Source payload differs')
    for row in binding['external_source_pins']:
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE) and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Qualified dependency differs: ' + row['path'])
    require(release.get('cohort_sha256') == sha(HERE / 'COHORT.json') and release.get('caps') == cohort['stages'][stage]['caps'] and release.get('invocation') == cohort['stages'][stage]['invocation'], 'Frozen cohort/caps/invocation differ')
    # These are scalar prerequisite receipts. No engineering state file is opened.
    for name, specification in binding['prerequisite_receipts'].items():
        row = release.get('prerequisites', {}).get(name, {})
        require(row.get('verified_by_root') is True and row.get('sha256') and row.get('path') == specification['path'], 'Root prerequisite absent: ' + name)
        if specification.get('root_reported_sha256'):
            require(row['sha256'] == specification['root_reported_sha256'], 'Root-reported prerequisite pin differs: ' + name)
        path = PHASE / row['path']
        require(not path.is_symlink() and sha(path) == row['sha256'], 'Prerequisite custody differs: ' + name)
        scalar = json.loads(path.read_text())
        require(scalar.get('status') == specification['required_status'], 'Prerequisite incomplete: ' + name)
    require(release.get('feature_authority', {}).get('verified') is True and release['feature_authority'].get('evidence') and release['feature_authority'].get('origin'), 'Feature authority absent')
    require(release.get('negative_pool_authority', {}).get('verified') is True and release['negative_pool_authority'].get('semantics') and release['negative_pool_authority'].get('evidence'), 'Released VALID semantics absent')
    require(release.get('feature_equivalence_receipt_sha256') == binding['feature_equivalence_receipt_sha256'], 'Root feature prerequisite differs')
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
        terminal = EXECUTION / 'supervision/fit_cohort/run01/SUPERVISOR_TERMINAL.json'
        require(release.get('fit_cohort_freeze_sha256') == sha(freeze) and release.get('fit_supervisor_terminal_sha256') == sha(terminal), 'Completed owned fit custody absent')
        terminal_value = json.loads(terminal.read_text())
        require(json.loads(freeze.read_text()).get('status') == 'COMPLETE' and terminal_value.get('status') == 'COMPLETE', 'Fit did not complete within its cap')
        require(terminal_value['result']['sha256'] == sha(freeze) and terminal_value['final_custody']['sha256'] == sha(freeze.parent / 'FINAL_CUSTODY.json'), 'Completed fit terminal custody differs')
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
        write(output / 'PROGRESS.json', progress)
        require(all(peaks[key] <= caps[key] for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')), 'CUDA peak cap exceeded')
    def watch():
        while not stopped.wait(.25):
            try:
                sample()
            except BaseException as error:
                write(output / 'MONITOR_FAILURE.json', {'status': 'FAILED', 'type': type(error).__name__, 'condition': str(error), 'progress': progress})
                os._exit(88)
    observer = threading.Thread(target=watch, daemon=True)
    observer.start()
    return stopped, observer, peaks, sample
