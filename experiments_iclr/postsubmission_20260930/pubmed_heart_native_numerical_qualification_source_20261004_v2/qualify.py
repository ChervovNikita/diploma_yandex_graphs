#!/usr/bin/env python3
"""Engineering qualification only, separately root released; no scientific fit."""
import argparse
import ast
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
EXECUTION = PHASE / 'pubmed_heart_native_numerical_qualification_execution_root_20261004_v1'
ADAPTER = PHASE / 'pubmed_heart_available_inspector_native_adapter_20261004_v1'
AUTHOR = ADAPTER / 'public_author_code/HeaRT/benchmarking'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
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
    require(Path.cwd().resolve() == REPO and os.environ.get('GNNM_SSH_DESTINATION') == 'shmelev@192.168.18.77', 'Exact repository and route required')
    require(release_path.resolve() == EXECUTION / ('ROOT_RELEASE_' + stage + '.json') and sha(release_path) == pin, 'Exact external release required')
    release = json.loads(release_path.read_text())
    require(release.get('status') == 'APPROVED' and release.get('root_authorization_reference') and release.get('root_source_review_approved') is True, 'Root source review/release absent')
    require(release.get('authorized_stages') == [stage] and release.get('automatic_retry') is False and release.get('scientific_fit_admitted') is False, 'One engineering stage only')
    manifest = HERE / 'MANIFEST.json'
    require(sha(manifest) == release['source_manifest_sha256'], 'Qualification source manifest differs')
    for row in json.loads(manifest.read_text())['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE) and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Qualification source payload differs')
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
    for row in binding['external_source_pins']:
        path = PHASE / row['path']
        require(path.resolve().is_relative_to(PHASE) and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], 'Pinned source differs: ' + row['path'])
    review_path = Path(release['independent_source_review_path']).resolve()
    require(review_path.is_relative_to(PHASE) and sha(review_path) == release['independent_source_review_sha256'], 'Independent review receipt differs')
    review = json.loads(review_path.read_text())
    require(review.get('status') == 'PASS' and review.get('candidate_manifest_sha256') == release['source_manifest_sha256'] and review.get('execution_authorized') is False, 'Exact independent source PASS required')
    plan = json.loads((HERE / 'PLAN.json').read_text())
    require(release['caps'] == plan['stages'][stage]['caps'] and release['invocation'] == plan['stages'][stage]['invocation'], 'Engineering invocation/caps differ')
    profile = plan['execution_profile']
    require(sys.platform.startswith('linux') and socket.gethostname() == profile['host'], 'Exact normal Linux host required')
    require(os.environ.get('PYTHONPATH') == profile['project_PYTHONPATH'] and os.environ.get('PYTHONHASHSEED') == '0'
            and os.environ.get('OMP_NUM_THREADS') == '2' and os.environ.get('MKL_NUM_THREADS') == '2', 'Declared process environment differs')
    require(Path(sys.executable).resolve() == Path(profile['interpreter_path']).resolve() and sha(profile['interpreter_path']) == profile['interpreter_sha256'], 'Interpreter differs')
    for name, version in profile['distribution_versions'].items():
        require(importlib.metadata.version(name) == version, 'Distribution differs: ' + name)
    authority = json.loads((PHASE / binding['runtime_authority']['path']).read_text())
    for row in authority['runtime_source_pins'] + authority['runtime_binary_files'] + [authority['negative_sampler']]:
        require(sha(row['path']) == row['sha256'], 'Native runtime source/binary differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ('' if stage == 'cpu_bookkeeping' else profile['CUDA_VISIBLE_DEVICES']), 'Stage device visibility differs')
    if stage == 'real_graph':
        require(release.get('input_semantics_prerequisite_verified') is True and release.get('feature_equivalence_receipt_sha256') == binding['root_reported_feature_prerequisite']['receipt_sha256'], 'Root feature input prerequisite absent')
        require(release.get('released_VALID_pool_semantics_approved') is True and release.get('cpu_bookkeeping_receipt_sha256'), 'VALID semantics/CPU bookkeeping not released')
        cpu = EXECUTION / 'cpu_bookkeeping/run01/QUALIFICATION.json'
        require(sha(cpu) == release['cpu_bookkeeping_receipt_sha256'] and json.loads(cpu.read_text()).get('status') == 'PASS', 'Exact CPU bookkeeping PASS absent')
        cpu_terminal = EXECUTION / 'supervision/cpu_bookkeeping/run01/SUPERVISOR_TERMINAL.json'
        require(sha(cpu_terminal) == release['cpu_bookkeeping_supervision_terminal_sha256'] and json.loads(cpu_terminal.read_text()).get('status') == 'COMPLETE', 'CPU physical completion/caps absent')
    return release, plan


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


def compare(left, right, torch, atol, rtol, label, reports):
    maximum_abs = maximum_scaled = 0.0
    tensors = elements = 0
    def visit(a, b, name):
        nonlocal maximum_abs, maximum_scaled, tensors, elements
        require(type(a) is type(b), label + ': type differs at ' + name)
        if isinstance(a, torch.Tensor):
            require(a.shape == b.shape and a.dtype == b.dtype and a.layout == b.layout, label + ': tensor schema differs at ' + name)
            tensors += 1
            elements += a.numel()
            if a.is_floating_point():
                require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), label + ': nonfinite at ' + name)
                if a.numel():
                    delta = (a.to(torch.float64) - b.to(torch.float64)).abs()
                    allowed = atol + rtol * b.to(torch.float64).abs()
                    maximum_abs = max(maximum_abs, float(delta.max()))
                    maximum_scaled = max(maximum_scaled, float((delta / allowed.clamp_min(1e-300)).max()))
                    require(bool((delta <= allowed).all()), label + ': engineering tolerance exceeded at ' + name)
            else:
                require(torch.equal(a, b), label + ': discrete tensor differs at ' + name)
        elif isinstance(a, dict):
            require(a.keys() == b.keys(), label + ': keys differ at ' + name)
            for key in a:
                visit(a[key], b[key], name + '/' + str(key))
        elif isinstance(a, (list, tuple)):
            require(len(a) == len(b), label + ': sequence length differs at ' + name)
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, name + '/' + str(index))
        elif isinstance(a, float):
            require(abs(a - b) <= atol + rtol * abs(b), label + ': scalar differs at ' + name)
        else:
            require(a == b, label + ': value differs at ' + name)
    try:
        visit(left, right, '')
    finally:
        reports.append({'label': label, 'atol': atol, 'rtol': rtol, 'tensor_count': tensors, 'element_count': elements,
                        'observed_max_absolute_difference': maximum_abs, 'observed_max_fraction_of_allowed_error': maximum_scaled,
                        'accuracy_claim': 'Fixed-profile engineering parity only; no universal numerical-accuracy guarantee'})


def cpu_bookkeeping(output, torch, np, bodies, reports):
    from torch.utils.data import DataLoader
    h = torch.zeros((8, 1))
    fixture = lambda count: torch.tensor([[0, 1]], dtype=torch.long).expand(count, 2).clone()
    valid = fixture(2216)
    pool = fixture(2216 * 500).reshape(2216, 500, 2)
    omitted_a, omitted_b = fixture(2216), fixture(2051)
    omitted_pool = fixture(2051 * 500).reshape(2051, 500, 2)
    counts = {'reference_scorer_calls': 0, 'adapter_scorer_calls': 0}
    def score(which):
        def constant(a, b):
            counts[which] += 1
            return torch.zeros((*a.shape[:-1], 1))
        return constant
    bodies.seed_native(0)
    before = clone(rng(torch, np, False), torch)
    bodies.reference_sage_test_edge(score('reference_scorer_calls'), omitted_a, h, 1024)
    reference = bodies.reference_sage_test_edge(score('reference_scorer_calls'), valid, h, 1024, pool)
    bodies.reference_sage_test_edge(score('reference_scorer_calls'), omitted_b, h, 1024, omitted_pool)
    after_reference = clone(rng(torch, np, False), torch)
    reference_order = torch.cat(list(DataLoader(range(37676), 1024, shuffle=True)))
    next_reference = clone(rng(torch, np, False), torch)
    restore_rng(before, torch, np, False)
    iter(DataLoader((), batch_size=1024))
    candidate = bodies.candidate_sage_test_edge(score('adapter_scorer_calls'), valid, h, 1024, pool)
    iter(DataLoader((), batch_size=1024))
    compare(reference, candidate, torch, 0, 0, 'fabricated_complete_served_scores', reports)
    compare(after_reference, rng(torch, np, False), torch, 0, 0, 'omitted_iterator_full_CPU_RNG_cadence', reports)
    candidate_order = torch.cat(list(DataLoader(range(37676), 1024, shuffle=True)))
    compare(reference_order, candidate_order, torch, 0, 0, 'next_complete_native_SAGE_shuffle', reports)
    compare(next_reference, rng(torch, np, False), torch, 0, 0, 'after_next_complete_shuffle_RNG', reports)
    require(counts == {'reference_scorer_calls': 15, 'adapter_scorer_calls': 6}, 'Fabricated scorer coverage differs')
    return {'fabricated_inputs_only': True, 'omitted_view_row_counts': [2216, 2051], 'fabricated_VALID_shape': [2216, 500, 2],
            'evaluation_iterators': {'reference': 3, 'adapter': 3}, 'full_next_shuffle_rows_per_copy': 37676,
            'full_next_shuffle_batches_per_copy': 37, 'full_next_shuffle_tail': 812, 'counts': counts,
            'actual_TEST_rows_or_shape_queried': False, 'model_construction_or_optimizer_updates': 0}


def real_graph(output, torch, np, bodies, reports, plan, progress):
    sys.path.insert(0, str(ADAPTER))
    from inspect_available import inspect_available, load_available
    from evalutors import evaluate_mrr, eval_mrr
    inspection = inspect_available()
    write(output / 'AVAILABLE_INSPECTION.json', inspection)
    require(inspection['available_geometry_checks_pass'], 'Available geometry qualification failed')
    x, train_rows, valid_rows, pool, identities = load_available()
    require(list(x.shape) == [19717, 500], 'Root-reported raw feature geometry differs')
    train = torch.tensor(train_rows, dtype=torch.long)
    valid = torch.tensor(valid_rows, dtype=torch.long)
    negative = torch.from_numpy(np.array(pool, copy=True))
    atol, rtol = plan['engineering_tolerance']['atol'], plan['engineering_tolerance']['rtol']
    # Scalar-only observers preserve each native method's arguments, result and RNG.
    # They expose top-level and recursive work; no sampler or evaluator is changed.
    def observe_unit(model_name, which, unit):
        model, predictor, optimizer, ux, data, train_device = unit
        counters = progress['observed_calls'][model_name][which]
        def before_encoder(module, args):
            counters['encoder_forwards_started'] += 1
            write(output / 'PROGRESS.json', progress)
        def after_encoder(module, args, result):
            counters['encoder_forwards_completed'] += 1
            write(output / 'PROGRESS.json', progress)
        model.register_forward_pre_hook(before_encoder)
        model.register_forward_hook(after_encoder)
        if model_name == 'SAGE':
            def before_score(module, args):
                counters['scorer_calls_started'] += 1
                counters['scorer_pair_rows_started'] += args[0].numel() // args[0].shape[-1]
                write(output / 'PROGRESS.json', progress)
            def after_score(module, args, result):
                counters['scorer_calls_completed'] += 1
                counters['scorer_pair_rows_completed'] += result.numel()
                write(output / 'PROGRESS.json', progress)
            predictor.register_forward_pre_hook(before_score)
            predictor.register_forward_hook(after_score)
        else:
            original = predictor.multidomainforward
            def measured(*args, **kwargs):
                depth = kwargs.get('depth', args[5] if len(args) > 5 else None)
                depth = predictor.depth if depth is None else depth
                key = 'root' if depth == 1 else 'residual'
                counters[key + '_multidomain_calls_started'] += 1
                counters[key + '_pair_rows_started'] += int(args[2].shape[1])
                write(output / 'PROGRESS.json', progress)
                result = original(*args, **kwargs)
                counters[key + '_multidomain_calls_completed'] += 1
                counters[key + '_pair_rows_completed'] += int(args[2].shape[1])
                write(output / 'PROGRESS.json', progress)
                return result
            predictor.multidomainforward = measured
    def capture(unit):
        model, predictor, optimizer, ux, data, train_device = unit
        return clone({'encoder': model.state_dict(), 'predictor': predictor.state_dict(), 'Adam': optimizer.state_dict(),
                      'gradients': [{name: parameter.grad for name, parameter in module.named_parameters()} for module in (model, predictor)],
                      'flags': [[module.training for module in root.modules()] for root in (model, predictor)],
                      'invest': [getattr(module, 'invest', None) for module in (model, predictor)], 'RNG': rng(torch, np, True)}, torch)
    def restore(unit, state):
        model, predictor, optimizer, ux, data, train_device = unit
        model.load_state_dict(state['encoder'], strict=True)
        predictor.load_state_dict(state['predictor'], strict=True)
        optimizer.load_state_dict(state['Adam'])
        for index, module in enumerate((model, predictor)):
            for name, parameter in module.named_parameters():
                grad = state['gradients'][index][name]
                parameter.grad = None if grad is None else grad.to(parameter.device).clone()
            for child, flag in zip(module.modules(), state['flags'][index]):
                child.training = flag
            if state['invest'][index] is not None:
                module.invest = state['invest'][index]
        restore_rng(state['RNG'], torch, np, True)
    def serve(which, model_name, unit, start):
        restore(unit, start)
        model, predictor, optimizer, ux, data, train_device = unit
        scope = dict(vars(bodies))
        scope.update(model=model, predictor=predictor, model_name=model_name, x=ux, data=data, device=torch.device('cuda:0'),
                     valid=valid, valid_neg=negative, split_edge={'valid': {'edge': valid, 'edge_neg': negative}})
        function = bodies.bound_validation(which, model_name, scope)
        scores = clone(function(), torch)
        require(scores[0].shape == (2216,) and scores[1].shape == (2216, 500), 'Complete served score geometry differs')
        progress['complete_VALID_serves'] += 1
        write(output / 'PROGRESS.json', progress)
        return scores, capture(unit)
    def check_serves(model_name, a, b, label):
        compare(a, b, torch, atol, rtol, label + '/served_scores', reports)
        compare(eval_mrr(*a), eval_mrr(*b), torch, 0, 0, label + '/all_per_query_MRR_and_Hits', reports)
        require(evaluate_mrr(None, *a) == evaluate_mrr(None, *b), label + ': complete rounded native evaluator differs')
        # Evaluator agreement is reported as a boolean; no fitted metric value is published.
        progress['complete_native_evaluator_agreements'] += 1
    def roundtrip(model_name, unit, state, label):
        path = output / (model_name + '_' + label + '_ENGINEERING_ONLY_NEVER_DONOR.pt')
        torch.save(state, path)
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        compare(state, loaded, torch, 0, 0, model_name + '/' + label + '/serialized_full_state', reports)
        restore(unit, loaded)
        compare(state, capture(unit), torch, 0, 0, model_name + '/' + label + '/restored_next_step_state', reports)
        progress['owned_serializations'] += 1
        progress['owned_weights_only_loads'] += 1
        return loaded
    for model_name in ('SAGE', 'NCNC'):
        progress['current_model'] = model_name
        reference = bodies.reference_factory(model_name, 0, x, train)
        reference_initial = capture(reference)
        candidate = bodies.candidate_factory(model_name, 0, x, train)
        candidate_initial = capture(candidate)
        compare(reference_initial, candidate_initial, torch, 0, 0, model_name + '/initialized_complete_state', reports)
        for name, unit in (('reference', reference), ('candidate', candidate)):
            observe_unit(model_name, name, unit)
            def pre(optimizer, args, kwargs, name=name):
                progress['optimizer_calls_started'][model_name][name] += 1
                write(output / 'PROGRESS.json', progress)
            def post(optimizer, args, kwargs, name=name):
                progress['optimizer_calls_completed'][model_name][name] += 1
                write(output / 'PROGRESS.json', progress)
            unit[2].register_step_pre_hook(pre)
            unit[2].register_step_post_hook(post)
        for epoch in (0, 1, 2):
            a_scores, a_after = serve('reference', model_name, reference, reference_initial)
            b_scores, b_after = serve('candidate', model_name, candidate, candidate_initial)
            check_serves(model_name, a_scores, b_scores, model_name + '/after_epoch' + str(epoch))
            compare(a_after, b_after, torch, atol, rtol, model_name + '/after_serve_epoch' + str(epoch) + '/complete_state', reports)
            compare(a_after['RNG'], b_after['RNG'], torch, 0, 0, model_name + '/after_serve_epoch' + str(epoch) + '/exact_full_RNG', reports)
            if epoch == 2:
                break
            candidate_initial = roundtrip(model_name, candidate, b_after, 'initialized' if epoch == 0 else 'after_epoch1')
            reference_initial = a_after
            losses = []
            for which, unit, start in (('reference', reference, reference_initial), ('candidate', candidate, candidate_initial)):
                restore(unit, start)
                model, predictor, optimizer, ux, data, train_device = unit
                progress['native_epoch_calls_started'] += 1
                progress['current_call'] = {'model': model_name, 'copy': which, 'epoch': epoch + 1}
                write(output / 'PROGRESS.json', progress)
                if model_name == 'SAGE':
                    function = bodies.reference_sage_train if which == 'reference' else bodies.candidate_sage_train
                    loss = function(model, predictor, train_device, ux, optimizer, 1024)
                else:
                    function = bodies.reference_ncnc_train if which == 'reference' else bodies.candidate_ncnc_train
                    loss = function(model, predictor, data, {'train': {'edge': train}}, optimizer, 1024, True, [], None)
                losses.append(float(loss))
                progress['native_epoch_calls_completed'] += 1
                state = capture(unit)
                if which == 'reference':
                    reference_initial = state
                else:
                    candidate_initial = state
            compare(losses[0], losses[1], torch, atol, rtol, model_name + '/epoch' + str(epoch + 1) + '/native_loss', reports)
            compare(reference_initial, candidate_initial, torch, atol, rtol, model_name + '/epoch' + str(epoch + 1) + '/model_Adam_gradients_RNG_next_step_state', reports)
            compare(reference_initial['RNG'], candidate_initial['RNG'], torch, 0, 0, model_name + '/epoch' + str(epoch + 1) + '/exact_full_RNG', reports)
        del reference, candidate, reference_initial, candidate_initial
    require(progress['optimizer_calls_completed'] == {'SAGE': {'reference': 74, 'candidate': 74}, 'NCNC': {'reference': 72, 'candidate': 72}}, 'Native full epoch/tail update coverage differs')
    require(progress['native_epoch_calls_completed'] == 8 and progress['complete_VALID_serves'] == 12 and progress['owned_serializations'] == 4 and progress['complete_native_evaluator_agreements'] == 6, 'Qualification schedule incomplete')
    for which in ('reference', 'candidate'):
        sage = progress['observed_calls']['SAGE'][which]
        ncnc = progress['observed_calls']['NCNC'][which]
        require(sage['encoder_forwards_completed'] == 77 and sage['scorer_calls_completed'] == 166 and sage['scorer_pair_rows_completed'] == 3481352, 'Observed complete SAGE calls/rows differ')
        require(ncnc['encoder_forwards_completed'] == 75 and ncnc['root_multidomain_calls_completed'] == 174 and ncnc['root_pair_rows_completed'] == 3478104 and ncnc['residual_multidomain_calls_completed'] == 348, 'Observed complete NCNC calls/rows differ')
        for counters in (sage, ncnc):
            for key in counters:
                if key.endswith('_started'):
                    require(counters[key] == counters[key.replace('_started', '_completed')], 'Incomplete native observed call/rows')
    return {'input_hashes': identities, 'native_work': progress, 'selected_scientific_checkpoint': False,
            'negative_pool_preserved_without_global_nonlink_claim': True, 'data_class_labels_accessed': False}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--stage', required=True, choices=('cpu_bookkeeping', 'real_graph'))
    args = parser.parse_args()
    release, plan = gate(args.root_release, args.release_sha256, args.stage)
    output = EXECUTION / args.stage / 'run01'
    require(not output.exists(), 'Fresh owned output required; no retry/resume')
    output.mkdir(parents=True)
    reports = []
    progress = {'native_epoch_calls_started': 0, 'native_epoch_calls_completed': 0, 'complete_VALID_serves': 0,
                'complete_native_evaluator_agreements': 0, 'owned_serializations': 0, 'owned_weights_only_loads': 0,
                'optimizer_calls_started': {m: {c: 0 for c in ('reference', 'candidate')} for m in ('SAGE', 'NCNC')},
                'optimizer_calls_completed': {m: {c: 0 for c in ('reference', 'candidate')} for m in ('SAGE', 'NCNC')},
                'observed_calls': {m: {c: {key + suffix: 0 for key in (('encoder_forwards', 'scorer_calls', 'scorer_pair_rows') if m == 'SAGE' else
                   ('encoder_forwards', 'root_multidomain_calls', 'root_pair_rows', 'residual_multidomain_calls', 'residual_pair_rows')) for suffix in ('_started', '_completed')}
                   for c in ('reference', 'candidate')} for m in ('SAGE', 'NCNC')}}
    stopped = threading.Event()
    cuda_peaks = {'CUDA_observed': False, 'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0}
    caps = plan['stages'][args.stage]['caps']
    def observe_cuda():
        while not stopped.wait(.25):
            try:
                module = sys.modules.get('torch')
                if module is None or not module.cuda.is_initialized():
                    continue
                cuda_peaks['CUDA_observed'] = True
                for key, fn in (('cuda_peak_allocated_bytes', module.cuda.max_memory_allocated), ('cuda_peak_reserved_bytes', module.cuda.max_memory_reserved)):
                    cuda_peaks[key] = max(cuda_peaks[key], int(fn(0)))
                write(output / 'CUDA_PEAKS.json', cuda_peaks)
                if any(cuda_peaks[key] > caps[key] for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')):
                    write(output / 'FAILURE.json', {'status': 'FAILED', 'reason': 'CUDA peak cap', 'progress': progress, 'peaks': cuda_peaks})
                    os._exit(88)
            except BaseException as error:
                write(output / 'CUDA_MONITOR_FAILURE.json', {'status': 'FAILED', 'type': type(error).__name__, 'condition': str(error)})
                os._exit(89)
    observer = threading.Thread(target=observe_cuda, daemon=True)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(torch.__version__ == plan['execution_profile']['torch_version'] and torch.are_deterministic_algorithms_enabled() is False, 'Torch profile differs')
        if args.stage == 'real_graph':
            require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.version.cuda == '12.6', 'One native CUDA device required')
            properties = torch.cuda.get_device_properties(0)
            require(properties.name == plan['execution_profile']['gpu_name'] and properties.total_memory == plan['execution_profile']['gpu_total_memory_bytes'], 'Physical GPU profile differs')
            observer.start()
        sys.path.insert(0, str(AUTHOR))
        bodies = load_module('released_pubmed_native_qualification_bodies', HERE / 'native_bodies.py')
        result = cpu_bookkeeping(output, torch, np, bodies, reports) if args.stage == 'cpu_bookkeeping' else real_graph(output, torch, np, bodies, reports, plan, progress)
        actual_profile = {'default_dtype': str(torch.get_default_dtype()), 'torch_num_threads': torch.get_num_threads(),
                          'torch_num_interop_threads': torch.get_num_interop_threads(), 'matmul_TF32': torch.backends.cuda.matmul.allow_tf32,
                          'matmul_precision': torch.get_float32_matmul_precision(),
                          'cudnn_TF32': torch.backends.cudnn.allow_tf32, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
                          'cudnn_benchmark': torch.backends.cudnn.benchmark, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                          'autocast': torch.is_autocast_enabled()}
        expected_profile = {key: plan['execution_profile'][key] for key in actual_profile}
        expected_profile['default_dtype'] = 'torch.float32'
        require(actual_profile == expected_profile, 'Effective final runtime profile differs')
        if args.stage == 'cpu_bookkeeping':
            require(not torch.cuda.is_initialized(), 'CPU bookkeeping initialized CUDA unexpectedly')
        if args.stage == 'real_graph':
            torch.cuda.synchronize(0)
            cuda_peaks.update(CUDA_observed=True, cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)), cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
            require(all(cuda_peaks[key] <= caps[key] for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')), 'Final CUDA peak cap')
        write(output / 'QUALIFICATION.json', {'schema': 'pubmed-native-engineering-qualification-v1', 'status': 'PASS', 'stage': args.stage,
              'source_manifest_sha256': release['source_manifest_sha256'], 'root_release_sha256': args.release_sha256,
              'engineering_only': True, 'scientific_fit_admitted': False, 'state_donor': False, 'TEST_opened': False,
              'profile': plan['execution_profile'], 'actual_effective_runtime_profile': actual_profile, 'result': result, 'engineering_comparisons': reports,
              'cuda_peaks': cuda_peaks, 'child_inclusive_wall_seconds': time.monotonic() - started,
              'terminal_write_tail_measured': False, 'universal_numerical_accuracy_claim': False})
    except BaseException as error:
        write(output / 'FAILURE.json', {'status': 'FAILED', 'exception': type(error).__name__, 'condition': str(error), 'progress': progress,
              'engineering_comparisons': reports, 'cuda_peaks': cuda_peaks, 'child_inclusive_wall_seconds': time.monotonic() - started,
              'scientific_fit_admitted': False, 'automatic_retry': False})
        raise
    finally:
        stopped.set()
        if observer.is_alive():
            observer.join()
        write(output / 'CUDA_PEAKS.json', cuda_peaks)
        custody = [{'path': file.name, 'bytes': file.stat().st_size, 'sha256': sha(file)} for file in sorted(output.iterdir()) if file.is_file() and not file.name.endswith('.tmp')]
        write(output / 'FINAL_CUSTODY.json', {'files': custody, 'all_pt_files': 'Engineering-only owned continuation states; permanently barred as scientific donors',
              'no_tensor_or_state_payload_export_to_Mac': True, 'child_inclusive_wall_seconds': time.monotonic() - started, 'terminal_write_tail_measured': False})


if __name__ == '__main__':
    main()
