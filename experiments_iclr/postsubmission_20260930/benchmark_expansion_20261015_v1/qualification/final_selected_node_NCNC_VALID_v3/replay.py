"""Two retained own selections, original devices, one current whole VALID call."""
import argparse
from contextlib import contextmanager
import copy
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'final_selected_molecular_VALID_cpu_v1'))
from support import allocation, digest, inside, receipt, verify, write_json
from qualify import no_training, tree_identity


def require(value, message):
    if not value: raise ValueError(message)


def relative_pin(pin):
    return {'path': str(EXPANSION / pin['path_relative_to_expansion']),
            'bytes': pin['bytes'], 'sha256': pin['sha256']}


def admitted(case_id, root, request_path):
    allocation()
    root = Path(root).resolve(); inside(HERE, root)
    request = json.loads(inside(request_path, root).read_text())
    require(request.get('schema') == 'genuine-selected-node-NCNC-VALID-admission-v3'
            and request.get('status') == 'ADMITTED', 'Explicit root admission required')
    require(request.get('harness_manifest_sha256') == receipt(HERE/'MANIFEST.json')['sha256'],
            'Admission must pin this exact source')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        verify({'path': str(HERE/row['path']), 'bytes': row['bytes'], 'sha256': row['sha256']}, root)
    inputs = json.loads((HERE/'INPUTS.json').read_text())
    require(request.get('INPUTS_sha256') == receipt(HERE/'INPUTS.json')['sha256'], 'Frozen inputs differ')
    for pin in [inputs['mapping'], inputs['scorer'], inputs['node_provider_authority'], *inputs['reused_harness']]:
        verify(relative_pin(pin), root)
    limits = next(row for row in inputs['cases'] if row['id'] == case_id)
    require(request.get('case_id') == case_id and request.get('resources') == limits,
            'One exact finite original-device case required')
    require(request.get('TRAIN_updates') == 0 and request.get('TEST_calls') == 0
            and request.get('automatic_retries') == 0 and request.get('whole_VALID_calls') == 1
            and request.get('torch_threads') == 2 and request.get('torch_interop_threads') == 1,
            'Fixed one-VALID/no-training/no-TEST scope required')
    require(request.get('python_executable') == sys.executable, 'Root-qualified interpreter required')
    verify(request['qualified_python_review'], root); verify(request['peer_resource_review'], root)
    require(request.get('exclusive_shared_GPU_slot_held') is (limits['device'] != 'cpu'),
            'Root must hold the shared GPU slot for the CUDA case')
    visible = '' if limits['device'] == 'cpu' else 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == visible, 'Exact visible device required before tensor import')
    output = inside(request['output'], root)
    require(output.is_relative_to(EXPANSION/'qualification') and not output.exists(),
            'Fresh own qualification output required; preserve every attempt')
    mapping = json.loads(verify(relative_pin(inputs['mapping']), root).read_text())
    case = next(row for row in mapping['existing_selected_inputs'] if row['id'] == case_id)
    require(mapping['fixed_tolerances'] == inputs['fixed_tolerances'] == {'rtol': 1e-4, 'atol': 1e-5},
            'Original fixed tolerance required')
    return root, output, request, inputs, limits, case


@contextmanager
def inference_only(torch, adapter, work, no_updates):
    protected = []
    def blocked(*args, **kwargs):
        work['forbidden_call_attempts'] += 1
        raise RuntimeError('This replay forbids backward, TRAIN, and extra validation callbacks')
    owners = [adapter, adapter.session, getattr(adapter.session, 'native', None)]
    for owner in owners:
        for name in ('validate', 'train_update', '_next_train_batch'):
            if owner is not None and hasattr(owner, name):
                protected.append((owner, name, getattr(owner, name))); setattr(owner, name, blocked)
    for owner, name in [(torch.Tensor, 'backward'), (torch.autograd, 'backward'), (torch.autograd, 'grad')]:
        protected.append((owner, name, getattr(owner, name))); setattr(owner, name, blocked)
    try:
        with no_training(adapter, work, no_updates): yield
    finally:
        for owner, name, value in reversed(protected): setattr(owner, name, value)


def execute(case_id, root, request_path):
    root, output, request, inputs, limits, case = admitted(case_id, root, request_path)
    output.mkdir(parents=True)
    started = time.perf_counter(); adapter = None; torch = None; hooks = []
    device = limits['device']
    work = {'native_constructions_started': 0, 'native_constructions_completed': 0,
            'selected_state_loads': 0, 'whole_VALID_calls': 0, 'TRAIN_callback_attempts': 0,
            'optimizer_step_attempts': 0, 'optimizer_steps': 0, 'forbidden_call_attempts': 0,
            'native_forwards_attempted': 0, 'native_forwards_completed': 0,
            'predictor_top_level_attempted': 0, 'predictor_top_level_completed': 0}
    result = {'schema': 'genuine-selected-node-NCNC-VALID-result-v1', 'case_id': case_id,
              'status': 'running', 'role': 'VALID', 'TEST_accessed': False, 'device': device,
              'scope': 'genuine retained producer and original device plus current role-serving components; not current portable runtime restoration',
              'job_hash': case['job_hash'], 'selected_epoch': case['selected_epoch'],
              'selected_checkpoint': case['selected_checkpoint'], 'selected_VALID_export': case['selected_VALID_export'],
              'mapping': relative_pin(inputs['mapping']), 'admission': receipt(request_path),
              'harness': receipt(HERE/'MANIFEST.json'), 'actual_work': work,
              'automatic_retries': 0, 'selection_changed': False, 'rtol': 1e-4, 'atol': 1e-5}
    write_json(output/'START.json', result)
    def timeout(*_): raise TimeoutError('Fixed selected VALID replay wall bound reached')
    signal.signal(signal.SIGALRM, timeout); signal.signal(signal.SIGTERM, timeout)
    signal.alarm(limits['wall_seconds'])
    try:
        require('torch' not in sys.modules, 'Fresh process required')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '2'
        import torch as tensor_library
        torch = tensor_library
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        if device == 'cpu': require(not torch.cuda.is_initialized(), 'Node replay cannot initialize CUDA')
        else:
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(limits['GPU_reservation_bytes']/torch.cuda.get_device_properties(0).total_memory, 0)
            torch.cuda.reset_peak_memory_stats(0)
        job = json.loads(verify(relative_pin(case['descriptor_receipt']), root).read_text())
        metadata = json.loads(verify(relative_pin(case['metadata_receipt']), root).read_text())
        fit = metadata['invocations'][-1]['result']
        binding = case['genuine_runtime_binding']
        require(metadata['status'] == 'passed' and fit['status'] == 'complete'
                and fit['runtime_binding'] == binding and fit['job_hash'] == digest(job) == case['job_hash']
                and job['id'] == case['job_id'] and job['device'] == device
                and job['procedure'] == case['procedure'] and job['family'] == case['family']
                and job['TEST_access'] is False and fit['TEST_accessed'] is False,
                'Exact completed genuine producer identity required')
        sources = fit['runtime_sources']
        require({name: pin['sha256'] for name, pin in sources.items()} == binding['runtime_sha256'],
                'Genuine original runtime source binding differs')
        paths = [verify(pin, root) for pin in sources.values()]; runtime = paths[0].parent
        require(all(path.parent == runtime for path in paths), 'One coherent original runtime required')
        for path, sha in binding['dependency_sha256'].items(): verify({'path': path, 'sha256': sha}, root)
        verify({'path': job['callback_module'], 'sha256': binding['callback_sha256']}, root)
        if case_id == 'node_C4':
            provider = json.loads(verify(relative_pin(inputs['node_provider_authority']), root).read_text())
            data_pins = provider['node_data_files']
            extension = job['session_transform']
            verify({'path': extension['path'], 'sha256': binding['extension_sha256']}, root)
        else:
            data_pins = fit['source_metadata']['native_integration']['native_binding']['data_files']
            require(set(data_pins) == {'graph', 'train', 'valid'}, 'VALID contract excludes TEST')
            require({key: pin['path'] for key, pin in data_pins.items()} == job['recipe']['data_files'],
                    'Native graph/TRAIN/VALID provider paths differ')
            verify({'path': job['link_policy_module'], 'sha256': binding['policy_sha256']}, root)
        for pin in data_pins.values(): verify(pin, root)
        result['original_data_pins'] = data_pins
        checkpoint = torch.load(verify(case['selected_checkpoint'], root), map_location='cpu', weights_only=False)
        saved = torch.load(verify(case['selected_VALID_export'], root), map_location='cpu', weights_only=False)
        epoch = checkpoint['selector']['selected_epoch']
        require(checkpoint['schema'] == 'native-full-epoch-recovery-v1' and checkpoint['job_hash'] == case['job_hash']
                and checkpoint['runtime_binding'] == binding and checkpoint['TEST_accessed'] is False
                and type(epoch) is int and epoch > 0
                and epoch == case['selected_epoch'] == checkpoint['selector']['completed_epoch'] == fit['selected_epoch'],
                'Exact own positive selected checkpoint required')
        require(saved['schema'] == 'own-selected-VALID-logits-v1' and saved['job_hash'] == case['job_hash']
                and saved['role'] == 'VALID' and saved['TEST_accessed'] is False
                and saved['selected_checkpoint'] == case['selected_checkpoint'] and saved['selected_epoch'] == epoch,
                'Exact own selected VALID export required')
        state_identity = digest(tree_identity(torch, checkpoint['adapter']))
        rng_identity = digest(tree_identity(torch, checkpoint['global_rng']))
        require(not any(name in sys.modules for name in ('adapters', 'bookkeeping', 'common', 'controllers')),
                'Fresh original-runtime namespace required')
        sys.path.insert(0, str(runtime))
        factory = importlib.import_module('adapters'); common = importlib.import_module('common')
        policy, _, policy_pin = importlib.import_module('controllers').controller(job, root)
        require((None if policy_pin is None else policy_pin['sha256']) == binding['policy_sha256'], 'Policy differs')
        sys.path.insert(0, str(EXPANSION))
        serving = importlib.import_module('final_selected_scoring_v5.role_serving')
        backend = importlib.import_module('final_selected_scoring_v5.backend')
        backend.source_authority(root, relative_pin(inputs['scorer']))
        helper = importlib.import_module('final_test_serving_v1.link.backend') if case_id == 'NCNC_C4' else None
        common.seed_all(torch, job['seed'], device=device)
        work['native_constructions_started'] += 1
        adapter = factory.create_adapter(torch, copy.deepcopy(job), policy, root)
        work['native_constructions_completed'] += 1
        require(adapter.callback_receipt['sha256'] == binding['callback_sha256']
                and adapter.extension_receipt['sha256'] == binding['extension_sha256']
                and {p['path']: p['sha256'] for p in adapter.dependency_receipts} == binding['dependency_sha256'],
                'Actual constructed producer differs')
        adapter.load_state(copy.deepcopy(checkpoint['adapter'])); work['selected_state_loads'] += 1
        if hasattr(adapter, 'bind_loaded_selection'):
            adapter.bind_loaded_selection(checkpoint, selected_checkpoint=case['selected_checkpoint'],
                                          job_hash=case['job_hash'], runtime_binding=binding)
        require(adapter.completed_epochs == epoch, 'Restored selected epoch differs')
        require(digest(tree_identity(torch, adapter.state())) == state_identity, 'Complete selected adapter restore differs')
        before = backend.model_fingerprint(adapter.model)
        def objects():
            return [(scope, name, id(value), value.data_ptr(), str(value.device), str(value.dtype), list(value.shape))
                    for scope, iterator in [('parameter', adapter.model.named_parameters()), ('buffer', adapter.model.named_buffers())]
                    for name, value in iterator]
        before_objects = objects()
        optimizer_before = digest(tree_identity(torch, [opt.state_dict() for opt in adapter.optimizers]))
        updates_before = adapter.session.updates
        def count(module, prefix, maximum):
            def attempted(*_):
                require(work[prefix+'_attempted'] < maximum, 'Fixed native forward bound exhausted')
                work[prefix+'_attempted'] += 1
                with (output/'EVENTS.jsonl').open('a') as stream:
                    stream.write(json.dumps({'event': prefix, 'attempt': work[prefix+'_attempted'], 'UTC_unix': time.time()})+'\n')
            def completed(*_): work[prefix+'_completed'] += 1
            hooks.extend([module.register_forward_pre_hook(attempted), module.register_forward_hook(completed)])
        if case_id == 'node_C4':
            native = adapter.model.native; contract = None
            expected_ids = torch.nonzero(adapter.session.data.val_mask, as_tuple=False).flatten().cpu()
        else:
            native = adapter.session.native.encoder
            require(adapter.session.native.members == 4 and adapter.session.native.test_batch_size == 131072,
                    'Original NCNC four routes and source query batching required')
            count(adapter.session.native.predictor, 'predictor_top_level', 8)
            contract = {'architecture': 'NCNC', 'data_files': data_pins}
            expected_ids = torch.arange(160084)
        count(native, 'native_forwards', 4)
        common.restore_rng(torch, checkpoint['global_rng'], device=device)
        require(digest(tree_identity(torch, common.global_rng(torch, device=device))) == rng_identity, 'Original full RNG restore differs')
        result['construction_and_restore_seconds'] = time.perf_counter()-started
        with inference_only(torch, adapter, work, backend.no_updates):
            if device != 'cpu': torch.cuda.synchronize()
            forward_start = time.perf_counter(); work['whole_VALID_calls'] += 1
            replay = serving.predict(torch, adapter, job, 'VALID', contract, root, link_helper=helper, cache_root=output/'unused_cache')
            if device != 'cpu': torch.cuda.synchronize()
            result['VALID_seconds'] = time.perf_counter()-forward_start
        for name in ('ids', 'labels') + (('query_endpoints',) if case_id == 'NCNC_C4' else ()):
            require(replay[name].dtype == saved[name].dtype and replay[name].shape == saved[name].shape
                    and torch.equal(replay[name], saved[name]), 'Exact ordered VALID '+name+' differs')
        require(torch.equal(replay['ids'], expected_ids) and len(replay['ids']) == limits['VALID_records'],
                'Whole original VALID population required')
        require(work['native_forwards_attempted'] == work['native_forwards_completed'] == 4, 'Four native route forwards required')
        require(all(work[name] == 0 for name in ('TRAIN_callback_attempts', 'optimizer_step_attempts',
                                                'optimizer_steps', 'forbidden_call_attempts')),
                'Forbidden work was attempted')
        if case_id == 'NCNC_C4':
            n = adapter.session.native
            require(work['predictor_top_level_attempted'] == work['predictor_top_level_completed'] == 8,
                    'Eight source top-level predictor calls required')
            require(replay['work']['NCNC_output_routing'] == metadata['VALID_receipts'][-1]['work']['NCNC_output_routing']
                    and not n.task_call_scope.depth and not adapter.session._depth,
                    'Original task-only output and recursive bypass scope differs')
            require(torch.equal(replay['query_endpoints'], torch.cat((n.valid['edge'], n.valid['edge_neg'])).cpu()),
                    'Native ordered query endpoints differ')
        require(objects() == before_objects and backend.model_fingerprint(adapter.model) == before
                and digest(tree_identity(torch, [opt.state_dict() for opt in adapter.optimizers])) == optimizer_before
                and adapter.session.updates == updates_before, 'Inference changed model/optimizer/TRAIN state')
        require(state_identity == digest(tree_identity(torch, checkpoint['adapter']))
                and rng_identity == digest(tree_identity(torch, checkpoint['global_rng'])), 'Saved state/RNG payload changed')
        mean = saved['same_bank_mean_VALID']
        require(mean['schema'] == 'own-selected-same-bank-mean-VALID-v1' and mean['role'] == 'VALID'
                and mean['TEST_accessed'] is False and mean['extra_forward_calls'] == 0
                and mean['selected_checkpoint'] == case['selected_checkpoint'] and mean['selected_epoch'] == epoch
                and mean['job_hash'] == case['job_hash'] and mean['runtime_binding'] == binding,
                'Own selected same-bank mean authority differs')
        for name in ('ids', 'labels') + (('query_endpoints',) if case_id == 'NCNC_C4' else ()):
            require(mean[name].dtype == saved[name].dtype and mean[name].shape == saved[name].shape
                    and torch.equal(mean[name], saved[name]), 'Saved actual/mean identity differs: '+name)
        parity = {}
        for name, reference in [('logits', saved['raw_logits']), ('mean', mean['raw_logits'])]:
            value = replay[name]
            require(value.shape == reference.shape and value.dtype == reference.dtype
                    and torch.isfinite(value).all().item() and torch.isfinite(reference).all().item(), 'Finite selected logit schema differs')
            delta = (value.double()-reference.double()).abs()
            parity[name] = {'passed': bool(torch.allclose(value, reference, rtol=1e-4, atol=1e-5)),
                            'max_absolute_difference': float(delta.max()),
                            'coordinates_outside_tolerance': int((delta > 1e-5+1e-4*reference.double().abs()).sum())}
        require(torch.isfinite(replay['members']).all().item(), 'Nonfinite native route output')
        result.update(status='passed' if all(row['passed'] for row in parity.values()) else 'failed_numerical_reconstruction',
                      parity=parity, serving_work=replay['work'], readout=replay['readout'],
                      exact_IDs_labels_endpoints_order=True, parameters_buffers_optimizer_unchanged=True,
                      exact_selected_state_and_RNG_restored=True, current_serving_source=relative_pin(inputs['scorer']))
        torch.save({'schema': 'genuine-selected-node-NCNC-VALID-replay-v1', 'case_id': case_id,
                    'job_hash': case['job_hash'], 'selected_epoch': epoch, 'selected_checkpoint': case['selected_checkpoint'],
                    'role': 'VALID', 'TEST_accessed': False, 'replay': replay, 'parity': parity}, output/'VALID.pt')
        result['VALID_export'] = receipt(output/'VALID.pt')
        for pin in [case['selected_checkpoint'], case['selected_VALID_export'], *data_pins.values()]: verify(pin, root)
    except BaseException as error:
        result.update(status='failed', failure={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(limit=12)})
    finally:
        signal.alarm(0)
        for hook in hooks: hook.remove()
        if adapter is not None:
            try: adapter.close()
            except Exception as error: result.update(status='failed', close_error=str(error))
        if torch is not None:
            result['CUDA_context_initialized'] = bool(torch.cuda.is_initialized())
            if device == 'cpu' and result['CUDA_context_initialized']: result.update(status='failed', CUDA_violation=True)
            elif device != 'cpu' and result['CUDA_context_initialized']:
                result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        use = resource.getrusage(resource.RUSAGE_SELF)
        result.update(wall_seconds=time.perf_counter()-started, own_peak_RSS_bytes=use.ru_maxrss*1024,
                      process_user_seconds=use.ru_utime, process_system_seconds=use.ru_stime)
        if result['own_peak_RSS_bytes'] > limits['host_RSS_reservation_bytes']:
            result.update(status='failed', host_reservation_exceeded=True)
        write_json(output/'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True); parser.add_argument('--admission', required=True)
    parser.add_argument('--case', required=True, choices=['node_C4', 'NCNC_C4'])
    args = parser.parse_args(); result = execute(args.case, args.root, args.admission)
    print(json.dumps({'case_id': result['case_id'], 'status': result['status'], 'wall_seconds': result['wall_seconds']}))
    raise SystemExit(0 if result['status'] == 'passed' else 2)
