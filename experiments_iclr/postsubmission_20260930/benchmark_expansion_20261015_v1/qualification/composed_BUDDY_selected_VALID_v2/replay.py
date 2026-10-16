"""One repaired composed-BUDDY selected VALID replay; no TRAIN or reselection."""
import argparse
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
E = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'final_selected_molecular_VALID_cpu_v1'))
from support import allocation, digest, inside, receipt, verify, write_json
from qualify import tree_identity


def require(value, message):
    if not value: raise ValueError(message)


def relative(pin):
    require(set(pin) == {'path_relative_to_expansion', 'bytes', 'sha256'}, 'Exact expansion-relative receipt required')
    path = (E/pin['path_relative_to_expansion']).resolve()
    require(not Path(pin['path_relative_to_expansion']).is_absolute() and path.is_relative_to(E.resolve()),
            'Relative source receipt escapes expansion')
    return {'path': str(path), 'bytes': pin['bytes'], 'sha256': pin['sha256']}


def execute(root, admission_path):
    allocation(); root = Path(root).resolve(); inside(HERE, root)
    request = json.loads(inside(admission_path, root).read_text())
    require(request.get('schema') == 'composed-BUDDY-selected-VALID-replay-admission-v1'
            and request.get('status') == 'ADMITTED', 'Separate explicit root admission required')
    require(request['harness_manifest_sha256'] == receipt(HERE/'MANIFEST.json')['sha256']
            and request['INPUTS_sha256'] == receipt(HERE/'INPUTS.json')['sha256'], 'Exact replay source required')
    for pin in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        verify({**pin, 'path': str(HERE/pin['path'])}, root)
    inputs = json.loads((HERE/'INPUTS.json').read_text()); limits = inputs['resources']
    require(request['resources'] == limits and request['TRAIN_updates'] == request['TEST_calls'] == request['automatic_retries'] == 0
            and request['whole_VALID_calls'] == 1 and request['torch_threads'] == 2 and request['torch_interop_threads'] == 1
            and request['exclusive_shared_GPU_slot_held'] is True and request['python_executable'] == sys.executable,
            'Exact one-VALID, no-training, held-slot admission required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == 'GPU-44039938-fd82-41d2-fefd-de71514e2fac', 'Exact visible GPU required')
    for pin in [request['qualified_python_review'], request['peer_resource_review'], *inputs['reused_helpers']]: verify(pin, root)
    metadata = json.loads(verify(inputs['failed_qualification'], root).read_text())
    fit = json.loads(verify(inputs['completed_fit'], root).read_text())
    require(metadata['status'] == 'failed' and metadata['failure']['type'] == 'AttributeError'
            and metadata['failure']['message'] == "'types.SimpleNamespace' object has no attribute 'subgraph_feature_batch_size'"
            and metadata['TRAIN_updates'] == metadata['optimizer_completed'] == metadata['optimizer_attempts'] == 1585
            and metadata['native_constructions'] == 3 and metadata['mean_export_calls'] == 2,
            'Exact retained serving failure and paid completed training required')
    require([row['stage'] for row in metadata['invocations']] == ['one_complete_epoch', 'stopped_finalization_resume']
            and [row['new_updates'] for row in metadata['invocations']] == [1585, 0]
            and all(row['result']['status'] == 'complete' for row in metadata['invocations'])
            and fit == metadata['invocations'][-1]['result'], 'Successful original fit/resume must remain the authority')
    job = json.loads(inside(inputs['job_path'], root).read_text()); binding = fit['runtime_binding']
    require(digest(job) == inputs['job_sha256'] == fit['job_hash'] and job['TEST_access'] is False
            and job['family'] == 'link' and job['recipe_id'] == 'collab-buddy' and job['seed'] == 1
            and job['procedure'] == 'interaction_input_conditionalJ' and job['device'] == 'cuda:0'
            and job.get('torch_threads', 2) == 2 and job.get('normalization_composition'), 'Exact original composed JOB required')
    require(fit['selected_checkpoint'] == inputs['selected_checkpoint'] and fit['selected_VALID_export'] == inputs['selected_VALID_export']
            and fit['selected_epoch'] == fit['completed_epochs'] == 1 and fit['finalization_only_resume'] is True
            and not any(fit['optimizer_calls_this_invocation']['attempts'])
            and not any(fit['optimizer_calls_this_invocation']['completed']) and fit['TEST_accessed'] is False,
            'Exact completed own selection required')
    for pin in [inputs['producer_manifest'], inputs['producer_entrypoint'], inputs['selected_checkpoint'], inputs['selected_VALID_export']]: verify(pin, root)
    role = json.loads(verify(inputs['VALID_role_contract'], root).read_text())
    require(role['architecture'] == 'BUDDY' and role['rtol'] == 1e-4 and role['atol'] == 1e-5, 'Fixed BUDDY tolerance required')
    data = fit['source_metadata']['native_integration']['native_binding']['data_files']
    require(set(data) == {'graph', 'train', 'valid'} and all(role['data_files'][key] == pin for key, pin in data.items()), 'Exact original graph/TRAIN/VALID receipts required')
    for pin in data.values(): verify(pin, root)
    contract = {'architecture': 'BUDDY', 'data_files': data, 'topology': 'weighted_year2007_TRAIN'}
    output = inside(request['output'], root)
    require(output.is_relative_to(E/'qualification') and not output.exists(), 'Fresh replay output required')
    output.mkdir(parents=True); started = time.perf_counter(); owner = None; torch = None; hooks = []
    work = {'native_constructions_started': 0, 'native_constructions_completed': 0, 'selected_state_loads': 0,
            'whole_VALID_calls': 0, 'native_forwards_attempted': 0, 'native_forwards_completed': 0,
            'TRAIN_callback_attempts': 0, 'optimizer_step_attempts': 0, 'optimizer_steps': 0, 'forbidden_call_attempts': 0}
    result = {'schema': 'composed-BUDDY-selected-VALID-replay-v1', 'status': 'running', 'role': 'VALID',
              'TEST_accessed': False, 'TRAIN_updates': 0, 'automatic_retries': 0, 'full_campaign_GO': False,
              'admission': receipt(admission_path), 'retained_failed_qualification': inputs['failed_qualification'],
              'retained_paid_updates': 1585, 'selected_checkpoint': inputs['selected_checkpoint'],
              'selected_VALID_export': inputs['selected_VALID_export'], 'actual_work': work}
    write_json(output/'START.json', result)
    def timeout(*_): raise TimeoutError('Root-admitted selected-replay wall bound reached')
    signal.signal(signal.SIGALRM, timeout); signal.signal(signal.SIGTERM, timeout); signal.alarm(limits['wall_seconds'])
    try:
        sys.path.insert(0, str(E)); backend = importlib.import_module('composable_normalization_final_scoring_v3.backend')
        backend.source_authority(root, relative(inputs['scorer']))
        guards = backend.load_module(Path(inputs['reused_helpers'][0]['path']), '_retained_zero_training_guards')
        sys.path.insert(0, str(E/'composable_normalization_runtime_v1'))
        bridge = importlib.import_module('runtime_bridge')
        require(bridge.verify_sources(root) == inputs['producer_manifest'], 'Original genuine producer source differs')
        require('torch' not in sys.modules, 'Fresh worker required')
        import torch as tensor_library
        torch = tensor_library; torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(limits['GPU_reservation_bytes']/torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats(0)
        selected = torch.load(inputs['selected_checkpoint']['path'], map_location='cpu', weights_only=False)
        saved = torch.load(inputs['selected_VALID_export']['path'], map_location='cpu', weights_only=False)
        require(selected['schema'] == 'native-full-epoch-recovery-v1' and selected['job_hash'] == digest(job)
                and selected['runtime_binding'] == binding and selected['TEST_accessed'] is False
                and selected['selector']['selected_epoch'] == selected['selector']['completed_epoch'] == 1, 'Own selected checkpoint differs')
        require(saved['schema'] == 'own-selected-VALID-logits-v1' and saved['job_hash'] == digest(job)
                and saved['role'] == 'VALID' and saved['TEST_accessed'] is False and saved['selected_epoch'] == 1
                and saved['selected_checkpoint'] == inputs['selected_checkpoint'], 'Own selected VALID export differs')
        identity = digest(tree_identity(torch, selected['adapter'])); rng = digest(tree_identity(torch, selected['global_rng']))
        work['native_constructions_started'] += 1
        runtime, owner, actual_binding = bridge.load_runtime({'runtime_entrypoint': inputs['producer_entrypoint']}, job, root)
        work['native_constructions_completed'] += 1
        require(actual_binding == binding, 'Genuine producer binding differs; no checkpoint migration allowed')
        owner.load_state(copy.deepcopy(selected['adapter'])); work['selected_state_loads'] += 1
        owner.bind_loaded_selection(selected, selected_checkpoint=inputs['selected_checkpoint'], job_hash=digest(job), runtime_binding=binding)
        require(owner.completed_epochs == 1 and digest(tree_identity(torch, owner.state())) == identity, 'Complete selected model/bank restore differs')
        before = owner.state_signature(); optimizer = digest(tree_identity(torch, [o.state_dict() for o in owner.optimizers])); updates = owner.session.updates
        def objects():
            return [(scope, name, id(value), value.data_ptr()) for scope, values in [('parameters', owner.model.named_parameters()),
                    ('buffers', owner.model.named_buffers()), ('private_BN', owner.manager.named_buffers())] for name, value in values]
        pointers = objects(); native = owner.session.native
        require(native.members == 4, 'Exactly four original BUDDY routes required')
        def attempted(*_):
            require(work['native_forwards_attempted'] < 4, 'Four native forward bound exhausted'); work['native_forwards_attempted'] += 1
            with (output/'EVENTS.jsonl').open('a') as stream: stream.write(json.dumps({'native_forward_attempt': work['native_forwards_attempted'], 'UTC_unix': time.time()})+'\n')
        def completed(*_): work['native_forwards_completed'] += 1
        hooks = [native.model.register_forward_pre_hook(attempted), native.model.register_forward_hook(completed)]
        helper = importlib.import_module('final_test_serving_v1.link.backend')
        runtime.restore_rng(torch, selected['global_rng'], device='cuda:0')
        require(digest(tree_identity(torch, runtime.global_rng(torch, device='cuda:0'))) == rng, 'Saved full RNG restore differs')
        with guards.inference_only(torch, owner, work, backend.no_updates):
            work['whole_VALID_calls'] += 1
            replay = backend.predict(torch, owner, job, 'VALID', contract, root, link_helper=helper, cache_root=output/'serving_cache')
        require(owner.state_signature() == before and objects() == pointers and owner.session.updates == updates
                and digest(tree_identity(torch, [o.state_dict() for o in owner.optimizers])) == optimizer, 'Serving changed model/private banks/optimizer/TRAIN clock')
        require(work['native_forwards_attempted'] == work['native_forwards_completed'] == 4
                and replay['work']['native_BUDDY_forwards'] == 4 and all(work[k] == 0 for k in
                ('TRAIN_callback_attempts', 'optimizer_step_attempts', 'optimizer_steps', 'forbidden_call_attempts')), 'Native work differs')
        mean = saved['same_bank_mean_VALID']
        require(mean['schema'] == 'own-selected-same-bank-mean-VALID-v1' and mean['role'] == 'VALID' and mean['TEST_accessed'] is False
                and mean['extra_forward_calls'] == 0 and mean['selected_checkpoint'] == inputs['selected_checkpoint']
                and mean['selected_epoch'] == 1 and mean['job_hash'] == digest(job) and mean['runtime_binding'] == binding, 'Own same-bank mean authority differs')
        for name in ('ids', 'labels', 'query_endpoints'):
            require(replay[name].dtype == saved[name].dtype == mean[name].dtype and replay[name].shape == saved[name].shape == mean[name].shape
                    and torch.equal(replay[name], saved[name]) and torch.equal(saved[name], mean[name]), 'Exact VALID '+name+' differs')
        require(len(replay['ids']) == 160084 and torch.equal(replay['ids'], torch.arange(160084)), 'Whole native VALID population required')
        parity = {}
        for name, reference in [('logits', saved['raw_logits']), ('mean', mean['raw_logits'])]:
            value = replay[name]; require(value.shape == reference.shape and value.dtype == reference.dtype
                    and torch.isfinite(value).all().item() and torch.isfinite(reference).all().item(), 'Finite logit schema differs')
            delta = (value.double()-reference.double()).abs()
            parity[name] = {'passed': bool(torch.allclose(value, reference, rtol=1e-4, atol=1e-5)), 'max_absolute_difference': float(delta.max()),
                            'coordinates_outside_tolerance': int((delta > 1e-5+1e-4*reference.double().abs()).sum())}
        require(identity == digest(tree_identity(torch, selected['adapter'])) and rng == digest(tree_identity(torch, selected['global_rng'])), 'Saved state/RNG payload changed')
        result.update(status='passed' if all(v['passed'] for v in parity.values()) else 'failed_numerical_reconstruction', parity=parity,
                      serving_work=replay['work'], readout=replay['readout'], runtime_binding=binding, current_serving_source=relative(inputs['scorer']),
                      exact_selected_state_and_RNG_restored=True, exact_IDs_labels_endpoints=True, parameters_BN_buffers_optimizer_unchanged=True)
        torch.save({'schema': result['schema'], 'role': 'VALID', 'TEST_accessed': False, 'selected_checkpoint': inputs['selected_checkpoint'],
                    'replay': replay, 'parity': parity}, output/'VALID.pt'); result['VALID_export'] = receipt(output/'VALID.pt')
        for pin in [inputs['failed_qualification'], inputs['completed_fit'], inputs['selected_checkpoint'], inputs['selected_VALID_export'], *data.values()]: verify(pin, root)
    except BaseException as error:
        result.update(status='failed', failure={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(limit=16)})
    finally:
        signal.alarm(0)
        for hook in hooks: hook.remove()
        if owner is not None:
            try: owner.close()
            except Exception as error: result.update(status='failed', close_error=str(error))
        if torch is not None and torch.cuda.is_initialized():
            result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        use = resource.getrusage(resource.RUSAGE_SELF)
        result.update(wall_seconds=time.perf_counter()-started, own_peak_RSS_bytes=use.ru_maxrss*1024, process_user_seconds=use.ru_utime, process_system_seconds=use.ru_stime)
        if result['own_peak_RSS_bytes'] > limits['host_RSS_reservation_bytes']: result.update(status='failed', host_reservation_exceeded=True)
        write_json(output/'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); parser.add_argument('--admission', required=True)
    args = parser.parse_args(); result = execute(args.root, args.admission)
    print(json.dumps({key: result.get(key) for key in ('status', 'actual_work', 'parity', 'failure')}))
    raise SystemExit(0 if result['status'] == 'passed' else 2)
