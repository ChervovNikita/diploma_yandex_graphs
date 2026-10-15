"""Final scoring APIs; no imports or execution of model/data code at import time.

Call only in a fresh reservation-owned worker after the same campaign GO has
authorized its frozen training and final-scoring package. This module does
not launch workers, allocate another GPU, train, choose checkpoints or push Git.
"""
from contextlib import contextmanager
from collections import OrderedDict
import copy
import importlib.util
import json
from pathlib import Path
import sys
import struct
import time
import traceback
from .contracts import (authorize, digest, inside, readout_policy, receipt, require,
                       selected_role_contract, verify)
from .role_serving import metrics, predict, tensor_sha

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source_authority(root, expected):
    require(expected['path'] == str(HERE / 'MANIFEST.json'), 'Use this frozen scoring source manifest')
    manifest = verify(expected, root, json_metadata=True)
    require(manifest['schema'] == 'composed-normalization-final-scoring-source-v2', 'Wrong final-scoring source schema')
    for row in manifest['files']:
        verify({'path': str(HERE / row['path']), 'bytes': row['bytes'], 'sha256': row['sha256']}, root)
    bindings = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        verify({'path': str(EXPANSION.parent / row['path_relative_to_phase']),
                'bytes': row['bytes'], 'sha256': row['sha256']}, root)
    return manifest


def context(request, go_pin, root, admit_runtime):
    require(callable(admit_runtime), 'Existing resource/host admission callback required')
    admit_runtime()  # Host/resource/source route gate precedes project/data reads.
    root = Path(root).resolve()
    expected = request['campaign_scoring_authority']['source_manifest']
    source_authority(root, expected)
    go = verify(go_pin, root, json_metadata=True)
    task = authorize(request, go, root, expected)
    roster = verify(request['campaign_scoring_authority']['roster'], root, json_metadata=True)
    index = verify(roster['registry_index'], root, json_metadata=True)
    require(request['profile'] in index['preset_files'], 'Profile absent from the frozen exact reporting registry')
    profile = verify(request['profile'], root, json_metadata=True)
    role = verify(task['role_contract'], root, json_metadata=True)
    require(task['preset_id'] == profile['registry']['presets'][0]['id'] == role['preset_id'], 'Preset identity differs')
    output = inside(request['output'], root)
    require(not output.exists(), 'Preserve previous final-scoring outcomes; no automatic retry')
    output.mkdir(parents=True)
    return root, task, profile, role, output


def provider_authority(contract, identity, root):
    population = contract['population']
    proof = verify(population['provider_evidence'], root, json_metadata=True)
    require(proof.get('schema') == 'native-final-TEST-population-evidence-v2'
            and proof.get('role') == 'TEST' and proof.get('complete_official_population') is True
            and proof.get('dataset') == identity['dataset'] and proof.get('preset_id') == identity['preset_id']
            and proof.get('split_index') == identity['split_index']
            and proof.get('records') == population['records'] and proof.get('ids_sha256') == population['ids_sha256'],
            'Official complete TEST population authority differs')
    require(proof.get('source_receipts'), 'Official role evidence must bind the actual input sources')
    for pin in proof['source_receipts']:
        verify(pin, root)
    if contract['family'] == 'link':
        require(proof.get('query_file') == contract['data_files']['test']
                and proof.get('positive') == population['positive'] and proof.get('negative') == population['negative'],
                'Link official-query authority differs')
    if contract['family'] == 'ogb':
        require(proof.get('test_indices') == contract['test_indices'], 'Molecular scaffold role authority differs')
    return proof


def load_runtime(spec, job, root):
    entry = verify(spec['runtime_entrypoint'], root)
    if job.get('normalization_composition') is not None:
        bridge = load_module(EXPANSION / 'composable_normalization_runtime_v1/runtime_bridge.py',
                             '_normalization_final_runtime_bridge')
        return bridge.load_runtime(spec, job, root)
    family = 'runtime_link' if job['family'] == 'link' else 'runtime_node_molecular'
    require(entry == EXPANSION / 'portable_dispatcher_v3' / family / 'run_fit.py', 'Exact coherent portablev3 runtime required')
    folder = entry.parent
    for name in ('common', 'controllers', 'bookkeeping', 'source_rng', 'adapters', 'link_authority'):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve().parent == folder,
                    'Fresh scoring worker required; conflicting generic runtime import')
    sys.path.insert(0, str(folder))
    runtime = load_module(entry, '_final_scoring_runtime_' + spec['runtime_entrypoint']['sha256'])
    runtime.bound_project_root(root)
    runtime.validate_host_job(job, root)
    policy = runtime.validate_job(job, root)
    policy, _, policy_pin = runtime.controller(job, root)
    from adapters import create_adapter
    torch = sys.modules['torch']
    torch.set_num_threads(int(job.get('torch_threads', 2)))
    runtime.seed_all(torch, job['seed'], device=job.get('device', 'cuda:0'))
    adapter = create_adapter(torch, job, policy, root)
    sources = {name: receipt(folder / name, root) for name in
               ('run_fit.py', 'adapters.py', 'bookkeeping.py', 'common.py', 'controllers.py', 'source_rng.py')}
    binding = {'runtime_sha256': {name: p['sha256'] for name, p in sources.items()},
               'policy_sha256': None if policy_pin is None else policy_pin['sha256'],
               'callback_sha256': adapter.callback_receipt['sha256'],
               'extension_sha256': None if adapter.extension_receipt is None else adapter.extension_receipt['sha256'],
               'dependency_sha256': {p['path']: p['sha256'] for p in adapter.dependency_receipts}}
    return runtime, adapter, binding


@contextmanager
def no_updates(adapter):
    originals = [(optimizer, optimizer.step) for optimizer in adapter.optimizers]
    def refused(*args, **kwargs):
        raise RuntimeError('Final scoring forbids every optimizer update')
    for optimizer, _ in originals:
        optimizer.step = refused
    try:
        yield
    finally:
        for optimizer, original in originals:
            optimizer.step = original


def _state_fingerprint(value, is_tensor):
    """Preserve tensors and typed extra state; never drop or stringify objects."""
    if is_tensor(value):
        # Keep the existing tensor-only fingerprint representation unchanged.
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': tensor_sha(value)}
    kind = type(value)
    if value is None:
        return {'state_type': 'NoneType'}
    if kind in (bool, int, str):
        return {'state_type': kind.__name__, 'value': value}
    if kind is float:
        # IEEE bytes preserve signed zero and nonfinite payloads without NaN
        # equality or JSON-number ambiguities. No numeric coercion is applied.
        return {'state_type': 'float', 'ieee754_hex': struct.pack('>d', value).hex()}
    if kind is bytes:
        return {'state_type': 'bytes', 'hex': value.hex()}
    if kind in (dict, OrderedDict):
        return {'state_type': kind.__name__, 'items': [
            [_state_fingerprint(key, is_tensor), _state_fingerprint(item, is_tensor)]
            for key, item in value.items()]}
    if kind in (list, tuple):
        return {'state_type': kind.__name__, 'items': [_state_fingerprint(item, is_tensor) for item in value]}
    raise TypeError('Unsupported model extra-state type: ' + kind.__module__ + '.' + kind.__qualname__)

def model_fingerprint(adapter):
    if getattr(adapter, 'composed_producer', False):
        return adapter.state_signature()
    import torch
    model = adapter.model
    return {name: _state_fingerprint(value, torch.is_tensor) for name, value in model.state_dict().items()}


def attach_metrics(torch, result, job):
    require(torch.isfinite(result['logits']).all().item()
            and torch.isfinite(result['members']).all().item(), 'Nonfinite final prediction')
    score, loss = metrics(torch, job['family'], job['recipe']['dataset'], result['logits'], result['labels'],
                          positive=result.get('positive_records'))
    mean_score, mean_loss = metrics(torch, job['family'], job['recipe']['dataset'], result['mean'], result['labels'],
                                    positive=result.get('positive_records'))
    result.update(metrics=score, diagnostic_loss=loss, same_bank_mean_metrics=mean_score,
                  same_bank_mean_diagnostic_loss=mean_loss)
    result['member_metrics'] = [metrics(torch, job['family'], job['recipe']['dataset'], row, result['labels'],
                                         positive=result.get('positive_records'))[0] for row in result['members']]
    return result


def score_fit(request, go_pin, root, *, admit_runtime):
    root, task, profile, role_manifest, output = context(request, go_pin, root, admit_runtime)
    require(task['kind'] == 'selected_fit_TEST', 'Physical selected-fit scoring task required')
    spec = profile['fit_specs'][task['training_fit_id']]
    for field in ('descriptor', 'runtime_entrypoint', 'job_hash'):
        require(task[field] == spec[field], 'Scoring task changed exact training execution')
    job = verify(spec['descriptor'], root, json_metadata=True)
    require(digest(job) == spec['job_hash'], 'Selected fit descriptor changed')
    readout_policy(job['procedure'], job['family'])  # Reject unknown readouts before model/data work.
    role = selected_role_contract(role_manifest, spec['identity'])
    require(role['family'] == job['family'], 'Role/family differs')
    sealed = verify(request['training_seal'], root, json_metadata=True)['outcomes'][job['id']]
    training_release = verify(request['campaign_scoring_authority']['training_release'], root, json_metadata=True)
    training_root = Path(training_release['output_root'])
    expected_result = inside(training_root / spec['output_relative'] / 'RESULT.json', root)
    require(request.get('training_result') == sealed['training_result'], 'Use the exact sealed training outcome')
    raw = None if sealed['training_result'] is None else verify(sealed['training_result'], root, json_metadata=True)
    if raw is not None:
        require(request['training_result']['path'] == str(expected_result)
                and raw['job_id'] == job['id'] and raw['job_hash'] == spec['job_hash']
                and raw['TEST_accessed'] is False, 'Training RESULT authority differs')
    record = {'schema': 'final-selected-fit-scoring-result-v2', 'task_id': task['id'],
              'training_fit_id': job['id'], 'job_hash': spec['job_hash'], 'role': 'TEST',
              'identity': copy.deepcopy(spec['identity']), 'role_contract': task['role_contract'],
              'source_reference_or_matched': spec['identity']['track'],
              'training_RESULT': request['training_result'], 'request_sha256': digest(request),
              'campaign_GO': go_pin, 'physical_training_fits': 0, 'optimizer_updates': 0,
              'TEST_accessed': False, 'automatic_retries': 0, 'training_seal': request['training_seal'],
              'training_stop_reason': None if raw is None else raw.get('stop_reason'),
              'training_convergence_established': None if raw is None else raw.get('convergence_established'),
              'source_reproduction_caveats_preserved': True}
    if sealed['status'] != 'complete' or sealed['selected_state'] is None:
        record.update(status='missing_selected_state', training_status=sealed['status'],
                      reason='Retain declared training failure/no-winner; no fallback checkpoint')
        write_json(output / 'RESULT.json', record)
        return record
    started, adapter, runtime = time.perf_counter(), None, None
    try:
        # No import of torch/model/data occurs above this point. Execution is
        # reachable only through real campaign GO and existing runtime admission.
        import torch
        for pin in spec['observed_input_files']:
            verify(pin, root)
        if job['family'] == 'link':
            for key in ('graph', 'train', 'valid'):
                require(Path(role['data_files'][key]['path']).resolve() == Path(job['recipe']['data_files'][key]).resolve(),
                        'TEST graph ancestry differs from the selected training source')
                verify(role['data_files'][key], root)
        selected_pin, valid_pin = raw['selected_checkpoint'], raw['selected_VALID_export']
        require(sealed['selected_state'] == {'checkpoint': selected_pin, 'epoch': raw['selected_epoch'],
                                             'VALID_export': valid_pin, 'runtime_binding': raw['runtime_binding']},
                'Selected checkpoint changed after the all-campaign seal')
        require(Path(selected_pin['path']).name == 'selected.pt'
                and inside(selected_pin['path'], root).parent == expected_result.parent,
                'Only the exact own selected checkpoint can serve')
        selected = torch.load(verify(selected_pin, root), map_location='cpu', weights_only=False)
        valid = torch.load(verify(valid_pin, root), map_location='cpu', weights_only=False)
        epoch = selected['selector']['selected_epoch']
        require(selected['schema'] == 'native-full-epoch-recovery-v1'
                and selected['job_hash'] == spec['job_hash'] and selected['TEST_accessed'] is False
                and type(epoch) is int and epoch > 0 and epoch == selected['selector']['completed_epoch']
                and epoch == raw['selected_epoch'], 'Exact positive VALID-selected state required')
        require(valid['schema'] == 'own-selected-VALID-logits-v1' and valid['job_hash'] == spec['job_hash']
                and valid['selected_checkpoint'] == selected_pin and valid['selected_epoch'] == epoch
                and valid['role'] == 'VALID' and valid['TEST_accessed'] is False,
                'Recorded complete selected VALID export differs')
        runtime, adapter, binding = load_runtime(spec, job, root)
        require(binding == selected['runtime_binding'] == raw['runtime_binding'], 'Current source/runtime differs from selected fit')
        adapter.load_state(copy.deepcopy(selected['adapter']))
        if getattr(adapter, 'composed_producer', False):
            adapter.bind_loaded_selection(selected, selected_checkpoint=selected_pin,
                job_hash=spec['job_hash'], runtime_binding=binding)
        require(adapter.completed_epochs == epoch, 'Restored native/selector clocks differ')
        if job['family'] == 'link' and job['recipe']['architecture'].upper() == 'BUDDY' \
                and job['procedure'] == 'gnnm_private_bn_state_v3':
            inference = load_module(EXPANSION / 'buddy_private_bn_production_v3/session.py',
                                    '_final_scoring_BUDDY_owned_inference')
            expected = receipt(EXPANSION / 'buddy_private_bn_production_v3/MANIFEST.json', root)['sha256']
            record['inference_API_binding'] = inference.bind_inference_owner(adapter.session, job, root, expected)
        original = model_fingerprint(adapter)
        helper = load_module(EXPANSION / 'final_test_serving_v1/link/backend.py', '_final_scoring_link_source') \
            if job['family'] == 'link' else None
        with no_updates(adapter):
            runtime.restore_rng(torch, selected['global_rng'], device=job.get('device', 'cuda:0'))
            replay_started = time.perf_counter()
            replay = predict(torch, adapter, job, 'VALID', role, root, link_helper=helper, cache_root=output / 'native_cache')
            record['VALID_reconstruction_seconds'] = time.perf_counter() - replay_started
            require(torch.equal(replay['ids'], valid['ids'])
                    and replay['labels'].shape == valid['labels'].shape
                    and torch.allclose(replay['labels'].double(), valid['labels'].double(), rtol=0, atol=0, equal_nan=True),
                    'Selected VALID replay population/order/labels differ')
            require(torch.allclose(replay['logits'], valid['raw_logits'], rtol=role['rtol'], atol=role['atol']),
                    'Source-specific inference does not preserve the selected actual readout')
            if job['family'] == 'link':
                require(torch.equal(replay['query_endpoints'], valid['query_endpoints']), 'Selected VALID query order differs')
            if 'same_bank_mean_identity' in spec:
                saved_mean = valid.get('same_bank_mean_VALID')
                require(saved_mean is not None and saved_mean['selected_checkpoint'] == selected_pin
                        and saved_mean['selected_epoch'] == epoch and torch.allclose(replay['mean'], saved_mean['raw_logits'],
                                                                                   rtol=role['rtol'], atol=role['atol']),
                        'Declared same-bank mean failed selected VALID reconstruction')
            qualification = {'schema': 'selected-serving-VALID-reconstruction-v2', 'passed': True,
                             'selected_checkpoint': selected_pin, 'recorded_VALID_export': valid_pin,
                             'role': 'VALID', 'TEST_accessed': False, 'readout': replay['readout'],
                             'max_absolute_difference': float((replay['logits'] - valid['raw_logits']).abs().max()),
                             'rtol': role['rtol'], 'atol': role['atol'], 'work': replay['work']}
            write_json(output / 'VALID_QUALIFICATION.json', qualification)
            write_json(output / 'SELECTION_BINDING.json', {
                'schema': 'deterministic-own-VALID-selection-binding-v2',
                'training_RESULT': request['training_result'], 'descriptor': spec['descriptor'],
                'selected_checkpoint': selected_pin, 'selected_VALID_export': valid_pin,
                'selected_epoch': epoch, 'role_contract': task['role_contract'],
                'campaign_GO': go_pin, 'selection_rule': task['selection_rule'],
                'new_human_action_claimed': False, 'TEST_used_for_selection': False})
            require(model_fingerprint(adapter) == original, 'Inference changed selected parameters/buffers')
            # Return to the exact selected RNG, not the replay's advanced loader state.
            adapter.load_state(copy.deepcopy(selected['adapter']))
            if getattr(adapter, 'composed_producer', False):
                adapter.bind_loaded_selection(selected, selected_checkpoint=selected_pin,
                    job_hash=spec['job_hash'], runtime_binding=binding)
            runtime.restore_rng(torch, selected['global_rng'], device=job.get('device', 'cuda:0'))
            record['TEST_accessed'] = True  # Attempted role access is retained even on later failure.
            test_started = time.perf_counter()
            provider_authority(role, spec['identity'], root)
            served = predict(torch, adapter, job, 'TEST', role, root, link_helper=helper, cache_root=output / 'native_cache')
            record['TEST_serving_seconds'] = time.perf_counter() - test_started
            skipped = served['work'].get('native_single_node_skipped_graph_ids', [])
            require(len(served['ids']) + len(skipped) == role['population']['records']
                    and tensor_sha(served['ids']) == role['population']['ids_sha256'],
                    'Final serving did not retain the complete official/native-eligible population')
            require(model_fingerprint(adapter) == original, 'TEST inference changed selected parameters/buffers')
            attach_metrics(torch, served, job)
        payload = {'schema': 'own-selected-TEST-logits-v2', 'role': 'TEST', 'TEST_accessed': True,
                   'identity': copy.deepcopy(spec['identity']), 'job_id': job['id'], 'job_hash': spec['job_hash'],
                   'body': job.get('body'), 'procedure': job['procedure'], 'seed': job['seed'],
                   'selected_checkpoint': selected_pin, 'selected_epoch': epoch, 'runtime_binding': binding,
                   'role_contract': task['role_contract'], 'request_sha256': digest(request),
                   'raw_logits': served['logits'], 'member_logits': served['members'],
                   'ids': served['ids'], 'labels': served['labels'],
                   'query_endpoints': served.get('query_endpoints'), 'positive_records': served.get('positive_records'),
                   'metrics': served['metrics'], 'diagnostic_loss': served['diagnostic_loss'],
                   'member_metrics': served['member_metrics'],
                   'readout': served['readout'], 'work': served['work'], 'training_stop_reason': raw.get('stop_reason'),
                   'training_convergence_established': raw.get('convergence_established'),
                   'optimizer_updates': 0, 'physical_training_fits': 0}
        if 'same_bank_mean_identity' in spec:
            payload['same_selected_bank_mean'] = {'identity': spec['same_bank_mean_identity'],
                'raw_logits': served['mean'], 'metrics': served['same_bank_mean_metrics'],
                'diagnostic_loss': served['same_bank_mean_diagnostic_loss'],
                'selected_checkpoint': selected_pin, 'selected_epoch': epoch,
                'diagnostic_only': True, 'extra_forward_calls': 0, 'selection_changed': False}
        torch.save(payload, output / 'selected_TEST.pt')
        record.update(status='complete', selected_checkpoint=selected_pin, selected_epoch=epoch,
                      TEST_export=receipt(output / 'selected_TEST.pt', root), metrics=served['metrics'],
                      member_metrics=served['member_metrics'],
                      diagnostic_loss=served['diagnostic_loss'], official_records=role['population']['records'],
                      evaluated_records=len(served['ids']), VALID_qualification=receipt(output / 'VALID_QUALIFICATION.json', root),
                      readout=served['readout'], work=served['work'])
        if 'same_bank_mean_identity' in spec:
            record['same_selected_bank_mean'] = {'identity': spec['same_bank_mean_identity'],
                'metrics': served['same_bank_mean_metrics'],
                'diagnostic_loss': served['same_bank_mean_diagnostic_loss'],
                'selected_checkpoint': selected_pin, 'selected_epoch': epoch,
                'diagnostic_only': True, 'extra_forward_calls': 0, 'selection_changed': False}
    except BaseException as error:
        record.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                                                'traceback': traceback.format_exc(limit=10)})
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                record['close_error'] = str(error)
        if runtime is not None:
            record['resources'] = runtime.resource_record(sys.modules['torch'], device=job.get('device', 'cuda:0'))
        record['worker_wall_seconds'] = time.perf_counter() - started
        write_json(output / 'RESULT.json', record)
    return record


def aggregate_i4(request, go_pin, root, *, admit_runtime):
    root, task, profile, role_manifest, output = context(request, go_pin, root, admit_runtime)
    require(task['kind'] == 'I4_TEST_aggregate', 'I4 TEST aggregate task required')
    spec = profile['I4_specs'][task['training_I4_id']]
    sealed = verify(request['training_seal'], root, json_metadata=True)['outcomes'][task['training_I4_id']]
    require(spec['component_fit_ids'] == task['component_fit_ids'] and len(request['component_results']) == 4,
            'Four exact predetermined own native TEST results required')
    started = time.perf_counter()
    record = {'schema': 'final-I4-TEST-scoring-result-v2', 'task_id': task['id'], 'role': 'TEST',
              'identity': copy.deepcopy(spec['identity']), 'role_contract': task['role_contract'],
              'training_seal': request['training_seal'], 'campaign_GO': go_pin,
              'physical_training_fits': 0, 'optimizer_updates': 0, 'native_forwards': 0,
              'components': request['component_results'], 'automatic_retries': 0, 'TEST_accessed': False}
    try:
        if sealed['status'] != 'complete':
            record.update(status='missing_training_I4', training_status=sealed['status'])
            return record
        import torch
        rows, authorities = [], []
        for fit_id, result_pin in zip(task['component_fit_ids'], request['component_results']):
            physical = verify(result_pin, root, json_metadata=True)
            require(physical['training_fit_id'] == fit_id and physical['task_id'] == 'TEST/' + fit_id
                    and physical['campaign_GO'] == go_pin, 'Foreign I4 component or campaign')
            if physical['status'] != 'complete':
                record.update(status='missing_component', missing_fit_id=fit_id)
                return record
            record['TEST_accessed'] = True
            row = torch.load(verify(physical['TEST_export'], root), map_location='cpu', weights_only=False)
            require(row['schema'] == 'own-selected-TEST-logits-v2' and row['procedure'] == 'native_single'
                    and row['job_id'] == fit_id and row['role'] == 'TEST'
                    and row['identity'] == profile['fit_specs'][fit_id]['identity']
                    and row['job_hash'] == profile['fit_specs'][fit_id]['job_hash']
                    and row['selected_checkpoint'] == physical['selected_checkpoint'],
                    'I4 consumes its exact own native TEST exports only')
            rows.append(row); authorities.append(physical['TEST_export'])
        first = rows[0]
        require([row['body'] for row in rows] == [0, 1, 2, 3] and len({row['seed'] for row in rows}) == 4,
                'I4 body identities/seeds differ')
        require(first['identity']['scientific_method_id'] == 'native_M1', 'Matched M1 must be reused as exact body0')
        for row in rows:
            for name in ('dataset', 'preset_id', 'cohort_id', 'split_index', 'track'):
                require(row['identity'][name] == first['identity'][name], 'I4 native cohorts differ')
            require(row['role_contract'] == first['role_contract'] == task['role_contract']
                    and torch.equal(row['ids'], first['ids']) and torch.allclose(row['labels'].double(), first['labels'].double(),
                                                                               rtol=0, atol=0, equal_nan=True),
                    'I4 official populations/order/labels differ')
            require(row['raw_logits'].shape == first['raw_logits'].shape
                    and row['raw_logits'].dtype == first['raw_logits'].dtype, 'I4 native output shape/type differs')
            if first.get('query_endpoints') is not None:
                require(torch.equal(row['query_endpoints'], first['query_endpoints']), 'I4 query order/endpoints differ')
        logits = torch.stack([row['raw_logits'] for row in rows]).mean(0)
        family = 'link' if first['identity']['dataset'] == 'ogbl-collab' else \
            'ogb' if first['identity']['dataset'].startswith('ogbg-') else 'node'
        scores, loss = metrics(torch, family, first['identity']['dataset'], logits, first['labels'],
                               positive=first.get('positive_records'))
        payload = {'schema': 'I4-own-selected-TEST-bank-v2', 'role': 'TEST', 'TEST_accessed': True,
                   'identity': spec['identity'],
                   'raw_logits': logits, 'member_logits': torch.stack([row['raw_logits'] for row in rows]),
                   'ids': first['ids'], 'labels': first['labels'], 'query_endpoints': first.get('query_endpoints'),
                   'components': authorities, 'selected_epochs': [row['selected_epoch'] for row in rows],
                   'selected_checkpoints': [row['selected_checkpoint'] for row in rows],
                   'aggregation': 'mean_four_own_VALID_selected_native_raw_logits', 'metrics': scores,
                   'diagnostic_loss': loss, 'physical_training_fits': 0, 'native_forwards': 0}
        torch.save(payload, output / 'I4_TEST.pt')
        record.update(status='complete', TEST_export=receipt(output / 'I4_TEST.pt', root), metrics=scores,
                      diagnostic_loss=loss, selected_epochs=payload['selected_epochs'])
    except BaseException as error:
        record.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                                                'traceback': traceback.format_exc(limit=10)})
    finally:
        record['worker_wall_seconds'] = time.perf_counter() - started
        write_json(output / 'RESULT.json', record)
    return record
