"""Prepare-only source: exact 30 serialized checkpoint inference replay.

run is a future, separately released scientific process. close is stdlib-only
and consumes root's physical process receipt after its own child has been reaped.
No training, optimizer construction, initialization update, or label load occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
PHASE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
REPOSITORY = PHASE.parents[1]
INTERPRETER = REPOSITORY / '.venv/bin/python'
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
DRIVER = PHASE / 'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v3_precision/prototype/graph_init_driver.py'
DRIVER_SHA256 = '76873c66454fce606c3b54f609036706bc640676a300430be841e33fbb0a6e4f'
ARMS = ('graph', 'common_only', 'random_tangent', 'topology_permuted', 'warm_copy')
EXPECTED = {(g, s, a) for g in ('Photo', 'Squirrel') for s in (17, 29, 43) for a in ARMS}
LOGIT_ATOL, LOGIT_RTOL = 1e-6, 1e-5


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def verified(record):
    require(isinstance(record, dict) and {'path', 'sha256'} <= set(record) <= {'path', 'sha256', 'bytes'},
            'Exact path/hash descriptor with optional bytes required')
    path = Path(record['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'Canonical file path required')
    require(sha(path) == record['sha256'], 'Fingerprint mismatch: ' + str(path))
    if 'bytes' in record:
        require(type(record['bytes']) is int and path.stat().st_size == record['bytes'], 'Byte count changed')
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def payload_record(freeze, name):
    matches = [row for row in freeze['payload'] if row['path'] == name]
    require(len(matches) == 1, 'Missing/duplicate frozen payload: ' + name)
    return dict(matches[0], path=str(Path(freeze['output_root']) / name))


def manifest_guard(record):
    require(verified(record) == PACKAGE / 'MANIFEST.json', 'Wrong replay source manifest')
    manifest = read_json(PACKAGE / 'MANIFEST.json')
    require(manifest['schema'] == 'graph-init-checkpoint-replay-source-manifest-v1', 'Wrong source schema')
    names = []
    for row in manifest['payload']:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source payload')
        verified(dict(row, path=str(PACKAGE / relative)))
        names.append(row['path'])
    require(len(names) == len(set(names)) and 'replay_selected_checkpoints.py' in names and 'COHORT.json' in names,
            'Incomplete/duplicate source inventory')
    return manifest


def admission_guard(path):
    admission_record = descriptor(path)
    admission = read_json(path)
    require(admission['schema'] == 'graph-init-checkpoint-replay-root-admission-v1' and
            admission['authorized_checkpoint_inference'] is True and admission['root_release'] is True,
            'Separate exact root checkpoint inference release is required')
    manifest_guard(admission['source_manifest'])
    require(verified(admission['cohort']) == PACKAGE / 'COHORT.json', 'Wrong replay cohort')
    review = read_json(verified(admission['independent_source_review']))
    require(review['approved_for_checkpoint_replay_execution'] is True and
            review['source_manifest'] == admission['source_manifest'], 'Independent exact-source review required')
    require(admission['interpreter'] == str(INTERPRETER) and admission['repository'] == str(REPOSITORY)
            and admission['gpu_uuid'] == GPU_UUID and admission['device'] == 'cuda:0', 'Runtime route changed')
    require(admission['label_packs_may_be_opened'] is False and admission['training_or_reselection_allowed'] is False
            and admission['canonical_ledger_or_results_may_change'] is False, 'Replay scope changed')
    require(admission['logit_tolerance'] == dict(atol=LOGIT_ATOL, rtol=LOGIT_RTOL, reference='saved selected member logits'),
            'Original fixed full-tensor logit tolerance changed')
    for key in ('wall_cap_seconds', 'host_memory_cap_bytes', 'gpu_memory_cap_bytes'):
        require(type(admission[key]) is int and admission[key] > 0, 'Positive integer resource bound required')
    out = Path(admission['output'])
    require(out.is_absolute() and out.resolve() == out and out.is_relative_to(PHASE)
            and not out.is_relative_to(PACKAGE) and 'graph_init_serialized_checkpoint_replay_execution_root_' in out.parts[-2],
            'A dedicated fresh replay output beneath the phase is required')
    return admission, admission_record, out


def cohort_guard():
    cohort = read_json(PACKAGE / 'COHORT.json')
    audit = read_json(verified(cohort['source_valid_audit']))
    freeze = read_json(verified(cohort['source_valid_audit_freeze']))
    comparison = read_json(verified(cohort['comparison_freeze']))
    registry = read_json(verified(cohort['attempt_registry']))
    require(audit['completed'] is True and audit['source_cells'] == 30 and audit['registered_phases'] == 72
            and audit['serialized_checkpoint_inference_replayed'] is False, 'Wrong completed saved-array audit')
    require(freeze['completed'] is True and freeze['audit']['sha256'] == cohort['source_valid_audit']['sha256']
            and freeze['comparison_freeze']['sha256'] == cohort['comparison_freeze']['sha256'], 'Audit freeze link changed')
    require(audit['comparison_freeze'] == freeze['comparison_freeze'] == cohort['comparison_freeze'] and
            audit['attempt_registry'] == comparison['attempt_registry'] == cohort['attempt_registry'],
            'Audit/comparison/registry linkage changed')
    require(comparison['source_selection_closed'] is True and comparison['complete_cells'] == 30
            and comparison['final_labels_read'] is False and comparison['arms'] == list(ARMS)
            and comparison['configs'] == [0] and comparison['expected_contexts'] == registry['contexts'],
            'Closed source comparison changed')
    require(len(cohort['cases']) == 30 and len(cohort['contexts']) == 6 and
            {(r['graph'], r['seed'], r['arm']) for r in cohort['cases']} == EXPECTED,
            'Exactly all 30 original cases are required')
    contexts = {object_hash(context): context for context in registry['contexts']}
    require(len(contexts) == 6 and cohort['contexts'] == registry['contexts'], 'Context identities changed')
    rows = {(r['graph'], r['seed'], r['arm']): r for r in comparison['rows']}
    metrics = {(r['graph'], r['seed'], r['arm']): r for r in audit['metrics']}
    require(len(rows) == len(metrics) == 30 and set(rows) == set(metrics) == EXPECTED, 'Source cell coverage changed')
    for case in cohort['cases']:
        key = (case['graph'], case['seed'], case['arm'])
        row, metric = rows[key], metrics[key]
        require(case['context_sha256'] in contexts and row['context_sha256'] == case['context_sha256'], 'Cell context changed')
        for field in ('fit_freeze', 'selected_checkpoint', 'selected_logits'):
            require(case[field] == row[field] == metric[field], 'Source descriptor changed: ' + field)
    return cohort, contexts


def load_driver():
    require(sha(DRIVER) == DRIVER_SHA256, 'Pinned original driver changed')
    spec = importlib.util.spec_from_file_location('checkpoint_replay_original_driver', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_identity():
    # Linux /proc field22; process comm can contain spaces or parentheses.
    tail = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), parent_pid=os.getppid(), start_ticks=int(tail[19]))


def tensor_record(rt, value):
    require(rt.torch.is_tensor(value) and value.layout == rt.torch.strided, 'Dense tensor required')
    require(bool(rt.torch.isfinite(value).all()), 'Nonfinite checkpoint/state/logit tensor')
    array = value.detach().cpu().contiguous().numpy()
    raw = array.tobytes(order='C')
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def install_complete_state(rt, model, frozen):
    current = model.state_dict()
    require(set(current) == set(frozen), 'Complete strict state key inventory changed')
    for name, expected in current.items():
        actual = frozen[name]
        require(rt.torch.is_tensor(actual) and actual.dtype == expected.dtype and actual.shape == expected.shape,
                'State tensor shape/dtype differs: ' + name)
        tensor_record(rt, actual)
    # Every alias value is checked again after strict loading. Conflicting tied
    # values cannot silently overwrite each other through load_state_dict order.
    model.load_state_dict(frozen, strict=True)
    return verify_complete_state(rt, model, frozen)


def verify_complete_state(rt, model, frozen):
    current = model.state_dict()
    require(set(current) == set(frozen), 'State inventory changed')
    result = []
    for name in sorted(current):
        loaded, serialized = tensor_record(rt, current[name]), tensor_record(rt, frozen[name])
        require(loaded == serialized, 'Restored full tensor bytes differ: ' + name)
        result.append(dict(name=name, **serialized))
    return result


def optimizer_metadata_guard(rt, model, frozen):
    require(set(frozen) == {'schema', 'aliases', 'groups', 'state'} and frozen['schema'] == 'named-aliased-adam-v1'
            and frozen['aliases'] == rt.integration.alias_inventory(model), 'Serialized optimizer alias metadata changed')
    parameters = dict(model.named_parameters())
    names = [name for group in frozen['groups'] for name in group['names']]
    require(len(names) == len(set(names)) and set(names) == set(parameters) == set(frozen['state']),
            'Optimizer parameter coverage changed')
    for group in frozen['groups']:
        require(set(group) == {'names', 'options'}, 'Optimizer group metadata changed')
    for name, state in frozen['state'].items():
        require(set(state) <= {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'}, 'Unexpected Adam state')
        for key, value in state.items():
            if rt.torch.is_tensor(value):
                tensor_record(rt, value)
                require(value.numel() == 1 if key == 'step' else
                        value.shape == parameters[name].shape and value.dtype == parameters[name].dtype,
                        'Optimizer metadata tensor shape/dtype differs')
            else:
                require(key == 'step' and type(value) in (int, float) and math.isfinite(value), 'Unexpected optimizer scalar')
    return dict(alias_groups=len(frozen['aliases']), parameter_coverage=len(names),
                optimizer_constructed=False, optimizer_restored=False, optimizer_updates=0)


def case_metadata(case, context):
    fit = read_json(verified(case['fit_freeze']))
    selection = read_json(verified(case['selection_receipt']))
    warm = read_json(verified(case['warm_freeze']))
    warm_receipt = read_json(verified(case['warm_receipt']))
    require(fit['schema'] == warm['schema'] == 'graph-init-phase-freeze-v1' and
            fit['phase'] == 'fit' and warm['phase'] == 'warm' and fit['completed'] is True and warm['completed'] is True
            and fit['context'] == warm['context'] == context and fit['context_sha256'] == case['context_sha256']
            and fit['arm'] == case['arm'] and fit['warm_freeze'] == case['warm_freeze'], 'Fit/warm custody changed')
    require(case['selection_receipt'] == payload_record(fit, 'SELECTION.json') and
            case['warm_receipt'] == payload_record(warm, 'WARM_RECEIPT.json'), 'Metadata payload custody changed')
    require(fit['selected_checkpoint'] == selection['selected_checkpoint'] == case['selected_checkpoint'] and
            fit['selected_logits'] == case['selected_logits'] and
            warm['warm_checkpoint'] == selection['warm_checkpoint'] == warm_receipt['warm_checkpoint'] == case['warm_checkpoint'],
            'Checkpoint/logit/warm binding changed')
    require(selection['selection'] == fit['selection'] and selection['preprocessing'] == fit['preprocessing'] ==
            warm['preprocessing'] == warm_receipt['preprocessing'] and
            selection['initialized_from'] == fit['initialization_freeze'], 'Selection/preprocessing construction differs')
    require(fit['selection']['configuration'] == 0 and fit['selection']['members'] == 4 and
            fit['selection']['global_only'] is (context['backbone'] == 'polynormer_r'), 'Native selected branch changed')
    return fit, selection, warm_receipt


def replay_case(driver, rt, context, case, out):
    start = time.monotonic()
    fit, selection, warm_receipt = case_metadata(case, context)
    graph_manifest = read_json(verified(context['graph_input']))
    role = read_json(verified(context['role_freeze']))
    certificate = read_json(verified(context['modern_qualification_certificate']))
    require(role['graph_input'] == context['graph_input'] and role['labels_read'] is False and
            certificate['admission_eligible'] is True and certificate['backbone'] == context['backbone'] and
            certificate['qualified_config'] == 0 and certificate['environment'] == rt.environment and
            certificate['preprocessing'] == fit['preprocessing'], 'Qualified graph/preprocessing identity changed')
    x_np, edges_np = driver.canonical_graph(rt, graph_manifest)
    native_shape = rt.adapter.NATIVE[context['backbone']]
    require((graph_manifest['num_nodes'], graph_manifest['num_features'], graph_manifest['num_classes']) ==
            (native_shape['nodes'], native_shape['features'], native_shape['classes']), 'Full native graph dimensions changed')
    x = rt.torch.from_numpy(x_np.copy()).to(rt.device)
    edges = rt.torch.from_numpy(edges_np.copy()).to(rt.device)
    graph = rt.adapter.prepare_graph(x, edges, context['backbone'], rt.environment,
                                     certificate['preprocessing']['input_binding'])
    require(graph.preprocessing == certificate['preprocessing'] == fit['preprocessing'], 'Actual preprocessing identity changed')
    warm = rt.torch.load(verified(case['warm_checkpoint']), map_location=rt.device, weights_only=True)
    require(set(warm) == {'schema', 'model', 'optimizer', 'rng', 'specification', 'global_stage', 'model_training', 'metadata'}
            and warm['schema'] == 'graph-init-native-warm-checkpoint-v1', 'Warm checkpoint field/schema changed')
    expected_specification = rt.adapter.specification(context['backbone'], 'single_author', 0, context['seed'])
    require(warm['specification'] == warm_receipt['specification'] == expected_specification and
            warm['metadata'] == warm_receipt['metadata'] and
            warm['global_stage'] is (context['backbone'] == 'polynormer_r') and type(warm['model_training']) is bool,
            'Matched native warm construction metadata changed')
    # Same native TeacherFamily constructor and warm native state as restore_native,
    # without constructing an optimizer or restoring training RNG for eval/no_grad.
    native = rt.adapter.TeacherFamily(warm['specification']).to(rt.device)
    install_complete_state(rt, native, warm['model'])
    native.set_global_stage(warm['global_stage'])
    native.train(warm['model_training'])
    warm_optimizer = optimizer_metadata_guard(rt, native, warm['optimizer'])
    raw = rt.integration.clone_boundary(native, rt.boundary, 4)
    selected = rt.torch.load(verified(case['selected_checkpoint']), map_location=rt.device, weights_only=True)
    require(set(selected) == {'schema', 'arm', 'context_sha256', 'global_stage', 'state', 'optimizer',
                             'validation_nll', 'continuation_epoch', 'actual_update'} and
            selected['schema'] == 'graph-init-selected-checkpoint-v1' and selected['arm'] == case['arm'] and
            selected['context_sha256'] == case['context_sha256'] and
            selected['global_stage'] is (context['backbone'] == 'polynormer_r'), 'Selected checkpoint identity/schema changed')
    chosen = fit['selection']
    require(type(selected['continuation_epoch']) is int and type(selected['actual_update']) is int and
            selected['continuation_epoch'] == chosen['selected_continuation_epoch'] and
            selected['actual_update'] == chosen['selected_actual_update'] and
            type(selected['validation_nll']) in (float, int) and math.isfinite(selected['validation_nll']) and
            selected['validation_nll'] == chosen['primary_validation_nll'], 'Serialized selection metadata changed')
    complete_state = install_complete_state(rt, raw, selected['state'])
    if context['backbone'] == 'polynormer_r':
        raw.set_global_stage(True)
        require(raw.core._global is True, 'Photo final global branch not enabled')
    selected_optimizer = optimizer_metadata_guard(rt, raw, selected['optimizer'])
    require(rt.integration.storage_receipt(raw) == selection['storage'], 'Native parameter ownership/storage changed')
    boundaries = tuple(rt.integration.BOUNDARIES[context['backbone']].values())
    factor_names = {boundary + '.' + factor for boundary in boundaries for factor in ('R', 'S', 'B')}
    require(factor_names <= set(selected['state']), 'Every private boundary factor must be restored')
    factors = [row for row in complete_state if row['name'] in factor_names]
    require(len(factors) == 3 * len(boundaries) and all(row['shape'][0] == 4 for row in factors), 'K4 factor state changed')
    del native, warm
    raw.eval()
    require(all(not module.training for module in raw.modules()), 'Every native module must be in eval mode')
    with rt.torch.no_grad():
        inferred = raw(*rt.integration.raw_arguments(graph, context['backbone']))
    driver.sync(rt)
    require(inferred.dtype == rt.torch.float32 and tuple(inferred.shape) ==
            (4, native_shape['nodes'], native_shape['classes']), 'Complete inference logit contract changed')
    inferred_record = tensor_record(rt, inferred)
    expected = rt.np.load(verified(case['selected_logits']), allow_pickle=False)
    require(expected.dtype == rt.np.float32 and expected.shape == tuple(inferred.shape) and rt.np.isfinite(expected).all(),
            'Complete saved logit contract changed')
    reference = rt.torch.from_numpy(expected.copy())
    inferred_cpu = inferred.detach().cpu()
    difference = rt.integration.difference(inferred_cpu, reference, LOGIT_ATOL, LOGIT_RTOL)
    require(difference['finite'] is True and all(math.isfinite(difference[key]) for key in ('max_absolute', 'max_relative')),
            'Nonfinite full-tensor comparison diagnostic')
    output = out / (case['graph'] + '_seed' + str(case['seed']) + '__' + case['arm'] + '__inferred_member_logits.npy')
    with output.open('xb') as handle:
        rt.np.save(handle, inferred_cpu.numpy(), allow_pickle=False)
    require(verify_complete_state(rt, raw, selected['state']) == complete_state, 'Inference changed selected tensors')
    result = dict(graph=case['graph'], seed=case['seed'], arm=case['arm'], context_sha256=case['context_sha256'],
        passed=difference['passed'], checkpoint=case['selected_checkpoint'], saved_logits=case['selected_logits'],
        fit_freeze=case['fit_freeze'], warm_checkpoint=case['warm_checkpoint'], warm_freeze=case['warm_freeze'],
        selection_receipt=case['selection_receipt'], warm_receipt=case['warm_receipt'],
        inferred_logits=descriptor(output), inferred_tensor=inferred_record,
        tensor_payload_byte_equal=inferred_record == tensor_record(rt, reference),
        npy_file_byte_equal=sha(output) == case['selected_logits']['sha256'], complete_tensor_comparison=difference,
        restored_complete_state_sha256=object_hash(complete_state), restored_state_tensor_count=len(complete_state),
        complete_state_restored_byte_exact=True, complete_state_unchanged_after_inference=True,
        factor_tensors=factors, storage=selection['storage'], warm_optimizer_metadata=warm_optimizer,
        selected_optimizer_metadata=selected_optimizer, preprocessing=graph.preprocessing,
        eval_all_modules=True, no_grad=True, global_stage=selected['global_stage'], label_packs_opened=False,
        seconds=time.monotonic()-start)
    del raw, selected, graph, x, edges, x_np, edges_np, inferred, inferred_cpu, expected, reference
    gc.collect()
    rt.torch.cuda.empty_cache()
    return result


def run(path):
    admission, admission_record, out = admission_guard(path)
    require(str(sys.executable) == str(INTERPRETER) and PACKAGE.parent == PHASE,
            'Use the exact repository interpreter and original remote phase')
    cohort, contexts = cohort_guard()
    require(not out.exists(), 'Fresh replay output required; retries need a new separately released output')
    out.mkdir(parents=True, exist_ok=False)
    started = dict(schema='graph-init-checkpoint-replay-start-v1', UTC=utc(), admission=admission_record,
                   source_manifest=admission['source_manifest'], cohort=admission['cohort'], output=str(out),
                   child_identity=process_identity(), command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    write_json(out / 'STARTED.json', started)
    start, rt, results, error = time.monotonic(), None, [], None
    try:
        driver = load_driver()
        for case in cohort['cases']:
            context = contexts[case['context_sha256']]
            source_directory = driver.source_guard(context)
            if rt is None:
                rt = driver.load_runtime(context, source_directory)
                require(rt.torch.cuda.device_count() == 1, 'The retained single GPU inventory changed')
                require(rt.integration.TOLERANCES['logits_atol'] == LOGIT_ATOL and
                        rt.integration.TOLERANCES['logits_rtol'] == LOGIT_RTOL, 'Original logit tolerance changed')
                rt.torch.cuda.reset_peak_memory_stats(rt.device)
            else:
                _, _, environment = driver.runtime(context['device'])
                require(environment == context['environment'] == rt.environment, 'Frozen per-context runtime changed')
            results.append(replay_case(driver, rt, context, case, out))
            with (out / 'CASE_RESULTS.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(results[-1], allow_nan=False) + '\n')
    except Exception as exception:
        error = dict(type=type(exception).__name__, message=str(exception), traceback=traceback.format_exc(),
                     next_case=cohort['cases'][len(results)] if len(results) < 30 else None)
    peak = dict(allocated_bytes=rt.torch.cuda.max_memory_allocated(rt.device),
                reserved_bytes=rt.torch.cuda.max_memory_reserved(rt.device)) if rt is not None else None
    passed = len(results) == 30 and error is None and all(row['passed'] for row in results)
    result = dict(schema='graph-init-checkpoint-replay-inference-result-v1', UTC=utc(), inference_completed=passed,
        physical_closure=False, accepted_checkpoint_verification=False, started=descriptor(out / 'STARTED.json'),
        admission=admission_record, source_manifest=admission['source_manifest'], cohort=admission['cohort'],
        child_identity=started['child_identity'], command=started['command'], output=str(out),
        requested_cells=30, replayed_cells=len(results), cases=results, error=error,
        runtime=rt.environment if rt is not None else None, gpu_allocator_peak=peak,
        process_peak_resident_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        process_peak_resident_scope='Linux process lifetime; final root own-wait4 usage is authoritative',
        internal_seconds=time.monotonic()-start, label_packs_opened=False, optimizer_updates=0,
        scientific_recipe_or_canonical_results_changed=False, automatic_retry=False,
        source_VALID_NLL_recomputed=False, prior_saved_array_source_VALID_audit=cohort['source_valid_audit'])
    write_json(out / 'INFERENCE_RESULT.json', result)
    return 0 if passed else 1


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def route_guard(route):
    require(route['repository'] == str(REPOSITORY) and route['gpu_uuid'] == GPU_UUID and
            route['ssh_destination'] == 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru' and
            route['ssh_port'] == 2222, 'Physical route changed')
    require(verified(route['stdout']).read_text().strip().splitlines() == [GPU_UUID],
            'Actual GPU UUID probe must show exactly the retained sole GPU')
    require(verified(route['stderr']).read_text() == '', 'Physical GPU UUID probe failed')


def close(path, physical_path, physical_sha256):
    admission, admission_record, out = admission_guard(path)
    cohort_guard()
    physical_record = dict(path=str(Path(physical_path).resolve()), sha256=physical_sha256)
    physical = read_json(verified(physical_record))
    result_record = descriptor(out / 'INFERENCE_RESULT.json')
    result = read_json(verified(result_record))
    started = read_json(verified(result['started']))
    require(physical['schema'] == 'graph-init-checkpoint-replay-physical-closure-v1' and
            physical['source_manifest'] == result['source_manifest'] == admission['source_manifest'] and
            physical['admission'] == result['admission'] == admission_record and
            physical['result'] == result_record and physical['started'] == result['started'] and
            physical['child_identity'] == result['child_identity'] == started['child_identity'] and
            physical['command'] == result['command'] == started['command'] and physical['output'] == str(out),
            'Physical custody must bind this exact inference process and output')
    require(physical['own_wait4_reaped'] is True and physical['child_exit_code'] == 0 and
            physical['timed_out'] is False and physical['automatic_retry'] is False, 'Successful physical child reap required')
    for key in ('dispatch_UTC', 'reap_UTC'):
        stamp = datetime.fromisoformat(physical[key])
        require(stamp.utcoffset() is not None, 'Physical timestamps must include timezone')
    require(datetime.fromisoformat(physical['dispatch_UTC']) <= datetime.fromisoformat(started['UTC']) <=
            datetime.fromisoformat(result['UTC']) <= datetime.fromisoformat(physical['reap_UTC']), 'Physical time order changed')
    require(finite_nonnegative(physical['whole_process_seconds']) and
            result['internal_seconds'] <= physical['whole_process_seconds'] <= admission['wall_cap_seconds'],
            'Whole physical process time must fit the released cap')
    usage = physical['child_usage']
    require(all(finite_nonnegative(usage[k]) for k in ('user_seconds', 'system_seconds', 'peak_resident_bytes')) and
            result['process_peak_resident_bytes'] <= usage['peak_resident_bytes'] <= admission['host_memory_cap_bytes'],
            'Final root child usage exceeds released memory bound or contradicts child usage')
    require(result['inference_completed'] is True and result['replayed_cells'] == 30 and result['error'] is None and
            len(result['cases']) == 30 and result['label_packs_opened'] is False and result['optimizer_updates'] == 0 and
            {(r['graph'], r['seed'], r['arm']) for r in result['cases']} == EXPECTED and
            result['gpu_allocator_peak']['reserved_bytes'] <= admission['gpu_memory_cap_bytes'], 'Incomplete/out-of-bound replay')
    route_guard(physical['route_before']); route_guard(physical['route_after'])
    verified(physical['stdout']); verified(physical['stderr'])
    dependencies = read_json(PACKAGE / 'DEPENDENCIES.json')
    for key in ('unchanged_source', 'required_raw_graph_payloads', 'metadata_text_bindings'):
        for record in dependencies[key]:
            verified(record)
    for row in result['cases']:
        require(row['passed'] is True and row['complete_state_restored_byte_exact'] is True and
                row['complete_state_unchanged_after_inference'] is True and row['complete_tensor_comparison']['passed'] is True,
                'A replay case failed')
        for key in ('checkpoint', 'saved_logits', 'fit_freeze', 'warm_checkpoint', 'warm_freeze',
                    'selection_receipt', 'warm_receipt', 'inferred_logits'):
            verified(row[key])
    receipt = dict(schema='graph-init-serialized-checkpoint-replay-receipt-v1', UTC=utc(), completed=True,
        accepted_checkpoint_verification=True, physical_closure=True, cells=30,
        inference_result=result_record, physical_closure_receipt=physical_record,
        source_manifest=admission['source_manifest'], cohort=admission['cohort'], admission=admission_record,
        independent_source_review=admission['independent_source_review'],
        full_logits_compared=True, original_logit_tolerance=admission['logit_tolerance'],
        tensor_byte_equal_cells=sum(r['tensor_payload_byte_equal'] for r in result['cases']),
        npy_file_byte_equal_cells=sum(r['npy_file_byte_equal'] for r in result['cases']),
        whole_process_seconds=physical['whole_process_seconds'], child_usage=usage,
        gpu_allocator_peak=result['gpu_allocator_peak'], prior_source_VALID_audit=result['prior_saved_array_source_VALID_audit'],
        label_packs_opened=False, optimizer_updates=0, source_VALID_NLL_recomputed=False,
        canonical_ledger_or_original_results_changed=False, manuscript_acceptance=False)
    write_json(out / 'SERIALIZED_CHECKPOINT_REPLAY_RECEIPT.json', receipt)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    for name in ('run', 'close'):
        command = commands.add_parser(name)
        command.add_argument('--admission', type=Path, required=True)
        if name == 'close':
            command.add_argument('--physical-receipt', type=Path, required=True)
            command.add_argument('--physical-sha256', required=True)
    args = parser.parse_args()
    if args.operation == 'run':
        return run(args.admission)
    return close(args.admission, args.physical_receipt, args.physical_sha256)


if __name__ == '__main__':
    sys.exit(main())
