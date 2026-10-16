"""One new-producer MolBACE epoch plus stopped finalization through genuine reporting."""
import argparse
from contextlib import contextmanager
import copy
import importlib
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback
from witness_support import E, ROOT, HERE, GPU, digest, guard, inside, limits, module, own_source, read, receipt, require, verify, write
from runtime_guards import projection_host, guard_training_transition, native_update_call


def prepare_profile(inputs, output, python):
    """Use the successor's actual scalar profile projection before importing Torch."""
    pins = inputs['molbace']
    old_fit = read(verify(pins['completed_fit']))
    job_path = inside(pins['job_path'])
    job = read(job_path)
    require(digest(job) == pins['job_sha256'] == old_fit['job_hash'], 'Exact original engineering descriptor required')
    require(job['engineering_only'] is True and job['family'] == 'ogb' and job['seed'] == 1
            and job['recipe_id'] == 'ogb-ogbg-molbace-gin-drop0.0'
            and job['procedure'] == 'interaction_input_conditionalJ' and job['TEST_access'] is False
            and job['device'] == 'cuda:0' and job['policy']['maximum_epochs'] == job['policy']['minimum_epochs'] == 1,
            'One native MolBACE epoch with declared engineering settings required')
    require(job['normalization_composition']['parent_descriptor'] == pins['engineering_parent'], 'Original parent changed')
    verify(pins['engineering_parent'])
    successor = E/'server51_campaign_port_v1'
    producer = successor/'composed'
    actual_producer = receipt(producer/'MANIFEST.json')
    require(old_fit['runtime_binding']['schema'] == 'composed-normalization-genuine-runtime-binding-v1'
            and actual_producer != old_fit['runtime_binding']['composition_producer']['source_manifest'],
            'This finite new-producer witness must not claim old-checkpoint portability')
    data = old_fit['source_metadata']['native_session_metadata']['interaction']['source_data']
    observed = [receipt(job_path), pins['completed_fit'], pins['engineering_parent'],
                *old_fit['source_metadata']['source_dependency_receipts'],
                old_fit['source_metadata']['callback_receipt'], old_fit['source_metadata']['extension_receipt']]
    for item in [*data['scaffold_roles'].values(), *data['encoding_dependencies'].values()]:
        actual = receipt(inside(item['path']))
        require(actual['sha256'] == item['sha256'], 'Original VALID/data or encoder bytes changed')
        observed.append(actual)
    observed = list({p['path']: p for p in observed}.values())
    for pin in observed:
        verify(pin)
    bindings = {'scope': 'engineering_only_new_producer_one_epoch_reporting_witness',
        'native_source': digest(old_fit['runtime_binding']['composition_producer']['unchanged_helper_runtime']),
        'data': digest(data), 'controller': digest(job['policy']),
        'evaluator': digest(old_fit['source_metadata']['callback_receipt']),
        'wrapper': digest(job['normalization_composition'])}
    parent_spec = {'descriptor': receipt(job_path), 'job_hash': digest(job),
        'runtime_entrypoint': old_fit['runtime_binding']['composition_producer']['entrypoint'],
        'observed_input_files': observed, 'bindings': bindings,
        'identity': {'scope': 'engineering_only', 'fit_id': job['id'], 'bindings': copy.deepcopy(bindings)},
        'wrapper_authority': {'descriptor': receipt(job_path), 'normalization_composition': copy.deepcopy(job['normalization_composition'])}}
    parent_profile = {'fit_specs': {job['id']: parent_spec}, 'I4_specs': {},
        'registry': {'engineering_only': True}, 'contrast_plan': {}, 'campaign_profile_or_fit_claim': False}
    parent_path = output/'PARENT_ENGINEERING_PROFILE.json'
    write(parent_path, parent_profile)
    execution = {'python_executable': python, 'runtime_entrypoint': receipt(producer/'runtime/run_fit.py'),
                 'job_descriptor': receipt(job_path)}
    task = {'id': job['id'], 'kind': 'fit', 'worker': 'allocation/gpu0', 'cell_id': 'engineering_molbace',
        'binding': {'execution': execution}, 'runtime_job_hash': digest(job),
        'runtime_binding': {'schema': 'engineering-unchanged-allocation-job-new-producer-v1',
                            'producer_manifest': actual_producer, 'campaign_launch_admitted': False}}
    candidate = {'_job': job, 'job_descriptor': receipt(job_path), 'runtime_entrypoint': execution['runtime_entrypoint']}
    require('descriptor_binding' not in sys.modules, 'Fresh scalar projection import required')
    sys.path.insert(0, str(successor))
    projection = module(successor/'reporting_projection.py', '_remaining_actual_profile_projection')
    def source_pin(path, expected=None):
        actual = receipt(inside(path))
        if expected is not None:
            require(actual == expected, 'Allocation profile input changed')
        return actual
    try:
        projected = projection.project_profile(parent_profile, receipt(parent_path), {job['id']: task},
            {job['id']: candidate}, projection_host(read(successor/'hosts/allocation_gpu0.json')), source_pin)
    finally:
        # The real producer requires fresh generic modules. Remove only this scalar helper.
        loaded = sys.modules.get('descriptor_binding')
        require(loaded is not None and Path(loaded.__file__).resolve() == successor/'descriptor_binding.py',
                'Unexpected projection module collision')
        del sys.modules['descriptor_binding']
        sys.path.remove(str(successor))
    spec = projected['fit_specs'][job['id']]
    require(spec['bindings'] == bindings and spec['runtime_entrypoint'] == execution['runtime_entrypoint']
            and spec['composed_runtime_authority']['producer_manifest'] == actual_producer,
            'Actual successor profile/producer required')
    profile_path = output/'engineering_molbace.json'
    write(profile_path, projected)
    index_path = output/'ENGINEERING_INDEX.json'
    write(index_path, {'preset_files': [receipt(profile_path)], 'engineering_only': True})
    sys.path.insert(0, str(successor/'reporting'))
    authority = module(successor/'reporting/composed_authority.py', 'composed_authority')
    authority.verify_sources(spec['composed_runtime_authority'])
    bridge = module(successor/'reporting/backend_bridge.py', '_remaining_actual_argv_bridge')
    transform = bridge.ObservedArgvTransform(index_path, receipt(index_path)['sha256'], ROOT)
    target = output/'fits'/digest(job['id'])
    target.parent.mkdir()
    write(output/'PRODUCER_DECISION.json', {'old_checkpoint_reused': False, 'new_TRAIN_updates_admitted': 38,
        'reason': 'New producer schema and manifest must exist in its own genuine checkpoint; old bytes cannot be relabeled',
        'old_producer': old_fit['runtime_binding']['composition_producer']['source_manifest'],
        'new_producer': actual_producer, 'original_job_unchanged': True, 'campaign_profile_or_fit_claim': False})
    return job, old_fit, spec, target, task, transform


def execute(admission_path):
    guard()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'Explicit sole physical CUDA mask required')
    admission = read(inside(admission_path))
    require(admission.get('schema') == 'remaining-runtime-native-child-admission-v1'
            and admission.get('status') == 'ADMITTED' and admission['case'] == 'bound_observer_molbace'
            and admission['TRAIN_updates'] == 38 and admission['TEST_calls'] == admission['automatic_retries'] == 0
            and admission['python_executable'] == sys.executable and admission['exclusive_shared_GPU_slot_held'] is True,
            'Explicit one-epoch/38-update admission required')
    inputs = own_source(admission['source_manifest_sha256'])
    cap = limits(admission['case'])
    require(admission['resources'] == cap, 'Exact resource reservations required')
    verify(admission['peer_resource_review'])
    output = inside(admission['output'])
    require(output.is_relative_to(E/'qualification') and not output.exists(), 'Fresh engineering output required')
    output.mkdir(parents=True)
    started = time.monotonic()
    work = {'constructors_started': 0, 'constructors_completed': 0, 'state_restores': 0,
        'whole_VALID_calls': 0, 'native_forwards_attempted': 0, 'native_forwards_completed': 0,
        'mean_exports': 0, 'TRAIN_epochs_started': 0, 'TRAIN_epochs_completed': 0,
        'TRAIN_update_attempts': 0, 'TRAIN_updates': 0, 'TRAIN_callback_calls': 0,
        'TRAIN_exhaustion_probe_attempts': 0, 'TRAIN_exhaustion_probes_completed': 0,
        'optimizer_attempts': 0, 'optimizer_completed': 0,
        'forbidden_calls': 0}
    result = {'schema': 'successor-bound-composed-observer-runtime-witness-v1', 'status': 'running', 'actual_work': work,
        'admission': receipt(admission_path), 'TEST_accessed': False, 'TRAIN_updates': 0, 'automatic_retries': 0,
        'full_campaign_GO': False, 'scientific_accuracy_claim': False, 'campaign_profile_or_fit_claim': False,
        'old_checkpoint_reused': False, 'retained_old_paid_updates': 38, 'invocations': []}
    handles, originals = [], []
    torch = None
    training_open = False
    stage = None
    def save():
        result['wall_seconds'] = time.monotonic()-started
        result['TRAIN_updates'] = work['TRAIN_updates']
        write(output/'RESULT.json', result)
    def forbid(message):
        work['forbidden_calls'] += 1
        save()
        raise ValueError(message)
    def timeout(*_):
        raise TimeoutError('Bound observer wall limit reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.alarm(cap['wall_seconds'])
    save()
    try:
        job, old_fit, spec, target, task, transform = prepare_profile(inputs, output, sys.executable)
        require('torch' not in sys.modules, 'Authenticate genuine successor/profile before tensor import')
        successor = E/'server51_campaign_port_v1'
        import torch as tensor_library
        torch = tensor_library
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(cap['GPU_reservation_bytes']/torch.cuda.get_device_properties(0).total_memory, 0)
        for obj, name in [(torch.Tensor, 'backward'), (torch.autograd, 'backward'), (torch.autograd, 'grad')]:
            original = getattr(obj, name)
            originals.append((obj, name, original))
            def reverse(*a, function=original, **kw):
                guard_training_transition(work, training_open, forbid)
                return function(*a, **kw)
            setattr(obj, name, reverse)
        factory = None
        def construct(*a, **kw):
            require(work['constructors_started'] < 2, 'Only fit and stopped-finalizer constructors admitted')
            work['constructors_started'] += 1
            save()
            owner = factory(*a, **kw)
            work['constructors_completed'] += 1
            native = owner.manager._native_ref()
            native_helper = sys.modules[type(owner.normalized.inner).__module__]
            require(Path(native_helper.__file__).resolve() == E/'portable_dispatcher_v3/runtime_node_molecular/adapters.py',
                    'The original molecular adapter must own the epoch-completion exception')
            epoch_complete = native_helper.EpochComplete
            def before(*_):
                if training_open:
                    guard_training_transition(work, training_open, forbid)
                require(work['native_forwards_attempted'] < 212, 'Finite native route forward budget exhausted')
                work['native_forwards_attempted'] += 1
                save()
            def after(*_):
                work['native_forwards_completed'] += 1
            handles.extend([native.register_forward_pre_hook(before), native.register_forward_hook(after)])
            original_load, original_valid, original_mean = owner.load_state, owner.validate, owner.selected_mean_export
            original_train, original_update = owner.train_epoch, owner.session.train_update
            def train():
                nonlocal training_open
                if stage != 'one_complete_epoch' or work['TRAIN_epochs_started']:
                    forbid('Only one complete native TRAIN epoch admitted')
                work['TRAIN_epochs_started'] += 1
                training_open = True
                save()
                try:
                    value = original_train()
                finally:
                    training_open = False
                require(value['complete_native_TRAIN'] is True and work['TRAIN_updates'] == 38,
                        'Complete native MolBACE epoch must honestly charge all 38 updates')
                work['TRAIN_epochs_completed'] += 1
                save()
                return value
            def update(*a, **kw):
                return native_update_call(original_update, work, training_open, epoch_complete, save, forbid, *a, **kw)
            for optimizer in owner.optimizers:
                original_step = optimizer.step
                def step(*a, function=original_step, **kw):
                    guard_training_transition(work, training_open, forbid)
                    if work['optimizer_attempts'] >= 38:
                        forbid('No extra optimizer transition admitted')
                    work['optimizer_attempts'] += 1
                    save()
                    value = function(*a, **kw)
                    work['optimizer_completed'] += 1
                    save()
                    return value
                optimizer.step = step
            def restore(value):
                require(work['state_restores'] < 3, 'Only own selected/recovery restores admitted')
                value = original_load(value)
                work['state_restores'] += 1
                save()
                return value
            def valid():
                require(work['whole_VALID_calls'] < 3 and not training_open, 'Only epoch and selected VALID calls admitted')
                before_state = owner.state_signature()
                optimizer_before = owner.normalized._state_signature([o.state_dict() for o in owner.optimizers])
                work['whole_VALID_calls'] += 1
                save()
                value = original_valid()
                require(owner.state_signature() == before_state and optimizer_before == owner.normalized._state_signature([o.state_dict() for o in owner.optimizers]),
                        'VALID modified selected model, optimizer or private BN state')
                require(value['role'] == 'VALID' and value['complete_source_VALID'] is True
                        and value['TEST_accessed'] is False and len(value['ids']) == 151, 'Whole native VALID required')
                require(value['ids'].detach().cpu().tolist() == old_fit['source_metadata']['native_session_metadata']['interaction']['source_data']['VALID_ids'],
                        'Original native VALID graph order required')
                return value
            def mean(actual, **context):
                before_count = work['native_forwards_attempted']
                value = original_mean(actual, **context)
                require(work['mean_exports'] < 2 and value is not None and value['extra_forward_calls'] == 0
                        and work['native_forwards_attempted'] == before_count, 'Same-call mean must add zero forwards')
                work['mean_exports'] += 1
                save()
                return value
            owner.load_state, owner.validate, owner.selected_mean_export = restore, valid, mean
            owner.train_epoch, owner.session.train_update = train, update
            owner.train_diagnostic = lambda: forbid('No TRAIN diagnostic admitted')
            return owner
        observer_module = module(successor/'reporting/observe_exports.py', 'observe_exports')
        original_observe = observer_module.observe
        @contextmanager
        def guarded_observe(runtime, observed_job, observed_output, binding_spec=None):
            nonlocal factory
            # The real entry has now installed/imported its runtime. No generic
            # host_binding/common/adapters module is preloaded ahead of it.
            if factory is None:
                adapters = importlib.import_module('adapters')
                require(Path(adapters.__file__).resolve() == successor/'composed/runtime/adapters.py',
                        'Actual entry must resolve its own genuine producer factory')
                factory = adapters.create_adapter
                adapters.create_adapter = construct
                originals.append((adapters, 'create_adapter', factory))
            with original_observe(runtime, observed_job, observed_output, binding_spec=binding_spec) as observer:
                yield observer
        observer_module.observe = guarded_observe
        originals.append((observer_module, 'observe', original_observe))
        entry = module(successor/'reporting/runtime_entry.py', '_remaining_actual_reporting_entry')
        frozen_checkpoints = None
        for stage, resume in [('one_complete_epoch', None), ('stopped_finalization_resume', target/'recovery.pt')]:
            checkpoint = None if resume is None else receipt(resume)
            original_argv = [sys.executable, '-B', '-u', spec['runtime_entrypoint']['path'], '--project-root', str(ROOT),
                             '--job', spec['descriptor']['path'], '--output', str(target)]
            argv = transform(task, target, checkpoint, {}, original_argv)
            invocation_output = output/'invocations'/stage
            invocation_output.mkdir(parents=True)
            write(invocation_output/'BRIDGE_COMMAND.json', {'task': task, 'original_argv': original_argv, 'observed_argv': argv})
            previous_indices = set((output/'reporting').rglob('INDEX.json')) if (output/'reporting').exists() else set()
            before_updates = work['TRAIN_updates']
            old_argv = sys.argv
            try:
                sys.argv = [argv[3], *argv[4:]]
                exit_code = entry.main()
            finally:
                sys.argv = old_argv
            fit = read(target/'RESULT.json')
            require(exit_code == 0 and fit['status'] == 'complete' and fit['selected_epoch'] == fit['completed_epochs'] == 1
                    and fit['runtime_binding']['schema'] == 'server51-composed-normalization-genuine-runtime-binding-v1'
                    and fit['runtime_binding']['composition_producer']['source_manifest'] == spec['composed_runtime_authority']['producer_manifest']
                    and fit['TEST_accessed'] is False, 'Actual successor own fit/selected state required')
            expected_updates = 38 if resume is None else 0
            require(work['TRAIN_updates']-before_updates == expected_updates
                    and sum(fit['optimizer_calls_this_invocation']['attempts']) == expected_updates
                    and sum(fit['optimizer_calls_this_invocation']['completed']) == expected_updates
                    and fit['training_epochs_this_invocation'] == (1 if resume is None else 0)
                    and fit['finalization_only_resume'] is (resume is not None), 'Exact epoch/finalizer cost differs')
            indices = set((output/'reporting').rglob('INDEX.json'))-previous_indices
            require(len(indices) == 1, 'One genuine reporting invocation per original entry call required')
            index_path = indices.pop()
            folder = index_path.parent
            index, invocation, sidecar = read(index_path), read(folder/'INVOCATION_001.json'), read(folder/'SELECTED_EXPORT_001.json')
            require(index['exports'] == index['invocations'] == 1 and index['binding_spec_attached'] is True
                    and index['extra_native_work'] == index['RNG_calls'] == index['tensor_deserializations'] == 0,
                    'Exact scalar observer scope differs')
            require(invocation['bindings'] == spec['bindings'] and not invocation['binding_unresolved_reasons']
                    and invocation['composed_runtime_witness'] == sidecar['composed_runtime_witness']
                    and invocation['selected_sidecar'] == sidecar and sidecar['selected_checkpoint'] == fit['selected_checkpoint']
                    and sidecar['selected_export'] == fit['selected_VALID_export']
                    and sidecar['same_selected_bank_mean']['extra_forward_calls'] == 0,
                    'Actual bound selected-export sidecar and invocation must agree')
            preserved = []
            for name in ('RESULT.json', 'selected_VALID.pt'):
                pin = receipt(target/name)
                shutil.copyfile(target/name, invocation_output/name)
                preserved.append({'original': pin, 'preserved_copy': receipt(invocation_output/name)})
                require(all(preserved[-1]['preserved_copy'][k] == pin[k] for k in ('bytes', 'sha256')), 'Preserved invocation bytes differ')
            current_checkpoints = [receipt(target/name) for name in ('selected.pt', 'recovery.pt')]
            if frozen_checkpoints is None:
                frozen_checkpoints = current_checkpoints
            require(current_checkpoints == frozen_checkpoints, 'Stopped finalization changed own selected/recovery bytes')
            result['invocations'].append({'stage': stage, 'new_updates': expected_updates, 'preserved_exports': preserved,
                'observer_index': receipt(index_path), 'invocation': receipt(folder/'INVOCATION_001.json'),
                'selected_sidecar': receipt(folder/'SELECTED_EXPORT_001.json'), 'runtime_binding': fit['runtime_binding']})
            save()
        require(work == {'constructors_started': 2, 'constructors_completed': 2, 'state_restores': 3,
            'whole_VALID_calls': 3, 'native_forwards_attempted': 212, 'native_forwards_completed': 212,
            'mean_exports': 2, 'TRAIN_epochs_started': 1, 'TRAIN_epochs_completed': 1,
            'TRAIN_update_attempts': 38, 'TRAIN_updates': 38, 'TRAIN_callback_calls': 39,
            'TRAIN_exhaustion_probe_attempts': 1, 'TRAIN_exhaustion_probes_completed': 1,
            'optimizer_attempts': 38, 'optimizer_completed': 38,
            'forbidden_calls': 0}, 'Exact finite native work differs')
        for pin in spec['observed_input_files']:
            verify(pin)
        result.update(status='passed', actual_new_producer=spec['composed_runtime_authority']['producer_manifest'],
            bound_engineering_bindings=spec['bindings'], selected_checkpoint=fit['selected_checkpoint'],
            selected_export=fit['selected_VALID_export'], VALID_graphs_per_call=151, VALID_batches_per_call=5,
            original_artifacts_preserved=True, genuine_successor_projection_entry_bridge_writer_connected=True)
    except BaseException as error:
        result.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                      'traceback': traceback.format_exc(limit=16)})
    finally:
        signal.alarm(0)
        for handle in handles:
            handle.remove()
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
        if torch is not None:
            result['resources'] = {'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                                   'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(0)}
        result['own_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if result['own_peak_RSS_bytes'] > cap['host_RSS_reservation_bytes']:
            result.update(status='failed', host_reservation_exceeded=True)
        save()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True)
    args = parser.parse_args()
    result = execute(args.admission)
    print({key: result.get(key) for key in ('status', 'actual_work', 'failure')})
    raise SystemExit(0 if result['status'] == 'passed' else 1)
