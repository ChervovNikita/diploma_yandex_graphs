"""Root-admitted actual one-epoch ownership, selected restore/resume and VALID serving.

One process is one case. This source does not launch or admit itself. It reads no
TEST payload and cannot authorize the campaign. The first two updates record
finite shared/private/gate gradients; the same trajectory completes its native
epoch so that the selected binding is positive-epoch and genuine.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
from runtime_bridge import (RUNTIME, build_runtime_binding, digest, load, receipt, require, verify_sources)

CASES = {
    'molbace': {'recipe_id': 'ogb-ogbg-molbace-gin-drop0.0', 'family': 'ogb', 'optimizer_updates': 38},
    'buddy_seed1': {'recipe_id': 'collab-buddy', 'family': 'link', 'optimizer_updates': 1585},
}


def write(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verified(pin, root):
    path = Path(pin['path']).resolve()
    require(path.is_relative_to(root) and receipt(path) == pin, 'Exact confined input receipt required')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--admission', required=True)
    args = parser.parse_args()
    host = load(HERE / 'host_binding.py', 'host_binding')
    host.guard()  # Before project metadata, data, Torch and source-closure reads.
    root = host.bound_project_root(args.project_root)
    admission_path = Path(args.admission).resolve()
    require(admission_path.is_relative_to(root), 'Admission must be inside the repository')
    admission = json.loads(admission_path.read_text())
    source = verify_sources(root)
    require(admission.get('schema') == 'root-admitted-composed-native-qualification-v1'
            and admission.get('root_reviewed_source') is True
            and admission.get('root_verified_two_worker_ceiling') is True
            and admission.get('source_manifest_sha256') == source['sha256']
            and admission.get('case') in CASES, 'Reviewed finite root/resource admission required')
    case = CASES[admission['case']]
    output = Path(admission['output']).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'Fresh repository-confined qualification output required')
    original_pin = admission['parent_descriptor']
    original = json.loads(verified(original_pin, root).read_text())
    require(original.get('recipe_id') == case['recipe_id'] and original.get('family') == case['family']
            and original.get('scientific_method_id') == 'interaction_input_conditionalJ'
            and original.get('procedure') == 'interaction_input_conditionalJ'
            and original.get('seed') == 1 and original.get('TEST_access') is False,
            'Exact declared conditional-input candidate on the new native dataset, seed1, required')
    require(admission.get('maximum_optimizer_updates') == case['optimizer_updates'], 'Exact one-epoch update bound required')
    require(type(admission.get('maximum_wall_seconds')) is int and 1 <= admission['maximum_wall_seconds'] <= 21600,
            'Finite root-reviewed wall bound of at most six hours required')
    role_path = verified(admission['VALID_role_contract'], root)
    role = json.loads(role_path.read_text())
    output.mkdir(parents=True)
    report = {'schema': 'composed-native-qualification-result-v1', 'status': 'running', 'case': admission['case'],
              'source_manifest': source, 'admission': receipt(admission_path), 'parent_descriptor': original_pin,
              'TRAIN_updates': 0, 'optimizer_attempts': 0, 'optimizer_completed': 0,
              'native_constructions': 0, 'native_forward_calls': 0, 'mean_export_calls': 0,
              'gradient_observations': [], 'BN_counters': [], 'invocations': [],
              'TEST_accessed': False, 'full_campaign_GO': False, 'scientific_accuracy_claim': False}
    started = time.perf_counter()
    def save():
        report['wall_seconds'] = time.perf_counter() - started
        write(output / 'RESULT.json', report)
    save()
    def timeout(signum, frame):
        raise TimeoutError('Root-admitted native qualification wall bound reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(admission['maximum_wall_seconds'])
    try:
        parent = copy.deepcopy(original)
        parent['id'] += '/composed-native-qualification-v1'
        parent.update(engineering_only=True, parent_compiler_descriptor=copy.deepcopy(original_pin),
                      parent_runtime_job_hash=digest(original))
        if parent['family'] == 'link':
            engineering = load(HERE / 'qualification_policy.py', '_composed_engineering_policy')
            parent['normalization_engineering_parent_policy'] = copy.deepcopy(parent['policy'])
            parent['policy'] = copy.deepcopy(engineering.FINITE)
            parent['link_policy_module'] = str(HERE / 'qualification_policy.py')
            parent['link_policy_sha256'] = receipt(HERE / 'qualification_policy.py')['sha256']
        else:
            parent['policy'].update(track='matched_pilot', maximum_epochs=1, minimum_epochs=1, patience=1,
                                    evaluate_TRAIN_each_epoch=False)
        write(output / 'PARENT_ENGINEERING_JOB.json', parent)
        composition = load(HERE.parent / 'composable_normalization_v2/contracts.py', '_native_qualification_composition_contract')
        declaration = {'schema': composition.SCHEMA, 'policy': composition.POLICY,
            'source_manifest_sha256': receipt(HERE.parent / 'composable_normalization_v2/MANIFEST.json')['sha256'],
            'banks_sha256': receipt(HERE.parent / 'private_bn_state_v3/banks.py')['sha256'],
            'parent_descriptor': receipt(output / 'PARENT_ENGINEERING_JOB.json'), 'parent_job_sha256': digest(parent),
            'matched_control_method_id': composition.CONTROL}
        roster = json.loads((HERE.parent / 'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text())
        job = composition.migrated_job(parent, declaration, roster)
        write(output / 'JOB.json', job)
        host.validate_host_job(job, root)
        require('torch' not in sys.modules, 'Fresh worker must authenticate source and descriptor before Torch')
        sys.path.insert(0, str(RUNTIME))
        import torch
        import adapters
        runtime = load(RUNTIME / 'run_fit.py', '_composed_qualification_run_fit')
        factory = adapters.create_adapter
        def construct(*arguments, **keywords):
            owner = factory(*arguments, **keywords)
            report['native_constructions'] += 1
            native = owner.manager._native_ref()
            source_owner = owner.session.native if job['family'] == 'link' else owner.session
            anchors = {id(p) for p in source_owner.source_parameter_references.values()}
            original_step = owner.optimizers[0].step
            original_update = owner.session.train_update
            original_train, original_export, original_close = owner.train_epoch, owner.selected_mean_export, owner.close
            def counter():
                return [[int(bank.num_batches_tracked.item()) for bank in banks] for banks in owner.manager.banks]
            def forwards(module, inputs):
                report['native_forward_calls'] += 1
            handle = native.register_forward_pre_hook(forwards)
            def step(*a, **kw):
                report['optimizer_attempts'] += 1; save()
                require(report['optimizer_attempts'] <= case['optimizer_updates'], 'No extra optimizer transition admitted')
                groups = {'shared': [], 'private': [], 'correction': [], 'gate': []}
                for name, parameter in owner.model.named_parameters():
                    if parameter.grad is None: continue
                    grad = parameter.grad.coalesce().values() if parameter.grad.is_sparse else parameter.grad
                    require(bool(torch.isfinite(grad).all()), 'Nonfinite actual gradient: ' + name)
                    kind = 'shared' if id(parameter) in anchors else 'gate' if ('gate' in name or 'readout' in name) \
                        else 'correction' if 'residual' in name else 'private'
                    if report['optimizer_attempts'] <= 2:
                        groups[kind].append({'name': name, 'norm': float(grad.double().norm().item())})
                if report['optimizer_attempts'] <= 2:
                    report['gradient_observations'].append({'update': report['optimizer_attempts'], 'groups': groups})
                value = original_step(*a, **kw); report['optimizer_completed'] += 1; save(); return value
            def update():
                value = original_update(); report['TRAIN_updates'] += 1; save(); return value
            def train():
                before = counter(); value = original_train(); after = counter()
                require(value['complete_native_TRAIN'] is True and owner.completed_epochs == 1,
                        'One complete native epoch required before selection')
                require(report['TRAIN_updates'] == case['optimizer_updates'], 'Native epoch update count differs')
                deltas = [[b-a for a,b in zip(old,new)] for old,new in zip(before,after)]
                require(deltas and any(min(row) > 0 for row in deltas)
                        and all(len(row) == 4 and min(row) >= 0 and len(set(row)) == 1 for row in deltas),
                        'Each active tracked BN must advance routes equally; unused native BN must remain unchanged')
                report['BN_counters'].append({'names': list(owner.manager.names), 'before': before, 'after': after,
                    'deltas': deltas, 'unused_native_modules': [name for name,row in zip(owner.manager.names,deltas) if row == [0]*4]})
                save(); return value
            def export(actual, **context):
                before = report['native_forward_calls']; value = original_export(actual, **context)
                require(report['native_forward_calls'] == before and value is not None and value['extra_forward_calls'] == 0,
                        'Exact selected same-call mean export must add zero native forwards')
                report['mean_export_calls'] += 1; save(); return value
            def close():
                handle.remove(); original_close()
            owner.optimizers[0].step, owner.session.train_update = step, update
            owner.train_epoch, owner.selected_mean_export, owner.close = train, export, close
            return owner
        adapters.create_adapter = construct
        fit = output / 'own_fit'
        frozen_training_checkpoints = None
        for stage, resume in [('one_complete_epoch', None), ('stopped_finalization_resume', fit / 'recovery.pt')]:
            before = report['optimizer_completed']
            result = runtime.run(copy.deepcopy(job), root, fit, resume=resume)
            report['invocations'].append({'stage': stage, 'result': result, 'new_updates': report['optimizer_completed']-before})
            save(); require(result['status'] == 'complete', 'Genuine producer failed: ' + str(result.get('failure')))
            snapshot = output / 'invocations' / stage
            snapshot.mkdir(parents=True)
            saved_exports = []
            for filename in ('RESULT.json', 'selected_VALID.pt'):
                original_path = fit / filename; copied_path = snapshot / filename
                shutil.copy2(original_path, copied_path)
                original_receipt, copy_receipt = receipt(original_path), receipt(copied_path)
                require(original_receipt['sha256'] == copy_receipt['sha256'], 'Invocation export snapshot differs')
                saved_exports.append({'original': original_receipt, 'preserved_copy': copy_receipt})
            report['invocations'][-1]['preserved_exports'] = saved_exports
            current_checkpoints = {name: receipt(fit/name) for name in ('selected.pt','recovery.pt')}
            if frozen_training_checkpoints is None: frozen_training_checkpoints = current_checkpoints
            else: require(current_checkpoints == frozen_training_checkpoints, 'Stopped resume changed selected/recovery training state')
            save()
            require(result['selected_epoch'] == 1 and result['completed_epochs'] == 1, 'Genuine positive selected epoch required')
            if resume is not None:
                require(result['finalization_only_resume'] and report['optimizer_completed'] == before,
                        'Stopped finalization resume must add no TRAIN work')
        require(report['mean_export_calls'] == 2, 'Both actual selected finalizations require their same-call mean export')
        require(report['optimizer_attempts'] == report['optimizer_completed'] == report['TRAIN_updates'] == case['optimizer_updates'],
                'Complete and no-extra native TRAIN trajectory required')
        for kind in ('shared', 'private', 'correction', 'gate'):
            require(any(row['norm'] > 0 for observation in report['gradient_observations']
                        for row in observation['groups'][kind]), 'No finite nonzero actual ' + kind + ' gradient in first two updates')
        report['zero_initialization_gradient_policy'] = 'Individual zero gradients are permitted; no down-gradient requirement at zero up. Nonzero correction-group credit required across updates1..2.'
        # Fresh genuine serving construction, no optimizer step or TEST role.
        adapters.create_adapter = factory
        selected_pin = receipt(fit / 'selected.pt')
        selected = torch.load(selected_pin['path'], map_location='cpu', weights_only=False)
        saved_valid = torch.load(fit / 'selected_VALID.pt', map_location='cpu', weights_only=False)
        bridge = load(HERE / 'runtime_bridge.py', 'runtime_bridge')
        _, owner, binding = bridge.load_runtime({'runtime_entrypoint': receipt(RUNTIME / 'run_fit.py')}, job, root)
        report['native_constructions'] += 1
        try:
            require(binding == selected['runtime_binding'], 'Serving must reproduce the exact genuine producer identity')
            owner.load_state(selected['adapter'])
            owner.bind_loaded_selection(selected, selected_checkpoint=selected_pin, job_hash=digest(job), runtime_binding=binding)
            serving = load(HERE / 'final_scoring/__init__.py', '_composed_native_selected_serving')
            from importlib import import_module
            backend = import_module('_composed_native_selected_serving.backend')
            before = owner.state_signature()
            helper = load(HERE.parent / 'final_test_serving_v1/link/backend.py', '_composed_qualification_link_helper') if job['family'] == 'link' else None
            runtime.restore_rng(torch, selected['global_rng'], device=job.get('device', 'cuda:0'))
            with backend.no_updates(owner):
                replay = backend.predict(torch, owner, job, 'VALID', role, root, link_helper=helper,
                                         cache_root=output / 'serving_cache')
            require(owner.state_signature() == before, 'VALID serving changed selected model or private BN banks')
            require(torch.equal(replay['ids'], saved_valid['ids']) and torch.equal(replay['labels'], saved_valid['labels']),
                    'Whole native selected VALID population/order/labels differ')
            require(torch.allclose(replay['logits'], saved_valid['raw_logits'], rtol=role['rtol'], atol=role['atol']),
                    'Source serving changed selected actual readout')
            if job['family'] == 'link':
                require(torch.equal(replay['query_endpoints'], saved_valid['query_endpoints']), 'Official query order differs')
            report['selected_serving'] = {'role': 'VALID', 'TEST_accessed': False,
                'records': len(replay['ids']), 'max_absolute_difference': float((replay['logits']-saved_valid['raw_logits']).abs().max()),
                'rtol': role['rtol'], 'atol': role['atol'], 'work': replay['work'], 'runtime_binding': binding,
                'whole_selected_population_preserved': True, 'model_and_BN_state_unchanged': True}
        finally:
            owner.close()
        report.update(status='passed',native_qualification=True, source_scope='one native molecular or BUDDY representative; no transfer/accuracy inference')
    except BaseException as error:
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                                              'traceback': traceback.format_exc(limit=24)})
    finally:
        signal.alarm(0)
        if 'torch' in sys.modules:
            import resource
            report['resources'] = {'own_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        save()
    print(json.dumps({key: report.get(key) for key in ('status','case','wall_seconds','TRAIN_updates','failure')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
