"""Run two native optimizer updates for one exact deployed fit, then VALID/reload."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from smoke_control import install_step_cap, train_two_updates
from support import (E, PORT, PORT_SHA, digest, guard, inside, module, own_source,
                     pin, read, read_pinned, require, verify, write)


def equal_state(torch, left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(torch, left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(torch, a, b) for a, b in zip(left, right))
    if hasattr(left, 'tobytes') and hasattr(left, 'shape'):
        return type(left) is type(right) and left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()
    return type(left) is type(right) and left == right


def execute(args):
    config, config_pin = guard(args.project_root)
    root = Path(args.project_root).resolve()
    inventory, inventory_pin = read_pinned(args.inventory, args.inventory_sha256, root)
    require(inventory['schema'] == 'all-fit-smoke-inventory-v1', 'Wrong inventory schema')
    require(inventory['project_root'] == str(root) and inventory['source_closure_verified_once'] is True, 'Inventory root/source verification differs')
    source = own_source()
    require(source == inventory['source_manifest'], 'Inventory belongs to another smoke source')
    require(pin(PORT / 'MANIFEST.json') == inventory['producer_manifest'] and inventory['producer_manifest']['sha256'] == PORT_SHA, 'Port source manifest changed')
    matches = [row for row in inventory['fits'] if row['id'] == args.fit_id]
    require(len(matches) == 1, 'Exactly one inventoried fit required')
    row = matches[0]
    require(row['worker'] == config['worker'] and args.updates == row['TRAIN_updates_requested'] == 2, 'Exact worker/two-update request required')
    output = inside(args.output, root)
    require(output.is_relative_to(E) and not output.is_relative_to(E / 'campaign_outputs'), 'Smoke output must be separate from scientific fits')
    require(row['smoke_output_relative'] in output.parts and 'smokes' in output.parts, 'Use separate smokes/<fit-hash>/attempts/<attempt> output')
    require(not output.exists(), 'Preserve previous attempt; choose a fresh smoke attempt')
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {'schema': 'all-fit-two-update-smoke-result-v1', 'status': 'running',
        'id': row['id'], 'worker': row['worker'], 'inventory': inventory_pin,
        'descriptor': row['descriptor'], 'runtime_entrypoint': row['runtime_entrypoint'],
        'runtime_job_hash': row['runtime_job_hash'], 'canonical_runtime_job_hash': row['canonical_runtime_job_hash'],
        'science_contract_hash': row['science_contract_hash'], 'source_manifest': source,
        'producer_manifest': inventory['producer_manifest'], 'host_config': config_pin,
        'TRAIN_updates_requested': 2, 'TRAIN_updates': 0, 'TRAIN_finite': False,
        'VALID_finite': False, 'VALID_calls': 0, 'checkpoint_reload': {'status': 'not_started'},
        'optimizer_calls': None, 'TEST_accessed': False, 'scientific_checkpoint_reused': False,
        'scientific_fit_policy_changed': False, 'scientific_accuracy_claim': False,
        'full_training_admitted': False, 'automatic_retries': 0, 'child_pid': os.getpid()}
    adapter = observer = torch = runtime = None
    checkpoint_path = output / 'smoke_state.pt'

    def save():
        report['wall_seconds'] = time.monotonic() - started
        if observer is not None:
            report['optimizer_calls'] = observer.state()
            report['TRAIN_updates'] = sum(report['optimizer_calls']['completed'])
        write(output / 'RESULT.json', report)

    def interrupted(signum, _frame):
        raise InterruptedError('Smoke interrupted by signal ' + str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(signum, interrupted)
    require(1 <= args.wall_seconds <= 86400, 'Positive bounded child wall required')
    signal.alarm(args.wall_seconds)
    save()
    try:
        job = read(verify(row['descriptor'], root))
        require(job['id'] == row['id'] and digest(job) == row['runtime_job_hash'], 'Exact unchanged execution descriptor required')
        require(job['TEST_access'] is False and job['device'] == 'cuda:0', 'Only native CUDA TRAIN/VALID')
        entry = verify(row['runtime_entrypoint'], root)
        expected = PORT / ('composed/runtime' if job.get('normalization_composition') else 'runtime_link' if job['family'] == 'link' else 'runtime_node_molecular') / 'run_fit.py'
        require(entry == expected, 'Use actual deployed native runtime')
        require('torch' not in sys.modules and 'adapters' not in sys.modules, 'Each fit needs a fresh child interpreter')
        sys.path.insert(0, str(entry.parent))
        runtime = module(entry, '_all_fit_original_runtime')
        runtime.bound_project_root(root)
        runtime.validate_host_job(job, root)
        policy = runtime.validate_job(job, root)
        import torch as tensor_library
        torch = tensor_library
        import adapters
        require(Path(adapters.__file__).resolve() == entry.parent / 'adapters.py', 'Actual runtime adapter module required')
        torch.set_num_threads(int(job.get('torch_threads', 2)))
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        runtime.seed_all(torch, job['seed'], device=job['device'])
        adapter = adapters.create_adapter(torch, copy.deepcopy(job), policy, root)
        require(len(adapter.optimizers) == 1, 'One exact body/shared-bank optimizer required')
        observer = runtime.OptimizerObserver(torch, adapter.optimizers)
        install_step_cap(observer, adapter.optimizers)
        report.update(construction_seconds=time.monotonic() - started,
            actual_callback=adapter.callback_receipt, actual_extension=adapter.extension_receipt,
            actual_source_dependencies_sha256=digest(adapter.dependency_receipts),
            native_settings={'family': job['family'], 'recipe_id': job['recipe_id'], 'procedure': job['procedure'],
                             'seed': job['seed'], 'split': job.get('split', 0), 'body': job.get('body')})
        save()
        train_two_updates(adapter, observer, report, save)
        for parameter in adapter.model.parameters():
            require(torch.isfinite(parameter.detach()).all().item(), 'Nonfinite parameter after TRAIN')
        report['TRAIN_finite_checks'] = ['native_returned_loss', 'native_gradients_before_each_optimizer_step', 'model_parameters_after_two_updates']
        validation = adapter.validate()
        report['VALID_calls'] += 1
        summary = runtime.compact_validation(validation)
        require(torch.isfinite(validation['outputs']).all().item(), 'Nonfinite native VALID output')
        require(summary['records'] > 0, 'Empty VALID population')
        report.update(VALID_finite=True, VALID=summary, VALID_scope='complete_native_VALID', VALID_subsampled=False)
        del validation
        save()
        # This is explicitly a smoke snapshot, never a complete-epoch recovery.
        # No additional TRAIN call is made to reach an epoch boundary.
        state = runtime.cpu_clone(torch, adapter.state())
        rng = runtime.cpu_clone(torch, runtime.global_rng(torch, device=job['device']))
        payload = {'schema': 'all-fit-smoke-only-state-v1', 'id': row['id'],
                   'runtime_job_hash': row['runtime_job_hash'], 'descriptor': row['descriptor'],
                   'runtime_entrypoint': row['runtime_entrypoint'], 'source_manifest': source,
                   'adapter': state, 'global_rng': rng, 'optimizer_calls': observer.state(),
                   'usable_for_scientific_resume': False}
        checkpoint_pin = runtime.atomic_torch(torch, checkpoint_path, payload)
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        require(equal_state(torch, payload, loaded), 'Serialized smoke state differs on reload')
        require(loaded['runtime_job_hash'] == row['runtime_job_hash'] and loaded['descriptor'] == row['descriptor'], 'Wrong checkpoint identity')
        parameter = next(p for p in adapter.model.parameters() if p.is_floating_point() and p.numel())
        with torch.no_grad():
            first = parameter[tuple(0 for _ in parameter.shape)]
            before = first.detach().clone()
            first.add_(1.0)
            require(not torch.equal(first, before), 'Checkpoint restore probe must change one actual model parameter')
            moment_name, moment = next((name, value)
                for optimizer in adapter.optimizers for item in optimizer.state.values()
                for name, value in item.items()
                if name in ('exp_avg', 'exp_avg_sq', 'momentum_buffer')
                and torch.is_tensor(value) and value.is_floating_point() and value.numel())
            first_moment = moment[tuple(0 for _ in moment.shape)]
            previous_moment = first_moment.detach().clone()
            first_moment.add_(1.0)
            require(not torch.equal(first_moment, previous_moment), 'Checkpoint restore probe must change one initialized optimizer moment')
        adapter.load_state(loaded['adapter'])
        runtime.restore_rng(torch, loaded['global_rng'], device=job['device'])
        require(equal_state(torch, state, runtime.cpu_clone(torch, adapter.state())), 'Adapter state did not restore exactly')
        require(observer.state() == {'attempts': [2], 'completed': [2]}, 'Reload/VALID must add no optimizer work')
        report['checkpoint_reload'] = {'status': 'PASS', 'checkpoint': checkpoint_pin,
            'exact_adapter_state_restored': True, 'parameter_perturbation_restored': True,
            'optimizer_state_reloaded': True, 'optimizer_moment_perturbation_restored': True,
            'perturbed_optimizer_field': moment_name, 'TRAIN_updates_after_reload': 0,
            'complete_epoch_or_resume_claim': False, 'retained': True}
        torch.cuda.synchronize()
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                                               'traceback': traceback.format_exc(limit=14)})
    finally:
        if observer is not None:
            report['optimizer_calls'] = observer.state()
            try:
                observer.close()
            except BaseException as error:
                report.update(status='failed', observer_close_error=str(error))
        if adapter is not None:
            try:
                adapter.close()
            except BaseException as error:
                report.update(status='failed', close_error=str(error))
        if torch is not None:
            try:
                report['resources'] = runtime.resource_record(torch, device='cuda:0')
            except BaseException as error:
                report.update(status='failed', resource_record_error=str(error))
        # Successful temporary reload payloads are disposable engineering files.
        # Failures keep any checkpoint for diagnosis; no earlier output is touched.
        if report['status'] == 'PASS' and checkpoint_path.exists() and not args.keep_checkpoint:
            checkpoint_path.unlink()
            report['checkpoint_reload']['retained'] = False
            report['checkpoint_reload']['disposition'] = 'own temporary smoke checkpoint removed only after exact reload PASS'
        signal.alarm(0)
        save()
    print(json.dumps({k: report.get(k) for k in ('status', 'id', 'TRAIN_updates', 'VALID_finite', 'checkpoint_reload', 'failure')}))
    return 0 if report['status'] == 'PASS' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root', 'inventory', 'inventory-sha256', 'fit-id', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--updates', type=int, required=True, choices=[2])
    p.add_argument('--wall-seconds', type=int, default=1800)
    p.add_argument('--keep-checkpoint', action='store_true')
    return execute(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
