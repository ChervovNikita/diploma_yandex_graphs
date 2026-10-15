"""One variant's bounded actual CPU lifecycle; prepared, not executed here.

Two same-32 TRAIN transitions, actual F/shared and W/private gradient checks,
three full VALID calls and fresh genuine-state restore. No selector or TEST.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback


def guard():
    if socket.gethostname() != 'anogena-2-0' or subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines() != [
            'GPU-44039938-fd82-41d2-fefd-de71514e2fac'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Dedicated allocation with CUDA hidden required before project reads')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--variant', choices=('structured', 'permuted'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); guard()
    root, here, output = Path(args.project_root).resolve(), Path(__file__).resolve().parent, Path(args.output).resolve()
    if not here.is_relative_to(root) or not output.is_relative_to(root) or output.exists():
        raise ValueError('Project-contained source and fresh engineering output required')
    spec = importlib.util.spec_from_file_location('_structural_residual_CPU_source', here / 'extension.py')
    extension = importlib.util.module_from_spec(spec); sys.modules[spec.name] = extension; spec.loader.exec_module(extension)
    receipts = extension.source_bindings(root, args.manifest_sha256)
    parent_spec = json.loads((extension.PARENT / 'SPEC.json').read_text())
    for pin in parent_spec['source_pins'] + parent_spec['data_role_pins']:
        path = Path(pin['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError('Frozen source/TRAIN/VALID integrity changed: ' + str(path))
    if any(Path(p).exists() for p in parent_spec['absent_role_overrides']): raise ValueError('Unexpected source role override')
    parent_path = extension.PARENT / 'JOB_seed101.json'
    parent = json.loads(parent_path.read_text())
    method = list(extension.METHODS)[int(args.variant == 'permuted')]
    planned = extension.descriptor(parent, method, args.manifest_sha256)
    sys.path.insert(0, str(extension.E / 'runtime'))
    from run_fit import validate_job, OptimizerObserver
    from adapters import create_adapter
    from common import seed_all, global_rng, restore_rng, cpu_clone
    policy = validate_job(planned)
    cpu_job = copy.deepcopy(planned); cpu_job['device'] = 'cpu'
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'schema': 'structural-residual-CPU-lifecycle-v1', 'status': 'running', 'variant': args.variant,
        'source_receipts': receipts, 'script': extension.pin(__file__), 'parent_JOB': extension.pin(parent_path),
        'in_memory_descriptor_sha256': extension.digest(planned), 'execution_override': {'device': 'cpu'},
        'scientific_fit': False, 'selector_calls': 0, 'TEST_accessed': False,
        'bounds': {'native_constructions': 2, 'optimizer_transitions': 2, 'native_forwards': 73,
                   'complete_VALID_calls': 3, 'reverse_collections_including_checks': 8},
        'native_construction_attempts': 0, 'native_constructions_completed': 0,
        'native_forward_attempts': 0, 'native_forwards_completed': 0,
        'VALID_attempts': 0, 'VALID_completed': 0, 'reverse_attempts_including_checks': 0,
        'reverse_completed_including_checks': 0, 'optimizer_calls': {'attempts': [0], 'completed': [0]},
        'CUDA_qualification': False, 'full_epoch_selector_qualification': False}
    adapters, hooks, observer = [], [], None

    def write():
        if observer is not None: report['optimizer_calls'] = observer.state()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(wall_seconds=time.perf_counter() - started, peak_RSS_KiB=usage.ru_maxrss,
                      process_user_seconds=usage.ru_utime, process_system_seconds=usage.ru_stime)
        (output / 'RESULT.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    try:
        import torch
        from torch_geometric.data import Batch
        torch.set_num_threads(2); report.update(torch=str(torch.__version__), device='cpu', threads=2)

        def construct():
            report['native_construction_attempts'] += 1
            if report['native_construction_attempts'] > 2: raise AssertionError('Construction bound exceeded')
            seed_all(torch, 101)
            adapter = create_adapter(torch, copy.deepcopy(cpu_job), policy, root); adapters.append(adapter)
            report['native_constructions_completed'] += 1
            if any(p.device.type != 'cpu' for p in adapter.model.parameters()): raise AssertionError('Only CPU placement allowed')
            def before(module, arguments):
                report['native_forward_attempts'] += 1
                if report['native_forward_attempts'] > 73: raise AssertionError('Native forward bound exceeded')
            def after(module, arguments, result): report['native_forwards_completed'] += 1
            hooks.extend([adapter.model.native.register_forward_pre_hook(before), adapter.model.native.register_forward_hook(after)])
            return adapter

        def validate(adapter):
            report['VALID_attempts'] += 1
            if report['VALID_attempts'] > 3: raise AssertionError('VALID bound exceeded')
            value = adapter.validate()
            if value['role'] != 'VALID' or value['TEST_accessed'] is not False or not value['complete_source_VALID'] \
                    or len(value['ids']) != 151 or not torch.equal(value['ids'], adapter.session.valid_ids.cpu()) \
                    or not torch.isfinite(value['outputs']).all().item(): raise AssertionError('Complete original VALID151 required')
            report['VALID_completed'] += 1
            return value

        source = construct(); owner, model = source.session, source.model
        batch = Batch.from_data_list([owner.train_view[i] for i in range(32)])
        positions = torch.arange(32, dtype=torch.long)
        owner._next_train_batch = lambda: (batch, positions, 0)
        report['TRAIN_graph_ids'] = batch.smoke_graph_id.reshape(-1).tolist()
        initial_state = source.state()
        report['table_summary'] = initial_state['session']['method']['table']
        model.eval()
        with torch.no_grad(), model.route_bn_state.route(0): reference = model.native(batch).detach().clone()
        with torch.no_grad(): initial = torch.stack([model.run_member(m, batch)[1] for m in range(4)])
        if not torch.equal(initial, reference[None].expand_as(initial)): raise AssertionError('Native initial evaluation function changed')
        report['initial_native_eval_bit_exact'] = True

        def checked_update(adapter, *, first):
            nonlocal observer
            session, current = adapter.session, adapter.model
            shared = tuple(p for _, p in current.native.named_parameters())
            private = tuple(current.private_parameters()); all_parameters = shared + private
            old = [p.detach().clone() for p in all_parameters]
            counters = {n: b.clone() for n, b in current.route_bn_state.named_buffers() if n.endswith('num_batches_tracked')}
            original_grad, original_step = torch.autograd.grad, session.optimizer.step
            collected, wrong = [], []

            def reverse(loss, parameters, **kwargs):
                index = len(collected)
                expected = shared if index == 0 else private
                if index > 1 or tuple(map(id, parameters)) != tuple(map(id, expected)):
                    raise AssertionError('F/shared then W/private partition required')
                if any(not torch.equal(p, saved) for p, saved in zip(all_parameters, old)):
                    raise AssertionError('Parameters changed before all old-state reverses')
                def charged(parameters, options):
                    report['reverse_attempts_including_checks'] += 1
                    if report['reverse_attempts_including_checks'] > 8: raise AssertionError('Reverse bound exceeded')
                    value = original_grad(loss, parameters, **options)
                    report['reverse_completed_including_checks'] += 1
                    return value
                wrong.append(charged(private if index == 0 else shared,
                    {'retain_graph': True, 'create_graph': False, 'allow_unused': True}))
                gradients = charged(parameters, kwargs)
                collected.append((float(loss.detach()), parameters, gradients))
                return gradients

            def step(*arguments, **keywords):
                if len(collected) != 2 or any(not torch.equal(p, saved) for p, saved in zip(all_parameters, old)):
                    raise AssertionError('One step must follow both old-state collections')
                for _, parameters, gradients in collected:
                    for p, g in zip(parameters, gradients):
                        if (g is None) != (p.grad is None) or (g is not None and not torch.equal(p.grad, g)):
                            raise AssertionError('Assigned gradient differs from correct block objective')
                if first:
                    down_grad, up_grad = collected[1][2]
                    if down_grad is None or down_grad.count_nonzero().item() != 0 or up_grad is None or up_grad.count_nonzero().item() == 0:
                        raise AssertionError('Zero-up start must allow up learning with initial zero down gradient')
                return original_step(*arguments, **keywords)

            torch.autograd.grad, session.optimizer.step = reverse, step
            try: result = session.train_update()
            finally: torch.autograd.grad, session.optimizer.step = original_grad, original_step
            if collected[0][0] != result['native_unweighted_F'] or collected[1][0] != result['private_weighted_W']:
                raise AssertionError('Block objective labels differ from actual reverse inputs')
            differing = any(g is not None and h is not None and not torch.equal(g, h)
                            for g, h in zip(collected[1][2], wrong[0]))
            if first and not differing: raise AssertionError('First native fixture did not discriminate weighted private credit')
            after = dict(current.route_bn_state.named_buffers())
            if any(not torch.equal(after[n], value + 1) for n, value in counters.items()):
                raise AssertionError('Exactly one TRAIN forward per BN route required')
            result.update(actual_reverse_collections_including_checks=4, correct_block_gradients_verified=True,
                          private_F_vs_W_gradients_differ=differing, original_BN_route_updates_verified=True)
            return result

        observer = OptimizerObserver(torch, source.optimizers)
        report['first_update'] = checked_update(source, first=True)
        if observer.state() != {'attempts': [1], 'completed': [1]}: raise AssertionError('Exactly one first Adam transition required')
        valid = validate(source)
        checkpoint = output / 'ENGINEERING_after_one.pt'
        torch.save({'adapter': cpu_clone(torch, source.state()), 'global_rng': cpu_clone(torch, global_rng(torch)),
                    'optimizer_calls': observer.state(), 'engineering_only_not_scientific_initialization': True}, checkpoint)
        report['checkpoint'] = extension.pin(checkpoint)
        observer.close(); observer = None; report['optimizer_calls'] = {'attempts': [1], 'completed': [1]}
        fresh = construct(); fresh.session._next_train_batch = lambda: (batch, positions, 0)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for field in ('table_record', 'weights', 'method'):
            bad = copy.deepcopy(saved['adapter'])
            if field == 'table_record': bad['session'][field]['metrics'][0] += 1
            elif field == 'weights': bad['session'][field][0, 0] += 1
            else: bad['session'][field]['seed'] = 202
            try: fresh.load_state(bad)
            except ValueError: pass
            else: raise AssertionError('Changed composition state accepted: ' + field)
        fresh.load_state(saved['adapter'])
        replay = validate(fresh)
        if not torch.equal(replay['outputs'], valid['outputs']) or not torch.equal(replay['ids'], valid['ids']):
            raise AssertionError('Fresh complete VALID restoration differs')
        fresh.load_state(saved['adapter']); restore_rng(torch, saved['global_rng'])
        observer = OptimizerObserver(torch, fresh.optimizers, previous=saved['optimizer_calls'])
        report['second_update_after_fresh_restore'] = checked_update(fresh, first=False)
        validate(fresh)
        if observer.state() != {'attempts': [2], 'completed': [2]} or fresh.session.updates != 2:
            raise AssertionError('Exactly two total Adam transitions required')
        report.update(status='passed', fresh_restore_VALID_bit_exact=True, changed_state_rejected=True,
                      correct_shared_F_private_W_same_old_state=True, candidate_metadata=fresh.metadata(),
                      output_checkpoint_is_not_scientific_initialization=True)
    except BaseException as error:
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                      'traceback': traceback.format_exc(limit=12)})
    finally:
        write()
        if observer is not None: observer.close()
        for handle in hooks: handle.remove()
        for adapter in adapters:
            try: adapter.close()
            except Exception as error:
                report.setdefault('close_errors', []).append(str(error)); report['status'] = 'failed'
        write(); gc.collect()
    print(json.dumps({k: report[k] for k in ('status', 'variant', 'optimizer_calls', 'native_forwards_completed',
                                           'reverse_completed_including_checks', 'VALID_completed')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
