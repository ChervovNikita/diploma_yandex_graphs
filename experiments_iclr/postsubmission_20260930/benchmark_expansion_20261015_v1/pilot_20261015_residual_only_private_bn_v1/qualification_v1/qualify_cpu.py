"""Exact frozen residual-only/private-BN MolBACE CPU lifecycle, two updates.

Uses existing CPU helper and runtime adapter construction. No fit/selector,
TEST access, automatic retry, or mutation of the frozen parent packet.
"""
import argparse
import copy
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
    # Before reading/loading any project source or metadata.
    if socket.gethostname() != 'anogena-2-0' or subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
            text=True, timeout=10).splitlines() != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        raise ValueError('Exact authorized allocation required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU qualification requires CUDA_VISIBLE_DEVICES empty')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    guard()
    root = Path(args.project_root).resolve()
    here = Path(__file__).resolve().parent
    parent, expansion = here.parent, here.parent.parent
    job_path, output = Path(args.job).resolve(), Path(args.output).resolve()
    if not all(p.is_relative_to(root) for p in (here, job_path, output)) or output.exists():
        raise ValueError('Use repository-contained sources and fresh engineering output')
    contract = json.loads((here / 'CONTRACT.json').read_text())
    for pin in contract['pins']:
        path = (root / pin['path_relative_to_project']).resolve()
        if not path.is_relative_to(root) or digest(path) != pin['sha256']:
            raise ValueError('Exact qualification input changed: ' + str(path))
    if job_path != parent / 'JOB_seed101.json' or digest(job_path) != contract['job_sha256']:
        raise ValueError('This mechanical lifecycle qualification uses exact first planned seed101 job')
    manifest = json.loads((parent / 'MANIFEST.json').read_text())
    for pin in manifest['files']:
        if digest(parent / pin['path']) != pin['sha256']:
            raise ValueError('Frozen parent artifact changed: ' + pin['path'])
    spec = json.loads((parent / 'SPEC.json').read_text())
    for pin in spec['source_pins'] + spec['data_role_pins']:
        path = Path(pin['path']).resolve()
        if not path.is_relative_to(root) or digest(path) != pin['sha256']:
            raise ValueError('Frozen source/data differs: ' + str(path))
    if any(Path(p).exists() for p in spec['absent_role_overrides']):
        raise ValueError('Unexpected TRAIN/VALID role override')
    helper_path = expansion / 'portable_private_input_v2/qualify_native_cpu.py'
    loader = importlib.util.spec_from_file_location('_retained_molbace_cpu_helper', helper_path)
    helper = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = helper
    loader.loader.exec_module(helper)
    sys.path.insert(0, str(expansion / 'runtime'))
    from run_fit import validate_job, OptimizerObserver
    from adapters import create_adapter
    from common import seed_all, global_rng, restore_rng, cpu_clone
    frozen = json.loads(job_path.read_text())
    policy = validate_job(frozen)  # Original CUDA descriptor/controller stays frozen.
    cpu_job = copy.deepcopy(frozen)
    cpu_job['device'] = 'cpu'  # Only execution override; not a scientific job rewrite.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {'schema': 'residual-only-private-bn-exact-CPU-lifecycle-v1', 'status': 'running',
              'source_job': helper.pin(job_path), 'parent_manifest': helper.pin(parent / 'MANIFEST.json'),
              'contract': helper.pin(here / 'CONTRACT.json'), 'script': helper.pin(__file__),
              'reused_CPU_helper': helper.pin(helper_path), 'execution_override': {'device': 'cpu'},
              'scientific_fit': False, 'scientific_selection': False, 'TEST_accessed': False,
              'maximum_optimizer_transitions': 2, 'maximum_native_constructions': 2,
              'same32_real_TRAIN_graphs_for_both_updates': True, 'expected_full_VALID_calls': 3,
              'native_construction_attempts': 0, 'native_constructions_completed': 0,
              'native_forward_attempts': 0, 'native_forwards_completed': 0,
              'VALID_attempts': 0, 'VALID_completed': 0, 'optimizer_calls': {'attempts': [0], 'completed': [0]},
              'full_epoch_selector_qualification': False, 'CUDA_numerical_qualification': False}
    adapters, hooks, observer = [], [], None

    def write():
        if observer is not None:
            report['optimizer_calls'] = observer.state()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter() - started,
                      process_user_seconds=usage.ru_utime, process_system_seconds=usage.ru_stime,
                      process_peak_RSS_KiB=usage.ru_maxrss)
        (output / 'RESULT.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    try:
        import torch
        from torch_geometric.data import Batch
        torch.set_num_threads(2)
        report.update(torch=str(torch.__version__), device='cpu', threads=2)

        def construct():
            report['native_construction_attempts'] += 1
            if report['native_construction_attempts'] > 2:
                raise AssertionError('Two-construction bound exceeded')
            seed_all(torch, frozen['seed'])
            adapter = create_adapter(torch, copy.deepcopy(cpu_job), policy, root)
            adapters.append(adapter)
            report['native_constructions_completed'] += 1
            if any(p.device.type != 'cpu' for p in adapter.model.parameters()):
                raise AssertionError('CPU-only parameter placement required')
            def before(module, arguments):
                report['native_forward_attempts'] += 1
            def after(module, arguments, result):
                report['native_forwards_completed'] += 1
            hooks.extend([adapter.model.native.register_forward_pre_hook(before),
                          adapter.model.native.register_forward_hook(after)])
            return adapter

        def validate(adapter):
            report['VALID_attempts'] += 1
            value = adapter.validate()
            if value['role'] != 'VALID' or value['complete_source_VALID'] is not True \
                    or value['TEST_accessed'] is not False or len(value['ids']) != 151 \
                    or not torch.equal(value['ids'], adapter.session.valid_ids.cpu()) \
                    or not torch.isfinite(value['outputs']).all().item():
                raise AssertionError('Exact full finite VALID151 and original order required')
            report['VALID_completed'] += 1
            return value

        def native_identity(adapter):
            current = dict(adapter.model.named_parameters())
            if any(current.get(name) is not p for name, p in adapter.session.source_parameter_references.items()):
                raise AssertionError('Native parameter identity changed')
            adapter.assert_identity()
            adapter.model.assert_source_identity()

        source = construct()
        owner, model = source.session, source.model
        manager = model.route_bn_state
        batch = Batch.from_data_list([owner.train_view[i] for i in range(32)])
        positions = torch.arange(32, dtype=torch.long)
        report['TRAIN_graph_ids'] = batch.smoke_graph_id.reshape(-1).tolist()
        owner._next_train_batch = lambda: (batch, positions, 0)
        model.eval()
        # Native reference on the same original body: inactive residual hook,
        # with a route BN context holding exact cloned initial native buffers.
        with torch.no_grad(), manager.route(0):
            reference = model.native(batch).detach().clone()
        with torch.no_grad():
            initial = torch.stack([model.run_member(m, batch)[1] for m in range(4)])
        report['initial_native_eval_max_abs_difference'] = float((initial - reference.unsqueeze(0)).abs().max())
        if not torch.equal(initial, reference.unsqueeze(0).expand_as(initial)):
            raise AssertionError('Initial residual-only routes do not equal native eval exactly')
        if hasattr(model, 'input_bank') or hasattr(model, 'output_bank'):
            raise AssertionError('Residual-only wrapper retained a BE bank')
        native_identity(source)

        # One charged TRAIN forward tests route ownership, then restore its
        # diagnostic buffer/RNG changes before either original optimizer update.
        banks = copy.deepcopy(manager.state_dict())
        canonical = {name: (id(b), b.detach().clone()) for name, b in model.native.named_buffers()}
        rng_before_isolation = global_rng(torch)
        model.train()
        with torch.no_grad():
            model.run_member(2, batch)
        after = manager.state_dict()
        for key, before in banks.items():
            route = int(key.split('.')[2])
            if route != 2 and not torch.equal(before, after[key]):
                raise AssertionError('A non-active BN route changed: ' + key)
            if route == 2 and key.endswith('num_batches_tracked') and not torch.equal(after[key], before + 1):
                raise AssertionError('Active route did not receive exactly one BN update: ' + key)
        for name, b in model.native.named_buffers():
            old_id, old_value = canonical[name]
            if id(b) != old_id or not torch.equal(b, old_value):
                raise AssertionError('Canonical native buffer changed outside route context')
        manager.load_state_dict(banks, strict=True)
        restore_rng(torch, rng_before_isolation)
        if any(not torch.equal(banks[k], v) for k, v in manager.state_dict().items()):
            raise AssertionError('Isolation diagnostic state was not restored')
        report.update(BN_isolation_active_route=2, BN_isolation_TRAIN_forwards=1,
                      BN_other_routes_unchanged=True, BN_canonical_identity_and_values_unchanged=True,
                      BN_diagnostic_state_and_RNG_restored=True)

        observer = OptimizerObserver(torch, source.optimizers)
        report['first_original_update'] = owner.train_update()
        if observer.state() != {'attempts': [1], 'completed': [1]} or owner.updates != 1:
            raise AssertionError('Exactly one original optimizer update required before checkpoint')
        valid = validate(source)
        native_identity(source)
        checkpoint = output / 'ENGINEERING_after_one.pt'
        torch.save({'adapter': cpu_clone(torch, source.state()), 'global_rng': cpu_clone(torch, global_rng(torch)),
                    'source_job_sha256': contract['job_sha256'], 'optimizer_calls': observer.state(),
                    'engineering_only_not_scientific_resume': True}, checkpoint)
        report['checkpoint'] = helper.pin(checkpoint)
        previous_calls = observer.state()
        observer.close()
        observer = None
        report['optimizer_calls'] = previous_calls
        fresh = construct()
        fresh.session._next_train_batch = lambda: (batch, positions, 0)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['source_job_sha256'] != contract['job_sha256']:
            raise AssertionError('Engineering checkpoint job differs')
        fresh.load_state(saved['adapter'])
        native_identity(fresh)
        replay = validate(fresh)
        report['fresh_restore_VALID_max_abs_difference'] = float((replay['outputs'] - valid['outputs']).abs().max())
        if not torch.equal(replay['outputs'], valid['outputs']) or not torch.equal(replay['ids'], valid['ids']):
            raise AssertionError('Fresh restore failed bit-exact full VALID parity')
        fresh.load_state(saved['adapter'])
        restore_rng(torch, saved['global_rng'])
        observer = OptimizerObserver(torch, fresh.optimizers, previous=saved['optimizer_calls'])
        report['second_original_update_after_fresh_restore'] = fresh.session.train_update()
        if observer.state() != {'attempts': [2], 'completed': [2]} or fresh.session.updates != 2:
            raise AssertionError('Two total original optimizer updates required')
        validate(fresh)
        native_identity(fresh)
        report['candidate_metadata'] = fresh.metadata()
        report.update(status='passed', native_update_clock=2, initial_native_eval_bit_exact=True,
                      fresh_restore_VALID_bit_exact=True, restored_native_parameter_identity=True,
                      engineering_completed_epochs=0, no_selector_called=True,
                      output_checkpoint_is_not_scientific_fit_initialization=True)
    except BaseException as error:
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error),
                      'traceback': traceback.format_exc(limit=15)})
    finally:
        write()
        if observer is not None:
            observer.close()
        for handle in hooks:
            handle.remove()
        for adapter in adapters:
            try:
                adapter.close()
            except Exception as error:
                report.setdefault('close_errors', []).append(str(error))
                report['status'] = 'failed'
        write()
    print(json.dumps({k: report[k] for k in ['status', 'optimizer_calls', 'native_constructions_completed',
                                           'native_forwards_completed', 'VALID_completed']}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
