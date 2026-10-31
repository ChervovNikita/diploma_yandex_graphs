"""Eight-case exact GNCL/retrieval/recurrent-decoder source proposal; root executes only.

Imports numerical APIs only after exact allocation identity and root receipt
checks. Reuses previously reviewed optimizer/role/commit denial helpers. A
fresh reviewed engineering receipt is required; this program cannot make GO.
No remote connection, fit registry, selection, evaluator or optimizer kernel.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock
HOST = 'anogena-2-0'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
ORIGINAL_FIVE = {'romanempire', 'amazonratings', 'minesweeper', 'questions', 'tolokers'}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def verify_route():
    if socket.gethostname() != HOST:
        raise ValueError('Exact authorized allocation hostname required')
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    if result.returncode or result.stdout.split() != [UUID]:
        raise ValueError('Exact sole authorized allocation UUID required')

def local(root, relative):
    value = Path(relative)
    result = root / value
    if value.is_absolute() or '..' in value.parts or result.is_symlink() or (not result.resolve().is_relative_to(root)):
        raise ValueError('Confined non-symlink project path required')
    return result

def receipt(root, path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).resolve().relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def bound(root, row):
    path = local(root, row['path'])
    if receipt(root, path) != row:
        raise ValueError('Exact source/request/input binding changed')
    return path

def module(root, row, name):
    path = bound(root, row)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

def state_hashes(torch, h, z, adapter):
    return {'parameters': h.tensor_digest(torch, adapter.model.named_parameters()), 'moments': h.tensor_digest(torch, h.optimizer_tensors(adapter)), 'buffers': h.tensor_digest(torch, adapter.model.named_buffers()), 'external': z.tree_digest(torch, adapter.capture_state())}

def sync(torch, model):
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def require_finite_gradients(torch, model):
    if any((p.grad is not None and (not torch.isfinite(p.grad).all().item()) for p in model.parameters())):
        raise FloatingPointError('Nonfinite exact-source random-state gradient')

def node_probe(torch, h, z, root, case, study, run, output, admission):
    """Exact full factory with only unused SELECT authority explicitly replaced."""
    actual_inputs = run['config']['node_inputs']
    allowed = [bound(root, actual_inputs[key]) for key in ('graph', 'train')]
    if run['config'].get('sparse_support'):
        allowed.append(bound(root, run['config']['sparse_support']))
    forbidden = [local(root, path) for (role, value) in study['data']['roles'].items() if role != 'train' for path in h.role_paths(value)]
    if not set(allowed).isdisjoint(forbidden):
        raise ValueError('Declared TRAIN inputs overlap forbidden role paths')
    runtime = local(root, admission['runtime_package'])
    sys.path.insert(0, str(runtime))
    with h.OptimizerDenial(torch) as denial, h.RoleIO(torch, root, allowed, forbidden, output) as io_guard:
        (clone, engineering_run, stub) = h.engineering_clone(torch, root, study, {'run': run}, output)
        io_guard.allowed.add(stub.resolve())
        clone['runs'] = [engineering_run]
        from run_study import load_factory
        factory = load_factory(root, engineering_run['adapter'])
        adapter = factory(torch=torch, project_root=root, study=clone, run=engineering_run, device='cuda:0')
        for name in ('validate', 'observe_validation', 'evaluate', 'export'):
            if hasattr(adapter, name):
                setattr(adapter, name, lambda *a, **k: (_ for _ in ()).throw(h.ForbiddenRoleAccess('Selectors/targets/metrics prohibited')))
        h.allocate_adam_moments(torch, adapter)
        moments = [v for (_, v) in h.optimizer_tensors(adapter)]
        extra_workspace = [torch.empty_like(p) for p in adapter.model.parameters() for _ in range(2)]
        initial = copy.deepcopy(adapter.capture_state())
        modes = {name: module.training for (name, module) in adapter.model.named_modules()}
        before = state_hashes(torch, h, z, adapter)
        fixed_weights = getattr(adapter, 'bootstrap_weights', None)
        fixed_hash = None if fixed_weights is None else h.tensor_digest(torch, [('fixed_weights', fixed_weights)])
        counts = {key: 0 for key in ('backward_calls', 'autograd_grad_VJP_calls', 'optimizer_attempts')}
        cycle_counts = []
        times = []
        gradient_bytes = 0
        (original_grad, original_backward) = (torch.autograd.grad, torch.Tensor.backward)

        def counted_grad(*args, **kwargs):
            counts['autograd_grad_VJP_calls'] += 1
            return original_grad(*args, **kwargs)

        def counted_backward(value, *args, **kwargs):
            counts['backward_calls'] += 1
            return original_backward(value, *args, **kwargs)
        hooks = []
        with ExitStack() as stack:
            for hook in hooks:
                stack.callback(hook.remove)
            stack.enter_context(mock.patch.object(torch.autograd, 'grad', counted_grad))
            stack.enter_context(mock.patch.object(torch.Tensor, 'backward', counted_backward))
            for index in range(2):
                adapter.restore_state(copy.deepcopy(initial))
                for (name, module) in adapter.model.named_modules():
                    module.training = modes[name]
                counts.update({key: 0 for key in counts})
                attempts = denial.denied_step_calls
                sync(torch, adapter.model)
                tick = time.perf_counter()
                with z.TensorCommitDenial(torch, list(adapter.model.parameters()), moments) as commits:
                    ignored = adapter.train_step(index + 1)
                    reported = dict(ignored.get("operation_counts", {}))
                    del ignored
                sync(torch, adapter.model)
                times.append(time.perf_counter() - tick)
                counts['optimizer_attempts'] = denial.denied_step_calls - attempts
                if commits.denied or denial.denied_functional_calls:
                    raise ValueError('Protected source action attempted')
                require_finite_gradients(torch, adapter.model)
                gradient_bytes = max(gradient_bytes, sum((p.grad.numel() * p.grad.element_size() for p in adapter.model.parameters() if p.grad is not None)))
                if counts['optimizer_attempts'] < 1 or counts['backward_calls'] + counts['autograd_grad_VJP_calls'] < 1:
                    raise ValueError('Complete source gradient callback did not reach a denied optimizer')
                for key in ('backward_calls', 'autograd_grad_VJP_calls'):
                    if key in reported and counts[key] != reported[key]:
                        raise ValueError('Source-reported gradient work differs from actual calls')
                cycle_counts.append(dict(counts))
                current = state_hashes(torch, h, z, adapter)
                if current['parameters'] != before['parameters'] or current['moments'] != before['moments']:
                    raise ValueError('Parameter or manual-moment bytes changed')
        adapter.restore_state(copy.deepcopy(initial))
        for (name, module) in adapter.model.named_modules():
            module.training = modes[name]
        if before != state_hashes(torch, h, z, adapter):
            raise ValueError('Initial source state not exactly restored')
        adapter.model.eval()
        sync(torch, adapter.model)
        tick = time.perf_counter()
        with torch.no_grad():
            logits = adapter.logits_fn(adapter.model, False, 0)
            if logits.ndim != 3 or not torch.isfinite(logits).all().item():
                raise ValueError('Finite full-graph source serving shape required')
            serving_shape = list(logits.shape)
            serving_cpu_bytes = logits.numel() * logits.element_size()
            del logits
        sync(torch, adapter.model)
        serving_seconds = time.perf_counter() - tick
        adapter.restore_state(copy.deepcopy(initial))
        for (name, module) in adapter.model.named_modules():
            module.training = modes[name]
        from engine import capture_checkpoint, restore_checkpoint, _atomic_checkpoint
        if Path(sys.modules['engine'].__file__).resolve() != runtime / 'engine.py':
            raise ValueError('Frozen engine runtime differs')
        tick = time.perf_counter()
        payload = capture_checkpoint(torch, adapter, fit_id='ENGINEERING_RANDOM_UNTRAINED_UNSELECTED', step=0, validation={}, decision={})
        clone_seconds = time.perf_counter() - tick
        snapshot = output / 'DISPOSABLE_RANDOM_ENGINEERING_STATE.pt'
        io_guard.allowed.add(snapshot.resolve())
        tick = time.perf_counter()
        size = _atomic_checkpoint(torch, snapshot, payload)
        save_seconds = time.perf_counter() - tick
        saved = receipt(root, snapshot)
        del payload
        torch.rand(3, device='cuda:0')
        adapter.model.eval()
        tick = time.perf_counter()
        restore_checkpoint(torch, snapshot, adapter, expected_fit_id='ENGINEERING_RANDOM_UNTRAINED_UNSELECTED', verify=True)
        restore_seconds = time.perf_counter() - tick
        if before != state_hashes(torch, h, z, adapter):
            raise ValueError('Engine model/moment/source/private RNG restore mismatch')
        if fixed_weights is not None and fixed_hash != h.tensor_digest(torch, [('fixed_weights', fixed_weights)]):
            raise ValueError('Fixed untrained table changed')
        snapshot.unlink()
        role_shapes = admission.get('node_bank_role_shapes', {}).get(case['dataset'])
        CPU_bank = {'qualified': False, 'scope': 'root metadata role counts not supplied'}
        if role_shapes is not None:
            if set(role_shapes) != {'cal', 'evaluate', 'custody_receipt'} or any((type(role_shapes[key]) is not int or role_shapes[key] < 1 for key in ('cal', 'evaluate'))) or role_shapes['cal'] + role_shapes['evaluate'] > case['model_graph_shape']['nodes']:
                raise ValueError('Finite shape-only CPU bank role counts required')
            custody = json.loads(bound(root, role_shapes['custody_receipt']).read_text())
            expected_shapes = {role: max((row['role_counts'][role] for row in custody['splits'])) for role in ('cal', 'evaluate')}
            if custody.get('dataset_id') != case['dataset'] or any((expected_shapes[role] != role_shapes[role] for role in expected_shapes)):
                raise ValueError('CPU bank shape counts differ from frozen dataset custody metadata')
            views = {role: [torch.zeros((1, role_shapes[role], case['shape_config']['classes']), dtype=torch.float32) for _ in range(4)] for role in ('cal', 'evaluate')}
            banks = {role: torch.cat(value, dim=0) for (role, value) in views.items()}
            coordinates = sum((value.numel() for value in banks.values()))
            bank_path = output / 'DISPOSABLE_CPU_BANK_SHAPE.pt'
            io_guard.allowed.add(bank_path.resolve())
            tick = time.perf_counter()
            torch.save({'engineering_dummy_shape_only': True, 'role_tensors': banks}, bank_path)
            bank_save_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            loaded = torch.load(bank_path, map_location='cpu', weights_only=True)
            bank_load_seconds = time.perf_counter() - tick
            if not loaded['engineering_dummy_shape_only'] or any((not torch.equal(banks[key], loaded['role_tensors'][key]) for key in banks)):
                raise ValueError('Dummy CPU shape serializer roundtrip differs')
            bank_receipt = receipt(root, bank_path)
            bank_path.unlink()
            CPU_bank = {'qualified': True, 'dummy_values_no_predictions': True, 'roles': expected_shapes, 'served_routes': 4, 'retained_route_logits_bytes': 4 * coordinates, 'cat_transient_bytes': 4 * coordinates, 'snapshot': bank_receipt, 'save_seconds': bank_save_seconds, 'load_seconds': bank_load_seconds, 'removed': True, 'actual_endpoint_ID_or_target_access': False}
            del loaded, views, banks
        stub_receipt = receipt(root, stub)
        stub.unlink()
        result = {'cold_callback_seconds': times[0], 'warm_callback_seconds': times[1:], 'warm_median_seconds': statistics.median(times[1:]), 'cycle_action_counts': cycle_counts, 'source_state_hashes_before': before, 'source_state_hashes_after': state_hashes(torch, h, z, adapter), 'full_graph_serving_shape': serving_shape, 'full_graph_serving_seconds': serving_seconds, 'full_graph_logits_CPU_bytes': serving_cpu_bytes, 'parameter_bytes': sum((p.numel() * p.element_size() for p in adapter.model.parameters())), 'gradient_bytes': gradient_bytes, 'manual_moment_bytes': sum((v.numel() * v.element_size() for v in moments)), 'extra_workspace_bytes': sum((v.numel() * v.element_size() for v in extra_workspace)), 'fixed_table_hash': fixed_hash, 'input_load_paths': io_guard.tensor_load_paths, 'denied_role_reads': io_guard.denied, 'unused_SELECT_stub': {**stub_receipt, 'removed': True, 'synthetic_one_row_never_validated': True}, 'random_engine_restore': {'snapshot': saved, 'bytes': size, 'clone_seconds': clone_seconds, 'save_seconds': save_seconds, 'load_and_verify_restore_seconds': restore_seconds, 'removed': True, 'selected_checkpoint': False}, 'optimizer_attempts': denial.denied_step_calls, 'optimizer_functional_attempts': denial.denied_functional_calls, 'original_optimizer_kernels': denial.actual_optimizer_kernels, 'actual_optimizer_updates': 0, 'CPU_bank_export_proxy': CPU_bank, 'real_CAL_evaluate_export': False}
        return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify_route()
    root = Path(args.project_root).resolve()
    output = local(root, args.output)
    admission_path = local(root, args.admission)
    admission = json.loads(admission_path.read_text())
    self_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if admission.get('schema') != 'root-approved-zero-update-resource-admission-v1' or admission.get('root_source_review_confirmed') is not True or admission.get('helper_sha256') != self_hash or (admission.get('scientific_GO') is not False) or (args.case_id not in admission.get('approved_case_ids', [])) or (admission.get('max_actual_optimizer_updates') != 0):
        raise ValueError('Fresh root exact-source engineering admission required; no scientific GO')
    proposal = json.loads(bound(root, admission['proposal_receipt']).read_text())
    for dependency in admission['runtime_dependency_receipts'] + admission['source_dependency_receipts']:
        bound(root, dependency)
    case = next((c for c in proposal['requested_cases'] if c['case_id'] == args.case_id))
    study_row = admission['case_study_receipts'][args.case_id]
    raw = bound(root, study_row).read_bytes()
    study = json.loads(gzip.decompress(raw))
    if study['dataset'].lower().replace('_', '').replace('-', '') in ORIGINAL_FIVE:
        raise ValueError('Original-paper datasets excluded for every method')
    run = next((r for r in study['runs'] if r['run_id'] == case['source_run_id']))
    if sha(run) != case['source_run_hash']:
        raise ValueError('Exact prospective source run changed')
    if output.exists():
        raise ValueError('Fresh output required; failures never overwritten or silently retried')
    h = module(root, admission['role_optimizer_cap_helper'], 'new_source_inherited_role_optimizer_cap')
    z = module(root, admission['tensor_commit_tree_helper'], 'new_source_inherited_tensor_commit_tree')
    timeout = case['timeout_seconds']
    old = signal.getsignal(signal.SIGALRM)

    def timed_out(*ignored):
        raise h.ResourceTimeout('Declared source engineering child deadline')
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(timeout)
    output.mkdir(parents=True)
    owner = h.own_identity()
    started = time.monotonic()
    (output / 'START.json').write_text(json.dumps({'case_id': args.case_id, 'owner': owner, 'admission': receipt(root, admission_path), 'helper_sha256': self_hash, 'timeout_seconds': timeout, 'scientific_fits': 0, 'GO': False}, indent=2) + '\n')
    try:
        import torch
        torch.set_num_threads(2)
        (cap, before_usage, after_init) = h.configure_cuda_allocator(torch, output)
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        if case['family'] not in ('neighborhood_distribution', 'sparse_feature_kernel', 'shared_message', 'private_hop'):
            raise ValueError('Only the reviewed closed15 source/provider cases are admitted')
        result = node_probe(torch, h, z, root, case, study, run, output, admission)
        result.update(schema='new-source-zero-update-resource-result-v1', status='RANDOM_SOURCE_ZERO_UPDATE_ENGINEERING_COMPLETE', case_id=args.case_id, owner=owner, constructor_and_probe_seconds=time.monotonic() - tick, elapsed_wall_seconds=time.monotonic() - started, CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(), current_child_allocator_cap=cap, GPU_usage_before_init=before_usage, GPU_usage_after_init=after_init, GPU_usage_at_completion=h.GPU_process_usage(), process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, scientific_fits=0, GO_created=False, scores_emitted=0, loss_values_emitted=0, heldout_target_files_opened=0, actual_optimizer_updates=0, whole_master_admitted=False)
        (output / 'RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps({'case_id': args.case_id, 'status': result['status'], 'actual_optimizer_updates': 0, 'scientific_fits': 0}))
    except BaseException as error:
        failure = h.terminal_failure(error, owner, started)
        failure.update(case_id=args.case_id, helper_sha256=self_hash, GO_created=False, optimizer_updates_authorized=0, automatic_retry=False)
        (output / 'FAILURE.json').write_text(json.dumps(failure, indent=2, sort_keys=True) + '\n')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
if __name__ == '__main__':
    main()
