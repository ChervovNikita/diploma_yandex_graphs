"""One common NCN CUDA lifecycle gate, at most three optimizer transitions."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import socket
import subprocess
import sys
import time
import traceback


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def pin(path):
    path = Path(path).resolve(); h = hashlib.sha256(); count = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block); count += len(block)
    return {'path': str(path), 'bytes': count, 'sha256': h.hexdigest()}


def main():
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.root).resolve(); output = Path(args.output).resolve()
    package = Path(__file__).resolve().parent; expansion = package.parents[1]; phase = expansion.parent
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or output.exists() or not all(p.is_relative_to(root) for p in (package, output, phase)):
        raise ValueError('One visible allocation GPU and fresh own repository output required')
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'schema': 'C3-NCN-v2-common-CUDA-lifecycle-v2', 'status': 'running', 'device': 'cuda:0',
        'stage': 'before_numerical_import', 'actual_optimizer_transitions': 0, 'optimizer_attempts': 0,
        'maximum_optimizer_transitions': 3, 'VALID_calls': 0, 'maximum_VALID_calls': 4,
        'native_constructions': 0, 'additional_auxiliary_gradient_collections': 0,
        'TEST_accessed': False, 'scientific_fit': False, 'script': pin(__file__),
        'memory_events_before': Path('/sys/fs/cgroup/memory.events').read_text()}
    torch = None; adapter = None
    def write():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(elapsed_seconds=time.perf_counter()-started, process_user_seconds=usage.ru_utime,
            process_system_seconds=usage.ru_stime, process_peak_RSS_KiB=usage.ru_maxrss)
        if torch is not None and torch.cuda.is_initialized():
            report.update(peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(), peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved())
        tmp = output/'RESULT.json.part'; tmp.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n'); tmp.replace(output/'RESULT.json')
    write()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Actual sole CUDA device required')
        manifest = phase/'reference_wrapper_redesign_20261013_v1/smoke_20261014/MANIFEST.json'
        case = next(r for r in json.loads(manifest.read_text())['cases'] if r['id'] == 'collab-ncn/gnnm_boundary_native')
        recipe = copy.deepcopy(case['recipe']); recipe.update(_repository_root=str(root), _seed=0, seed=0)
        method_package = expansion/'context_affinity_link_port_v2'; callback = method_package/'callback.py'
        job = {'schema': 'native-full-epoch-fit-v1', 'id': 'C3-NCN-v2-CUDA-engineering/common',
            'cohort_id': 'NCN-CPU-engineering/split0/seed0', 'family': 'link', 'procedure': 'context_affinity_link_common_v2',
            'base_procedure': 'gnnm_boundary_native', 'seed': 0, 'split': 0, 'body': 0, 'recipe': recipe,
            'data_root': case['data_root'], 'device': 'cuda:0', 'TEST_access': False,
            'callback_module': str(callback), 'callback_sha256': pin(callback)['sha256'],
            'method_settings': {'manifest_sha256': pin(method_package/'MANIFEST.json')['sha256']},
            'policy': {'family': 'link', 'track': 'matched_extended', 'maximum_epochs': 1000, 'minimum_epochs': 100,
                'patience': 50, 'metric': 'hits@50', 'direction': 'max', 'ties': 'first',
                'evaluate_TRAIN_each_epoch': False, 'evaluation_mode': 'VALID_only_disclosed_adaptation'}}
        policy = load(expansion/'link_policy_v1/policy.py', '_C3_NCN_v2_CUDA_policy').create_policy(job, root)
        source = load(callback, '_C3_NCN_v2_CUDA_callback')
        report.update(job=job, callback=pin(callback), method_manifest=pin(method_package/'MANIFEST.json'),
            torch=str(torch.__version__), device_name=torch.cuda.get_device_name(),
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled())
        def exact(a, b, path='state'):
            if isinstance(a, torch.Tensor):
                if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
                    raise AssertionError('Exact tensor mismatch at '+path)
            elif isinstance(a, dict):
                if not isinstance(b, dict) or set(a) != set(b):
                    raise AssertionError('Exact dictionary keys differ at '+path)
                for key in a: exact(a[key], b[key], path+'.'+str(key))
            elif isinstance(a, (tuple, list)):
                if type(a) is not type(b) or len(a) != len(b):
                    raise AssertionError('Exact sequence differs at '+path)
                for i, (x, y) in enumerate(zip(a, b)): exact(x, y, path+'['+str(i)+']')
            elif type(a) is not type(b) or a != b:
                raise AssertionError('Exact scalar differs at '+path)
        def replay_compare(a, b, path='replay'):
            # A repeated CUDA floating calculation may use nondeterministic
            # sparse reductions. State restoration itself is checked exactly
            # above; report re-execution errors with fixed prior thresholds.
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
                    raise AssertionError('Replay tensor type/shape changed at '+path)
                x, y = a.detach().cpu().double(), b.detach().cpu().double()
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise AssertionError('Nonfinite replay tensor at '+path)
                difference = (x-y).abs()
                row = {'path': path, 'elements': a.numel(),
                    'max_absolute_difference': float(difference.max()) if a.numel() else 0.,
                    'RMS_difference': float(difference.square().mean().sqrt()) if a.numel() else 0.,
                    'coordinates_outside_fixed_threshold': int((difference > 1e-5 + 1e-4*x.abs()).sum()),
                    'rtol': 1e-4, 'atol': 1e-5}
                report.setdefault('CUDA_reexecution_diagnostics', []).append(row)
            elif isinstance(a, dict):
                if not isinstance(b, dict) or set(a) != set(b):
                    raise AssertionError('Replay dictionary keys changed at '+path)
                for key in a: replay_compare(a[key], b[key], path+'.'+str(key))
            elif isinstance(a, (tuple, list)):
                if type(a) is not type(b) or len(a) != len(b):
                    raise AssertionError('Replay sequence changed at '+path)
                for i, (x, y) in enumerate(zip(a, b)): replay_compare(x, y, path+'['+str(i)+']')
            else:
                exact(a, b, path)
        def construct():
            current = source.create_adapter(torch, copy.deepcopy(job), policy, root)
            report['native_constructions'] += 1
            if any(p.device.type != 'cuda' for p in current.model.parameters()):
                raise AssertionError('All original model Parameters must use actual CUDA')
            original = current.optimizers[0].step
            def step(*a, **kw):
                report['optimizer_attempts'] += 1
                if report['actual_optimizer_transitions'] >= 3:
                    raise AssertionError('Finite three-transition authorization exceeded')
                value = original(*a, **kw); report['actual_optimizer_transitions'] += 1; write(); return value
            current.optimizers[0].step = step
            return current
        def valid(current):
            value = current.validate(); report['VALID_calls'] += 1; torch.cuda.synchronize()
            if (value['complete_VALID_positive_records'], value['complete_VALID_negative_records']) != (60084, 100000) \
                    or value.get('complete_source_VALID') is not True or value.get('TEST_accessed') is not False:
                raise AssertionError('Whole source VALID and closed TEST required')
            if not torch.equal(value['query_endpoints'], torch.cat((current.session.native.valid['edge'], current.session.native.valid['edge_neg'])).cpu()):
                raise AssertionError('Full VALID endpoint identities/order changed')
            if not torch.isfinite(value['outputs']).all():
                raise AssertionError('Nonfinite served full VALID')
            return {'outputs': value['outputs'].clone(), 'member_outputs': value['member_outputs'].clone(),
                'labels': value['labels'].clone(), 'query_endpoints': value['query_endpoints'].clone()}
        report['stage'] = 'construct_first_source'; write(); adapter = construct()
        report['stage'] = 'first_source_update'; write(); report['first_update'] = adapter.session.train_update()
        torch.cuda.synchronize(); report['stage'] = 'first_whole_VALID'; write(); first_outputs = valid(adapter)
        saved = adapter.state(); checkpoint = output/'after_one_update.pt'; torch.save({'session': saved, 'job': job}, checkpoint)
        report['checkpoint'] = pin(checkpoint)
        report['stage'] = 'uninterrupted_second_source_update'; write(); report['uninterrupted_second_update'] = adapter.session.train_update()
        torch.cuda.synchronize(); report['stage'] = 'uninterrupted_second_whole_VALID'; write(); uninterrupted_outputs = valid(adapter)
        uninterrupted_state = adapter.state()
        adapter.close(); del adapter; adapter = None; gc.collect(); torch.cuda.empty_cache()
        report['stage'] = 'construct_fresh_source'; write(); adapter = construct()
        references = {name: id(p) for name, p in adapter.session.native.source_parameter_references.items()}
        saved_from_disk = torch.load(checkpoint, map_location='cpu', weights_only=False)['session']
        report['stage'] = 'device_safe_restore_and_exact_state_comparison'; write(); adapter.load_state(saved_from_disk)
        if references != {name: id(p) for name, p in adapter.session.native.source_parameter_references.items()}:
            raise AssertionError('Fresh original Parameter identities changed on restore')
        exact(saved, adapter.state()); report['fresh_restored_all_native_optimizer_sampler_RNG_state_exact'] = True
        report['stage'] = 'fresh_restored_first_whole_VALID'; write(); restored_outputs = valid(adapter)
        replay_compare(first_outputs, restored_outputs, 'restored_step1_VALID'); report['fresh_restored_full_VALID_checked'] = True
        # Reset the complete saved source state after the VALID observer,
        # keeping the resumed update comparison independent of observer RNG.
        adapter.load_state(saved_from_disk)
        report['stage'] = 'replayed_second_source_update'; write(); report['replayed_second_update'] = adapter.session.train_update()
        torch.cuda.synchronize(); report['stage'] = 'replayed_second_whole_VALID'; write(); replay_outputs = valid(adapter)
        replay_compare(uninterrupted_outputs, replay_outputs, 'replayed_step2_VALID')
        replay_compare(uninterrupted_state, adapter.state(), 'replayed_step2_state')
        if report['actual_optimizer_transitions'] != 3 or report['VALID_calls'] != 4 or adapter.session.native.updates != 2:
            raise AssertionError('Finite actual work count or restored update clock differs')
        warnings = [v for v in report.get('CUDA_reexecution_diagnostics', []) if v['coordinates_outside_fixed_threshold']]
        report.update(status='passed' if not warnings else 'complete_with_numerical_warning', stage='complete',
            numerical_warnings=warnings, replayed_second_state_checked=True, replayed_second_full_VALID_checked=True, native_update_clock=adapter.session.native.updates,
            source_parameter_identity_preserved=True, full_VALID_positive_records=60084, full_VALID_negative_records=100000)
    except BaseException as exc:
        report.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(limit=14)})
    finally:
        if adapter is not None:
            try: adapter.close()
            except Exception as exc: report['close_error'] = str(exc)
        report['memory_events_after'] = Path('/sys/fs/cgroup/memory.events').read_text()
        report['limits'] = ['One common NCN CUDA lifecycle engineering gate; no scientific fit or TEST',
            'Fixed pre-execution rtol1e-4/atol1e-5 diagnostics; no tolerance expansion, retry, or accuracy claim',
            'Checkpoint restore/identities/sampler/integer/RNG state exact; repeated CUDA float calculations reported separately',
            'Split, NCNC/BUDDY, pair-resource admission, full worker and final TEST reconstruction remain unqualified']
        write()
    print(json.dumps({k: report[k] for k in ('status', 'stage', 'actual_optimizer_transitions', 'VALID_calls', 'native_constructions', 'elapsed_seconds')}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
