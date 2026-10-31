"""Source-bound TRAIN-only resource prototypes; never a scientific runner.

Manifest preparation is stdlib metadata only. Numerical execution is a separate
one-case command after root source review. The approved study is never edited:
its unused SELECT interface is replaced in an explicit engineering data-authority
clone, optimizer actions are denied, and no validation/selection endpoint runs.
Molecular, link and author costs remain unknown in this node-only qualifier.
"""
from __future__ import annotations

import argparse
import builtins
import copy
from contextlib import ExitStack
import gzip
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from unittest import mock


class ForbiddenRoleAccess(RuntimeError):
    pass


class ForbiddenOptimizerKernel(RuntimeError):
    pass


NODE_FACTORIES = {'node.py', 'coordinate.py', 'context_keys.py', 'context_exclusion.py',
                  'polyformer.py', 'polyformer_boundary.py'}
ORIGINAL_FIVE = {'romanempire', 'amazonratings', 'minesweeper', 'questions', 'tolokers'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def local(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Use immutable project-relative paths')
    result = root / path
    if result.is_symlink() or not result.resolve().is_relative_to(root):
        raise ValueError('Source/input may not escape project or use symlinks')
    return result


def receipt(root, path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(root)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def bound(root, binding):
    path = local(root, binding['path'])
    raw = path.read_bytes()
    if len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('Immutable source/input bytes changed')
    return path


def role_paths(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        if set(value) >= {'path', 'sha256', 'bytes'}:
            return {value['path']}
        return set().union(*(role_paths(v) for v in value.values())) if value else set()
    return set()


def input_binding(study, value):
    if isinstance(value, dict):
        return value
    rows = [r for r in study['source']['files'] if r['path'] == value]
    if len(rows) != 1:
        raise ValueError('Every actual GRAPH/TRAIN file must have one immutable receipt')
    return rows[0]


def dimensions(config):
    values = config.get('native', config)
    return (int(values.get('width', values.get('hidden', 0))),
            int(values.get('depth', values.get('nlayer', 0))), float(values['dropout']))


def is_anchor(run):
    c = run['config']
    return bool(c.get('practical_source_anchor') or c.get('native_anchor')
                or c.get('is_native_anchor')
                or (('context_key_arm' in c or 'context_arm' in c)
                    and c.get('learning_rate') == .001 and c.get('dropout') == .2))


def representative_candidates(candidates):
    """Exact source points on a shape frontier plus source anchors.

    Equal shape/branch points differ only in optimizer LR/seed in these frozen
    cells. This is a declared random-weight timing/memory envelope proxy, not
    qualification of every lower point or mask. No predictive result is read.
    """
    by_shape = {}
    for row in candidates:
        by_shape.setdefault(dimensions(row['run']['config']), []).append(row)
    chosen = []
    for shape, values in sorted(by_shape.items()):
        anchors = [v for v in values if v.get('source_anchor') or is_anchor(v['run'])]
        frontier = not any(all(a >= b for a, b in zip(other, shape)) and other != shape for other in by_shape)
        if frontier or anchors:
            chosen.append(sorted(anchors or values, key=lambda v: v['config_id'])[0])
    return chosen


def custody_counts(root, study):
    for binding in study['source']['files']:
        if Path(binding['path']).name == 'CUSTODY.json':
            data = json.loads(bound(root, binding).read_text())
            if (data.get('schema') == 'node-provider-role-custody-v1'
                    and data.get('dataset_id') == study['dataset']):
                return {r['split_id']: r['role_counts']['train'] for r in data['splits']}
    raise ValueError('Exact prepared node custody TRAIN counts required; no array/count inference')


def prepare_manifest(root, plan_path, expected_hash):
    plan = json.loads(plan_path.read_text())
    if digest(plan) != expected_hash or plan.get('scientific_launch_enabled') is not False:
        raise ValueError('Bind the exact current disabled master')
    cases, blocked = [], []
    for entry in plan['studies']:
        path = local(root, entry['path']);raw = path.read_bytes()
        study = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
        if digest(study) != entry['spec_hash']:
            raise ValueError('Master study changed')
        if (study['data'].get('task') != 'node_classification'
                or not any(r['kind'] == 'model_fit' for r in study['runs'])):
            blocked.append({'study_id': study['study_id'], 'reason': 'node-only prototype; molecular/link/author resource costs remain unknown'})
            continue
        if ''.join(c for c in study['dataset'].lower() if c.isalnum()) in ORIGINAL_FIVE:
            raise ValueError('Original-paper dataset resource cases are excluded')
        if any(Path(r['adapter'].partition(':')[0]).name not in NODE_FACTORIES
               or r['adapter'].partition(':')[2] != 'make_adapter'
               for r in study['runs'] if r['kind'] == 'model_fit'):
            blocked.append({'study_id': study['study_id'], 'reason': 'Factory outside reviewed node resource scope'})
            continue
        counts = custody_counts(root, study)
        runs = {r['run_id']: r for r in study['runs']}
        for selection in study['runs']:
            if selection['kind'] != 'selection':
                continue
            candidates = []
            for c in selection['config']['configurations']:
                candidate_runs = [runs[k] for k in c['run_ids']]
                # Each assigned source objective remains represented separately.
                bodies = sorted({r['body'] for r in candidate_runs})
                for body in bodies:
                    first = min((r for r in candidate_runs if r['body'] == body), key=lambda r: (r['seed'], r['run_id']))
                    candidates.append({'config_id': c['config_id'], 'run': first, 'body': body,
                                       'source_anchor': bool(c.get('source_anchor'))})
            for body in sorted({c['body'] for c in candidates}):
                current = [c for c in candidates if c['body'] == body]
                fitted = [r for r in study['runs'] if r['kind'] == 'model_fit'
                          and r['config'].get('selected_config_from') == selection['run_id']]
                available_masks = sorted({r['split'] for r in fitted})
                if not available_masks or set(available_masks) - set(counts):
                    raise ValueError('Every prospective comparative mask needs actual custody counts')
                maximum = max(counts[m] for m in available_masks)
                maximum_masks = [m for m in available_masks if counts[m] == maximum]
                mask = maximum_masks[0]
                data_run = min((r for r in fitted if r['split'] == mask), key=lambda r: (r['body'], r['seed']))
                for representative in representative_candidates(current):
                    run = copy.deepcopy(representative['run']);run['split'] = mask
                    for key in ('node_inputs', 'context_targets'):
                        if key in data_run['config']:
                            run['config'][key] = copy.deepcopy(data_run['config'][key])
                    shape = dimensions(run['config'])
                    covered = [c['config_id'] for c in current
                               if all(a >= b for a, b in zip(shape, dimensions(c['run']['config'])))]
                    identifier = digest({'study': entry['spec_hash'], 'selection': selection['run_id'],
                                         'run': run, 'body': body})[:24]
                    cases.append({'case_id': identifier, 'study_binding': receipt(root, path),
                        'original_spec_hash': entry['spec_hash'], 'study_id': study['study_id'],
                        'selection_run_id': selection['run_id'], 'source_search_run_id': representative['run']['run_id'],
                        'source_input_cohort_run_id': data_run['run_id'], 'run': run,
                        'representative_dimensions': list(shape), 'actual_TRAIN_rows': maximum,
                        'representative_mask': mask, 'all_prepared_comparative_masks': available_masks,
                        'equal_maximum_TRAIN_row_masks': maximum_masks, 'covered_config_ids_shape_proxy_only': sorted(set(covered)),
                        'coverage_relation': 'declared_shape_envelope_random_timing_proxy;lower_configs_masks_seeds_not_individually_qualified',
                        'scientific_fits': 0, 'optimizer_updates': 0, 'real_data_execution_review_pending': True})
    return {'schema': 'practical-node-zero-update-manifest-v1', 'master_plan': receipt(root, plan_path),
            'master_hash': expected_hash, 'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'cases': cases, 'unqualified_studies': blocked, 'scientific_fits': 0,
            'scientific_launch_enabled': False, 'scope': 'TRAINmemory_time_and_random_checkpoint_proxy_only',
            'actual_optimizer_kernels': 0, 'original_selector_or_endpoint_qualification': False}


class OptimizerDenial:
    """Intercept concrete optimizer steps before construction; never call originals."""
    def __init__(self, torch):
        self.t = torch;self.denied_step_calls = 0;self.denied_functional_calls = 0
        self.actual_optimizer_kernels = 0;self.stack = ExitStack()

    def __enter__(self):
        def denied_step(optimizer, *args, **kwargs):
            self.denied_step_calls += 1
            return None  # Source loops may finish; their nominal update counters are discarded.
        def denied_function(*args, **kwargs):
            self.denied_functional_calls += 1
            raise ForbiddenOptimizerKernel('Optimizer functional action denied')
        for value in vars(self.t.optim).values():
            if isinstance(value, type) and issubclass(value, self.t.optim.Optimizer) and value is not self.t.optim.Optimizer:
                self.stack.enter_context(mock.patch.object(value, 'step', denied_step))
        functional = importlib.import_module('torch.optim._functional')
        for name, value in vars(functional).items():
            if callable(value) and not name.startswith('_') and name not in {'Tensor', 'List'}:
                self.stack.enter_context(mock.patch.object(functional, name, denied_function))
        for module_name, name in [('torch.optim.adam', 'adam'), ('torch.optim.adamw', 'adamw')]:
            module = importlib.import_module(module_name)
            self.stack.enter_context(mock.patch.object(module, name, denied_function))
        return self

    def __exit__(self, *args):
        self.stack.__exit__(*args)


class RoleIO:
    """Deny original heldout files even for hashes; permit only actual TRAIN inputs."""
    def __init__(self, torch, root, allowed, forbidden, output):
        self.t = torch;self.root = root;self.allowed = {Path(p).resolve() for p in allowed}
        self.forbidden = {Path(p).resolve() for p in forbidden};self.output = output.resolve()
        self.stack = ExitStack();self.tensor_load_paths = [];self.denied = 0

    def __enter__(self):
        original_load = self.t.load
        def checked_load(value, *args, **kwargs):
            if not isinstance(value, (str, os.PathLike)) or Path(value).resolve() not in self.allowed:
                self.denied += 1;raise ForbiddenRoleAccess('Tensor load is outside GRAPH/TRAIN/engineering-stub authorities')
            self.tensor_load_paths.append(str(Path(value).resolve()))
            return original_load(value, *args, **kwargs)
        def wrapper(original):
            def checked(value, mode='r', *args, **kwargs):
                if isinstance(value, (str, os.PathLike)):
                    path = Path(value).resolve()
                    if path in self.forbidden:
                        self.denied += 1;raise ForbiddenRoleAccess('Original heldout bytes are forbidden, including hash reads')
                    if path.is_relative_to(self.root):
                        if any(c in mode for c in 'wax+') and not path.is_relative_to(self.output):
                            raise ForbiddenRoleAccess('Qualifier cannot mutate frozen source/data')
                        if path.suffix.lower() in {'.pt', '.pth', '.pkl', '.pickle', '.npy', '.npz'} and path not in self.allowed and not path.is_relative_to(self.output):
                            raise ForbiddenRoleAccess('Undeclared project binary input')
                return original(value, mode, *args, **kwargs)
            return checked
        self.stack.enter_context(mock.patch.object(self.t, 'load', checked_load))
        self.stack.enter_context(mock.patch.object(builtins, 'open', wrapper(builtins.open)))
        self.stack.enter_context(mock.patch.object(io, 'open', wrapper(io.open)))
        return self

    def __exit__(self, *args):
        self.stack.__exit__(*args)


def tensor_digest(torch, values):
    h = hashlib.sha256()
    for name, value in values:
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def optimizers(adapter):
    return list(getattr(adapter.optimizer, 'optimizers', [adapter.optimizer]))


def allocate_adam_moments(torch, adapter):
    """Explicit synthetic zero moment storage, never initialized by an update."""
    for optimizer in optimizers(adapter):
        if type(optimizer).__name__ not in {'Adam', 'AdamW'} or optimizer.state:
            raise ValueError('Only fresh actual Adam/AdamW objects have qualified manual-moment layout')
        for group in optimizer.param_groups:
            for parameter in group['params']:
                state = optimizer.state[parameter]
                device = parameter.device if group.get('capturable') or group.get('fused') else 'cpu'
                state['step'] = torch.tensor(0., dtype=torch.float32, device=device)
                state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                if group.get('amsgrad'):
                    state['max_exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)


def optimizer_tensors(adapter):
    return [(str(i) + '/' + str(j) + '/' + key, value)
            for i, optimizer in enumerate(optimizers(adapter))
            for j, state in enumerate(optimizer.state.values())
            for key, value in sorted(state.items()) if hasattr(value, 'dtype')]


def engineering_clone(torch, root, study, case, output):
    run = copy.deepcopy(case['run']);inputs = run['config']['node_inputs']
    graph = torch.load(bound(root, input_binding(study, inputs['graph'])), map_location='cpu', weights_only=True)
    train = torch.load(bound(root, input_binding(study, inputs['train'])), map_location='cpu', weights_only=True)
    n = len(graph['x']);mask = torch.ones(n, dtype=torch.bool);mask[train['ids']] = False
    unused = torch.nonzero(mask, as_tuple=False).flatten()
    if not len(unused):
        raise ValueError('NonTRAIN public GRAPH row required for unused synthetic SELECT interface')
    stub = output / 'SYNTHETIC_UNUSED_SELECT_INTERFACE.pt'
    torch.save({'ids': unused[:1].long(), 'y': torch.zeros(1, dtype=torch.long)}, stub)
    binding = receipt(root, stub);old = inputs['select']
    replacement = binding if isinstance(old, dict) else binding['path']
    clone = copy.deepcopy(study);clone['stage'] = 'engineering'
    for row in clone['runs']:
        if row['kind'] == 'model_fit' and row['config'].get('node_inputs', {}).get('select') == old:
            row['config']['node_inputs']['select'] = copy.deepcopy(replacement)
    run['config']['node_inputs']['select'] = copy.deepcopy(replacement)
    clone['data']['roles']['select'][str(run['split'])] = copy.deepcopy(replacement)
    # The actual resource run may use another prepared mask than search split0.
    clone['data']['roles']['train'][str(run['split'])] = copy.deepcopy(run['config']['node_inputs']['train'])
    clone['runs'] = [row for row in clone['runs'] if row['run_id'] != run['run_id']] + [copy.deepcopy(run)]
    return clone, run, stub


def measure_adapter(torch, adapter, denial, output, warm_iterations=3, capture=None):
    if warm_iterations < 1:
        raise ValueError('At least one warm observation required')
    for name in ('validate', 'observe_validation', 'evaluate', 'export'):
        if hasattr(adapter, name):
            setattr(adapter, name, lambda *a, **k: (_ for _ in ()).throw(ForbiddenRoleAccess('Validation/selection/heldout endpoints are prohibited')))
    allocate_adam_moments(torch, adapter)
    before = tensor_digest(torch, adapter.model.named_parameters())
    moment_before = tensor_digest(torch, optimizer_tensors(adapter))
    parameter_bytes = sum(p.numel()*p.element_size() for p in adapter.model.parameters())
    gradient_bytes = 0;times = [];forward_calls = [];routes = []
    device = next(adapter.model.parameters()).device
    def synchronize():
        if device.type == 'cuda':torch.cuda.synchronize(device)
    def counted(module, inputs, output):
        forward_calls.append(1)
        if torch.is_tensor(output) and output.ndim == 3:routes.append(int(output.shape[0]))
    hook = adapter.model.register_forward_hook(counted)
    start_attempts = denial.denied_step_calls
    try:
        for index in range(1 + warm_iterations):
            synchronize();start = time.perf_counter()
            ignored = adapter.train_step(index + 1)
            del ignored  # Never persist returned TRAIN loss/nominal-update/metric values.
            synchronize();times.append(time.perf_counter()-start)
            gradient_bytes = max(gradient_bytes, sum(p.grad.numel()*p.grad.element_size() for p in adapter.model.parameters() if p.grad is not None))
    finally:
        hook.remove()
    after = tensor_digest(torch, adapter.model.named_parameters())
    moment_after = tensor_digest(torch, optimizer_tensors(adapter))
    if before != after or moment_before != moment_after or denial.denied_functional_calls:
        raise ValueError('Parameter/moment bytes changed or a forbidden optimizer kernel was attempted')
    if denial.denied_step_calls == start_attempts:
        raise ValueError('The exact source train path did not reach its intercepted optimizer action')
    adapter.model.eval();synchronize();start = time.perf_counter()
    with torch.no_grad():
        logits = adapter.logits_fn(adapter.model, False, 0)
        train_only_probabilities = adapter.probabilities(logits)[adapter.train_ids]
        del logits, train_only_probabilities
    synchronize();inference_seconds = time.perf_counter()-start
    serialization = {'scope': 'unmeasured;capture function not supplied'}
    if capture is not None:
        start = time.perf_counter();payload = capture(torch, adapter, fit_id='ENGINEERING_RANDOM_NOT_A_SCIENTIFIC_ANCESTOR', step=0, validation={}, decision={})
        clone_seconds = time.perf_counter()-start
        destination = output / 'RANDOM_ENGINEERING_SNAPSHOT.pt'
        start = time.perf_counter();torch.save({'engineering_only': True, 'never_selected_checkpoint': True, 'payload': payload}, destination)
        save_seconds = time.perf_counter()-start
        serialization = {'scope': 'actual_full_root_clone_and_torch_serializer_random_state_proxy', 'bytes': destination.stat().st_size,
                         'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'clone_seconds': clone_seconds,
                         'save_seconds': save_seconds, 'selected_checkpoint': False}
        destination.unlink();del payload
    return {'scope': 'RANDOM_WEIGHT_TRAIN_FWD_BACKWARD_NO_UPDATE_PROXY_NOT_ETA', 'parameter_bytes': parameter_bytes,
            'gradient_bytes': gradient_bytes, 'optimizer_moment_bytes': sum(v.numel()*v.element_size() for _, v in optimizer_tensors(adapter)),
            'optimizer_objects': len(optimizers(adapter)), 'parameters_sha256_before': before, 'parameters_sha256_after': after,
            'moments_sha256_before': moment_before, 'moments_sha256_after': moment_after,
            'cold_train_seconds': times[0], 'warm_train_seconds': times[1:], 'warm_train_median_seconds': statistics.median(times[1:]),
            'TRAIN_only_full_graph_serving_proxy_seconds': inference_seconds, 'model_forward_calls_observed': len(forward_calls),
            'route_predictions_observed': sum(routes), 'denied_optimizer_step_calls': denial.denied_step_calls,
            'actual_optimizer_kernel_calls': denial.actual_optimizer_kernels, 'actual_optimizer_updates': 0,
            'optimizer_kernel_timing': None, 'SELECT_CAL_TEST_endpoint_timing': None, 'all_fit_ETA': None,
            'serialization': serialization, 'scientific_fits': 0, 'accuracy_calls': 0, 'scores_emitted': 0}


def execute_case(root, manifest, identifier, output, warm_iterations):
    if manifest['helper_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Reviewed helper changed; prepare and review a fresh manifest')
    master = json.loads(bound(root, manifest['master_plan']).read_text())
    if digest(master) != manifest['master_hash'] or master.get('scientific_launch_enabled') is not False:
        raise ValueError('The exact disabled master changed')
    matches = [c for c in manifest['cases'] if c['case_id'] == identifier]
    if len(matches) != 1:raise ValueError('Exactly one reviewed resource case required')
    case = matches[0]
    if not any(e['path'] == case['study_binding']['path'] and e['spec_hash'] == case['original_spec_hash'] for e in master['studies']):
        raise ValueError('Resource case is outside the exact master')
    path = bound(root, case['study_binding']);raw = path.read_bytes()
    study = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
    if digest(study) != case['original_spec_hash'] or study['data']['task'] != 'node_classification':
        raise ValueError('Only exact declared node factory cases are supported')
    if Path(case['run']['adapter'].partition(':')[0]).name not in NODE_FACTORIES or case['run']['adapter'].partition(':')[2] != 'make_adapter':
        raise ValueError('Factory is outside reviewed node resource scope')
    expected = copy.deepcopy(next(r for r in study['runs'] if r['run_id'] == case['source_search_run_id']))
    cohort = next(r for r in study['runs'] if r['run_id'] == case['source_input_cohort_run_id'])
    expected['split'] = cohort['split']
    for key in ('node_inputs', 'context_targets'):
        if key in cohort['config']:expected['config'][key] = copy.deepcopy(cohort['config'][key])
    if expected != case['run'] or cohort['config'].get('selected_config_from') != case['selection_run_id']:
        raise ValueError('Resource case differs from actual source configuration and prospective mask cohort')
    forbidden = set().union(*(role_paths(study['data']['roles'][role]) for role in ('select', 'cal', 'evaluate')))
    for row in study['runs']:
        endpoints = row['config'].get('endpoint_inputs', {})
        forbidden.update(role_paths(endpoints))
    inputs = case['run']['config']['node_inputs']
    allowed = set().union(*(role_paths(inputs[role]) for role in ('graph', 'train')))
    allowed.update(role_paths(case['run']['config'].get('context_targets', {})))
    for binding in study['source']['files']:
        if Path(binding['path']).suffix == '.py':bound(root, binding)
    import torch  # First numerical import; manifest preparation never reaches here.
    sys.dont_write_bytecode = True;torch.set_num_threads(2)
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
    io_guard = RoleIO(torch, root, [root/p for p in allowed], [root/p for p in forbidden], output)
    with OptimizerDenial(torch) as denial, io_guard:
        clone, run, stub = engineering_clone(torch, root, study, case, output)
        io_guard.allowed.add(stub.resolve())
        from run_study import load_factory
        from engine import capture_checkpoint
        start = time.perf_counter()
        adapter = load_factory(root, run['adapter'])(torch=torch, project_root=root, study=clone, run=run, device='cuda:0')
        torch.cuda.synchronize();constructor_seconds = time.perf_counter()-start
        if denial.denied_step_calls or denial.denied_functional_calls:
            raise ValueError('Factory attempted an update before the audited train proxy')
        result = measure_adapter(torch, adapter, denial, output, warm_iterations, capture_checkpoint)
        result.update(constructor_seconds=constructor_seconds,
                      CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                      process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      tensor_load_paths=io_guard.tensor_load_paths, heldout_original_file_accesses=io_guard.denied,
                      synthetic_unused_SELECT_interface=receipt(root, stub), original_spec_unchanged=True,
                      original_selector_resource_qualified=False, data_authority='explicit_engineering_clone_GRAPH_TRAIN_plus_synthetic_unused_SELECT')
    result.update(case_id=identifier, master_hash=manifest['master_hash'], helper_sha256=manifest['helper_sha256'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['prepare-manifest', 'execute-node-case'], default='prepare-manifest')
    parser.add_argument('--plan');parser.add_argument('--expected-plan-hash');parser.add_argument('--manifest');parser.add_argument('--case-id')
    parser.add_argument('--reviewed-helper-sha256');parser.add_argument('--warm-iterations', type=int, default=3)
    args = parser.parse_args()
    from prepare_node_data import verify_allocation
    verify_allocation()  # Before any project metadata or tensor access.
    root = Path(args.project_root).resolve();output = local(root, args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == 'prepare-manifest':
        if not args.plan or not args.expected_plan_hash:raise ValueError('Bind exact disabled master')
        result = prepare_manifest(root, local(root, args.plan), args.expected_plan_hash)
        (output/'MANIFEST.json').write_text(json.dumps(result, indent=2)+'\n')
        print(canonical({'metadata_only': True, 'cases': len(result['cases']), 'unqualified_studies': len(result['unqualified_studies']), 'scientific_fits': 0}))
    else:
        current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.reviewed_helper_sha256 != current:raise ValueError('Execution requires the exact root-reviewed helper hash')
        if not args.manifest or not args.case_id:raise ValueError('Execute one reviewed exact manifest case per fresh process')
        manifest = json.loads(local(root, args.manifest).read_text())
        result = execute_case(root, manifest, args.case_id, output, args.warm_iterations)
        (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(canonical({'resource_proxy_completed': True, 'actual_optimizer_updates': 0, 'scientific_fits': 0, 'scores_emitted': 0}))


if __name__ == '__main__':
    main()
