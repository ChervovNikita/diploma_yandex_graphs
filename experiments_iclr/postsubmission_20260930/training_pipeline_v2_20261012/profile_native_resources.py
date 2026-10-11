"""Allocation-only native resource qualification, with no optimizer or scores.

Every case runs in a fresh process. The bank class is extracted from the frozen
node bridge's AST; the bridge factory (which constructs AdamW) is never called.
Only custody metadata and the prepared graph file are opened. Main execution is for the
root after source review; importing this module performs no scientific work.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

from prepare_node_data import (ALLOCATION_GPU, ALLOCATION_HOST, ALLOCATION_ROOT,
                               ORIGINAL_FIVE, canonical, digest, normalized,
                               verify_allocation)


MIB = 1024 ** 2
BACKBONES = ('ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep', 'TAG')
DATASETS = {'cora', 'citeseer', 'pubmed', 'wikics', 'amazon-computers',
            'amazon-photo', 'coauthor-cs', 'coauthor-physics',
            'squirrel-filtered', 'chameleon-filtered'}
BASE_CASES = (
    {'case': 'original_boundary4', 'architecture': 'original_gnnm', 'variant': 'own',
     'coefficients': {}, 'covers': ['original']},
    {'case': 'allmap4_own', 'architecture': 'internal_be', 'variant': 'own',
     'coefficients': {}, 'covers': ['single', 'allmap_shared4', 'allmap_single']},
    {'case': 'untied4_own', 'architecture': 'joint_untied', 'variant': 'own',
     'coefficients': {}, 'covers': ['joint_untied4']},
)
SAGE_CASES = (
    {'case': 'allmap4_gncl_supcon', 'architecture': 'internal_be',
     'variant': 'gncl_within_route_supcon',
     'coefficients': {'pool_weight': .75, 'supcon_weight': .1, 'temperature': .1},
     'covers': ['gncl_shared4', 'supcon_shared4', 'gncl_supcon_shared4',
                'supcon_single', 'supcon_allmap_single']},
    {'case': 'allmap4_two_view_alignment_residual', 'architecture': 'internal_be',
     'variant': 'two_view_internal_be',
     'coefficients': {'alignment_weight': .1, 'residual_weight': .1, 'temperature': .1},
     'covers': ['two_view_class_full', 'two_view_alignment_residual']},
    {'case': 'allmap4_two_view_supcon_eq2', 'architecture': 'internal_be',
     'variant': 'two_view_supcon_eq2',
     'coefficients': {'supcon_weight': .1, 'temperature': .1},
     'covers': ['two_view_supcon_eq2']},
)


def positive(value, name, *, integer=False):
    types = (int,) if integer else (int, float)
    if type(value) not in types or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite ' + name + ' required')
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def local(root, value):
    raw = Path(value).expanduser()
    raw = raw if raw.is_absolute() else root / raw
    # Reject symlink components, including ones whose target stays in root.
    for item in (raw, *raw.parents):
        if item.is_symlink():
            raise ValueError('Immutable inputs/output may not traverse a symlink: ' + str(raw))
    path = raw.resolve()
    if not path.is_relative_to(root):
        raise ValueError('Profiler path escapes the allocation project: ' + str(value))
    return path


def binding(root, path):
    path = local(root, path)
    if not path.is_file():
        raise ValueError('Missing profiler source/input: ' + str(path))
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
            'sha256': digest(path)}


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.free',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError('Sole allocation GPU required')
    gpu_uuid, name, total, free = [x.strip() for x in lines[0].split(',')]
    if gpu_uuid != ALLOCATION_GPU:
        raise RuntimeError('Allocation identity changed')
    return {'uuid': gpu_uuid, 'name': name, 'total_bytes': int(total) * MIB,
            'free_bytes': int(free) * MIB}


def process_memory():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory,process_name',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    rows = []
    for line in csv.reader(result.stdout.splitlines()):
        if not line:
            continue
        pid, gpu_uuid, memory, name = [x.strip() for x in line]
        if gpu_uuid != ALLOCATION_GPU:
            raise RuntimeError('Unexpected GPU in process inventory')
        rows.append({'pid': int(pid), 'gpu_uuid': gpu_uuid, 'process_name': name,
                     'used_bytes': int(memory) * MIB if memory.isdigit() else None})
    return rows


def read_custody(root, path):
    path = local(root, path)
    receipt = json.loads(path.read_text())
    if receipt.get('schema') != 'node-provider-role-custody-v1' or receipt.get('status') != 'prepared':
        raise ValueError('Actual prepared node custody required')
    dataset = receipt['dataset_id']
    if dataset not in DATASETS or normalized(dataset) in ORIGINAL_FIVE:
        raise ValueError('Dataset outside the new-dataset full-graph native scope')
    if receipt.get('benchmark_fits') != 0 or receipt.get('predictive_scores') != 0:
        raise ValueError('Custody unexpectedly contains fitting/scoring')
    graph = receipt['model_graph']
    if any(type(graph[k]) is not int or graph[k] < 1 for k in ('nodes', 'features', 'classes', 'edges')):
        raise ValueError('Positive native graph dimensions required')
    if graph['classes'] < 3:
        raise ValueError('Binary BCE requires a separate native packet')
    splits = receipt['splits']
    ids = [row['split_id'] for row in splits]
    if 0 not in ids or len(ids) != len(set(ids)):
        raise ValueError('Unique roster including search split0 required')
    declared = {row['path']: row for row in receipt['files']}
    if len(declared) != len(receipt['files']):
        raise ValueError('Ambiguous custody file binding')
    # Never hash/open any target-bearing role file, including TRAIN/SELECT.
    # Only custody metadata and the graph array are read.
    files = [binding(root, path)]
    observed = {}
    for row in splits:
        if set(row['node_inputs']) != {'graph', 'train', 'select'}:
            raise ValueError('Separate graph/TRAIN/SELECT custody required')
        if type(row['split_id']) is not int or row['split_id'] < 0:
            raise ValueError('Nonnegative integer split required')
        for role in ('train', 'select'):
            positive(row['role_counts'][role], role + ' count', integer=True)
        for relative in [row['node_inputs']['graph']]:
            raw = Path(relative)
            if raw.is_absolute() or 'sealed' in raw.parts or relative not in declared:
                raise ValueError('Only the declared unsealed graph input may be read')
            target = local(root, path.parent / relative)
            if not target.is_relative_to(path.parent):
                raise ValueError('Node input escapes its custody packet')
            if relative not in observed:
                item = binding(root, target)
                if (item['bytes'], item['sha256']) != (declared[relative]['bytes'], declared[relative]['sha256']):
                    raise ValueError('Node input changed after custody')
                observed[relative] = item
                files.append(item)
    if len({r['node_inputs']['graph'] for r in splits}) != 1:
        raise ValueError('This native packet expects one fixed full graph')
    # Use maximum declared TRAIN and SELECT sizes. These are shape proxies;
    # no actual role IDs/targets are loaded or selected by any case worker.
    representative = max(splits, key=lambda r: (r['role_counts']['train'],
                                               r['role_counts']['select'], -r['split_id']))
    return receipt, representative, files


def bank_ast(path):
    source = Path(path).read_text()
    module = ast.parse(source, filename=str(path))
    factories = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'make_adapter']
    if len(factories) != 1:
        raise ValueError('Expected one frozen native bridge factory')
    banks = [node for node in factories[0].body if isinstance(node, ast.ClassDef) and node.name == 'NativeBank']
    if len(banks) != 1:
        raise ValueError('Expected one directly declared frozen NativeBank')
    # Extraction compiles only the bank, never the enclosing optimizer factory.
    return compile(ast.fix_missing_locations(ast.Module(body=[banks[0]], type_ignores=[])),
                   str(path), 'exec'), hashlib.sha256(ast.dump(banks[0], include_attributes=False).encode()).hexdigest()


def import_source(path, label):
    name = label + '_' + digest(path)[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def parameter_digest(model):
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((tuple(value.shape), str(value.dtype))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def worker(request_path, result_path):
    # This must precede request, source, graph or role metadata reads.
    verify_allocation()
    started = time.perf_counter()
    result = {'schema': 'native-resource-case-v1', 'status': 'started', 'pid': os.getpid(),
              'optimizer_objects': 0, 'optimizer_updates': 0, 'benchmark_fits': 0,
              'predictive_scores': 0, 'accuracy_computations': 0,
              'heldout_target_files_opened': 0, 'target_bearing_role_files_opened': 0, 'phase': 'request'}
    torch = None
    try:
        request = json.loads(Path(request_path).read_text())
        root = Path(request['project_root']).resolve()
        if not root.is_relative_to(ALLOCATION_ROOT):
            raise ValueError('Worker project escapes the allocation repository')
        result_path = local(root, result_path)
        result.update(dataset_id=request['dataset_id'], backbone=request['backbone'],
                      case=request['case'], split_id=request['split']['split_id'],
                      shape=request['shape'], covers=request['covers'])
        for item in request['bindings']:
            path = local(root, item['path'])
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError('Worker input/source binding changed')
        import torch as torch_module
        torch = torch_module
        import torch_geometric
        from torch.nn import functional as F
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('One visible allocation CUDA device required')
        torch.set_num_threads(request['torch_threads'])
        torch.cuda.set_device(0)
        torch.cuda.init()
        device = torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        snapshots = []

        def snapshot(phase):
            torch.cuda.synchronize(device)
            rows = process_memory()
            current = next((row['used_bytes'] for row in rows if row['pid'] == os.getpid()), None)
            other = [r for r in rows if r['pid'] != os.getpid()]
            item = {'phase': phase, 'allocated_bytes': torch.cuda.memory_allocated(device),
                    'reserved_bytes': torch.cuda.memory_reserved(device),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                    'process_used_bytes': current, 'other_processes': other}
            snapshots.append(item)
            return item

        snapshot('cuda_context')
        current_gpu = gpu_snapshot()
        # A process-local allocator limit preserves concurrent work. This is
        # not a host change or a claim that budget OOM means physical OOM.
        allocator_budget = min(request['free_memory_budget_bytes'], current_gpu['free_bytes']) - request['context_headroom_bytes']
        if allocator_budget <= 0:
            raise RuntimeError('No free-memory headroom for a resource probe')
        torch.cuda.set_per_process_memory_fraction(allocator_budget / current_gpu['total_bytes'], device)
        result['allocator_budget'] = {'limit_bytes': allocator_budget,
            'context_headroom_bytes': request['context_headroom_bytes'],
            'free_memory_at_worker_start': current_gpu['free_bytes'],
            'scope': 'per-process allocator cap under current concurrent workload; cap OOM is not idle-GPU physical infeasibility'}
        result['environment'] = {'hostname': socket.gethostname(), 'gpu': gpu_snapshot(),
            'torch_version': torch.__version__, 'pyg_version': torch_geometric.__version__,
            'cuda_version': torch.version.cuda, 'torch_threads': torch.get_num_threads(),
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}

        def reject_optimizer(self, *args, **kwargs):
            result['optimizer_constructor_attempts'] = result.get('optimizer_constructor_attempts', 0) + 1
            raise RuntimeError('Optimizer construction is forbidden in this resource profiler')

        torch.optim.Optimizer.__init__ = reject_optimizer
        result['optimizer_constructor_guard'] = 'torch.optim.Optimizer.__init__ raises'
        result['phase'] = 'node_inputs'

        graph_inputs = torch.load(local(root, request['graph_input']), map_location='cpu', weights_only=True)
        if set(graph_inputs) != {'x', 'edge_index'}:
            raise ValueError('Unexpected prepared node input schema')
        nodes, features, classes = (request['model_graph'][key] for key in ('nodes', 'features', 'classes'))
        if (graph_inputs['x'].shape != (nodes, features) or graph_inputs['x'].dtype != torch.float32
                or graph_inputs['edge_index'].shape != (2, request['model_graph']['edges'])
                or graph_inputs['edge_index'].dtype != torch.long):
            raise ValueError('Prepared graph shape/dtype differs from custody')
        if not bool(torch.isfinite(graph_inputs['x']).all()):
            raise ValueError('Nonfinite graph features')
        train_rows, select_rows = request['maximum_train_rows'], request['maximum_select_rows']
        if min(train_rows, select_rows) < 1 or max(train_rows, select_rows) > nodes:
            raise ValueError('Invalid declared resource-proxy row counts')
        x = graph_inputs['x'].to(device)
        graph = SimpleNamespace(edge_index=graph_inputs['edge_index'].to(device))
        train_ids = torch.arange(train_rows, dtype=torch.long, device=device)
        select_ids = torch.arange(select_rows, dtype=torch.long, device=device)
        synthetic_y = torch.arange(train_rows, dtype=torch.long, device=device) % classes
        synthetic_select_storage = torch.zeros(select_rows, dtype=torch.long, device=device)
        # SELECT CE/NLL are forbidden. Reserve their possible gather/log-softmax
        # tensor shapes without computing a target-based loss or any score.
        select_shape_reserve = torch.empty(select_rows * classes * 8, dtype=torch.uint8, device=device)
        snapshot('node_inputs_resident')
        result['phase'] = 'model_construction'
        native = import_source(local(root, request['native_model_source']), 'resource_native')
        losses = import_source(local(root, request['node_losses']), 'resource_node_losses')
        method, members = request['architecture'], 4
        config = {'fast_input_initialization': 'rademacher'}
        needs_hidden = request['variant'] not in {'own', 'gncl'}
        shape = request['shape']
        arguments = (request['backbone'], shape['depth'], features, shape['width'], classes,
                     shape['ffn_multiplier'], shape['heads'], shape['normalization'], shape['dropout'])
        torch.manual_seed(request['seed'])
        torch.cuda.manual_seed_all(request['seed'])
        code, bank_hash = bank_ast(local(root, request['node_bridge']))
        environment = dict(torch=torch, native=native, method=method, members=members,
                           config=config, needs_hidden=needs_hidden, arguments=arguments,
                           device=device, graph=graph, x=x, __name__='native_resource_extracted_bank')
        exec(code, environment)
        model = environment['NativeBank']().to(device)
        result['bank_ast_sha256'] = bank_hash
        parameters = list(model.parameters())
        parameter_bytes = sum(tensor_bytes(p) for p in parameters)
        buffer_bytes = sum(tensor_bytes(p) for p in model.buffers())
        result['model_storage'] = {'parameter_bytes': parameter_bytes,
                                  'parameter_tensors': len(parameters), 'buffer_bytes': buffer_bytes,
                                  'physical_slow_bodies': 4 if method == 'joint_untied' else 1}
        before_hash = parameter_digest(model)
        before_versions = [p._version for p in parameters]
        # Explicit storage allowance, never an optimizer. Two moment-sized and
        # two temporary-sized pools coexist even with live backward activations.
        # This is a conservative declared AdamW allowance, not a measured step.
        result['phase'] = 'adamw_storage_allowance'
        adamw_storage = [torch.zeros_like(p, memory_format=torch.preserve_format)
                         for _ in range(4) for p in parameters]
        adamw_step_storage = torch.zeros(len(parameters), dtype=torch.float64, device=device)
        result['adamw_storage_allowance'] = {'bytes': 4 * parameter_bytes + 8 * len(parameters),
            'moment_parameter_multiples': 2, 'temporary_parameter_multiples': 2,
            'parameter_updates_performed': 0, 'actual_optimizer_step_measured': False}
        # A whole-panel class-gather allowance covers the changed synthetic
        # class-group layout in paired residual losses. Actual class identities
        # remain unopened; this is an explicit storage allowance.
        auxiliary_shape_reserve = torch.empty(4 * 4 * min(train_rows, 512) * shape['width'],
            dtype=torch.float32, device=device) if request['variant'] == 'two_view_internal_be' else None
        result['synthetic_class_group_storage_allowance_bytes'] = tensor_bytes(auxiliary_shape_reserve) if auxiliary_shape_reserve is not None else 0
        snapshot('model_and_allowance_resident')
        paired = request['variant'].startswith('two_view_')

        def backward_probe():
            model.train()
            model.zero_grad(set_to_none=True)
            first = model()
            hidden = model.last_hidden
            first_logits = first[:, train_ids]
            first_hidden = hidden[:, train_ids] if hidden is not None else None
            second_logits, second_hidden = None, None
            if paired:
                second = model()
                second_logits = second[:, train_ids]
                second_hidden = model.last_hidden[:, train_ids]
            auxiliary_labels = synthetic_y
            if paired and len(train_ids) > 512:
                first_hidden = first_hidden[:, :512]
                second_hidden = second_hidden[:, :512]
                auxiliary_labels = synthetic_y[:512]
            terms = losses.node_objective(torch, F, first_logits, first_hidden, synthetic_y,
                variant=request['variant'], coefficients=request['coefficients'],
                paired_logits=second_logits, paired_hidden=second_hidden,
                auxiliary_labels=auxiliary_labels, independent=False)
            terms['objective'].backward()
            if any(p.grad is None for p in parameters):
                raise RuntimeError('A declared native parameter did not receive a gradient')
            if any(not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError('Nonfinite resource-probe gradient')
            # Mirrors the production engine's model/moment finite scans without
            # optimizer-state construction or a parameter update.
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError('Nonfinite unchanged native parameter')
            for value in adamw_storage[:2 * len(parameters)]:
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError('Nonfinite synthetic moment storage')

        def select_forward_probe():
            model.eval()
            with torch.no_grad():
                logits = model()
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('Nonfinite label-free SELECT-forward logits')
                probabilities = (logits.mean(0).softmax(-1) if method == 'original_gnnm'
                                 else logits.softmax(-1).mean(0))
                selected = probabilities[select_ids]
                # Force the complete gather without targets, argmax, proper
                # loss, score, selection decision, or persistence of logits.
                if not bool(torch.isfinite(selected).all()):
                    raise FloatingPointError('Nonfinite label-free SELECT gather')

        def timed(callback):
            torch.cuda.synchronize(device)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            begin.record()
            callback()
            end.record()
            end.synchronize()
            return {'wall_seconds': time.perf_counter() - wall,
                    'cuda_event_seconds': begin.elapsed_time(end) / 1000}

        iterations = []
        result['phase'] = 'backward_and_label_free_select'
        for index in range(request['warmup_iterations'] + request['measured_iterations']):
            train_time = timed(backward_probe)
            snapshot('backward_' + str(index))
            select_time = timed(select_forward_probe)
            snapshot('select_forward_' + str(index))
            iterations.append({'index': index, 'warmup': index < request['warmup_iterations'],
                               'backward_probe': train_time, 'select_forward_probe': select_time})
        after_hash = parameter_digest(model)
        if before_hash != after_hash or before_versions != [p._version for p in parameters]:
            raise RuntimeError('Resource probe modified native parameters')
        result.update(status='complete', phase='complete', parameters_unchanged=True,
                      parameter_sha256_before=before_hash, parameter_sha256_after=after_hash,
                      iterations=iterations, memory_snapshots=snapshots,
                      backward_calls=len(iterations), model_forwards=4 * (2 if paired else 1) * len(iterations),
                      label_free_select_model_forwards=4 * len(iterations),
                      synthetic_target_law='synthetic row index modulo class count; no target-bearing role files opened',
                      auxiliary_rows=min(len(train_ids), 512) if paired else len(train_ids),
                      objective_values_saved=False, logits_saved=False)
        result['gradient_storage_bytes'] = sum(tensor_bytes(p.grad) for p in parameters if p.grad is not None)
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        result['wall_seconds'] = time.perf_counter() - started
        # Keep all declared pools live until the final peak is recorded.
        assert adamw_storage and adamw_step_storage is not None and synthetic_select_storage is not None
        assert select_shape_reserve is not None
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), wall_seconds=time.perf_counter() - started)
        result['oom'] = (torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError)) or 'out of memory' in str(exc).lower()
        if result['oom']:
            result['oom_scope'] = 'within declared process-local allocator/headroom budget; physical idle-GPU feasibility unqualified'
        if torch is not None and torch.cuda.is_initialized():
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        # Partial snapshots remain useful even if a later case phase OOMs.
        if 'snapshots' in locals():
            result['memory_snapshots'] = snapshots
    atomic_json(result_path, result)
    return 0 if result['status'] == 'complete' else 2


def run_case(request_path, result_path, log_path, *, timeout, sample_seconds, root, threads):
    samples, sampling_errors = [], []
    stop = threading.Event()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', GNNM_TORCH_THREADS=str(threads),
               OMP_NUM_THREADS=str(threads))
    start = time.perf_counter()
    with Path(log_path).open('x') as log:
        process = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
            '--worker-request', str(request_path), '--worker-result', str(result_path)],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)

        def sample():
            while not stop.is_set():
                try:
                    rows = process_memory()
                    samples.append({'elapsed_seconds': time.perf_counter() - start,
                                    'used_bytes': next((r['used_bytes'] for r in rows if r['pid'] == process.pid), None),
                                    'other_pids': [r['pid'] for r in rows if r['pid'] != process.pid],
                                    'other_processes': [r for r in rows if r['pid'] != process.pid]})
                except Exception as exc:
                    sampling_errors.append(str(exc))
                stop.wait(sample_seconds)

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        termination = None
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            termination = 'case_timeout'
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            stop.set()
            monitor.join(timeout=15)
    if Path(result_path).is_file():
        result = json.loads(Path(result_path).read_text())
    else:
        result = {'schema': 'native-resource-case-v1', 'status': 'failed',
                  'error': termination or 'worker_exited_without_receipt', 'oom': None,
                  'optimizer_objects': 0, 'optimizer_updates': 0, 'benchmark_fits': 0,
                  'predictive_scores': 0, 'accuracy_computations': 0,
                  'heldout_target_files_opened': 0, 'target_bearing_role_files_opened': 0}
    result['parent_monitor'] = {'pid': process.pid, 'returncode': code,
        'wall_seconds': time.perf_counter() - start, 'sample_seconds': sample_seconds,
        'samples': samples, 'sampling_errors': sampling_errors, 'termination': termination,
        'peak_process_used_bytes': max((r['used_bytes'] for r in samples if r['used_bytes'] is not None), default=None)}
    if code != 0 or termination or sampling_errors:
        result['status'] = 'failed'
        result.setdefault('error', 'Nonclean worker exit or unavailable process-memory monitor')
    result['concurrent_cuda_workload_observed'] = any(r['other_pids'] for r in samples)
    result['timing_scope'] = 'measured under concurrent CUDA workload' if result['concurrent_cuda_workload_observed'] else 'no other CUDA compute process observed during samples'
    if result['status'] == 'complete' and (result['parent_monitor']['peak_process_used_bytes'] is None
            or not any(r['process_used_bytes'] is not None for r in result['memory_snapshots'])):
        result.update(status='failed', error='Per-process VRAM measurement unavailable')
    # Replace only this freshly created case result to add its monitor. Never
    # overwrite an earlier output directory or a previous qualification attempt.
    atomic_json(result_path, result)
    return result


def summarize_case(result, *, timing_factor, memory_factor, margin_mib):
    if result.get('status') != 'complete':
        return None
    measured = [row for row in result['iterations'] if not row['warmup']]
    walls = [r['backward_probe']['wall_seconds'] + r['select_forward_probe']['wall_seconds'] for r in measured]
    cuda = [r['backward_probe']['cuda_event_seconds'] + r['select_forward_probe']['cuda_event_seconds'] for r in measured]
    process_peak = result['parent_monitor']['peak_process_used_bytes']
    snapshot_process = [r['process_used_bytes'] for r in result['memory_snapshots'] if r['process_used_bytes'] is not None]
    if process_peak is None or not snapshot_process:
        raise ValueError('Per-process VRAM measurement unavailable; case is not admitted')
    outside_allocator = max(0, max(r['process_used_bytes'] - r['reserved_bytes']
        for r in result['memory_snapshots'] if r['process_used_bytes'] is not None))
    measured_envelope = max(process_peak, max(snapshot_process),
                            result['peak_reserved_bytes'] + outside_allocator)
    reservation = math.ceil((measured_envelope * memory_factor + margin_mib * MIB) / MIB)
    return {'case': result['case'], 'covers': result['covers'],
            'timing_scope': result.get('timing_scope', 'unrecorded'),
            'concurrent_cuda_workload_observed': result.get('concurrent_cuda_workload_observed', False),
            'process_allocator_limit_bytes': result.get('allocator_budget', {}).get('limit_bytes'),
            'measured_iteration_wall_seconds_median': statistics.median(walls),
            'measured_iteration_wall_seconds_max': max(walls),
            'measured_iteration_cuda_event_seconds_median': statistics.median(cuda),
            'estimated_seconds_per_full_case_update': max(walls) * timing_factor,
            'measured_process_peak_bytes': max(process_peak, max(snapshot_process)),
            'sampled_nonallocator_allowance_bytes': outside_allocator,
            'measured_envelope_bytes': measured_envelope, 'reservation_mib': reservation,
            'parameter_bytes': result['model_storage']['parameter_bytes'],
            'parameter_tensors': result['model_storage']['parameter_tensors'],
            'buffer_bytes': result['model_storage']['buffer_bytes'],
            'adamw_storage_allowance_bytes': result['adamw_storage_allowance']['bytes']}


def arm_counts(backbone, split_count, scalar_mode):
    """Exact starts from the frozen five bases/eight SAGE scalar templates."""
    result = {arm: {'search': 200, 'comparative': 3 * split_count}
              for arm in ('single', 'original', 'allmap_shared4', 'allmap_single', 'joint_untied4')}
    # Independently trained I4 members reuse member0; only three extra starts.
    result['single']['comparative'] += 9 * split_count
    result['allmap_single']['comparative'] += 9 * split_count
    if backbone == 'SAGE':
        coefficient_fields = {'gncl_shared4': 1, 'supcon_shared4': 2, 'gncl_supcon_shared4': 3,
            'supcon_single': 2, 'supcon_allmap_single': 2, 'two_view_class_full': 2,
            'two_view_supcon_eq2': 2, 'two_view_alignment_residual': 3}
        for arm, fields in coefficient_fields.items():
            result[arm] = {'search': 200 * (3 ** fields if scalar_mode == 'neighbors' else 1),
                           'comparative': 3 * split_count}
        result['supcon_allmap_single']['comparative'] += 9 * split_count
    return result


def cost_projection(custodies, backbones, case_summaries, *, scalar_mode, fit_overhead_seconds,
                    checkpoint_io_mib_per_second):
    """Planning arithmetic only; missing measurements stay unknown, never zero."""
    cells = []
    counts = Counter()
    known_compute = known_storage = known_io_min = known_io_max = 0.0
    unknown = []
    for receipt in custodies:
        dataset = receipt['dataset_id']
        split_count = len(receipt['splits'])
        for backbone in backbones:
            key = dataset + '/' + backbone
            arms = arm_counts(backbone, split_count, scalar_mode)
            mappings = {arm: summary for summary in case_summaries.get(key, []) for arm in summary['covers']}
            starts = {stage: sum(row[stage] for row in arms.values()) for stage in ('search', 'comparative')}
            counts.update({'search_optimizer_starts': starts['search'], 'comparative_optimizer_starts': starts['comparative'],
                           'registered_optimizer_updates': 5000 * sum(starts.values())})
            counts.update({'optimizer_starts_total': sum(starts.values()),
                'search_physical_slow_body_instances': sum(v['search'] * (4 if arm == 'joint_untied4' else 1) for arm, v in arms.items()),
                'comparative_physical_slow_body_instances': sum(v['comparative'] * (4 if arm == 'joint_untied4' else 1) for arm, v in arms.items()),
                'I4_banks_and_reused_member0_fits': (9 if backbone == 'SAGE' else 6) * split_count})
            banks_per_split = 48 if backbone == 'SAGE' else 21
            route_views_per_split = 156 if backbone == 'SAGE' else 66
            counts.update({'comparative_banks': banks_per_split * split_count,
                           'bank_exports': banks_per_split * split_count,
                           'scalar_calibrations': banks_per_split * split_count,
                           'registered_scalar_calibration_updates': 500 * banks_per_split * split_count,
                           'evaluation_jobs': banks_per_split * split_count})
            missing = sorted(set(arms) - set(mappings))
            row = {'dataset_id': dataset, 'backbone': backbone, 'prepared_split_count': split_count,
                   'optimizer_starts_by_stage': starts, 'registered_updates': 5000 * sum(starts.values()),
                   'unmeasured_arms': missing}
            if missing:
                row['estimates'] = None
                unknown.append(key)
                cells.append(row)
                continue
            compute = storage = io_min = io_max = 0.0
            for arm, stage_counts in arms.items():
                count = sum(stage_counts.values())
                profile = mappings[arm]
                compute += count * (5000 * profile['estimated_seconds_per_full_case_update'] + fit_overhead_seconds)
                # Shapes use the measured M4 source bank even for M1/I4 jobs.
                # Checkpoints include model plus two Adam moments. No optimizer
                # is instantiated, serialized, or timed by this arithmetic.
                payload = (3 * profile['parameter_bytes'] + profile['buffer_bytes']
                           + 8 * profile['parameter_tensors'] + 8 * MIB)
                retained = 4 * payload + 16 * MIB  # selected, final, recovery+embedded selected, traces
                storage += count * retained
                # 51 recovery writes (each bounded by 2 payloads), one final,
                # and 1..5000 strict-improvement selected overwrites. Initial
                # recovery is deliberately bounded as a full two-payload file.
                io_min += count * 104 * payload
                io_max += count * 5103 * payload
            endpoint_bytes = sum((r['role_counts']['cal'] + r['role_counts']['evaluate'])
                * (4 * receipt['model_graph']['classes'] * route_views_per_split + 8 * banks_per_split)
                for r in receipt['splits'])
            row['estimates'] = {'gpu_compute_seconds': compute, 'gpu_compute_days': compute / 86400,
                'retained_fit_checkpoint_and_trace_bytes': math.ceil(storage),
                'label_free_endpoint_tensor_bytes': endpoint_bytes,
                'checkpoint_bytes_written_planning_min': math.ceil(io_min),
                'checkpoint_bytes_written_planning_max': math.ceil(io_max),
                'checkpoint_io_seconds_planning_min': io_min / (checkpoint_io_mib_per_second * MIB),
                'checkpoint_io_seconds_planning_max': io_max / (checkpoint_io_mib_per_second * MIB)}
            known_compute += compute
            known_storage += storage + endpoint_bytes
            known_io_min += io_min
            known_io_max += io_max
            cells.append(row)
    return {'schema': 'native-resource-cost-projection-v1', 'counts': dict(counts), 'cells': cells,
        'all_cells_estimated': not unknown, 'unestimated_cells': unknown,
        'measured_subset_planning_totals': {'gpu_compute_seconds': known_compute,
            'gpu_compute_days': known_compute / 86400, 'retained_checkpoint_trace_endpoint_bytes': math.ceil(known_storage),
            'checkpoint_bytes_written_planning_min': math.ceil(known_io_min),
            'checkpoint_bytes_written_planning_max': math.ceil(known_io_max)},
        'complete_packet_planning_totals': None if unknown else {
            'gpu_compute_seconds': known_compute, 'gpu_compute_days': known_compute / 86400,
            'retained_checkpoint_trace_endpoint_bytes': math.ceil(known_storage),
            'checkpoint_io_seconds_planning_min': known_io_min / (checkpoint_io_mib_per_second * MIB),
            'checkpoint_io_seconds_planning_max': known_io_max / (checkpoint_io_mib_per_second * MIB)},
        'assumptions': {'updates_per_start': 5000, 'scalar_coefficients': scalar_mode,
            'maximum_shape_substituted_for_all_100_configurations': True,
            'M4_proxy_substituted_for_dominated_M1_jobs': True,
            'fit_setup_seconds_per_start': fit_overhead_seconds,
            'checkpoint_io_mib_per_second': checkpoint_io_mib_per_second,
            'checkpoint_io_throughput_measured': False,
            'retained_all_search_checkpoints': True, 'recovery_interval': 100,
            'active_prefixes': 0, 'failed_attempts_and_retries_included': False,
            'CAL_evaluation_compute_estimated': False,
            'projection_is_strict_time_or_disk_upper_bound': False,
            'case_timings_may_include_existing_concurrent_workload': True,
            'limitation': 'Optimizer kernels, target-based metric/selector CPU work, calibration/evaluation compute, '
                          'filesystem serialization, metadata/trace growth and concurrency are unmeasured. '
                          'Timing factors and payload padding are declared planning allowances, not qualification of those phases.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root')
    parser.add_argument('--custody', action='append')
    parser.add_argument('--native-model-source')
    parser.add_argument('--runtime-package')
    parser.add_argument('--output')
    parser.add_argument('--expected-datasets', type=int, default=8)
    parser.add_argument('--backbone', choices=BACKBONES, action='append')
    parser.add_argument('--worker-capacities', help='Planning-only JSON of worker capacities; no remote verification')
    parser.add_argument('--scalar-coefficients', choices=('neighbors', 'native_fixed'), default='neighbors')
    parser.add_argument('--warmup-iterations', type=int, default=1)
    parser.add_argument('--measured-iterations', type=int, default=3)
    parser.add_argument('--torch-threads', type=int, default=2)
    parser.add_argument('--timing-factor', type=float, default=1.5)
    parser.add_argument('--memory-factor', type=float, default=1.15)
    parser.add_argument('--memory-margin-mib', type=int, default=512)
    parser.add_argument('--fit-overhead-seconds', type=float, default=60)
    parser.add_argument('--checkpoint-io-mib-per-second', type=float, default=100)
    parser.add_argument('--process-sample-seconds', type=float, default=.25)
    parser.add_argument('--case-timeout-seconds', type=int, default=300)
    parser.add_argument('--worker-request', help=argparse.SUPPRESS)
    parser.add_argument('--worker-result', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_request or args.worker_result:
        if not (args.worker_request and args.worker_result):
            parser.error('Internal worker request/result must be paired')
        return worker(args.worker_request, args.worker_result)
    if not all((args.project_root, args.custody, args.native_model_source, args.runtime_package, args.output)):
        parser.error('project-root, custody, native-model-source, runtime-package and output are required')
    for field in ('expected_datasets', 'measured_iterations', 'torch_threads', 'memory_margin_mib', 'case_timeout_seconds'):
        positive(getattr(args, field), field, integer=True)
    for field in ('timing_factor', 'memory_factor', 'fit_overhead_seconds', 'checkpoint_io_mib_per_second', 'process_sample_seconds'):
        positive(getattr(args, field), field)
    if args.warmup_iterations < 0 or args.timing_factor < 1 or args.memory_factor < 1:
        parser.error('Nonnegative warmup and factors >=1 required')
    # Identity check before project metadata, directory creation, or data access.
    verify_allocation()
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_relative_to(ALLOCATION_ROOT):
        raise ValueError('Project root must stay in the allocation repository')
    output = local(root, args.output)
    if output.exists():
        raise ValueError('Preserve prior qualifications; output must be fresh')
    gpu = gpu_snapshot()
    concurrent_at_start = process_memory()
    native = local(root, args.native_model_source)
    runtime = local(root, args.runtime_package)
    if runtime.name == 'training_pipeline_v2_20261012' or native.parent.name == 'common_wrapper_native_source_20261010_v1':
        raise ValueError('Use frozen runtime/native copies outside the live source directories')
    node_bridge, node_losses = runtime / 'adapters/node.py', runtime / 'adapters/node_losses.py'
    bank_ast(node_bridge)  # Source-only validation before output creation.
    sources = [binding(root, p) for p in (native, node_bridge, node_losses, Path(__file__).resolve())]
    custodies, datasets, inputs = [], set(), {}
    for path in args.custody:
        receipt, representative, files = read_custody(root, path)
        if receipt['dataset_id'] in datasets:
            raise ValueError('One custody per dataset required')
        datasets.add(receipt['dataset_id'])
        custodies.append(receipt)
        inputs[receipt['dataset_id']] = (representative, files, local(root, path).parent)
    if len(custodies) != args.expected_datasets:
        raise ValueError('Dataset roster count differs from --expected-datasets')
    backbones = tuple(dict.fromkeys(args.backbone or BACKBONES))
    capacities = {'allocation/gpu0': gpu['free_bytes']}
    capacity_binding = None
    if args.worker_capacities:
        capacity_path = local(root, args.worker_capacities)
        capacity_binding = binding(root, capacity_path)
        capacity_document = json.loads(capacity_path.read_text())
        capacities = capacity_document.get('worker_capacities_cuda_bytes', capacity_document)
        if not isinstance(capacities, dict) or not capacities:
            raise ValueError('Nonempty explicit planning capacities required')
        for name, capacity in capacities.items():
            if name not in {'allocation/gpu0', 'server77/gpu0', 'server77/gpu1'}:
                raise ValueError('Unknown logical planning worker')
            positive(capacity, name + ' capacity', integer=True)
        if 'allocation/gpu0' not in capacities or capacities['allocation/gpu0'] > gpu['total_bytes']:
            raise ValueError('Allocation planning capacity exceeds verified physical memory')
        capacities = dict(capacities)
        capacities['allocation/gpu0'] = min(capacities['allocation/gpu0'], gpu['free_bytes'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'requests').mkdir()
    (output / 'cases').mkdir()
    (output / 'logs').mkdir()
    manifest = {'schema': 'native-resource-qualification-v1', 'status': 'running',
        'project_root': str(root), 'hostname': socket.gethostname(), 'gpu': gpu,
        'sources': sources, 'capacity_binding': capacity_binding,
        'other_cuda_processes_at_start': concurrent_at_start,
        'allocation_capacity_basis': 'initially observed free memory under current workload; rechecked per case',
        'worker_capacities_cuda_bytes': capacities, 'remote_capacity_live_verified': False,
        'arguments': {k: v for k, v in vars(args).items() if not k.startswith('worker_')},
        'scientific_launch_enabled': False, 'human_GO_required': True,
        'optimizer_objects': 0, 'optimizer_updates': 0, 'benchmark_fits': 0,
        'predictive_scores': 0, 'accuracy_computations': 0, 'heldout_target_files_opened': 0,
        'target_bearing_role_files_opened': 0,
        'cases': [], 'dataset_roster': sorted(datasets), 'backbones': list(backbones)}
    atomic_json(output / 'MANIFEST.json', manifest)
    summaries, profiles = {}, {}
    started = time.perf_counter()
    try:
        for custody in custodies:
            dataset = custody['dataset_id']
            split, files, data_dir = inputs[dataset]
            for backbone in backbones:
                key = dataset + '/' + backbone
                summaries[key] = []
                for case in BASE_CASES + (SAGE_CASES if backbone == 'SAGE' else ()):
                    name = dataset + '--' + backbone + '--' + case['case']
                    request_path, result_path = output / 'requests' / (name + '.json'), output / 'cases' / (name + '.json')
                    used = {str((data_dir / split['node_inputs']['graph']).relative_to(root))}
                    # Workers hash only graph, custody and frozen source files,
                    # never a target-bearing role file or endpoint file.
                    selected_files = [row for row in files if row['path'] in used or row is files[0]]
                    request = {**case, 'project_root': str(root), 'dataset_id': dataset, 'backbone': backbone,
                        'native_model_source': str(native.relative_to(root)), 'node_bridge': str(node_bridge.relative_to(root)),
                        'node_losses': str(node_losses.relative_to(root)), 'bindings': sources + selected_files,
                        'split': {'split_id': split['split_id'], 'role_counts': {r: split['role_counts'][r] for r in ('train', 'select')}},
                        'graph_input': str((data_dir / split['node_inputs']['graph']).relative_to(root)),
                        'model_graph': {k: custody['model_graph'][k] for k in ('nodes', 'features', 'classes', 'edges')},
                        'maximum_select_rows': max(r['role_counts']['select'] for r in custody['splits']),
                        'maximum_train_rows': max(r['role_counts']['train'] for r in custody['splits']),
                        'free_memory_budget_bytes': min(capacities['allocation/gpu0'], gpu_snapshot()['free_bytes']),
                        'context_headroom_bytes': 512 * MIB,
                        'shape': {'width': 512, 'depth': 5, 'ffn_multiplier': 1, 'heads': 8,
                                  'normalization': 'LayerNorm', 'dropout': .2, 'routes': 4},
                        'seed': 79001, 'warmup_iterations': args.warmup_iterations,
                        'measured_iterations': args.measured_iterations, 'torch_threads': args.torch_threads}
                    atomic_json(request_path, request)
                    print(canonical({'status': 'profiling', 'cell': key, 'case': case['case']}), flush=True)
                    result = run_case(request_path, result_path, output / 'logs' / (name + '.log'),
                        timeout=args.case_timeout_seconds, sample_seconds=args.process_sample_seconds,
                        root=root, threads=args.torch_threads)
                    summary = summarize_case(result, timing_factor=args.timing_factor,
                        memory_factor=args.memory_factor, margin_mib=args.memory_margin_mib)
                    if summary is not None:
                        summaries[key].append(summary)
                    manifest['cases'].append({'cell': key, 'case': case['case'], 'status': result['status'],
                        'oom': result.get('oom', False), 'receipt': str(result_path.relative_to(root)),
                        'receipt_sha256': digest(result_path), 'summary': summary})
                    atomic_json(output / 'MANIFEST.json', manifest)
                    print(canonical({'status': result['status'], 'cell': key, 'case': case['case'],
                                     'oom': result.get('oom', False)}), flush=True)
                required = 6 if backbone == 'SAGE' else 3
                if len(summaries[key]) == required:
                    maximum = max(s['estimated_seconds_per_full_case_update'] for s in summaries[key])
                    reservation = max(s['reservation_mib'] for s in summaries[key])
                    admitted = reservation * MIB <= capacities['allocation/gpu0']
                    profiles[key] = {'reservation_mib': reservation,
                        'wall_seconds': math.ceil(maximum * 5000 * 4 + args.fit_overhead_seconds),
                        # Existing compiler multiplies this by members. Retain
                        # the full M4 case time here; no unproved /4 amortization.
                        'estimated_seconds_per_update': maximum,
                        'allocation_memory_admitted': admitted,
                        'two_slot_allocation_memory_admitted': 2 * reservation * MIB <= capacities['allocation/gpu0'],
                        'two_slot_reservation_bytes': 2 * reservation * MIB,
                        'memory_admission_basis': 'separate measured M4 cases plus live 4P AdamW storage and headroom',
                        'full_training_runtime_qualified': False,
                        'case_profiles': summaries[key],
                        'evidence': [str((output / 'MANIFEST.json').relative_to(root)),
                            'fresh CUDA process for each required architecture/objective under recorded concurrent workload; max measured wall times x '
                            + str(args.timing_factor), 'memory headroom factor ' + str(args.memory_factor)
                            + ' plus ' + str(args.memory_margin_mib) + 'MiB; parameters unchanged; zero optimizer updates',
                            'compiler member multiplier retains additional conservative M4 time for all jobs']}
        resources = {'schema': 'native-resource-profiles-v1', 'profiles': profiles,
            'worker_capacities_cuda_bytes': capacities, 'remote_capacity_live_verified': False,
            'missing_or_failed_cells': sorted(dataset + '/' + backbone for dataset in datasets for backbone in backbones
                if dataset + '/' + backbone not in profiles),
            'unadmitted_cells': sorted(k for k, p in profiles.items() if not p['allocation_memory_admitted']),
            'one_slot_only_cells': sorted(k for k, p in profiles.items() if p['allocation_memory_admitted']
                and not p['two_slot_allocation_memory_admitted']),
            'two_slot_worst_pair_reservation_bytes': 2 * max((p['reservation_mib'] * MIB for p in profiles.values()), default=0),
            'two_slot_concurrent_execution_measured': False,
            'scientific_launch_enabled': False, 'human_GO_required': True}
        atomic_json(output / 'RESOURCES.json', resources)
        projection = cost_projection(custodies, backbones, summaries, scalar_mode=args.scalar_coefficients,
            fit_overhead_seconds=args.fit_overhead_seconds, checkpoint_io_mib_per_second=args.checkpoint_io_mib_per_second)
        atomic_json(output / 'COST_PROJECTION.json', projection)
        manifest.update(status='measured_memory_admitted' if not resources['missing_or_failed_cells']
            and not resources['unadmitted_cells'] else 'unqualified',
            wall_seconds=time.perf_counter() - started,
            resources=binding(root, output / 'RESOURCES.json'), cost_projection=binding(root, output / 'COST_PROJECTION.json'),
            qualification_scope='no-update backward/label-free SELECT/memory allowance; actual optimizer and endpoints unmeasured')
        atomic_json(output / 'MANIFEST.json', manifest)
        print(canonical({'status': manifest['status'], 'cells': len(profiles),
                         'expected_cells': len(datasets) * len(backbones), 'output': str(output),
                         'optimizer_updates': 0, 'benchmark_fits': 0, 'predictive_scores': 0}), flush=True)
        return 0 if manifest['status'] == 'measured_memory_admitted' else 2
    except BaseException as exc:
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                        error_type=type(exc).__name__, error=str(exc), wall_seconds=time.perf_counter() - started)
        atomic_json(output / 'MANIFEST.json', manifest)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
