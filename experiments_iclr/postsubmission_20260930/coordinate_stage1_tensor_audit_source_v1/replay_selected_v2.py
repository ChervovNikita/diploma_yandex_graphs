"""One bounded selected-state/logit integrity replay for all54 frozen Stage1 cells.

Prepared source only. Execution needs a new root-admitted exact request/allocation.
No optimizer application, training, test bundles, pickle fallback or tensor writes.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parents[1]
GPU_UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
HELPER = PHASE / 'coordinate_stage1_assessment_source_v2/validate_stage1_v2.py'
HELPER_SHA = 'e5f870f62a0d9635113517bc80a8ff1fe8b858b9b2c8d4e00b22854770fc4bab'
TOLERANCES = {'logits_atol': 1e-6, 'logits_rtol': 1e-5, 'NLL_atol': 2e-6,
              'NLL_rtol': 1e-5, 'fraction_atol': 1e-7, 'exact_argmax_required': True}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, 'Source module absent')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def independent_metrics(torch, logits, labels):
    """CPU float64 metrics, independent of runner metric helpers/reductions."""
    z, y = logits.to(device='cpu', dtype=torch.float64), labels.to(device='cpu')
    classes = z.shape[-1]
    log_member = z - torch.logsumexp(z, dim=-1, keepdim=True)
    pooled = torch.logsumexp(log_member, dim=0) - math.log(z.shape[0])
    mean_logit = z.mean(dim=0)
    sensitivity = mean_logit - torch.logsumexp(mean_logit, dim=-1, keepdim=True)

    def metrics(log_prob):
        prediction = log_prob.argmax(dim=-1)
        nll = float(-log_prob[torch.arange(y.numel()), y].mean().item())
        accuracy = float((prediction == y).to(torch.float64).mean().item())
        scores = []
        for c in range(classes):
            truth, predicted = y == c, prediction == c
            tp = int((truth & predicted).sum().item())
            denominator = int(truth.sum().item()) + int(predicted.sum().item())
            scores.append(2 * tp / denominator if denominator else 0.0)
        return {'nll': nll, 'accuracy': accuracy, 'macro_f1': math.fsum(scores) / classes}

    member_predictions = z.argmax(dim=-1)
    pairs = [{'member_i': left, 'member_j': right,
              'disagreement': float((member_predictions[left] != member_predictions[right]).to(torch.float64).mean().item())}
             for left in range(z.shape[0]) for right in range(left + 1, z.shape[0])]
    return {'primary_probability_pool': metrics(pooled),
            'same_checkpoint_mean_logit_sensitivity': metrics(sensitivity),
            'members': [metrics(log_member[i]) for i in range(z.shape[0])],
            'member_argmax_disagreement': {'pairs': pairs, 'unordered_pair_count': len(pairs),
                'mean_pairwise': math.fsum(p['disagreement'] for p in pairs) / len(pairs) if pairs else None,
                'nodes': y.numel(), 'single_member_value': None, 'argmax_ties': 'first_class_index'}}, pooled.argmax(dim=-1), member_predictions


def compare_metrics(actual, reported):
    differences = {}
    groups = [('primary_probability_pool', actual['primary_probability_pool'], reported['primary_probability_pool']),
              ('same_checkpoint_mean_logit_sensitivity', actual['same_checkpoint_mean_logit_sensitivity'], reported['same_checkpoint_mean_logit_sensitivity'])]
    require(len(actual['members']) == len(reported['members']), 'Member count differs')
    groups += [(f'member_{i}', a, b) for i, (a, b) in enumerate(zip(actual['members'], reported['members']))]
    for name, a, b in groups:
        for key in ('nll', 'accuracy', 'macro_f1'):
            require(type(b[key]) in (int, float) and math.isfinite(b[key]), 'Nonfinite reported metric')
            differences[name + '_' + key] = abs(a[key] - b[key])
            tolerance = (TOLERANCES['NLL_atol'] + TOLERANCES['NLL_rtol'] * abs(b[key])
                         if key == 'nll' else TOLERANCES['fraction_atol'])
            require(differences[name + '_' + key] <= tolerance,
                    f'Saved logits do not reproduce {name}_{key}: actual={a[key]}, reported={b[key]}, absolute_error={differences[name + "_" + key]}, tolerance={tolerance}')
    a, b = actual['member_argmax_disagreement'], reported['member_argmax_disagreement']
    for key in ('nodes', 'unordered_pair_count', 'single_member_value', 'argmax_ties'):
        require(a[key] == b[key], 'Disagreement scope differs')
    require(len(a['pairs']) == len(b['pairs']), 'Disagreement pair count differs')
    for left, right in zip(a['pairs'], b['pairs']):
        require(left['member_i'] == right['member_i'] and left['member_j'] == right['member_j'] and
                abs(left['disagreement'] - right['disagreement']) <= TOLERANCES['fraction_atol'],
                'Member disagreement differs')
    require(a['mean_pairwise'] == b['mean_pairwise'] if a['mean_pairwise'] is None else
            abs(a['mean_pairwise'] - b['mean_pairwise']) <= TOLERANCES['fraction_atol'],
            'Mean disagreement differs')
    return differences


def audit_case(torch, models, evidence, protocol, sources_binding, data, load, cell):
    dataset, arm, seed, recipe_name = (cell[k] for k in ('dataset', 'arm', 'seed', 'recipe'))
    manifest, graph, validation = data[dataset]
    recipe = protocol['recipes'][recipe_name]
    suffix = f'{dataset}__core0__seed{seed}__{arm}__{recipe_name}'
    root = evidence.relative(protocol['phases']['fit']['output_root']) + '/' + suffix
    artifacts = evidence.json(root + '/artifacts.json')['files']
    inventory = {item['path']: item for item in artifacts}
    require(len(inventory) == len(artifacts), 'Duplicate cell artifact entry')
    for name in ['cell_report.json', 'selected_state.pt', 'selected_deployment_state.pt', 'selected_validation_member_logits.pt']:
        indexed = evidence.entries[root + '/' + name]
        require(indexed['bytes'] == inventory[name]['bytes'] and indexed['sha256'] == inventory[name]['sha256'],
                'Selected artifact inventory/index differs: ' + name)
    report = evidence.json(root + '/cell_report.json')
    expected_provenance = {'protocol_sha256': canonical_protocol_sha, 'source_manifest_sha256': sources_binding['sha256'],
        'data_manifest_sha256': protocol['data_bindings'][dataset]['data_manifest_sha256'], 'cell': cell, 'recipe': recipe,
        'optimizer': recipe['optimizer'], 'optimizer_strategy': 'independent_member_optimizers_sum_member_ce_backward' if arm == 'untied' else 'one_optimizer_mean_member_ce_backward',
        'optimizer_count': 4 if arm == 'untied' else 1, 'graph_artifact_sha256': manifest['graph']['sha256'],
        'development_target_artifact_sha256': {role: manifest['splits']['core0'][role]['sha256'] for role in ('train', 'validation')},
        'mask_indices_sha256': {**manifest['splits']['core0']['mask_indices_sha256'], 'test_indices_only': manifest['test_indices_sha256']}}
    require(report['provenance'] == expected_provenance and report['mode'] == 'fit' and report['status'] == 'FIT_COMPLETE' and
            1000 <= report['completed_updates'] <= 2000 and 1 <= report['selected_epoch'] <= report['completed_updates'] and
            report['test_labels_read'] is False and report['test_scores_or_logits_written'] is False,
            'Report identity/provenance differs')
    checkpoint = load(root + '/selected_state.pt')
    deployment = load(root + '/selected_deployment_state.pt')
    saved = load(root + '/selected_validation_member_logits.pt')
    require(set(checkpoint) == {'epoch', 'model_state', 'optimizer_states', 'provenance'} and
            set(deployment) == {'model_state', 'factory_kwargs', 'epoch', 'protocol_sha256', 'source_manifest_sha256'} and
            set(saved) == {'epoch', 'node_ids', 'member_logits', 'provenance'}, 'Native bundle keys differ')
    require(checkpoint['provenance'] == saved['provenance'] == expected_provenance and
            checkpoint['epoch'] == deployment['epoch'] == saved['epoch'] == report['selected_epoch'] and
            deployment['protocol_sha256'] == canonical_protocol_sha and
            deployment['source_manifest_sha256'] == sources_binding['sha256'], 'Selected tensor provenance/epochs differ')
    require(len(checkpoint['optimizer_states']) == expected_provenance['optimizer_count'], 'Optimizer snapshot count differs')
    kwargs = {'arm': arm, 'input_dim': manifest['num_features'], 'hidden_dim': recipe['width'],
              'output_dim': manifest['num_classes'], 'seed': seed, 'permutation_seed': cell['permutation_seed'],
              'members': 4, 'num_layers': recipe['num_layers'], 'dropout': recipe['dropout'], **recipe['model_kwargs']}
    require(deployment['factory_kwargs'] == kwargs, 'Native deployment factory differs')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = models.build_model(**kwargs)
    expected_state, state = model.state_dict(), checkpoint['model_state']
    require(set(state) == set(expected_state) == set(deployment['model_state']), 'State keys differ')
    buffers = dict(model.named_buffers())
    for name, initial in expected_state.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
                value.shape == initial.shape and value.dtype == initial.dtype and torch.equal(value, deployment['model_state'][name]),
                'Checkpoint/deployment state shape/dtype/value differs: ' + name)
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Nonfinite selected state')
        if name in buffers:
            require(torch.equal(value, initial), 'A declared fixed buffer changed during training: ' + name)
    model.load_state_dict(state, strict=True)
    storage = models.storage_report(model)
    require(storage['total_model_tensor_bytes'] == cell['storage_gate']['expected_model_tensor_bytes'] and
            storage['index_buffer_bytes'] == cell['storage_gate']['expected_index_buffer_bytes'], 'Native model byte gates differ')
    ids, labels, logits = validation['indices'], validation['labels'], saved['member_logits']
    members = 1 if arm in ('single', 'gt_sep_single') else 4
    require(saved['node_ids'].dtype == torch.int64 and torch.equal(saved['node_ids'], ids) and
            logits.dtype == torch.float32 and logits.device.type == 'cpu' and
            tuple(logits.shape) == (members, ids.numel(), manifest['num_classes']) and bool(torch.isfinite(logits).all()),
            'Saved logits dtype/shape/order/finite check differs')
    metrics, saved_prediction, saved_members = independent_metrics(torch, logits, labels)
    differences = compare_metrics(metrics, report['selected_validation'])
    del checkpoint, deployment, state, expected_state, buffers
    torch.cuda.reset_peak_memory_stats()
    model = model.to('cuda:0').eval()
    x, edges, gpu_ids = graph['x'].to('cuda:0'), graph['edge_index'].to('cuda:0'), ids.to('cuda:0')
    torch.cuda.synchronize()
    begun = time.monotonic()
    with torch.no_grad():
        full = model(SimpleNamespace(edge_index=edges), x)
        require(tuple(full.shape) == (members, manifest['num_nodes'], manifest['num_classes']) and
                bool(torch.isfinite(full).all()), 'Independent forward shape/finite check failed')
        replay = full.index_select(1, gpu_ids).detach().cpu()
    torch.cuda.synchronize()
    forward_seconds = time.monotonic() - begun
    error = (replay - logits).abs()
    require(bool(torch.allclose(replay, logits, atol=TOLERANCES['logits_atol'], rtol=TOLERANCES['logits_rtol'])),
            'Independent selected-state forward disagrees with saved logits; max_absolute_error=' + str(float(error.max().item())))
    replay_metrics, replay_prediction, replay_members = independent_metrics(torch, replay, labels)
    require(torch.equal(saved_prediction, replay_prediction) and torch.equal(saved_members, replay_members),
            'Independent forward changed pooled/member argmax predictions; pooled_mismatches=' +
            str(int((saved_prediction != replay_prediction).sum().item())) + '; member_mismatches=' +
            str(int((saved_members != replay_members).sum().item())))
    compare_metrics(replay_metrics, report['selected_validation'])
    peak = {'allocated_bytes': torch.cuda.max_memory_allocated(), 'reserved_bytes': torch.cuda.max_memory_reserved()}
    require(peak['allocated_bytes'] <= 8 * (1 << 30) and peak['reserved_bytes'] <= 8 * (1 << 30), 'Replay GPU working bound exceeded')
    return {'cell': cell, 'status': 'TENSOR_REPLAY_VERIFIED', 'selected_epoch': report['selected_epoch'],
            'saved_logit_metrics_CPU64_integrity_only': metrics, 'report_metric_absolute_differences': differences,
            'forward_member_logits_max_absolute_error': float(error.max().item()),
            'pooled_and_member_argmax_match': True, 'forward_gather_CPU_copy_seconds': forward_seconds,
            'gpu_peak_memory': peak, 'original_FP32_report_gate_metrics_preserved': True}


def main():
    global canonical_protocol_sha
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--supervisor', type=Path, required=True)
    args = parser.parse_args()
    require(sha(HELPER) == HELPER_SHA, 'Frozen stdlib guard source differs')
    guard = module(HELPER, 'coordinate_replay_frozen_stdlib_guards')
    request_path = guard.safe(PHASE, str(args.request.relative_to(PHASE)))
    request, request_sha = guard.read(request_path), sha(request_path)
    require(request['schema'] == 'coordinate-stage1-selected-tensor-replay-request-v1' and
            request['root_admitted'] is True and request['test_data_or_labels_admitted'] is False and
            request['training_or_optimizer_steps_admitted'] is False and request['tolerances'] == TOLERANCES and
            request['comparative_tensor_or_metric_outcomes_inspected_before_replay_rule_adoption'] is False,
            'No exact root replay admission')
    require(str(PHASE) == request['phase_root'] and Path.cwd().resolve() == REPO and
            Path(os.path.abspath(sys.executable)) == REPO / '.venv/bin/python' and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('PYTHONDONTWRITEBYTECODE') == '1' and
            os.environ.get('GNNM_PHASE_ROOT') == str(PHASE), 'Require exact remote single-GPU wrapper/runtime')
    index_path = guard.safe(PHASE, str(Path(request['evidence_index']['path']).relative_to(PHASE)))
    require(sha(index_path) == request['evidence_index']['sha256'], 'Immutable evidence index differs')
    evidence = guard.Evidence(PHASE, str(PHASE), guard.read(index_path))
    for binding in request['protected_files'] + [request[k] for k in ('protocol', 'sources', 'stdlib_specification', 'stdlib_adoption', 'allocation')]:
        evidence.binding(binding)
    require(request['stdlib_specification']['sha256'] == '6da6a44bbc2b55118296b8e0de007330fd5d83983b57fb3d17fdd093e0cef44d' and
            request['stdlib_adoption']['sha256'] == '0f3f6c4e8dac4f695e8291d3b9eb65897f3984f9f8c09879dc46cb42d8bfad17',
            'Only the frozen adopted v2 analysis is admitted')
    protected = {binding['path']: binding['sha256'] for binding in request['protected_files']}
    require(len(protected) == len(request['protected_files']), 'Duplicate protected replay source')
    for path in [Path(__file__).resolve(), HELPER, *[PHASE / ('protocols/' + name) for name in
                 ['bounded_run_v1.py', 'run_authorized_v2.py', 'run_logged.py', 'repo_env.sh', 'AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json']]]:
        require(str(path) in protected and protected[str(path)] == sha(path), 'Required protected replay source absent or changed')
    protocol = guard.read(evidence.binding(request['protocol']))
    canonical_protocol_sha = request['protocol']['sha256']
    require(canonical_protocol_sha == '7817d3b818ee9deec11dae6bb2867ff91d150bcea350f5428d23c95bee241444' and
            len(protocol['cells']) == 54 and all(c['split'] == 'core0' for c in protocol['cells']), 'Only exact Stage1 packet admitted')
    sources = guard.read(evidence.binding(request['sources']))
    require(request['sources']['sha256'] == protocol['source_manifest_sha256'], 'Frozen source manifest differs')
    for item in sources['files']:
        evidence.binding({'path': item['path'], 'sha256': item['sha256']})
    bound_path = Path(os.environ['GNNM_BOUND_START_JSON'])
    bound = guard.read(bound_path)
    own = [str(REPO / '.venv/bin/python'), str(Path(__file__).resolve()), '--request', str(request_path), '--supervisor', str(args.supervisor)]
    require(bound['child_argv'] == own and bound['root_request']['sha256'] == request_sha and
            bound['research_script_sha256'] == sha(Path(__file__)) and bound['whole_cap_seconds'] == 900 and
            bound['normal_child_budget_seconds'] == 895 and bound['inner_supervisor_directory'] == str(args.supervisor),
            'Whole replay supervisor binding differs')
    supervisor = guard.read(args.supervisor / 'command.json')
    require(supervisor['argv'] == own and supervisor['shell'] is False, 'Inner exact replay argv differs')
    require(os.environ.get('GNNM_SSH_DESTINATION') == 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
            'Replay route differs from the corrected authorization')
    row = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu',
                          '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True, timeout=10).stdout.strip().splitlines()
    require(len(row) == 1, 'Only one physical GPU admitted')
    gpu = [part.strip() for part in row[0].split(',')]
    require(gpu[0] == '0' and gpu[1] == GPU_UUID and gpu[2] == 'NVIDIA A100-SXM4-80GB' and
            int(gpu[3]) <= 100 and int(gpu[4]) == 81920 and int(gpu[5]) == 0, 'Exact authorized idle GPU differs')
    output = guard.safe(PHASE, evidence.relative(request['output']))
    require(not output.exists() and output.parent == PHASE / 'coordinate_ensemble_execution_root_v1' and
            output.name == 'tensor_replay_selected_v1_run01', 'Replay output must be the new separate receipt root')
    output.mkdir(parents=True)
    write(output / 'START.json', {'request_sha256': request_sha, 'source_sha256': sha(Path(__file__)),
          'UTC': datetime.now(timezone.utc).isoformat(), 'cases': 54, 'tensor_loads_weights_only': True,
          'test_data_or_labels_read': False, 'training_steps': 0, 'tensor_writes': 0, 'gpu_metadata': gpu})
    lock_path = guard.safe(PHASE, evidence.relative(protocol['runtime']['gpu_lock_file']))
    lock = lock_path.open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = protocol['runtime']['cublas_workspace_config']
    import torch
    import torch_geometric
    runtime = protocol['runtime']
    require(torch.__version__ == runtime['torch_version'] and torch_geometric.__version__ == runtime['torch_geometric_version'],
            'Replay framework differs from the frozen fit')
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(runtime['cpu_threads'])
    torch.use_deterministic_algorithms(runtime['deterministic_algorithms'])
    torch.backends.cuda.matmul.allow_tf32 = runtime['allow_tf32']
    torch.backends.cudnn.allow_tf32 = runtime['allow_tf32']
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.1)
    allowed = set()
    for cell in protocol['cells']:
        suffix = f'{cell["dataset"]}__core0__seed{cell["seed"]}__{cell["arm"]}__{cell["recipe"]}'
        root = evidence.relative(protocol['phases']['fit']['output_root']) + '/' + suffix
        allowed.update(root + '/' + name for name in ['selected_state.pt', 'selected_deployment_state.pt', 'selected_validation_member_logits.pt'])
    manifests = {}
    for dataset in ['AmazonPhoto', 'CoauthorCS']:
        root = protocol['phases']['acquire']['public_root'] + '/' + dataset
        manifest = evidence.remote_json(root + '/data_manifest.json')
        require(sha(evidence.file(evidence.relative(root + '/data_manifest.json'))) == protocol['data_bindings'][dataset]['data_manifest_sha256'],
                'Public manifest differs')
        manifests[dataset] = manifest
        for item in [manifest['graph'], manifest['splits']['core0']['validation']]:
            indexed = evidence.entries[evidence.relative(root + '/' + item['path'])]
            require(indexed['sha256'] == item['sha256'] and indexed['bytes'] == item['bytes'],
                    'Public tensor index/manifest hash or length differs')
        allowed.update(evidence.relative(root + '/' + item['path']) for item in [manifest['graph'], manifest['splits']['core0']['validation']])
    original_load = torch.load

    def load(relative):
        require(relative in allowed, 'Tensor input outside exact selected/development allowlist')
        path = guard.safe(PHASE, relative)
        indexed = evidence.entries[relative]
        contents = path.read_bytes()
        require(len(contents) == indexed['bytes'] and hashlib.sha256(contents).hexdigest() == indexed['sha256'],
                'Native input hash/length differs before load')
        evidence.checked[relative] = indexed
        return original_load(io.BytesIO(contents), map_location='cpu', weights_only=True)

    def reject(*args_, **kwargs_):
        raise ValueError('Direct torch.load/save outside the digest-verified replay loader is forbidden')

    torch.load, torch.save = reject, reject
    models = module(evidence.binding({'path': sources['model_entry'], 'sha256': next(i['sha256'] for i in sources['files'] if i['path'] == sources['model_entry'])}), 'coordinate_replay_exact_models')
    data = {}
    for dataset, manifest in manifests.items():
        root = evidence.relative(manifest['public_root'])
        graph, validation = load(root + '/' + manifest['graph']['path']), load(root + '/' + manifest['splits']['core0']['validation']['path'])
        require(set(graph) == {'x', 'edge_index'} and set(validation) == {'indices', 'labels'}, 'Public bundle keys differ')
        x, edges, ids, labels = graph['x'], graph['edge_index'], validation['indices'], validation['labels']
        require(x.dtype == torch.float32 and tuple(x.shape) == (manifest['num_nodes'], manifest['num_features']) and
                bool(torch.isfinite(x).all()) and edges.dtype == torch.int64 and tuple(edges.shape) == (2, manifest['num_edges']) and
                int(edges.min()) >= 0 and int(edges.max()) < manifest['num_nodes'], 'Public graph shape/dtype differs')
        require(ids.dtype == labels.dtype == torch.int64 and ids.ndim == labels.ndim == 1 and ids.numel() == labels.numel() and
                ids.unique().numel() == ids.numel() and int(ids.min()) >= 0 and int(ids.max()) < manifest['num_nodes'] and
                int(labels.min()) >= 0 and int(labels.max()) < manifest['num_classes'] and
                canonical(ids.tolist()) == manifest['splits']['core0']['mask_indices_sha256']['validation'], 'Validation bundle identity differs')
        data[dataset] = manifest, graph, validation
    begun, records = time.monotonic(), []
    for cell in protocol['cells']:
        try:
            require(sha(request_path) == request_sha, 'Replay request changed')
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(bound['start_UTC'])).total_seconds()
            require(age < 860, 'Insufficient whole replay runway')
            record = audit_case(torch, models, evidence, protocol, request['sources'], data, load, cell)
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 8 * (1 << 30), 'CPU RSS working bound exceeded')
        except Exception as exc:
            record = {'cell': cell, 'status': 'INCONCLUSIVE_TENSOR_REPLAY', 'reason': str(exc), 'type': type(exc).__name__}
        records.append(record)
        write(output / f'case{len(records):02d}.json', record)
        torch.cuda.empty_cache()
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()
    for binding in request['protected_files'] + [request['protocol'], request['sources']]:
        evidence.binding(binding)
    complete = len(records) == 54 and all(r['status'] == 'TENSOR_REPLAY_VERIFIED' for r in records) and sha(request_path) == request_sha
    write(output / 'TERMINAL.json', {'complete': complete, 'records': records, 'cases': len(records),
          'elapsed_seconds': time.monotonic() - begun, 'test_data_or_labels_read': False,
          'training_steps': 0, 'tensor_writes': 0, 'tolerances': TOLERANCES,
          'CPU64_metrics_are_integrity_checks_not_gate_replacements': True,
          'failed_cases_are_inconclusive_no_tolerance_widening': True})
    raise SystemExit(0 if complete else 1)


if __name__ == '__main__':
    main()
