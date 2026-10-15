"""Prepared selected-state link inference; importing executes no model or TEST read."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[2]
SOURCE_BINDINGS_SHA256 = 'f08e52380d5e89a093f7b79707e6fd66316e20fd8f70b9838d55b8d8206aae1c'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inside(path, root):
    root, path = Path(root).resolve(), Path(path)
    path = (path if path.is_absolute() else root/path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Serving path escapes the repository root: '+str(path))
    return path


def receipt(path):
    h, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block); size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def backend_sources():
    return {name: receipt(HERE/name)['sha256'] for name in ('backend.py', 'SOURCE_BINDINGS.json')}


def _source_bindings(root):
    path = inside(HERE/'SOURCE_BINDINGS.json', root)
    if receipt(path)['sha256'] != SOURCE_BINDINGS_SHA256:
        raise ValueError('Prepared serving source bindings changed')
    bindings = json.loads(path.read_text())
    for row in bindings['files']:
        actual = receipt(inside(PHASE/row['path_relative_to_phase'], root))
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('Frozen serving dependency changed: '+row['path_relative_to_phase'])
    return bindings


def _descriptor(row, root):
    if not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}:
        raise ValueError('Every input requires an explicit path/bytes/SHA256 descriptor')
    inside(row['path'], root)
    if type(row['bytes']) is not int or row['bytes'] <= 0 or not isinstance(row['sha256'], str) or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
        raise ValueError('Invalid serving input receipt')


def _verified(row, root):
    path = inside(row['path'], root)
    actual = receipt(path)
    if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
        raise ValueError('Serving input bytes changed: '+str(path))
    return path


def validate_request(request, root):
    """Metadata/source-only preflight; never opens a graph/query/checkpoint payload."""
    bindings = _source_bindings(root)
    if request.get('schema') != 'own-selected-link-serving-v1' or request.get('role') not in ('VALID', 'TEST'):
        raise ValueError('Explicit selected-link serving schema and role required')
    role, family = request['role'], request.get('native_family')
    cells = {row['architecture']: row for row in bindings['cells']}
    if family not in cells or request.get('dataset') != 'ogbl-collab' or request.get('recipe_id') != cells[family]['id']:
        raise ValueError('Only the three frozen Collab link cells are supported')
    if request.get('backend_sources') != backend_sources():
        raise ValueError('The request must pin the exact prepared inference sources')
    expected_files = {'graph', 'train', 'valid'} | ({'test'} if role == 'TEST' else set())
    if set(request.get('data_files', {})) != expected_files:
        raise ValueError('VALID must exclude TEST; TEST requires its separately pinned official query file')
    for row in request['data_files'].values(): _descriptor(row, root)
    population = request.get('query_population', {})
    if set(population) != {'positive', 'negative', 'provider_evidence'} or any(type(population[k]) is not int or population[k] <= 0 for k in ('positive', 'negative')):
        raise ValueError('Explicit complete official query counts and provider evidence are required')
    _descriptor(population['provider_evidence'], root)
    if role == 'VALID' and (population['positive'], population['negative']) != (60084, 100000):
        raise ValueError('Qualification requires complete official VALID60084/100000')
    if request.get('evaluator') != {'name': 'ogbl-collab', 'metric': 'hits@50', 'K': 50}:
        raise ValueError('Native complete-population Collab Hits@50 evaluator required')
    topology = ('weighted_year2007_TRAIN_plus_weighted_VALID' if role == 'TEST' else 'weighted_year2007_TRAIN') if family == 'BUDDY' else ('unweighted_TRAIN_plus_VALID' if role == 'TEST' else 'unweighted_TRAIN')
    if request.get('topology') != topology:
        raise ValueError('Serving topology differs from the pinned native role')
    if request.get('inference_batch_size') != (1000000 if family == 'BUDDY' else 131072):
        raise ValueError('Retain the source inference batch size')
    rule = request.get('serving_rule')
    if rule not in ('own_selected_raw_logits', 'mean_four_individually_selected_native_logits'):
        raise ValueError('Only the frozen raw-logit serving rules are supported')
    selected = request.get('selected', [])
    if len(selected) != (4 if rule.startswith('mean_four') else 1):
        raise ValueError('One selected state per own native/shared fit; I4 requires four own states')
    for row in selected:
        if set(row) != {'job', 'checkpoint', 'runtime_binding'} or not isinstance(row['runtime_binding'], dict):
            raise ValueError('Each selection must bind JOB.json, selected.pt and its exact runtime binding')
        _descriptor(row['job'], root); _descriptor(row['checkpoint'], root)
        if Path(row['checkpoint']['path']).name != 'selected.pt':
            raise ValueError('Final/current/recovery weights cannot replace an own selected checkpoint')
    for key in ('rtol', 'atol'):
        if type(request.get(key)) not in (float, int) or not 0 <= request[key] < 1:
            raise ValueError('Declare finite engineering comparison tolerances before qualification')
    if family == 'BUDDY':
        if request.get('buddy_cache_mode') != 'fresh_original_HashDataset' or 'cache_directory' not in request:
            raise ValueError('BUDDY requires a fresh role-specific original HashDataset cache; TRAIN caches cannot serve TEST')
        inside(request['cache_directory'], root)
    if role == 'TEST': _descriptor(request.get('VALID_qualification'), root)
    return {'source_bound': True, 'role': role, 'TEST_payload_opened': False,
            'request_sha256': digest(request), 'native_family': family}


def _qualification_binding(request):
    return {'native_family': request['native_family'], 'recipe_id': request['recipe_id'],
            'serving_rule': request['serving_rule'], 'selected': request['selected'],
            'data_files': {key: request['data_files'][key] for key in ('graph', 'train', 'valid')},
            'backend_sources': request['backend_sources'], 'rtol': request['rtol'], 'atol': request['atol']}


def authorize_TEST(request, approval, root):
    """No TEST file is opened while checking this exact-request human GO envelope."""
    if request.get('role') != 'TEST' or not isinstance(approval, dict) or approval.get('action') != 'score_final_selected_TEST' or approval.get('approved_by') != 'human' or approval.get('authorized') is not True or approval.get('selection_frozen') is not True or approval.get('request_sha256') != digest(request) or not approval.get('authorization_reference'):
        raise PermissionError('Explicit human GO for this exact frozen selected-state TEST request is required')
    report = json.loads(_verified(request['VALID_qualification'], root).read_text())
    if report.get('schema') != 'selected-link-VALID-qualification-v1' or report.get('passed') is not True or report.get('role') != 'VALID' or report.get('TEST_accessed') is not False or report.get('binding') != _qualification_binding(request):
        raise ValueError('Matching actual VALID backend qualification must precede TEST access')
    return {'request_sha256': digest(request), 'authorization_reference': approval['authorization_reference'],
            'qualification_sha256': request['VALID_qualification']['sha256']}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _definition(path, name, namespace):
    raw = Path(path).read_text(); tree = ast.parse(raw, filename=str(path))
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    scope = dict(namespace, __name__='_prepared_exact_link_source', __file__=str(path))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def _open_selected(torch, request, root, device):
    integration = _load(PHASE/'benchmark_expansion_20261015_v1/link_native_integration_v1/session.py', '_final_link_native_integration')
    policy_module = _load(PHASE/'benchmark_expansion_20261015_v1/link_policy_v1/policy.py', '_final_link_policy')
    adapters, jobs, checkpoints = [], [], []
    try:
        for row in request['selected']:
            job = json.loads(_verified(row['job'], root).read_text())
            if job['recipe']['architecture'] != request['native_family'] or job['recipe']['id'] != request['recipe_id'] or job['family'] != 'link' or job['TEST_access'] is not False:
                raise ValueError('Selected job differs from the frozen serving cell')
            if job.get('device', 'cuda:0') != device:
                raise ValueError('Device migration requires a separately declared and qualified serving request')
            if inside(job['callback_module'], root) != Path(integration.__file__).resolve() or job.get('callback_sha256') != receipt(integration.__file__)['sha256']:
                raise ValueError('Selected job must pin the frozen native integration entrypoint')
            for key in ('graph', 'train', 'valid'):
                if inside(job['recipe']['data_files'][key], root) != inside(request['data_files'][key]['path'], root):
                    raise ValueError('Selected job and serving data binding differ')
            checkpoint = torch.load(_verified(row['checkpoint'], root), map_location='cpu', weights_only=False)
            if checkpoint.get('schema') != 'native-full-epoch-recovery-v1' or checkpoint.get('job_hash') != digest(job) or checkpoint.get('runtime_binding') != row['runtime_binding'] or checkpoint.get('TEST_accessed') is not False:
                raise ValueError('Invalid own selected checkpoint/runtime binding')
            selected = checkpoint['selector']['selected_epoch']
            if type(selected) is not int or selected <= 0 or selected != checkpoint['selector']['completed_epoch']:
                raise ValueError('Selected checkpoint must be its own positive completed winning epoch')
            if checkpoint['validation'].get('role') != 'VALID' or checkpoint['validation'].get('complete_source_VALID') is not True or checkpoint['validation'].get('records') != 160084:
                raise ValueError('Only a complete official VALID-selected checkpoint may serve')
            policy = policy_module.create_policy(job, root)
            selector = policy_module.create_selector(policy); selector.load_state(checkpoint['selector'])
            integration._seed(torch, job['seed'])
            adapter = integration.create_adapter(torch, job, policy, root)
            adapters.append(adapter)
            adapter.load_state(checkpoint['adapter'])
            if adapter.completed_epochs != selected:
                raise ValueError('Selected native epoch and selector clock differ')
            integration._restore_rng(torch, checkpoint['global_rng'])
            jobs.append(job); checkpoints.append(checkpoint)
        if request['serving_rule'].startswith('mean_four'):
            first = jobs[0]
            if [job.get('body') for job in jobs] != [0, 1, 2, 3] or any(job['procedure'] != 'native_single' or job['cohort_id'] != first['cohort_id'] for job in jobs):
                raise ValueError('I4 requires four distinct native bodies in one declared cohort')
            cell = next(row for row in _source_bindings(root)['cells'] if row['architecture'] == request['native_family'])
            offsets = cell['native_seed_offsets']
            if any(job['recipe'].get('native_seed_offsets') != offsets for job in jobs):
                raise ValueError('Independent offsets must match the frozen native link cell')
            if [job['seed']-first['seed'] for job in jobs] != offsets or len(set(job['seed'] for job in jobs)) != 4:
                raise ValueError('I4 selected members do not match their frozen seed offsets')
            stores = [{(str(p.device), p.untyped_storage().data_ptr()) for p in a.model.parameters()} for a in adapters]
            if any(stores[i] & stores[j] for i in range(4) for j in range(i)):
                raise ValueError('Independent selected bodies alias Parameter storage')
        return integration, adapters, jobs, checkpoints
    except BaseException:
        for adapter in adapters: adapter.close()
        raise


def _queries(torch, request, root):
    role = request['role'].lower()
    split = torch.load(_verified(request['data_files'][role], root), map_location='cpu', weights_only=False)
    if not isinstance(split, dict) or not {'edge', 'edge_neg'} <= set(split):
        raise ValueError('Official Collab query schema edge/edge_neg is required')
    for name, count in (('edge', request['query_population']['positive']), ('edge_neg', request['query_population']['negative'])):
        value = split[name]
        if not torch.is_tensor(value) or value.dtype != torch.long or tuple(value.shape) != (count, 2) or not bool(((value >= 0) & (value < 235868)).all()):
            raise ValueError('Complete original query count/endpoints/dtype differ from the provider binding')
    return split


def _ncn_predict(torch, session, split, role):
    n = session.native; n.model.eval()
    adj = n.adj
    if role == 'TEST':
        # Exact native loaddataset ordering: concatenate original graph and raw
        # VALID edges, coalesce, then symmetrize. Edge weights are absent.
        edges = torch.cat((n.edge_index, n.valid['edge'].to(n.device).t()), dim=-1)
        adj = n.SparseTensor.from_edge_index(edges, sparse_sizes=(235868, 235868)).coalesce().to_symmetric()
    rows = []
    with torch.no_grad():
        for member in range(n.members):
            with n._route(member, False):
                h = n.encoder(n.x, adj); parts = []
                for key in ('edge', 'edge_neg'):
                    pairs = split[key].to(n.device)
                    parts.extend(n.predictor(h, adj, pairs[p].t()).reshape(-1).cpu()
                                 for p in n.utils.PermIterator(n.device, len(pairs), n.test_batch_size, False))
                rows.append(torch.cat(parts))
    member = torch.stack(rows)
    return member.mean(0), member, {'encoder_forwards': n.members, 'topology': role, 'source_test_batch_size': 131072}


def _buddy_dataset(torch, integration, session, job, split, request, root):
    import os
    import time
    import scipy.sparse as ssp
    import torch_sparse
    from torch_geometric.data import Dataset
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import to_undirected
    n = session.native
    callback = integration._native_factory(n.recipe)
    graph, train, valid, _ = callback.load_collab(torch, n.recipe, inside(job['data_root'], root))
    loader = callback._local_module('source_loader')
    _, _, Hashes, filter_year, _, _, _ = loader.load_buddy(torch)
    graph, _ = filter_year(graph, {'train': train}, 2007)
    if request['role'] == 'TEST':
        if 'weight' not in valid or len(valid['weight']) != len(valid['edge']):
            raise ValueError('BUDDY TEST topology needs original official VALID weights')
        vei, vw = to_undirected(valid['edge'].t(), valid['weight'])
        graph.edge_index = torch.cat((graph.edge_index, vei), dim=1)
        graph.edge_weight = torch.cat((graph.edge_weight, vw.unsqueeze(-1)), dim=0)
    args = copy.deepcopy(n.args)
    args.load_features = args.load_hashes = False
    args.cache_subgraph_features = args.use_struct_feature = True
    args.dataset_name = 'ogbl-collab'; args.num_negs = 1
    args.eval_batch_size = 1000000; args.dynamic_train = args.dynamic_val = args.dynamic_test = False
    args.wandb = False
    cache = inside(request['cache_directory'], root)
    if cache.exists():
        raise ValueError('A fresh role-specific BUDDY cache directory is required; never reuse TRAIN or stale TEST context')
    cache.mkdir(parents=True)
    cls = _definition(PHASE/'reference_wrapper_redesign_20261013_v1/native_sources/buddy/buddy_src_datasets_elph.py',
        'HashDataset', {'torch': torch, 'Dataset': Dataset, 'ElphHashes': Hashes,
            'coalesce': torch_sparse.coalesce, 'to_undirected': to_undirected, 'ssp': ssp,
            'torch_sparse': torch_sparse, 'gcn_norm': gcn_norm, 'os': os, 'time': time.time})
    dataset = cls(str(cache/'elph_'), request['role'].lower(), graph, split['edge'], split['edge_neg'], args,
                  use_coalesce=True, directed=False)
    if tuple(dataset.x.shape) != (235868, 128) or tuple(dataset.subgraph_features.shape) != (len(dataset.links), 8):
        raise ValueError('Original BUDDY role context has an unexpected full-native shape')
    provenance = {'schema': 'native-role-specific-BUDDY-context-v1', 'role': request['role'],
        'topology': request['topology'], 'request_sha256': digest(request),
        'query_population': request['query_population'], 'source_bindings_sha256': SOURCE_BINDINGS_SHA256,
        'backend_sources': backend_sources(), 'data_files': request['data_files'],
        'source_class': 'unchanged HashDataset AST', 'source_cache_files': [
            {'path': str(path), **receipt(path)} for path in sorted(cache.rglob('*.pt'))]}
    (cache/'CONTEXT_BINDING.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return dataset, args, provenance


def _buddy_predict(torch, session, dataset, args):
    import time
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    samples = _definition(HERE/'primary/buddy_src_runners_inference.py', 'get_split_samples', {})
    predict = _definition(HERE/'primary/buddy_src_runners_inference.py', 'get_buddy_preds',
        {'torch': torch, 'DataLoader': DataLoader, 'tqdm': tqdm, 'time': time, 'get_split_samples': samples})
    n = session.native; n.model.eval(); rows = []
    with torch.no_grad():
        for member in range(n.members):
            with n._route(member, False):
                _, _, values, labels = predict(n.model, SimpleNamespace(dataset=dataset), n.device, args, split=dataset.split)
                rows.append(values)
    member = torch.stack(rows)
    return member.mean(0), member, {'native_BUDDY_forwards': n.members*((len(dataset.links)+999999)//1000000),
                                  'source_combined_positive_negative_query_batches': True}


def _run(torch, request, root, device, qualify):
    for key in ('graph', 'train', 'valid'): _verified(request['data_files'][key], root)
    provider = json.loads(_verified(request['query_population']['provider_evidence'], root).read_text())
    if provider.get('schema') != 'official-collab-query-population-v1' or provider.get('dataset') != 'ogbl-collab' or provider.get('role') != request['role'] or provider.get('complete_official_split') is not True or provider.get('query_file') != request['data_files'][request['role'].lower()] or (provider.get('positive_records'), provider.get('negative_records')) != (request['query_population']['positive'], request['query_population']['negative']) or not provider.get('provider_release'):
        raise ValueError('Authenticated provider evidence for the entire official query population is required')
    integration = _load(PHASE/'benchmark_expansion_20261015_v1/link_native_integration_v1/session.py', '_final_link_rng')
    outside = integration._capture_rng(torch)
    adapters = []
    try:
        integration, adapters, jobs, checkpoints = _open_selected(torch, request, root, device)
        split = _queries(torch, request, root)  # TEST reaches this only after human GO and VALID qualification.
        outputs, member_outputs, parity, work = [], [], [], []
        buddy_context = None
        if request['native_family'] == 'BUDDY':
            buddy_context = _buddy_dataset(torch, integration, adapters[0].session, jobs[0], split, request, root)
        for adapter, job, checkpoint in zip(adapters, jobs, checkpoints):
            integration._restore_rng(torch, checkpoint['global_rng'])
            before = adapter.state() if qualify else None
            source_reference = adapter.validate()['outputs'] if qualify else None
            if qualify: adapter.load_state(before)
            integration._restore_rng(torch, checkpoint['global_rng'])
            values, members, counts = _buddy_predict(torch, adapter.session, buddy_context[0], buddy_context[1]) if buddy_context else _ncn_predict(torch, adapter.session, split, request['role'])
            if not bool(torch.isfinite(values).all()): raise FloatingPointError('Nonfinite selected raw logits')
            if qualify:
                passed = torch.allclose(source_reference, values, rtol=request['rtol'], atol=request['atol'])
                parity.append({'passed': bool(passed), 'max_absolute_difference': float((source_reference-values).abs().max()),
                               'selected_epoch': adapter.completed_epochs, 'optimizer_maps_executed': 0})
                if not passed: raise ValueError('Original prepared VALID inference differs from the frozen selected native VALID port')
            outputs.append(values); member_outputs.append(members); work.append(counts)
        logits = torch.stack(outputs).mean(0)
        positive = request['query_population']['positive']
        from ogb.linkproppred import Evaluator
        evaluator = Evaluator(name='ogbl-collab'); evaluator.K = 50
        metrics = evaluator.eval({'y_pred_pos': logits[:positive], 'y_pred_neg': logits[positive:]})
        result = {'role': request['role'], 'raw_logits': logits, 'member_outputs': member_outputs,
            'query_endpoints': torch.cat((split['edge'], split['edge_neg'])),
            'labels': torch.cat((torch.ones(positive, dtype=torch.long), torch.zeros(len(logits)-positive, dtype=torch.long))),
            'ids': torch.arange(len(logits)), 'metrics': {k: float(v) for k, v in metrics.items()},
            'selected_epochs': [c['selector']['selected_epoch'] for c in checkpoints],
            'selected_job_ids': [job['id'] for job in jobs], 'selected_checkpoint_receipts': [row['checkpoint'] for row in request['selected']],
            'serving_rule': request['serving_rule'], 'request_sha256': digest(request), 'work': work,
            'TEST_accessed': request['role'] == 'TEST', 'optimizer_maps_executed': 0}
        if buddy_context: result['buddy_context_binding'] = buddy_context[2]
        if qualify:
            result['qualification'] = {'schema': 'selected-link-VALID-qualification-v1', 'role': 'VALID',
                'passed': all(row['passed'] for row in parity), 'TEST_accessed': False,
                'binding': _qualification_binding(request), 'per_selected_fit': parity,
                'source_BUDDY_combined_batch_vs_frozen_split_batch_difference': request['native_family'] == 'BUDDY'}
        return result
    finally:
        for adapter in adapters: adapter.close()
        integration._restore_rng(torch, outside)


def qualify_VALID(torch, request, root, *, admit_runtime, device='cuda:0'):
    if not callable(admit_runtime): raise ValueError('Caller must provide its actual allocation admission gate')
    admit_runtime()
    validate_request(request, root)
    if request['role'] != 'VALID': raise PermissionError('This qualification API cannot open TEST')
    return _run(torch, request, root, device, True)


def serve_TEST(torch, request, approval, root, *, admit_runtime, device='cuda:0'):
    if not callable(admit_runtime): raise ValueError('Caller must provide its actual allocation admission gate')
    admit_runtime()
    validate_request(request, root)
    authorization = authorize_TEST(request, approval, root)
    result = _run(torch, request, root, device, False)
    result['authorization'] = authorization
    return result
