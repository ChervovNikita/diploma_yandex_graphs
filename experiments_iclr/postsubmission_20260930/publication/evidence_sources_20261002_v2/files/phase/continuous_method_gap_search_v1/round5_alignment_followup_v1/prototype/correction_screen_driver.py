"""Future correction screen CLI. SOURCE TEXT ONLY: never imported/run in authoring.

prepare reads graph/masks but no labels; fit reads only compact source-role labels;
report alone reads compact final-pool labels, after verifying an immutable freeze.
Every command requires a prospectively completed, fingerprinted manifest. A failed
attempt is retained and never silently retried or replaced with a favorable cell.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

# Future execution must not write Python cache files beside immutable sources.
sys.dont_write_bytecode = True

PRIMARY = ('aligned', 'marginal', 'shuffled', 'pooled', 'head', 'cf')
SEEDS = (17, 29, 43)
MODEL_SHA = 'a74a87dc26b7675a2d3fa0aaf0e7734b4786fa6c49411ef52f9bb21c0516dbc1'
FROZEN_SHA = '07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303'
TEACHER = dict(members=4, hidden_dim=128, num_layers=3, dropout=0.5,
               hidden_dim_multiplier=1.0, normalization='layer', num_heads=4,
               optimizer='AdamW', lr=0.001, weight_decay=0.01,
               betas=[0.9, 0.999], eps=1e-8, max_epochs=2000,
               min_epochs=300, patience=200, eval_every=1,
               loss='mean_member_cross_entropy', selector='mean_raw_logit_validation_nll')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, exclusive=False):
    with open(path, 'x' if exclusive else 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def descriptor(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))


def verified(record):
    require(set(record) >= {'path', 'sha256'}, 'Missing path/hash')
    require(isinstance(record['sha256'], str) and len(record['sha256']) == 64,
            'Fill all input fingerprints before launch')
    p = Path(record['path']).resolve()
    require(sha(p) == record['sha256'], f'Fingerprint mismatch: {p}')
    return p


def implementation_bindings():
    directory = Path(__file__).resolve().parent
    return {name: sha(directory/name) for name in ('correction_screen_driver.py',
             'aligned_score_correction.py', 'cf_gnn_source_adapter.py')}


def check_implementation(expected):
    require(implementation_bindings() == expected, 'Correction implementation differs from frozen code hashes')


def tree_records(root):
    return [dict(path=str(p.relative_to(root)), sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(root.rglob('*')) if p.is_file()]


def verify_tree(root, records):
    for record in records:
        p = root / record['path']
        require(sha(p) == record['sha256'], f'Frozen payload changed: {p}')


def runtime(device):
    import numpy as np
    import torch
    import torch_geometric
    require(device == 'cpu' or (device.startswith('cuda') and torch.cuda.is_available()),
            'Requested execution device unavailable')
    # CUDA indexed graph reductions may be nondeterministic. No cross-device
    # bitwise reproducibility is claimed; exact environment/random arrays are saved.
    torch.set_default_dtype(torch.float32)
    env = dict(python=platform.python_version(), numpy=np.__version__,
               torch=torch.__version__, torch_geometric=torch_geometric.__version__,
               cuda=torch.version.cuda, device=device,
               gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               float32_matmul_precision=torch.get_float32_matmul_precision(),
               cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
               cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
               cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    return np, torch, env


def sync(torch, device):
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)


def measured(torch, device, function):
    sync(torch, device)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        result = function()
    except Exception as error:
        sync(torch, device)
        error.screen_cost = dict(seconds=time.perf_counter()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                    if device.startswith('cuda') else None,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device)
                    if device.startswith('cuda') else None)
        raise
    sync(torch, device)
    return result, dict(seconds=time.perf_counter() - start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                        if device.startswith('cuda') else None,
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)
                        if device.startswith('cuda') else None,
                        process_maxrss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        cpu_peak_scope='process lifetime, not isolated method peak')


def np_load(np, record):
    return np.load(verified(record), allow_pickle=False)


def canonical_graph(np, manifest):
    x = np_load(np, manifest['features'])
    e = np_load(np, manifest['edges'])
    n, f, c = manifest['num_nodes'], manifest['num_features'], manifest['num_classes']
    require(x.dtype == np.float32 and x.shape == (n, f) and np.isfinite(x).all(),
            'Features must be finite FP32 [N,F]')
    require(e.dtype == np.int64 and e.ndim == 2 and e.shape[0] == 2, 'Edges int64 [2,E]')
    require(e.shape[1] == manifest['num_edges'] and c >= 2, 'Metadata shape mismatch')
    require((e >= 0).all() and (e < n).all() and (e[0] < e[1]).all(), 'Require 0<=u<v<N')
    require(np.unique(e.T, axis=0).shape[0] == e.shape[1], 'Duplicate correction edges')
    require(np.array_equal(np.lexsort((e[1], e[0])), np.arange(e.shape[1])),
            'Edges must be sorted lexicographically before freeze')
    return x, e


def prepare(args):
    """Future unlabeled preparation. Does not accept any label file argument."""
    import numpy as np
    import networkx as nx
    manifest = read_json(args.input)
    require(manifest['graph'] in ('Squirrel', 'Photo'), 'Only the recommended two graphs')
    require(manifest['release_provenance'] and manifest['preprocessing'], 'Complete provenance first')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    x, edges = canonical_graph(np, manifest)
    n, c = x.shape[0], manifest['num_classes']
    masks = None
    if manifest['graph'] == 'Squirrel':
        masks = np_load(np, manifest['published_masks'])
        require(set(masks.files) == {'train', 'validation', 'pool'}, 'Mask keys must be exact')
        for key in masks.files:
            require(masks[key].dtype == np.bool_ and masks[key].shape[0] >= 3 and
                    masks[key].shape[1:] == (n,), 'Published masks [S,N], S>=3')
    for split_index, seed in enumerate(SEEDS):
        cell = out / f'seed{seed}_split{split_index}'
        cell.mkdir()
        rng = np.random.Generator(np.random.PCG64(seed + 10000))
        if masks is not None:
            published = {key: np.flatnonzero(masks[key][split_index]) for key in masks.files}
            require(np.array_equal(np.sort(np.concatenate(list(published.values()))), np.arange(n)),
                    'Published roles must partition all released nodes exactly once')
            published_train = rng.permutation(published['train'])
            train_count, reservoir_count = n*20//100, n*50//100-n*20//100
            require(len(published_train) >= train_count and len(published['validation']) <= reservoir_count,
                    'Published mask cannot support the frozen derived-role rule')
            train, remainder = published_train[:train_count], published_train[train_count:]
            fill_count = reservoir_count-len(published['validation'])
            require(len(remainder) >= fill_count, 'Insufficient published train reservoir fill')
            reservoir = rng.permutation(np.concatenate((published['validation'], remainder[:fill_count])))
            pool = np.concatenate((published['pool'], remainder[fill_count:]))
            cut = [len(reservoir) * k // 6 for k in (2, 3, 5)]
            val, a, b, d = np.split(reservoir, cut)
            mask_rule = 'derived: floor(.20N) uniform published-train teacher nodes; reservoir floor(.50N)-floor(.20N) retains every published-validation node, fills from same train permutation, re-permutes then2:1:2:1 floors; pool retains every published-test node plus remaining train'
        else:
            order = rng.permutation(n)
            cut = [n * k // 100 for k in (20, 30, 35, 45, 50)]
            train, val, a, b, d, pool = np.split(order, cut)
            mask_rule = 'PCG64 uniform permutation; cumulative floors20/30/35/45/50%; no class balancing'
        roles = dict(train=train, validation=val, A=a, B=b, D=d, pool=pool)
        all_nodes = np.concatenate(list(roles.values()))
        require(len(all_nodes) == n and np.array_equal(np.sort(all_nodes), np.arange(n)),
                'Roles must partition every released node exactly once')
        require(all(len(v) > 0 for v in roles.values()) and len(a) >= 9, 'Insufficient source roles')
        for name, ids in roles.items():
            np.save(cell / f'{name}_nodes.npy', np.sort(ids).astype(np.int64), allow_pickle=False)
        uniforms = np.random.Generator(np.random.PCG64(seed + 30000)).random((n, c), dtype=np.float32)
        permutations = np.random.Generator(np.random.PCG64(seed + 40000)).random((n, 4)).argsort(1).astype(np.int64)
        np.save(cell / 'aps_uniforms.npy', uniforms, allow_pickle=False)
        np.save(cell / 'member_permutations.npy', permutations, allow_pickle=False)
        pool = np.sort(pool)
        for j in range(20):
            order = np.random.Generator(np.random.PCG64(seed + 50000 + j)).permutation(pool)
            np.save(cell / f'calibration{j:02d}.npy', np.sort(order[:len(pool)//2]), allow_pickle=False)
            np.save(cell / f'test{j:02d}.npy', np.sort(order[len(pool)//2:]), allow_pickle=False)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(edges.T.tolist())
        communities = list(nx.community.asyn_lpa_communities(graph, weight=None, seed=seed + 60000))
        communities.sort(key=lambda group: min(group))
        community = np.empty(n, dtype=np.int64)
        for index, group in enumerate(communities):
            community[list(group)] = index
        np.save(cell / 'communities.npy', community, allow_pickle=False)
        write_json(cell / 'ROLE_FREEZE.json', dict(graph_input=descriptor(args.input),
                   preparation_driver_sha256=sha(__file__),
                   graph=manifest['graph'], seed=seed, source_split_index=split_index,
                   mask_rule=mask_rule, source_counts={k: len(v) for k, v in roles.items()},
                   randomness='NumPy Generator PCG64; uniforms FP32; independent fixed per-node permutations',
                   community_recipe=dict(algorithm='networkx.asyn_lpa_communities', version=nx.__version__,
                                         seed=seed + 60000, edge_order='canonical lexicographic'),
                   labels_read=False, payload=tree_records(cell)), exclusive=True)
    write_json(out / 'PREPARATION_RECEIPT.json', dict(input=descriptor(args.input),
               preparation_seconds=time.perf_counter()-started, numpy=np.__version__,
               labels_read=False, source_fits=False), exclusive=True)


def load_source(np, torch, record, expected_nodes, device, classes):
    pack = np_load(np, record)
    require(set(pack.files) == {'nodes', 'labels'}, 'Compact role pack needs nodes/labels only')
    nodes, labels = pack['nodes'], pack['labels']
    require(nodes.dtype == labels.dtype == np.int64 and np.array_equal(nodes, expected_nodes),
            'Source label pack must match committed sorted role identities')
    require(labels.shape == nodes.shape and ((labels >= 0) & (labels < classes)).all(), 'Invalid labels')
    from aligned_score_correction import SourceLabels
    return SourceLabels(torch.from_numpy(nodes.copy()).to(device), torch.from_numpy(labels.copy()).to(device))


def load_model_source(admission):
    path = verified(admission['model_source'])
    require(sha(path) == MODEL_SHA, 'Wrong qualified original wrapper')
    dependency = path.parent.parent / 'propagation_cost_impl_v1/_pinned/frozen_models.py'
    require(sha(dependency) == FROZEN_SHA, 'Wrong immutable model dependency')
    spec = importlib.util.spec_from_file_location('screen_original_model_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_bytes(model):
    return sum(v.numel()*v.element_size() for v in model.state_dict().values())


def fit_teacher(torch, source, graph, x, train, validation, seed, qualify, out):
    model = source.build_model(arm='original', input_dim=x.shape[1], output_dim=graph.classes,
              seed=seed, permutation_seed=seed + 70000,
              **{k: TEACHER[k] for k in ('members', 'hidden_dim', 'num_layers', 'dropout',
                                        'hidden_dim_multiplier', 'normalization', 'num_heads')}).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEACHER['lr'],
                weight_decay=TEACHER['weight_decay'], betas=tuple(TEACHER['betas']), eps=TEACHER['eps'])
    best_nll, best_epoch, selected = float('inf'), None, None
    epochs = 3 if qualify else TEACHER['max_epochs']
    with open(out / 'teacher_trace.jsonl', 'x') as trace:
        for epoch in range(1, epochs+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph, x)
            require(logits.shape == (4, x.shape[0], graph.classes), 'Teacher [K,N,C] contract')
            selected_logits = logits[:, train.nodes]
            loss = torch.nn.functional.cross_entropy(selected_logits.reshape(-1, graph.classes), train.labels.repeat(4))
            require(bool(torch.isfinite(loss)), 'Nonfinite teacher loss')
            loss.backward()
            grad_count = sum(int(p.grad is not None and bool((p.grad != 0).any())) for p in model.parameters())
            require(grad_count > 0, 'No nonzero teacher gradients')
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_logits = model(graph, x).mean(0)[validation.nodes]
                nll = float(torch.nn.functional.cross_entropy(validation_logits, validation.labels))
            require(math.isfinite(nll), 'Nonfinite teacher validation loss')
            trace.write(json.dumps(dict(epoch=epoch, loss=float(loss.detach()), validation_nll=nll,
                                        nonzero_gradient_tensors=grad_count))+'\n')
            trace.flush()
            if nll < best_nll:
                best_nll, best_epoch = nll, epoch
                selected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not qualify and epoch >= TEACHER['min_epochs'] and epoch-best_epoch >= TEACHER['patience']:
                break
    model.load_state_dict(selected)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.save(selected, out / 'teacher_state.pt')
    write_json(out / 'teacher_selection.json', dict(selected_epoch=best_epoch, epochs_completed=epoch,
               source_validation_nll=best_nll, specification=model.specification,
               schedule=TEACHER, model_tensor_bytes=tensor_bytes(model),
               label_scope='train and predictor-validation only', qualification_only=qualify))
    del optimizer
    return model


def trace_writer(path):
    def sink(row):
        with open(path, 'a') as handle:
            handle.write(json.dumps(dict(epoch=row[0], loss=row[1], development_coverage=row[2],
                                         development_mean_size=row[3], nonzero_gradient_tensors=row[4] if len(row)>4 else None))+'\n')
    return sink


def select_results(results, inputs, a, d):
    from aligned_score_correction import _development_metrics
    choices = []
    aps_coverage, aps_size = _development_metrics(inputs.base_scores, a, d, .1)
    if aps_coverage >= .9:
        choices.append((aps_size, False, -1, 0, None, None))
    for index, (result, state) in enumerate(results):
        if result is None or result.no_source_feasible_choice:
            continue
        coverage, size = _development_metrics(result.frozen_scores, a, d, 0.1)
        if coverage >= 0.9:
            choices.append((size, not result.selected_zero_diffusion, index,
                            result.selected_epoch or 0, result, state))
    if not choices:
        return inputs.base_scores, None, dict(no_source_feasible_choice=True, selected_zero_diffusion=True,
                                              selected_config=None, selected_epoch=None)
    chosen = min(choices, key=lambda row: row[:4])
    if chosen[4] is None:
        return inputs.base_scores, None, dict(no_source_feasible_choice=False,
                   selected_zero_diffusion=True, selected_config=None, selected_epoch=0)
    return chosen[4].frozen_scores, chosen[5], dict(no_source_feasible_choice=False,
               selected_zero_diffusion=chosen[4].selected_zero_diffusion,
               selected_config=chosen[2], selected_epoch=chosen[3])


def cf_qualification(torch, inputs, uniforms, a, b, d, pool, out, device, seed):
    from cf_gnn_source_adapter import PooledGraphCorrection, native_source_loss
    from aligned_score_correction import randomized_aps, _development_metrics
    directed = torch.cat((inputs.edges, inputs.edges.flip(0)), 1)
    # Two actual full-graph updates qualify both native loss branches. This is
    # not a 1001-epoch fitted trajectory and is ineligible for final reporting.
    for index, backbone in enumerate(('GCN', 'GraphSAGE')):
        torch.manual_seed(seed + 20000 + index)
        model = PooledGraphCorrection(inputs.classes, backbone).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=5e-4)
        for branch_epoch in (1, 1001):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = native_source_loss(model(inputs.point_probabilities, directed), a, b, branch_epoch)
            require(bool(torch.isfinite(loss)), 'Nonfinite qualification CF loss')
            loss.backward()
            require(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters()),
                    'CF qualification requires nonzero gradients')
            optimizer.step()
            model.eval()
            with torch.no_grad():
                scores = randomized_aps(model(inputs.point_probabilities, directed).softmax(1), uniforms)
                coverage, size = _development_metrics(scores, a, d, .1)
            trace_writer(out / f'cf_{index}_trace.jsonl')((branch_epoch, float(loss.detach()), coverage, size))
        del optimizer, model


def fit(args):
    started = time.perf_counter()
    admission_record = descriptor(args.admission)
    admission = read_json(args.admission)
    require(sha(args.admission) == admission_record['sha256'], 'Admission changed while reading')
    check_implementation(admission['implementation_sha256'])
    require(admission['execution_authorized'] is True and admission['teacher_schedule'] == TEACHER,
            'A separate completed admission must authorize this exact prospective schedule')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'ATTEMPT_STARTED.json', dict(admission=admission_record,
               mode=args.command, timestamp=time.time()), exclusive=True)
    try:
        fit_body(args, admission, out, admission_record, started)
    except Exception as error:
        write_json(out / 'FAILED_ATTEMPT.json', dict(error_type=type(error).__name__, message=str(error),
                   traceback=traceback.format_exc(), fit_wall_seconds=time.perf_counter()-started,
                   automatic_retry=False), exclusive=True)
        raise


def fit_body(args, admission, out, admission_record, started):
    np, torch, env = runtime(admission['device'])
    device = admission['device']
    require(env == admission['environment'], 'Freeze the actual runtime fingerprint before qualification/fit')
    role_path = verified(admission['role_freeze'])
    role_freeze, cell = read_json(role_path), role_path.parent
    require(role_freeze['preparation_driver_sha256'] == admission['implementation_sha256']['correction_screen_driver.py'],
            'Role preparation used different driver source')
    verify_tree(cell, role_freeze['payload'])
    graph_manifest = read_json(verified(role_freeze['graph_input']))
    x_np, edge_np = canonical_graph(np, graph_manifest)
    seed = role_freeze['seed']
    require(seed in SEEDS and role_freeze['source_split_index'] == SEEDS.index(seed), 'Fixed split/seed pairing')
    require(admission['graph'] == graph_manifest['graph'] == role_freeze['graph'], 'Graph identity mismatch')
    require(admission['roles_frozen_before_label_extraction'] is True and
            admission['prior_exposure_disclosure'] and admission['independent_of_stage1_outcomes'] is True,
            'Complete role provenance and exposure disclosure before execution')
    require(set(admission['source_labels']) == {'train', 'validation', 'A', 'B', 'D'},
            'Fit accepts exactly five compact source label packs; no pool pack')
    # All label-independent roles, random arrays and source hashes were frozen
    # before opening compact labels. The admission itself binds their hashes.
    source = load_model_source(admission)
    expected = {k: np.load(cell / f'{k}_nodes.npy', allow_pickle=False) for k in admission['source_labels']}
    roles = {k: load_source(np, torch, admission['source_labels'][k], expected[k], device,
                           graph_manifest['num_classes']) for k in expected}
    pool = torch.from_numpy(np.load(cell / 'pool_nodes.npy', allow_pickle=False)).to(device)
    x = torch.from_numpy(x_np.copy()).to(device)
    edges = torch.from_numpy(edge_np.copy()).to(device)
    from types import SimpleNamespace
    graph = SimpleNamespace(edge_index=torch.cat((edges, edges.flip(0)), 1), classes=graph_manifest['num_classes'])
    uniforms = torch.from_numpy(np.load(cell / 'aps_uniforms.npy', allow_pickle=False)).to(device)
    permutations = torch.from_numpy(np.load(cell / 'member_permutations.npy', allow_pickle=False)).to(device)
    torch.manual_seed(seed)
    qualify = args.command == 'qualify'
    teacher, teacher_cost = measured(torch, device, lambda: fit_teacher(torch, source, graph, x,
                   roles['train'], roles['validation'], seed, qualify, out))
    with torch.no_grad():
        logits, inference_cost = measured(torch, device, lambda: teacher(graph, x).detach())
    np.save(out / 'teacher_member_logits.npy', logits.cpu().numpy(), allow_pickle=False)
    from aligned_score_correction import (prepare_fixed_inputs, SymmetricEdgeGate, fit_source_gate,
                    head_family_scores, _development_metrics, randomized_aps)
    inputs, feature_cost = measured(torch, device, lambda: prepare_fixed_inputs(logits, edges, uniforms, permutations))
    a, b, d = roles['A'], roles['B'], roles['D']
    attempts, chosen, scores, probabilities = [], {}, {}, {}
    for arm in ('aligned', 'marginal', 'shuffled', 'pooled'):
        results = []
        for config, lr in enumerate((.001, .0003)):
            torch.manual_seed(seed + 20000)
            gate = SymmetricEdgeGate(inputs.node_features.shape[1]).to(device)
            attempt = dict(arm=arm, config=config, learning_rate=lr)
            try:
                result, cost = measured(torch, device, lambda: fit_source_gate(inputs, gate, arm, a, b, d, pool,
                     max_epochs=2 if qualify else 2000, min_epochs=2 if qualify else 200, patience=200,
                     learning_rate=lr, trace_sink=trace_writer(out / f'{arm}_{config}_trace.jsonl')))
                state = None if result.selected_state is None else {k: v.cpu() for k, v in result.selected_state.items()}
                if qualify:
                    require(any(len(row)>4 and row[4]>0 for row in result.source_trace),
                            'Gate qualification did not obtain nonzero gradients')
                torch.save(state, out / f'{arm}_{config}_state.pt')
                attempt.update(status='completed', cost=cost, selected_epoch=result.selected_epoch,
                               no_source_feasible_choice=result.no_source_feasible_choice)
                results.append((result, state))
            except (RuntimeError, ValueError, FloatingPointError) as error:
                sync(torch, device)
                attempt.update(status='failed', error_type=type(error).__name__, message=str(error),
                               cost=getattr(error, 'screen_cost', None),
                               partial_trace=f'{arm}_{config}_trace.jsonl', automatic_retry=False)
                results.append((None, None))
            finally:
                attempts.append(attempt)
                write_json(out / 'SOURCE_ATTEMPTS.json', attempts)
                del gate
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
        scores[arm], state, chosen[arm] = select_results(results, inputs, a, d)
        scores[arm] = scores[arm].detach()
        torch.save(state, out / f'{arm}_selected_state.pt')
        del results
    family, head_cost = measured(torch, device, lambda: head_family_scores(inputs))
    head_choices = []
    for index, (name, table) in enumerate(family.items()):
        coverage, size = _development_metrics(table, a, d, .1)
        attempts.append(dict(arm='head', config=name, status='completed', development_coverage=coverage,
                             development_mean_size=size))
        if coverage >= .9:
            head_choices.append((size, name != 'aps', index, name))
    name = min(head_choices, key=lambda row: row[:3])[3] if head_choices else 'aps'
    scores['head'] = family[name].detach()
    chosen['head'] = dict(selected_variant=name, selected_zero_diffusion=name == 'aps',
                          no_source_feasible_choice=not bool(head_choices))
    # Declared isolated-node convention: arithmetic mean over nonisolated nodes.
    p = inputs.point_probabilities
    u, v = edges
    total = torch.zeros(x.shape[0], device=device)
    dot = (p[u] * p[v]).sum(1)
    total.index_add_(0, u, dot); total.index_add_(0, v, dot)
    nonisolated = inputs.degree > 0
    average_h = float((total[nonisolated] / inputs.degree[nonisolated]).mean()) if bool(nonisolated.any()) else 0.0
    published_variant = 'head_signed' if average_h < .4 else 'head_edge' if average_h < .6 else 'head_v3'
    scores.update(aps=family['aps'].detach(), daps=family['daps'].detach(),
                  head_published=family[published_variant].detach())
    chosen['head_published'] = dict(variant=published_variant, average_soft_homophily=average_h,
                         isolated_convention='exclude isolates from selector mean; all-isolated mean0',
                         source_parity_unresolved=True)
    if qualify:
        _, cf_cost = measured(torch, device, lambda: cf_qualification(torch, inputs, uniforms, a, b, d, pool,
                                                                    out, device, seed))
        write_json(out / 'SOURCE_ATTEMPTS.json', attempts)
        write_json(out / 'QUALIFICATION_ONLY.json', dict(environment=env, teacher_cost=teacher_cost,
                   teacher_inference_cost=inference_cost, feature_cost=feature_cost, cf_cost=cf_cost,
                   head_cost=head_cost, all_gate_attempts_completed=all(row['status']=='completed' for row in attempts),
                   complete_graph=True, teacher_updates=3, gate_updates_per_config=2,
                   cf_updates_per_config=2, cf_branch_epochs=[1, 1001], trained_utility=False,
                   final_pool_labels_read=False, report_eligible=False,
                   resource_extrapolation='Multiply measured complete-update times by capped schedules; not a measured completion forecast'), exclusive=True)
        check_implementation(admission['implementation_sha256'])
        verified(admission_record)
        write_json(out / 'FIT_WALL_RECEIPT.json', dict(seconds=time.perf_counter()-started,
                   scope='Residual-inclusive qualification wall from admission read through input load, construction, updates, source selection and artifact writes',
                   final_labels_read=False), exclusive=True)
        return
    from cf_gnn_source_adapter import PooledGraphCorrection, fit_cf_source_corrector
    results, cf_probabilities = [], []
    for config, backbone in enumerate(('GCN', 'GraphSAGE')):
        torch.manual_seed(seed + 20000 + config)
        corrector = PooledGraphCorrection(inputs.classes, backbone).to(device)
        attempt = dict(arm='cf', config=config, backbone=backbone)
        try:
            (result, cp), cost = measured(torch, device, lambda: fit_cf_source_corrector(inputs, corrector, uniforms,
                 a, b, d, pool, trace_sink=trace_writer(out / f'cf_{config}_trace.jsonl')))
            state = None if result.selected_state is None else {k: value.cpu() for k, value in result.selected_state.items()}
            torch.save(state, out / f'cf_{config}_state.pt')
            results.append((result, state)); cf_probabilities.append(cp)
            attempt.update(status='completed', cost=cost, selected_epoch=result.selected_epoch,
                           no_source_feasible_choice=result.no_source_feasible_choice)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            attempt.update(status='failed', error_type=type(error).__name__, message=str(error), automatic_retry=False)
            attempt['cost'] = getattr(error, 'screen_cost', None)
            results.append((None, None)); cf_probabilities.append(None)
        finally:
            attempts.append(attempt)
            write_json(out / 'SOURCE_ATTEMPTS.json', attempts)
            del corrector
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
    scores['cf'], cf_state, chosen['cf'] = select_results(results, inputs, a, d)
    config = chosen['cf']['selected_config']
    cp = inputs.point_probabilities if config is None or chosen['cf']['selected_zero_diffusion'] else cf_probabilities[config]
    probabilities['cf'] = cp.detach()
    scores['cf_unrandomized_shared_kth'] = randomized_aps(cp, torch.ones_like(uniforms)).detach()
    torch.save(cf_state, out / 'cf_selected_state.pt')
    np.save(out / 'point_probabilities.npy', inputs.point_probabilities.cpu().numpy(), allow_pickle=False)
    np.save(out / 'cf_probabilities.npy', probabilities['cf'].cpu().numpy(), allow_pickle=False)
    np.save(out / 'base_scores.npy', inputs.base_scores.cpu().numpy(), allow_pickle=False)
    for arm, table in scores.items():
        require(bool(torch.isfinite(table).all()), f'Nonfinite frozen {arm} scores')
        np.save(out / f'{arm}_scores.npy', table.cpu().numpy(), allow_pickle=False)
    # Label-independent strata fixed before final labels. Balanced rank quartiles
    # use node identity as tie-breaker; all class groups are reported later.
    degree = inputs.degree.cpu().numpy()
    # Disagreement is actual member argmax variation, not uncertainty entropy.
    member_top = logits.argmax(-1).cpu().numpy()
    disagreement = sum((member_top[i] != member_top[j]).astype(np.float64)
                       for i in range(4) for j in range(i+1, 4)) / 6
    for name, values in (('degree', degree), ('disagreement', disagreement)):
        order = np.lexsort((np.arange(len(values)), values))
        group = np.empty(len(values), dtype=np.int64)
        group[order] = np.minimum(3, np.arange(len(values))*4//len(values))
        np.save(out / f'{name}_quartiles.npy', group, allow_pickle=False)
    np.save(out / 'communities.npy', np.load(cell / 'communities.npy', allow_pickle=False), allow_pickle=False)
    write_json(out / 'SOURCE_SELECTION.json', chosen)
    write_json(out / 'COSTS.json', dict(teacher_fit=teacher_cost, teacher_inference=inference_cost,
               preparation_all_features=feature_cost, head_transforms=head_cost, source_attempts=attempts,
               accounting='All attempts and traces retained; no matched-compute claim'))
    del results, cf_probabilities, scores, cp, inputs, logits, teacher, roles, pool, uniforms, permutations
    del x, edges, graph
    result = state = cf_state = p = u = v = total = nonisolated = None
    probabilities = family = {}
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    serving = profile_serving(np, torch, source, admission, role_freeze, graph_manifest, cell, out, chosen)
    write_json(out / 'SERVING_COSTS.json', serving)
    check_implementation(admission['implementation_sha256'])
    verified(admission_record)
    verified(admission['role_freeze'])
    write_json(out / 'FIT_WALL_RECEIPT.json', dict(seconds=time.perf_counter()-started,
               scope='Residual-inclusive fit wall from admission read through input/source-label loading, all constructions/fits/checkpoint saves/selections and serving profiling; excludes final score-freeze serialization',
               final_labels_read=False), exclusive=True)
    payload = tree_records(out)
    write_json(out / 'SCORE_FREEZE.json', dict(admission=admission_record, role_freeze=admission['role_freeze'],
               implementation_sha256=admission['implementation_sha256'],
               environment=env, graph=graph_manifest['graph'], seed=seed,
               source_split_index=role_freeze['source_split_index'], primary_methods=list(PRIMARY),
               final_allocations=20, alpha=.1, selected=chosen,
               canonical_point='softmax(mean raw member logits), first-index argmax',
               final_labels_read=False, payload=payload), exclusive=True)


def profile_serving(np, torch, source, admission, roles, graph_manifest, cell, out, chosen):
    from types import SimpleNamespace
    from aligned_score_correction import (prepare_fixed_inputs, SymmetricEdgeGate, corrected_scores,
                                         randomized_aps, head_variant_scores)
    from cf_gnn_source_adapter import PooledGraphCorrection
    device, seed = admission['device'], roles['seed']
    reference_logits = np.load(out/'teacher_member_logits.npy', allow_pickle=False)
    costs = {}
    for arm in PRIMARY:
        context = {}
        def cold_setup():
            # Full cold local serving: read graph/random arrays and checkpoint,
            # construct/load/move teacher and selected correction; no optimizer.
            x_np, e_np = canonical_graph(np, graph_manifest)
            context['x'] = torch.from_numpy(x_np.copy()).to(device)
            context['e'] = torch.from_numpy(e_np.copy()).to(device)
            context['graph'] = SimpleNamespace(edge_index=torch.cat((context['e'], context['e'].flip(0)), 1))
            context['u'] = torch.from_numpy(np.load(cell/'aps_uniforms.npy', allow_pickle=False)).to(device)
            if arm in ('aligned', 'marginal', 'shuffled', 'pooled') and not chosen[arm]['selected_zero_diffusion']:
                context['perm'] = torch.from_numpy(np.load(cell/'member_permutations.npy', allow_pickle=False)).to(device)
            context['capture_replay'] = True
            model = source.build_model(arm='original', input_dim=x_np.shape[1], output_dim=graph_manifest['num_classes'],
                    seed=seed, permutation_seed=seed+70000,
                    **{k: TEACHER[k] for k in ('members', 'hidden_dim', 'num_layers', 'dropout',
                                              'hidden_dim_multiplier', 'normalization', 'num_heads')}).to(device)
            model.load_state_dict(torch.load(out/'teacher_state.pt', map_location=device, weights_only=True))
            context['teacher'] = model.eval()
            if arm in ('aligned', 'marginal', 'shuffled', 'pooled') and not chosen[arm]['selected_zero_diffusion']:
                node_dim = 6*graph_manifest['num_classes']+1
                gate = SymmetricEdgeGate(node_dim).to(device)
                gate.load_state_dict(torch.load(out/f'{arm}_selected_state.pt', map_location=device, weights_only=True))
                context['correction'] = gate.eval()
            elif arm == 'cf' and not chosen[arm]['selected_zero_diffusion']:
                backbone = ('GCN', 'GraphSAGE')[chosen[arm]['selected_config']]
                corrector = PooledGraphCorrection(graph_manifest['num_classes'], backbone).to(device)
                corrector.load_state_dict(torch.load(out/'cf_selected_state.pt', map_location=device, weights_only=True))
                context['correction'] = corrector.eval()
            return serve()
        def serve():
            with torch.no_grad():
                z = context['teacher'](context['graph'], context['x'])
                if context['capture_replay']:
                    context['last_member_logits'] = z.detach()
                p = z.mean(0).softmax(-1)
                if chosen[arm]['selected_zero_diffusion']:
                    return randomized_aps(p, context['u'])
                if arm == 'cf':
                    cp = context['correction'](p, context['graph'].edge_index).softmax(-1)
                    return randomized_aps(cp, context['u'])
                if arm == 'head':
                    inp = SimpleNamespace(point_probabilities=p, base_scores=randomized_aps(p,context['u']),
                          edges=context['e'], degree=torch.bincount(context['e'].flatten(), minlength=p.shape[0]),
                          classes=p.shape[1])
                    return head_variant_scores(inp, chosen[arm]['selected_variant'])
                mode = arm if arm in ('aligned', 'shuffled', 'pooled') else 'none'
                inp = prepare_fixed_inputs(z, context['e'], context['u'], context['perm'], feature_mode=mode)
                return corrected_scores(inp, context['correction'], arm)
        _, cold = measured(torch, device, cold_setup)
        replay_difference = np.abs(context['last_member_logits'].cpu().numpy()-reference_logits)
        del context['last_member_logits']
        context['capture_replay'] = False
        samples = []
        for _ in range(10):
            _, warm = measured(torch, device, serve)
            samples.append(warm)
        costs[arm] = dict(cold=cold, warm=samples,
                         cold_teacher_replay_max_absolute_logit_difference=float(replay_difference.max()),
                         replay_policy='Report numeric differences and saved-logit metric recomputation; no bitwise claim, no hidden retry/tolerance rescue',
                         warm_mean_seconds=sum(row['seconds'] for row in samples)/len(samples),
                         teacher_member_trajectories_per_request=4,
                         scope='Full local graph/checkpoint loading in cold; full fresh teacher forward and correction in every warm sample; no cached-logit serving')
        context.clear()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return costs


def metrics(np, scores, base, probabilities, nodes, labels, threshold, canonical_probabilities):
    sets = scores[nodes] <= threshold
    hit = sets[np.arange(len(nodes)), labels]
    size = sets.sum(1)
    singleton = size == 1
    point = canonical_probabilities[nodes].argmax(1)
    minimum = scores[nodes].argmin(1)
    singleton_label = sets.argmax(1)
    dis = singleton & (singleton_label != point)
    order = np.argsort(scores[nodes], axis=1, kind='stable')
    base_order = np.argsort(base[nodes], axis=1, kind='stable')
    def mean_or_none(value):
        return float(value.mean()) if len(value) else None
    return dict(count=len(nodes), coverage=mean_or_none(hit), mean_size=mean_or_none(size),
           p90_size=float(np.percentile(size, 90, method='higher')) if len(size) else None,
           empty_frequency=mean_or_none(size == 0), singleton_frequency=mean_or_none(singleton),
           full_frequency=mean_or_none(size == scores.shape[1]),
           changed_class_order_frequency=mean_or_none((order != base_order).any(1)),
           canonical_point_vs_minimum_score_agreement=mean_or_none(point == minimum),
           singleton_count=int(singleton.sum()),
           singleton_vs_canonical_agreement=mean_or_none((singleton_label == point)[singleton]),
           singleton_hit_frequency=mean_or_none(singleton & hit),
           singleton_conditional_accuracy=mean_or_none(hit[singleton]),
           singleton_point_disagreement_count=int(dis.sum()),
           singleton_disagreement_accuracy=mean_or_none(hit[dis]),
           point_accuracy=mean_or_none(point == labels),
           point_nll=mean_or_none(-np.log(np.maximum(canonical_probabilities[nodes, labels], np.finfo(np.float32).tiny))),
           point_entropy=mean_or_none(-(canonical_probabilities[nodes]*np.log(np.maximum(canonical_probabilities[nodes], np.finfo(np.float32).tiny))).sum(1)),
           corrected_probability_point_accuracy=mean_or_none(probabilities[nodes].argmax(1) == labels),
           corrected_probability_nll=mean_or_none(-np.log(np.maximum(probabilities[nodes, labels], np.finfo(np.float32).tiny))),
           corrected_probability_entropy=mean_or_none(-(probabilities[nodes]*np.log(np.maximum(probabilities[nodes], np.finfo(np.float32).tiny))).sum(1)))


def report(args):
    """Only this command opens final-pool labels; nothing is refitted here."""
    started = time.perf_counter()
    import numpy as np
    frozen_path = Path(args.frozen).resolve()
    freeze, root = read_json(frozen_path), frozen_path.parent
    check_implementation(freeze['implementation_sha256'])
    frozen_admission = read_json(verified(freeze['admission']))
    require(frozen_admission['implementation_sha256'] == freeze['implementation_sha256'],
            'Frozen admission/code binding mismatch')
    require(freeze['final_labels_read'] is False and freeze['primary_methods'] == list(PRIMARY), 'Wrong freeze')
    verify_tree(root, freeze['payload'])
    role_path = verified(freeze['role_freeze'])
    roles, cell = read_json(role_path), role_path.parent
    verify_tree(cell, roles['payload'])
    labels_record = read_json(args.pool_labels)
    require(labels_record['score_freeze_sha256'] == sha(frozen_path), 'Pool labels must bind this score freeze')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Exclusive receipt in frozen directory prevents repeated reporting even to
    # a new output path; failed reports keep the receipt and failure details.
    write_json(root/'REPORT_STARTED.json', dict(score_freeze=descriptor(frozen_path),
               pool_labels_manifest=descriptor(args.pool_labels), output=str(out)), exclusive=True)
    try:
        pool = np.load(cell/'pool_nodes.npy', allow_pickle=False)
        pack = np_load(np, labels_record['labels'])
        require(set(pack.files) == {'nodes', 'labels'} and np.array_equal(pack['nodes'], pool), 'Compact pool pack mismatch')
        labels = pack['labels']
        point = np.load(root/'point_probabilities.npy', allow_pickle=False)
        require(labels.dtype == np.int64 and labels.shape == pool.shape and
                ((labels >= 0) & (labels < point.shape[1])).all(), 'Invalid pool labels')
        base = np.load(root/'base_scores.npy', allow_pickle=False)
        cp = np.load(root/'cf_probabilities.npy', allow_pickle=False)
        saved_logits = np.load(root/'teacher_member_logits.npy', allow_pickle=False).astype(np.float64)
        reference_logits = saved_logits.mean(0)
        reference_exp = np.exp(reference_logits-reference_logits.max(1, keepdims=True))
        reference_probabilities = reference_exp/reference_exp.sum(1, keepdims=True)
        stratum = {name: np.load(root/f'{name}.npy', allow_pickle=False)
                   for name in ('degree_quartiles', 'disagreement_quartiles', 'communities')}
        stratum['class'] = np.full(point.shape[0], -1, dtype=np.int64)
        stratum['class'][pool] = labels
        label_by_node = np.full(point.shape[0], -1, dtype=np.int64)
        label_by_node[pool] = labels
        tables = {name: np.load(root/f'{name}_scores.npy', allow_pickle=False)
                  for name in (*PRIMARY, 'aps', 'daps', 'head_published', 'cf_unrandomized_shared_kth')}
        allocations = []
        for j in range(20):
            allocation_started = time.perf_counter()
            cal = np.load(cell/f'calibration{j:02d}.npy', allow_pickle=False)
            test = np.load(cell/f'test{j:02d}.npy', allow_pickle=False)
            require(np.array_equal(np.sort(np.concatenate((cal, test))), pool) and
                    len(np.intersect1d(cal, test)) == 0, 'Final masks not a uniform prepared partition')
            rank = math.ceil((len(cal)+1)*.9)
            methods = {}
            for name, table in tables.items():
                calibration_started = time.perf_counter()
                cal_scores = table[cal, label_by_node[cal]]
                threshold = float(np.sort(cal_scores, kind='stable')[rank-1]) if rank <= len(cal) else float('inf')
                calibration_seconds = time.perf_counter()-calibration_started
                pred_probs = cp if name in ('cf', 'cf_unrandomized_shared_kth') else point
                main = metrics(np, table, base, pred_probs, test, label_by_node[test], threshold, point)
                groups = {}
                for group_name, group in stratum.items():
                    values = range(point.shape[1]) if group_name == 'class' else np.unique(group[pool]).tolist()
                    rows = {}
                    for value in values:
                        nodes = test[group[test] == value]
                        row = metrics(np, table, base, pred_probs, nodes, label_by_node[nodes], threshold, point)
                        row['pool_count'] = int((group[pool] == value).sum())
                        row['subgroup_gate_eligible'] = row['pool_count'] >= 100 and len(nodes) >= 50
                        rows[str(value)] = row
                    groups[group_name] = rows
                methods[name] = dict(threshold=threshold if math.isfinite(threshold) else '+inf',
                                     calibration_seconds=calibration_seconds,
                                     overall=main, groups=groups)
            allocations.append(dict(allocation=j, calibration_count=len(cal), test_count=len(test), methods=methods,
                     allocation_wall_seconds=time.perf_counter()-allocation_started,
                     saved_logit_cpu64_point_reference=dict(
                     accuracy=float((reference_probabilities[test].argmax(1)==label_by_node[test]).mean()),
                     nll=float(-np.log(reference_probabilities[test,label_by_node[test]]).mean()),
                     entropy=float(-(reference_probabilities[test]*np.log(np.maximum(reference_probabilities[test],np.finfo(np.float64).tiny))).sum(1).mean()),
                     point_argmax_agreement_with_fp32=float((reference_probabilities[test].argmax(1)==point[test].argmax(1)).mean()))))
        serving = read_json(root/'SERVING_COSTS.json')
        necessary_gates = {}
        aligned_size = np.mean([row['methods']['aligned']['overall']['mean_size'] for row in allocations])
        for comparator in ('pooled', 'marginal', 'shuffled', 'head', 'cf'):
            comparator_size = np.mean([row['methods'][comparator]['overall']['mean_size'] for row in allocations])
            worsening = []
            for allocation in allocations:
                for group_name in ('class', 'degree_quartiles'):
                    for key, group in allocation['methods']['aligned']['groups'][group_name].items():
                        other = allocation['methods'][comparator]['groups'][group_name][key]
                        if group['subgroup_gate_eligible']:
                            worsening.append(other['coverage']-group['coverage'])
            necessary_gates[comparator] = dict(mean_size_reduction=1-aligned_size/comparator_size if comparator_size else None,
                     five_percent_size_gate=bool(comparator_size > 0 and aligned_size <= .95*comparator_size),
                     worst_eligible_coverage_deterioration=max(worsening) if worsening else None,
                     subgroup_gate=bool(worsening and max(worsening) <= .02),
                     absent_eligible_groups_are_success=False)
        ratio = serving['aligned']['warm_mean_seconds']/serving['marginal']['warm_mean_seconds']
        result = dict(score_freeze=descriptor(frozen_path), graph=freeze['graph'], seed=freeze['seed'],
                 source_split_index=freeze['source_split_index'], selected=freeze['selected'], allocations=allocations,
                 necessary_gates=necessary_gates, serving_latency_ratio_to_marginal=ratio,
                 latency_gate=bool(ratio <= 1.1), coverage_interpretation='Marginal random-allocation procedure; no node-IID intervals or realized90% requirement',
                 promotion='Cannot promote automatically: six cells must be complete and validity/coverage reviewed; crossed split/seed confirmation remains conditional',
                 report_wall_seconds_before_serialization=time.perf_counter()-started,
                 secondary_cf_calibration='Unrandomized APS on source-selected CF probabilities with shared kth threshold; not the author np.quantile(higher) calibration',
                 significance_claim=False, repeated_allocations_independent_graphs=False)
        write_json(out/'FINAL_REPORT.json', result, exclusive=True)
        write_json(root/'REPORT_COMPLETED.json', dict(output=descriptor(out/'FINAL_REPORT.json')), exclusive=True)
    except Exception as error:
        write_json(out/'FAILED_REPORT.json', dict(error_type=type(error).__name__, message=str(error),
                   traceback=traceback.format_exc(), automatic_retry=False), exclusive=True)
        raise


def forecast(args):
    """Metadata-only prospective byte/work forecast, with no scientific arrays."""
    graph = read_json(args.input)
    n, e, f, c = (graph[k] for k in ('num_nodes', 'num_edges', 'num_features', 'num_classes'))
    require(all(type(v) is int and v > 0 for v in (n, e, f, c)), 'Complete source metadata dimensions first')
    node_dim = 6*c+1
    gate_parameters = (2*node_dim+2)*32+32 + 32*32+32 + 32+1
    result = dict(graph=graph['graph'], teacher_fits=3, teacher_epoch_cap_per_graph=6000,
          gate_fits_per_graph=24, gate_epoch_cap_per_graph=48000,
          cf_fits_per_graph=6, cf_epoch_cap_per_graph=30000,
          two_graph_training_epoch_cap=168000,
          immutable_array_bytes=dict(features=4*n*f, canonical_edges=16*e,
             directed_teacher_edges=32*e, teacher_logits=16*n*c, member_probabilities=16*n*c,
             aps_uniforms=4*n*c, member_permutations=32*n,
             full_node_features=4*n*node_dim, aligned_and_shuffled_features=16*e,
             one_class_score_table=4*n*c, ten_frozen_score_tables=40*n*c,
             twenty_final_partition_masks=160*n),
          gate_parameters=gate_parameters, gate_fp32_weight_bytes=4*gate_parameters,
          teacher_activation_note='K=4 complete propagations; saved autograd messages and residual activations need measured full-graph peak',
          gate_training_activation_lower_order_bytes=2*e*32*4*2,
          gate_activation_note='Illustrative two directional first/second hidden activations only; autograd inputs, masks, gradients, diffusion and optimizer add cost. Chunking retains state.',
          time_forecast='Unmeasured. Qualification uses full graph/update and loss branches. Extrapolate measured per-update time to caps, add selection/inference/reporting; report uncertainty, no numeric duration invented.')
    write_json(args.output, result, exclusive=True)


def environment(args):
    _, _, env = runtime(args.device)
    write_json(args.output, env, exclusive=True)


def summarize(args):
    reports = [read_json(path) for path in args.reports]
    expected = {(g, s, i) for g in ('Squirrel', 'Photo') for i, s in enumerate(SEEDS)}
    actual = {(r['graph'], r['seed'], r['source_split_index']) for r in reports}
    complete = actual == expected and len(reports) == 6
    rows = [dict(graph=r['graph'], seed=r['seed'], source_split_index=r['source_split_index'],
                 necessary_gates=r['necessary_gates'], latency_gate=r['latency_gate'],
                 source_failures={k: v['no_source_feasible_choice'] for k, v in r['selected'].items()
                                  if k in PRIMARY}) for r in reports]
    write_json(args.output, dict(complete_six_cell_screen=complete, cells=rows,
               missing=[list(v) for v in sorted(expected-actual)],
               all_necessary_size_subgroup_latency_gates=complete and all(r['latency_gate'] and
                 all(v['five_percent_size_gate'] and v['subgroup_gate'] for v in r['necessary_gates'].values()) for r in reports),
               promotion='Human validity/coverage review required; screen only, no publication significance or originality claim',
               input_reports=[descriptor(p) for p in args.reports]), exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    for command in ('qualify', 'fit'):
        p = sub.add_parser(command); p.add_argument('--admission', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('report'); p.add_argument('--frozen', required=True); p.add_argument('--pool-labels', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('forecast'); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('environment'); p.add_argument('--device', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('summarize'); p.add_argument('--reports', nargs='+', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'fit': fit, 'qualify': fit, 'report': report, 'forecast': forecast,
     'environment': environment, 'summarize': summarize}[args.command](args)


if __name__ == '__main__':
    main()
