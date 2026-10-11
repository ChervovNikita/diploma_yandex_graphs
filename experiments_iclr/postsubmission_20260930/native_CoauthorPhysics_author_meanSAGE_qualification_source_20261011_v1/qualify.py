"""One inactive full-Physics native author resource integration check.

One TRAIN update, full selected-state poison/restore, one label-free SELECT
forward. No SELECT labels, quality score, fit continuation or parity suite.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import resource
import sys
import time

CONTEXT_HASHES = {
    'author_model.py': 'dfd19cf6da0460a4de49697e2c5afe185e85e8192db4bb51e548868b9a2d1401',
    'consumer.py': '84f076664ad409b674e8a7ba89362e41670080635feedc512a7fd68e25821db0',
}
SOURCE_ARM = 'author_meanSAGE_M1'
PROTOCOL_ARM = 'published_mean_SAGE_source_port_M1'


def receipt(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def load_source(name, path, digest):
    actual = receipt(path)
    if actual['sha256'] != digest: raise ValueError('Pinned author context source differs')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, actual


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def clone(torch, value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: clone(torch, v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(clone(torch, v) for v in value)
    return value


def qualify(cfg, output, root_validate):
    """Root entry, separate from consumer scientific fit admission.

    cfg: status ROOT_AUTHOR_CONTEXT_RESOURCE_QUALIFICATION_ADMITTED,
    context_directory, seed, device='cuda:0', point=sealed author.POINT,
    profile{dataset,nodes,input_dim,classes,edges,self_edges},
    paths exact{train,select_ids}: pathstrings; no SELECT NPZ/label path,
    limits exact{wall_seconds,cuda_bytes,rss_bytes}: positive finite integers.
    root_validate(cfg,action='fixed_Physics_author_meanSAGE_resource_qualification',
                  source_receipts=...) must return primitive JSON custody with
    the same admitted status and bind source/runtime/provider/full safe TRAIN,
    label-free150SELECT membership, factual support, caps and host/GPU.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); torch = device = None
    result = {'status': 'RUNNING', 'dataset': 'CoauthorPhysics', 'source_arm': SOURCE_ARM,
              'protocol_arm': PROTOCOL_ARM, 'fits': 0, 'TRAIN_routes': 0, 'backwards': 0,
              'Adam_steps': 0, 'SELECT_label_free_routes': 0, 'SELECT_labels_or_scores': False,
              'CAL_CONFIRM_truth_access': False, 'automatic_scientific_fit_release': False,
              'new_equivalence_or_parity_suite': False}
    try:
        status = 'ROOT_AUTHOR_CONTEXT_RESOURCE_QUALIFICATION_ADMITTED'
        limits = cfg.get('limits', {})
        expected = {'dataset': 'CoauthorPhysics', 'nodes': 34493, 'input_dim': 8415,
                    'classes': 5, 'edges': 530417, 'self_edges': 34493}
        if (cfg.get('status') != status or not callable(root_validate)
                or cfg.get('device') != 'cuda:0'
                or set(cfg.get('paths', {})) != {'train', 'select_ids'}
                or set(limits) != {'wall_seconds', 'cuda_bytes', 'rss_bytes'}
                or any(type(v) is not int or v <= 0 for v in limits.values())
                or any(cfg.get('profile', {}).get(k) != v for k, v in expected.items())):
            raise ValueError('Inactive source: finite root native-resource admission and complete Physics required')
        context = Path(cfg['context_directory'])
        author, author_receipt = load_source('qualified_Physics_author_model', context / 'author_model.py', CONTEXT_HASHES['author_model.py'])
        helper, helper_receipt = load_source('qualified_Physics_author_state_helpers', context / 'consumer.py', CONTEXT_HASHES['consumer.py'])
        if cfg.get('seed') not in author.SEEDS or cfg.get('point') != author.POINT:
            raise ValueError('Frozen root seed and exact author point required')
        source_receipts = {'qualification': receipt(__file__), 'author_model': author_receipt, 'consumer': helper_receipt}
        custody = root_validate(cfg, action='fixed_Physics_author_meanSAGE_resource_qualification', source_receipts=source_receipts)
        if not isinstance(custody, dict) or custody.get('status') != status:
            raise ValueError('Root must bind separate one-update qualification source/runtime/provider/role admission')
        json.dumps(custody, allow_nan=False)
        result.update(custody=custody, source_receipts=source_receipts, limits=limits, seed=cfg['seed'])
        import numpy as np
        import torch
        import torch.nn.functional as F
        if torch.get_default_dtype() != torch.float32: raise ValueError('Native float32 runtime required')
        device = torch.device(cfg['device']); torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        def synchronize(): torch.cuda.synchronize(device)
        def budget():
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            result['CUDA_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            result['CUDA_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
            if (time.perf_counter() - started > limits['wall_seconds'] or rss > limits['rss_bytes']
                    or result['CUDA_peak_allocated_bytes'] > limits['cuda_bytes']):
                raise RuntimeError('Finite native qualification wall/RSS/CUDA cap exceeded')
        budget()
        with np.load(cfg['paths']['train'], allow_pickle=False) as a:
            if set(a.files) != {'x', 'edge_index', 'ids', 'y'}: raise ValueError('Exact safe complete TRAIN fields')
            x, edges, train_ids, train_y = [a[k].copy() for k in ('x', 'edge_index', 'ids', 'y')]
        # IDs only: no SELECT target file is accepted or deserialized.
        select_ids = np.load(cfg['paths']['select_ids'], allow_pickle=False)
        if (x.shape != (34493, 8415) or x.dtype != np.float32 or not np.isfinite(x).all()
                or not np.logical_or(x == 0., x == 1.).all()
                or edges.dtype != np.int64 or edges.shape != (2, 530417)
                or edges.min() < 0 or edges.max() >= 34493
                or int((edges[0] == edges[1]).sum()) != 34493
                or train_ids.dtype != np.int64 or train_ids.shape != (100,)
                or train_y.dtype != np.int64 or train_y.shape != (100,)
                or train_y.min() < 0 or train_y.max() >= 5
                or select_ids.dtype != np.int64 or select_ids.shape != (150,)):
            raise ValueError('Complete factual Physics support/raw features/safe roles required')
        joined = np.concatenate((train_ids, select_ids))
        if len(np.unique(joined)) != 250 or joined.min() < 0 or joined.max() >= 34493:
            raise ValueError('Unique disjoint complete TRAIN and label-free SELECT IDs required')
        tensor = lambda a: torch.from_numpy(a).to(device)
        train_idx, targets, select_idx = map(tensor, (train_ids, train_y, select_ids))
        features = torch.from_numpy(x).to_sparse_coo().coalesce().to(device)
        adjacency = torch.sparse_coo_tensor(torch.from_numpy(edges).flip(0), torch.ones(530417),
                                             (34493, 34493)).coalesce().to(device)
        if adjacency._nnz() != 530417: raise ValueError('Common prepared support changed')
        degrees = torch.bincount(adjacency.indices()[0], minlength=34493).to(torch.float32).unsqueeze(1)
        if degrees.min().item() < 1.: raise ValueError('Explicit-self common support requires positive degrees')
        result['graph'] = {**expected, 'TRAIN_rows': 100, 'SELECT_ID_rows': 150,
                           'input_stored_values': features._nnz(), 'added_edges_or_loops': 0}
        del x, edges, train_ids, train_y, select_ids, joined
        synchronize(); budget(); result['preparation_seconds'] = time.perf_counter() - started
        tick = time.perf_counter()
        seed = cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = author.model_class(torch, F)().to(device)
        result['construction'] = author.inspect_model(model)
        optimizer = author.optimizer_class(torch)(model.parameters())
        owned = [id(p) for group in optimizer.param_groups for p in group['params']]
        if len(owned) != len(set(owned)) or set(owned) != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError('Every registered published parameter exactly once in optimizer')
        synchronize(); phase = time.perf_counter(); model.train(); optimizer.zero_grad()
        logits = model(features, adjacency, degrees); result['TRAIN_routes'] += 1
        if logits.shape != (34493, 5) or not torch.isfinite(logits).all().item():
            raise ValueError('Complete native TRAIN graph output differs')
        loss = F.cross_entropy(logits[train_idx], targets) + model.regularizer()
        if not torch.isfinite(loss).item(): raise ValueError('Nonfinite native full-TRAIN total CE+L2')
        loss.backward(); result['backwards'] += 1
        gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()]
        if (any(g is None or not torch.isfinite(g).all().item() for name, g in gradients)
                or not any(g.ne(0).any().item() for name, g in gradients)):
            raise ValueError('Every registered native gradient finite and some native gradient active')
        optimizer.step(); result['Adam_steps'] += 1
        if optimizer.state['tf1_global']['updates'] != 1 or any(not optimizer.state[p] for p in model.parameters()):
            raise ValueError('Exactly one TF1 Adam advancement and all parameter moments required')
        del logits, loss, gradients
        synchronize(); result['TRAIN_update_seconds'] = time.perf_counter() - phase; budget()
        def state():
            return clone(torch, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'global_rng': helper.rng(torch, np, device), 'model_training': model.training})
        saved = state(); saved_fingerprint = helper.fingerprint(torch, saved)
        checkpoint = output / 'discarded_one_update_state.pt'; torch.save(saved, checkpoint)
        synchronize(); phase = time.perf_counter()
        with torch.no_grad():
            for parameter in model.parameters(): parameter.add_(1.)
            for values in optimizer.state.values():
                for value in values.values():
                    if torch.is_tensor(value): value.add_(1.)
            optimizer.state['tf1_global']['updates'] += 7
            optimizer.param_groups[0]['lr'] = .7
        model.eval(); random.seed(997); np.random.seed(997); torch.manual_seed(997); torch.cuda.manual_seed_all(997)
        if helper.fingerprint(torch, state()) == saved_fingerprint: raise ValueError('Full-state poison was inactive')
        loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(loaded['model'], strict=True); optimizer.load_state_dict(loaded['optimizer'])
        helper.restore_rng(torch, np, device, loaded['global_rng']); model.train(loaded['model_training'])
        if helper.fingerprint(torch, state()) != saved_fingerprint:
            raise ValueError('Strict model/Adam/beta-powers/iteration/mode/Python/NumPy/Torch RNG restore failed')
        result['strict_full_state_restore'] = True
        result['discarded_checkpoint'] = receipt(checkpoint)
        synchronize(); result['poison_and_restore_seconds'] = time.perf_counter() - phase; budget()
        model.eval(); synchronize(); phase = time.perf_counter()
        with torch.no_grad():
            output_logits = model(features, adjacency, degrees)[select_idx]
            probability = output_logits.softmax(-1)
        result['SELECT_label_free_routes'] += 1
        if output_logits.shape != (150, 5) or not torch.isfinite(output_logits).all().item() or not torch.isfinite(probability).all().item():
            raise ValueError('Complete label-free150SELECT output dimensions/finiteness differ')
        del output_logits, probability
        synchronize(); result['SELECT_label_free_seconds'] = time.perf_counter() - phase; budget()
        if (result['TRAIN_routes'], result['backwards'], result['Adam_steps'], result['SELECT_label_free_routes']) != (1, 1, 1, 1):
            raise ValueError('Exactly one native update and one label-free serving route required')
        result.update(status='PASS', actual_full_graph_native_resource_certified=True,
                      work_seconds=time.perf_counter() - tick, meanSAGE_layer_calls=4)
    except BaseException as error:
        result.update(status='FAILED_RETAINED', actual_full_graph_native_resource_certified=False,
                      failure={'type': type(error).__name__, 'message': str(error)}, retry=False)
        raise
    finally:
        if torch is not None and device is not None:
            result['CUDA_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            result['CUDA_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        result['process_lifetime_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        result['owner_seconds'] = time.perf_counter() - started
        write(output / 'AUTHOR_MEANSAGE_RESOURCE_QUALIFICATION.json', result)
    return result


if __name__ == '__main__':
    raise SystemExit('Inactive source. Root separately admits and executes exactly one native resource check.')
