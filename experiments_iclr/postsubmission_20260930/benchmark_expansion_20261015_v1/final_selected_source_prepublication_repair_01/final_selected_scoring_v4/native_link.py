"""Raw Collab split conversion and original BUDDY context; no top-level model imports."""
from pathlib import Path
import copy


def load_split(torch, path):
    # Same API as OGB PygLinkPropPredDataset.get_edge_split. It preserves
    # NumPy dtype and nested container shape; no casts or cache rewrite.
    from ogb.utils.torch_util import replace_numpy_with_torchtensor
    return replace_numpy_with_torchtensor(torch.load(path, map_location='cpu', weights_only=False))


def load_collab(torch, recipe, data_root):
    from torch_geometric.data import Data
    declared = recipe.get('data_files')
    if declared is not None:
        paths = {k: Path(declared[k]) for k in ('graph', 'train', 'valid')}
        paths = {k: p if p.is_absolute() else Path(data_root) / p for k, p in paths.items()}
    else:
        raise ValueError('Exact explicit source graph/TRAIN/VALID paths required')
    # Source-preserving graph deserialization in the admitted runtime. Nothing
    # calls PygLinkPropPredDataset.get_edge_split or opens a TEST split.
    packed = torch.load(paths['graph'], map_location='cpu', weights_only=False)
    raw = packed[0] if isinstance(packed, tuple) else packed
    if isinstance(raw, dict):
        raw = Data.from_dict(raw)
    if not hasattr(raw, 'x') or not hasattr(raw, 'edge_index'):
        raise ValueError('Unsupported official processed Collab graph format')
    graph = Data(x=raw.x.detach().cpu().clone(), edge_index=raw.edge_index.detach().cpu().clone(),
                 num_nodes=int(raw.num_nodes))
    weights = getattr(raw, 'edge_weight', None)
    if weights is not None:
        graph.edge_weight = weights.detach().cpu().clone()
    if graph.num_nodes != 235868 or tuple(graph.x.shape) != (235868, 128):
        raise ValueError('Full native Collab235868x128 required; graph/feature shrink is unsupported')
    train = load_split(torch, paths['train'])
    valid = load_split(torch, paths['valid'])
    for k in ('edge', 'edge_neg'):
        if valid[k].ndim != 2 or valid[k].shape[1] != 2:
            raise ValueError('Complete official VALID pair records required: ' + k)
    if tuple(valid['edge'].shape) != (60084, 2) or tuple(valid['edge_neg'].shape) != (100000, 2):
        raise ValueError('Native complete Collab VALID60084/100000 required')
    if train['edge'].ndim != 2 or train['edge'].shape[1] != 2:
        raise ValueError('Native complete TRAIN edge records required')
    return graph, train, valid, {k: str(p) for k, p in paths.items()}


def buddy_dataset(torch, integration, session, job, split, request, root, helper):
    import os
    inside, digest, receipt = helper.inside, helper.digest, helper.receipt
    _definition, PHASE = helper._definition, helper.PHASE
    SOURCE_BINDINGS_SHA256 = helper.SOURCE_BINDINGS_SHA256
    backend_sources = helper.backend_sources
    import time
    import scipy.sparse as ssp
    import torch_sparse
    from torch_geometric.data import Dataset
    from torch_geometric.nn.conv.gcn_conv import gcn_norm
    from torch_geometric.utils import to_undirected
    n = session.native
    callback = integration._native_factory(n.recipe)
    graph, train, valid, _ = load_collab(torch, n.recipe, inside(job['data_root'], root))
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
        'source_class': 'unchanged HashDataset AST',
        'native_split_conversion_source': receipt(__file__),
        'native_split_conversion': 'OGB replace_numpy_with_torchtensor; dtype/order unchanged; no raw cache rewrite',
        'source_cache_files': [
            {'path': str(path), **receipt(path)} for path in sorted(cache.rglob('*.pt'))]}
    (cache/'CONTEXT_BINDING.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return dataset, args, provenance
