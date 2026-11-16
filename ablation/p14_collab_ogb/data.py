"""Official ogbl-collab splits as edge lists for full-graph GCN and GraphSAGE."""

import hashlib
import json

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import coalesce, to_undirected

from .protocol import file_hash, inside_repository


def dataset_identity(data_dir):
    root = inside_repository(data_dir) / 'ogbl_collab'
    files = sorted((root / 'raw').glob('*')) + sorted((root / 'split').rglob('*.pt'))
    if not files:
        raise FileNotFoundError(f'Install official ogbl-collab under {root}')
    hashes = {str(path.relative_to(root)): file_hash(path) for path in files if path.is_file()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def tensor_identity(tensor):
    tensor = tensor.contiguous().cpu()
    digest = hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def message_graph(edges, weights, num_nodes):
    """Undirected graph with summed collaboration counts and GCN weights."""
    edge_index, weight = to_undirected(edges.t(), weights.reshape(-1).float(),
                                       num_nodes=num_nodes, reduce='add')
    edge_index, weight = coalesce(edge_index, weight, num_nodes=num_nodes, reduce='add')
    gcn_index, gcn_weight = gcn_norm(edge_index, weight, num_nodes, add_self_loops=True)
    return {'edge_index': edge_index, 'gcn_index': gcn_index, 'gcn_weight': gcn_weight}


def evaluation_links(positive, negative):
    links = torch.cat((positive, negative), dim=0)
    return {'links': links, 'num_positive': len(positive), 'edge_order_sha256': tensor_identity(links)}


def build_bundle(x, split, with_test):
    """TRAIN positives, VALID links on the training graph and, after selection,
    TEST links on the training graph plus validation edges."""
    num_nodes = x.size(0)
    train = split['train']
    bundle = {
        'x': x.float(), 'num_nodes': num_nodes,
        'train_positive': train['edge'],
        'train_graph': message_graph(train['edge'], train['weight'], num_nodes),
        'valid': evaluation_links(split['valid']['edge'], split['valid']['edge_neg']),
    }
    if with_test:
        edges = torch.cat((train['edge'], split['valid']['edge']))
        weights = torch.cat((train['weight'].reshape(-1), split['valid']['weight'].reshape(-1)))
        bundle['test_graph'] = message_graph(edges, weights, num_nodes)
        bundle['test'] = evaluation_links(split['test']['edge'], split['test']['edge_neg'])
    return bundle


def load_bundle(data_dir, with_test=False):
    from ogb.linkproppred import PygLinkPropPredDataset
    dataset = PygLinkPropPredDataset(name='ogbl-collab', root=str(inside_repository(data_dir)))
    return build_bundle(dataset[0].x, dataset.get_edge_split(), with_test)
