"""Pinned BUDDY preprocessing with official Collab splits and bounded caches."""

import contextlib
import fcntl
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, negative_sampling, to_undirected
from torch_sparse import spmm

from ablation.common.resumable import atomic_torch_save, capture_rng, restore_rng
from ablation.common.worker import Preempted
from .protocol import REFERENCE, RECIPE, file_hash, inside_repository, namespace

if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
from src.hashing import ElphHashes


def dataset_identity(data_dir):
    root = inside_repository(data_dir) / 'ogbl_collab'
    files = sorted((root / 'raw').glob('*')) + sorted((root / 'split').rglob('*.pt'))
    if not files:
        raise FileNotFoundError(f'Install official ogbl-collab under {root} before starting BUDDY')
    hashes = {str(path.relative_to(root)): file_hash(path) for path in files if path.is_file()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def tensor_identity(tensor):
    tensor = tensor.contiguous().cpu()
    digest = hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def cache_lock(path, store, stop):
    """An advisory lock is released by the kernel after a crashed preprocessor."""
    with path.open('a') as stream:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                store.heartbeat()
                if stop.requested:
                    raise Preempted()
                time.sleep(2)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def bounded_hash_tables(hashes, num_nodes, edge_index, batch_size, heartbeat):
    """Native sketch propagation and estimators with bounded edge messages."""
    edges, _ = add_self_loops(edge_index)
    tables = {0: {'minhash': hashes.initialise_minhash(num_nodes),
                  'hll': hashes.initialise_hll(num_nodes)}}
    cards = torch.zeros((num_nodes, hashes.max_hops))
    for hop in range(1, hashes.max_hops + 1):
        previous = tables[hop - 1]
        current = {
            'minhash': torch.full_like(previous['minhash'], torch.iinfo(torch.int64).max),
            'hll': torch.zeros_like(previous['hll']),
        }
        # This is the same min/max neighborhood reduction as native MessagePassing.
        for batch in edges.split(batch_size, dim=1):
            source, target = batch
            current['minhash'].scatter_reduce_(
                0, target[:, None].expand(-1, hashes.num_perm), previous['minhash'][source],
                reduce='amin', include_self=True)
            current['hll'].scatter_reduce_(
                0, target[:, None].expand(-1, hashes.hll_size), previous['hll'][source],
                reduce='amax', include_self=True)
            heartbeat()
        # Native MessagePassing returns zero for nodes with no incoming edge.
        current['minhash'].masked_fill_(current['minhash'] == torch.iinfo(torch.int64).max, 0)
        tables[hop] = current
        for start in range(0, num_nodes, batch_size):
            cards[start:start + batch_size, hop - 1] = hashes.hll_count(current['hll'][start:start + batch_size])
            heartbeat()
    return tables, cards


def make_graph(x, positive_edges, weights, heartbeat, chunk_size):
    edges, weights = to_undirected(positive_edges.t(), weights.flatten().float(),
                                   num_nodes=x.size(0), reduce='add')
    # Collab has symmetric edges, so column sums equal the native scipy degree.
    degrees = torch.zeros(x.size(0), dtype=torch.float64)
    degrees.index_add_(0, edges[1], weights.to(torch.float64))
    normalized_edges, normalized_weights = gcn_norm(edges, weights, x.size(0))
    propagated = spmm(normalized_edges, normalized_weights, x.size(0), x.size(0), x)
    heartbeat()
    hashes = ElphHashes(namespace())
    tables, cards = bounded_hash_tables(hashes, x.size(0), edges, chunk_size, heartbeat)
    return {'x': propagated, 'degrees': degrees.float(), 'edge_index': edges}, hashes, tables, cards


def hydrate(graph, hashes, tables, cards, positive, negative, chunk_size, heartbeat):
    links = torch.cat((positive, negative), dim=0)
    features = []
    for chunk in links.split(chunk_size):
        features.append(hashes.get_subgraph_features(chunk, tables, cards, chunk_size))
        heartbeat()
    return {
        'links': links, 'subgraph_features': torch.cat(features).float(),
        'x': graph['x'], 'degrees': graph['degrees'], 'num_positive': len(positive),
        'edge_order_sha256': tensor_identity(links),
    }


def load_official(data_dir, smoke):
    dataset = PygLinkPropPredDataset(name='ogbl-collab', root=str(inside_repository(data_dir)))
    graph = dataset[0]
    split = dataset.get_edge_split()
    selected = split['train']['year'].reshape(-1) >= RECIPE['year']
    positive = split['train']['edge'][selected]
    weights = split['train']['weight'][selected].reshape(-1)
    valid = {key: split['valid'][key] for key in ('edge', 'edge_neg', 'weight')}
    return graph.x, positive, weights, valid


def prepare_training(config, args, store, stop):
    """Create shared immutable TRAIN/VALID data without evaluating TEST."""
    identity = {name: config[name] for name in ('source_sha256', 'dataset_sha256', 'recipe', 'dependencies')}
    identity['train_edge_cap'] = config['smoke']['train_edges'] if config['smoke'] else 0
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = inside_repository(args.cache_dir) / cache_key
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'train_valid.pt'

    def heartbeat():
        store.heartbeat()
        if stop.requested:
            raise Preempted()

    with cache_lock(root / 'train_valid.lock', store, stop):
        if path.exists():
            bundle = torch.load(path, map_location='cpu', weights_only=False)
            if bundle['identity'] != identity:
                raise RuntimeError('BUDDY preprocessing cache identity mismatch')
            return bundle, root
        cpu = torch.device('cpu')
        rng = capture_rng(cpu)
        try:
            random.seed(RECIPE['preprocessing_seed'])
            np.random.seed(RECIPE['preprocessing_seed'])
            torch.manual_seed(RECIPE['preprocessing_seed'])
            x, positive, weights, valid = load_official(args.data_dir, config['smoke'])
            graph, hashes, tables, cards = make_graph(x, positive, weights, heartbeat, args.preprocess_batch_size)
            if config['smoke'] and config['smoke']['train_edges']:
                positive = positive[:config['smoke']['train_edges']]
            exclusions, _ = add_self_loops(graph['edge_index'], num_nodes=x.size(0))
            negative = negative_sampling(exclusions, num_nodes=x.size(0), num_neg_samples=len(positive)).t()
            if len(negative) != len(positive):
                raise RuntimeError('The Collab graph did not yield one negative per positive')
            bundle = {
                'identity': identity,
                'train': hydrate(graph, hashes, tables, cards, positive, negative,
                                 args.preprocess_batch_size, heartbeat),
                'valid': hydrate(graph, hashes, tables, cards, valid['edge'], valid['edge_neg'],
                                 args.preprocess_batch_size, heartbeat),
            }
            atomic_torch_save(bundle, path)
        finally:
            restore_rng(rng, cpu)
    return bundle, root


def prepare_test(config, args, cache_root, store, stop):
    """Only called after a checkpoint has been selected using validation."""
    if config['smoke']:
        raise RuntimeError('Smoke runs cannot access TEST preprocessing')
    path = cache_root / 'test.pt'

    def heartbeat():
        store.heartbeat()
        if stop.requested:
            raise Preempted()

    with cache_lock(cache_root / 'test.lock', store, stop):
        if path.exists():
            return torch.load(path, map_location='cpu', weights_only=False)
        x, positive, weights, valid = load_official(args.data_dir, None)
        if RECIPE['use_valedges_as_input']:
            positive = torch.cat((positive, valid['edge']))
            weights = torch.cat((weights, valid['weight'].flatten()))
        graph, hashes, tables, cards = make_graph(x, positive, weights, heartbeat, args.preprocess_batch_size)
        dataset = PygLinkPropPredDataset(name='ogbl-collab', root=str(inside_repository(args.data_dir)))
        test = dataset.get_edge_split()['test']
        result = hydrate(graph, hashes, tables, cards, test['edge'], test['edge_neg'],
                         args.preprocess_batch_size, heartbeat)
        atomic_torch_save(result, path)
        return result
