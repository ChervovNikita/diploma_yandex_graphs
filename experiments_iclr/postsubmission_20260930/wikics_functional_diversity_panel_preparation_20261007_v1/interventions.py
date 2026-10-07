"""Fixed public-input interventions; no predictor, labels, RNG or fitted stats."""
import hashlib

NAMESPACE = 'wikics-functional-panel-20261007-v1'
LEVELS = (10, 25)
NODES, FEATURES = 11701, 300


def array_digest(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str((value.shape, value.dtype)).encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def variants(x, edge_index):
    """Yield baseline plus four fixed variants; original arrays are never written."""
    import numpy as np
    if x.shape != (NODES, FEATURES) or x.dtype != np.float32:
        raise ValueError('Exact public WikiCS feature table')
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != np.int64:
        raise ValueError('Exact prepared directed integer edge table')
    if not np.isfinite(x).all() or edge_index.min() < 0 or edge_index.max() >= NODES:
        raise ValueError('Finite original inputs and public node-ID domain')
    keys = edge_index.min(0) * NODES + edge_index.max(0)
    pairs, inverse = np.unique(keys, return_inverse=True)
    scores = np.array([int.from_bytes(hashlib.sha256(
        (NAMESPACE + '|undirected_pair|' + str(int(k))).encode('ascii')).digest()[:8],
        'big') for k in pairs], dtype=np.uint64)
    loops = edge_index[0] == edge_index[1]
    ranks = sorted(range(FEATURES), key=lambda j: hashlib.sha256(
        (NAMESPACE + '|feature_column|' + str(j)).encode('ascii')).digest())
    yield 'baseline', x, edge_index, {'kind': 'none'}
    for percent in LEVELS:
        drop = (scores[inverse] < np.uint64(percent * (1 << 64) // 100)) & ~loops
        keep = ~drop
        edges = np.ascontiguousarray(edge_index[:, keep])
        yield 'edge_thin_' + str(percent), x, edges, {
            'kind': 'undirected_edge_thinning', 'percent_threshold': percent,
            'kept_entry_mask_sha256': array_digest(keep),
            'removed_directed_entries': int(drop.sum()),
            'removed_distinct_nonself_pairs': int(len(np.unique(keys[drop]))),
            'selfloops_retained': int((edges[0] == edges[1]).sum())}
    for percent in LEVELS:
        columns = ranks[:FEATURES * percent // 100]
        features = x.copy()
        features[:, columns] = 0.0
        yield 'feature_zero_' + str(percent), features, edge_index, {
            'kind': 'fixed_feature_column_zeroing', 'percent': percent,
            'columns': columns, 'column_count': len(columns),
            'fill': 0.0, 'all_public_rows_masked': True}


def materialization(x, edge_index):
    rows = {}
    for name, features, edges, description in variants(x, edge_index):
        rows[name] = {'description': description,
            'x': {'shape': list(features.shape), 'dtype': str(features.dtype),
                  'sha256': array_digest(features)},
            'edge_index': {'shape': list(edges.shape), 'dtype': str(edges.dtype),
                           'sha256': array_digest(edges)}}
    return rows
