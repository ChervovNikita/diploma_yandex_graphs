"""Fixed TRAIN-label positive relations from attributed graph filter contexts.

Numpy reference preparation only. Native predictor topology is unchanged.
No validation labels or model predictions are accepted by this interface.
"""
import numpy as np


def graph_contexts(x, edge_index, train_ids, chunk_edges=1024):
    """Return TRAIN rows of [X, X-PX, PX, P^2X], with P row-normalized A+I.

    Caller supplies the public graph and feature preprocessing used by its
    backbone. This signature computation does not alter the predictor graph.
    Duplicate ordered edges are removed, then exactly one self-loop is added.
    Directed input is not silently symmetrized. Destination rows aggregate
    source features; the edge convention is explicit and must be bound.
    """
    x = np.asarray(x)
    edges = np.asarray(edge_index)
    ids = np.asarray(train_ids)
    if x.ndim != 2 or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError('finite floating point node features required')
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('integer [source,destination] edges required')
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError('unique integer TRAIN IDs required')
    n = x.shape[0]
    if n < 1 or np.any(edges < 0) or np.any(edges >= n) or np.any(ids < 0) or np.any(ids >= n):
        raise ValueError('object or edge ID outside graph')
    if chunk_edges < 1:
        raise ValueError('positive preparation chunk required')
    loops = np.arange(n, dtype=np.int64)
    encoded = edges[0].astype(np.int64) * n + edges[1]
    encoded = np.unique(np.concatenate([encoded, loops*n+loops]))
    src, dst = encoded // n, encoded % n
    degree = np.bincount(dst, minlength=n)
    def propagate(field):
        out = np.zeros_like(field)
        for first in range(0, len(src), chunk_edges):
            selected = slice(first, first+chunk_edges)
            np.add.at(out, dst[selected], field[src[selected]])
        return out / degree[:, None]
    first = propagate(x)
    second = propagate(first)
    return np.stack([x[ids], (x-first)[ids], first[ids], second[ids]])


def positive_masks(contexts, train_labels, neighbors=16):
    """Four directed top-k same-class relations, always including self.

    Stable descending sort resolves equal cosine similarities using TRAIN row
    order. Zero signature rows have zero similarities; no outcome-dependent
    fallback is allowed. All same-class rows not selected are denominator
    distractors in the proposed objective, a deliberate and disclosed change.
    """
    contexts = np.asarray(contexts)
    labels = np.asarray(train_labels)
    if contexts.ndim != 3 or contexts.shape[0] != 4 or contexts.shape[1] != len(labels):
        raise ValueError('exact [4,TRAIN,D] contexts and TRAIN labels required')
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('finite nonnegative integer TRAIN classes required')
    if not np.isfinite(contexts).all() or neighbors < 1:
        raise ValueError('finite contexts and positive neighbor count required')
    norm = np.linalg.norm(contexts, axis=-1, keepdims=True)
    normalized = np.divide(contexts, norm, out=np.zeros_like(contexts, dtype=np.float64), where=norm > 0)
    similarities = normalized @ normalized.transpose(0, 2, 1)
    n = len(labels)
    mask = np.zeros((4, n, n), dtype=bool)
    for route in range(4):
        for row in range(n):
            eligible = np.flatnonzero((labels == labels[row]) & (np.arange(n) != row))
            order = np.argsort(-similarities[route, row, eligible], kind='stable')
            mask[route, row, eligible[order[:neighbors]]] = True
            mask[route, row, row] = True
    return mask


def row_weights(masks):
    """Normalized positive weights; self ensures every row has mass."""
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.shape[0] != 4 or masks.shape[1] != masks.shape[2] or masks.dtype != bool:
        raise ValueError('four square Boolean masks required')
    if not np.all(np.diagonal(masks, axis1=1, axis2=2)):
        raise ValueError('own-object cross-view positive required')
    return masks / masks.sum(-1, keepdims=True)


def within_class_permuted(masks, train_labels, seed, panel_rows):
    """Independent fixed class-preserving permutation of each route relation.

    Preserves label and original panel membership. Within the scored panel,
    further restrict each route permutation by restricted positive degree.
    Thus every scored anchor preserves its exact positive count after panel
    restriction, as well as self/label. A column-degree multiset is preserved
    but its assignment to individual nodes can change. Randomization is fixed
    prospectively, not selected by model outcomes.
    """
    labels = np.asarray(train_labels)
    masks = np.asarray(masks)
    row_weights(masks)
    if len(labels) != masks.shape[1]:
        raise ValueError('TRAIN labels must match mask identities')
    panel = np.asarray(panel_rows)
    if panel.ndim != 1 or not np.issubdtype(panel.dtype,np.integer) or len(np.unique(panel)) != len(panel):
        raise ValueError('exact unique integer scored-panel rows required')
    if len(panel)<1 or np.any(panel<0) or np.any(panel>=len(labels)):
        raise ValueError('invalid scored panel')
    in_panel=np.zeros(len(labels),dtype=bool);in_panel[panel]=True
    generator = np.random.default_rng(seed)
    out = np.empty_like(masks)
    permutations = []
    for route in range(4):
        permutation = np.arange(len(labels))
        restricted_degrees = masks[route][:,panel].sum(-1)
        for cls in np.unique(labels):
            outside=np.flatnonzero((labels==cls)&~in_panel)
            permutation[outside]=generator.permutation(outside)
            class_panel=(labels==cls)&in_panel
            for degree in np.unique(restricted_degrees[class_panel]):
                ids=np.flatnonzero(class_panel&(restricted_degrees==degree))
                permutation[ids]=generator.permutation(ids)
        out[route] = masks[route][permutation[:, None], permutation[None, :]]
        permutations.append(permutation)
    return out, np.stack(permutations)


def sampled_weights(masks, sampled_train_rows, mode, update_index=None):
    """Restrict exact masks to original auxiliary TRAIN sample, normalize after.

    route: route-m gets relation-m. common: every route gets the arithmetic mean
    of four row-normalized relations. Their aggregate positive target mass is
    exactly equal. cycle: every route gets relation update_index modulo four;
    this is a separate temporal control, not equal per-update target mass.
    """
    masks = np.asarray(masks)
    rows = np.asarray(sampled_train_rows)
    row_weights(masks)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(np.unique(rows)) != len(rows):
        raise ValueError('unique integer auxiliary TRAIN rows required')
    if len(rows) < 1 or np.any(rows < 0) or np.any(rows >= masks.shape[1]):
        raise ValueError('invalid auxiliary TRAIN sample')
    restricted = masks[:, rows[:, None], rows[None, :]]
    weights = row_weights(restricted)
    if mode == 'route':
        return weights
    if mode == 'common':
        return np.broadcast_to(weights.mean(0, keepdims=True), weights.shape).copy()
    if mode == 'cycle':
        if not isinstance(update_index, int) or update_index < 0:
            raise ValueError('cycle requires a nonnegative integer update index')
        chosen = weights[update_index % 4]
        return np.broadcast_to(chosen[None], weights.shape).copy()
    raise ValueError('mode must be route, common or cycle')
