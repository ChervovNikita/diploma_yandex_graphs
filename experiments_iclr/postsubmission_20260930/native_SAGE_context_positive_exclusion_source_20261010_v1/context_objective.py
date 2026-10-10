"""Pure caller-supplied NumPy/Torch helpers; no model, data reader or launcher.

Known weighted supervised contrast/false-negative elimination ingredients.
Fixed targets do not establish learned class modes or competence protection.
"""

CONTEXT_NAMES = ("X", "X-PX", "PX", "P^2X")
NEIGHBORS = 16
SHUFFLE_SEED = 9100019
TEMPERATURE = .2
COEFFICIENT = .05


def prepare_targets(np, x, edge_index, train_ids, train_y):
    """Prepare all four factual/shuffled relations from public X/E and TRAIN.

    Full ordered TRAIN is the auxiliary panel. Signatures/cosines use float64;
    native graph/features stay untouched. RNG is a dedicated local generator.
    """
    if x.ndim != 2 or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Finite native float32 public features required")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != np.int64:
        raise ValueError("Directed int64 source/destination COO required")
    if (train_ids.ndim != 1 or train_ids.dtype != np.int64 or train_y.shape != train_ids.shape
            or train_y.dtype != np.int64 or len(np.unique(train_ids)) != len(train_ids)):
        raise ValueError("Exact unique ordered TRAIN IDs and labels required")
    n, b, d = len(x), len(train_ids), x.shape[1]
    if (b < 2 or train_ids.min() < 0 or train_ids.max() >= n or train_y.min() < 0
            or edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n)):
        raise ValueError("TRAIN or public graph index outside its domain")
    classes, counts = np.unique(train_y, return_counts=True)
    if (counts < 2).any():
        raise ValueError("Every TRAIN class needs a nonself same-class positive")
    original_codes = edge_index[0] * n + edge_index[1]
    codes = np.unique(np.concatenate((original_codes, np.arange(n, dtype=np.int64) * (n + 1))))
    src, dst = codes // n, codes % n
    degree = np.bincount(dst, minlength=n)

    def propagate(field):
        output = np.zeros_like(field, dtype=np.float64)
        for first in range(0, len(src), 1024):
            part = slice(first, first + 1024)
            np.add.at(output, dst[part], field[src[part]])
        return output / degree[:, None]

    features = x.astype(np.float64)
    first = propagate(features)
    second = propagate(first)
    contexts = np.stack((features[train_ids], (features - first)[train_ids],
                         first[train_ids], second[train_ids]))
    norms = np.linalg.norm(contexts, axis=-1, keepdims=True)
    normalized = np.divide(contexts, norms, out=np.zeros_like(contexts), where=norms > 0)
    cosine = normalized @ normalized.transpose(0, 2, 1)
    positive = np.zeros((4, b, b), dtype=bool)
    sort_entries = 0
    for context in range(4):
        for row in range(b):
            eligible = np.flatnonzero((train_y == train_y[row]) & (np.arange(b) != row))
            order = np.argsort(-cosine[context, row, eligible], kind="stable")
            positive[context, row, eligible[order[:NEIGHBORS]]] = True
            sort_entries += len(eligible)
    # Whole TRAIN is the panel. Preserve each anchor's positive-relation degree;
    # public-graph degree is not the preserved quantity.
    generator = np.random.default_rng(SHUFFLE_SEED)
    shuffled = np.empty_like(positive)
    permutations = np.empty((4, b), dtype=np.int64)
    for context in range(4):
        permutation = np.arange(b, dtype=np.int64)
        positive_degree = positive[context].sum(-1)
        for cls in classes:
            for count in np.unique(positive_degree[train_y == cls]):
                rows = np.flatnonzero((train_y == cls) & (positive_degree == count))
                permutation[rows] = generator.permutation(rows)
        shuffled[context] = positive[context][permutation[:, None], permutation[None, :]]
        permutations[context] = permutation
        if not np.array_equal(positive[context].sum(-1), shuffled[context].sum(-1)):
            raise ValueError("Incidence shuffle changed an anchor's positive degree")
    changed_by_context = np.count_nonzero(positive != shuffled, axis=(1, 2))
    if (changed_by_context == 0).any():
        raise ValueError("Fixed incidence shuffle is trivial in a context; no retry/repair")
    factual_weights = positive / positive.sum(-1, keepdims=True)
    shuffled_weights = shuffled / shuffled.sum(-1, keepdims=True)
    same_class = train_y[:, None] == train_y[None, :]
    different_class = ~same_class
    retained = positive | different_class[None]
    retained_shuffled = shuffled | different_class[None]
    allowed = ~np.eye(b, dtype=bool)
    if (np.diagonal(positive, axis1=1, axis2=2).any()
            or np.diagonal(shuffled, axis1=1, axis2=2).any()
            or (positive & different_class[None]).any()
            or (shuffled & different_class[None]).any()):
        raise ValueError("Targets must remain nonself same-class")
    metadata = {
        "contexts": list(CONTEXT_NAMES), "neighbors": NEIGHBORS,
        "shuffle_seed": SHUFFLE_SEED, "TRAIN_rows": b, "TRAIN_panel": "all ordered TRAIN",
        "signature_dtype": "float64", "native_features_changed": False,
        "directed_original_edge_entries": edge_index.shape[1],
        "signature_unique_edges_with_exact_self_loops": len(codes),
        "signature_edge_dedup_sort_entries": len(original_codes) + n,
        "feature_aggregate_calls": 2, "feature_aggregate_edge_visits": 2 * len(codes),
        "feature_aggregate_coordinates": 2 * len(codes) * d,
        "signature_gram_calls": 4, "signature_gram_entries": 4 * b * b,
        "signature_gram_multiply_accumulates": 4 * b * b * d,
        "signature_zero_rows_per_context": (norms[..., 0] == 0).sum(-1).tolist(),
        "same_class_stable_sort_entries": sort_entries,
        "class_labels": classes.tolist(), "class_counts": counts.tolist(),
        "positive_counts_per_context": positive.sum((1, 2)).tolist(),
        "retained_counts_per_context": retained.sum((1, 2)).tolist(),
        "excluded_nonself_sameclass_counts_per_context": ((~retained) & allowed[None]).sum((1, 2)).tolist(),
        "changed_incidence_entries_per_context": changed_by_context.tolist(),
        "factual_shuffle_positive_degree_equal": True,
        "shuffle_preserves_public_graph_degree": False,
        "targets_use_VALID_or_predictions": False,
        "native_RNG_draws": 0,
    }
    return {"positive": positive, "shuffled_positive": shuffled,
            "weights": factual_weights, "shuffled_weights": shuffled_weights,
            "retained": retained, "shuffled_retained": retained_shuffled,
            "full_allowed": allowed, "permutations": permutations}, metadata


def tensor_bank(torch, arrays, order, device):
    """Joint row/column reordering aligns the same targets to native H order."""
    output = {}
    for key in ("weights", "shuffled_weights", "retained", "shuffled_retained"):
        values = arrays[key][:, order[:, None], order[None, :]].copy()
        output[key] = torch.as_tensor(values, device=device,
                                      dtype=torch.float32 if "weights" in key else torch.bool)
    output["full_allowed"] = torch.as_tensor(arrays["full_allowed"][order[:, None], order[None, :]].copy(),
                                              device=device, dtype=torch.bool)
    return output


def context_auxiliary(torch, functional, hidden, targets, mode):
    """One live H Gram; no extra model forward or cross-route contrast.

    MIX averages complete (Q,E) losses. It never uses a Qbar union denominator.
    Fixed float32 target row mass multiplies logZ for literal weighted CE.
    """
    modes = ("private", "mixture", "private_full", "shuffled_private")
    if mode not in modes or hidden.ndim != 3 or hidden.shape[0] not in (1, 4) or hidden.shape[1] != 580:
        raise ValueError("Fixed mode and live [1or4,all580,D] native H required")
    members, rows, width = hidden.shape
    if mode != "mixture" and members != 4:
        raise ValueError("Private context allocation is defined only for four native routes")
    weights = targets["shuffled_weights" if mode == "shuffled_private" else "weights"]
    retained = targets["shuffled_retained" if mode == "shuffled_private" else "retained"]
    if weights.shape != (4, rows, rows) or retained.shape != weights.shape:
        raise ValueError("Four aligned fixed target/support pairs required")
    if weights.requires_grad or retained.requires_grad or weights.device != hidden.device or weights.dtype != hidden.dtype:
        raise ValueError("Frozen same-device native-float32 target bank required")
    normalized = functional.normalize(hidden, p=2, dim=-1, eps=1e-12)
    scores = (normalized @ normalized.transpose(-2, -1)) / TEMPERATURE
    if mode == "mixture":
        losses = []
        for context in range(4):
            q, e = weights[context], retained[context]
            log_z = torch.logsumexp(scores.masked_fill(~e[None], float("-inf")), dim=-1)
            losses.append((q.sum(-1)[None] * log_z - (q[None] * scores).sum(-1)).mean(-1))
        pair_losses = torch.stack(losses, dim=-1)
        route_losses = pair_losses.mean(-1)
        reductions = members * 4
    else:
        allowed = targets["full_allowed"][None] if mode == "private_full" else retained
        log_z = torch.logsumexp(scores.masked_fill(~allowed, float("-inf")), dim=-1)
        route_losses = (weights.sum(-1) * log_z - (weights * scores).sum(-1)).mean(-1)
        pair_losses = route_losses[:, None]
        reductions = members
    counts = {
        "context_batched_gram_calls": 1,
        "context_route_grams": members,
        "context_normalized_coordinates": members * rows * width,
        "context_cosine_gram_entries": members * rows * rows,
        "context_gram_multiply_accumulates": members * rows * rows * width,
        "context_score_anchor_rows": members * rows,
        "context_target_support_pair_reductions": reductions,
        "context_logsumexp_anchor_reductions": reductions * rows,
        "context_dense_masked_logsumexp_entries": reductions * rows * rows,
        "context_dense_weighted_score_entries": reductions * rows * rows,
        "context_extra_native_forwards": 0,
        "context_serving_calls": 0,
    }
    return route_losses.mean(), route_losses, pair_losses, counts
