"""Pure caller-supplied NumPy/Torch helpers; no imports, model or launcher.

MLCL and weighted InfoNCE/y-Aware/X-CLR are attributed prior ingredients.
Fixed unequal affinities request ratios, without a collapse or utility guarantee.
"""

CONTEXT_NAMES = ("X", "X-PX", "PX", "P^2X")
TEMPERATURE = .2
TEACHER_TEMPERATURE = .2
COEFFICIENT = .05
TARGET_POLICY = "continuous_all_classmate_full_denominator_v1"


def prepare_targets(np, x, edge_index, train_ids, train_y):
    """Prepare fixed affinities from public X/E and ordered TRAIN only.

    P changes signature preparation only. Every nonself classmate is a positive;
    every nonself TRAIN node belongs to the student denominator. Zero signature
    rows give cosine zero and hence uniform affinities over their classmates.
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
        raise ValueError("Every TRAIN class needs a nonself classmate; no omission or repair")
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
    allowed = ~np.eye(b, dtype=bool)
    positive = (train_y[:, None] == train_y[None, :]) & allowed
    positive_counts = positive.sum(-1)
    unnormalized = np.zeros_like(cosine)
    unnormalized[:, positive] = np.exp(cosine[:, positive] / TEACHER_TEMPERATURE)
    weights = unnormalized / unnormalized.sum(-1, keepdims=True)
    uniform_weights = positive / positive_counts[:, None]
    if (not np.isfinite(weights).all() or not np.isfinite(uniform_weights).all()
            or (weights[:, positive] <= 0).any() or weights[:, ~positive].any()
            or np.diagonal(weights, axis1=1, axis2=2).any()):
        raise ValueError("Finite strictly positive classmate targets with exact zero other/self support required")
    metadata = {
        "target_policy": TARGET_POLICY, "contexts": list(CONTEXT_NAMES),
        "teacher_temperature": TEACHER_TEMPERATURE, "student_temperature": TEMPERATURE,
        "TRAIN_rows": b, "TRAIN_panel": "all ordered TRAIN", "signature_dtype": "float64",
        "native_features_changed": False, "native_predictor_graph_changed": False,
        "positive_support": "every other TRAIN classmate, identical weighted/uniform support",
        "denominator_support": "every other TRAIN node, identical in every arm/context",
        "zero_signature_cosine": 0, "singleton_class_policy": "fail without omission or repair",
        "directed_original_edge_entries": edge_index.shape[1],
        "signature_unique_edges_with_exact_self_loops": len(codes),
        "signature_edge_dedup_sort_entries": len(original_codes) + n,
        "feature_aggregate_calls": 2, "feature_aggregate_edge_visits": 2 * len(codes),
        "feature_aggregate_coordinates": 2 * len(codes) * d,
        "signature_gram_calls": 4, "signature_gram_entries": 4 * b * b,
        "signature_gram_multiply_accumulates": 4 * b * b * d,
        "signature_zero_rows_per_context": (norms[..., 0] == 0).sum(-1).tolist(),
        "teacher_exponential_entries": 4 * int(positive.sum()),
        "teacher_row_normalizations": 4 * b,
        "teacher_dense_row_sum_entries": 4 * b * b,
        "uniform_row_normalizations": b, "uniform_dense_target_entries": b * b,
        "class_labels": classes.tolist(), "class_counts": counts.tolist(),
        "positive_counts_per_anchor": positive_counts.tolist(),
        "positive_counts_per_context": [int(positive.sum())] * 4,
        "full_denominator_entries_per_route": int(allowed.sum()),
        "nonuniform_target_rows_per_context_exact_diagnostic":
            (weights != uniform_weights[None]).any(-1).sum(-1).tolist(),
        "targets_use_VALID_or_predictions": False, "native_RNG_draws": 0,
        "target_or_coefficient_grid": False,
    }
    return {"weights": weights, "uniform_weights": uniform_weights,
            "positive": positive, "full_allowed": allowed}, metadata


def tensor_bank(torch, arrays, order, device):
    """Apply one joint row/column permutation for native bank or single H order."""
    output = {}
    for key in ("weights", "uniform_weights", "positive", "full_allowed"):
        original = arrays[key]
        values = (original[:, order[:, None], order[None, :]] if original.ndim == 3
                  else original[order[:, None], order[None, :]]).copy()
        output[key] = torch.as_tensor(values, device=device,
                                      dtype=torch.float32 if "weights" in key else torch.bool)
    return output


def context_auxiliary(torch, functional, hidden, targets, mode, context_index=None):
    """One live native H Gram with a complete nonself TRAIN denominator.

    Weighted mixture explicitly averages all four outside-log objectives for
    each route, including each independent single/I4 body. The common logZ is
    computed once per route. Float32 Q row mass multiplies logZ literally.
    Assigned I4 body m uses only Q_m, with an explicitly bound context index.
    """
    modes = ("weighted_private", "weighted_mixture", "uniform_allclassmate", "weighted_assigned")
    if (mode not in modes or hidden.ndim != 3 or hidden.shape[0] not in (1, 4)
            or hidden.shape[1:] != (580, 128) or hidden.dtype != torch.float32):
        raise ValueError("Fixed mode and live native float32 [1or4,580,128] H required")
    members, rows, width = hidden.shape
    if mode not in ("weighted_mixture", "weighted_assigned") and members != 4:
        raise ValueError("Private/uniform shared allocation requires four native routes")
    if mode == "weighted_assigned":
        if (members != 1 or not isinstance(context_index, int) or isinstance(context_index, bool)
                or not 0 <= context_index < 4):
            raise ValueError("One independent native body and explicit context index0..3 required")
    elif context_index is not None:
        raise ValueError("Only the assigned independent-body mode accepts a context index")
    weights, uniform, allowed = targets["weights"], targets["uniform_weights"], targets["full_allowed"]
    if weights.shape != (4, rows, rows) or uniform.shape != (rows, rows) or allowed.shape != (rows, rows):
        raise ValueError("Four aligned affinities and common all-classmate/full-denominator targets required")
    if any(value.requires_grad or value.device != hidden.device for value in targets.values()):
        raise ValueError("Frozen same-device target/support bank required")
    if weights.dtype != hidden.dtype or uniform.dtype != hidden.dtype or allowed.dtype != torch.bool:
        raise ValueError("Native float32 target weights and Boolean denominator required")
    normalized = functional.normalize(hidden, p=2, dim=-1, eps=1e-12)
    scores = (normalized @ normalized.transpose(-2, -1)) / TEMPERATURE
    log_z = torch.logsumexp(scores.masked_fill(~allowed[None], float("-inf")), dim=-1)
    if mode == "weighted_mixture":
        losses = []
        for context in range(4):
            q = weights[context]
            losses.append((q.sum(-1)[None] * log_z - (q[None] * scores).sum(-1)).mean(-1))
        pair_losses = torch.stack(losses, dim=-1)
        route_losses = pair_losses.mean(-1)
        reductions, target_matrices = members * 4, 4
    elif mode == "weighted_assigned":
        q = weights[context_index][None]
        route_losses = (q.sum(-1) * log_z - (q * scores).sum(-1)).mean(-1)
        pair_losses = route_losses[:, None]
        reductions, target_matrices = 1, 1
    else:
        q = uniform[None] if mode == "uniform_allclassmate" else weights
        route_losses = (q.sum(-1) * log_z - (q * scores).sum(-1)).mean(-1)
        pair_losses = route_losses[:, None]
        reductions, target_matrices = members, 1 if mode == "uniform_allclassmate" else 4
    counts = {
        "context_batched_gram_calls": 1, "context_route_grams": members,
        "context_normalized_coordinates": members * rows * width,
        "context_cosine_gram_entries": members * rows * rows,
        "context_gram_multiply_accumulates": members * rows * rows * width,
        "context_score_anchor_rows": members * rows,
        "context_target_support_pair_reductions": reductions,
        "context_logsumexp_anchor_reductions": members * rows,
        "context_dense_masked_logsumexp_entries": members * rows * rows,
        "context_dense_weighted_score_entries": reductions * rows * rows,
        "context_target_rowmass_reductions": target_matrices * rows,
        "context_target_rowmass_entries": target_matrices * rows * rows,
        "context_extra_native_forwards": 0, "context_serving_calls": 0,
    }
    return route_losses.mean(), route_losses, pair_losses, counts
