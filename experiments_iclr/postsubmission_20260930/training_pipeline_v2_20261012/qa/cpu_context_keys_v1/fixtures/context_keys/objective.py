"""Caller-supplied NumPy/Torch helpers; no imports or automatic execution.

MLCL and weighted InfoNCE/y-Aware/X-CLR are attributed prior ingredients.
Fixed feature keys/means provide an asymmetric query target; no novelty or utility guarantee.
"""

CONTEXT_NAMES = ("X", "X-PX", "PX", "P^2X")
TEMPERATURE = .2
TEACHER_TEMPERATURE = .2
COEFFICIENT = .05
TARGET_POLICY = "continuous_all_classmate_full_denominator_v1"
KEY_POLICY = "fixed_normalized_graph_context_keys_unit_logZ_v1"
SHUFFLE_SEED = 11711
QUERY_SEED_OFFSET = 5000011
QUERY_WIDTH = 300
HIDDEN_WIDTH = 128


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
    # One prospective class-preserving incidence permutation, common to all
    # contexts/seeds. Only Q axes move. Fixed normalized key rows stay factual.
    generator = np.random.RandomState(SHUFFLE_SEED)
    permutation = np.arange(b, dtype=np.int64)
    for label in classes:
        indices = np.flatnonzero(train_y == label)
        permutation[indices] = generator.permutation(indices)
    shuffled = weights[:, permutation[:, None], permutation[None, :]].copy()
    if (not np.array_equal(np.sort(permutation), np.arange(b))
            or not np.array_equal(train_y[permutation], train_y)
            or shuffled[:, ~positive].any()
            or np.diagonal(shuffled, axis1=1, axis2=2).any()):
        raise ValueError("Fixed class-preserving both-axis Q permutation required")
    metadata.update({
        "key_policy": KEY_POLICY, "key_width": d,
        "fixed_key_coordinates": 4 * b * d,
        "key_normalization_elements": 4 * b * d,
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_moves": "both Q axes only; factual fixed keys unchanged",
        "shuffle_permutation_entries": b,
        "shuffle_target_copy_entries": 4 * b * b,
        "shuffle_support_self_zero_preserved": True,
        "shuffle_preserves_rowmass_multiset_and_class_support": True,
        "target_means": "computed once per actual float32 bank/single order from actual Q/K; no normalization",
        "key_or_teacher_GNN_fits": 0,
    })
    return {"weights": weights, "uniform_weights": uniform_weights,
            "positive": positive, "full_allowed": allowed,
            "keys": normalized, "shuffled_weights": shuffled,
            "shuffle_permutation": permutation,
            "train_ids": train_ids.copy(), "train_y": train_y.copy()}, metadata


def tensor_bank(torch, arrays, order, device):
    """Bind factual keys and both Q axes to the actual native H row order.

    Moments are literal actual float32 Q @ K reductions, paid once for each
    bank/single order. They are not normalized and not cast float64 moments.
    """
    output = {}
    for key in ("weights", "shuffled_weights", "uniform_weights", "positive", "full_allowed"):
        original = arrays[key]
        values = (original[:, order[:, None], order[None, :]] if original.ndim == 3
                  else original[order[:, None], order[None, :]]).copy()
        output[key] = torch.as_tensor(values, device=device,
            dtype=torch.float32 if "weights" in key else torch.bool)
    output["keys"] = torch.as_tensor(arrays["keys"][:, order].copy(), device=device, dtype=torch.float32)
    for key in ("train_ids", "train_y"):
        output[key] = torch.as_tensor(arrays[key][order].copy(), device=device, dtype=torch.long)
    inverse = arrays["shuffle_permutation"].copy()
    inverse[order] = range(len(order))
    output["shuffle_permutation"] = torch.as_tensor(
        inverse[arrays["shuffle_permutation"][order]].copy(), device=device, dtype=torch.long)
    # Exact self exclusion is checked by factual node identity after reordering.
    by_id = output["train_ids"][:, None].ne(output["train_ids"][None, :])
    by_class = output["train_y"][:, None].eq(output["train_y"][None, :]) & by_id
    if not torch.equal(output["full_allowed"], by_id) or not torch.equal(output["positive"], by_class):
        raise ValueError("Transferred TRAIN identities/support/order mismatch")
    output["moments"] = torch.bmm(output["weights"], output["keys"])
    output["shuffled_moments"] = torch.bmm(output["shuffled_weights"], output["keys"])
    output["uniform_moments"] = torch.bmm(
        output["uniform_weights"][None].expand(4, -1, -1), output["keys"])
    return output


def make_query_map(torch, functional, seed, device):
    """A bare shared 128->300 map, separately initialized without global draws.

    Every new arm/body at one paired seed receives identical initial A bytes;
    independent bodies own separate parameters, optimizers and learned states.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed + QUERY_SEED_OFFSET)
    raw = torch.randn(QUERY_WIDTH, HIDDEN_WIDTH, generator=generator, dtype=torch.float64, device="cpu")
    basis, upper = torch.linalg.qr(raw, mode="reduced")
    sign = torch.where(upper.diagonal() < 0, -torch.ones(HIDDEN_WIDTH, dtype=torch.float64, device="cpu"),
                       torch.ones(HIDDEN_WIDTH, dtype=torch.float64, device="cpu"))
    initial = (basis * sign[None]).to(dtype=torch.float32, device=device)

    class QueryMap(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight)

        def forward(self, hidden):
            return functional.linear(hidden, self.weight, bias=None)

    return QueryMap(initial)


def context_auxiliary(torch, functional, hidden, targets, mode, context_index=None, query_map=None):
    """Complete key objectives on one live factual native pre-head H.

    Unit logZ coefficient is the prospectively requested fixed-key formula;
    actual inherited Q row masses are custody diagnostics, not loss weights.
    Peer queries/keys are both live. All fixed buffers remain detached.
    """
    modes = ("key_private", "key_mixture", "key_uniform", "key_assigned",
             "projected_peer_private", "key_moment_private", "key_shuffled_private")
    if (mode not in modes or hidden.ndim != 3 or hidden.shape[0] not in (1, 4)
            or hidden.shape[1:] != (580, 128) or hidden.dtype != torch.float32):
        raise ValueError("Fixed mode and native float32 [1or4,580,128] H required")
    members, rows, width = hidden.shape
    if mode not in ("key_mixture", "key_assigned") and members != 4:
        raise ValueError("Private/key controls require four shared native routes")
    if mode == "key_assigned":
        if (members != 1 or not isinstance(context_index, int) or isinstance(context_index, bool)
                or not 0 <= context_index < 4):
            raise ValueError("One independent body with explicit assigned index0..3 required")
    elif context_index is not None:
        raise ValueError("Only assigned independent bodies accept a context index")
    if (query_map is None or tuple(query_map.weight.shape) != (300, 128)
            or query_map.weight.dtype != hidden.dtype or query_map.weight.device != hidden.device):
        raise ValueError("One bare same-device float32 128->300 query map required")
    if any(v.requires_grad or v.device != hidden.device for v in targets.values()):
        raise ValueError("Frozen same-device key/target/order buffers required")
    keys, weights, allowed = targets["keys"], targets["weights"], targets["full_allowed"]
    if (keys.shape != (4, rows, 300) or weights.shape != (4, rows, rows)
            or allowed.shape != (rows, rows) or allowed.dtype != torch.bool
            or keys.dtype != hidden.dtype or weights.dtype != hidden.dtype):
        raise ValueError("Four aligned fixed key/target pairs and exact nonself denominator required")
    queries = functional.normalize(query_map(hidden), p=2, dim=-1, eps=1e-12)
    counts = {"context_query_projection_calls": 1,
        "context_query_projection_rows": members * rows,
        "context_query_projection_MACs": members * rows * width * 300,
        "context_normalized_query_coordinates": members * rows * 300,
        "context_dictionary_score_calls": 0, "context_dictionary_score_matrices": 0,
        "context_dictionary_score_entries": 0, "context_dictionary_score_MACs": 0,
        "context_logsumexp_anchor_reductions": 0,
        "context_dense_masked_logsumexp_entries": 0,
        "context_fixed_moment_dot_entries": 0,
        "context_live_peer_target_score_entries": 0,
        "context_complete_objective_anchor_rows": 0,
        "context_extra_native_forwards": 0, "context_teacher_GNN_forwards": 0,
        "context_serving_calls": 0}
    if mode == "projected_peer_private":
        # q on both sides; no detach of peer keys or their incoming gradients.
        scores = torch.bmm(queries, queries.transpose(-2, -1)) / TEMPERATURE
        log_z = torch.logsumexp(scores.masked_fill(~allowed[None], float("-inf")), dim=-1)
        route_losses = (log_z - (weights * scores).sum(-1)).mean(-1)
        pair_losses = route_losses[:, None]
        matrices = members
        counts["context_live_peer_target_score_entries"] = members * rows * rows
    elif mode == "key_moment_private":
        route_losses = ((1 - (queries * targets["moments"]).sum(-1)) / TEMPERATURE).mean(-1)
        pair_losses = route_losses[:, None]
        matrices = 0
        counts["context_fixed_moment_dot_entries"] = members * rows * 300
    elif mode == "key_mixture":
        # Each dictionary has its own log denominator. Projection is reused.
        complete = []
        for context in range(4):
            scores = queries @ keys[context].transpose(-2, -1) / TEMPERATURE
            log_z = torch.logsumexp(scores.masked_fill(~allowed[None], float("-inf")), dim=-1)
            positive = (queries * targets["moments"][context][None]).sum(-1) / TEMPERATURE
            complete.append((log_z - positive).mean(-1))
        pair_losses = torch.stack(complete, dim=-1)
        route_losses = pair_losses.mean(-1)
        matrices = members * 4
        counts["context_fixed_moment_dot_entries"] = matrices * rows * 300
    else:
        if mode == "key_assigned":
            selected_keys = keys[context_index][None]
            moments = targets["moments"][context_index][None]
        else:
            selected_keys = keys
            moments = targets[{"key_private": "moments", "key_uniform": "uniform_moments",
                               "key_shuffled_private": "shuffled_moments"}[mode]]
        scores = torch.bmm(queries, selected_keys.transpose(-2, -1)) / TEMPERATURE
        log_z = torch.logsumexp(scores.masked_fill(~allowed[None], float("-inf")), dim=-1)
        positive = (queries * moments).sum(-1) / TEMPERATURE
        route_losses = (log_z - positive).mean(-1)
        pair_losses = route_losses[:, None]
        matrices = members
        counts["context_fixed_moment_dot_entries"] = members * rows * 300
    counts.update({"context_dictionary_score_calls": 4 if mode == "key_mixture" else int(matrices > 0),
        "context_dictionary_score_matrices": matrices,
        "context_dictionary_score_entries": matrices * rows * rows,
        "context_dictionary_score_MACs": matrices * rows * rows * 300,
        "context_logsumexp_anchor_reductions": matrices * rows,
        "context_dense_masked_logsumexp_entries": matrices * rows * rows,
        "context_complete_objective_anchor_rows": (members * 4 if mode == "key_mixture" else members) * rows})
    return route_losses.mean(), route_losses, pair_losses, counts
