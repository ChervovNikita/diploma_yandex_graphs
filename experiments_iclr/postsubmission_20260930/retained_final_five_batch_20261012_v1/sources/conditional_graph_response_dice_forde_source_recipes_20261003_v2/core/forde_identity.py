"""Prepared, UNEXECUTED author-force FoRDE Identity graph implementation.

Requires immutable native float32 COO P, qualified node-row-local native forwards,
and separate numerical authorization. Gram contraction is exact algebra, not a
new metric. Float32 direct-pullback equivalence remains unqualified.
"""
import math


def _torch():
    import torch
    return torch


def _finite_nonnegative(value, label):
    torch = _torch()
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise FloatingPointError(label + " is nonfinite/negative; no silent clamp")


def native_row_grams(operator, target_ids, maximum_power):
    """G[b,k,l] from a[b,k,:]=row_target(P**k), retaining native dtype.

    Row selection is graph-only. The input operator is neither altered nor
    coalesced. A=[B,K+1,N] is an ephemeral allocation; G is detached graph state.
    """
    torch = _torch()
    if (operator.layout != torch.sparse_coo or operator.dtype != torch.float32
            or operator.requires_grad or operator.ndim != 2
            or operator.shape[0] != operator.shape[1]):
        raise ValueError("Immutable native square float32 COO operator required")
    if (target_ids.ndim != 1 or target_ids.dtype != torch.long
            or target_ids.device != operator.device or target_ids.numel() == 0
            or torch.unique(target_ids).numel() != target_ids.numel()):
        raise ValueError("Distinct device-local target IDs required")
    if type(maximum_power) is not int or maximum_power < 0:
        raise ValueError("Declared nonnegative native maximum power required")
    n, batch = operator.shape[0], target_ids.numel()
    if bool((target_ids < 0).any()) or bool((target_ids >= n).any()):
        raise ValueError("Target ID outside complete graph")
    with torch.no_grad():
        current = torch.zeros((n, batch), dtype=operator.dtype, device=operator.device)
        current[target_ids, torch.arange(batch, device=operator.device)] = 1
        rows = [current.transpose(0, 1)]
        transpose = operator.transpose(0, 1)
        for _ in range(maximum_power):
            current = torch.sparse.mm(transpose, current)
            rows.append(current.transpose(0, 1))
        a = torch.stack(rows, dim=1)
        grams = torch.bmm(a, a.transpose(1, 2))
        if not bool(torch.isfinite(grams).all()):
            raise FloatingPointError("Nonfinite graph row Gram")
    return grams


def _quadratic(coefficients, grams):
    torch = _torch()
    return torch.einsum("bkl,bkf,blf->b", grams, coefficients, coefficients)


def normalize_full_feature_coefficients(coefficients, grams):
    """Normalize q[M,B,K+1,F] in full-X metric, retaining live mixed graph."""
    torch = _torch()
    if (coefficients.ndim != 4 or grams.ndim != 3
            or grams.shape != (coefficients.shape[1], coefficients.shape[2], coefficients.shape[2])
            or coefficients.dtype != torch.float32 or grams.dtype != coefficients.dtype
            or grams.device != coefficients.device or grams.requires_grad):
        raise ValueError("Live float32 coefficients and fixed matching graph Grams required")
    squared_norms = torch.stack([_quadratic(q, grams) for q in coefficients], dim=0)
    _finite_nonnegative(squared_norms, "Full-feature squared norm")
    return coefficients / torch.sqrt(squared_norms[..., None, None] + 1e-24)


def first_argument_distances(normalized_coefficients, grams, reference_coefficients=None):
    """D[i,j,b], with ONLY live first argument differentiated.

    Explicit reference input lets finite-difference oracles freeze source
    stop-gradients across perturbations, rather than recomputing them.
    """
    torch = _torch()
    reference = (normalized_coefficients.detach() if reference_coefficients is None
                 else reference_coefficients.detach())
    if reference.shape != normalized_coefficients.shape:
        raise ValueError("Reference particle shape mismatch")
    distances = []
    for live in normalized_coefficients:
        row = []
        for frozen in reference:
            row.append(_quadratic(live - frozen, grams))
        distances.append(torch.stack(row, dim=0))
    result = torch.stack(distances, dim=0)
    _finite_nonnegative(result, "Full-feature pair distance")
    return result


def author_column_bandwidth(distances):
    """Author jnp.median(D,0), not median over the entire member matrix."""
    torch = _torch()
    if distances.ndim != 3 or distances.shape[0] != distances.shape[1] or distances.shape[0] < 2:
        raise ValueError("D[M_live,M_reference,B] with M>=2 required")
    count = distances.shape[0]
    values = torch.sort(distances.detach(), dim=0).values
    if count % 2:
        median = values[count // 2]
    else:
        median = (values[count // 2 - 1] + values[count // 2]) / 2
    return median / math.log(count) + 1e-12


def author_repulsion(distances, frozen_bandwidth=None):
    """R_sum=(1/B)sum_i log(sum_j mean_b exp(-D_ijb/h_jb))."""
    torch = _torch()
    if distances.ndim != 3 or distances.shape[2] == 0:
        raise ValueError("Nonempty target batch required")
    bandwidth = (author_column_bandwidth(distances) if frozen_bandwidth is None
                 else frozen_bandwidth.detach())
    if bandwidth.shape != distances.shape[1:]:
        raise ValueError("Author bandwidth must have [reference_member,B] shape")
    if not bool(torch.isfinite(bandwidth).all()) or bool((bandwidth <= 0).any()):
        raise FloatingPointError("Invalid bandwidth")
    batch = distances.shape[2]
    log_kde = torch.logsumexp((-distances / bandwidth.unsqueeze(0)).flatten(1), dim=1) - math.log(batch)
    return log_kde.sum() / batch, bandwidth


def source_style_objective(forward_member, native_token_rows, labels, grams, members=4):
    """Return sum-member CE + exact author-force FoRDE Identity surrogate.

    forward_member(tokens,m) must be audited row-local, eval, return raw logits,
    preserve live parameters, and restore native member-selection state.
    It receives native cached target rows; the cache and labels are read only.
    No optimizer action, shared-map update, RNG mutation or guard occurs here.
    """
    torch = _torch()
    if (type(members) is not int or members < 2 or native_token_rows.ndim != 3
            or native_token_rows.dtype != torch.float32 or labels.ndim != 1
            or labels.numel() != native_token_rows.shape[0]):
        raise ValueError("Qualified native rows, target labels and M>=2 required")
    coefficients, ce_terms, logits_all = [], [], []
    for member in range(members):
        token_leaf = native_token_rows.detach().clone().requires_grad_(True)
        logits = forward_member(token_leaf, member)
        if logits.ndim != 2 or logits.shape[0] != labels.numel():
            raise ValueError("One raw-logit vector per selected target required")
        score = logits.gather(1, labels[:, None]).sum()
        q = torch.autograd.grad(score, token_leaf, create_graph=True, retain_graph=True)[0]
        coefficients.append(q)
        ce_terms.append(torch.nn.functional.cross_entropy(logits, labels, reduction="mean"))
        logits_all.append(logits)
    q_all = torch.stack(coefficients, dim=0)
    normalized = normalize_full_feature_coefficients(q_all, grams)
    distances = first_argument_distances(normalized, grams)
    repulsion, bandwidth = author_repulsion(distances)
    ce = torch.stack(ce_terms).sum()
    return ce + repulsion, {
        "ce_sum": ce.detach(), "repulsion_sum": repulsion.detach(),
        "bandwidth": bandwidth.detach(), "logits": torch.stack(logits_all).detach(),
        "distance_shape": tuple(distances.shape), "batch_size": labels.numel(),
    }

