"""Same fixed ROUTE targets; omit only unselected same-class denominator entries.

No COMMON symmetry, competence, scientific quality or novelty claim.
"""


def retained_entries(weights, labels):
    import torch
    if weights.ndim != 3 or weights.shape[-2] != weights.shape[-1]:
        raise ValueError('Exact member/anchor/candidate target axes required')
    if labels.ndim != 1 or len(labels) != weights.shape[-1] or labels.dtype != torch.long:
        raise ValueError('Original ordered auxiliary TRAIN class labels required')
    positive = weights > 0
    different_class = labels[:, None] != labels[None, :]
    if (positive & different_class[None]).any():
        raise ValueError('Original positives must remain same-class')
    retained = positive | different_class[None]
    if not retained.any(-1).all():
        raise ValueError('Every anchor must retain its original positives')
    return retained


def directional_target_ce(scores, weights, retained):
    """Finite weighted CE; -inf occurs only in logZ, never in target products."""
    import torch
    if scores.shape != weights.shape or retained.shape != scores.shape or retained.dtype != torch.bool:
        raise ValueError('Exact score/target/retained axes required')
    if not torch.isfinite(scores).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Finite scores and frozen nonnegative targets required')
    if weights.requires_grad or retained.requires_grad or not retained.any(-1).all():
        raise ValueError('Frozen nonempty retained sets required')
    if (weights[~retained] != 0).any():
        raise ValueError('No original target mass may be removed')
    mass = weights.sum(-1)
    if not torch.allclose(mass, torch.ones_like(mass), atol=1e-6, rtol=1e-6):
        raise ValueError('Original row-normalized targets required')
    log_z = torch.logsumexp(scores.masked_fill(~retained, -torch.inf), dim=-1)
    # Use the stored row mass to preserve literal weighted-CE behavior under
    # float32 row-sum rounding. Every target product uses the finite scores.
    return (mass * log_z - (weights * scores).sum(-1)).mean()


def masked_context_alignment(a, b, positive_weights, labels, temperature=.2, retained=None):
    import torch
    from torch.nn import functional as F
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] not in (1, 4):
        raise ValueError('Original paired member/object/feature axes required')
    if temperature != .2:
        raise ValueError('Original fixed temperature0.2 required')
    weights = torch.as_tensor(positive_weights, device=a.device, dtype=a.dtype)
    if weights.shape != (a.shape[0], a.shape[1], a.shape[1]):
        raise ValueError('Original exact auxiliary targets required')
    labels = labels.to(device=a.device)
    expected = retained_entries(weights, labels)
    if retained is None:
        retained = expected
    elif not torch.equal(retained, expected):
        raise ValueError('Retained entries differ from original targets/classes')
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    scores = torch.einsum('mbd,mcd->mbc', a, b) / temperature
    # Directed target i->j is the same relation in either anchor view. The
    # scores transpose; the anchor-indexed target/retained mask does not.
    return .5 * (directional_target_ce(scores, weights, retained) +
                 directional_target_ce(scores.transpose(1, 2), weights, retained))
