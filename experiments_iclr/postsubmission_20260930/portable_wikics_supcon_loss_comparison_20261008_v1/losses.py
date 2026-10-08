"""Two class-positive graph loss adaptations, with no model/data imports.

The public baseline facade delegates the pinned original cross-view operation.
The standalone cross-view function below supplies a semantic equivalence check.
SupCon is Eq2's outside-log positive average, adapted to the native task-head
input and joint own CE; it is not a reproduction of the original full recipe.
Numerical imports are lazy so source/CLI inspection requires only stdlib.
"""


def _validate(a, b, labels, temperature):
    import torch
    if a.ndim != 3 or a.shape != b.shape or min(a.shape) < 1:
        raise ValueError('Exact paired [member, object, feature] representations')
    if not a.is_floating_point() or a.dtype != b.dtype or a.device != b.device:
        raise ValueError('Matching floating point representation dtype/device')
    if labels.ndim != 1 or labels.shape[0] != a.shape[1] or labels.dtype != torch.long:
        raise ValueError('Exact one-dimensional integer class labels')
    if labels.device != a.device or (labels < 0).any():
        raise ValueError('Nonnegative labels on representation device')
    if not 0 < temperature < float('inf'):
        raise ValueError('Finite positive temperature')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError('Finite paired representations required')


def class_full_cross_view(a, b, labels, temperature=.2):
    """Pinned original WikiCS alignment algebra, mean across both directions.

    Every opposite-view same-class object (including counterpart) is positive;
    all B opposite-view objects enter each denominator. No cross-member pairs.
    """
    import torch
    from torch.nn import functional as F
    _validate(a, b, labels, temperature)
    if a.shape[1] < 2:
        return a.sum() * 0  # Preserve the original operation's small-B guard.
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    scores = torch.einsum('mbd,mcd->mbc', a, b) / temperature
    positives = labels[:, None] == labels[None, :]

    def direction(s):
        lp = F.log_softmax(s, -1)
        return -(lp * positives).sum(-1) / positives.sum(-1)

    return .5 * (direction(scores).mean() + direction(scores.transpose(1, 2)).mean())


def canonical_supcon_eq2(a, b, labels, temperature=.2):
    """Mean SupCon Eq2 over M*2B anchors; exact anchor excluded, both views live.

    Canonical target support is all same-label positions across the two views,
    except the exact anchor. Positive weights are uniform outside the log.
    Self-excluded infinities enter only logsumexp, never positive multiplication.
    F.normalize's original default epsilon/floor is preserved.
    """
    import torch
    from torch.nn import functional as F
    _validate(a, b, labels, temperature)
    z = torch.cat((F.normalize(a, dim=-1), F.normalize(b, dim=-1)), dim=1)
    scores = torch.matmul(z, z.transpose(1, 2)) / temperature
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Finite raw SupCon scores required')
    count = 2 * a.shape[1]
    self_mask = torch.eye(count, dtype=torch.bool, device=a.device)
    y = labels.repeat(2)
    positive = (y[:, None] == y[None, :]) & ~self_mask
    positives_per_anchor = positive.sum(-1)
    # The other view of the same object guarantees a positive for every anchor.
    log_denominator = torch.logsumexp(scores.masked_fill(self_mask, -float('inf')), dim=-1)
    positive_mean_score = (scores * positive).sum(-1) / positives_per_anchor
    result = (log_denominator - positive_mean_score).mean()
    if not torch.isfinite(result):
        raise FloatingPointError('Nonfinite canonical SupCon Eq2')
    return result
