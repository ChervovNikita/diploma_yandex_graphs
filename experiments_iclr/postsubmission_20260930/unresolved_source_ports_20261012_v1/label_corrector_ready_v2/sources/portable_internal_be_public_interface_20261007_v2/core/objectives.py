"""Explicit graph adaptation; not a DICE/CDLG reproduction or MI estimator."""
import torch
from torch.nn import functional as F


def own_supervision(logits, labels, task):
    if logits.ndim != 3:
        raise ValueError("Expected member, object, output axes")
    if task == "wikics":
        return torch.stack([F.cross_entropy(x, labels) for x in logits])
    target = labels.reshape(-1, 1).to(logits)
    if not torch.isfinite(target).all():
        raise ValueError("No missing/bad finite target may be silently dropped")
    losses = F.binary_cross_entropy_with_logits(logits, target.expand_as(logits), reduction="none")
    if task == "collab":
        # Retain the native sum of positive and negative mean logistic losses.
        pos, neg = target[:, 0] == 1, target[:, 0] == 0
        if not pos.any() or not neg.any():
            raise ValueError("Both edge labels required")
        return losses[:, pos].mean((1, 2)) + losses[:, neg].mean((1, 2))
    return losses.mean((1, 2))


def alignment_loss(a, b, labels, task, temperature=.2, identities=None):
    """Same member across two stochastic views, symmetric contrastive loss.

    WikiCS/molecule: all same TRAIN-label objects are positives in the opposite
    view (supervised contrast). Collab: all records of the same canonical target edge are positives;
    other edges are instance negatives, not asserted semantic class equivalents.
    No validation/test labels or graph changes enter these pair targets.
    """
    if a.shape != b.shape or a.ndim != 3 or len(labels) != a.shape[1]:
        raise ValueError("Exact aligned member/object rows required")
    if a.shape[1] < 2:
        return a.sum() * 0
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    scores = torch.einsum("mbd,mcd->mbc", a, b) / temperature
    if task=='collab':
        if identities is None:
            raise ValueError('TRAIN canonical target identities required')
        canonical=identities.sort(dim=1).values
        positives=(canonical[:,None,:]==canonical[None,:,:]).all(-1)
        if ((labels[:,None]!=labels[None,:]) & positives).any():
            raise ValueError('Contradictory labels for same canonical target')
    else:
        positives=labels[:,None]==labels[None,:]
    def direction(s):
        lp = F.log_softmax(s, -1)
        return -(lp * positives).sum(-1) / positives.sum(-1)
    return .5 * (direction(scores).mean() + direction(scores.transpose(1, 2)).mean())


def residual_member_contrast(a, b, labels, temperature=.2):
    """CDLG-inspired same-object cross-member negatives after TRAIN class centering.

    Positive: same member/object across views. Negatives: other members, same
    object, opposite view. Centering preserves class means algebraically; it
    does NOT ensure preserved class information or useful prediction diversity.
    Singleton classes are skipped. No stop-gradient center or private projector
    can manufacture a detached loss. M=1 gives exactly zero.
    """
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("Matched [M,B,D] representations")
    m, n, _ = a.shape
    if m == 1:
        return a.sum() * 0
    ra, rb = a.clone(), b.clone()
    valid = torch.zeros(n, dtype=torch.bool, device=a.device)
    for cls in torch.unique(labels):
        mask = labels == cls
        if int(mask.sum()) >= 2:
            valid |= mask
            ra[:, mask] = a[:, mask] - a[:, mask].mean(1, keepdim=True)
            rb[:, mask] = b[:, mask] - b[:, mask].mean(1, keepdim=True)
    if not valid.any():
        return a.sum() * 0
    ra, rb = F.normalize(ra[:, valid], dim=-1), F.normalize(rb[:, valid], dim=-1)
    s = torch.einsum("mbd,kbd->bmk", ra, rb) / temperature
    targets = torch.arange(m, device=a.device).expand(s.shape[0], -1).reshape(-1)
    return .5 * (F.cross_entropy(s.reshape(-1, m), targets)
                 + F.cross_entropy(s.transpose(1, 2).reshape(-1, m), targets))


def objective(logits_a, repr_a, logits_b, repr_b, labels, task, enabled, independent, identities=None):
    # Two CE views in EVERY arm; contrastive compute does not grant more labels.
    own = .5 * (own_supervision(logits_a, labels, task) + own_supervision(logits_b, labels, task))
    alignment = alignment_loss(repr_a, repr_b, labels, task, identities=identities) if enabled else own.sum() * 0
    residual = residual_member_contrast(repr_a, repr_b, labels) if enabled else own.sum() * 0
    aux = .05 * alignment + .05 * residual
    # Separate independently trained members retain their unscaled own gradients.
    total = own.sum() + len(own) * aux if independent else own.mean() + aux
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite total objective")
    return total, {"own": own.detach(), "alignment": alignment.detach(), "residual": residual.detach()}
