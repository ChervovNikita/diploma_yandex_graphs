"""Source-only route-target contrast; no scientific or CUDA qualification.

This is a supervised contrastive positive-relation adaptation, not an MI
estimator or a guarantee of independent members. All predictor parameters
receive the same scalar objective, retaining ordinary joint optimization.
"""
import torch
from torch.nn import functional as F


def context_alignment(a, b, positive_weights, temperature=.2):
    """Symmetric cross-view CE with externally frozen [M,B,B] targets.

    Native members and both stochastic views remain unchanged. No new auxiliary
    projection head, predictor, inference gate or graph edit is supplied.
    Same-class unselected rows remain denominator distractors by design.
    """
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] not in (1, 4):
        raise ValueError('one/four exact paired member/object/feature axes required')
    weights = torch.as_tensor(positive_weights, device=a.device, dtype=a.dtype)
    if weights.shape != (a.shape[0], a.shape[1], a.shape[1]):
        raise ValueError('frozen targets must match exact auxiliary identities')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('finite nonnegative targets required')
    if weights.requires_grad:
        raise ValueError('positive targets must be frozen independently of the predictor')
    if not torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6, rtol=1e-6):
        raise ValueError('each target row must have total mass one')
    if not 0 < temperature:
        raise ValueError('positive temperature required')
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    scores = torch.einsum('mbd,mcd->mbc', a, b) / temperature
    # Both views have the same object IDs: a directed target relation i->j
    # remains i->j when switching views. Only the score matrix transposes.
    return -.5 * ((F.log_softmax(scores, -1) * weights).sum(-1).mean()
                  + (F.log_softmax(scores.transpose(1, 2), -1) * weights).sum(-1).mean())


def objective(logits_a, repr_a, logits_b, repr_b, train_labels,
              positive_weights, coefficient=.05, temperature=.2, independent=False):
    """Ordinary two-view mean own CE plus one fixed target alignment term.

    This dispatch is WikiCS multiclass only; molecular/ranking labels require a
    separately justified positive relation. Private internal factor, stem/head,
    and shared-weight gradients are not silently restricted or projected.
    """
    if logits_a.shape != logits_b.shape or logits_a.ndim != 3 or logits_a.shape[0] not in (1, 4):
        raise ValueError('one/four paired member TRAIN logits required')
    if coefficient < 0:
        raise ValueError('fixed nonnegative coefficient required')
    labels = train_labels.to(device=logits_a.device)
    own = torch.stack([.5*(F.cross_entropy(xa, labels)+F.cross_entropy(xb, labels))
                       for xa, xb in zip(logits_a, logits_b)])
    alignment = context_alignment(repr_a, repr_b, positive_weights, temperature)
    # Untied members keep their ordinary unscaled own gradients. Their target
    # losses are separable, so this is sum_m [own_m+.05 alignment_m]. The shared
    # bank follows the existing mean-member convention.
    total = own.sum()+len(own)*coefficient*alignment if independent else own.mean()+coefficient*alignment
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite candidate objective')
    return total, {'own': own.detach(), 'context_alignment': alignment.detach()}
