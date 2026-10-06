"""Disabled CMCL exact-KL objective transplant; pure supplied-logit functions.

Attributed to Confident Multiple Choice Learning, PMLR70/lee17b and author
chhwang/cmcl commit57f41f4b166b8544c56580f9d32ee98b997e3f59 (Apache2.0).
Independently written PyTorch equations; no native model/loader/trainer/CLI.
M4/K3/beta.75, stopped member-index ties, no optional feature exchange.
"""
import math

SOURCE_RELEASED = False
MEMBERS, CLASSES, OWNER_K, BETA = 4, 5, 3, 0.75
ROLE_MASS = 0.5


def require(value, message):
    if not value:
        raise RuntimeError(message)


def _node_terms(logits, targets):
    import torch
    require(isinstance(logits, torch.Tensor) and isinstance(targets, torch.Tensor)
        and targets.ndim == 1 and targets.numel() > 0
        and logits.shape == (MEMBERS, targets.numel(), CLASSES)
        and logits.dtype in (torch.float32, torch.float64)
        and targets.dtype == torch.int64 and targets.device == logits.device
        and bool(torch.isfinite(logits).all()) and bool(((targets >= 0) & (targets < CLASSES)).all()),
        "Fixed M4/C5 supplied finite logits and nonempty in-range int64 targets required")
    log_probs = torch.log_softmax(logits, dim=-1)
    ce = -log_probs.gather(2, targets[None, :, None].expand(MEMBERS, -1, 1)).squeeze(-1)
    # Exact KL(U_C || P_m), not KL(P_m || U_C); no probability clipping.
    uniform_to_member_KL = -math.log(CLASSES) - log_probs.mean(dim=-1)
    owner_score = ce - BETA * uniform_to_member_KL
    with torch.no_grad():
        # Stable member-axis ordering preserves index0,1,2,3 at exact ties.
        owners = torch.argsort(owner_score.detach(), dim=0, stable=True)[:OWNER_K]
        owner_mask = torch.zeros_like(ce, dtype=torch.bool)
        owner_mask.scatter_(0, owners, True)
    # Published overlap objective: sum across members, then mean across nodes.
    # Nonowners receive KL only. No all-member CE anchor or M/K division.
    losses = torch.where(owner_mask, ce, BETA * uniform_to_member_KL)
    node_loss = losses.sum(dim=0)
    require(bool(torch.isfinite(node_loss).all()) and bool((owner_mask.sum(dim=0) == OWNER_K).all()),
        "Finite exact three-owner CMCL objective required")
    return node_loss, {"owner_indices": owners, "owner_mask": owner_mask,
        "CE": ce, "KL_uniform_to_member": uniform_to_member_KL,
        "owner_score": owner_score, "member_losses": losses}


def _engineering_role_objective(logits_S, targets_S, logits_R, targets_R, *, engineering_authorized=False):
    require(engineering_authorized is True, "Disabled: separately admitted pure engineering caller required")
    require(logits_S.device == logits_R.device and logits_S.dtype == logits_R.dtype,
        "Both supplied roles must use the same incoming-state device/dtype")
    node_S, terms_S = _node_terms(logits_S, targets_S)
    node_R, terms_R = _node_terms(logits_R, targets_R)
    loss_S, loss_R = node_S.mean(), node_R.mean()
    total = ROLE_MASS * (loss_S + loss_R)
    return total, {"S_loss": loss_S, "R_loss": loss_R, "S": terms_S, "R": terms_R,
        "M": MEMBERS, "K": OWNER_K, "beta": BETA, "role_mass": ROLE_MASS,
        "optional_feature_sharing": False, "owners_stopped": True,
        "all_member_CE_anchor_added": False, "member_sum_not_mean": True}


def training_role_objective(logits_S, targets_S, logits_R, targets_R):
    require(SOURCE_RELEASED is True, "Disabled source preparation has no training authority")
    return _engineering_role_objective(logits_S, targets_S, logits_R, targets_R,
        engineering_authorized=True)


def _engineering_mean_probabilities(logits, *, engineering_authorized=False):
    require(engineering_authorized is True, "Disabled: separately admitted pure engineering serving caller required")
    import torch
    require(isinstance(logits, torch.Tensor) and logits.ndim == 3 and logits.shape[0] == MEMBERS
        and logits.shape[2] == CLASSES and logits.dtype in (torch.float32, torch.float64)
        and bool(torch.isfinite(logits).all()), "Four complete finite member logit tensors required")
    return torch.softmax(logits, dim=-1).mean(dim=0)


def mean_probability_serving(logits):
    require(SOURCE_RELEASED is True, "Disabled source preparation has no serving authority")
    return _engineering_mean_probabilities(logits, engineering_authorized=True)
