"""Passive exact same-context diagnostics; no optimizer or selection hook.

This new DDI schedule does not fulfill the older full NCNC census diagnostic
plan. Any future probe invocation pays its cost under a separate root budget.
"""
from math import log
import torch
from exact_conditional_loss import _grouped_side_nll, training_pattern_losses, TrainPatternLabels


def entropy(responsibility):
    members = responsibility.shape[0]
    return -(responsibility * responsibility.clamp_min(torch.finfo(responsibility.dtype).tiny).log()).sum(dim=0) / log(members)


@torch.no_grad()
def same_state_probe(eta_left, eta_right, queries, supports, bits, visible_degree):
    """Use identical ordered full supports and detached TRAIN bits throughout.

    Call on a prereleased TRAIN probe with model eval mode and restored RNG.
    Return per-query quantities for population-aware descriptive aggregation.
    """
    left_rows, left_nodes = supports.left
    right_rows, right_nodes = supports.right
    count = len(queries)
    labels = TrainPatternLabels(bits, 'complete_TRAIN_observation_membership')
    laws = training_pattern_losses(eta_left, eta_right, left_rows, right_rows, labels, count)
    a, nl, kl, rl = _grouped_side_nll(eta_left, left_rows, bits[:len(left_rows)], count)
    b, nr, kr, rr = _grouped_side_nll(eta_right, right_rows, bits[len(left_rows):], count)
    denominator = (nl + nr).clamp_min(1)
    log_rho_left, log_rho_right = torch.log_softmax(-a, dim=0), torch.log_softmax(-b, dim=0)
    rho_left, rho_right = log_rho_left.exp(), log_rho_right.exp()
    # Keep the identity in log space even when a displayed overlap underflows.
    log_overlap = log(eta_left.shape[0]) + torch.logsumexp(log_rho_left + log_rho_right, dim=0)
    overlap = log_overlap.exp()
    alignment_identity = -log_overlap / denominator
    uniform = (torch.lgamma((nl + 1).to(eta_left.dtype)) - torch.lgamma((kl + 1).to(eta_left.dtype))
               - torch.lgamma((nl - kl + 1).to(eta_left.dtype))
               + torch.lgamma((nr + 1).to(eta_left.dtype)) - torch.lgamma((kr + 1).to(eta_left.dtype))
               - torch.lgamma((nr - kr + 1).to(eta_left.dtype))) / denominator
    degree_left = torch.log1p(visible_degree[left_nodes].to(eta_left.dtype))[None, :]
    degree_right = torch.log1p(visible_degree[right_nodes].to(eta_right.dtype))[None, :]
    degree = training_pattern_losses(degree_left, degree_right, left_rows, right_rows, labels, count)['J_K']
    return {
        'J_K': laws['J_K'], 'J_K_sep': laws['J_K_sep'], 'W_K': laws['W_K'],
        'uniform_subset_nll': uniform, 'visible_degree_subset_nll': degree,
        'rho_left': rho_left, 'rho_right': rho_right, 'rho_joint': laws['rho'],
        'entropy_left': entropy(rho_left), 'entropy_right': entropy(rho_right),
        'entropy_joint': entropy(laws['rho']), 'scaled_endpoint_overlap': overlap,
        'same_state_gap': laws['J_K'] - laws['J_K_sep'], 'alignment_identity': alignment_identity,
        'identity_residual': laws['J_K'] - laws['J_K_sep'] - alignment_identity,
        'both_variable': (rl > 0) & (rr > 0), 'one_variable': (rl > 0) ^ (rr > 0),
        'support_counts': (nl, nr), 'teacher_counts': (kl, kr),
    }
