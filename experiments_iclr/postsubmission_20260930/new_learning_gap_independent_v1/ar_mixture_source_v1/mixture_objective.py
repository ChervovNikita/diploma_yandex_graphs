"""Unexecuted fixed-mixture reverse-KL score-function estimator.

This module supplies no optimizer, fit, RNG, normalization-constant estimator,
leave-one-out density or current-sample-including baseline.
"""
from dataclasses import dataclass

from .domain import FixedMixture, IndependentBaseline


@dataclass(frozen=True)
class ReverseKLResult:
    surrogate: object
    objective_without_log_partition: object
    detached_rewards: object
    detached_log_q_mix: object
    detached_log_gamma: object
    generating_log_q: object
    objective_kind: str
    sampling_design: str


def fixed_mixture_reverse_kl_surrogate(*, torch_api, encoded, samples,
                                     cross_log_q, mixture: FixedMixture,
                                     log_fugacities,
                                     baseline: IndependentBaseline,
                                     sampling_design: str,
                                     objective_kind: str, ledger):
    """Return a minimization surrogate for the stated fixed-pi objective.

    cross_log_q[s,m] is the FULL teacher-forced complete density of sample s
    under branch m. Every branch is included, including the generating branch.
    Its generating entries retain score gradients; other entries need none.

    iid_mixture: caller independently draws branches from fixed pi, then y~q_m.
    stratified_per_branch: caller draws S_m independent schedules from every
    branch; each scored sample has weight pi_m/S_m. No S_m is chosen here.

    mixture_reverse_kl uses detached log q_mix - log gamma in the reward.
    mean_branch_reverse_kl uses detached log q_generating - log gamma instead.
    Both calls receive/evaluate the same full matrix for a work-matched ablation.
    """
    samples = tuple(samples)
    S, M = len(samples), len(mixture.masses)
    if not S or tuple(cross_log_q.shape) != (S, M):
        raise ValueError("Complete nonempty [samples,branches] cross-density matrix required")
    if not cross_log_q.requires_grad or not torch_api.isfinite(cross_log_q).all().item():
        raise ValueError("Finite full densities with generating-score gradients required")
    if len(baseline.values_by_branch) != M:
        raise ValueError("One independently certified baseline value per branch required")
    if tuple(log_fugacities.shape) != (encoded.domain.n_nodes,) or log_fugacities.requires_grad:
        raise ValueError("Known theta-independent full-instance log fugacities required")
    if log_fugacities.device != cross_log_q.device or log_fugacities.dtype != cross_log_q.dtype:
        raise ValueError("Explicitly matched density/fugacity device and dtype required")
    if not torch_api.isfinite(log_fugacities).all().item():
        raise ValueError("Finite log fugacities for positive known lambda required")
    if objective_kind not in ("mixture_reverse_kl", "mean_branch_reverse_kl"):
        raise ValueError("Declared fixed-mixture or mean-branch objective required")
    branches, assignments = [], []
    counts = [0] * M
    for sample in samples:
        branch = sample.generating_branch
        if type(branch) is not int or not 0 <= branch < M:
            raise ValueError("Generating branch outside the fixed mixture")
        if sample.graph_id != encoded.domain.graph_id or sample.order_id != encoded.domain.order_id:
            raise ValueError("Sample graph and fixed order must agree with the scored law")
        if not encoded.domain.feasible(sample.assignment):
            raise ValueError("Mixture samples must lie on the common feasible support")
        branches.append(branch)
        assignments.append(encoded.domain.assignment(sample.assignment))
        counts[branch] += 1
    if sampling_design == "iid_mixture":
        sample_weights = [1 / S] * S
    elif sampling_design == "stratified_per_branch":
        if any(count == 0 for count in counts):
            raise ValueError("Stratified estimation requires samples from every positive-weight branch")
        pi = mixture.probabilities
        sample_weights = [pi[branch] / counts[branch] for branch in branches]
    else:
        raise ValueError("Sampling design must be frozen and supplied explicitly")

    row_index = torch_api.arange(S, device=cross_log_q.device)
    branch_index = torch_api.as_tensor(branches, dtype=torch_api.long, device=cross_log_q.device)
    generating_log_q = cross_log_q[row_index, branch_index]
    log_pi = cross_log_q.new_tensor(mixture.log_probabilities)
    # Fixed pi and every full density appear in this reward. No LOO surrogate.
    log_q_mix = torch_api.logsumexp(log_pi + cross_log_q.detach(), dim=-1)
    ledger.record_mixture_reduction(S, M)
    labels = torch_api.as_tensor(assignments, dtype=log_fugacities.dtype,
                                device=log_fugacities.device).reshape(S, encoded.domain.n_nodes)
    log_gamma = (labels * log_fugacities.detach()).sum(dim=-1)
    density_in_reward = log_q_mix if objective_kind == "mixture_reverse_kl" else generating_log_q.detach()
    rewards = density_in_reward - log_gamma
    if not torch_api.isfinite(rewards).all().item():
        raise ArithmeticError("Reverse-KL reward lost numerical finiteness")
    baseline_values = cross_log_q.new_tensor([baseline.values_by_branch[branch] for branch in branches])
    weights = cross_log_q.new_tensor(sample_weights)
    advantages = (rewards - baseline_values).detach()
    surrogate = (weights * advantages * generating_log_q).sum()
    objective_without_log_partition = (weights * rewards).sum().detach()
    return ReverseKLResult(
        surrogate, objective_without_log_partition, rewards.detach(),
        log_q_mix.detach(), log_gamma.detach(), generating_log_q,
        objective_kind, sampling_design,
    )
