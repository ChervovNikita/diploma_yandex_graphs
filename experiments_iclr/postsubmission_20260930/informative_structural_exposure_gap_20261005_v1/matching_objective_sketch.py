"""Unexecuted source sketch; no dataset, optimizer, model or serving integration.

Input scores are actual native target logits evaluated on one common masked
TRAIN context. Bundle construction and duplicate-safe masking are specified in
ASSESSMENT.md. Numerical root tolerance and iteration budget must be supplied
by a future prospectively specified implementation; no values are selected here.
"""

import itertools
import math

import torch


PERMUTATIONS = tuple(itertools.permutations(range(3)))
PARITY = tuple(
    1 if sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) % 2 == 0 else -1
    for p in PERMUTATIONS
)
OBSERVED_DIAGONAL = PERMUTATIONS.index((0, 1, 2))


def member_log_laws(native_target_logits):
    """Return [members, bundles, 6] log probabilities from [M, B, 3, 3]."""
    if native_target_logits.ndim != 4 or native_target_logits.shape[-2:] != (3, 3):
        raise ValueError("Expected native target logits with shape [M, B, 3, 3].")
    energies = torch.stack(
        [sum(native_target_logits[..., i, p[i]] for i in range(3)) for p in PERMUTATIONS],
        dim=-1,
    )
    return energies.log_softmax(dim=-1)


class _SameMarginalMaximumEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_joint, root_atol, max_iterations):
        if q_joint.shape[-1] != 6:
            raise ValueError("Expected six permutation probabilities.")
        if not torch.isfinite(q_joint).all() or not (q_joint > 0).all():
            raise ValueError("Strictly positive finite probabilities are required; no silent clamps.")
        if root_atol <= 0 or max_iterations <= 0:
            raise ValueError("Caller must supply a positive tolerance and iteration budget.")
        signs = q_joint.new_tensor(PARITY)
        even = signs > 0
        odd = signs < 0
        lower = -q_joint[..., even].min(dim=-1).values
        upper = q_joint[..., odd].min(dim=-1).values
        for _ in range(max_iterations):
            middle = (lower + upper) / 2
            candidate = q_joint + middle[..., None] * signs
            if not (candidate > 0).all():
                raise RuntimeError("Probability underflow or stalled root interval; qualify arithmetic.")
            residual = (candidate.log() * signs).sum(dim=-1)
            lower = torch.where(residual < 0, middle, lower)
            upper = torch.where(residual >= 0, middle, upper)
            if bool(((upper - lower) <= root_atol).all()):
                break
        else:
            raise RuntimeError("Maximum-entropy root did not reach the supplied tolerance.")
        q_me = q_joint + ((lower + upper) / 2)[..., None] * signs
        if not (q_me > 0).all():
            raise RuntimeError("Strictly positive maximum-entropy output required.")
        ctx.save_for_backward(q_me, signs)
        return q_me

    @staticmethod
    def backward(ctx, grad_output):
        q_me, signs = ctx.saved_tensors
        # f(c,r)=sum_pi signs_pi log(r_pi + signs_pi*c)=0.
        # dc/dr_j = -signs_j / (q_me_j * sum_pi 1/q_me_pi).
        denominator = q_me.reciprocal().sum(dim=-1, keepdim=True)
        dc_dr = -signs / (q_me * denominator)
        chain = (grad_output * signs).sum(dim=-1, keepdim=True)
        return grad_output + chain * dc_dr, None, None


def auxiliary_losses(native_target_logits, *, root_atol, max_iterations):
    """Compute alternatives, not a combined training objective or chosen weight.

    q_joint and q_me have the same row marginals at this fixed model state.
    Separately trained arms need not retain the same learned marginals.
    """
    log_members = member_log_laws(native_target_logits)
    member_count = native_target_logits.shape[0]
    log_joint = torch.logsumexp(log_members, dim=0) - math.log(member_count)
    q_joint = log_joint.exp()
    q_me = _SameMarginalMaximumEntropy.apply(q_joint, root_atol, max_iterations)
    return {
        "joint": -log_joint[..., OBSERVED_DIAGONAL].mean(),
        "same_marginal_maximum_entropy": -q_me[..., OBSERVED_DIAGONAL].log().mean(),
        "own_member_mean": -log_members[..., OBSERVED_DIAGONAL].mean(),
        "joint_law": q_joint,
        "same_marginal_law": q_me,
    }


def row_marginals(permutation_probabilities):
    """[... , 6] -> [... , 3, 3]; intended for later source qualification."""
    return torch.stack(
        [
            torch.stack(
                [permutation_probabilities[..., [k for k, p in enumerate(PERMUTATIONS) if p[i] == j]].sum(-1)
                 for j in range(3)],
                dim=-1,
            )
            for i in range(3)
        ],
        dim=-2,
    )
