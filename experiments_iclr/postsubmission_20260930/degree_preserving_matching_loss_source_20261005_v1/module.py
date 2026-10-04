"""Attributed3x3 conditional matching losses; no scorer, fit or graph loader.

Inputs are actual target logits on the same nine edges/context for every
member. The caller owns target exclusion and TRAIN-only assignment labels.
FP64 likelihood arithmetic leaves the native scorer's forward dtype unchanged.
"""
import torch
from torch.autograd.function import once_differentiable

PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
PARITY = (1, -1, -1, 1, 1, -1)


def require(value, message):
    if not value:
        raise ValueError(message)


def _parity(like):
    return like.new_tensor(PARITY)


def _equation(q, c):
    sign = _parity(q)
    r = q + c[:, None] * sign
    require(bool((r > 0).all()) and bool(torch.isfinite(r).all()), "Root candidate is not finite/interior")
    logs = r.log()
    residual = (logs * sign).sum(dim=-1)
    # Fixed rounding budget for six FP64 logs and their sum; not user-tunable.
    resolution = 16 * torch.finfo(torch.float64).eps * (1 + logs.abs().sum(dim=-1))
    return r, residual, resolution


class _EntropyRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q):
        require(q.ndim == 2 and q.shape[1] == 6 and q.dtype == torch.float64,
                "Expected FP64 [bundles,6] law")
        require(bool(torch.isfinite(q).all()) and bool((q > 0).all()), "qJ must be strictly finite/interior")
        eps = torch.finfo(torch.float64).eps
        require(bool(((q.sum(-1) - 1).abs() <= 32 * eps).all()), "qJ does not normalize at declared FP64 resolution")
        sign = _parity(q)
        lo = -q[:, sign > 0].min(-1).values
        hi = q[:, sign < 0].min(-1).values
        best = torch.zeros_like(lo)
        _, best_f, tolerance = _equation(q, best)
        done = best_f.abs() <= tolerance
        # On [-1,1],1077 halvings cover FP64's52-bit significand and the
        # subnormal exponent range. This is an arithmetic bound, not a search.
        for _ in range(1077):
            if bool(done.all()):
                break
            mid = lo + (hi - lo) / 2
            stalled = (mid <= lo) | (mid >= hi)
            active = ~done & ~stalled
            if not bool(active.any()):
                break
            candidate = torch.where(active, mid, best)
            _, f, tolerance = _equation(q, candidate)
            improve = active & ((f.abs() < best_f.abs()) | (f.abs() <= tolerance))
            best = torch.where(improve, candidate, best)
            best_f = torch.where(improve, f, best_f)
            lo = torch.where(active & (f < 0), candidate, lo)
            hi = torch.where(active & (f >= 0), candidate, hi)
            done = done | (active & (f.abs() <= tolerance))
        r, residual, resolution = _equation(q, best)
        require(bool((residual.abs() <= resolution).all()), "Interior entropy root unresolved at fixed FP64 resolution")
        # Use scaled reciprocals to avoid overflow when positive probabilities
        # are subnormal: (1/r_i)/sum_j(1/r_j) = (min(r)/r_i)/sum_j(min(r)/r_j).
        inverse_scaled = r.min(-1, keepdim=True).values / r
        derivative = -sign * inverse_scaled / inverse_scaled.sum(-1, keepdim=True)
        require(bool(torch.isfinite(derivative).all()), "Nonfinite implicit root derivative")
        ctx.save_for_backward(derivative)
        return best

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (derivative,) = ctx.saved_tensors
        return grad_output[:, None] * derivative


def maximum_entropy_same_margins(q):
    """Unique interior entropy maximum on q+parity*c; first-order differentiable."""
    c = _EntropyRoot.apply(q)
    r, residual, resolution = _equation(q, c)
    return r, c, residual, resolution


def matching_law_edge_marginals(law):
    require(law.ndim == 2 and law.shape[1] == 6, "Expected [bundles,6] matching law")
    incidence = law.new_zeros((6, 3, 3))
    for index, permutation in enumerate(PERMUTATIONS):
        for row, column in enumerate(permutation):
            incidence[index, row, column] = 1
    return torch.einsum("bp,pij->bij", law, incidence)


def matching_losses(native_logits, observed_assignment):
    """Return both mean NLLs, per-bundle NLLs, six laws and nine edge margins.

    native_logits:[members>=1,bundles>=1,3,3], finiteFP32/FP64.
    observed_assignment:[bundles], detached TRAIN-observed lexicographic index.
    No label enters the construction of either law. No lambda is selected.
    """
    require(native_logits.ndim == 4 and native_logits.shape[0] >= 1 and native_logits.shape[1] >= 1
            and tuple(native_logits.shape[-2:]) == (3, 3) and native_logits.dtype in (torch.float32, torch.float64),
            "Expected finite native [members,bundles,3,3] floating logits")
    require(bool(torch.isfinite(native_logits).all()), "Nonfinite native target logits")
    bundles = native_logits.shape[1]
    require(observed_assignment.shape == (bundles,) and observed_assignment.dtype == torch.long
            and observed_assignment.device == native_logits.device and not observed_assignment.requires_grad
            and bool(((observed_assignment >= 0) & (observed_assignment < 6)).all()), "Invalid TRAIN assignment index")
    logits = native_logits.to(torch.float64)
    scores = torch.stack([sum(logits[:, :, row, column] for row, column in enumerate(permutation))
                          for permutation in PERMUTATIONS], dim=-1)
    require(bool(torch.isfinite(scores).all()), "Nonfinite additive assignment scores")
    member_laws = torch.softmax(scores, dim=-1)
    q_joint = member_laws.mean(dim=0)
    q_me, c, residual, resolution = maximum_entropy_same_margins(q_joint)
    margins_joint = matching_law_edge_marginals(q_joint)
    margins_me = matching_law_edge_marginals(q_me)
    eps = torch.finfo(torch.float64).eps
    require(bool(((q_me.sum(-1) - 1).abs() <= 64 * eps).all())
            and bool(((margins_joint - margins_me).abs() <= 64 * eps).all()),
            "Normalization/marginal preservation exceeds fixed FP64 resolution")
    rows = torch.arange(bundles, device=logits.device)
    nll_joint = -q_joint[rows, observed_assignment].log()
    nll_me = -q_me[rows, observed_assignment].log()
    return {"nll_joint": nll_joint.mean(), "nll_max_entropy": nll_me.mean(),
            "per_bundle_nll_joint": nll_joint, "per_bundle_nll_max_entropy": nll_me,
            "member_laws": member_laws, "q_joint": q_joint, "q_max_entropy": q_me,
            "matching_law_edge_marginals_joint": margins_joint, "matching_law_edge_marginals_max_entropy": margins_me,
            "entropy_root": c, "entropy_root_residual": residual, "entropy_root_resolution": resolution}


def cpu_correctness_check():
    """Bounded constructed-input check, unexecuted in this source preparation.

    Checks one finite mixture, two nulls and one central-difference gradient.
    No model, optimizer, graph, data acquisition, fitting or update exists here.
    """
    dtype = torch.float64
    teacher = torch.tensor([0], dtype=torch.long, device="cpu")
    first = torch.eye(3, dtype=dtype, device="cpu")
    # Three equal components favor the three even permutations. For a=1,
    # exp(3a)+2>3exp(a), so qJ has greater even mass while every edge margin
    # is1/3. The same-margin maximum-entropy law is uniform on all six.
    fixture = torch.stack((first, first.roll(1, dims=1), first.roll(2, dims=1)))[:, None].requires_grad_()
    result = matching_losses(fixture, teacher)
    eps = torch.finfo(dtype).eps
    resolution = 64 * eps
    assert bool(((result["q_joint"].sum(-1) - 1).abs() <= resolution).all())
    assert bool(((result["q_max_entropy"].sum(-1) - 1).abs() <= resolution).all())
    assert bool(((result["matching_law_edge_marginals_joint"] - result["matching_law_edge_marginals_max_entropy"]).abs() <= resolution).all())
    assert bool(((result["matching_law_edge_marginals_joint"] - 1 / 3).abs() <= resolution).all())
    assert bool(((result["q_max_entropy"] - 1 / 6).abs() <= resolution).all())
    assert float(result["entropy_root"].abs().max()) > resolution
    entropy_j = -(result["q_joint"] * result["q_joint"].log()).sum(-1)
    entropy_me = -(result["q_max_entropy"] * result["q_max_entropy"].log()).sum(-1)
    assert bool((entropy_me > entropy_j + resolution).all())
    for null in (first[None, None], first[None, None].expand(3, 1, 3, 3)):
        value = matching_losses(null, teacher)
        assert bool(((value["q_joint"] - value["q_max_entropy"]).abs() <= resolution).all())
    gradient = torch.autograd.grad(result["nll_joint"] + result["nll_max_entropy"], fixture)[0]
    h = eps ** (1 / 3)  # Fixed central-difference rounding/truncation balance.
    oracle = torch.empty_like(fixture)
    flat = fixture.detach().flatten()
    for coordinate in range(flat.numel()):
        plus, minus = flat.clone(), flat.clone()
        plus[coordinate] += h
        minus[coordinate] -= h
        right = matching_losses(plus.reshape_as(fixture), teacher)
        left = matching_losses(minus.reshape_as(fixture), teacher)
        oracle.flatten()[coordinate] = ((right["nll_joint"] + right["nll_max_entropy"])
                                        - (left["nll_joint"] + left["nll_max_entropy"])) / (2 * h)
    fd_resolution = 128 * h * h * (1 + torch.maximum(oracle.abs(), gradient.abs()))
    assert bool(torch.isfinite(gradient).all()) and bool(((gradient - oracle).abs() <= fd_resolution).all())
    assert fixture.grad is None
    return {"normalization_and_marginals": True, "single_and_identical_member_nulls": True,
            "finite_mixture_witness": True, "autograd_vs_central_difference": True,
            "probability_absolute_resolution": resolution,
            "maximum_gradient_difference": float((gradient - oracle).abs().max()), "fits_or_updates": 0}
