"""Functional native Adam step, with a stable exact zero-history root.

No imports, model construction or execution occur at module import. Previous
moments are frozen episode state; this differentiates the current gradient only.
No decay, AMSGrad, maximize, smoothing, clipping or implicit SGD substitution.
"""
import math

LR, BETAS, EPS = .001, (.9, .999), 1e-8


def initial_state(torch, parameters):
    return {name: {"step": 0, "exp_avg": torch.zeros_like(value),
                   "exp_avg_sq": torch.zeros_like(value)}
            for name, value in parameters.items()}


def step(torch, parameters, gradients, previous, *, lr=LR, betas=BETAS, eps=EPS):
    """Return virtual parameters and moments without mutating episode state.

    At zero previous second moment, sqrt((1-b2)*g*g) is written as
    sqrt(1-b2)*abs(g). abs uses its zero subgradient at g=0. For reachable
    zero-history m=v=0, the complete parameter-update map is differentiable
    there, with derivative -lr*(1-b1)/(bias1*eps) with respect to g. This is
    a first derivative of the update, not a claim of its second differentiability.
    """
    if set(parameters) != set(gradients) or set(parameters) != set(previous):
        raise ValueError("Private Adam parameter/gradient/state coverage differs")
    if (lr, tuple(betas), eps) != (LR, BETAS, EPS):
        raise ValueError("Only the prospectively fixed native Adam constants are admitted")
    b1, b2 = betas
    updated, states = {}, {}
    for name, parameter in parameters.items():
        gradient, old = gradients[name], previous[name]
        if gradient.shape != parameter.shape or old["exp_avg"].shape != parameter.shape or old["exp_avg_sq"].shape != parameter.shape:
            raise ValueError("Private Adam geometry differs: " + name)
        if type(old["step"]) is not int or old["step"] < 0:
            raise ValueError("Private Adam step must be a nonnegative integer")
        m0, v0 = old["exp_avg"].detach(), old["exp_avg_sq"].detach()
        if not bool(torch.isfinite(parameter).all() and torch.isfinite(gradient).all()
                    and torch.isfinite(m0).all() and torch.isfinite(v0).all()) or bool((v0 < 0).any()):
            raise FloatingPointError("Private Adam input/moment not finite nonnegative: " + name)
        positive_history = v0 > 0
        if bool(((~positive_history) & (m0 != 0)).any()):
            # Exact floating-point underflow could create this state. Do not
            # silently choose a derivative at an unreachable real-valued state.
            raise FloatingPointError("Zero second moment with nonzero first moment: " + name)
        count = old["step"] + 1
        m1 = torch.lerp(m0, gradient, 1 - b1)
        v1 = torch.addcmul(v0 * b2, gradient, gradient, value=1 - b2)
        if bool(((gradient != 0) & (v1 == 0)).any()):
            raise FloatingPointError("Nonzero private gradient squared-moment underflow: " + name)
        # Avoid evaluating sqrt(0) in an inactive branch: where alone would
        # otherwise still allow zero-times-infinity in its backward.
        safe_argument = torch.where(positive_history, v1, torch.ones_like(v1))
        root = torch.where(positive_history, safe_argument.sqrt(), math.sqrt(1-b2) * gradient.abs())
        denominator = root / math.sqrt(1 - b2**count) + eps
        value = torch.addcdiv(parameter, m1, denominator, value=-lr / (1 - b1**count))
        if not bool(torch.isfinite(value).all() and torch.isfinite(m1).all() and torch.isfinite(v1).all()):
            raise FloatingPointError("Private Adam result not finite: " + name)
        updated[name] = value
        states[name] = {"step": count, "exp_avg": m1, "exp_avg_sq": v1}
    return updated, states


def detached_state(states):
    return {name: {"step": row["step"], "exp_avg": row["exp_avg"].detach().clone(),
                   "exp_avg_sq": row["exp_avg_sq"].detach().clone()}
            for name, row in states.items()}


def gradient_jacobian(torch, gradients, previous):
    """Independent scalar formula for d(phi')/d(g), used only by qualification.

    Returns constants for an explicit mixed-Hessian VJP oracle. It is not the
    training derivative, and no finite-difference approximation is used.
    """
    result = {}
    b1, b2 = BETAS
    with torch.no_grad():
        for name, gradient in gradients.items():
            old = previous[name]
            g, m0, v0 = gradient.detach(), old["exp_avg"], old["exp_avg_sq"]
            count = old["step"] + 1
            m1 = torch.lerp(m0, g, 1-b1)
            v1 = torch.addcmul(v0*b2, g, g, value=1-b2)
            history = v0 > 0
            safe = torch.where(history, v1, torch.ones_like(v1))
            root = torch.where(history, safe.sqrt(), math.sqrt(1-b2)*g.abs())
            droot = torch.where(history, (1-b2)*g/safe.sqrt(), math.sqrt(1-b2)*g.sign())
            correction = math.sqrt(1-b2**count)
            denominator = root/correction+EPS
            result[name] = -LR/(1-b1**count) * ((1-b1)/denominator-m1*(droot/correction)/denominator.square())
    return result
