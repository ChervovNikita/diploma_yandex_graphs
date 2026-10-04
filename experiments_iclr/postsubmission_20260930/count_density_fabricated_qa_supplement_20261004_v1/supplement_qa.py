"""CPU-only fabricated supplemental gates. No science data or model imports."""
from itertools import product
import torch
import hardened_count_density as density


def assert_close(a, b, message, atol=2e-10, rtol=2e-10):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(message)


def enumerated(left, right, potential):
    size = len(left) + len(right)
    states = torch.tensor(list(product((0., 1.), repeat=size)), dtype=torch.float64).reshape(2**size, size)
    kl, kr = states[:, :len(left)].sum(1).long(), states[:, len(left):].sum(1).long()
    logw = states @ torch.cat((left, right)) + potential[kl, kr]
    return states, logw, torch.logsumexp(logw, 0)


def rejection(fn, message):
    try:
        fn()
    except (RuntimeError, ValueError, TypeError):
        return
    raise AssertionError("Invalid input accepted: " + message)


def run():
    cases, pattern_checks, direct_mass_checks = 0, 0, 0
    for size in range(9):
        for nl in range(size + 1):
            eta = torch.linspace(-2.7, 2.3, size, dtype=torch.float64)
            left, right = eta[:nl], eta[nl:]
            shape = (nl + 1, size - nl + 1)
            zeros = torch.zeros(shape, dtype=torch.float64)
            expected_z = torch.nn.functional.softplus(eta).sum()
            assert_close(density.log_normalizer(left, right, zeros), expected_z, "g=0 independent partition")
            assert_close(density.unary_marginals(left, right, zeros)[0], torch.sigmoid(eta), "g=0 sigmoid marginals")
            k, l = torch.meshgrid(torch.arange(shape[0], dtype=torch.float64),
                                  torch.arange(shape[1], dtype=torch.float64), indexing="ij")
            potential = .3 * k.square() - .8 * k * l + .5 * l
            # Constant potential, arbitrary scalar potential shift, and a
            # compensated two-side unary gauge shift preserve the full law.
            variants = ((zeros, zeros + 17.25, left, right),
                        (potential, potential - 12.75, left, right),
                        (potential, potential - 1.3 * k + .7 * l, left + 1.3, right - .7))
            for original, altered, altered_left, altered_right in variants:
                states, logw, _ = enumerated(left, right, original)
                _, altered_logw, _ = enumerated(altered_left, altered_right, altered)
                logz = density.log_normalizer(left, right, original)
                new_z = density.log_normalizer(altered_left, altered_right, altered)
                mass = (logw - logz).exp()
                altered_mass = (altered_logw - new_z).exp()
                assert_close(mass.sum(), mass.new_tensor(1.), "direct total pattern mass")
                assert_close(altered_mass.sum(), mass.new_tensor(1.), "shifted direct total mass")
                assert_close(mass, altered_mass, "complete-law shift invariance")
                assert_close(density.unary_marginals(left, right, original)[0],
                             density.unary_marginals(altered_left, altered_right, altered)[0], "shifted unary marginals")
                direct_mass_checks += 2
                for z in states:
                    before = density.query_nll(left, right, original, z)
                    after = density.query_nll(altered_left, altered_right, altered, z)
                    assert_close(before, after, "every-pattern shift-invariant NLL")
                    pattern_checks += 1
            cases += 1
    # Explicit dependence sign and count-boundary concentration, not inferred
    # only from recurrence identities. Nonzero probabilities are retained.
    eta = torch.zeros(1, dtype=torch.float64)
    negative_g = torch.tensor([[-3., 3.], [3., -3.]], dtype=torch.float64)
    states, logw, _ = enumerated(eta, eta, negative_g)
    probabilities = (logw - density.log_normalizer(eta, eta, negative_g)).exp()
    mu = density.unary_marginals(eta, eta, negative_g)[0]
    covariance = (probabilities * states[:, 0] * states[:, 1]).sum() - mu.prod()
    if float(covariance) >= -.2:
        raise AssertionError("Negative-correlation witness failed")
    for target in ((0, 0), (2, 2), (0, 2), (2, 0)):
        eta = torch.tensor([-.2, .3], dtype=torch.float64)
        g = torch.zeros((3, 3), dtype=torch.float64)
        g[target] = 40.
        _, result = density.unary_marginals(eta, eta, g)
        if float(result["count_probability"][target]) < 1 - 1e-14:
            raise AssertionError("Boundary count concentration failed")
    empty = torch.empty(0, dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[7.]], dtype=torch.float64, requires_grad=True)
    loss = density.query_nll(empty, empty, g, empty)
    de, dg = torch.autograd.grad(loss, (empty, g))
    assert_close(de, torch.empty(0, dtype=torch.float64), "empty unary gradient")
    assert_close(dg, torch.zeros_like(g), "empty potential gradient")
    rejections = 0
    for invalid in (torch.zeros(1, dtype=torch.float32), torch.tensor([float("nan")], dtype=torch.float64),
                    torch.tensor([float("inf")], dtype=torch.float64), torch.zeros((1, 1), dtype=torch.float64)):
        rejection(lambda: density.log_prefix(invalid), "unary public DP")
        rejection(lambda: density.unary_marginals(invalid, torch.empty(0, dtype=torch.float64),
                                                   torch.zeros((2, 1), dtype=torch.float64)), "marginal unary")
        rejections += 2
    rejection(lambda: density.query_nll(torch.empty(0, dtype=torch.float32), empty,
                                        torch.zeros((1, 1), dtype=torch.float64), empty), "empty lower precision")
    rejection(lambda: density.query_nll(empty, empty, torch.tensor([[float("nan")]], dtype=torch.float64), empty),
              "empty nonfinite potential")
    rejections += 2
    return {"status": "PASS", "geometries": cases, "every_pattern_shift_checks": pattern_checks,
            "direct_mass_checks": direct_mass_checks, "invalid_input_rejections": rejections,
            "negative_correlation_witness": True, "boundary_count_concentration": True,
            "empty_zero_gradients": True, "native_model_or_feasibility_qualified": False}
