"""Prepared density QA only. Root admission required; never invoked here.

No dataset/checkpoint/split reader. Enumeration independently defines the law.
The complete native model/replay QA is separately required in QUALIFICATION.md.
"""
from itertools import product
import torch
from count_density import log_normalizer, query_nll, unary_marginals, require


def enumeration(left, right, g):
    size = len(left) + len(right)
    states = torch.tensor(list(product((0., 1.), repeat=len(left) + len(right))),
                          dtype=torch.float64, device=left.device).reshape(2**size, size)
    kl, kr = states[:, :len(left)].sum(1).long(), states[:, len(left):].sum(1).long()
    log_weight = states @ torch.cat((left, right)) + g[kl, kr]
    logz = torch.logsumexp(log_weight, 0)
    probability = torch.softmax(log_weight, 0)
    return states, log_weight, logz, probability, probability @ states


def close(actual, expected, name, atol=2e-10, rtol=2e-10):
    require(torch.allclose(actual, expected, atol=atol, rtol=rtol), name + " parity failed")


def density_qualification():
    """CPU tensors; exhaustive supports0..8; no stochastic/model/data work."""
    cases, patterns, gradient_cases = 0, 0, 0
    for size in range(9):
        for nl in range(size + 1):
            nr = size - nl
            eta = torch.linspace(-2.1, 2.4, size, dtype=torch.float64) if size else torch.empty(0, dtype=torch.float64)
            k, l = torch.meshgrid(torch.arange(nl + 1, dtype=torch.float64),
                                  torch.arange(nr + 1, dtype=torch.float64), indexing="ij")
            g = .7 * (k - l).square() - .4 * k * l + .2 * (k + 2 * l)
            left, right = eta[:nl], eta[nl:]
            states, log_weight, expected_z, probability, expected_mu = enumeration(left, right, g)
            close(log_normalizer(left, right, g), expected_z, "normalizer")
            mu, detail = unary_marginals(left, right, g)
            close(mu, expected_mu, "fixed-context unary marginal")
            count_probability = torch.zeros_like(g)
            for z, p in zip(states, probability):
                count_probability[int(z[:nl].sum()), int(z[nl:].sum())] += p
            close(detail["count_probability"], count_probability, "full joint count probability")
            for z, logw in zip(states, log_weight):
                expected = (expected_z - logw) / max(size, 1) if size else expected_z * 0.
                close(query_nll(left, right, g, z), expected, "every-pattern NLL")
                patterns += 1
            # Independent unary derivative with fixed g is mu. Autograd through
            # neural g(eta) would be a total derivative and is a different object.
            t = eta.clone().requires_grad_()
            fixed = g.clone().requires_grad_()
            normal = log_normalizer(t[:nl], t[nl:], fixed)
            dt, dg = torch.autograd.grad(normal, (t, fixed), allow_unused=True)
            if size:
                close(dt, expected_mu, "independent unary derivative")
            else:
                require(dt is None, "Empty unary input became active")
            close(dg, count_probability, "potential normalizer derivative")
            if size:
                z = states[len(states) // 2]
                t, fixed = eta.clone().requires_grad_(), g.clone().requires_grad_()
                loss = query_nll(t[:nl], t[nl:], fixed, z)
                dt, dg = torch.autograd.grad(loss, (t, fixed))
                close(dt, (expected_mu - z) / size, "source unary gradient")
                expected_g = count_probability.clone()
                expected_g[int(z[:nl].sum()), int(z[nl:].sum())] -= 1.
                close(dg, expected_g / size, "source potential gradient")
                gradient_cases += 1
            # Every slot remains included after arbitrary within-side permutation.
            perm_l = torch.arange(nl - 1, -1, -1)
            perm_r = torch.arange(nr - 1, -1, -1)
            perm_mu, _ = unary_marginals(left[perm_l], right[perm_r], g)
            close(perm_mu, torch.cat((expected_mu[:nl][perm_l], expected_mu[nl:][perm_r])), "slot permutation")
            swapped, _ = unary_marginals(right, left, g.T)
            close(swapped, torch.cat((expected_mu[nl:], expected_mu[:nl])), "side exchange")
            cases += 1
    # Witness: g suppresses mixed counts, yielding a correlated00/11 law.
    eta = torch.zeros(1, dtype=torch.float64)
    g = torch.tensor([[3., -3.], [-3., 3.]], dtype=torch.float64)
    states, _, _, p, mu = enumeration(eta, eta, g)
    covariance = (p * states[:, 0] * states[:, 1]).sum() - mu.prod()
    require(float(covariance) > .2, "Control collapsed to an independent-bit law")
    close(unary_marginals(eta, eta, g)[0], mu, "correlated witness marginal")
    # No epsilon/clipping fallback for extreme but finite full-support fixtures.
    for eta in (torch.tensor([-1000., 1000., 25., -25.], dtype=torch.float64),
                torch.full((4,), 1000., dtype=torch.float64),
                torch.full((4,), -1000., dtype=torch.float64)):
        g = torch.tensor([[0., 10., -10.], [20., -20., 5.], [-5., 3., 0.]], dtype=torch.float64)
        _, _, logz, _, mu = enumeration(eta[:2], eta[2:], g)
        close(log_normalizer(eta[:2], eta[2:], g), logz, "extreme normalizer", atol=1e-9)
        close(unary_marginals(eta[:2], eta[2:], g)[0], mu, "extreme marginal", atol=1e-9)
    # Fixed-context caveat witness: eta appears in g but is held fixed by API.
    t = torch.tensor([.2, -.6], dtype=torch.float64, requires_grad=True)
    g = torch.stack((t.new_zeros(2), 4 * t)).reshape(2, 2)
    total = torch.autograd.grad(log_normalizer(t[:1], t[1:], g), t)[0]
    mu, _ = unary_marginals(t[:1], t[1:], g)
    require(not torch.allclose(total, mu), "Context-derivative witness degenerate")
    # Meaningful first-order double precision gradcheck on all density inputs.
    left = torch.tensor([-.4, .8], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([.3], dtype=torch.float64, requires_grad=True)
    g = torch.arange(6, dtype=torch.float64).reshape(3, 2).div(7).requires_grad_()
    z = torch.tensor([1., 0., 1.], dtype=torch.float64)
    require(torch.autograd.gradcheck(lambda a, b, c: query_nll(a, b, c, z),
                                     (left, right, g), eps=1e-6, atol=1e-6, rtol=1e-5), "Density first-order gradcheck failed")
    return {"status": "PASS", "query_cases": cases, "enumerated_patterns": patterns,
            "nonempty_gradient_cases": gradient_cases, "correlated_witness": True,
            "full_native_model_qualified": False, "replay_qualified": False,
            "real_batch_feasibility_qualified": False, "fit_authorized": False}
