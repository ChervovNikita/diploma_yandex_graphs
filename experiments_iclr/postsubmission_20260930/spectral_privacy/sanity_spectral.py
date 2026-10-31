"""Synthetic algebra only; stdlib, no datasets, torch, or GPU.

The covariance is over output feature directions for out-by-in matrices.
The conditional risk identity concerns fixed projectors and fresh graph-unit
gradients. It does not assert unbiased risk after selecting a projector.
"""

import json
import math
import random
from pathlib import Path


def zeros(a, b):
    return [[0.0] * b for _ in range(a)]


def eye(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def add(a, b, scale=1.0):
    return [[x + scale * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scaled(a, scale):
    return [[scale * x for x in row] for row in a]


def multiply(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def inner(a, b):
    return sum(x * y for ar, br in zip(a, b) for x, y in zip(ar, br))


def average(matrices):
    out = zeros(len(matrices[0]), len(matrices[0][0]))
    for a in matrices:
        out = add(out, a)
    return scaled(out, 1.0 / len(matrices))


def centered(matrices):
    mean = average(matrices)
    return [add(a, mean, -1.0) for a in matrices]


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def output_covariance(a, b):
    return average([multiply(x, transpose(y)) for x, y in zip(a, b)])


def sym(a):
    return scaled(add(a, transpose(a)), 0.5)


def covariance_statistics(units, reference_batch):
    """All units are [K,out,in] and centered across K first.

    C is the off-diagonal graph-unit U-statistic. Sigma estimates per-unit
    covariance; N=Sigma/reference_batch is the noise of a fresh batch mean.
    """
    units = [centered(unit) for unit in units]
    n, k = len(units), len(units[0])
    if n < 2 or reference_batch < 1:
        raise ValueError("Need at least two units and positive batch size")
    means = [average([unit[m] for unit in units]) for m in range(k)]
    sigma = zeros(len(means[0]), len(means[0]))
    for unit in units:
        delta = [add(unit[m], means[m], -1.0) for m in range(k)]
        sigma = add(sigma, output_covariance(delta, delta))
    sigma = scaled(sigma, 1.0 / (n - 1))
    c = add(output_covariance(means, means), sigma, -1.0 / n)
    noise = scaled(sigma, 1.0 / reference_batch)
    t = add(c, noise, -1.0)
    return {"mean": means, "C": c, "Sigma": sigma, "N": noise, "T": t}


def jacobi(a):
    """Small symmetric eigensystem; columns of U match descending values."""
    a = [row[:] for row in a]
    n = len(a)
    u = eye(n)
    for _ in range(100 * n * n):
        p, q = max(((p, q) for p in range(n) for q in range(p + 1, n)),
                   key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) < 1e-13:
            break
        angle = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        rot = eye(n)
        rot[p][p], rot[q][q] = c, c
        rot[p][q], rot[q][p] = s, -s
        a = multiply(transpose(rot), multiply(a, rot))
        u = multiply(u, rot)
    order = sorted(range(n), key=lambda i: (-a[i][i], i))
    return [a[i][i] for i in order], [[row[i] for i in order] for row in u]


def projector(u, indices):
    v = [[row[i] for i in indices] for row in u]
    if not indices:
        return zeros(len(u), len(u))
    return multiply(v, transpose(v))


def helmert(k):
    c = zeros(k, k - 1)
    for q in range(k - 1):
        denominator = math.sqrt((q + 1) * (q + 2))
        for m in range(q + 1):
            c[m][q] = 1.0 / denominator
        c[q + 1][q] = -(q + 1) / denominator
    return c


def run():
    rng = random.Random(20260930)
    h, k = 3, 4
    ga = [[[rng.gauss(0, 1) for _ in range(h)] for _ in range(h)] for _ in range(k)]
    gb = [[[rng.gauss(0, 1) for _ in range(h)] for _ in range(h)] for _ in range(k)]
    da, db = centered(ga), centered(gb)
    cross = sym(output_covariance(da, db))
    values, u = jacobi(cross)
    p = projector(u, [0])
    v = [[row[0]] for row in u]
    contrasts = helmert(k)
    eta = 0.013
    coefficients = []
    for q in range(k - 1):
        gradient = zeros(1, h)
        for m in range(k):
            gradient = add(gradient, multiply(transpose(v), ga[m]), contrasts[m][q] / k)
        coefficients.append(scaled(gradient, -eta))
    private = []
    for m in range(k):
        correction = zeros(h, h)
        for q in range(k - 1):
            correction = add(correction, multiply(v, coefficients[q]), contrasts[m][q])
        private.append(correction)
    native_expected = [scaled(multiply(p, d), -eta / k) for d in da]
    coordinate_error = max(abs(x - y) for a, b in zip(private, native_expected)
                           for ar, br in zip(a, b) for x, y in zip(ar, br))
    shared = scaled(average(ga), -eta)
    losses_shared = [(inner(g, shared) + 0.5 * inner(shared, shared)) for g in gb]
    weights_private = [add(shared, d) for d in private]
    losses_private = [(inner(g, w) + 0.5 * inner(w, w)) for g, w in zip(gb, weights_private)]
    actual = sum(b - a for a, b in zip(losses_shared, losses_private)) / k
    recurrence = trace(multiply(p, cross))
    predicted = -eta * recurrence / k
    predicted += eta * eta * sum(inner(multiply(p, d), multiply(p, d)) for d in da) / (2 * k**3)
    local_error = abs(actual - predicted)

    # K=2: flattened member contrast rank one, output covariance rank h=3.
    diag = [[float(i == j) * (i + 1) for j in range(h)] for i in range(h)]
    feature_rank_values, _ = jacobi(output_covariance([diag, scaled(diag, -1)],
                                                      [diag, scaled(diag, -1)]))
    assert all(value > 0 for value in feature_rank_values)

    # Positive cross-unit recurrence alone can retain a noisy feature.
    # With two units and a reference mean of two units, the feature whose
    # gradients are 10 and 1 has C=10, N=(10-1)^2/4=20.25, and T=-10.25.
    paired_units = []
    for feature_two in (10.0, 1.0):
        diagonal = [[float(i == j) * ([1.0, feature_two, 0.0][i])
                     for j in range(h)] for i in range(h)]
        paired_units.append([diagonal, scaled(diagonal, -1.0)])
    paired_stats = covariance_statistics(paired_units, reference_batch=2)
    recurrence_values, _ = jacobi(paired_stats["C"])
    corrected_values, _ = jacobi(paired_stats["T"])
    assert recurrence_values == [10.0, 1.0, 0.0]
    assert corrected_values == [1.0, 0.0, -10.25]

    # Fresh graph-unit mean-gradient denoising, fixed P selecting axis one.
    signal = [4.0, 0.5, 0.0]
    n, n0, repeats = 6, 2, 4000
    member_scales = [-1.5, -0.5, 0.5, 1.5]
    scale_norm = math.sqrt(sum(x*x for x in member_scales) / k)
    member_scales = [x / scale_norm for x in member_scales]
    truth = [[[member_scales[m] * math.sqrt(signal[i]) * float(i == j)
               for j in range(h)] for i in range(h)] for m in range(k)]
    noise_sd = math.sqrt(n0 / (h * (1 - 1 / k)))
    def draw():
        error = centered([[[rng.gauss(0, noise_sd) for _ in range(h)]
                           for _ in range(h)] for _ in range(k)])
        return [add(truth[m], error[m]) for m in range(k)]
    accumulated = zeros(h, h)
    risk_delta = 0.0
    fixed_p = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    shared_risk = sum(inner(d, d) for d in truth) / k
    for _ in range(repeats):
        stats = covariance_statistics([draw() for _ in range(n)], n0)
        accumulated = add(accumulated, stats["T"])
        fresh = [average(group) for group in zip(*[draw() for _ in range(n0)])]
        residuals = [add(multiply(fixed_p, fresh[m]), truth[m], -1.0) for m in range(k)]
        risk_delta += sum(inner(d, d) for d in residuals) / k - shared_risk
    empirical_t = scaled(accumulated, 1.0 / repeats)
    empirical_risk_delta = risk_delta / repeats
    expected_t = [[float(i == j) * (signal[i] - 1) for j in range(h)] for i in range(h)]
    covariance_error = max(abs(x-y) for ar, br in zip(empirical_t, expected_t) for x,y in zip(ar,br))
    risk_error = abs(empirical_risk_delta + 3)

    # Dependence counterexample: identical graph noises produce positive signal
    # while population member-demand is zero. No IID graph-unit theorem applies.
    common_noise = centered([[[rng.gauss(0, 1) for _ in range(h)]
                              for _ in range(h)] for _ in range(k)])
    dependent = covariance_statistics([common_noise] * n, n0)
    dependence_false_signal = trace(dependent["T"])
    dependence_noise_estimate = trace(dependent["N"])
    assert dependence_false_signal > 0 and abs(dependence_noise_estimate) < 1e-20

    # An indefinite cross covariance is allowed; clipping it would change the
    # unbiased entrywise estimator. Squaring it would lose recurrence's sign.
    anti = sym(output_covariance([diag, scaled(diag,-1)], [scaled(diag,-1), diag]))
    anti_values, _ = jacobi(anti)
    assert all(value < 0 for value in anti_values)

    assert coordinate_error < 1e-12 and local_error < 1e-12
    assert covariance_error < 0.15 and risk_error < 0.15
    return {
        "status": "PASS",
        "scope": "synthetic stdlib algebra and fixed-projector Monte Carlo only; no dataset or GPU",
        "native_mean_objective_coordinate_error": coordinate_error,
        "native_quadratic_identity_error": local_error,
        "output_feature_eigenvalues_with_two_members": feature_rank_values,
        "flattened_member_contrast_rank_in_that_example": 1,
        "positive_recurrence_example_C_eigenvalues": recurrence_values,
        "positive_recurrence_example_T_eigenvalues": corrected_values,
        "fixed_projector_expected_T_diagonal": [3.0,-0.5,-1.0],
        "fixed_projector_empirical_T": empirical_t,
        "fixed_projector_covariance_max_error": covariance_error,
        "fixed_projector_expected_risk_delta": -3.0,
        "fixed_projector_empirical_risk_delta": empirical_risk_delta,
        "monte_carlo_repeats": repeats,
        "dependent_graph_counterexample_false_signal": dependence_false_signal,
        "dependent_graph_counterexample_estimated_noise": dependence_noise_estimate,
        "negative_recurrence_eigenvalues": anti_values,
        "selected_projector_unbiasedness_claimed": False,
    }


if __name__ == "__main__":
    report = run()
    output = Path(__file__).resolve().parent / "SANITY_REPORT.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
