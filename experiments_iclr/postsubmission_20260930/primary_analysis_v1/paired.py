"""Prospectively fixed paired summaries for the three-seed MolHIV pilot.

These describe optimization variation on one development split. They do not
estimate variation over datasets or establish independent confirmation.
"""
from itertools import product
import math
import statistics

T_CRITICAL_DF2_975 = 4.302652729911275


def paired_summary(candidate, control):
    if len(candidate) != 3 or len(control) != 3:
        raise ValueError("Require every paired seed 0/1/2, in that order")
    if any(not math.isfinite(x) for x in [*candidate, *control]):
        raise ValueError("Incomplete/nonfinite contrast")
    differences = [a - b for a, b in zip(candidate, control)]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    half_width = T_CRITICAL_DF2_975 * sd / math.sqrt(3)
    observed = abs(math.fsum(differences))
    # Enumerating all eight assignments avoids Monte Carlo error. This test
    # requires sign exchangeability/symmetry under the paired null; it is not
    # an assumption-free permutation test or a test across independent graphs.
    assignments = [abs(math.fsum(s * d for s, d in zip(signs, differences)))
                   for signs in product((-1, 1), repeat=3)]
    tolerance = 16 * math.ulp(max(1.0, observed))
    tail = sum(value >= observed - tolerance for value in assignments)
    return {"seeds": [0, 1, 2], "paired_differences": differences,
            "mean_difference": mean, "sd_difference": sd,
            "paired_t_95_interval": [mean - half_width, mean + half_width],
            "paired_t_assumption": "independent approximately normal seed differences on this fixed split",
            "exact_two_sided_sign_flip_p": tail / 8,
            "sign_flip_assignments": 8,
            "sign_flip_assumption": "paired null differences are jointly sign exchangeable",
            "smallest_attainable_two_sided_p_with_three_nonzero_pairs": 0.25,
            "practical_primary_signal": mean >= 0.005 and all(x > 0 for x in differences),
            "statistical_significance_claim": False,
            "independent_graph_or_task_confirmation": False}


def holm_adjust(p_values):
    if len(p_values) != 8 or len(set(p_values)) != 8:
        raise ValueError("Require the complete eight-contrast primary family")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in p_values.values()):
        raise ValueError("Invalid p-value")
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted, running = {}, 0.0
    for position, name in enumerate(ordered):
        running = max(running, (len(ordered) - position) * p_values[name])
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in p_values}
