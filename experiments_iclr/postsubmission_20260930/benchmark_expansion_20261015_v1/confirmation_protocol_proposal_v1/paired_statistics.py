"""Prospective repeat-training summaries; no model, dataset or automatic I/O.

The exact sign test concerns P(paired improvement > 0), not the mean effect.
Bootstrap intervals are explicitly approximate and conditional on these trials.
"""
from __future__ import annotations

import math
import random
import statistics


def sign_pvalue(wins: int, total: int) -> float:
    if type(wins) is not int or type(total) is not int or not 0 <= wins <= total or total < 1:
        raise ValueError("Fixed positive trial count and integer wins required")
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / (2 ** total)


def paired_summary(candidate, baseline, *, bootstrap_seed: int, draws: int = 10000):
    a, b = list(candidate), list(baseline)
    if len(a) != len(b) or len(a) < 2 or type(bootstrap_seed) is not int or draws < 1000:
        raise ValueError("Aligned paired trials and explicit bootstrap seed required")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in a + b):
        raise ValueError("Keep unresolved outcomes in the ledger; no silent finite-only summary")
    differences = [x - y for x, y in zip(a, b)]
    n = len(differences)
    rng = random.Random(bootstrap_seed)
    boot = sorted(statistics.fmean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    wins = sum(d > 0 for d in differences)
    return {
        "paired_trials": n,
        "differences": differences,
        "mean_difference": statistics.fmean(differences),
        "SD_of_paired_differences": statistics.stdev(differences),
        "median_difference": statistics.median(differences),
        "wins": wins,
        "ties": sum(d == 0 for d in differences),
        "losses": sum(d < 0 for d in differences),
        "one_sided_exact_sign_p": sign_pvalue(wins, n),
        "approximate_paired_bootstrap_mean_95pct": [boot[int(.025 * draws)], boot[math.ceil(.975 * draws) - 1]],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": draws,
        "interval_scope": "approximate repeat-training variability conditional on the declared data protocol",
        "sign_test_estimand": "probability of a strictly positive paired difference; ties are non-wins",
        "mean_superiority_test": False,
    }


def joint_confirmation(candidate, baselines, *, unresolved=None, expected_trials=20):
    """One frozen joint claim versus exactly M1, base GNNM and native I4.

    Inference-only unresolved entries count as non-wins in the fixed trial count.
    They never acquire fabricated metric values or enter a finite-only mean.
    """
    if set(baselines) != {"native_M1", "base_GNNM", "native_I4"}:
        raise ValueError("Exact three-baseline family required")
    a = list(candidate)
    flags = list(unresolved) if unresolved is not None else [False] * len(a)
    if expected_trials != 20 or len(a) != expected_trials or len(flags) != len(a) or any(type(x) is not bool for x in flags):
        raise ValueError("Frozen20-trial confirmation and aligned explicit unresolved flags required")
    components = {}
    for name, raw in baselines.items():
        b = list(raw)
        if len(b) != len(a):
            raise ValueError("Every comparator needs all declared trial positions")
        wins = 0
        for x, y, failed in zip(a, b, flags):
            if failed:
                continue
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y)):
                raise ValueError("Nonfinite trial must be explicitly recorded as unresolved")
            wins += x > y
        components[name] = {"wins": wins, "total_including_unresolved": expected_trials,
                            "one_sided_exact_sign_p": sign_pvalue(wins, expected_trials)}
    joint = max(row["one_sided_exact_sign_p"] for row in components.values())
    return {"components": components, "joint_intersection_union_p": joint,
            "alpha": .05, "joint_repeat_win_claim_rejected_null": joint <= .05,
            "unresolved_trials": sum(flags), "mean_superiority_test": False,
            "graph_or_dataset_population_test": False}
