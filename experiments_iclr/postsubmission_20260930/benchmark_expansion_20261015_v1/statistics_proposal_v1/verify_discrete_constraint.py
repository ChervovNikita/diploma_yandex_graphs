"""Exact arithmetic and local design metadata only; no results/models/data."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DESIGN = PROJECT / "reference_wrapper_redesign_20261013_v1"


def exact_two_sided_tail(differences):
    """Literal complete sign enumeration using integer sums, not floating p."""
    observed = abs(sum(differences))
    extreme = sum(abs(sum(s * d for s, d in zip(signs, differences))) >= observed
                  for signs in product((-1, 1), repeat=len(differences)))
    patterns = 2 ** len(differences)
    return extreme, patterns, Fraction(extreme, patterns)


def holm_rejected(pvalues, alpha):
    ordered = sorted(enumerate(pvalues), key=lambda pair: pair[1])
    rejected = []
    for rank, (index, pvalue) in enumerate(ordered):
        if pvalue > alpha / (len(ordered) - rank):
            break
        rejected.append(index)
    return rejected


def number(value):
    return {"exact": str(value), "decimal": float(value)}


def main():
    paths = [DESIGN / name for name in ("README.md", "ROSTER.json", "PLAN.json", "build_design.py")]
    pins = [{"path": str(p.relative_to(PROJECT)), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    roster = json.loads((DESIGN / "ROSTER.json").read_text())
    plan = json.loads((DESIGN / "PLAN.json").read_text())
    banks = {bank["id"]: bank for bank in plan["logical_banks"]}
    fit_ids = {job["id"] for job in plan["jobs"] if job["kind"] == "fit"}
    contrasts, cohorts, tasks = [], {}, defaultdict(lambda: {"presets": 0, "wrapper_preset_cells": 0,
                                                               "contrasts": 0})
    methods = Counter()
    for recipe in roster["recipes"]:
        rid, task = recipe["id"], recipe["task"]
        trial_rows = recipe["cohorts_by_track"]["long_convergence"]
        assert len(trial_rows) == 10
        assert len({(t["seed"], t["split"]) for t in trial_rows}) == 10
        cohorts[rid] = trial_rows
        tasks[task]["presets"] += 1
        for wrapper in recipe["wrappers"]:
            assert "long_convergence" in wrapper["tracks"]
            wid = wrapper["id"]
            tasks[task]["wrapper_preset_cells"] += 1
            for trial in trial_rows:
                prefix = f"{rid}/long_convergence/split-{trial['split']}/seed-{trial['seed']}"
                wb = banks[prefix + "/wrapper-bank-" + wid]
                mb, ib = banks[prefix + "/native-M1"], banks[prefix + "/native-I4"]
                assert wb["selection"] == "own selected complete shared bank"
                assert ib["selection"] == "four own selected states"
                assert len(ib["fit_ids"]) == 4 and len(set(ib["fit_ids"])) == 4
                assert mb["fit_ids"] == [ib["fit_ids"][0]] and ib["exact_M1_body0_reuse"]
                assert set(wb["fit_ids"] + mb["fit_ids"] + ib["fit_ids"]) <= fit_ids
            for comparator in ("native-M1", "native-I4"):
                contrasts.append({"id": f"{rid}/{wid}-minus-{comparator}", "recipe_id": rid,
                    "task": task, "dataset": recipe["dataset"], "architecture": recipe["architecture"],
                    "wrapper": wid, "comparator": comparator,
                    "source_metric": recipe["official_evaluator"]["source_metric"], "paired_trials": 10,
                    "track": "long_convergence"})
                tasks[task]["contrasts"] += 1
                methods[wid] += 1
    m, n, alpha = len(contrasts), 10, Fraction(1, 20)
    assert m == roster["reporting_policy"]["primary_contrast_count"] == 226
    assert plan["reporting_policy"] == roster["reporting_policy"]
    fit_tracks = Counter(job["contract"]["track"] for job in plan["jobs"] if job["kind"] == "fit")
    assert len(fit_ids) == 2480 and fit_tracks["native_replica"] == 270
    family = {"schema": "registered-local-design-contrast-audit-v1", "design_pins": pins,
        "family_count": m, "wrapper_preset_cells": m // 2, "paired_trials_each": n,
        "task_breakdown": dict(tasks), "wrapper_contrast_counts": dict(methods),
        "cohorts_by_recipe": cohorts, "contrasts": contrasts,
        "plan_bank_references_and_I4_body0_reuse_verified": True,
        "source_reference_fits_unchanged": 270, "total_model_fits_unchanged": 2480,
        "frozen_reporting_policy": roster["reporting_policy"],
        "alpha_point_zero_five_explicit_in_reporting_policy": "alpha" in roster["reporting_policy"],
        "alpha_for_this_audit": number(alpha),
        "alpha_basis": "Root audit instruction uses .05; existing reporting_policy does not explicitly store alpha.",
        "scope": "Local declared proposal, not a claim of external preregistration; no outcome arrays read."}
    (HERE / "REGISTERED_FAMILY.json").write_text(json.dumps(family, indent=2) + "\n")

    tail, patterns, pmin = exact_two_sided_tail(tuple(range(1, n + 1)))
    threshold = alpha / m
    assert tail == 2 and patterns == 1024 and pmin == Fraction(1, 512)
    assert pmin > threshold
    assert holm_rejected([pmin] * m, alpha) == []
    maximum_family = int(alpha // pmin)
    assert maximum_family == 25
    assert len(holm_rejected([pmin] * 25, alpha)) == 25
    assert holm_rejected([pmin] * 26, alpha) == []
    zero_tail, zero_patterns, zero_p = exact_two_sided_tail(tuple(range(1, 10)) + (0,))
    assert zero_p == Fraction(1, 256)
    table = []
    for trials in range(10, 17):
        extreme, count, pvalue = exact_two_sided_tail(tuple(range(1, trials + 1)))
        assert extreme == 2 and pvalue == Fraction(2, 2 ** trials)
        allowed = int(threshold * count)
        table.append({"nonzero_sign_flippable_trials": trials, "patterns": count,
            "minimum_two_sided_p": number(pvalue), "best_possible_first_Holm_adjusted_p": number(m * pvalue),
            "first_rejection_mathematically_possible": pvalue <= threshold,
            "largest_allowed_even_two_sided_tail_count_at_first_step": allowed - allowed % 2})
    first_possible = next(row["nonzero_sign_flippable_trials"] for row in table
                          if row["first_rejection_mathematically_possible"])
    assert first_possible == 14
    # With14, the first-step cutoff permits only2 of16384 sign patterns, so a
    # nonzero mixed-sign observed vector cannot meet that cutoff.
    mixed = exact_two_sided_tail(tuple(range(1, 14)) + (-1,))
    assert mixed[0] >= 4 and mixed[2] > threshold
    assert exact_two_sided_tail((1,) * 10)[2] == pmin
    demonstration = {"schema": "exact-discrete-sign-flip-Holm-demonstration-v1", "alpha": number(alpha),
        "registered_family_m": m, "observed_artificial_differences": list(range(1, 11)),
        "extreme_sign_patterns": tail, "all_sign_patterns": patterns,
        "minimum_two_sided_p_n10": number(pmin), "first_Holm_cutoff": number(threshold),
        "best_possible_first_Holm_adjusted_p_n10": number(m * pmin),
        "Holm_rejections_when_all226_pvalues_equal_the_minimum": 0,
        "largest_equal_weight_Holm_family_with_first_rejection_possible_at_n10": maximum_family,
        "minimum_trials_for_first_possible_rejection_at_m226": first_possible,
        "one_zero_difference_at_n10": {"tail_patterns": zero_tail, "patterns": zero_patterns,
                                       "p": number(zero_p)},
        "trial_count_table": table,
        "n14_first_cutoff_requires_unanimous_nonzero_signs": True,
        "illustrative_iid_sign_probability_of_meeting_initial_cutoff_for_one_cell": [
            {"Pr_positive_difference": float(q), "unanimity_probability": float(q ** 14 + (1-q) ** 14)}
            for q in (Fraction(4, 5), Fraction(9, 10), Fraction(19, 20))],
        "illustration_limit": "Unanimity probability for one cell under an illustrative iid-sign model, not full Holm power, an estimated effect model or a study guarantee.",
        "power_guarantee": False, "models_or_dataset_records_read": 0, "server_operations": 0,
        "frozen_files_modified": False}
    (HERE / "NUMERICAL_DEMONSTRATION.json").write_text(json.dumps(demonstration, indent=2) + "\n")
    for pin, path in zip(pins, paths):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]
    print(json.dumps({"verified_contrasts": m, "task_breakdown": dict(tasks),
        "minimum_p_n10": number(pmin), "first_Holm_cutoff": number(threshold),
        "best_possible_adjusted_p_n10": number(m * pmin), "first_possible_n": first_possible,
        "source270_total2480_unchanged": True, "frozen_file_hashes_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
