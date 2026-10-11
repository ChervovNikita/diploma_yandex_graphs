"""Reused complete native reader math; explicit served-log override for raw-logit comparator."""
import itertools
import json
import math
import statistics
from pathlib import Path


METRICS = ("pooled_accuracy_pct", "pooled_nll", "pooled_brier",
           "mean_member_accuracy_pct", "worst_member_accuracy_pct",
           "mean_member_nll", "worst_member_nll",
           "mean_member_brier", "worst_member_brier")

def require(value, message):
    if not value:
        raise ValueError(message)

def logsumexp(np, x, axis):
    high = x.max(axis=axis, keepdims=True)
    return (high + np.log(np.exp(x - high).sum(axis=axis, keepdims=True))).squeeze(axis)

def strict_mask(np, logits, labels):
    target = logits[:, np.arange(len(labels)), labels]
    better = (logits > target[:, :, None]).all(0)
    better[np.arange(len(labels)), labels] = False
    return better.any(-1)

def cohort(np, mask, correct, labels):
    return {"nodes": int(mask.sum()), "corrected": int((mask & correct).sum()),
            "classes": [{"class": c, "nodes": int((mask & (labels == c)).sum()),
                         "corrected": int((mask & correct & (labels == c)).sum())} for c in range(10)]}

def score(np, log_members, probability, member_errors, pooled_errors, labels, *, pool_log_probability=None):
    member_correct, pooled_correct = ~member_errors, ~pooled_errors
    coverage = member_correct.any(0)
    log_pool = logsumexp(np, log_members, 0) - math.log(len(log_members)) if pool_log_probability is None else pool_log_probability
    truth = np.eye(10, dtype=np.float64)[labels]
    member_nll = -log_members[:, np.arange(len(labels)), labels]
    pool_nll = -log_pool[np.arange(len(labels)), labels]
    member_brier = ((np.exp(log_members) - truth[None]) ** 2).sum(-1)
    pool_brier = ((probability.astype(np.float64) - truth) ** 2).sum(-1)
    def summary(mask):
        n = int(mask.sum())
        require(n > 0, "Every reported class must occur in VALID")
        ma, mn, mb = member_correct[:, mask].mean(1) * 100, member_nll[:, mask].mean(1), member_brier[:, mask].mean(1)
        q = {"pooled_accuracy_pct": float(pooled_correct[mask].mean() * 100),
             "pooled_nll": float(pool_nll[mask].mean()), "pooled_brier": float(pool_brier[mask].mean()),
             "member_accuracy_pct": ma.tolist(), "member_nll": mn.tolist(), "member_brier": mb.tolist(),
             "mean_member_accuracy_pct": float(ma.mean()), "worst_member_accuracy_pct": float(ma.min()),
             "mean_member_nll": float(mn.mean()), "worst_member_nll": float(mn.max()),
             "mean_member_brier": float(mb.mean()), "worst_member_brier": float(mb.max())}
        counts = {"nodes": n, "pooled_correct": int(pooled_correct[mask].sum()),
                  "coverage": int(coverage[mask].sum()),
                  "lost_correct_alternatives": int((coverage & ~pooled_correct & mask).sum()),
                  "aggregation_only_correct": int((~coverage & pooled_correct & mask).sum()),
                  "unavailable_alternatives": int((~coverage & mask).sum()),
                  "served_alternatives": int((coverage & pooled_correct & mask).sum()),
                  "member_correct": member_correct[:, mask].sum(1).tolist()}
        require(counts["pooled_correct"] == counts["coverage"] - counts["lost_correct_alternatives"] + counts["aggregation_only_correct"],
                "Exact coverage/loss/aggregation-only identity")
        return q, counts
    quality, counts = summary(np.ones(len(labels), dtype=bool))
    classes = []
    for c in range(10):
        q, z = summary(labels == c)
        classes.append(dict(q, **z, **{"class": c}))
    return {"quality": quality, "counts": counts, "classes": classes,
            "P": pooled_correct, "C": coverage, "L": coverage & ~pooled_correct,
            "G": ~coverage & pooled_correct, "member_correct": member_correct}

def paired(values):
    deltas = list(values)
    require(len(deltas) == 3 and all(math.isfinite(v) for v in deltas), "Three finite paired seed differences")
    mean, sd = statistics.mean(deltas), statistics.stdev(deltas)
    radius = 4.302652729911275 * sd / math.sqrt(3)
    flips = [abs(statistics.mean(sign * v for sign, v in zip(signs, deltas)))
             for signs in itertools.product((-1, 1), repeat=3)]
    return {"seed_deltas": deltas, "mean": mean, "sample_sd": sd,
            "min": min(deltas), "max": max(deltas),
            "descriptive_95pct_t_interval_df2": [mean - radius, mean + radius],
            "nonnegative_seed_count": sum(v >= 0 for v in deltas),
            "positive_seed_count": sum(v > 0 for v in deltas),
            "exact_sign_flip_two_sided_p": sum(v >= abs(mean) - 1e-12 for v in flips) / 8,
            "uncertainty_scope": "Three paired optimizer seeds on one encountered development graph; descriptive, not independent-node inference or significance."}

def error_diagnostic(np, candidate, reference, labels):
    p, r, c, d = candidate["P"], reference["P"], candidate["C"], reference["C"]
    repair, harm, new, removed = ~r & p, r & ~p, ~d & c, d & ~c
    def counts(mask):
        delta = {k: int(candidate["counts"][k] - reference["counts"][k]) for k in
                 ("pooled_correct", "coverage", "lost_correct_alternatives", "aggregation_only_correct")}
        if not mask.all():
            delta = {"pooled_correct": int(((p.astype(int) - r.astype(int)) * mask).sum()),
                     "coverage": int(((c.astype(int) - d.astype(int)) * mask).sum()),
                     "lost_correct_alternatives": int(((candidate["L"].astype(int) - reference["L"].astype(int)) * mask).sum()),
                     "aggregation_only_correct": int(((candidate["G"].astype(int) - reference["G"].astype(int)) * mask).sum())}
        repairs, harms = int((repair & mask).sum()), int((harm & mask).sum())
        require(repairs - harms == delta["pooled_correct"] == delta["coverage"] - delta["lost_correct_alternatives"] + delta["aggregation_only_correct"],
                "Complete paired repair/harm count identity")
        return {"repairs": repairs, "harms": harms, "count_decomposition_delta": delta,
                "new_coverage": int((new & mask).sum()), "removed_coverage": int((removed & mask).sum()),
                "new_coverage_served": int((new & p & mask).sum()),
                "new_coverage_lost": int((new & ~p & mask).sum()),
                "repair_causes": {"new_member_alternative": int((repair & c & ~d & mask).sum()),
                                  "both_banks_have_member_alternative": int((repair & c & d & mask).sum()),
                                  "aggregation_only": int((repair & ~c & mask).sum())},
                "harm_causes": {"available_alternative_lost": int((harm & c & mask).sum()),
                                "removed_member_alternative": int((harm & ~c & d & mask).sum()),
                                "lost_aggregation_only": int((harm & ~c & ~d & mask).sum())}}
    whole = counts(np.ones(len(labels), dtype=bool))
    whole["classes"] = [dict(counts(labels == c), **{"class": c}) for c in range(10)]
    return whole

def contrast(np, values, candidate, reference, labels, seeds, scope="full_F_procedures"):
    if any(values[s, candidate] is None or values[s, reference] is None for s in seeds):
        return {"candidate": candidate, "reference": reference, "status": "failed_calibration_retained",
                "quality_deltas": None, "missing_seed_readouts": [s for s in seeds if values[s, candidate] is None or values[s, reference] is None]}
    rows = [dict(error_diagnostic(np, values[s, candidate], values[s, reference], labels), seed=s) for s in seeds]
    summed = {"repairs": sum(r["repairs"] for r in rows), "harms": sum(r["harms"] for r in rows),
              "count_decomposition_delta": {k: sum(r["count_decomposition_delta"][k] for r in rows) for k in rows[0]["count_decomposition_delta"]}}
    member_count = len(values[seeds[0], candidate]["quality"]["member_accuracy_pct"])
    equal_members = member_count == len(values[seeds[0], reference]["quality"]["member_accuracy_pct"])
    members = [{"member": m, **{label: paired(values[s, candidate]["quality"][metric][m] - values[s, reference]["quality"][metric][m] for s in seeds)
                                for label, metric in (("accuracy_pp", "member_accuracy_pct"), ("nll", "member_nll"), ("brier", "member_brier"))}}
               for m in range(member_count)] if equal_members else None
    return {"candidate": candidate, "reference": reference, "status": "finite_complete",
            "quality_deltas": {k: paired(values[s, candidate]["quality"][k] - values[s, reference]["quality"][k] for s in seeds) for k in METRICS},
            "classes": [{"class": c, **{k: paired(values[s, candidate]["classes"][c][k] - values[s, reference]["classes"][c][k] for s in seeds) for k in METRICS}} for c in range(10)],
            "per_seed_error_diagnostics": rows, "summed_three_seed_readout_diagnostics": summed,
            "members": members, "member_pairing": "Route index only for equally sized banks; does not certify identical realized bodies or selected steps." if equal_members else "No route pairing for differently sized banks.",
            "comparison_scope": scope, "error_categories_are_descriptive_not_causal": True,
            "scope_explanation": "Fixed selected native bank; only equal-policy temperature changes; no member alternative acquired." if scope == "same_selected_state_calibration" else
                "Separately fitted, stopped and selected complete procedures; matched starts do not imply equal optimizer trajectories. No isolated same-state learning/averaging causal attribution.",
            "coverage_category_note": "Both-bank coverage does not certify the same route/vector or exclusive pooling causality.",
            "count_scope": "Three dependent readouts of the same 5274 development nodes."}

def public_score(value):
    return None if value is None else {k: v for k, v in value.items() if k in ("quality", "counts", "classes")}

def chunk_documents(category, rows):
    docs, chunk = [], []
    for row in rows:
        proposed = {"category": category, "offset": sum(len(d[1]["rows"]) for d in docs), "total_rows": len(rows), "rows": chunk + [row]}
        if len(json.dumps(proposed, indent=2, allow_nan=False).encode()) > 1800000:
            require(chunk, "Single compact report row exceeds transport bound")
            docs.append((f"{category}_part{len(docs) + 1}.json", dict(proposed, rows=chunk)))
            chunk = [row]
        else:
            chunk.append(row)
    if chunk:
        docs.append((f"{category}_part{len(docs) + 1}.json", {"category": category, "offset": sum(len(d[1]["rows"]) for d in docs), "total_rows": len(rows), "rows": chunk}))
    return docs
