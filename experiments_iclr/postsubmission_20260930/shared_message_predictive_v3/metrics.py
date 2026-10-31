"""Complete official validation metrics and descriptive paired seed summaries."""
import math
import statistics
from .contract import ARMS, SEEDS, TASKS


def full_split(member, pooled, targets):
    import torch
    from torch.nn import functional as F
    if member.device.type != "cpu" or pooled.device.type != "cpu" or targets.device.type != "cpu":
        raise ValueError("CPU official validation evidence required")
    m, n, c = member.shape
    y = targets.reshape(-1)
    if (m not in (1, 4) or pooled.shape != (n, c) or y.shape != (n,) or y.dtype != torch.long
            or n == 0 or y.min() < 0 or y.max() >= c
            or not torch.isfinite(member).all() or not torch.isfinite(pooled).all()):
        raise ValueError("Aligned finite complete-split logits and admitted class targets required")
    stats = {"nodes": n, "members": m, "classes": c, "pooled_CE_sum": 0.0,
             "pooled_correct": 0, "member_CE_sum": [0.0] * m, "member_correct": [0] * m,
             "pair_counts": {f"{a}_{b}": {"disagree": 0, "both_wrong": 0, "either_wrong": 0}
                             for a in range(m) for b in range(a)}}
    for start in range(0, n, 65536):
        raw, values, target = member[:, start:start + 65536], pooled[start:start + 65536].double(), y[start:start + 65536]
        stats["pooled_CE_sum"] += float(F.cross_entropy(values, target, reduction="sum"))
        stats["pooled_correct"] += int((values.argmax(-1) == target).sum())
        predictions = []
        for k in range(m):
            values = raw[k].double()
            stats["member_CE_sum"][k] += float(F.cross_entropy(values, target, reduction="sum"))
            predictions.append(values.argmax(-1))
            stats["member_correct"][k] += int((predictions[k] == target).sum())
        for a in range(m):
            for b in range(a):
                entry = stats["pair_counts"][f"{a}_{b}"]
                wrong_a, wrong_b = predictions[a] != target, predictions[b] != target
                entry["disagree"] += int((predictions[a] != predictions[b]).sum())
                entry["both_wrong"] += int((wrong_a & wrong_b).sum())
                entry["either_wrong"] += int((wrong_a | wrong_b).sum())
    return from_statistics(stats), stats


def from_statistics(stats, chosen_member=None):
    n, m, c = stats["nodes"], stats["members"], stats["classes"]
    if type(n) is not int or n <= 0 or m not in (1, 4) or type(c) is not int or c < 2:
        raise ValueError("Invalid sufficient-statistic dimensions")
    if len(stats["member_CE_sum"]) != m or len(stats["member_correct"]) != m:
        raise ValueError("Missing member statistics")
    if any(not math.isfinite(v) or v < 0 for v in [stats["pooled_CE_sum"], *stats["member_CE_sum"]]):
        raise ValueError("Invalid CE sums")
    if any(type(v) is not int or not 0 <= v <= n for v in [stats["pooled_correct"], *stats["member_correct"]]):
        raise ValueError("Invalid correct counts")
    expected = {f"{a}_{b}" for a in range(m) for b in range(a)}
    if set(stats["pair_counts"]) != expected:
        raise ValueError("Incomplete pair counts")
    for entry in stats["pair_counts"].values():
        if set(entry) != {"disagree", "both_wrong", "either_wrong"} or any(type(v) is not int or not 0 <= v <= n for v in entry.values()) or entry["both_wrong"] > entry["either_wrong"]:
            raise ValueError("Invalid pair counts")
    result = {"nodes": n, "members": m, "classes": c, "NLL": stats["pooled_CE_sum"] / n,
              "accuracy_percent": 100 * stats["pooled_correct"] / n,
              "member_NLL": [v / n for v in stats["member_CE_sum"]],
              "member_accuracy_percent": [100 * v / n for v in stats["member_correct"]],
              "pairwise_disagreement": {k: v["disagree"] / n for k, v in stats["pair_counts"].items()},
              "pairwise_error_intersection_fraction": {k: v["both_wrong"] / n for k, v in stats["pair_counts"].items()},
              "pairwise_error_Jaccard": {k: v["both_wrong"] / v["either_wrong"] if v["either_wrong"] else None for k, v in stats["pair_counts"].items()},
              "scoring_precision": "float32 exact-path raw-logit pool; chunked float64 CPU reductions"}
    result["pooled_gain_over_mean_member_accuracy_pp"] = result["accuracy_percent"] - statistics.mean(result["member_accuracy_percent"])
    if chosen_member is not None:
        if chosen_member not in range(m):
            raise ValueError("Frozen validation-selected member outside model")
        result["validation_chosen_member"] = chosen_member
        result["pooled_gain_over_chosen_member_accuracy_pp"] = result["accuracy_percent"] - result["member_accuracy_percent"][chosen_member]
    return result


def choose_member(metrics):
    return min(range(metrics["members"]), key=lambda k: (metrics["member_NLL"][k], k))


def interval(values):
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("Exactly three paired finite seed differences required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)
    return {"values": list(values), "mean": mean, "sample_sd": sd, "two_sided_95_interval": [mean - half, mean + half],
            "n": 3, "df": 2, "scope": "descriptive optimizer-seed variation on a fixed graph and split",
            "assumptions": "approximately normal paired differences; three seeds cannot diagnose normality",
            "significance_or_noninferiority_established": False}


def summarize(rows):
    if len(rows) != 30 or {(r["task"], r["seed"], r["arm"]) for r in rows} != {(t, s, a) for t in TASKS for s in SEEDS for a in ARMS}:
        raise ValueError("Every fixed task/arm/seed outcome is required")
    indexed = {(r["task"], r["seed"], r["arm"]): r for r in rows}
    tasks = {}
    for task in TASKS:
        contrasts = {}
        for arm in ARMS[1:]:
            contrasts[arm] = {metric: interval([indexed[task, s, arm]["final_validation"][metric] - indexed[task, s, "exact_member"]["final_validation"][metric] for s in SEEDS]) for metric in ("accuracy_percent", "NLL")}
        shared = contrasts["shared_mean_message"]
        accuracy, nll = shared["accuracy_percent"]["values"], shared["NLL"]["values"]
        failed = -statistics.mean(accuracy) > 0.5 or -min(accuracy) > 1.0 or statistics.mean(nll) > 0.01
        mixed = (min(accuracy) < 0 < max(accuracy)) or (min(nll) < 0 < max(nll))
        tasks[task] = {"all_final_primary_paired_contrasts": contrasts,
                       "practical_retention_screen": "fails" if failed else "inconclusive_mixed_seeds" if mixed else "within_exploratory_margins",
                       "continued_learning_sequences": [{"seed": s, "arm": a, "last20_training": indexed[task, s, a]["last20_training"], "last30_epoch_validation": indexed[task, s, a]["last30_epoch_validation"]} for s in SEEDS for a in ARMS]}
    return {"every_cell": rows, "tasks": tasks, "primary_epoch": 300, "selected_checkpoint_secondary_only": True,
            "no_node_pseudoreplication": True, "novelty_acceptance_or_convergence_established": False}
