"""Chunked full-split metrics; target tensors must already be separately admitted."""
import math
import statistics

T95 = 2.91998558


def interval(values):
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("Exactly three finite paired optimizer-seed differences required")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T95 * sd / math.sqrt(3)
    return {"values": list(values), "mean": mean, "sample_sd": sd,
            "one_sided_95_lower": mean - half, "one_sided_95_upper": mean + half,
            "two_sided_90_interval": [mean - half, mean + half],
            "n": 3, "df": 2, "t_one_sided_95": T95,
            "assumption": "paired optimizer-seed differences treated as approximately normal; n=3 cannot diagnose normality",
            "scope": "optimizer-seed variation on one fixed graph/split; not independent nodes or graph population"}


def full_split(logits, targets, selected_member=None, pooled_logits=None):
    import torch
    from torch.nn import functional as F
    if logits.device.type != "cpu" or targets.device.type != "cpu" or logits.ndim != 3:
        raise ValueError("CPU member-by-official-split logits and admitted targets required")
    members, n, classes = logits.shape
    targets = targets.reshape(-1)
    if n != targets.numel() or n == 0 or not torch.isfinite(logits).all():
        raise ValueError("Complete official split logits/target shape or finiteness failed")
    if targets.dtype != torch.long or targets.min() < 0 or targets.max() >= classes:
        raise ValueError("Admitted integer class targets required")
    if pooled_logits is not None and (pooled_logits.device.type != "cpu" or pooled_logits.shape != (n, classes)
                                       or not torch.isfinite(pooled_logits).all()):
        raise ValueError("Exact source-path raw pooled logits must align with complete split")
    member_ce = [0.0] * members
    member_correct = [0] * members
    ce = brier = 0.0
    correct = 0
    bins_n, bins_conf, bins_correct = [0] * 15, [0.0] * 15, [0] * 15
    disagreements = {(a, b): 0 for a in range(members) for b in range(a)}
    probability_distance = {pair: 0.0 for pair in disagreements}
    for start in range(0, n, 65536):
        raw = logits[:, start:start + 65536]
        y = targets[start:start + 65536]
        # Raw-logit pooling remains float32; scoring reductions are float64.
        pooled = (raw.mean(dim=0) if pooled_logits is None else pooled_logits[start:start + 65536]).double()
        probability = pooled.softmax(dim=-1)
        prediction = probability.argmax(dim=-1)
        confidence = probability.max(dim=-1).values
        ce += float(F.cross_entropy(pooled, y, reduction="sum"))
        correct += int((prediction == y).sum())
        brier += float((probability.square().sum(-1) - 2 * probability.gather(1, y[:, None]).squeeze(1) + 1).sum())
        bins = torch.clamp((confidence * 15).to(torch.long), max=14)
        for k in range(15):
            mask = bins == k
            bins_n[k] += int(mask.sum())
            bins_conf[k] += float(confidence[mask].sum())
            bins_correct[k] += int((prediction[mask] == y[mask]).sum())
        member_predictions, member_probabilities = [], []
        for m in range(members):
            values = raw[m].double()
            member_ce[m] += float(F.cross_entropy(values, y, reduction="sum"))
            member_predictions.append(values.argmax(-1))
            member_probabilities.append(values.softmax(-1))
            member_correct[m] += int((member_predictions[-1] == y).sum())
        for a, b in disagreements:
            disagreements[a, b] += int((member_predictions[a] != member_predictions[b]).sum())
            probability_distance[a, b] += float((member_probabilities[a] - member_probabilities[b]).square().sum())
    member_ce = [v / n for v in member_ce]
    result = {"nodes": n, "members": members, "classes": classes, "CE": ce / n,
              "accuracy_percent": 100 * correct / n, "Brier_unnormalized": brier / n,
              "ECE_15_equal_width": sum(abs(bins_correct[k] - bins_conf[k]) for k in range(15)) / n,
              "member_CE": member_ce, "member_accuracy_percent": [100 * v / n for v in member_correct],
              "mean_member_CE_minus_pooled_CE": statistics.mean(member_ce) - ce / n,
              "pair_prediction_disagreement": {"%d_%d" % p: v / n for p, v in disagreements.items()},
              "pair_probability_squared_distance": {"%d_%d" % p: v / n for p, v in probability_distance.items()},
              "scoring_precision": "float32_raw_logit_pool_then_float64_CPU_metric_reductions",
              "ECE_bins": {"count": bins_n, "confidence_sum": bins_conf, "correct_count": bins_correct}}
    if selected_member is not None:
        if selected_member not in range(members):
            raise ValueError("Frozen validation-chosen member index outside model")
        result["validation_chosen_member"] = selected_member
        result["chosen_member_CE_minus_pooled_CE"] = member_ce[selected_member] - result["CE"]
    return result


def choose_member(validation_metrics):
    # Index tie rule is fixed independently of test outcomes.
    return min(range(validation_metrics["members"]), key=lambda m: (validation_metrics["member_CE"][m], m))


def gates(rows, timings, fit_costs, preparations):
    """Every task/seed/arm is reported; missing component evidence cannot pass."""
    from .contract import TASKS, ARMS
    if {(r["task"], r["seed"], r["arm"]) for r in rows} != {(t, s, a) for t in TASKS for s in range(3) for a in ARMS} or len(rows) != 30:
        raise ValueError("Complete fixed30 final outcomes required")
    indexed = {(r["task"], r["seed"], r["arm"]): r for r in rows}
    output = {}
    for task in TASKS:
        report = {"quality": {}, "diversity": {}, "controls": {}, "cost": {}}
        def contrasts(split, first, second, metric):
            return [indexed[task, s, first][split][metric] - indexed[task, s, second][split][metric] for s in range(3)]
        for split in ("validation", "test"):
            ce = interval(contrasts(split, "candidate", "exact_member", "CE"))
            accuracy = interval(contrasts(split, "exact_member", "candidate", "accuracy_percent"))
            brier = interval(contrasts(split, "candidate", "exact_member", "Brier_unnormalized"))
            quality = (max(ce["values"]) <= 0.01 and ce["one_sided_95_upper"] <= 0.01
                       and max(accuracy["values"]) <= 0.5 and accuracy["one_sided_95_upper"] <= 0.5
                       and max(brier["values"]) <= 0.01 and brier["one_sided_95_upper"] <= 0.01)
            report["quality"][split] = {"CE_increase": ce, "accuracy_decrease_pp": accuracy,
                                         "Brier_increase": brier, "passed": quality}
        for control in ("naive_shared_sample", "detached_current_member_mean"):
            gain = interval(contrasts("test", control, "candidate", "CE"))
            loss = interval(contrasts("test", control, "candidate", "accuracy_percent"))
            report["controls"][control] = {"CE_gain": gain, "accuracy_loss_pp": loss,
                "passed": gain["mean"] >= 0.005 and gain["one_sided_95_lower"] > 0 and loss["mean"] <= 0.5}
        exact_base = interval(contrasts("test", "single_base", "exact_member", "CE"))
        exact_own = interval([indexed[task, s, "exact_member"]["test"]["chosen_member_CE_minus_pooled_CE"] for s in range(3)])
        candidate_base = interval(contrasts("test", "single_base", "candidate", "CE"))
        candidate_own = interval([indexed[task, s, "candidate"]["test"]["chosen_member_CE_minus_pooled_CE"] for s in range(3)])
        preserve_base = interval([candidate_base["values"][s] - 0.5 * exact_base["values"][s] for s in range(3)])
        candidate_exact_member = interval([indexed[task, s, "exact_member"]["test"]["member_CE"][
            indexed[task, s, "exact_member"]["test"]["validation_chosen_member"]]
            - indexed[task, s, "candidate"]["test"]["CE"] for s in range(3)])
        preserve_own = interval([candidate_exact_member["values"][s] - 0.5 * exact_own["values"][s] for s in range(3)])
        report["diversity"] = {"exact_over_single": exact_base, "exact_over_own_chosen_member": exact_own,
            "candidate_over_single": candidate_base, "candidate_over_own_chosen_member": candidate_own,
            "candidate_over_exact_validation_chosen_member": candidate_exact_member,
            "candidate_minus_half_exact_gain_single": preserve_base,
            "candidate_minus_half_exact_gain_against_exact_chosen_member": preserve_own,
            "passed": exact_base["mean"] >= 0.005 and exact_own["mean"] >= 0.005
                      and exact_base["one_sided_95_lower"] > 0 and exact_own["one_sided_95_lower"] > 0
                      and candidate_own["one_sided_95_lower"] > 0
                      and preserve_base["mean"] >= 0 and preserve_own["mean"] >= 0}
        costs = []
        for seed in range(3):
            candidate = timings[task][str(seed)]["candidate"]
            exact = timings[task][str(seed)]["exact_member"]
            component = candidate.get("propagation_median_seconds")
            reference = exact.get("propagation_median_seconds")
            passed = (candidate["median_seconds"] <= .75 * exact["median_seconds"]
                      and candidate["p90_seconds"] <= .90 * exact["p90_seconds"]
                      and component is not None and reference is not None and component <= .70 * reference)
            costs.append({"seed": seed, "candidate": candidate, "fastest_qualified_exact": exact, "passed": passed})
        candidate_fit = sum(fit_costs[task][str(s)]["candidate"] for s in range(3)) + preparations[task]["candidate"]
        exact_fit = sum(fit_costs[task][str(s)]["exact_member"] for s in range(3)) + preparations[task]["exact_member"]
        report["cost"] = {"paired_seeds": costs, "candidate_fit_validation_plus_preparation_seconds": candidate_fit,
                          "exact_fit_validation_plus_preparation_seconds": exact_fit,
                          "passed": all(c["passed"] for c in costs) and candidate_fit <= exact_fit}
        report["all_original_and_detached_gates_pass"] = (all(r["passed"] for r in report["quality"].values())
            and report["diversity"]["passed"] and all(r["passed"] for r in report["controls"].values()) and report["cost"]["passed"])
        output[task] = report
    return {"tasks": output, "promotable_scoped_pilot": all(r["all_original_and_detached_gates_pass"] for r in output.values()),
            "originality_or_general_acceptance_established": False,
            "all_outcomes_reported": True, "node_independence_assumed": False}
