"""Disabled post-score accounting for immutable Amazon served predictions.

No model calls, label access, fitting, selection, file writes or acceptance gate.
The existing complete-endpoint evaluation must finish before a released caller
can supply its already scored arrays here. Synthetic checks use _analyze only.
"""

import math

SOURCE_RELEASED = False
MEMBERS, CLASSES = 4, 5
DEGREE_BANDS = ("0", "1-2", "3-5", "6-10", "11-20", ">20")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _argmax(values):
    values = tuple(float(x) for x in values)
    return values.index(max(values))  # First class wins a tie, as in serving.


def _band(degree):
    for index, upper in enumerate((0, 2, 5, 10, 20)):
        if degree <= upper:
            return DEGREE_BANDS[index]
    return DEGREE_BANDS[-1]


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))


def _state_rows(state, labels):
    logits = state["native_FP32_logits"]
    members = state["served_FP32_member_probabilities"]
    pool = state["served_FP32_pool_probabilities"]
    n = len(labels)
    _require(len(logits) == len(members) == MEMBERS and len(pool) == n,
             "Require all four members and aligned served pool")
    _require(all(len(x) == n for x in logits) and all(len(x) == n for x in members),
             "Incomplete member rows")
    rows = []
    for i, label in enumerate(labels):
        p = tuple(float(x) for x in pool[i])
        z = [tuple(float(x) for x in logits[m][i]) for m in range(MEMBERS)]
        pm = [tuple(float(x) for x in members[m][i]) for m in range(MEMBERS)]
        _require(len(p) == CLASSES and all(len(x) == CLASSES for x in z + pm),
                 "Class shape differs")
        _require(all(math.isfinite(x) for values in [p, *z, *pm] for x in values),
                 "Nonfinite served values")
        for values in [p, *pm]:
            _require(all(0 <= x <= 1 for x in values)
                     and abs(math.fsum(values) - 1) <= 2e-6,
                     "Invalid served probability row")
        _require(all(abs(p[c] - math.fsum(pm[m][c] for m in range(MEMBERS)) / MEMBERS)
                     <= 2e-7 for c in range(CLASSES)), "Not the served mean pool")
        prediction = _argmax(p)
        member_predictions = [_argmax(values) for values in z]
        shifted = [[x - max(values) for x in values] for values in z]
        member_logs = [values[label] - _logsumexp(values) for values in shifted]
        stable_nll = -(_logsumexp(member_logs) - math.log(MEMBERS))
        rows.append({"prediction": prediction, "correct": prediction == label,
                     "all_members_wrong": all(x != label for x in member_predictions),
                     "correct_members": sum(x == label for x in member_predictions),
                     "shared_strict_opponents": tuple(c for c in range(CLASSES)
                         if c != label and all(values[c] > values[label] for values in pm)),
                     "NLL": stable_nll,
                     "Brier": math.fsum((x - int(c == label)) ** 2
                                        for c, x in enumerate(p)),
                     "confidence": max(p), "true_class_probability": p[label]})
    return rows


def _summarize(initial, endpoint, indices):
    n = len(indices)
    counts = {name: 0 for name in ("wrong_to_correct", "correct_to_wrong",
               "correct_to_correct", "wrong_to_wrong", "wrong_class_changed",
               "initial_all_members_wrong", "initial_all_members_wrong_pool_errors",
               "common_wrong_pool_errors_repaired", "common_wrong_member_created")}
    differences = {name: [] for name in ("NLL", "Brier", "confidence",
                                          "true_class_probability")}
    initial_wrong_differences = {name: [] for name in differences}
    for i in indices:
        before, after = initial[i], endpoint[i]
        transition = ("correct" if before["correct"] else "wrong") + "_to_" + (
            "correct" if after["correct"] else "wrong")
        counts[transition] += 1
        if not before["correct"] and not after["correct"]:
            counts["wrong_class_changed"] += before["prediction"] != after["prediction"]
        if before["all_members_wrong"]:
            counts["initial_all_members_wrong"] += 1
            if not before["correct"]:
                counts["initial_all_members_wrong_pool_errors"] += 1
                counts["common_wrong_pool_errors_repaired"] += after["correct"]
                counts["common_wrong_member_created"] += after["correct_members"] > 0
        for name in differences:
            change = after[name] - before[name]
            differences[name].append(change)
            if not before["correct"]:
                initial_wrong_differences[name].append(change)
    _require(sum(counts[x] for x in ("wrong_to_correct", "correct_to_wrong",
                 "correct_to_correct", "wrong_to_wrong")) == n, "Transition identity failed")
    net = counts["wrong_to_correct"] - counts["correct_to_wrong"]
    correct_change = sum(endpoint[i]["correct"] - initial[i]["correct"] for i in indices)
    _require(net == correct_change, "Net repair identity failed")
    def means(values):
        return {k: math.fsum(v) / len(v) if v else None for k, v in values.items()}
    def metrics(rows):
        return {"accuracy": sum(rows[i]["correct"] for i in indices) / n if n else None,
                **{name: math.fsum(rows[i][name] for i in indices) / n if n else None
                   for name in ("NLL", "Brier")}}
    return {"n": n, "counts": counts, "net_correct_nodes": net,
            "net_accuracy_change": net / n if n else None,
            "initial_metrics": metrics(initial), "endpoint_metrics": metrics(endpoint),
            "mean_endpoint_minus_initial": means(differences),
            "initial_pool_wrong_n": counts["wrong_to_correct"] + counts["wrong_to_wrong"],
            "initial_pool_wrong_mean_endpoint_minus_initial": means(initial_wrong_differences)}


def _analyze(initial_state, endpoint_state, labels, node_ids, degrees):
    n = len(labels)
    _require(n > 0 and len(node_ids) == len(degrees) == n, "Population alignment failed")
    _require(all(int(x) == x and int(x) >= 0 for x in node_ids), "Invalid node ID")
    _require(len(set(int(x) for x in node_ids)) == n, "Duplicate node IDs")
    for state in (initial_state, endpoint_state):
        _require(len(state["A_ids"]) == n
                 and all(int(x) == x and int(x) == int(y)
                         for x, y in zip(state["A_ids"], node_ids)),
                 "Baseline and endpoint IDs must match the same ordered population")
    _require(all(int(x) == x and 0 <= int(x) < CLASSES for x in labels), "Invalid label")
    _require(all(int(x) == x and int(x) >= 0 for x in degrees), "Invalid degree")
    labels = [int(x) for x in labels]
    initial = _state_rows(initial_state, labels)
    endpoint = _state_rows(endpoint_state, labels)
    all_indices = list(range(n))
    common = [i for i in all_indices if initial[i]["all_members_wrong"]
              and not initial[i]["correct"]]
    shared_opponent_errors = [i for i in all_indices
        if initial[i]["shared_strict_opponents"] and not initial[i]["correct"]]
    return {"schema": "amazon_fixed_paired_error_flow_v1",
            "full_population": _summarize(initial, endpoint, all_indices),
            "baseline_all_members_wrong_pool_errors": _summarize(initial, endpoint, common),
            "baseline_shared_strict_opponent_pool_errors": _summarize(
                initial, endpoint, shared_opponent_errors),
            "shared_opponent_repair_details": {
                "initial_pool_error_rows": len(shared_opponent_errors),
                "repaired_without_a_correct_member": sum(endpoint[i]["correct"]
                    and endpoint[i]["correct_members"] == 0 for i in shared_opponent_errors),
                "definition": "A non-target class has strictly larger saved served probability "
                    "than the target for every member at the baseline. The cohort also requires "
                    "an actual baseline served-pool error. Native-logit argmax defines member correctness. "
                    "No claim of exact agreement between FP32 pooling and real arithmetic is made."},
            "classes": [{"class": cls, **_summarize(initial, endpoint,
                            [i for i in all_indices if labels[i] == cls])}
                        for cls in range(CLASSES)],
            "degree_bands": [{"band": band, **_summarize(initial, endpoint,
                               [i for i in all_indices if _band(degrees[i]) == band])}
                             for band in DEGREE_BANDS],
            "interpretation": "Descriptive paired transitions on the fixed scored population; "
                "no independent-node significance test, causal attribution or acceptance decision. "
                "Negative NLL/Brier change means improved probability loss. "
                "Confidence is evaluated on the same baseline-error nodes."}


def analyze(initial_state, endpoint_state, labels, node_ids, degrees):
    if not SOURCE_RELEASED:
        raise RuntimeError("Disabled post-score source; immutable complete evaluation required")
    return _analyze(initial_state, endpoint_state, labels, node_ids, degrees)
