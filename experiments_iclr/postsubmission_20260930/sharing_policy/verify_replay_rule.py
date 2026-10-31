"""Exact constructed-Gram checks for v2; pure stdlib, no ML execution."""

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

from replay_recurrence import (ADAM_DIAGNOSTIC_CONVENTION, COLLECTOR_SCHEMA, FOLD_WEIGHTING,
                               LOSS_RULE, replay_and_decide)


def stats(a, b):
    def virtual_gain(fit, score):
        shared = [sum(row[j] for row in fit) / 4 for j in range(len(fit[0]))]
        shared_delta = [-0.001 * g / (abs(g) + 1e-8) for g in shared]
        private_delta = [[-0.001 * (g / 4) / (abs(g / 4) + 1e-8) for g in row] for row in fit]
        private_effect = sum(sum(g * d for g, d in zip(row, delta)) for row, delta in zip(score, private_delta)) / 4
        shared_effect = sum(sum(row[j] for row in score) / 4 * shared_delta[j] for j in range(len(shared)))
        return -(private_effect - shared_effect)
    def gram(left, right):
        return [[sum(x * y for x, y in zip(row, other)) for other in right] for row in left]
    matrices = {"AA": gram(a, a), "BB": gram(b, b), "AB": gram(a, b)}
    c = sum(matrices["AB"][i][i] for i in range(4)) / 4
    n = sum(sum((x - y) ** 2 for x, y in zip(left, right)) for left, right in zip(a, b)) / 16
    permutation_c = [sum(matrices["AB"][i][p[i]] for i in range(4)) / 4
                     for p in itertools.permutations(range(4))]
    ab, ba = virtual_gain(a, b), virtual_gain(b, a)
    return {"C": c, "N": n, "raw_gram": matrices, "centered_gram": copy.deepcopy(matrices),
            "member_permutation_C": permutation_c,
            "permutation_mean_C": sum(permutation_c) / len(permutation_c),
            "virtual_native_adam_first_step": {"convention": ADAM_DIAGNOSTIC_CONVENTION,
                                               "gain_A_to_B": ab, "gain_B_to_A": ba,
                                               "symmetric_gain": (ab + ba) / 2}}


def records_for(blocks):
    def fold(unit, count):
        return {"unit_ids": [unit], "unit_count": 1, "observed_entries": count,
                "observed_entries_by_unit": [count], "unit_weights": [1.0],
                "member_mean_loss": [1.0] * 4, "fold_weighting": FOLD_WEIGHTING, "loss_rule": LOSS_RULE}
    return [{"schema": COLLECTOR_SCHEMA, "checkpoint_epoch": epoch, "members": 4,
             "dataset_identity": {"kind": "constructed-v2-Gram-check-only"},
             "optimizer_seed": 0, "objective": "bce", "fold_weighting": FOLD_WEIGHTING,
             "probe_fit_unit_ids": ["P"], "training_unit_fingerprints":
             [{"unit_id": key, "split": "train"} for key in ("P", "A", "B")],
             "block_parameters": {key: {"count": 10, "names": [f"block{key}"]} for key in blocks},
             "collector_source_sha256": "constructed-no-autograd", "state_and_rng_unchanged": True,
             "partitions": [{"partition_id": str(partition), "A": fold("A", 121), "B": fold("B", 242),
                             "blocks": copy.deepcopy(blocks)} for partition in (701, 702, 703)]}
            for epoch in (0, 10, 25)]


def reject(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("Invalid evidence accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    output = args.output.resolve()
    if folder not in output.parents or output.exists():
        raise ValueError("Use a fresh output file inside sharing_policy")
    u, v = (-1, -1, 1, 1), (-1, 1, -1, 1)
    small = [[x, 0] for x in u]
    high_a, high_b = [[9 * x, 2 * y] for x, y in zip(u, v)], [[9 * x, 0] for x in u]
    records = records_for({"0": stats(small, small), "1": stats(high_a, high_b)})
    result = replay_and_decide(records, 30)
    assert result["scores"]["0"]["surrogate_gain"] == 1
    assert result["scores"]["1"]["surrogate_gain"] == 80
    assert result["scores"]["0"]["confidence_v1_score_per_parameter"] > result["scores"]["1"]["confidence_v1_score_per_parameter"]
    assert result["private_indices"] == [1]
    # Adam's epsilon=0 first-step sign normalization erases these positive
    # amplitude differences. This is a limitation, not the training recipe.
    raw_member_gradients = [x[0] + 2 for x in small]
    native_private_gradients = [g / 4 for g in raw_member_gradients]
    native_tied_gradient = sum(raw_member_gradients) / 4
    assert all(g / abs(g) == native_tied_gradient / abs(native_tied_gradient)
               for g in native_private_gradients)
    assert result["scores"]["0"]["S"] == 1
    assert result["scores"]["0"]["q"] == 1
    assert replay_and_decide(records, 60)["private_indices"] == [1]
    assert replay_and_decide(records, 0)["private_indices"] == []
    tied = records_for({"0": stats(small, small), "1": stats(small, small)})
    assert replay_and_decide(tied, 30)["private_indices"] == [0]
    opposed = records_for({"0": stats(small, [[-x, 0] for x in u])})
    assert replay_and_decide(opposed, 30)["private_indices"] == []
    malformed = copy.deepcopy(records)
    malformed[0]["partitions"][0]["A"]["unit_weights"] = [0.5]
    reject(lambda: replay_and_decide(malformed, 30))
    legacy = copy.deepcopy(records)
    legacy[0]["schema"] = "gnnm-recurrence-collector-v1"
    reject(lambda: replay_and_decide(legacy, 30))
    report = {"status": "PASS", "scope": "exact integer-vector/Gram arithmetic; no training or predictive evidence",
              "policy": "recurrence-budget-v2", "unequal_amplitude_gains": [1, 80],
              "v1_confidence_ranking_choice": 0, "v2_net_gain_ranking_choice": result["private_indices"],
              "at_most_one_with_large_budget": "PASS", "zero_budget": "PASS",
              "earlier_block_exact_tie": "PASS", "opposed_recurrence": "PASS",
              "wrong_weight_and_legacy_schema_rejection": "PASS",
              "positive_recurrence_does_not_force_distinct_first_Adam_step_epsilon0": "PASS",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), folder / "replay_recurrence.py")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
