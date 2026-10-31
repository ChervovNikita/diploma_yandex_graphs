"""Replay collected recurrence/Gram summaries and the budgeted hard decision.

Pure standard-library verifier: no torch, dataset, validation or test labels.
Use exactly three checkpoint records (epochs 0,10,25) from one dataset/seed.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import median


COLLECTOR_SCHEMA = "gnnm-recurrence-collector-v2"
FOLD_WEIGHTING = "equal_mean_per_training_unit"
LOSS_RULE = "unscaled own member loss; mean of unit-specific mean finite observed-entry losses"
ADAM_DIAGNOSTIC_CONVENTION = {
    "optimizer": "virtual_Adam_first_step", "moment_initialization": "zero",
    "learning_rate": 0.001, "epsilon": 1e-8, "weight_decay": 0.0,
    "shared_gradient": "mean own-member gradients",
    "private_gradient": "own-member gradient / members",
    "prediction": "negative linearized mean own-member scoring loss difference; symmetric A/B average",
    "scope": "secondary local zero-moment diagnostic; not actual checkpoint AdamW or final benefit",
}


def close(actual, expected, scale):
    if not math.isfinite(actual) or abs(actual - expected) > 2e-10 * max(1.0, scale):
        raise ValueError(f"Gram replay mismatch: {actual} versus {expected}")


def check_stats(stats, members):
    adam = stats["virtual_native_adam_first_step"]
    if adam["convention"] != ADAM_DIAGNOSTIC_CONVENTION:
        raise ValueError("Unexpected virtual Adam convention")
    if any(not math.isfinite(adam[key]) for key in ("gain_A_to_B", "gain_B_to_A", "symmetric_gain")):
        raise ValueError("Nonfinite virtual Adam diagnostic")
    close(adam["symmetric_gain"], (adam["gain_A_to_B"] + adam["gain_B_to_A"]) / 2,
          max(abs(adam["gain_A_to_B"]), abs(adam["gain_B_to_A"])))
    grams = stats["centered_gram"]
    for matrix in [*grams.values(), *stats["raw_gram"].values()]:
        if len(matrix) != members or any(len(row) != members for row in matrix):
            raise ValueError("Wrong member Gram shape")
        if any(not math.isfinite(v) for row in matrix for v in row):
            raise ValueError("Nonfinite Gram entry")
    aa, bb, ab = (grams[key] for key in ("AA", "BB", "AB"))
    c = sum(ab[i][i] for i in range(members)) / members
    va = sum(aa[i][i] for i in range(members)) / members
    vb = sum(bb[i][i] for i in range(members)) / members
    n = (va + vb - 2 * c) / 4
    scale = max(abs(va), abs(vb), abs(c))
    close(stats["C"], c, scale)
    close(stats["N"], n, scale)
    if stats["N"] < 0:
        raise ValueError("Negative recorded noise energy")
    # Centered Gram matrices must be the double-centering of the raw ones.
    for key in ("AA", "BB", "AB"):
        raw, centered = stats["raw_gram"][key], grams[key]
        rows = [sum(row) / members for row in raw]
        columns = [sum(raw[i][j] for i in range(members)) / members for j in range(members)]
        overall = sum(rows) / members
        raw_scale = max(abs(v) for row in raw for v in row)
        for i in range(members):
            for j in range(members):
                close(centered[i][j], raw[i][j] - rows[i] - columns[j] + overall, raw_scale)
    permutations = [sum(ab[i][p[i]] for i in range(members)) / members
                    for p in itertools.permutations(range(members))]
    if len(permutations) != len(stats["member_permutation_C"]):
        raise ValueError("Incomplete member-permutation control")
    for actual, expected in zip(stats["member_permutation_C"], permutations):
        close(actual, expected, scale)
    close(stats["permutation_mean_C"], sum(permutations) / len(permutations), scale)
    close(sum(permutations) / len(permutations), 0.0, scale)
    return c, max(0.0, n)


def replay_and_decide(records, parameter_budget):
    records = sorted(records, key=lambda record: record["checkpoint_epoch"])
    if [r["checkpoint_epoch"] for r in records] != [0, 10, 25]:
        raise ValueError("Need exactly epochs 0,10,25")
    reference = records[0]
    for record in records:
        if record["schema"] != COLLECTOR_SCHEMA or record["members"] != 4:
            raise ValueError("Unrecognized collector schema/member count")
        if record["fold_weighting"] != FOLD_WEIGHTING:
            raise ValueError("Unexpected scoring objective weighting")
        for field in ("dataset_identity", "optimizer_seed", "objective", "probe_fit_unit_ids",
                      "training_unit_fingerprints", "block_parameters", "collector_source_sha256"):
            if record[field] != reference[field]:
                raise ValueError(f"Probe records differ in {field}")
        if not record["state_and_rng_unchanged"]:
            raise ValueError("Failed no-mutation gate")
    partitions = [p["partition_id"] for p in reference["partitions"]]
    if len(partitions) != 3 or len(set(partitions)) != 3:
        raise ValueError("Need exactly three fixed scoring partitions")
    blocks = reference["block_parameters"]
    values = {key: {partition: [] for partition in partitions} for key in blocks}
    secondary_adam = {key: [] for key in blocks}
    fold_identity = {}
    fold_loss_metadata = {}
    for record in records:
        if [p["partition_id"] for p in record["partitions"]] != partitions:
            raise ValueError("Scoring partitions changed during probe")
        fit = set(record["probe_fit_unit_ids"])
        universe = {item["unit_id"] for item in record["training_unit_fingerprints"]}
        if any(item["split"] != "train" for item in record["training_unit_fingerprints"]):
            raise ValueError("Non-training unit in collector evidence")
        q = None
        for partition in record["partitions"]:
            key = partition["partition_id"]
            a, b = partition["A"]["unit_ids"], partition["B"]["unit_ids"]
            for side in ("A", "B"):
                fold = partition[side]
                ids, counts = fold["unit_ids"], fold["observed_entries_by_unit"]
                if (fold["fold_weighting"] != FOLD_WEIGHTING or fold["loss_rule"] != LOSS_RULE
                        or fold["unit_count"] != len(ids) or len(counts) != len(ids)
                        or not ids or any(type(count) is not int or count <= 0 for count in counts)
                        or fold["observed_entries"] != sum(counts)
                        or len(fold["unit_weights"]) != len(ids)):
                    raise ValueError("Invalid equal-unit fold-loss metadata")
                for weight in fold["unit_weights"]:
                    close(weight, 1.0 / len(ids), 1.0)
                if (len(fold["member_mean_loss"]) != record["members"]
                        or any(not math.isfinite(loss) for loss in fold["member_mean_loss"])):
                    raise ValueError("Invalid recorded member losses")
                metadata = {field: fold[field] for field in
                            ("unit_ids", "unit_count", "observed_entries", "observed_entries_by_unit",
                             "unit_weights", "fold_weighting", "loss_rule")}
                identity = (key, side)
                if identity in fold_loss_metadata and metadata != fold_loss_metadata[identity]:
                    raise ValueError("Scoring loss weights/counts changed across checkpoints")
                fold_loss_metadata[identity] = metadata
            if not a or not b or len(set(a)) != len(a) or len(set(b)) != len(b):
                raise ValueError("Empty/duplicate training fold units")
            if set(a) & set(b) or fit & (set(a) | set(b)):
                raise ValueError("Fit/score or A/B units overlap")
            if fit | set(a) | set(b) != universe:
                raise ValueError("Fit/score partition is incomplete")
            if q is not None and q != set(a) | set(b):
                raise ValueError("Q scoring units differ by partition")
            q = set(a) | set(b)
            if key in fold_identity and fold_identity[key] != (a, b):
                raise ValueError("Fold identity changed across checkpoints")
            fold_identity[key] = (a, b)
            if partition["blocks"].keys() != blocks.keys():
                raise ValueError("Missing/extra probed block")
            for block, stats in partition["blocks"].items():
                check_stats(stats, record["members"])
                # Use the recorded direct-vector reductions after verifying
                # their Gram equivalence, to retain the collector's rounding.
                values[block][key].append((stats["C"], stats["N"]))
                secondary_adam[block].append(stats["virtual_native_adam_first_step"]["symmetric_gain"])
    scores = {}
    for block, by_partition in values.items():
        s = max(0.0, median(c for pairs in by_partition.values() for c, _ in pairs))
        n = median(noise for pairs in by_partition.values() for _, noise in pairs)
        q = s / (s + n) if s + n > 0 else 0.0
        cost = 3 * blocks[block]["count"]
        if cost <= 0:
            raise ValueError("Empty block cost")
        stable = all(median(c for c, _ in pairs) > 0 for pairs in by_partition.values())
        eligible = s > n and stable
        scores[block] = {"S": s, "N": n, "q": q, "stable_positive_recurrence": stable,
                         "eligible": eligible, "additional_parameters": cost,
                         "surrogate_gain": s - n,
                         "score_per_parameter": (s - n) / cost,
                         "confidence_v1_score_per_parameter": (2 * q - 1) / cost,
                         "secondary_virtual_adam_zero_moment_median_gain": median(secondary_adam[block]),
                         "secondary_virtual_adam_zero_moment_gain_per_parameter": median(secondary_adam[block]) / cost}
    if parameter_budget < 0:
        raise ValueError("Negative private-parameter budget")
    eligible = sorted((key for key in scores if scores[key]["eligible"]),
                      key=lambda key: (-scores[key]["score_per_parameter"], int(key)))
    selected, remaining = [], parameter_budget
    for key in eligible:
        cost = scores[key]["additional_parameters"]
        if cost <= remaining:
            selected.append(int(key))
            remaining -= cost
            break  # First policy permits at most one private block at any budget.
    return {"status": "PASS", "policy": "recurrence-budget-v2", "scores": scores,
            "fold_weighting": FOLD_WEIGHTING, "maximum_private_blocks": 1,
            "ranking_rule": "(S-N)/additional_parameters; earlier block breaks exact ties",
            "secondary_virtual_adam_convention": ADAM_DIAGNOSTIC_CONVENTION,
            "secondary_virtual_adam_gram_replay_limit": "nonlinear vector diagnostic cannot be recovered from Gram matrices; requires fresh checkpoint autograd",
            "private_indices": selected, "private_parameter_budget": parameter_budget,
            "spent_private_parameters": parameter_budget - remaining,
            "checkpoint_epochs": [0, 10, 25], "optimizer_seed": reference["optimizer_seed"],
            "dataset_identity": reference["dataset_identity"],
            "scope": "train-only selector decision; no predictive score or generalization guarantee"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs=3, required=True)
    parser.add_argument("--parameter-budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = args.output.resolve()
    if here not in output.parents:
        raise SystemExit("Replay output must stay inside its sharing_policy folder")
    records = [json.loads(path.read_text()) for path in args.records]
    result = replay_and_decide(records, args.parameter_budget)
    result["input_record_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in args.records}
    result["replay_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
