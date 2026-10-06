"""Arithmetic and alignment checks only; contains no dataset/model access."""

from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("error_flow", HERE / "error_flow.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def state(predictions, ids):
    probabilities = [[0.8 if c == prediction else 0.05 for c in range(5)]
                     for prediction in predictions]
    return {"A_ids": ids,
            "native_FP32_logits": [[[math.log(x) for x in row] for row in probabilities]
                                    for _ in range(4)],
            "served_FP32_member_probabilities": [probabilities for _ in range(4)],
            "served_FP32_pool_probabilities": probabilities}


def main():
    labels = [0, 1, 2, 3, 4, 0]
    ids, degrees = [10, 11, 12, 13, 14, 15], [0, 2, 4, 8, 17, 21]
    initial = state([1, 0, 2, 1, 4, 2], ids)
    endpoint = state([0, 1, 0, 2, 4, 2], ids)
    result = module._analyze(initial, endpoint, labels, ids, degrees)
    full = result["full_population"]
    expected = {"wrong_to_correct": 2, "correct_to_wrong": 1,
                "correct_to_correct": 1, "wrong_to_wrong": 2,
                "wrong_class_changed": 1, "initial_all_members_wrong": 4,
                "initial_all_members_wrong_pool_errors": 4,
                "common_wrong_pool_errors_repaired": 2, "common_wrong_member_created": 2}
    assert full["counts"] == expected
    assert full["net_correct_nodes"] == 1 and full["net_accuracy_change"] == 1 / 6
    assert full["initial_metrics"]["accuracy"] == 2 / 6
    assert full["endpoint_metrics"]["accuracy"] == 3 / 6
    expected_nll_change = -math.log(16) / 6
    assert abs(full["mean_endpoint_minus_initial"]["NLL"] - expected_nll_change) < 1e-14
    assert full["initial_pool_wrong_n"] == 4
    assert full["initial_pool_wrong_mean_endpoint_minus_initial"]["confidence"] == 0
    assert sum(row["n"] for row in result["classes"]) == 6
    assert sum(row["n"] for row in result["degree_bands"]) == 6
    assert result["baseline_all_members_wrong_pool_errors"]["n"] == 4
    disabled = False
    try:
        module.analyze(initial, endpoint, labels, ids, degrees)
    except RuntimeError:
        disabled = True
    assert disabled
    reordered = dict(endpoint, A_ids=list(reversed(ids)))
    rejected = False
    try:
        module._analyze(initial, reordered, labels, ids, degrees)
    except ValueError:
        rejected = True
    assert rejected
    saturated = {"A_ids": [0], "native_FP32_logits": [[[1000, 0, 0, 0, 0]] for _ in range(4)],
                 "served_FP32_member_probabilities": [[[1, 0, 0, 0, 0]] for _ in range(4)],
                 "served_FP32_pool_probabilities": [[1, 0, 0, 0, 0]]}
    stable = module._analyze(saturated, saturated, [1], [0], [0])
    assert stable["full_population"]["initial_metrics"]["NLL"] == 1000
    assert stable["classes"][0]["n"] == 0
    assert stable["classes"][0]["initial_metrics"]["NLL"] is None
    assert stable["full_population"]["net_correct_nodes"] == 0
    artifact = {"status": "PASS_SYNTHETIC_ACCOUNTING_ONLY", "scientific_evaluation": False,
                "source_released": module.SOURCE_RELEASED, "python": sys.version,
                "source_sha256": hashlib.sha256((HERE / "error_flow.py").read_bytes()).hexdigest(),
                "checks": {"exact_transition_counts": True, "net_accuracy_identity": True,
                    "baseline_common_error_cohort": True, "stable_logit_NLL": True,
                    "class_degree_partition": True, "baseline_error_confidence": True,
                    "misordered_ID_rejection": True, "disabled_public_entry": True,
                    "probability_underflow_and_empty_slice": True},
                "six_node_fixture": full}
    target = HERE / "SYNTHETIC_ACCOUNTING_RESULT.json"
    with target.open("x") as stream:
        json.dump(artifact, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": artifact["status"], "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
