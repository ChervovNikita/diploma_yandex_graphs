"""Recalculate descriptive member/pooled decisions from compact stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
STAGES = {
    "cora": ROOT / "new_graph_studies",
    "chameleon_legacy": ROOT / "new_graph_studies",
    "chameleon_filtered": ROOT / "filtered_chameleon_study",
    "wikics": ROOT / "external_depth_sage",
    "actor": ROOT / "external_depth_sage",
}
DIRECTORY = {"cora": "cora", "chameleon_legacy": "chameleon",
             "chameleon_filtered": "chameleon_filtered", "wikics": "wikics",
             "actor": "actor"}
ARMS = ("tied", "untied_propagation")
METRICS = ("pooled_accuracy", "mean_member_accuracy", "pooling_gain",
           "pairwise_member_disagreement", "any_member_correct_coverage")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(path):
    with np.load(path, allow_pickle=False) as data:
        member_key = ("test_member_pred" if "test_member_pred" in data.files
                      else "test_member_predictions")
        pooled_key = ("test_pooled_pred" if "test_pooled_pred" in data.files
                      else "test_pooled_predictions")
        member = data[member_key].copy()
        pooled = data[pooled_key].copy()
        labels = data["test_labels"].copy()
        indices = data["test_indices"].copy()
    assert member.shape == (4, len(labels)) and pooled.shape == labels.shape
    assert indices.shape == labels.shape and len(set(indices.tolist())) == len(indices)
    member_accuracy = float(np.mean(member == labels[None, :]))
    pooled_accuracy = float(np.mean(pooled == labels))
    disagreement = float(np.mean([
        np.mean(member[i] != member[j]) for i in range(4) for j in range(i + 1, 4)]))
    coverage = float(np.mean((member == labels[None, :]).any(axis=0)))
    return {"pooled_accuracy": pooled_accuracy,
            "mean_member_accuracy": member_accuracy,
            "pooling_gain": pooled_accuracy - member_accuracy,
            "pairwise_member_disagreement": disagreement,
            "any_member_correct_coverage": coverage,
            "test_node_ids_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "test_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest()}


def main():
    result = {"status": "VERIFIED_COMPACT_DECISION_DECOMPOSITION",
              "definitions": {
                  "mean_member_accuracy": "mean of four member test accuracies",
                  "pooling_gain": "argmax-of-mean-raw-logit test accuracy minus mean member test accuracy",
                  "pairwise_member_disagreement": "mean over six member pairs of fraction of test nodes with different predicted classes",
                  "any_member_correct_coverage": "fraction of test nodes correctly predicted by at least one member",
              },
              "scope": "one fixed published split per graph; three optimizer seeds; descriptive held-out decisions",
              "graphs": {}}
    for graph, stage in STAGES.items():
        manifest = (stage / "evidence_manifest.json" if
                    (stage / "evidence_manifest.json").is_file() else
                    stage / "PRETRAIN_FREEZE.json")
        assert manifest.is_file()
        item = {"stage_anchor_file": manifest.name,
                "stage_anchor_sha256": sha(manifest), "depths": {}}
        for depth in (2, 5):
            groups = {}
            for arm in ARMS:
                runs = []
                for seed in range(3):
                    path = (stage / "results" / DIRECTORY[graph] / f"depth{depth}" /
                            f"seed{seed}" / arm / "selected_decisions.npz")
                    row = one(path)
                    source = json.loads((path.parent / "result.json").read_text())
                    assert abs(row["pooled_accuracy"] - source["test_accuracy"]) < 1e-6
                    row["seed"] = seed
                    row["decision_file_sha256"] = sha(path)
                    runs.append(row)
                assert len({r["test_node_ids_sha256"] for r in runs}) == 1
                assert len({r["test_labels_sha256"] for r in runs}) == 1
                groups[arm] = {"mean": {m: sum(r[m] for r in runs) / 3 for m in METRICS},
                               "runs": runs}
            item["depths"][str(depth)] = groups
        result["graphs"][graph] = item
    (ROOT / "DECISION_MECHANISM_CROSSGRAPH.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "graphs": list(result["graphs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
