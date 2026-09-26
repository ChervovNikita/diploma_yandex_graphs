"""CPU-only public check of all six original TIED default score projections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


GRAPHS = ("actor", "chameleon_filtered")
ORIGINAL_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    x = logits.astype(np.float64)
    high = x.max(axis=1)
    return float(np.mean(high + np.log(np.exp(x - high[:, None]).sum(axis=1))
                         - x[np.arange(len(labels)), labels]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--primary", type=Path)
    args = parser.parse_args()
    stage = args.stage
    primary = args.primary or stage.parent.parent / "validation_tuning"
    manifest = json.loads((stage / "MANIFEST.json").read_text())
    assert manifest["protocol"] == "original_primary_tied_default_actor_chameleon_all6_v1"
    assert sha(primary / "VALIDATION_SELECTION_LOCK.json") == manifest["original_primary_lock_sha256"] == ORIGINAL_LOCK_SHA
    assert sha(primary / "FINAL_SCORE_AUDIT.json") == manifest["original_final_score_audit_sha256"]
    assert sha(stage / "build_legacy_tied_default_projection.py") == manifest["source_builder_sha256"]
    audit = json.loads((primary / "FINAL_SCORE_AUDIT.json").read_text())
    hard = json.loads((primary / "HARD_DECISION_EXPORT_MANIFEST.json").read_text())
    expected = {f"{graph}/tied/lr0.001_wd0/seed{seed}" for graph in GRAPHS for seed in range(3)}
    assert set(manifest["cells"]) == expected and manifest["cell_count"] == 6
    for key in sorted(expected):
        graph = key.split("/")[0]
        row = manifest["cells"][key]
        ref_path = primary / hard["graph_references"][graph]["reference_path"]
        assert sha(ref_path) == hard["graph_references"][graph]["reference_sha256"]
        with np.load(ref_path, allow_pickle=False) as ref:
            labels = ref["full_labels"][ref["test_indices"]]
        score_path = primary / "scores" / key / "score.json"
        assert sha(score_path) == row["original_score_sha256"] == audit["scores"][key]["score_sha256"]
        score = json.loads(score_path.read_text())
        assert score["predeclared_default"] is True
        assert score["test_predictions_sha256"] == row["source_npz_sha256"] == audit["scores"][key]["predictions_sha256"]
        path = stage / f"{key}.npz"
        assert sha(path) == row["projected_npz_sha256"]
        with np.load(path, allow_pickle=False) as projection:
            assert set(projection.files) == {"pooled_logits", "member_class"}
            pool, classes = projection["pooled_logits"], projection["member_class"]
        assert pool.dtype == np.float32 and classes.dtype == np.uint8
        assert classes.shape == (4, len(labels)) and pool.shape[0] == len(labels)
        assert hashlib.sha256(pool.tobytes(order="C")).hexdigest() == row["pooled_logits_bytes_sha256"]
        assert hashlib.sha256(classes.tobytes(order="C")).hexdigest() == row["member_class_bytes_sha256"]
        with np.load(primary / "hard_decisions" / f"{key}.npz", allow_pickle=False) as saved:
            assert np.array_equal(pool.argmax(axis=1), saved["pooled_test_class"])
            assert np.array_equal(classes, saved["member_test_class"])
        accuracy = float(np.mean(pool.argmax(axis=1) == labels))
        cross_entropy = ce(pool, labels)
        assert abs(accuracy - row["test_accuracy"]) < 1e-12
        assert abs(accuracy - score["test_accuracy"]) < 1e-7
        assert abs(cross_entropy - row["test_ce"]) < 1e-12
        assert abs(cross_entropy - score["test_ce"]) < 1e-5
    print(json.dumps({"status": "ALL6_ORIGINAL_TIED_DEFAULT_PUBLIC_PASS", "cells": 6,
                      "manifest_sha256": sha(stage / "MANIFEST.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
