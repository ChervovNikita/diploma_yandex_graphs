"""Independent exact-array comparison of compact factor evidence to full source.

This author-side check needs original member-float archives and checkpoints.
It derives every expected key from fixed study matrices, not the compact
manifest, and compares all 108 validation and all allowed test arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


GRAPHS = {"all_layer": ("cora", "wikics", "actor", "chameleon_filtered"),
          "tied36": ("cora", "wikics")}
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01), (0.001, 0.0),
              (0.001, 0.01), (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)
FOLDERS = {"all_layer": "all_layer_factor_results",
           "tied36": "same_runtime_tied36_results"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(kind: str, graph: str, lr: float, wd: float, seed: int) -> str:
    prefix = f"{graph}/tied" if kind == "tied36" else graph
    return f"{prefix}/lr{lr:g}_wd{wd:g}/seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists(), "Refusing overwrite"
    rows = {"validation": {}, "test": {}}
    for kind in ("all_layer", "tied36"):
        root = args.full / FOLDERS[kind]
        lock = json.loads((root / "VALIDATION_SELECTION_LOCK.json").read_text())
        audit = json.loads((root / "FINAL_SCORE_AUDIT.json").read_text())
        for graph in GRAPHS[kind]:
            for lr, wd in CANDIDATES:
                for seed in range(3):
                    identity = key(kind, graph, lr, wd, seed)
                    folder = root / "results" / identity
                    original = folder / ("validation_predictions.npz" if kind == "all_layer"
                                         else "validation_companion.npz")
                    projected = args.compact / "validation_decisions" / kind / f"{identity}.npz"
                    frozen = lock["cells"][identity]
                    expected_sha = (frozen["validation_predictions_sha256"] if kind == "all_layer"
                                    else frozen["validation_companion_predictions_sha256"])
                    assert sha(original) == expected_sha
                    with np.load(original, allow_pickle=False) as full, np.load(projected, allow_pickle=False) as small:
                        if kind == "all_layer":
                            pool, member = full["pooled_logits"], full["member_logits"]
                        else:
                            pool, member = full["valid_pooled_logits"], full["valid_member_logits"]
                        assert small["pooled_class"].tobytes() == pool.argmax(axis=1).astype(np.uint8).tobytes()
                        assert small["member_class"].tobytes() == member.argmax(axis=2).astype(np.uint8).tobytes()
                        ref = args.compact / "validation_references" / f"{graph}.npz"
                        with np.load(ref, allow_pickle=False) as shared:
                            assert np.array_equal(shared["valid_indices"], full["valid_indices"])
                            assert np.array_equal(shared["valid_labels"], full["valid_labels"])
                        assert pool.dtype == member.dtype == np.float32
                        assert member.shape == (4, *pool.shape)
                    rows["validation"][f"{kind}/{identity}"] = {
                        "original_npz_sha256": sha(original),
                        "compact_npz_sha256": sha(projected),
                        "all_pooled_and_member_classes_exact": True,
                        "shared_validation_reference_exact": True,
                    }
            selected = tuple(lock["selections"][graph]["selected_candidate"])
            for lr, wd in dict.fromkeys((selected, DEFAULT)):
                for seed in range(3):
                    identity = key(kind, graph, lr, wd, seed)
                    original = root / "scores" / identity / "predictions.npz"
                    projected = args.compact / "test_predictions" / kind / f"{identity}.npz"
                    assert sha(original) == audit["scores"][identity]["predictions_sha256"]
                    with np.load(original, allow_pickle=False) as full, np.load(projected, allow_pickle=False) as small:
                        pool, member = full["test_pooled_logits"], full["test_member_logits"]
                        assert pool.dtype == member.dtype == np.float32
                        assert small["pooled_logits"].dtype == np.float32
                        assert small["pooled_logits"].tobytes(order="C") == pool.tobytes(order="C")
                        assert small["member_class"].tobytes(order="C") == \
                               member.argmax(axis=2).astype(np.uint8).tobytes(order="C")
                    rows["test"][f"{kind}/{identity}"] = {
                        "original_npz_sha256": sha(original),
                        "compact_npz_sha256": sha(projected),
                        "pooled_float32_bitwise_exact": True,
                        "all_member_classes_exact": True,
                    }
    assert len(rows["validation"]) == 108 and len(rows["test"]) == 36
    report = {"status": "INDEPENDENT_FULL_TO_COMPACT_ALL108_AND36_EXACT_ARRAY_PASS",
              "validation_cells": rows["validation"], "test_scores": rows["test"],
              "validation_count": len(rows["validation"]),
              "test_count": len(rows["test"]),
              "compact_manifest_sha256": sha(args.compact / "MANIFEST.json"),
              "audit_source_sha256": sha(Path(__file__))}
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "validation_count": 108,
                      "test_count": 36, "report_sha256": sha(args.report)}, sort_keys=True))


if __name__ == "__main__":
    main()
