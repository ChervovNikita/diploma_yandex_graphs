"""Transport all six original Actor/filtered-Chameleon TIED default scores."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np


GRAPHS = ("actor", "chameleon_filtered")
ORIGINAL_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    x = logits.astype(np.float64)
    peak = x.max(axis=1)
    return float(np.mean(peak + np.log(np.exp(x - peak[:, None]).sum(axis=1))
                         - x[np.arange(len(labels)), labels]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), "Refusing overwrite"
    with tarfile.open(args.raw_archive, "r:gz") as archive:
        audit_raw = archive.extractfile("FINAL_SCORE_AUDIT.json").read()
        lock_raw = archive.extractfile("VALIDATION_SELECTION_LOCK.json").read()
        assert sha_bytes(audit_raw) == sha(args.primary / "FINAL_SCORE_AUDIT.json")
        assert sha_bytes(lock_raw) == sha(args.primary / "VALIDATION_SELECTION_LOCK.json") == ORIGINAL_LOCK_SHA
        audit = json.loads(audit_raw)
        hard = json.loads((args.primary / "HARD_DECISION_EXPORT_MANIFEST.json").read_text())
        args.out.mkdir(parents=True)
        rows = {}
        for graph in GRAPHS:
            reference = hard["graph_references"][graph]
            ref_path = args.primary / reference["reference_path"]
            assert sha(ref_path) == reference["reference_sha256"]
            with np.load(ref_path, allow_pickle=False) as ref:
                labels = ref["full_labels"][ref["test_indices"]]
            for seed in range(3):
                key = f"{graph}/tied/lr0.001_wd0/seed{seed}"
                original_raw = archive.extractfile(f"scores/{key}/predictions.npz").read()
                assert sha_bytes(original_raw) == audit["scores"][key]["predictions_sha256"]
                score_path = args.primary / "scores" / key / "score.json"
                assert sha(score_path) == audit["scores"][key]["score_sha256"]
                score = json.loads(score_path.read_text())
                assert score["predeclared_default"] is True
                with np.load(io.BytesIO(original_raw), allow_pickle=False) as full:
                    assert set(full.files) == {"valid_pooled_logits", "test_pooled_logits",
                                               "test_member_logits"}
                    pool = full["test_pooled_logits"]
                    member = full["test_member_logits"]
                assert pool.dtype == member.dtype == np.float32
                assert member.shape == (4, *pool.shape) and len(labels) == pool.shape[0]
                classes = member.argmax(axis=2).astype(np.uint8)
                hard_path = args.primary / "hard_decisions" / f"{key}.npz"
                with np.load(hard_path, allow_pickle=False) as original_classes:
                    assert np.array_equal(pool.argmax(axis=1), original_classes["pooled_test_class"])
                    assert np.array_equal(classes, original_classes["member_test_class"])
                accuracy = float(np.mean(pool.argmax(axis=1) == labels))
                cross_entropy = ce(pool, labels)
                assert abs(accuracy - score["test_accuracy"]) < 1e-7
                assert abs(cross_entropy - score["test_ce"]) < 1e-5
                out = args.out / f"{key}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, pooled_logits=pool, member_class=classes)
                rows[key] = {"source_npz_sha256": sha_bytes(original_raw),
                             "original_score_sha256": sha(score_path),
                             "projected_npz_sha256": sha(out),
                             "pooled_logits_bytes_sha256": sha_bytes(pool.tobytes(order="C")),
                             "member_class_bytes_sha256": sha_bytes(classes.tobytes(order="C")),
                             "test_accuracy": accuracy,
                             "test_ce": cross_entropy}
    assert len(rows) == 6
    manifest = {"protocol": "original_primary_tied_default_actor_chameleon_all6_v1",
                "scope": "All three seeds at the declared default configuration for Actor and filtered Chameleon; no test filtering",
                "original_raw_archive_sha256": sha(args.raw_archive),
                "original_primary_lock_sha256": ORIGINAL_LOCK_SHA,
                "original_final_score_audit_sha256": sha(args.primary / "FINAL_SCORE_AUDIT.json"),
                "source_builder_sha256": sha(Path(__file__)),
                "cells": rows, "cell_count": len(rows)}
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.out / "build_legacy_tied_default_projection.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": "ALL6_ORIGINAL_TIED_DEFAULT_PROJECTION_PASS",
                      "cells": 6, "manifest_sha256": sha(args.out / "MANIFEST.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
