"""Verify all 72 validation-selected HPO test-logit companions.

Run from anywhere with NumPy installed. The sibling validation_tuning/ stage
must be present. This checks test logits, decisions, published score records,
and the cryptographic link to every original prediction file. Training replay
still requires the author-retained checkpoints and public graph downloads.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
STUDY = Path(os.environ.get("GNNM_VALIDATION_TUNING_DIR", str(ROOT.parent / "validation_tuning")))
EXPECTED_LOCK = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
EXPECTED_AUDIT = "63da49d3d797fa7856717d4b21a530c74419bfbdfa92614a03b036aa9950bdde"
GRAPHS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def ce_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    # Float64 reduction avoids numerical overflow and is compared with the
    # original float32 Torch score under a stated 1e-5 tolerance.
    x = logits.astype(np.float64)
    high = x.max(axis=1)
    logsumexp = high + np.log(np.exp(x - high[:, None]).sum(axis=1))
    return float(np.mean(logsumexp - x[np.arange(len(labels)), labels]))


def key_for(graph: str, arm: str, lr: float, wd: float, seed: int) -> str:
    return f"{graph}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"


def main() -> None:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    need(manifest["protocol"] == "all_selected_hpo72_test_member_logits_lossless_v1", "Protocol")
    need(manifest["cell_count"] == len(manifest["cells"]) == 72, "Cell count")
    need(sha_file(STUDY / "VALIDATION_SELECTION_LOCK.json") ==
         manifest["selection_lock_sha256"] == EXPECTED_LOCK, "Selection lock hash")
    need(sha_file(STUDY / "FINAL_SCORE_AUDIT.json") ==
         manifest["final_score_audit_sha256"] == EXPECTED_AUDIT, "Final score audit hash")
    lock = json.loads((STUDY / "VALIDATION_SELECTION_LOCK.json").read_text())
    audit = json.loads((STUDY / "FINAL_SCORE_AUDIT.json").read_text())
    hard = json.loads((STUDY / "HARD_DECISION_EXPORT_MANIFEST.json").read_text())
    need(len(lock["cells"]) == 432 and len(audit["scores"]) == 141, "Original audit cardinality")
    need(lock["selection_uses_test_labels"] is False and
         hard["selection_lock_sha256"] == EXPECTED_LOCK and
         hard["final_score_audit_sha256"] == EXPECTED_AUDIT,
         "Selection/hard-decision provenance")
    labels = {}
    for graph in GRAPHS:
        reference = hard["graph_references"][graph]
        ref_path = STUDY / reference["reference_path"]
        need(sha_file(ref_path) == reference["reference_sha256"], f"Test reference {graph}")
        with np.load(ref_path, allow_pickle=False) as ref:
            labels[graph] = ref["full_labels"][ref["test_indices"]]
        need(len(labels[graph]) == reference["test_nodes"], f"Test count {graph}")
    expected = set()
    for graph in GRAPHS:
        for arm in ARMS:
            lr, wd = lock["selections"][graph][arm]["selected_candidate"]
            for seed in range(3):
                key = key_for(graph, arm, lr, wd, seed)
                expected.add(key)
                row = manifest["cells"][key]
                source = audit["scores"][key]
                hard_row = hard["cells"][key]
                path = ROOT / "logits" / f"{key}.npz"
                need(sha_file(path) == row["derivative_npz_sha256"] and
                     path.stat().st_size == row["derivative_npz_bytes"],
                     f"Derivative file {key}")
                need(row["source_predictions_sha256"] ==
                     source["predictions_sha256"] ==
                     hard_row["source_float_predictions_sha256"] and
                     row["source_score_sha256"] ==
                     source["score_sha256"] ==
                     hard_row["source_score_sha256"],
                     f"Original file provenance {key}")
                score_path = STUDY / "scores" / key / "score.json"
                need(sha_file(score_path) == row["source_score_sha256"], f"Score file {key}")
                score = json.loads(score_path.read_text())
                need(score["selected_candidate"] is True and
                     score["validation_selection_lock_sha256"] == EXPECTED_LOCK and
                     score["test_predictions_sha256"] == row["source_predictions_sha256"],
                     f"Score selection {key}")
                with np.load(path, allow_pickle=False) as prediction:
                    need(set(prediction.files) == {"test_member_logits"}, f"Companion schema {key}")
                    member = prediction["test_member_logits"]
                need(member.dtype == np.float32 and list(member.shape) == row["shape"] and
                     member.shape[1] == len(labels[graph]) and np.isfinite(member).all(),
                     f"Companion values {key}")
                need(sha_bytes(member.tobytes(order="C")) ==
                     row["test_member_logits_bytes_sha256"], f"Original member array {key}")
                pooled = member.mean(axis=0)
                need(pooled.dtype == np.float32 and
                     sha_bytes(pooled.tobytes(order="C")) ==
                     row["test_pooled_logits_bytes_sha256"],
                     f"Bitwise original pooled logits {key}")
                hard_path = STUDY / hard_row["hard_path"]
                need(sha_file(hard_path) == hard_row["hard_sha256"], f"Hard decision hash {key}")
                with np.load(hard_path, allow_pickle=False) as decisions:
                    need(np.array_equal(pooled.argmax(axis=1), decisions["pooled_test_class"]) and
                         np.array_equal(member.argmax(axis=2), decisions["member_test_class"]),
                         f"Hard decisions {key}")
                accuracy = float(np.mean(pooled.argmax(axis=1) == labels[graph]))
                ce = ce_from_logits(pooled, labels[graph])
                need(abs(accuracy - row["test_accuracy"]) < 1e-12 and
                     abs(accuracy - score["test_accuracy"]) < 1e-7 and
                     abs(accuracy - source["test_accuracy"]) < 1e-7 and
                     abs(ce - score["test_ce"]) < 1e-5 and
                     abs(ce - source["test_ce"]) < 1e-5,
                     f"Test metrics {key}")
    need(set(manifest["cells"]) == expected, "Incomplete or extra selected cells")
    need(sum(row["derivative_npz_bytes"] for row in manifest["cells"].values()) ==
         manifest["derivative_npz_total_bytes"] == 17_301_843,
         "Payload size/count")
    print(json.dumps({"status": "ALL_72_SELECTED_TEST_LOGITS_PASS", "cells": len(expected),
                      "derivative_npz_total_bytes": manifest["derivative_npz_total_bytes"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
