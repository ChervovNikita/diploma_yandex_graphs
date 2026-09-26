"""Standalone NumPy replay of compact WikiCS matched54 public records."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ARMS = ("base", "ens", "private_last")
SEEDS = (0, 1, 2)
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01), (0.001, 0.0),
              (0.001, 0.01), (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key(arm, lr, wd, seed):
    return f"wikics/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"


def ce(logits, labels):
    values = logits.astype(np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_z - shifted[np.arange(len(labels)), labels]))


def main(root):
    manifest = json.loads((root / "COMPACT_MANIFEST.json").read_text())
    for relative, digest in manifest["files"].items():
        require(sha(root / relative) == digest, f"Compact file differs: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    require(actual == set(manifest["files"]) | {"COMPACT_MANIFEST.json"},
            "Compact file set differs")
    freeze = json.loads((root / "FROZEN_WIKICS_MATCHED54.json").read_text())
    lock = json.loads((root / "VALIDATION_SELECTION_LOCK.json").read_text())
    audit = json.loads((root / "FINAL_SCORE_AUDIT.json").read_text())
    require(freeze["protocol"] == lock["protocol"] == audit["protocol"] == manifest["protocol"] and
            sha(root / "FROZEN_WIKICS_MATCHED54.json") == manifest["freeze_sha256"] and
            sha(root / "VALIDATION_SELECTION_LOCK.json") == manifest["validation_lock_sha256"] and
            sha(root / "FINAL_SCORE_AUDIT.json") == manifest["score_audit_sha256"] and
            lock["freeze_sha256"] == manifest["freeze_sha256"] and
            audit["freeze_sha256"] == manifest["freeze_sha256"] and
            audit["validation_selection_lock_sha256"] == manifest["validation_lock_sha256"],
            "Freeze, selection, and score audit linkage differs")
    matrix = freeze["matrix"]
    require(matrix["dataset"] == "wikics" and tuple(matrix["arms"]) == ARMS and
            tuple(matrix["seeds"]) == SEEDS and matrix["epochs"] == 1000 and
            tuple((r["lr"], r["weight_decay"]) for r in matrix["candidates"]) == CANDIDATES and
            tuple(matrix["default_candidate"]) == DEFAULT and len(lock["cells"]) == 54,
            "Frozen 54-cell matrix differs")
    with np.load(root / "references/validation.npz", allow_pickle=False) as data:
        valid_y = data["labels"]
    with np.load(root / "references/test.npz", allow_pickle=False) as data:
        test_y = data["labels"]
    require(valid_y.ndim == test_y.ndim == 1 and len(valid_y) and len(test_y),
            "Reference label arrays invalid")
    for arm in ARMS:
        candidates = []
        for lr, wd in CANDIDATES:
            rows = []
            for seed in SEEDS:
                identity = key(arm, lr, wd, seed)
                saved = lock["cells"][identity]
                result_path = root / "results" / identity / "result.json"
                trace_path = root / "results" / identity / "validation_trace.csv"
                require(sha(result_path) == saved["result_sha256"] and
                        sha(trace_path) == saved["trace_sha256"],
                        f"Cell hashes differ: {identity}")
                result = json.loads(result_path.read_text())
                require(result["freeze_sha256"] == manifest["freeze_sha256"] and
                        result["dataset"] == "wikics" and result["arm"] == arm and
                        result["lr"] == lr and result["weight_decay"] == wd and
                        result["seed"] == seed and result["epochs_required"] == 1000 and
                        result["initialization"]["parameter_count"] == matrix["parameter_counts"][arm],
                        f"Cell identity differs: {identity}")
                with trace_path.open(newline="") as stream:
                    trace = list(csv.DictReader(stream))
                require(len(trace) == result["epochs_completed"] == 1000 and
                        result["failure"] is None, f"Incomplete cell: {identity}")
                best = (-1.0, float("inf"), 0)
                for epoch, row in enumerate(trace, 1):
                    va, vc = float(row["valid_accuracy"]), float(row["valid_ce"])
                    require(int(row["epoch"]) == epoch and math.isfinite(va) and
                            math.isfinite(vc) and 0 <= va <= 1 and vc >= 0,
                            f"Invalid validation trace: {identity}")
                    better = va > best[0] or (va == best[0] and vc < best[1])
                    require(int(row["selected_now"]) == int(better),
                            f"Selection flag differs: {identity}")
                    if better:
                        best = (va, vc, epoch)
                require((result["selected_valid_accuracy"], result["selected_valid_ce"],
                         result["selected_epoch"]) == best and
                        (saved["selected_valid_accuracy"], saved["selected_valid_ce"],
                         saved["selected_epoch"]) == best and
                        result["checkpoint_sha256"] == saved["checkpoint_sha256"],
                        f"Selected checkpoint record differs: {identity}")
                with np.load(root / "hard_decisions" / "validation" / (identity + ".npz"),
                             allow_pickle=False) as classes:
                    require(classes["pooled"].shape == valid_y.shape and
                            classes["members"].shape[1:] == valid_y.shape and
                            abs(float(np.mean(classes["pooled"] == valid_y)) - best[0]) <= 1e-7,
                            f"Validation hard decisions differ: {identity}")
                rows.append(best)
            candidates.append((lr, wd, float(np.mean([r[0] for r in rows])),
                               float(np.mean([r[1] for r in rows]))))
        chosen = max(candidates, key=lambda r: (r[2], -r[3], -r[0], -r[1]))
        recorded = lock["selections"][arm]
        require(tuple(recorded["selected_candidate"]) == chosen[:2] and
                abs(recorded["selected_mean_valid_accuracy"] - chosen[2]) <= 1e-15 and
                abs(recorded["selected_mean_valid_ce"] - chosen[3]) <= 1e-15,
                f"Validation candidate selection differs: {arm}")
    expected = set()
    for arm in ARMS:
        chosen = tuple(lock["selections"][arm]["selected_candidate"])
        for lr, wd in dict.fromkeys((chosen, DEFAULT)):
            for seed in SEEDS:
                identity = key(arm, lr, wd, seed)
                expected.add(identity)
                score_path = root / "scores" / identity / "score.json"
                score = json.loads(score_path.read_text())
                require(sha(score_path) == audit["scores"][identity]["score_sha256"] and
                        score["validation_selection_lock_sha256"] == manifest["validation_lock_sha256"] and
                        score["checkpoint_sha256"] == lock["cells"][identity]["checkpoint_sha256"] and
                        score["selected_candidate"] == ((lr, wd) == chosen) and
                        score["predeclared_default"] == ((lr, wd) == DEFAULT),
                        f"Allowed score record differs: {identity}")
                with np.load(root / "hard_decisions" / "test" / (identity + ".npz"),
                             allow_pickle=False) as classes:
                    pooled_classes = classes["pooled"].copy()
                    require(classes["pooled"].shape == test_y.shape and
                            classes["members"].shape[1:] == test_y.shape and
                            abs(float(np.mean(classes["pooled"] == test_y)) - score["test_accuracy"]) <= 1e-7,
                            f"Test hard decisions differ: {identity}")
                with np.load(root / "pooled_float_companion" / (identity + ".npz"),
                             allow_pickle=False) as data:
                    logits = data["logits"]
                    require(logits.dtype == np.float32 and logits.shape[0] == len(test_y) and
                            np.array_equal(logits.argmax(-1), pooled_classes) and
                            abs(ce(logits, test_y) - score["test_ce"]) <= 1e-5,
                            f"Exact float32 pooled replay differs: {identity}")
    require(set(audit["scores"]) == expected and 6 <= len(expected) <= 18,
            "Test score allowlist differs")
    print(json.dumps({"status": "PASS", "cells": 54, "audited_test_scores": len(expected),
                      "manifest_sha256": sha(root / "COMPACT_MANIFEST.json")}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve())
