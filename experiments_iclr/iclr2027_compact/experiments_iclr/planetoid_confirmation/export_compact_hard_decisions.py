"""Export audited selected/default test classes without float logits.

Postfreeze transport only. Requires complete validation and score/decision
audits. It never chooses a candidate or changes frozen training source.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import tuning as study


ROOT = Path(__file__).resolve().parent
DATASETS = ("citeseer", "pubmed")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
SEEDS = (0, 1, 2)
DEFAULT = (0.001, 0.0)
SOURCE_FILES = ("export_compact_hard_decisions.py",
                "HARD_DECISION_EXPORT_PROTOCOL.md")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def score_keys(lock: dict) -> set[str]:
    keys = set()
    for dataset in DATASETS:
        for arm in ARMS:
            chosen = tuple(lock["selections"][dataset][arm]["selected_candidate"])
            for lr, wd in dict.fromkeys((chosen, DEFAULT)):
                for seed in SEEDS:
                    keys.add(f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}")
    return keys


def full_label_and_indices(dataset: str, frozen: dict):
    _, labels, _, _, _, valid_mask, test_mask = study.load_planetoid(dataset)
    valid_indices = valid_mask.nonzero().flatten().long()
    indices = test_mask.nonzero().flatten().long()
    descriptor = frozen["graphs"][dataset]["descriptor"]
    require(study.tensor_sha(labels) == descriptor["tensor_sha256"]["labels"] and
            study.tensor_sha(valid_indices) ==
            descriptor["tensor_sha256"]["valid_indices"] and
            study.tensor_sha(indices) ==
            descriptor["tensor_sha256"]["test_indices"] and
            int(valid_indices.numel()) == descriptor["split_sizes"]["valid"] and
            int(indices.numel()) == descriptor["split_sizes"]["test"],
            f"Frozen public labels or official test index changed: {dataset}")
    return labels.numpy(), valid_indices.numpy(), indices.numpy()


def main() -> None:
    freeze_sha = study.check_freeze()
    frozen = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    lock_file = ROOT / "VALIDATION_SELECTION_LOCK.json"
    final_file = ROOT / "FINAL_SCORE_AUDIT.json"
    strict_file = ROOT / "STRICT_DECISION_AUDIT.json"
    for path in (lock_file, final_file, strict_file):
        require(path.is_file(), f"Required complete audit missing: {path.name}")
    lock = json.loads(lock_file.read_text())
    final = json.loads(final_file.read_text())
    strict = json.loads(strict_file.read_text())
    expected = score_keys(lock)
    require(freeze_sha == lock["freeze_sha256"] == final["freeze_sha256"] ==
            strict["freeze_sha256"] and
            len(lock["cells"]) == 216 and
            final["selection_lock_sha256"] ==
            strict["selection_lock_sha256"] == sha(lock_file) and
            strict["final_score_audit_sha256"] == sha(final_file) and
            strict["replayed_allowed_checkpoints"] == len(expected) and
            strict["total_valid_decision_mismatches"] == 0 and
            strict["total_test_decision_mismatches"] == 0 and
            strict["total_test_member_decision_mismatches"] == 0 and
            set(final["scores"]) == set(strict["per_checkpoint"]) == expected,
            "Incomplete selection, score, or strict decision audit")
    output = ROOT / "hard_decisions"
    manifest_file = ROOT / "HARD_DECISION_EXPORT_MANIFEST.json"
    require(not output.exists() and not manifest_file.exists(),
            "Refusing to overwrite prior hard-decision export")
    output.mkdir()
    graphs, cells = {}, {}
    for dataset in DATASETS:
        labels, valid_index, test_index = full_label_and_indices(dataset, frozen)
        require(labels.dtype == valid_index.dtype == test_index.dtype == np.int64 and
                int(labels.max()) < np.iinfo(np.int16).max and
                int(labels.min()) >= 0 and
                len(set(valid_index.tolist())) == len(valid_index) and
                len(set(test_index.tolist())) == len(test_index),
                f"Invalid fixed label/index dtype or range: {dataset}")
        reference = output / dataset / "test_reference.npz"
        reference.parent.mkdir()
        np.savez_compressed(reference, full_labels=labels,
                            valid_indices=valid_index,
                            test_indices=test_index)
        graphs[dataset] = {
            "reference_path": str(reference.relative_to(ROOT)),
            "reference_sha256": sha(reference),
            "full_label_tensor_sha256":
                frozen["graphs"][dataset]["descriptor"]["tensor_sha256"]["labels"],
            "valid_index_tensor_sha256":
                frozen["graphs"][dataset]["descriptor"]["tensor_sha256"]["valid_indices"],
            "test_index_tensor_sha256":
                frozen["graphs"][dataset]["descriptor"]["tensor_sha256"]["test_indices"],
            "valid_nodes": len(valid_index), "test_nodes": len(test_index),
            "classes": int(labels.max()) + 1,
        }
        truth = labels[test_index]
        for arm in ARMS:
            chosen = tuple(lock["selections"][dataset][arm]["selected_candidate"])
            for lr, wd in dict.fromkeys((chosen, DEFAULT)):
                for seed in SEEDS:
                    key = f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"
                    score_file = ROOT / "scores" / key / "score.json"
                    predictions_file = ROOT / "scores" / key / "predictions.npz"
                    scored = json.loads(score_file.read_text())
                    require(sha(score_file) == final["scores"][key]["score_sha256"] and
                            sha(predictions_file) ==
                            final["scores"][key]["predictions_sha256"] ==
                            scored["test_predictions_sha256"] and
                            scored["checkpoint_sha256"] ==
                            lock["cells"][key]["checkpoint_sha256"],
                            f"Audited score or source logits changed: {key}")
                    with np.load(predictions_file, allow_pickle=False) as raw:
                        pooled = raw["test_pooled_logits"]
                        members = raw["test_member_logits"]
                    expected_members = 1 if arm == "base" else 4
                    classes = graphs[dataset]["classes"]
                    require(pooled.shape == (len(truth), classes) and
                            members.shape ==
                            (expected_members, len(truth), classes) and
                            np.isfinite(pooled).all() and
                            np.isfinite(members).all(),
                            f"Unexpected source-logit shape/value: {key}")
                    pooled_class = pooled.argmax(-1).astype(np.int16)
                    member_class = members.argmax(-1).astype(np.int16)
                    accuracy = float(np.mean(pooled_class == truth))
                    require(abs(accuracy - scored["test_accuracy"]) <= 1e-7 and
                            abs(accuracy - final["scores"][key]["test_accuracy"]) <= 1e-7 and
                            abs(accuracy - strict["per_checkpoint"][key][
                                "test_accuracy"]) <= 1e-7,
                            f"Hard pooled classes do not reproduce audited accuracy: {key}")
                    hard_file = output / f"{key}.npz"
                    hard_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(hard_file,
                                        pooled_test_class=pooled_class,
                                        member_test_class=member_class)
                    cells[key] = {
                        "hard_path": str(hard_file.relative_to(ROOT)),
                        "hard_sha256": sha(hard_file),
                        "source_score_sha256": sha(score_file),
                        "source_float_predictions_sha256": sha(predictions_file),
                        "checkpoint_sha256": scored["checkpoint_sha256"],
                        "test_accuracy": accuracy,
                        "member_count": expected_members,
                        "mean_member_test_accuracy": float(np.mean(
                            member_class == truth[None, :])),
                    }
    require(set(cells) == expected,
            "Hard decisions do not cover the complete selected/default allowlist")
    manifest = {
        "protocol": "compact_selected_default_hard_decisions_v1",
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "frozen_study_sha256": freeze_sha,
        "selection_lock_sha256": sha(lock_file),
        "final_score_audit_sha256": sha(final_file),
        "strict_decision_audit_sha256": sha(strict_file),
        "graph_references": graphs,
        "cells": cells,
        "source_float_logits_included": False,
        "source_checkpoints_included": False,
    }
    study.write_json(manifest_file, manifest)
    print(json.dumps({"hard_cells": len(cells), "graphs": len(graphs),
                      "manifest_sha256": sha(manifest_file)}), flush=True)


if __name__ == "__main__":
    main()
