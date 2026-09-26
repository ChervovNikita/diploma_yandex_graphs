"""Public compact audit of the 216-cell selection and hard class scores.

Needs validation traces, result JSON, locks/audits, score JSON, and exported
hard classes. It uses no checkpoint, model, raw float logit, or GPU.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = ("citeseer", "pubmed")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01),
              (0.001, 0.0), (0.001, 0.01),
              (0.003, 0.0), (0.003, 0.01))
SEEDS = (0, 1, 2)
DEFAULT = (0.001, 0.0)
EPOCHS = 1000
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read(name: str) -> dict:
    path = ROOT / name
    require(path.is_file(), f"Missing compact audit artifact: {name}")
    return json.loads(path.read_text())


def cell_key(dataset: str, arm: str, lr: float, wd: float, seed: int) -> str:
    return f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"


def audit_trace(key: str, frozen_cell: dict, freeze_sha: str,
                dataset: str, arm: str, lr: float, wd: float, seed: int) -> None:
    cell = ROOT / "results" / key
    result_path, trace_path = cell / "result.json", cell / "validation_trace.csv"
    require(result_path.is_file() and trace_path.is_file() and
            sha(result_path) == frozen_cell["result_sha256"] and
            sha(trace_path) == frozen_cell["trace_sha256"],
            f"Locked result or trace missing/changed: {key}")
    result = json.loads(result_path.read_text())
    require(result["protocol"] == "validation_tuning_sensitivity_v1" and
            result["freeze_sha256"] == freeze_sha and
            result["dataset"] == dataset and result["arm"] == arm and
            result["lr"] == lr and result["weight_decay"] == wd and
            result["seed"] == seed and result["epochs_required"] == EPOCHS and
            result["failure"] == frozen_cell["failure"] and
            result["validation_trace_sha256"] == sha(trace_path),
            f"Cell identity or trace digest differs: {key}")
    with trace_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == TRACE_FIELDS,
                f"Validation trace schema differs: {key}")
        rows = list(reader)
    require(len(rows) == result["epochs_completed"] and len(rows) <= EPOCHS,
            f"Validation trace length differs: {key}")
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    for epoch, row in enumerate(rows, 1):
        acc, ce = float(row["valid_accuracy"]), float(row["valid_ce"])
        require(int(row["epoch"]) == epoch and math.isfinite(acc) and
                math.isfinite(ce) and 0 <= acc <= 1 and ce >= 0,
                f"Invalid validation row: {key}, epoch {epoch}")
        better = acc > best_acc or (acc == best_acc and ce < best_ce)
        require(int(row["selected_now"]) == int(better),
                f"Checkpoint flag differs: {key}, epoch {epoch}")
        if better:
            best_acc, best_ce, best_epoch = acc, ce, epoch
    if result["failure"] is not None:
        require(len(rows) < EPOCHS and "checkpoint_sha256" not in frozen_cell,
                f"Failed candidate has a complete trace/checkpoint lock: {key}")
    else:
        require(len(rows) == EPOCHS and
                result["selected_epoch"] == frozen_cell["selected_epoch"] ==
                best_epoch and
                result["selected_valid_accuracy"] ==
                frozen_cell["selected_valid_accuracy"] == best_acc and
                result["selected_valid_ce"] ==
                frozen_cell["selected_valid_ce"] == best_ce and
                result["checkpoint_sha256"] ==
                frozen_cell["checkpoint_sha256"],
                f"Selected validation checkpoint/metric differs: {key}")


def audit_selection(lock: dict, freeze_sha: str) -> set[str]:
    frozen = read("FROZEN_STUDY.json")
    require(frozen["protocol"] == lock["protocol"] ==
            "planetoid_two_graph_confirmation_v1" and
            sha(ROOT / "FROZEN_STUDY.json") == freeze_sha and
            tuple(frozen["matrix"]["datasets"]) == DATASETS and
            tuple(frozen["matrix"]["arms"]) == ARMS and
            tuple(frozen["matrix"]["seeds"]) == SEEDS and
            tuple((x["lr"], x["weight_decay"]) for x in
                  frozen["matrix"]["candidates"]) == CANDIDATES and
            frozen["matrix"]["epochs"] == EPOCHS and
            len(lock["cells"]) == 216,
            "Frozen matrix or full selection lock differs")
    for name, digest in frozen["source_sha256"].items():
        require(sha(ROOT / name) == digest, f"Frozen source differs: {name}")
    seen = set()
    for dataset in DATASETS:
        for arm in ARMS:
            calculated = []
            for lr, wd in CANDIDATES:
                seed_rows = []
                for seed in SEEDS:
                    key = cell_key(dataset, arm, lr, wd, seed)
                    require(key in lock["cells"], f"Locked cell missing: {key}")
                    seen.add(key)
                    frozen_cell = lock["cells"][key]
                    audit_trace(key, frozen_cell, freeze_sha,
                                dataset, arm, lr, wd, seed)
                    seed_rows.append(frozen_cell)
                valid = all(row["failure"] is None for row in seed_rows)
                acc = (sum(row["selected_valid_accuracy"]
                           for row in seed_rows) / 3 if valid else None)
                ce = (sum(row["selected_valid_ce"]
                          for row in seed_rows) / 3 if valid else None)
                calculated.append((lr, wd, valid, acc, ce))
            recorded = lock["selections"][dataset][arm]
            require(len(recorded["candidate_table"]) == 6,
                    f"Candidate table incomplete: {dataset}/{arm}")
            for (lr, wd, valid, acc, ce), old in zip(
                    calculated, recorded["candidate_table"]):
                require((old["lr"], old["weight_decay"], old["valid"],
                         old["mean_valid_accuracy"], old["mean_valid_ce"]) ==
                        (lr, wd, valid, acc, ce),
                        f"Candidate validation mean differs: {dataset}/{arm}")
            valid_rows = [row for row in calculated if row[2]]
            require(valid_rows, f"No finite candidate: {dataset}/{arm}")
            chosen = max(valid_rows, key=lambda row:
                         (row[3], -row[4], -row[0], -row[1]))
            require(recorded["selected_candidate"] == [chosen[0], chosen[1]] and
                    recorded["selected_mean_valid_accuracy"] == chosen[3] and
                    recorded["selected_mean_valid_ce"] == chosen[4],
                    f"Frozen candidate tie break differs: {dataset}/{arm}")
        first = lock["selections"][dataset]["private_first"]
        last = lock["selections"][dataset]["private_last"]
        if first["selected_mean_valid_accuracy"] > last[
                "selected_mean_valid_accuracy"]:
            chosen_arm = "private_first"
        elif first["selected_mean_valid_accuracy"] < last[
                "selected_mean_valid_accuracy"]:
            chosen_arm = "private_last"
        elif first["selected_mean_valid_ce"] < last["selected_mean_valid_ce"]:
            chosen_arm = "private_first"
        else:
            chosen_arm = "private_last"
        require(lock["partial_family_choice"][dataset]["selected_arm"] ==
                chosen_arm and
                lock["partial_family_choice"][dataset][
                    "family_configurations_searched"] == 12,
                f"Partial-family choice differs: {dataset}")
    actual = {str(path.parent.relative_to(ROOT / "results")) for path in
              (ROOT / "results").rglob("result.json")}
    require(seen == set(lock["cells"]) == actual and len(seen) == 216,
            "Unexpected or missing validation result cells")
    return seen


def audit_hard_scores(lock: dict, freeze_sha: str) -> int:
    final = read("FINAL_SCORE_AUDIT.json")
    strict = read("STRICT_DECISION_AUDIT.json")
    hard = read("HARD_DECISION_EXPORT_MANIFEST.json")
    lock_sha = sha(ROOT / "VALIDATION_SELECTION_LOCK.json")
    final_sha = sha(ROOT / "FINAL_SCORE_AUDIT.json")
    strict_sha = sha(ROOT / "STRICT_DECISION_AUDIT.json")
    require(final["protocol"] == "planetoid_two_graph_confirmation_v1" and
            final["freeze_sha256"] == freeze_sha and
            final["selection_lock_sha256"] == lock_sha and
            strict["freeze_sha256"] == freeze_sha and
            strict["selection_lock_sha256"] == lock_sha and
            strict["final_score_audit_sha256"] == final_sha and
            strict["total_valid_decision_mismatches"] == 0 and
            strict["total_test_decision_mismatches"] == 0 and
            strict["total_test_member_decision_mismatches"] == 0 and
            hard["protocol"] == "compact_selected_default_hard_decisions_v1" and
            hard["frozen_study_sha256"] == freeze_sha and
            hard["selection_lock_sha256"] == lock_sha and
            hard["final_score_audit_sha256"] == final_sha and
            hard["strict_decision_audit_sha256"] == strict_sha and
            not hard["source_float_logits_included"] and
            not hard["source_checkpoints_included"],
            "Score, decision, or hard export audit identity differs")
    require(strict["script_sha256"] ==
            sha(ROOT / "strict_score_decision_audit.py") and
            all(row["valid_decision_mismatches"] == 0 and
                row["test_decision_mismatches"] == 0 and
                row["test_member_decision_mismatches"] == 0
                for row in strict["per_checkpoint"].values()),
            "Strict per-checkpoint decision replay differs")
    for name, digest in hard["source_sha256"].items():
        require(sha(ROOT / name) == digest,
                f"Hard-decision exporter or protocol differs: {name}")
    references = {}
    frozen = read("FROZEN_STUDY.json")
    require(set(hard["graph_references"]) == set(DATASETS),
            "Hard-decision graph set differs")
    for dataset in DATASETS:
        record = hard["graph_references"][dataset]
        path = ROOT / record["reference_path"]
        require(sha(path) == record["reference_sha256"],
                f"Hard-decision test reference changed: {dataset}")
        with np.load(path, allow_pickle=False) as arrays:
            labels = arrays["full_labels"]
            valid_indices = arrays["valid_indices"]
            indices = arrays["test_indices"]
        descriptor = frozen["graphs"][dataset]["descriptor"]
        require(labels.dtype == valid_indices.dtype == indices.dtype == np.int64 and
                tensor_sha(labels) == record["full_label_tensor_sha256"] ==
                descriptor["tensor_sha256"]["labels"] and
                tensor_sha(valid_indices) == record["valid_index_tensor_sha256"] ==
                descriptor["tensor_sha256"]["valid_indices"] and
                tensor_sha(indices) == record["test_index_tensor_sha256"] ==
                descriptor["tensor_sha256"]["test_indices"] and
                len(valid_indices) == record["valid_nodes"] ==
                descriptor["split_sizes"]["valid"] and
                len(indices) == record["test_nodes"] ==
                descriptor["split_sizes"]["test"] and
                np.unique(valid_indices).size == len(valid_indices) and
                np.unique(indices).size == len(indices) and
                np.all((valid_indices >= 0) & (valid_indices < len(labels))) and
                np.all((indices >= 0) & (indices < len(labels))) and
                int(labels.max()) + 1 == record["classes"],
                f"Official test reference/fingerprint differs: {dataset}")
        references[dataset] = (labels, valid_indices, indices)

    for key, cell in lock["cells"].items():
        if cell["failure"] is not None:
            continue
        dataset, arm = key.split("/", 2)[:2]
        labels, valid_indices, _ = references[dataset]
        path = ROOT / "results" / key / "selected_validation_decisions.npz"
        require(sha(path) == cell["selected_validation_decisions_sha256"],
                f"Selected validation decisions changed: {key}")
        with np.load(path, allow_pickle=False) as arrays:
            pooled = arrays["pooled"]
            members = arrays["members"]
        member_count = 1 if arm == "base" else 4
        classes = int(labels.max()) + 1
        require(pooled.shape == (len(valid_indices),) and
                members.shape == (member_count, len(valid_indices)) and
                np.all(pooled < classes) and np.all(members < classes),
                f"Selected validation decision shape or class differs: {key}")
        accuracy = float(np.mean(pooled == labels[valid_indices]))
        require(abs(accuracy - cell["fresh_validation_replay_accuracy"]) <= 1e-7 and
                abs(accuracy - cell["selected_valid_accuracy"]) <= 1e-7 and
                abs(cell["fresh_validation_replay_ce"] - cell["selected_valid_ce"]) <= 1e-5,
                f"Selected validation replay accuracy differs: {key}")

    expected = set()
    for dataset in DATASETS:
        labels, _, indices = references[dataset]
        truth = labels[indices]
        for arm in ARMS:
            chosen = tuple(lock["selections"][dataset][arm]["selected_candidate"])
            for lr, wd in dict.fromkeys((chosen, DEFAULT)):
                for seed in SEEDS:
                    key = cell_key(dataset, arm, lr, wd, seed)
                    expected.add(key)
                    require(key in final["scores"] and
                            key in strict["per_checkpoint"] and
                            key in hard["cells"],
                            f"Audited selected/default score missing: {key}")
                    score_path = ROOT / "scores" / key / "score.json"
                    require(sha(score_path) ==
                            final["scores"][key]["score_sha256"],
                            f"Compact score JSON changed: {key}")
                    score = json.loads(score_path.read_text())
                    source = hard["cells"][key]
                    require(score["freeze_sha256"] == freeze_sha and
                            score["validation_selection_lock_sha256"] == lock_sha and
                            score["dataset"] == dataset and score["arm"] == arm and
                            score["lr"] == lr and score["weight_decay"] == wd and
                            score["seed"] == seed and
                            score["selected_candidate"] == ((lr, wd) == chosen) and
                            score["predeclared_default"] == ((lr, wd) == DEFAULT) and
                            score["checkpoint_sha256"] == source["checkpoint_sha256"] ==
                            lock["cells"][key]["checkpoint_sha256"] and
                            score["test_predictions_sha256"] ==
                            source["source_float_predictions_sha256"] ==
                            final["scores"][key]["predictions_sha256"] and
                            source["source_score_sha256"] == sha(score_path) and
                            all(math.isfinite(float(score[field])) for field in
                                ("valid_accuracy", "valid_ce", "test_accuracy", "test_ce")) and
                            0 <= score["valid_accuracy"] <= 1 and
                            0 <= score["test_accuracy"] <= 1 and
                            min(score["valid_ce"], score["test_ce"]) >= 0,
                            f"Score metadata or source hashes differ: {key}")
                    hard_path = ROOT / source["hard_path"]
                    require(sha(hard_path) == source["hard_sha256"],
                            f"Hard classes changed: {key}")
                    with np.load(hard_path, allow_pickle=False) as arrays:
                        pooled = arrays["pooled_test_class"]
                        members = arrays["member_test_class"]
                    member_count = 1 if arm == "base" else 4
                    classes = hard["graph_references"][dataset]["classes"]
                    require(pooled.dtype == members.dtype == np.int16 and
                            pooled.shape == (len(truth),) and
                            members.shape == (member_count, len(truth)) and
                            source["member_count"] == member_count and
                            np.all((pooled >= 0) & (pooled < classes)) and
                            np.all((members >= 0) & (members < classes)),
                            f"Hard class shape/range differs: {key}")
                    accuracy = float(np.mean(pooled == truth))
                    member_accuracy = float(np.mean(members == truth[None, :]))
                    require(abs(accuracy - score["test_accuracy"]) <= 1e-7 and
                            abs(accuracy - final["scores"][key]["test_accuracy"]) <= 1e-7 and
                            abs(accuracy - strict["per_checkpoint"][key][
                                "test_accuracy"]) <= 1e-7 and
                            abs(accuracy - source["test_accuracy"]) <= 1e-12 and
                            abs(member_accuracy - source[
                                "mean_member_test_accuracy"]) <= 1e-12,
                            f"Hard classes do not reproduce audited accuracy: {key}")
    actual_scores = {str(path.parent.relative_to(ROOT / "scores"))
                     for path in (ROOT / "scores").rglob("score.json")}
    actual_hard = {str(path.relative_to(ROOT / "hard_decisions"))
                   for path in (ROOT / "hard_decisions").rglob("*.npz")
                   if path.name != "test_reference.npz"}
    require(expected == set(final["scores"]) ==
            set(strict["per_checkpoint"]) == set(hard["cells"]) ==
            actual_scores and
            actual_hard == {f"{key}.npz" for key in expected} and
            strict["replayed_allowed_checkpoints"] == len(expected),
            "Unexpected or incomplete selected/default score or hard-class set")
    return len(expected)


def main() -> None:
    lock = read("VALIDATION_SELECTION_LOCK.json")
    freeze_sha = sha(ROOT / "FROZEN_STUDY.json")
    tools_lock = read("POSTFREEZE_AUDIT_EXPORT_SOURCE_LOCK.json")
    require(tools_lock["status"] == "PRETEST_INDEPENDENT_AUDIT_AND_EXPORT_SOURCE_LOCK" and
            tools_lock["study_freeze_sha256"] == freeze_sha and
            tools_lock["test_scores_opened_at_lock"] is False and
            all(sha(ROOT / name) == digest for name, digest in
                tools_lock["source_sha256"].items()),
            "Pretest exact-decision auditor or exporter source differs")
    audited_cells = audit_selection(lock, freeze_sha)
    audited_scores = audit_hard_scores(lock, freeze_sha)
    print(json.dumps({"status": "PASS",
                      "validation_cells": len(audited_cells),
                      "selected_default_hard_scores": audited_scores,
                      "scope": "trace and selection replay plus test accuracy from hard classes; model/logit replay requires omitted weights and float logits"},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
