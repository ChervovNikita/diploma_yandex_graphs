"""Public, CPU-only audit of optimizer validation traces and hard decisions.

This checks the 96-cell mechanism and 12-cell NORM-SYNC validation records,
their fixed score allowlists, and exported class decisions. It does not
replay omitted checkpoints or raw float logits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from export_compact_decisions import verify as verify_decisions


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("tied", "untied", "sync")
SEEDS = (0, 1, 2)
DEFAULT = (0.001, 0.0)
SYNC = ((0.0003, 0.0), (0.0003, 0.01), (0.001, 0.0),
        (0.001, 0.01), (0.003, 0.0), (0.003, 0.01))
EPOCHS = 1000
MECH_TRACE = ("epoch", "valid_accuracy", "valid_ce", "selected_now", "graph_weights_equal")
NORM_TRACE = ("epoch", "valid_accuracy", "valid_ce", "selected_now",
              "graph_weights_equal", "reference_graph_update_norm",
              "candidate_graph_update_norm", "applied_graph_update_norm",
              "relative_norm_error", "nominal_norm_scale", "applied_norm_scale",
              "relative_scale_correction", "rounding_correction_iterations",
              "direct_cast_abs_norm_error", "zero_case")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def key(dataset: str, arm: str, lr: float, wd: float, seed: int) -> str:
    return f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"


def trace_winner(path: Path, fields: tuple[str, ...], mode: str,
                 complete: bool = True) -> tuple[int, float, float, float, int]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == fields, f"Trace schema differs: {path}")
        rows = list(reader)
    require(len(rows) == EPOCHS if complete else len(rows) < EPOCHS,
            f"Trace completion flag differs: {path}")
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    max_norm_error = 0.0
    for epoch, row in enumerate(rows, 1):
        acc, ce = float(row["valid_accuracy"]), float(row["valid_ce"])
        require(int(row["epoch"]) == epoch and
                all(math.isfinite(v) for v in (acc, ce)) and
                0 <= acc <= 1 and ce >= 0,
                f"Invalid validation row: {path}, epoch {epoch}")
        better = acc > best_acc or (acc == best_acc and ce < best_ce)
        require(int(row["selected_now"]) == int(better),
                f"Checkpoint flag differs: {path}, epoch {epoch}")
        require(row["graph_weights_equal"] ==
                ("1" if mode in ("norm", "sync") else ""),
                f"Graph equality field differs: {path}, epoch {epoch}")
        if mode == "norm":
            ref = float(row["reference_graph_update_norm"])
            cand = float(row["candidate_graph_update_norm"])
            applied = float(row["applied_graph_update_norm"])
            error = float(row["relative_norm_error"])
            nominal = float(row["nominal_norm_scale"])
            scale = float(row["applied_norm_scale"])
            correction = float(row["relative_scale_correction"])
            iterations = int(row["rounding_correction_iterations"])
            direct_error = float(row["direct_cast_abs_norm_error"])
            require(all(math.isfinite(v) and v >= 0 for v in
                        (ref, cand, applied, error, nominal, scale, direct_error)) and
                    math.isfinite(correction) and iterations in (0, 48) and
                    abs(applied - ref) <= max(1e-8, 1e-5 * ref) and
                    abs(error - abs(applied - ref) / max(ref, 1e-8)) <= 1e-7,
                    f"NORM-SYNC step-norm gate differs: {path}, epoch {epoch}")
            if ref == 0.0:
                require(nominal == scale == correction == direct_error == 0.0 and
                        iterations == 0 and row["zero_case"] ==
                        ("both_zero" if cand == 0.0 else "zero_reference"),
                        f"NORM-SYNC zero rule differs: {path}, epoch {epoch}")
            else:
                require(cand > 0 and row["zero_case"] == "none" and
                        abs(nominal - ref / cand) <= 1e-6 * max(1.0, ref / cand) and
                        abs(correction) <= 0.001 + 1e-12 and
                        abs(scale / nominal - 1.0 - correction) <= 1e-10 and
                        ((iterations == 0 and direct_error <= max(1e-8, 1e-5 * ref)
                          and correction == 0.0) or
                         (iterations == 48 and direct_error > max(1e-8, 1e-5 * ref))),
                        f"NORM-SYNC scale differs: {path}, epoch {epoch}")
            max_norm_error = max(max_norm_error, error)
        if better:
            best_acc, best_ce, best_epoch = acc, ce, epoch
    return best_epoch, best_acc, best_ce, max_norm_error, len(rows)


def source_freeze(freeze: dict) -> None:
    for name, digest in freeze["source_sha256"].items():
        require(sha(ROOT / name) == digest, f"Frozen source differs: {name}")


def verify_descriptor_links(family: str, base: dict) -> None:
    manifest = read(ROOT / "compact_decisions" / family / "manifest.json")
    require(set(manifest["datasets"]) == set(DATASETS),
            f"Compact dataset set differs: {family}")
    for dataset in DATASETS:
        spec = manifest["datasets"][dataset]
        descriptor = base["graphs"][dataset]["descriptor"]
        require(spec["official_split"] == descriptor["split"] and
                spec["class_count"] == descriptor["classes"] and
                spec["test_count"] == descriptor["split_sizes"]["test"] and
                spec["full_label_tensor_sha256"] ==
                descriptor["tensor_sha256"]["labels"] and
                spec["test_indices_tensor_sha256"] ==
                descriptor["tensor_sha256"]["test_indices"],
                f"Compact official labels/split differ from base freeze: {family}/{dataset}")


def mechanism() -> tuple[dict, dict]:
    freeze_path = ROOT / "MECHANISM_FREEZE.json"
    lock_path = ROOT / "MECHANISM_VALIDATION_LOCK.json"
    base_path = ROOT / "FROZEN_STUDY.json"
    global_path = ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"
    audit_path = ROOT / "MECHANISM_FINAL_SCORE_AUDIT.json"
    freeze, lock, base, global_lock, audit = map(read,
        (freeze_path, lock_path, base_path, global_path, audit_path))
    source_freeze(freeze)
    source_freeze(base)
    # The frozen mechanism auditor's report has no protocol field. Its
    # freeze/lock identities and every scored cell are checked below.
    require(freeze["protocol"] == lock["protocol"] ==
            "optimizer_aggregation_mechanism_v1" and
            ("protocol" not in audit or audit["protocol"] == freeze["protocol"]) and
            lock["freeze_sha256"] == audit["freeze_sha256"] == sha(freeze_path) and
            freeze["base_validation_grid_freeze_sha256"] ==
            global_lock["freeze_sha256"] == sha(base_path) and
            len(global_lock["cells"]) == 432 and len(lock["cells"]) == 96 and
            audit["mechanism_validation_lock_sha256"] == sha(lock_path) and
            audit["original_432_validation_lock_sha256"] == sha(global_path),
            "Mechanism source, lock, or audit identity differs")
    matrix = freeze["matrix"]
    require(tuple(matrix["datasets"]) == DATASETS and
            tuple(matrix["arms"]) == ARMS and
            tuple(matrix["seeds"]) == SEEDS and matrix["epochs"] == EPOCHS and
            tuple((row["lr"], row["weight_decay"]) for row in
                  matrix["sync_candidates"]) == SYNC,
            "Mechanism frozen matrix differs")
    all_cells, allowed_scores = set(), set()
    for dataset in DATASETS:
        candidate_table = []
        for arm in ARMS:
            for lr, wd in (SYNC if arm == "sync" else (DEFAULT,)):
                rows = []
                for seed in SEEDS:
                    name = key(dataset, arm, lr, wd, seed)
                    all_cells.add(name)
                    folder = ROOT / "results" / name
                    result_path, trace_path = folder / "result.json", folder / "validation_trace.csv"
                    frozen_cell = lock["cells"][name]
                    require(sha(result_path) == frozen_cell["result_sha256"] and
                            sha(trace_path) == frozen_cell["trace_sha256"],
                            f"Locked mechanism cell changed: {name}")
                    result = read(result_path)
                    require(result["protocol"] == freeze["protocol"] and
                            result["freeze_sha256"] == sha(freeze_path) and
                            result["dataset"] == dataset and result["arm"] == arm and
                            result["learning_rate"] == lr and result["weight_decay"] == wd and
                            result["seed"] == seed and
                            result["failure"] == frozen_cell["failure"] and
                            result["validation_trace_sha256"] == sha(trace_path),
                            f"Mechanism result identity/failure differs: {name}")
                    failed = result["failure"] is not None
                    epoch, acc, ce, _, count = trace_winner(
                        trace_path, MECH_TRACE, "sync" if arm == "sync" else "control",
                        complete=not failed)
                    require(result["epochs_completed"] == count,
                            f"Mechanism trace length differs: {name}")
                    if failed:
                        require(arm == "sync" and "checkpoint_sha256" not in frozen_cell,
                                f"Failed default control or locked checkpoint: {name}")
                        rows.append(None)
                    else:
                        require(result["selected_epoch"] == frozen_cell["selected_epoch"] == epoch and
                                result["selected_valid_accuracy"] ==
                                frozen_cell["selected_valid_accuracy"] == acc and
                                result["selected_valid_ce"] ==
                                frozen_cell["selected_valid_ce"] == ce and
                                result["checkpoint_sha256"] == frozen_cell["checkpoint_sha256"],
                                f"Mechanism trace winner differs: {name}")
                        rows.append((acc, ce))
                if arm == "sync":
                    valid = all(row is not None for row in rows)
                    candidate_table.append((
                        lr, wd, valid,
                        sum(v[0] for v in rows) / 3 if valid else None,
                        sum(v[1] for v in rows) / 3 if valid else None))
        recorded = lock["sync_selections"][dataset]
        require(len(recorded["candidate_table"]) == len(SYNC),
                f"SYNC candidate table incomplete: {dataset}")
        for (lr, wd, valid, acc, ce), old in zip(candidate_table, recorded["candidate_table"]):
            require((old["lr"], old["weight_decay"], old["valid"],
                     old["mean_valid_accuracy"], old["mean_valid_ce"]) ==
                    (lr, wd, valid, acc, ce),
                    f"SYNC candidate mean differs: {dataset}")
        valid_candidates = [row for row in candidate_table if row[2]]
        require(valid_candidates, f"No valid SYNC candidate: {dataset}")
        chosen = max(valid_candidates, key=lambda row:
                     (row[3], -row[4], -row[0], -row[1]))
        require(recorded["selected_candidate"] == [chosen[0], chosen[1]] and
                recorded["selected_mean_valid_accuracy"] == chosen[3] and
                recorded["selected_mean_valid_ce"] == chosen[4],
                f"SYNC candidate choice differs: {dataset}")
        for arm in ARMS:
            for lr, wd in ({DEFAULT, (chosen[0], chosen[1])} if arm == "sync"
                           else {DEFAULT}):
                for seed in SEEDS:
                    allowed_scores.add(key(dataset, arm, lr, wd, seed))
    actual = {p.parent.relative_to(ROOT / "results").as_posix()
              for p in (ROOT / "results").rglob("result.json")}
    require(all_cells == actual == set(lock["cells"]) and len(all_cells) == 96,
            "Mechanism validation cell set differs")
    audit_scores = audit["scores"]
    actual_scores = {p.parent.relative_to(ROOT / "scores").as_posix()
                     for p in (ROOT / "scores").rglob("score.json")}
    require(set(audit_scores) == actual_scores == allowed_scores,
            "Mechanism score allowlist differs")
    for name in allowed_scores:
        path = ROOT / "scores" / name / "score.json"
        score = read(path)
        dataset, arm, candidate, seed_part = name.split("/")
        lr_part, wd_part = candidate.split("_")
        lr, wd, seed = float(lr_part[2:]), float(wd_part[2:]), int(seed_part[4:])
        selected = tuple(lock["sync_selections"][dataset]["selected_candidate"])
        require(sha(path) == audit_scores[name]["score_sha256"] and
                score["protocol"] == freeze["protocol"] and
                score["freeze_sha256"] == sha(freeze_path) and
                score["mechanism_validation_lock_sha256"] == sha(lock_path) and
                score["original_432_validation_lock_sha256"] == sha(global_path) and
                (score["dataset"], score["arm"], score["learning_rate"],
                 score["weight_decay"], score["seed"]) ==
                (dataset, arm, lr, wd, seed) and
                score["selected_sync_candidate"] ==
                (arm == "sync" and (lr, wd) == selected) and
                score["predeclared_default"] == ((lr, wd) == DEFAULT) and
                score["checkpoint_sha256"] == lock["cells"][name]["checkpoint_sha256"] and
                score["predictions_sha256"] == audit_scores[name]["predictions_sha256"] and
                all(math.isfinite(float(score[field])) for field in
                    ("valid_accuracy", "valid_ce", "test_accuracy", "test_ce")) and
                0 <= score["test_accuracy"] <= 1 and score["test_ce"] >= 0 and
                abs(score["test_accuracy"] - audit_scores[name]["test_accuracy"]) <= 1e-7,
                f"Mechanism audited score differs: {name}")
    verify_descriptor_links("mechanism", base)
    verify_decisions(ROOT / "compact_decisions" / "mechanism")
    return lock, audit


def norm_sync(mech_lock: dict, mech_audit: dict) -> None:
    v1_freeze_path = ROOT / "NORM_SYNC_FREEZE.json"
    v1_freeze = read(v1_freeze_path)
    v1_diag = read(ROOT / "NORM_SYNC_ROUNDING_DIAG_V1_CORA_SEED0.json")
    source_freeze(v1_freeze)
    require(v1_freeze["protocol"] == "norm_matched_sync_v1" and
            v1_diag["freeze_sha256"] == sha(v1_freeze_path) and
            v1_diag["failed_epoch"] == 510 and
            v1_diag["dataset"] == "cora" and v1_diag["seed"] == 0 and
            v1_diag["test_metrics_computed"] is False,
            "NORM-SYNC V1 failure chronology differs")
    freeze_path = ROOT / "NORM_SYNC_V2_FREEZE.json"
    lock_path = ROOT / "NORM_SYNC_V2_VALIDATION_LOCK.json"
    audit_path = ROOT / "NORM_SYNC_V2_FINAL_SCORE_AUDIT.json"
    freeze, lock, audit = map(read, (freeze_path, lock_path, audit_path))
    source_freeze(freeze)
    design = read(ROOT / "NORM_SYNC_V2_DESIGN_FREEZE.json")
    require(design["protocol"] == "norm_matched_sync_design_v2" and
            design["source_sha256"]["NORM_SYNC_FREEZE.json"] == sha(v1_freeze_path) and
            design["source_sha256"]["NORM_SYNC_ROUNDING_DIAG_V1_CORA_SEED0.json"] ==
            sha(ROOT / "NORM_SYNC_ROUNDING_DIAG_V1_CORA_SEED0.json") and
            design["norm_relative_tolerance"] == 1e-5 and
            design["norm_absolute_floor"] == 1e-8 and
            design["rounding_relative_radius"] == 0.001 and
            design["rounding_bisection_steps"] == 48,
            "NORM-SYNC V2 design/failure link differs")
    require(freeze["protocol"] == lock["protocol"] == audit["protocol"] ==
            "norm_matched_sync_v2" and
            freeze["design_freeze_sha256"] ==
            sha(ROOT / "NORM_SYNC_V2_DESIGN_FREEZE.json") and
            lock["freeze_sha256"] == audit["freeze_sha256"] == sha(freeze_path) and
            freeze["mechanism_freeze_sha256"] == sha(ROOT / "MECHANISM_FREEZE.json") and
            freeze["base_freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            lock["mechanism_freeze_sha256"] == sha(ROOT / "MECHANISM_FREEZE.json") and
            lock["base_freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            audit["norm_validation_lock_sha256"] == sha(lock_path) and
            audit["mechanism_validation_lock_sha256"] ==
            sha(ROOT / "MECHANISM_VALIDATION_LOCK.json") and
            audit["original_432_validation_lock_sha256"] ==
            sha(ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json") and
            len(lock["cells"]) == 12 and len(mech_lock["cells"]) == 96,
            "NORM-SYNC source, lock, or audit identity differs")
    matrix = freeze["matrix"]
    require(tuple(matrix["datasets"]) == DATASETS and
            tuple(matrix["seeds"]) == SEEDS and matrix["epochs"] == EPOCHS and
            matrix["planned_cells"] == 12 and matrix["learning_rate"] == 0.001 and
            matrix["weight_decay"] == 0 and matrix["norm_relative_tolerance"] == 1e-5 and
            matrix["norm_absolute_floor"] == 1e-8 and
            matrix["rounding_relative_radius"] == 0.001 and
            matrix["rounding_bisection_steps"] == 48,
            "NORM-SYNC frozen matrix differs")
    all_cells = set()
    for dataset in DATASETS:
        for seed in SEEDS:
            name = f"{dataset}/seed{seed}"
            all_cells.add(name)
            folder = ROOT / "results_norm_sync_v2" / name
            result_path, trace_path = folder / "result.json", folder / "validation_trace.csv"
            frozen_cell = lock["cells"][name]
            require(sha(result_path) == frozen_cell["result_sha256"] and
                    sha(trace_path) == frozen_cell["trace_sha256"],
                    f"Locked NORM-SYNC cell changed: {name}")
            result = read(result_path)
            require(result["protocol"] == freeze["protocol"] and
                    result["freeze_sha256"] == sha(freeze_path) and
                    result["dataset"] == dataset and result["seed"] == seed and
                    result["failure"] is None and
                    result["validation_trace_sha256"] == sha(trace_path),
                    f"NORM-SYNC result identity or finite gate differs: {name}")
            epoch, acc, ce, max_error, count = trace_winner(trace_path, NORM_TRACE, "norm")
            require(result["selected_epoch"] == frozen_cell["selected_epoch"] == epoch and
                    result["epochs_completed"] == count and
                    result["selected_valid_accuracy"] ==
                    frozen_cell["selected_valid_accuracy"] == acc and
                    result["selected_valid_ce"] ==
                    frozen_cell["selected_valid_ce"] == ce and
                    result["checkpoint_sha256"] == frozen_cell["checkpoint_sha256"] and
                    result["max_relative_norm_error"] ==
                    frozen_cell["max_relative_norm_error"] == max_error,
                    f"NORM-SYNC trace winner or norm summary differs: {name}")
    actual = {p.parent.relative_to(ROOT / "results_norm_sync_v2").as_posix()
              for p in (ROOT / "results_norm_sync_v2").rglob("result.json")}
    require(all_cells == actual == set(lock["cells"]),
            "NORM-SYNC validation cell set differs")
    audit_scores = audit["scores"]
    actual_scores = {p.parent.relative_to(ROOT / "scores_norm_sync_v2").as_posix()
                     for p in (ROOT / "scores_norm_sync_v2").rglob("score.json")}
    require(set(audit_scores) == actual_scores == all_cells,
            "NORM-SYNC score set differs")
    for name in all_cells:
        path = ROOT / "scores_norm_sync_v2" / name / "score.json"
        score = read(path)
        dataset, seed_part = name.split("/")
        require(sha(path) == audit_scores[name]["score_sha256"] and
                score["protocol"] == freeze["protocol"] and
                score["freeze_sha256"] == sha(freeze_path) and
                score["norm_validation_lock_sha256"] == sha(lock_path) and
                score["mechanism_validation_lock_sha256"] ==
                sha(ROOT / "MECHANISM_VALIDATION_LOCK.json") and
                score["original_432_validation_lock_sha256"] ==
                sha(ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json") and
                (score["dataset"], score["seed"]) ==
                (dataset, int(seed_part[4:])) and
                score["checkpoint_sha256"] == lock["cells"][name]["checkpoint_sha256"] and
                score["predictions_sha256"] == audit_scores[name]["predictions_sha256"] and
                all(math.isfinite(float(score[field])) for field in
                    ("valid_accuracy", "valid_ce", "test_accuracy", "test_ce")) and
                0 <= score["test_accuracy"] <= 1 and score["test_ce"] >= 0 and
                abs(score["test_accuracy"] - audit_scores[name]["test_accuracy"]) <= 1e-7,
                f"NORM-SYNC audited score differs: {name}")
    verify_descriptor_links("norm_sync_v2", read(ROOT / "FROZEN_STUDY.json"))
    verify_decisions(ROOT / "compact_decisions" / "norm_sync_v2")


def main() -> None:
    mechanism_lock, mechanism_audit = mechanism()
    norm_sync(mechanism_lock, mechanism_audit)
    print(json.dumps({"status": "PASS", "validation_cells": 108,
                      "mechanism_scored_cells": len(mechanism_audit["scores"]),
                      "norm_sync_scored_cells": 12,
                      "scope": "validation trace/selection and hard-class accuracy; checkpoint and float-logit replay require omitted files"},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
