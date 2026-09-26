"""Independent complete validation and selected-score audit for NORM-SYNC."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import mechanism as mech
import norm_sync_v2 as study
import tuning as base
import verify_mechanism as mech_audit


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
EPOCHS = 1000
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now",
                "graph_weights_equal", "reference_graph_update_norm",
                "candidate_graph_update_norm", "applied_graph_update_norm",
                "relative_norm_error", "nominal_norm_scale", "applied_norm_scale",
                "relative_scale_correction", "rounding_correction_iterations",
                "direct_cast_abs_norm_error", "zero_case")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return study.sha(path)


def key(dataset, seed):
    return f"{dataset}/seed{seed}"


def audit_one(dataset, seed, freeze_sha):
    cell = study.cell_dir(dataset, seed)
    result_path = cell / "result.json"
    trace_path = cell / "validation_trace.csv"
    require(result_path.is_file() and trace_path.is_file(),
            f"Missing NORM-SYNC cell {dataset}/{seed}")
    result = json.loads(result_path.read_text())
    require(result["protocol"] == "norm_matched_sync_v2" and
            result["freeze_sha256"] == freeze_sha and
            result["dataset"] == dataset and result["seed"] == seed and
            result["epochs_required"] == EPOCHS and
            result["validation_trace_sha256"] == sha(trace_path),
            f"NORM-SYNC cell identity/hash mismatch: {dataset}/{seed}")
    with trace_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == TRACE_FIELDS,
                f"NORM-SYNC trace schema mismatch: {dataset}/{seed}")
        rows = list(reader)
    require(len(rows) == result["epochs_completed"] and len(rows) <= EPOCHS,
            f"NORM-SYNC trace length mismatch: {dataset}/{seed}")
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    max_error, min_scale, max_scale = 0.0, float("inf"), 0.0
    max_correction, corrected_steps, max_direct_error = 0.0, 0, 0.0
    for epoch, row in enumerate(rows, 1):
        require(int(row["epoch"]) == epoch and row["graph_weights_equal"] == "1",
                f"Epoch/equal-graph gate failed: {dataset}/{seed}/{epoch}")
        acc, ce = float(row["valid_accuracy"]), float(row["valid_ce"])
        ref = float(row["reference_graph_update_norm"])
        candidate = float(row["candidate_graph_update_norm"])
        applied = float(row["applied_graph_update_norm"])
        error = float(row["relative_norm_error"])
        nominal = float(row["nominal_norm_scale"])
        scale = float(row["applied_norm_scale"])
        correction = float(row["relative_scale_correction"])
        iterations = int(row["rounding_correction_iterations"])
        direct_error = float(row["direct_cast_abs_norm_error"])
        require(all(math.isfinite(x) for x in
                    (acc, ce, ref, candidate, applied, error, nominal,
                     scale, correction, direct_error)) and
                0 <= acc <= 1 and ce >= 0 and
                min(ref, candidate, applied, error, nominal, scale, direct_error) >= 0 and
                iterations in (0, 48),
                f"Invalid NORM-SYNC trace number: {dataset}/{seed}/{epoch}")
        allowed = max(1e-8, 1e-5 * ref)
        require(abs(applied - ref) <= allowed and
                abs(error - abs(applied - ref) / max(ref, 1e-8)) <= 1e-7,
                f"Applied graph-step norm gate failed: {dataset}/{seed}/{epoch}")
        if ref == 0.0:
            require(nominal == scale == correction == direct_error == 0.0 and
                    iterations == 0 and row["zero_case"] ==
                    ("both_zero" if candidate == 0.0 else "zero_reference"),
                    f"Reference-zero rule mismatch: {dataset}/{seed}/{epoch}")
        else:
            require(candidate > 0 and row["zero_case"] == "none" and
                    abs(nominal - ref / candidate) <=
                    1e-6 * max(1.0, ref / candidate) and
                    abs(correction) <= 0.001 + 1e-12 and
                    abs(scale / nominal - 1.0 - correction) <= 1e-10 and
                    ((iterations == 0 and direct_error <= allowed and
                      correction == 0.0) or
                     (iterations == 48 and direct_error > allowed)),
                    f"V2 bounded rounding correction differs: {dataset}/{seed}/{epoch}")
        better = acc > best_acc or (acc == best_acc and ce < best_ce)
        require(int(row["selected_now"]) == int(better),
                f"NORM-SYNC checkpoint flag mismatch: {dataset}/{seed}/{epoch}")
        if better:
            best_acc, best_ce, best_epoch = acc, ce, epoch
        max_error = max(max_error, error)
        min_scale, max_scale = min(min_scale, scale), max(max_scale, scale)
        max_correction = max(max_correction, abs(correction))
        corrected_steps += int(iterations > 0)
        max_direct_error = max(max_direct_error, direct_error)
    initial = result["initialization"]
    require(initial["canonical_projector_state_sha256"] is not None and
            initial["paired_initial_logits_max_abs_diff"] <= 1e-5,
            f"NORM-SYNC initial-state gate failed: {dataset}/{seed}")
    record = {"result_sha256": sha(result_path), "trace_sha256": sha(trace_path),
              "failure": result["failure"], "initialization": initial}
    checkpoint_path = cell / "checkpoint.pt"
    if result["failure"] is not None:
        require(len(rows) < EPOCHS and not checkpoint_path.exists(),
                f"Invalid NORM-SYNC cell has complete trace/checkpoint: {dataset}/{seed}")
        return record
    require(len(rows) == EPOCHS and checkpoint_path.is_file(),
            f"Incomplete finite NORM-SYNC cell: {dataset}/{seed}")
    require(result["selected_epoch"] == best_epoch and
            result["selected_valid_accuracy"] == best_acc and
            result["selected_valid_ce"] == best_ce and
            result["max_relative_norm_error"] == max_error and
            result["min_norm_scale"] == min_scale and
            result["max_norm_scale"] == max_scale and
            result["max_abs_relative_scale_correction"] == max_correction and
            result["rounding_corrected_steps"] == corrected_steps and
            result["max_direct_cast_abs_norm_error"] == max_direct_error,
            f"Selected result/norm summary differs from trace: {dataset}/{seed}")
    moments = result["first_step_private_moments"]
    require(moments is not None and moments["separate_first_moment_tensors"] > 0 and
            moments["separate_first_moment_tensors"] == moments["separate_second_moment_tensors"] and
            result["reference_moment_tensor_count"] > 0 and
            (moments["first_moment_max_abs_member_difference"] > 0 or
             moments["second_moment_max_abs_member_difference"] > 0),
            f"Separate optimizer-state gate failed: {dataset}/{seed}")
    require(result["checkpoint_sha256"] == sha(checkpoint_path),
            f"NORM-SYNC checkpoint bytes changed: {dataset}/{seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["protocol"] == "norm_matched_sync_v2" and
            checkpoint["dataset"] == dataset and checkpoint["seed"] == seed and
            checkpoint["epoch"] == best_epoch and checkpoint["freeze_sha256"] == freeze_sha and
            mech_audit.state_sha(checkpoint["state_dict"]) == result["selected_state_sha256"],
            f"NORM-SYNC checkpoint identity/state mismatch: {dataset}/{seed}")
    expected = mech_audit.expected_collapsed_state(checkpoint["state_dict"])
    collapsed_path = cell / "collapsed_checkpoint.pt"
    require(collapsed_path.is_file() and
            result["collapsed_checkpoint_sha256"] == sha(collapsed_path),
            f"NORM-SYNC collapsed checkpoint changed: {dataset}/{seed}")
    collapsed = torch.load(collapsed_path, map_location="cpu", weights_only=True)
    require(collapsed["source_checkpoint_sha256"] == sha(checkpoint_path) and
            collapsed["freeze_sha256"] == freeze_sha and
            collapsed["dataset"] == dataset and collapsed["seed"] == seed and
            set(collapsed["state_dict"]) == set(expected) and
            all(torch.equal(collapsed["state_dict"][name], value)
                for name, value in expected.items()) and
            result["collapse_validation"]["collapsed_state_sha256"] ==
            mech_audit.state_sha(expected) and
            result["collapse_validation"]["max_abs_validation_member_logit_difference"] <= 1e-5 and
            result["collapse_validation"]["pooled_validation_decision_mismatches"] == 0,
            f"NORM-SYNC collapse gate failed: {dataset}/{seed}")
    record.update({"checkpoint_sha256": sha(checkpoint_path),
                   "collapsed_checkpoint_sha256": sha(collapsed_path),
                   "selected_epoch": best_epoch, "selected_valid_accuracy": best_acc,
                   "selected_valid_ce": best_ce,
                   "max_relative_norm_error": max_error,
                   "rounding_corrected_steps": corrected_steps,
                   "max_abs_relative_scale_correction": max_correction})
    return record


def audit_and_lock():
    lock_path = ROOT / "NORM_SYNC_V2_VALIDATION_LOCK.json"
    require(not lock_path.exists(), "NORM-SYNC validation lock exists; refusing overwrite")
    freeze_sha = study.check_freeze()
    frozen = json.loads((ROOT / "NORM_SYNC_V2_FREEZE.json").read_text())
    matrix = frozen["matrix"]
    require(tuple(matrix["datasets"]) == DATASETS and
            tuple(matrix["seeds"]) == SEEDS and
            matrix["planned_cells"] == 12 and matrix["epochs"] == EPOCHS and
            matrix["learning_rate"] == 0.001 and matrix["weight_decay"] == 0 and
            matrix["norm_relative_tolerance"] == 1e-5 and
            matrix["norm_absolute_floor"] == 1e-8 and
            matrix["rounding_relative_radius"] == 0.001 and
            matrix["rounding_bisection_steps"] == 48,
            "NORM-SYNC matrix differs from independent audit constants")
    cells = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            cells[key(dataset, seed)] = audit_one(dataset, seed, freeze_sha)
    actual = {str(path.parent.relative_to(ROOT / "results_norm_sync_v2"))
              for path in (ROOT / "results_norm_sync_v2").rglob("result.json")}
    require(len(cells) == 12 and actual == set(cells) and
            all(cell["failure"] is None for cell in cells.values()),
            "NORM-SYNC 12-cell validation matrix incomplete or invalid")
    lock = {"protocol": "norm_matched_sync_v2", "freeze_sha256": freeze_sha,
            "mechanism_freeze_sha256": sha(ROOT / "MECHANISM_FREEZE.json"),
            "base_freeze_sha256": sha(ROOT / "FROZEN_STUDY.json"),
            "cells": cells, "test_scoring_allowlist": "all 12 fixed default cells only",
            "selection_uses_test_labels": False}
    study.write_json(lock_path, lock)
    print(json.dumps({"norm_sync_lock_sha256": sha(lock_path),
                      "validated_cells": len(cells)}), flush=True)


def audit_scores(device: torch.device):
    freeze_sha = study.check_freeze()
    norm_lock_path = ROOT / "NORM_SYNC_V2_VALIDATION_LOCK.json"
    mech_lock_path = ROOT / "MECHANISM_VALIDATION_LOCK.json"
    base_lock_path = ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"
    norm_lock = json.loads(norm_lock_path.read_text())
    mech_lock = json.loads(mech_lock_path.read_text())
    base_lock = json.loads(base_lock_path.read_text())
    require(norm_lock["freeze_sha256"] == freeze_sha and len(norm_lock["cells"]) == 12 and
            mech_lock["freeze_sha256"] == sha(ROOT / "MECHANISM_FREEZE.json") and
            len(mech_lock["cells"]) == 96 and
            base_lock["freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            len(base_lock["cells"]) == 432,
            "All three complete validation locks are required for score replay")
    records, expected = {}, set()
    for dataset in DATASETS:
        bundle, _ = base.load_graph(dataset, device, include_test=True)
        for seed in SEEDS:
            name = key(dataset, seed)
            expected.add(name)
            cell = study.cell_dir(dataset, seed)
            score_dir = ROOT / "scores_norm_sync_v2" / name
            score = json.loads((score_dir / "score.json").read_text())
            locked = norm_lock["cells"][name]
            require(sha(cell / "result.json") == locked["result_sha256"] and
                    sha(cell / "checkpoint.pt") == locked["checkpoint_sha256"] and
                    sha(score_dir / "predictions.npz") == score["predictions_sha256"] and
                    score["protocol"] == "norm_matched_sync_v2" and
                    score["freeze_sha256"] == freeze_sha and
                    score["norm_validation_lock_sha256"] == sha(norm_lock_path) and
                    score["mechanism_validation_lock_sha256"] == sha(mech_lock_path) and
                    score["original_432_validation_lock_sha256"] == sha(base_lock_path) and
                    score["dataset"] == dataset and score["seed"] == seed and
                    score["checkpoint_sha256"] == locked["checkpoint_sha256"],
                    f"NORM-SYNC score/hash metadata mismatch: {name}")
            base.seed_all(seed)
            model, _ = mech.make_model("sync", bundle, device)
            checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            va, vc, vp, _ = mech_audit.fresh_metrics(
                model, "sync", bundle, bundle.valid_idx, bundle.valid_y)
            ta, tc, tp, tm = mech_audit.fresh_metrics(
                model, "sync", bundle, bundle.test_idx, bundle.test_y)
            compact, _ = base.make_model("tied", bundle, device)
            compact.load_state_dict(mech_audit.expected_collapsed_state(
                checkpoint["state_dict"]), strict=True)
            _, _, compact_pooled, compact_members = mech_audit.fresh_metrics(
                compact, "tied", bundle, bundle.test_idx, bundle.test_y)
            require(float((tm - compact_members).abs().max().item()) <= 1e-5 and
                    int((tp.argmax(-1) != compact_pooled.argmax(-1)).sum().item()) == 0 and
                    score["collapsed_test_replay"]["max_abs_member_logit_difference"] <= 1e-5 and
                    score["collapsed_test_replay"]["decision_mismatches"] == 0,
                    f"NORM-SYNC collapsed test replay failed: {name}")
            with np.load(score_dir / "predictions.npz", allow_pickle=False) as saved:
                old_valid, old_test, old_members = (saved["valid_pooled_logits"],
                                                    saved["test_pooled_logits"],
                                                    saved["test_member_logits"])
            require(np.allclose(old_valid, vp.numpy(), rtol=1e-5, atol=1e-5) and
                    np.allclose(old_test, tp.numpy(), rtol=1e-5, atol=1e-5) and
                    np.allclose(old_members, tm.numpy(), rtol=1e-5, atol=1e-5) and
                    np.array_equal(old_valid.argmax(-1), vp.numpy().argmax(-1)) and
                    np.array_equal(old_test.argmax(-1), tp.numpy().argmax(-1)) and
                    abs(va - score["valid_accuracy"]) <= 1e-7 and
                    abs(vc - score["valid_ce"]) <= 1e-5 and
                    abs(ta - score["test_accuracy"]) <= 1e-7 and
                    abs(tc - score["test_ce"]) <= 1e-5,
                    f"Fresh NORM-SYNC score/decision replay failed: {name}")
            records[name] = {"score_sha256": sha(score_dir / "score.json"),
                             "predictions_sha256": sha(score_dir / "predictions.npz"),
                             "test_accuracy": ta, "test_ce": tc}
    actual = {str(path.parent.relative_to(ROOT / "scores_norm_sync_v2"))
              for path in (ROOT / "scores_norm_sync_v2").rglob("score.json")}
    require(actual == expected, "NORM-SYNC test score exists outside fixed 12-cell set")
    report = {"protocol": "norm_matched_sync_v2", "freeze_sha256": freeze_sha,
              "norm_validation_lock_sha256": sha(norm_lock_path),
              "mechanism_validation_lock_sha256": sha(mech_lock_path),
              "original_432_validation_lock_sha256": sha(base_lock_path),
              "scores": records}
    study.write_json(ROOT / "NORM_SYNC_V2_FINAL_SCORE_AUDIT.json", report)
    print(json.dumps({"fresh_replayed_scores": len(records),
                      "report_sha256": sha(ROOT / "NORM_SYNC_V2_FINAL_SCORE_AUDIT.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-and-lock", "audit-scores"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "audit-and-lock":
        audit_and_lock()
    else:
        audit_scores(torch.device(args.device))


if __name__ == "__main__":
    main()
