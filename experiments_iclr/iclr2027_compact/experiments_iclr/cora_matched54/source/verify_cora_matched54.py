"""Independent validation lock and test replay for the WikiCS 54-cell study."""
from __future__ import annotations

import argparse
import json
import statistics

import numpy as np
import torch

import tuning as primary
import verify_tuning as original_auditor
import cora_matched54 as study


OUT = study.OUT
ARMS = ("base", "ens", "private_last")
SEEDS = (0, 1, 2)
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01), (0.001, 0.0),
              (0.001, 0.01), (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def independent_cell(arm, lr, wd, seed, freeze_sha, bundle, device):
    identity = study.key(arm, lr, wd, seed)
    old_root = original_auditor.ROOT
    try:
        original_auditor.ROOT = OUT
        row = original_auditor.audit_one("cora_normalized", arm, lr, wd, seed,
                                         freeze_sha, bundle, device)
    finally:
        original_auditor.ROOT = old_root
    cell = study.cell_dir(arm, lr, wd, seed)
    record = json.loads((cell / "result.json").read_text())
    require(record["initialization"]["parameter_count"] ==
            study.PARAMETER_COUNTS[arm] and
            record["epochs_required"] == 1000,
            f"Parameter count or epoch requirement differs: {identity}")
    companion_path = cell / "validation_companion.npz"
    manifest_path = cell / "validation_companion.json"
    if record["failure"] is not None:
        require(not companion_path.exists() and not manifest_path.exists(),
                f"Failed cell has companion: {identity}")
        return row
    require(companion_path.is_file() and manifest_path.is_file(),
            f"Missing validation companion: {identity}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["protocol"] == study.PROTOCOL and manifest["arm"] == arm and
            manifest["width"] == study.WIDTHS[arm] and manifest["lr"] == lr and
            manifest["weight_decay"] == wd and manifest["seed"] == seed and
            manifest["result_sha256"] == row["result_sha256"] and
            manifest["checkpoint_sha256"] == row["checkpoint_sha256"] and
            manifest["predictions_sha256"] == study.sha(companion_path),
            f"Companion identity differs: {identity}")
    primary.seed_all(seed)
    model, _ = primary.make_model(arm, bundle, device)
    payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    va, vc, pooled, members = original_auditor.independent_metrics(
        model, arm, bundle, bundle.valid_idx, bundle.valid_y)
    require(abs(va - record["selected_valid_accuracy"]) <= 1e-7 and
            abs(vc - record["selected_valid_ce"]) <= 1e-5 and
            abs(va - manifest["valid_accuracy"]) <= 1e-7 and
            abs(vc - manifest["valid_ce"]) <= 1e-5,
            f"Independent validation metric differs: {identity}")
    with np.load(companion_path, allow_pickle=False) as saved:
        require(set(saved.files) == {"valid_indices", "valid_labels",
                                     "valid_pooled_logits", "valid_member_logits"} and
                np.array_equal(saved["valid_indices"], bundle.valid_idx.cpu().numpy()) and
                np.array_equal(saved["valid_labels"], bundle.valid_y.cpu().numpy()) and
                np.allclose(saved["valid_pooled_logits"], pooled.numpy(), rtol=1e-5, atol=1e-5) and
                np.allclose(saved["valid_member_logits"], members.numpy(), rtol=1e-5, atol=1e-5) and
                np.array_equal(saved["valid_pooled_logits"].argmax(-1), pooled.argmax(-1).numpy()) and
                np.array_equal(saved["valid_member_logits"].argmax(-1), members.argmax(-1).numpy()),
                f"Independent validation floats/decisions differ: {identity}")
    row.update({"validation_companion_sha256": study.sha(companion_path),
                "validation_companion_manifest_sha256": study.sha(manifest_path)})
    return row


def lock(device):
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    require(not lock_path.exists(), "Validation-lock overwrite refused")
    freeze_sha = study.check_freeze()
    frozen = json.loads(study.FREEZE.read_text())
    matrix = frozen["matrix"]
    require(matrix["dataset"] == "cora_normalized" and tuple(matrix["arms"]) == ARMS and
            matrix["widths"] == study.WIDTHS and tuple(matrix["seeds"]) == SEEDS and
            matrix["epochs"] == 1000 and tuple(matrix["default_candidate"]) == DEFAULT and
            tuple((r["lr"], r["weight_decay"]) for r in matrix["candidates"]) == CANDIDATES,
            "Frozen 54-cell matrix differs")
    bundle, _ = primary.load_graph("cora_normalized", device, include_test=False)
    cells, selections = {}, {}
    saved_width = primary.WIDTH
    try:
        for arm in ARMS:
            study.with_width(arm)
            candidates = []
            for lr, wd in CANDIDATES:
                rows = []
                for seed in SEEDS:
                    row = independent_cell(arm, lr, wd, seed, freeze_sha, bundle, device)
                    cells[study.key(arm, lr, wd, seed)] = row
                    rows.append(row)
                valid = all(r["failure"] is None for r in rows)
                candidates.append({"lr": lr, "weight_decay": wd, "valid": valid,
                                   "mean_valid_accuracy": statistics.mean(r["selected_valid_accuracy"] for r in rows) if valid else None,
                                   "mean_valid_ce": statistics.mean(r["selected_valid_ce"] for r in rows) if valid else None,
                                   "seed_valid_accuracy": [r["selected_valid_accuracy"] for r in rows] if valid else None,
                                   "seed_selected_epoch": [r["selected_epoch"] for r in rows] if valid else None})
            valid = [r for r in candidates if r["valid"]]
            require(valid, f"No valid candidates for {arm}")
            chosen = max(valid, key=lambda r: (r["mean_valid_accuracy"], -r["mean_valid_ce"],
                                               -r["lr"], -r["weight_decay"]))
            selections[arm] = {"candidate_table": candidates,
                               "selected_candidate": [chosen["lr"], chosen["weight_decay"]],
                               "selected_mean_valid_accuracy": chosen["mean_valid_accuracy"],
                               "selected_mean_valid_ce": chosen["mean_valid_ce"],
                               "parameter_count": matrix["parameter_counts"][arm],
                               "width": study.WIDTHS[arm]}
    finally:
        primary.WIDTH = saved_width
    require(len(cells) == 54, "Incomplete 54-cell matrix")
    primary.write_json(lock_path, {"protocol": study.PROTOCOL, "freeze_sha256": freeze_sha,
                                   "cells": cells, "selections": selections,
                                   "selection_uses_test_labels": False,
                                   "test_scoring_allowlist": "selected and declared default only"})
    print(json.dumps({"validation_lock_sha256": study.sha(lock_path),
                      "cells": len(cells), "selections": selections}), flush=True)


def audit_scores(device):
    freeze_sha = study.check_freeze()
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    require(lock["protocol"] == study.PROTOCOL and lock["freeze_sha256"] == freeze_sha and
            len(lock["cells"]) == 54 and set(lock["selections"]) == set(ARMS),
            "Incomplete validation lock")
    bundle, _ = primary.load_graph("cora_normalized", device, include_test=True)
    expected, scores = set(), {}
    saved_width = primary.WIDTH
    try:
        for arm in ARMS:
            study.with_width(arm)
            selected = tuple(lock["selections"][arm]["selected_candidate"])
            for lr, wd in dict.fromkeys((selected, DEFAULT)):
                for seed in SEEDS:
                    identity = study.key(arm, lr, wd, seed)
                    expected.add(identity)
                    cell = study.cell_dir(arm, lr, wd, seed)
                    folder = OUT / "scores" / identity
                    score_path, prediction_path = folder / "score.json", folder / "predictions.npz"
                    require(score_path.is_file() and prediction_path.is_file(),
                            f"Missing allowed score: {identity}")
                    row = json.loads(score_path.read_text())
                    frozen = lock["cells"][identity]
                    require(study.sha(cell / "result.json") == frozen["result_sha256"] and
                            study.sha(cell / "checkpoint.pt") == frozen["checkpoint_sha256"] and
                            row["protocol"] == study.PROTOCOL and row["freeze_sha256"] == freeze_sha and
                            row["validation_selection_lock_sha256"] == study.sha(lock_path) and
                            row["dataset"] == "cora_normalized" and row["arm"] == arm and
                            row["width"] == study.WIDTHS[arm] and row["lr"] == lr and
                            row["weight_decay"] == wd and row["seed"] == seed and
                            row["selected_candidate"] == ((lr, wd) == selected) and
                            row["predeclared_default"] == ((lr, wd) == DEFAULT) and
                            row["checkpoint_sha256"] == frozen["checkpoint_sha256"] and
                            row["test_predictions_sha256"] == study.sha(prediction_path),
                            f"Score identity differs: {identity}")
                    primary.seed_all(seed)
                    model, _ = primary.make_model(arm, bundle, device)
                    payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
                    model.load_state_dict(payload["state_dict"], strict=True)
                    va, vc, vp, _ = original_auditor.independent_metrics(
                        model, arm, bundle, bundle.valid_idx, bundle.valid_y)
                    ta, tc, tp, tm = original_auditor.independent_metrics(
                        model, arm, bundle, bundle.test_idx, bundle.test_y)
                    with np.load(prediction_path, allow_pickle=False) as saved:
                        require(set(saved.files) == {"valid_pooled_logits", "test_pooled_logits", "test_member_logits"} and
                                np.allclose(saved["valid_pooled_logits"], vp.numpy(), rtol=1e-5, atol=1e-5) and
                                np.allclose(saved["test_pooled_logits"], tp.numpy(), rtol=1e-5, atol=1e-5) and
                                np.allclose(saved["test_member_logits"], tm.numpy(), rtol=1e-5, atol=1e-5) and
                                np.array_equal(saved["test_pooled_logits"].argmax(-1), tp.argmax(-1).numpy()) and
                                np.array_equal(saved["test_member_logits"].argmax(-1), tm.argmax(-1).numpy()),
                                f"Independent test floats/decisions differ: {identity}")
                    require(abs(va - row["valid_accuracy"]) <= 1e-7 and
                            abs(vc - row["valid_ce"]) <= 1e-5 and
                            abs(ta - row["test_accuracy"]) <= 1e-7 and
                            abs(tc - row["test_ce"]) <= 1e-5,
                            f"Independent test metric differs: {identity}")
                    scores[identity] = {"score_sha256": study.sha(score_path),
                                        "predictions_sha256": study.sha(prediction_path),
                                        "test_accuracy": ta, "test_ce": tc}
    finally:
        primary.WIDTH = saved_width
    actual = {str(path.parent.relative_to(OUT / "scores")) for path in (OUT / "scores").rglob("score.json")}
    require(actual == expected and len(expected) <= 18, "Unauthorized or missing test scores")
    output = OUT / "FINAL_SCORE_AUDIT.json"
    require(not output.exists(), "Score-audit overwrite refused")
    primary.write_json(output, {"protocol": study.PROTOCOL, "freeze_sha256": freeze_sha,
                                "validation_selection_lock_sha256": study.sha(lock_path),
                                "scores": scores})
    print(json.dumps({"score_audit_sha256": study.sha(output),
                      "fresh_replays": len(expected)}), flush=True)


def summarize():
    study.check_freeze()
    lock = json.loads((OUT / "VALIDATION_SELECTION_LOCK.json").read_text())
    audit = json.loads((OUT / "FINAL_SCORE_AUDIT.json").read_text())
    rows = []
    for role in ("selected", "default"):
        accuracies, candidates = {}, {}
        for arm in ARMS:
            candidate = tuple(lock["selections"][arm]["selected_candidate"]) if role == "selected" else DEFAULT
            candidates[arm] = candidate
            values = []
            for seed in SEEDS:
                identity = study.key(arm, *candidate, seed)
                score_path = OUT / "scores" / identity / "score.json"
                require(identity in audit["scores"] and
                        study.sha(score_path) == audit["scores"][identity]["score_sha256"],
                        f"Unaudited summary value: {identity}")
                values.append(json.loads(score_path.read_text())["test_accuracy"])
            accuracies[arm] = values
        rows.append({"role": role, "candidates": candidates,
                     "seed_test_accuracies": accuracies,
                     "mean_test_accuracy": {arm: statistics.mean(x) for arm, x in accuracies.items()},
                     "private_last_minus_base": statistics.mean(a-b for a,b in zip(accuracies["private_last"], accuracies["base"])),
                     "private_last_minus_ens": statistics.mean(a-b for a,b in zip(accuracies["private_last"], accuracies["ens"]))})
    output = OUT / "SUMMARY.json"
    require(not output.exists(), "Summary overwrite refused")
    primary.write_json(output, {"protocol": study.PROTOCOL, "freeze_sha256": study.sha(study.FREEZE),
                                "selection_lock_sha256": study.sha(OUT / "VALIDATION_SELECTION_LOCK.json"),
                                "score_audit_sha256": study.sha(OUT / "FINAL_SCORE_AUDIT.json"),
                                "rows": rows})
    print(json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("lock").add_argument("--device", default="cuda:0")
    command.add_parser("audit-scores").add_argument("--device", default="cuda:0")
    command.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "lock":
        lock(torch.device(args.device))
    elif args.command == "audit-scores":
        audit_scores(torch.device(args.device))
    else:
        summarize()


if __name__ == "__main__":
    main()
