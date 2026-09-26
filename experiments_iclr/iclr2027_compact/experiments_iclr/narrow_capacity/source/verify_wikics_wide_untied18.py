"""Independent validation lock and score replay for width-128 WikiCS UNTIED."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

import tuning as primary
import verify_tuning as original_auditor
import wikics_wide_untied18 as study
import narrow_untied72 as narrow


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "wikics_wide_untied18_results"
SEEDS = (0, 1, 2)
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01),
              (0.001, 0.0), (0.001, 0.01),
              (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)
PROTOCOL = "wikics_wide_untied18_same_runtime_v1"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def audit_cell(lr, wd, seed, freeze_sha, bundle, device):
    identity = study.key(lr, wd, seed)
    saved_root = original_auditor.ROOT
    try:
        original_auditor.ROOT = OUT
        row = original_auditor.audit_one("wikics", "untied", lr, wd, seed, freeze_sha)
    finally:
        original_auditor.ROOT = saved_root
    cell = study.cell_dir(lr, wd, seed)
    record = json.loads((cell / "result.json").read_text())
    require(record["initialization"]["parameter_count"] == 703232 and
            record["epochs_required"] == 1000,
            f"Wide WikiCS count or epoch requirement differs: {identity}")
    companion_path = cell / "validation_companion.npz"
    manifest_path = cell / "validation_companion.json"
    if record["failure"] is not None:
        require(not companion_path.exists() and not manifest_path.exists(),
                f"Failed wide cell has companion: {identity}")
        return row
    require(companion_path.is_file() and manifest_path.is_file(),
            f"Missing wide validation companion: {identity}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["protocol"] == PROTOCOL and manifest["lr"] == lr and
            manifest["weight_decay"] == wd and manifest["seed"] == seed and
            manifest["result_sha256"] == row["result_sha256"] and
            manifest["checkpoint_sha256"] == row["checkpoint_sha256"] and
            manifest["predictions_sha256"] == study.sha(companion_path) and
            manifest["selected_pooled_logits_sha256"] ==
            record["selected_valid_pooled_logits_sha256"],
            f"Wide validation companion identity differs: {identity}")
    primary.seed_all(seed)
    model, _ = primary.make_model("untied", bundle, device)
    payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    va, vc, pooled, members = original_auditor.independent_metrics(
        model, "untied", bundle, bundle.valid_idx, bundle.valid_y)
    require(abs(va - record["selected_valid_accuracy"]) <= 1e-7 and
            abs(vc - record["selected_valid_ce"]) <= 1e-5 and
            abs(va - manifest["valid_accuracy"]) <= 1e-7 and
            abs(vc - manifest["valid_ce"]) <= 1e-5,
            f"Independent wide validation metric differs: {identity}")
    with np.load(companion_path, allow_pickle=False) as saved:
        require(set(saved.files) == {"valid_indices", "valid_labels",
                                     "valid_pooled_logits", "valid_member_logits"} and
                np.array_equal(saved["valid_indices"], bundle.valid_idx.cpu().numpy()) and
                np.array_equal(saved["valid_labels"], bundle.valid_y.cpu().numpy()) and
                np.allclose(saved["valid_pooled_logits"], pooled.numpy(), rtol=1e-5, atol=1e-5) and
                np.allclose(saved["valid_member_logits"], members.numpy(), rtol=1e-5, atol=1e-5) and
                np.array_equal(saved["valid_pooled_logits"].argmax(-1), pooled.argmax(-1).numpy()) and
                np.array_equal(saved["valid_member_logits"].argmax(-1), members.argmax(-1).numpy()),
                f"Wide validation floats or decisions differ: {identity}")
    row.update({"validation_companion_sha256": study.sha(companion_path),
                "validation_companion_manifest_sha256": study.sha(manifest_path)})
    return row


def audit_and_lock(device):
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    require(not lock_path.exists(), "Refusing wide validation-lock overwrite")
    freeze_sha = study.check_freeze()
    narrow.original_locks()
    narrow_lock_path = narrow.OUT / "VALIDATION_SELECTION_LOCK.json"
    require(narrow_lock_path.is_file() and
            len(json.loads(narrow_lock_path.read_text())["cells"]) == 72,
            "Complete narrow72 validation lock required for linked wide18 lock")
    frozen = json.loads(study.FREEZE.read_text())
    matrix = frozen["matrix"]
    require(frozen["protocol"] == PROTOCOL and matrix["dataset"] == "wikics" and
            matrix["arm"] == "untied" and matrix["width"] == 128 and
            tuple(matrix["seeds"]) == SEEDS and matrix["epochs"] == 1000 and
            tuple((r["lr"], r["weight_decay"]) for r in matrix["candidates"]) == CANDIDATES and
            tuple(matrix["default_candidate"]) == DEFAULT,
            "Prospective wide WikiCS matrix differs")
    bundle, _ = primary.load_graph("wikics", device, include_test=False)
    cells, candidates = {}, []
    for lr, wd in CANDIDATES:
        rows = []
        for seed in SEEDS:
            row = audit_cell(lr, wd, seed, freeze_sha, bundle, device)
            cells[study.key(lr, wd, seed)] = row
            rows.append(row)
        valid = all(r["failure"] is None for r in rows)
        candidates.append({"lr": lr, "weight_decay": wd, "valid": valid,
                           "mean_valid_accuracy": sum(r["selected_valid_accuracy"] for r in rows) / 3 if valid else None,
                           "mean_valid_ce": sum(r["selected_valid_ce"] for r in rows) / 3 if valid else None,
                           "seed_valid_accuracy": [r["selected_valid_accuracy"] for r in rows] if valid else None,
                           "seed_selected_epoch": [r["selected_epoch"] for r in rows] if valid else None})
    valid = [r for r in candidates if r["valid"]]
    require(valid and len(cells) == 18, "Incomplete wide WikiCS validation matrix")
    chosen = max(valid, key=lambda r: (r["mean_valid_accuracy"], -r["mean_valid_ce"],
                                       -r["lr"], -r["weight_decay"]))
    selection = {"candidate_table": candidates,
                 "selected_candidate": [chosen["lr"], chosen["weight_decay"]],
                 "selected_mean_valid_accuracy": chosen["mean_valid_accuracy"],
                 "selected_mean_valid_ce": chosen["mean_valid_ce"],
                 "parameter_count": 703232, "width": 128}
    primary.write_json(lock_path, {"protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                                   "narrow_validation_lock_sha256": study.sha(narrow_lock_path),
                                   "original_validation_lock_sha256": narrow.PRIMARY_LOCK_SHA,
                                   "same_runtime_tied36_validation_lock_sha256": narrow.TIED36_LOCK_SHA,
                                   "cells": cells, "selection": selection,
                                   "selection_uses_test_labels": False,
                                   "test_scoring_allowlist": "selected and declared (0.001,0) default only"})
    print(json.dumps({"validation_lock_sha256": study.sha(lock_path),
                      "cells": len(cells), "selection": selection}), flush=True)


def audit_scores(device):
    freeze_sha = study.check_freeze()
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    narrow_lock_path = narrow.OUT / "VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    require(lock["protocol"] == PROTOCOL and lock["freeze_sha256"] == freeze_sha and
            len(lock["cells"]) == 18 and
            lock["narrow_validation_lock_sha256"] == study.sha(narrow_lock_path),
            "Wide WikiCS validation lock differs")
    bundle, _ = primary.load_graph("wikics", device, include_test=True)
    selected = tuple(lock["selection"]["selected_candidate"])
    expected, scores = set(), {}
    for lr, wd in dict.fromkeys((selected, DEFAULT)):
        for seed in SEEDS:
            identity = study.key(lr, wd, seed)
            expected.add(identity)
            cell = study.cell_dir(lr, wd, seed)
            folder = OUT / "scores" / identity
            score_path = folder / "score.json"
            prediction_path = folder / "predictions.npz"
            require(score_path.is_file() and prediction_path.is_file(),
                    f"Wide WikiCS selected/default score missing: {identity}")
            row = json.loads(score_path.read_text())
            frozen = lock["cells"][identity]
            require(study.sha(cell / "result.json") == frozen["result_sha256"] and
                    study.sha(cell / "checkpoint.pt") == frozen["checkpoint_sha256"] and
                    row["protocol"] == PROTOCOL and row["freeze_sha256"] == freeze_sha and
                    row["validation_selection_lock_sha256"] == study.sha(lock_path) and
                    row["narrow_validation_lock_sha256"] == study.sha(narrow_lock_path) and
                    row["dataset"] == "wikics" and row["arm"] == "untied" and
                    row["width"] == 128 and row["lr"] == lr and row["weight_decay"] == wd and
                    row["seed"] == seed and row["selected_candidate"] == ((lr, wd) == selected) and
                    row["predeclared_default"] == ((lr, wd) == DEFAULT) and
                    row["checkpoint_sha256"] == frozen["checkpoint_sha256"] and
                    row["test_predictions_sha256"] == study.sha(prediction_path),
                    f"Wide score identity/hash differs: {identity}")
            primary.seed_all(seed)
            model, _ = primary.make_model("untied", bundle, device)
            payload = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"], strict=True)
            va, vc, vp, _ = original_auditor.independent_metrics(
                model, "untied", bundle, bundle.valid_idx, bundle.valid_y)
            ta, tc, tp, tm = original_auditor.independent_metrics(
                model, "untied", bundle, bundle.test_idx, bundle.test_y)
            with np.load(prediction_path, allow_pickle=False) as saved:
                require(set(saved.files) == {"valid_pooled_logits", "test_pooled_logits", "test_member_logits"} and
                        np.allclose(saved["valid_pooled_logits"], vp.numpy(), rtol=1e-5, atol=1e-5) and
                        np.allclose(saved["test_pooled_logits"], tp.numpy(), rtol=1e-5, atol=1e-5) and
                        np.allclose(saved["test_member_logits"], tm.numpy(), rtol=1e-5, atol=1e-5) and
                        np.array_equal(saved["test_pooled_logits"].argmax(-1), tp.argmax(-1).numpy()) and
                        np.array_equal(saved["test_member_logits"].argmax(-1), tm.argmax(-1).numpy()),
                        f"Wide WikiCS test floats or decisions differ: {identity}")
            require(abs(va - row["valid_accuracy"]) <= 1e-7 and
                    abs(vc - row["valid_ce"]) <= 1e-5 and
                    abs(ta - row["test_accuracy"]) <= 1e-7 and
                    abs(tc - row["test_ce"]) <= 1e-5,
                    f"Wide WikiCS metric replay differs: {identity}")
            scores[identity] = {"score_sha256": study.sha(score_path),
                                "predictions_sha256": study.sha(prediction_path),
                                "test_accuracy": ta, "test_ce": tc}
    actual = {str(path.parent.relative_to(OUT / "scores"))
              for path in (OUT / "scores").rglob("score.json")}
    require(actual == expected and len(expected) <= 6,
            "Unauthorized or missing wide WikiCS score")
    output = OUT / "FINAL_SCORE_AUDIT.json"
    require(not output.exists(), "Refusing wide score-audit overwrite")
    primary.write_json(output, {"protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                                "validation_selection_lock_sha256": study.sha(lock_path),
                                "scores": scores})
    print(json.dumps({"score_audit_sha256": study.sha(output),
                      "fresh_replays": len(expected)}), flush=True)


def summarize():
    study.check_freeze()
    width_scan = json.loads(narrow.WIDTH_SCAN.read_text())["graphs"]["wikics"]
    original_preflight = json.loads((ROOT / "preflight.json").read_text())
    parameter_counts = {
        "wide128": original_preflight["graphs"]["wikics"]["parameter_counts"]["untied"],
        "narrow67": width_scan["selected_untied_parameters"],
        "tied128": width_scan["tied128_parameters"],
    }
    require(width_scan["selected_width"] == 67 and
            parameter_counts == {"wide128": 703232, "narrow67": 205228,
                                 "tied128": 207872}, "WikiCS parameter counts differ")
    wide_lock = json.loads((OUT / "VALIDATION_SELECTION_LOCK.json").read_text())
    narrow_lock = json.loads((narrow.OUT / "VALIDATION_SELECTION_LOCK.json").read_text())
    tied_lock = json.loads((ROOT / "same_runtime_tied36_results/VALIDATION_SELECTION_LOCK.json").read_text())
    wide_audit = json.loads((OUT / "FINAL_SCORE_AUDIT.json").read_text())
    narrow_audit = json.loads((narrow.OUT / "FINAL_SCORE_AUDIT.json").read_text())
    tied_audit = json.loads((ROOT / "same_runtime_tied36_results/FINAL_SCORE_AUDIT.json").read_text())
    require(len(wide_lock["cells"]) == 18 and len(narrow_lock["cells"]) == 72 and
            len(tied_lock["cells"]) == 36 and len(tied_audit["scores"]) == 12,
            "Linked validation/score audits incomplete")
    rows = []
    for setting in ("selected", "default"):
        candidates = {
            "wide128": tuple(wide_lock["selection"]["selected_candidate"]) if setting == "selected" else DEFAULT,
            "narrow67": tuple(narrow_lock["selections"]["wikics"]["selected_candidate"]) if setting == "selected" else DEFAULT,
            "tied128": tuple(tied_lock["selections"]["wikics"]["selected_candidate"]) if setting == "selected" else DEFAULT,
        }
        sources = {"wide128": (OUT, wide_audit), "narrow67": (narrow.OUT, narrow_audit),
                   "tied128": (ROOT / "same_runtime_tied36_results", tied_audit)}
        values = {}
        for arm in candidates:
            base, audit = sources[arm]
            acc = []
            for seed in SEEDS:
                identity = f"wikics/{'tied' if arm == 'tied128' else 'untied'}/{primary.candidate_name(*candidates[arm])}/seed{seed}"
                path = base / "scores" / identity / "score.json"
                require(identity in audit["scores"] and
                        study.sha(path) == audit["scores"][identity]["score_sha256"],
                        f"Audited comparison score missing: {arm}/{setting}/{seed}")
                acc.append(json.loads(path.read_text())["test_accuracy"])
            values[arm] = acc
        rows.append({"setting": setting, "candidates": candidates,
                     "seed_accuracies": values,
                     "means": {arm: statistics.mean(x) for arm, x in values.items()},
                     "wide_minus_narrow_mean": statistics.mean(
                         [a-b for a,b in zip(values["wide128"], values["narrow67"])]),
                     "wide_minus_tied_mean": statistics.mean(
                         [a-b for a,b in zip(values["wide128"], values["tied128"])])})
    output = OUT / "WIKICS_WIDTH_CONTRAST.json"
    require(not output.exists(), "Refusing wide contrast overwrite")
    primary.write_json(output, {"protocol": PROTOCOL, "rows": rows,
                                "parameter_counts": parameter_counts,
                                "interpretation": "post hoc same-runtime width sensitivity on one WikiCS split; width and capacity vary together",
                                "wide_score_audit_sha256": study.sha(OUT / "FINAL_SCORE_AUDIT.json"),
                                "narrow_score_audit_sha256": study.sha(narrow.OUT / "FINAL_SCORE_AUDIT.json"),
                                "tied_score_audit_sha256": study.sha(ROOT / "same_runtime_tied36_results/FINAL_SCORE_AUDIT.json")})
    print(json.dumps({"contrast_sha256": study.sha(output), "rows": rows}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-and-lock", "audit-scores", "summarize"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "audit-and-lock":
        audit_and_lock(torch.device(args.device))
    elif args.command == "audit-scores":
        audit_scores(torch.device(args.device))
    else:
        summarize()


if __name__ == "__main__":
    main()
