"""Independent validation lock, test replay, and paired summary for narrow UNTIED."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

import narrow_untied72 as study
import tuning as primary
import verify_tuning as original_auditor


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "narrow_untied72_results"
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01),
              (0.001, 0.0), (0.001, 0.01),
              (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)
EPOCHS = 1000
PROTOCOL = "narrow_untied72_capacity_sensitivity_v1"
PRIMARY_SCORE_AUDIT_SHA = "63da49d3d797fa7856717d4b21a530c74419bfbdfa92614a03b036aa9950bdde"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def audit_cell(dataset, lr, wd, seed, freeze_sha, bundle, device, width, count):
    identity = study.key(dataset, lr, wd, seed)
    saved_root = original_auditor.ROOT
    try:
        original_auditor.ROOT = OUT
        row = original_auditor.audit_one(dataset, "untied", lr, wd, seed, freeze_sha)
    finally:
        original_auditor.ROOT = saved_root
    folder = study.cell_dir(dataset, lr, wd, seed)
    result = json.loads((folder / "result.json").read_text())
    require(result["epochs_required"] == EPOCHS and
            result["initialization"]["parameter_count"] == count,
            f"Frozen narrow width/count or trace differs: {identity}")
    prediction_path = folder / "validation_companion.npz"
    manifest_path = folder / "validation_companion.json"
    if result["failure"] is not None:
        require(not prediction_path.exists() and not manifest_path.exists(),
                f"Failed cell has validation companion: {identity}")
        return row
    require(prediction_path.is_file() and manifest_path.is_file(),
            f"Complete validation companion missing: {identity}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["protocol"] == PROTOCOL and manifest["dataset"] == dataset and
            manifest["lr"] == lr and manifest["weight_decay"] == wd and
            manifest["seed"] == seed and manifest["width"] == width and
            manifest["result_sha256"] == row["result_sha256"] and
            manifest["checkpoint_sha256"] == row["checkpoint_sha256"] and
            manifest["predictions_sha256"] == study.sha(prediction_path) and
            manifest["selected_pooled_logits_sha256"] ==
            result["selected_valid_pooled_logits_sha256"],
            f"Validation companion provenance differs: {identity}")
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    primary.seed_all(seed)
    model, _ = primary.make_model("untied", bundle, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    va, vc, pooled, members = original_auditor.independent_metrics(
        model, "untied", bundle, bundle.valid_idx, bundle.valid_y)
    require(abs(va - result["selected_valid_accuracy"]) <= 1e-7 and
            abs(vc - result["selected_valid_ce"]) <= 1e-5 and
            abs(va - manifest["valid_accuracy"]) <= 1e-7 and
            abs(vc - manifest["valid_ce"]) <= 1e-5,
            f"Selected validation checkpoint replay differs: {identity}")
    with np.load(prediction_path, allow_pickle=False) as saved:
        require(set(saved.files) == {"valid_indices", "valid_labels",
                                     "valid_pooled_logits", "valid_member_logits"} and
                np.array_equal(saved["valid_indices"], bundle.valid_idx.cpu().numpy()) and
                np.array_equal(saved["valid_labels"], bundle.valid_y.cpu().numpy()) and
                np.allclose(saved["valid_pooled_logits"], pooled.numpy(), rtol=1e-5, atol=1e-5) and
                np.allclose(saved["valid_member_logits"], members.numpy(), rtol=1e-5, atol=1e-5) and
                np.array_equal(saved["valid_pooled_logits"].argmax(-1), pooled.argmax(-1).numpy()) and
                np.array_equal(saved["valid_member_logits"].argmax(-1), members.argmax(-1).numpy()),
                f"Bounded validation floats or hard decisions differ: {identity}")
    row.update({"validation_companion_sha256": study.sha(prediction_path),
                "validation_companion_manifest_sha256": study.sha(manifest_path)})
    return row


def audit_and_lock(device):
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    require(not lock_path.exists(), "Refusing narrow validation-lock overwrite")
    freeze_sha = study.check_freeze()
    study.original_locks()
    frozen = json.loads(study.FREEZE.read_text())
    matrix = frozen["matrix"]
    require(frozen["protocol"] == PROTOCOL and
            frozen["original_validation_lock_sha256"] == study.PRIMARY_LOCK_SHA and
            frozen["same_runtime_tied36_validation_lock_sha256"] == study.TIED36_LOCK_SHA and
            tuple(matrix["datasets"]) == DATASETS and matrix["arm"] == "untied" and
            tuple(matrix["seeds"]) == SEEDS and matrix["epochs"] == EPOCHS and
            tuple((r["lr"], r["weight_decay"]) for r in matrix["candidates"]) == CANDIDATES and
            tuple(matrix["default_candidate"]) == DEFAULT,
            "Prospective narrow matrix or comparator lock differs")
    scan = json.loads(study.WIDTH_SCAN.read_text())
    selected_widths = study.widths()
    cells, selections = {}, {}
    old_width = primary.WIDTH
    try:
        for dataset in DATASETS:
            bundle, _ = primary.load_graph(dataset, device, include_test=False)
            width = selected_widths[dataset]
            count = scan["graphs"][dataset]["selected_untied_parameters"]
            primary.WIDTH = width
            candidates = []
            for lr, wd in CANDIDATES:
                seeds = []
                for seed in SEEDS:
                    row = audit_cell(dataset, lr, wd, seed, freeze_sha, bundle, device, width, count)
                    cells[study.key(dataset, lr, wd, seed)] = row
                    seeds.append(row)
                valid = all(row["failure"] is None for row in seeds)
                candidates.append({"lr": lr, "weight_decay": wd, "valid": valid,
                                   "mean_valid_accuracy": sum(r["selected_valid_accuracy"] for r in seeds) / 3 if valid else None,
                                   "mean_valid_ce": sum(r["selected_valid_ce"] for r in seeds) / 3 if valid else None,
                                   "seed_valid_accuracy": [r["selected_valid_accuracy"] for r in seeds] if valid else None,
                                   "seed_selected_epoch": [r["selected_epoch"] for r in seeds] if valid else None})
            valid = [row for row in candidates if row["valid"]]
            require(valid, f"All narrow candidates failed: {dataset}")
            chosen = max(valid, key=lambda r: (r["mean_valid_accuracy"], -r["mean_valid_ce"],
                                               -r["lr"], -r["weight_decay"]))
            selections[dataset] = {"candidate_table": candidates,
                                   "selected_candidate": [chosen["lr"], chosen["weight_decay"]],
                                   "selected_mean_valid_accuracy": chosen["mean_valid_accuracy"],
                                   "selected_mean_valid_ce": chosen["mean_valid_ce"],
                                   "width": width, "parameter_count": count}
    finally:
        primary.WIDTH = old_width
    require(len(cells) == 72 and len(selections) == 4, "Incomplete narrow matrix")
    lock = {"protocol": PROTOCOL, "freeze_sha256": freeze_sha,
            "width_scan_sha256": study.sha(study.WIDTH_SCAN),
            "original_validation_lock_sha256": study.PRIMARY_LOCK_SHA,
            "same_runtime_tied36_validation_lock_sha256": study.TIED36_LOCK_SHA,
            "cells": cells, "selections": selections,
            "selection_uses_test_labels": False,
            "test_scoring_allowlist": "selected candidate and declared (0.001,0) default only"}
    primary.write_json(lock_path, lock)
    print(json.dumps({"validation_lock_sha256": study.sha(lock_path),
                      "cells": len(cells), "selections": selections}), flush=True)


def audit_scores(device):
    freeze_sha = study.check_freeze()
    study.original_locks()
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    require(lock["protocol"] == PROTOCOL and lock["freeze_sha256"] == freeze_sha and
            len(lock["cells"]) == 72 and set(lock["selections"]) == set(DATASETS) and
            lock["original_validation_lock_sha256"] == study.PRIMARY_LOCK_SHA and
            lock["same_runtime_tied36_validation_lock_sha256"] == study.TIED36_LOCK_SHA,
            "Complete validation lock differs")
    expected, scores = set(), {}
    selected_widths = study.widths()
    old_width = primary.WIDTH
    try:
        for dataset in DATASETS:
            bundle, _ = primary.load_graph(dataset, device, include_test=True)
            primary.WIDTH = selected_widths[dataset]
            selected = tuple(lock["selections"][dataset]["selected_candidate"])
            for lr, wd in dict.fromkeys((selected, DEFAULT)):
                for seed in SEEDS:
                    identity = study.key(dataset, lr, wd, seed)
                    expected.add(identity)
                    cell = study.cell_dir(dataset, lr, wd, seed)
                    folder = OUT / "scores" / identity
                    score_path = folder / "score.json"
                    prediction_path = folder / "predictions.npz"
                    require(score_path.is_file() and prediction_path.is_file(),
                            f"Selected/default narrow score missing: {identity}")
                    row = json.loads(score_path.read_text())
                    frozen = lock["cells"][identity]
                    require(study.sha(cell / "result.json") == frozen["result_sha256"] and
                            study.sha(cell / "checkpoint.pt") == frozen["checkpoint_sha256"] and
                            row["protocol"] == PROTOCOL and row["freeze_sha256"] == freeze_sha and
                            row["validation_selection_lock_sha256"] == study.sha(lock_path) and
                            row["original_validation_lock_sha256"] == study.PRIMARY_LOCK_SHA and
                            row["same_runtime_tied36_validation_lock_sha256"] == study.TIED36_LOCK_SHA and
                            row["dataset"] == dataset and row["arm"] == "untied" and
                            row["width"] == primary.WIDTH and row["lr"] == lr and
                            row["weight_decay"] == wd and row["seed"] == seed and
                            row["selected_candidate"] == ((lr, wd) == selected) and
                            row["predeclared_default"] == ((lr, wd) == DEFAULT) and
                            row["checkpoint_sha256"] == frozen["checkpoint_sha256"] and
                            row["test_predictions_sha256"] == study.sha(prediction_path),
                            f"Narrow score identity/hash differs: {identity}")
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
                                f"Narrow test logits/decisions replay differs: {identity}")
                    require(abs(va - row["valid_accuracy"]) <= 1e-7 and
                            abs(vc - row["valid_ce"]) <= 1e-5 and
                            abs(ta - row["test_accuracy"]) <= 1e-7 and
                            abs(tc - row["test_ce"]) <= 1e-5,
                            f"Narrow test metrics replay differs: {identity}")
                    scores[identity] = {"score_sha256": study.sha(score_path),
                                        "predictions_sha256": study.sha(prediction_path),
                                        "test_accuracy": ta, "test_ce": tc}
    finally:
        primary.WIDTH = old_width
    actual = {str(path.parent.relative_to(OUT / "scores"))
              for path in (OUT / "scores").rglob("score.json")}
    require(actual == expected and len(expected) <= 24,
            "Unauthorized, missing, or duplicate narrow test score")
    output = OUT / "FINAL_SCORE_AUDIT.json"
    require(not output.exists(), "Refusing narrow score-audit overwrite")
    primary.write_json(output, {"protocol": PROTOCOL, "freeze_sha256": freeze_sha,
                                "validation_selection_lock_sha256": study.sha(lock_path),
                                "scores": scores})
    print(json.dumps({"score_audit_sha256": study.sha(output),
                      "fresh_replays": len(expected)}), flush=True)


def summarize() -> None:
    study.check_freeze()
    study.original_locks()
    original_score_audit_path = ROOT / "FINAL_SCORE_AUDIT.json"
    require(study.sha(original_score_audit_path) == PRIMARY_SCORE_AUDIT_SHA,
            "Original 141-score audit differs")
    all_audit_path = OUT / "FINAL_SCORE_AUDIT.json"
    tied_audit_path = ROOT / "same_runtime_tied36_results/FINAL_SCORE_AUDIT.json"
    require(all_audit_path.is_file() and tied_audit_path.is_file(),
            "Complete narrow and TIED36 score audits required")
    narrow_lock = json.loads((OUT / "VALIDATION_SELECTION_LOCK.json").read_text())
    narrow_audit = json.loads(all_audit_path.read_text())
    original_lock = json.loads((ROOT / "VALIDATION_SELECTION_LOCK.json").read_text())
    original_audit = json.loads(original_score_audit_path.read_text())
    tied_lock = json.loads((ROOT / "same_runtime_tied36_results/VALIDATION_SELECTION_LOCK.json").read_text())
    tied_audit = json.loads(tied_audit_path.read_text())
    require(len(narrow_lock["cells"]) == 72 and len(original_lock["cells"]) == 432 and
            len(tied_lock["cells"]) == 36 and
            len(narrow_audit["scores"]) <= 24 and len(original_audit["scores"]) == 141 and
            len(tied_audit["scores"]) == 12,
            "Comparison audit completeness differs")
    report = []
    for dataset in DATASETS:
        same = dataset in ("cora", "wikics")
        comparator_lock = tied_lock if same else original_lock
        comparator_audit = tied_audit if same else original_audit
        comparator_root = ROOT / "same_runtime_tied36_results" if same else ROOT
        comparator_choice = comparator_lock["selections"][dataset]
        if not same:
            comparator_choice = comparator_choice["tied"]
        for setting in ("selected", "default"):
            narrow_candidate = (tuple(narrow_lock["selections"][dataset]["selected_candidate"])
                                if setting == "selected" else DEFAULT)
            tied_candidate = (tuple(comparator_choice["selected_candidate"])
                              if setting == "selected" else DEFAULT)
            n_acc, t_acc, differences = [], [], []
            for seed in SEEDS:
                n_key = study.key(dataset, *narrow_candidate, seed)
                t_key = f"{dataset}/tied/{primary.candidate_name(*tied_candidate)}/seed{seed}"
                require(n_key in narrow_audit["scores"] and t_key in comparator_audit["scores"],
                        f"Audited paired scores missing: {dataset}/{setting}/seed{seed}")
                n_path = OUT / "scores" / n_key / "score.json"
                t_path = comparator_root / "scores" / t_key / "score.json"
                require(study.sha(n_path) == narrow_audit["scores"][n_key]["score_sha256"] and
                        study.sha(t_path) == comparator_audit["scores"][t_key]["score_sha256"],
                        "Paired score bytes differ")
                n_value = json.loads(n_path.read_text())["test_accuracy"]
                t_value = json.loads(t_path.read_text())["test_accuracy"]
                n_acc.append(n_value)
                t_acc.append(t_value)
                differences.append(n_value - t_value)
            report.append({"dataset": dataset, "setting": setting,
                           "comparator": "same_runtime_tied36" if same else "original_same_runtime_tied",
                           "narrow_candidate": narrow_candidate,
                           "tied_candidate": tied_candidate,
                           "width": study.widths()[dataset],
                           "narrow_parameter_count": narrow_lock["selections"][dataset]["parameter_count"],
                           "tied128_parameter_count": json.loads(study.WIDTH_SCAN.read_text())["graphs"][dataset]["tied128_parameters"],
                           "narrow_seed_accuracy": n_acc, "tied_seed_accuracy": t_acc,
                           "narrow_mean": statistics.mean(n_acc), "tied_mean": statistics.mean(t_acc),
                           "paired_difference_mean": statistics.mean(differences),
                           "paired_difference_sample_sd": statistics.stdev(differences)})
    require(len(report) == 8, "Incomplete four-graph paired summary")
    output = OUT / "CAPACITY_SENSITIVITY_COMPARISON.json"
    require(not output.exists(), "Refusing comparison overwrite")
    primary.write_json(output, {"protocol": PROTOCOL, "rows": report,
                                "interpretation": "post hoc width-and-capacity sensitivity, three seeds on one split per graph; no causal capacity isolation",
                                "narrow_final_score_audit_sha256": study.sha(all_audit_path),
                                "original_final_score_audit_sha256": PRIMARY_SCORE_AUDIT_SHA,
                                "same_runtime_tied36_final_score_audit_sha256": study.sha(tied_audit_path)})
    print(json.dumps({"summary_sha256": study.sha(output), "rows": report}), flush=True)


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
