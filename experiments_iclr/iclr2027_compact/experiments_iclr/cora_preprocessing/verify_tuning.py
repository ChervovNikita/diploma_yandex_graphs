"""Independent validation-only audit and post-lock replay for matched Cora inputs.

`audit-and-lock` reads no test labels or predictions. `audit-scores` runs only
after the global lock and evaluates only selected/default checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import tuning as study


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora_raw", "cora_normalized")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
SEEDS = (0, 1, 2)
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01),
              (0.001, 0.0), (0.001, 0.01),
              (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)
EPOCHS = 1000
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().contiguous().cpu().numpy()
        h.update(key.encode() + b"\0")
        h.update(str(value.shape).encode() + b"\0")
        h.update(str(value.dtype).encode() + b"\0")
        h.update(value.tobytes())
    return h.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def cell_key(dataset, arm, lr, wd, seed):
    return f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"


def cell_dir(dataset, arm, lr, wd, seed):
    return ROOT / "results" / cell_key(dataset, arm, lr, wd, seed)


def audit_one(dataset, arm, lr, wd, seed, freeze_sha, bundle, device):
    key = cell_key(dataset, arm, lr, wd, seed)
    folder = cell_dir(dataset, arm, lr, wd, seed)
    result_path, trace_path = folder / "result.json", folder / "validation_trace.csv"
    require(result_path.is_file() and trace_path.is_file(), f"Missing completed cell: {key}")
    result = json.loads(result_path.read_text())
    require(result["protocol"] == "validation_tuning_sensitivity_v1" and
            result["freeze_sha256"] == freeze_sha and
            result["dataset"] == dataset and result["arm"] == arm and
            result["lr"] == lr and result["weight_decay"] == wd and
            result["seed"] == seed and result["epochs_required"] == EPOCHS,
            f"Identity/configuration mismatch: {key}")
    require(result["validation_trace_sha256"] == sha(trace_path), f"Trace digest mismatch: {key}")
    with trace_path.open(newline="") as f:
        reader = csv.DictReader(f)
        require(tuple(reader.fieldnames or ()) == TRACE_FIELDS, f"Trace schema changed: {key}")
        rows = list(reader)
    require(len(rows) == result["epochs_completed"] and len(rows) <= EPOCHS,
            f"Trace length mismatch: {key}")
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    for epoch, row in enumerate(rows, 1):
        require(int(row["epoch"]) == epoch, f"Nonsequential epoch: {key}")
        acc, ce = float(row["valid_accuracy"]), float(row["valid_ce"])
        require(math.isfinite(acc) and math.isfinite(ce) and 0 <= acc <= 1 and ce >= 0,
                f"Nonfinite/invalid validation metric in trace: {key}")
        better = acc > best_acc or (acc == best_acc and ce < best_ce)
        require(int(row["selected_now"]) == int(better), f"Checkpoint flag mismatch: {key}")
        if better:
            best_acc, best_ce, best_epoch = acc, ce, epoch
    cell = {"result_sha256": sha(result_path), "trace_sha256": sha(trace_path),
            "failure": result["failure"]}
    checkpoint_path = folder / "checkpoint.pt"
    if result["failure"] is None:
        require(len(rows) == EPOCHS and checkpoint_path.is_file(), f"Incomplete finite cell: {key}")
        require(result["selected_epoch"] == best_epoch and
                result["selected_valid_accuracy"] == best_acc and
                result["selected_valid_ce"] == best_ce,
                f"Selected validation result disagrees with trace: {key}")
        require(result["checkpoint_sha256"] == sha(checkpoint_path),
                f"Checkpoint bytes differ: {key}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(checkpoint["epoch"] == best_epoch and checkpoint["dataset"] == dataset and
                checkpoint["arm"] == arm and checkpoint["lr"] == lr and
                checkpoint["weight_decay"] == wd and checkpoint["seed"] == seed and
                checkpoint["freeze_sha256"] == freeze_sha,
                f"Checkpoint identity mismatch: {key}")
        require(state_sha(checkpoint["state_dict"]) == result["selected_state_sha256"],
                f"Checkpoint state differs: {key}")
        study.seed_all(seed)
        model, _ = study.make_model(arm, bundle, device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        replay_acc, replay_ce, pooled, members = independent_metrics(
            model, arm, bundle, bundle.valid_idx, bundle.valid_y)
        require(abs(replay_acc - result["selected_valid_accuracy"]) <= 1e-7 and
                abs(replay_ce - result["selected_valid_ce"]) <= 1e-5,
                f"Fresh selected validation replay differs: {key}")
        decisions_path = folder / "selected_validation_decisions.npz"
        pooled_decision = pooled.argmax(-1).numpy().astype(np.uint8)
        member_decision = members.argmax(-1).numpy().astype(np.uint8)
        if decisions_path.exists():
            with np.load(decisions_path, allow_pickle=False) as stored:
                require(np.array_equal(stored["pooled"], pooled_decision) and
                        np.array_equal(stored["members"], member_decision),
                        f"Stored validation decisions differ on replay: {key}")
        else:
            np.savez_compressed(decisions_path,
                                pooled=pooled_decision, members=member_decision)
        cell.update({"checkpoint_sha256": sha(checkpoint_path),
                     "selected_epoch": best_epoch, "selected_valid_accuracy": best_acc,
                     "selected_valid_ce": best_ce,
                     "fresh_validation_replay_accuracy": replay_acc,
                     "fresh_validation_replay_ce": replay_ce,
                     "selected_validation_decisions_sha256": sha(decisions_path)})
    else:
        require(len(rows) < EPOCHS and not checkpoint_path.exists(),
                f"Invalid run has a complete trace or checkpoint: {key}")
    return cell


def select_candidate(rows):
    valid = [r for r in rows if r["valid"]]
    if not valid:
        return None
    return max(valid, key=lambda r: (r["mean_valid_accuracy"],
                                     -r["mean_valid_ce"], -r["lr"], -r["weight_decay"]))


def audit_and_lock(device: torch.device):
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    require(not lock_path.exists(), "Selection lock exists; refusing overwrite")
    freeze_sha = study.check_freeze()
    frozen = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    matrix = frozen["matrix"]
    require(tuple(matrix["datasets"]) == DATASETS and tuple(matrix["arms"]) == ARMS and
            tuple(matrix["seeds"]) == SEEDS and matrix["epochs"] == EPOCHS and
            tuple((r["lr"], r["weight_decay"]) for r in matrix["candidates"]) == CANDIDATES,
            "Freeze matrix differs from independent audit constants")
    cells = {}
    selections, partial_choice = {}, {}
    for dataset in DATASETS:
        bundle, _ = study.load_graph(dataset, device, include_test=False)
        require(not hasattr(bundle, "test_idx") and not hasattr(bundle, "test_y"),
                f"Test labels exposed before lock: {dataset}")
        selections[dataset] = {}
        for arm in ARMS:
            candidates = []
            for lr, wd in CANDIDATES:
                seed_rows = []
                for seed in SEEDS:
                    key = cell_key(dataset, arm, lr, wd, seed)
                    row = audit_one(dataset, arm, lr, wd, seed, freeze_sha,
                                    bundle, device)
                    cells[key] = row
                    seed_rows.append(row)
                valid = all(row["failure"] is None for row in seed_rows)
                candidates.append({
                    "lr": lr, "weight_decay": wd, "valid": valid,
                    "mean_valid_accuracy": (sum(row["selected_valid_accuracy"] for row in seed_rows) / 3
                                            if valid else None),
                    "mean_valid_ce": (sum(row["selected_valid_ce"] for row in seed_rows) / 3
                                      if valid else None),
                    "seed_valid_accuracy": ([row["selected_valid_accuracy"] for row in seed_rows]
                                            if valid else None),
                    "seed_selected_epoch": ([row["selected_epoch"] for row in seed_rows]
                                            if valid else None),
                })
            chosen = select_candidate(candidates)
            require(chosen is not None, f"All candidates failed for {dataset}/{arm}")
            selections[dataset][arm] = {
                "candidate_table": candidates,
                "selected_candidate": [chosen["lr"], chosen["weight_decay"]],
                "selected_mean_valid_accuracy": chosen["mean_valid_accuracy"],
                "selected_mean_valid_ce": chosen["mean_valid_ce"],
                "selected_seed_valid_accuracy": chosen["seed_valid_accuracy"],
                "selected_seed_epoch": chosen["seed_selected_epoch"],
            }
        first = selections[dataset]["private_first"]
        last = selections[dataset]["private_last"]
        if first["selected_mean_valid_accuracy"] > last["selected_mean_valid_accuracy"]:
            chosen_arm = "private_first"
        elif first["selected_mean_valid_accuracy"] < last["selected_mean_valid_accuracy"]:
            chosen_arm = "private_last"
        elif first["selected_mean_valid_ce"] < last["selected_mean_valid_ce"]:
            chosen_arm = "private_first"
        else:
            chosen_arm = "private_last"  # Includes exact remaining tie.
        partial_choice[dataset] = {
            "selected_arm": chosen_arm,
            "selection_basis": "higher tuned mean validation accuracy, then lower CE, then private_last",
            "family_configurations_searched": 12,
            "individual_comparator_configurations_searched": 6,
            "private_first_mean_valid_accuracy": first["selected_mean_valid_accuracy"],
            "private_last_mean_valid_accuracy": last["selected_mean_valid_accuracy"],
        }
    require(len(cells) == 216 and sum(len(v) for v in selections.values()) == 12,
            "Full validation matrix incomplete")
    lock = {"protocol": "cora_feature_normalization_sensitivity_v1", "freeze_sha256": freeze_sha,
            "cells": cells, "selections": selections, "partial_family_choice": partial_choice,
            "fresh_validation_replay": "every finite selected checkpoint; exact accuracy within 1e-7 and cross-entropy within 1e-5",
            "test_scoring_allowlist": "only each arm's selected candidate and predeclared (0.001,0) default",
            "selection_uses_test_labels": False}
    study.write_json(lock_path, lock)
    print(json.dumps({"lock_sha256": sha(lock_path), "cells": len(cells),
                      "groups": 12, "partial_family_choice": partial_choice}), flush=True)


@torch.no_grad()
def independent_metrics(model, arm, bundle, idx, labels):
    model.eval()
    if arm == "base":
        logits = (model(bundle.graph, bundle.x),)
    elif arm == "ens":
        logits = tuple(member(bundle.graph, bundle.x) for member in model)
    else:
        logits = tuple(model(bundle.graph, bundle.x, tabm_seed=m) for m in range(4))
    members = torch.stack([x.index_select(0, idx) for x in logits]).detach().cpu()
    pooled = members.mean(0)
    labels = labels.detach().cpu()
    acc = float((pooled.argmax(-1) == labels).float().mean().item())
    ce = float(F.cross_entropy(pooled, labels).item())
    return acc, ce, pooled, members


def audit_scores(device: torch.device):
    freeze_sha = study.check_freeze()
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    require(lock["freeze_sha256"] == freeze_sha and len(lock["cells"]) == 216,
            "Invalid global selection lock")
    lock_sha = sha(lock_path)
    expected_keys = set()
    report = {"protocol": "cora_feature_normalization_sensitivity_v1", "freeze_sha256": freeze_sha,
              "selection_lock_sha256": lock_sha, "scores": {}}
    for dataset in DATASETS:
        bundle, _ = study.load_graph(dataset, device, include_test=True)
        for arm in ARMS:
            selected = tuple(lock["selections"][dataset][arm]["selected_candidate"])
            for lr, wd in dict.fromkeys((selected, DEFAULT)):
                for seed in SEEDS:
                    key = cell_key(dataset, arm, lr, wd, seed)
                    expected_keys.add(key)
                    cell = cell_dir(dataset, arm, lr, wd, seed)
                    score_dir = ROOT / "scores" / key
                    row = json.loads((score_dir / "score.json").read_text())
                    predictions_path = score_dir / "predictions.npz"
                    locked_cell = lock["cells"][key]
                    require(sha(cell / "result.json") == locked_cell["result_sha256"] and
                            sha(cell / "checkpoint.pt") == locked_cell["checkpoint_sha256"],
                            f"Cell changed after lock: {key}")
                    require(row["freeze_sha256"] == freeze_sha and
                            row["validation_selection_lock_sha256"] == lock_sha and
                            row["dataset"] == dataset and row["arm"] == arm and
                            row["lr"] == lr and row["weight_decay"] == wd and row["seed"] == seed and
                            row["checkpoint_sha256"] == locked_cell["checkpoint_sha256"] and
                            row["test_predictions_sha256"] == sha(predictions_path),
                            f"Score metadata/hash mismatch: {key}")
                    study.seed_all(seed)
                    model, _ = study.make_model(arm, bundle, device)
                    checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
                    model.load_state_dict(checkpoint["state_dict"], strict=True)
                    va, vc, vp, _ = independent_metrics(
                        model, arm, bundle, bundle.valid_idx, bundle.valid_y)
                    ta, tc, tp, tm = independent_metrics(
                        model, arm, bundle, bundle.test_idx, bundle.test_y)
                    with np.load(predictions_path, allow_pickle=False) as saved:
                        require(np.allclose(saved["valid_pooled_logits"], vp.numpy(), rtol=1e-5, atol=1e-5) and
                                np.allclose(saved["test_pooled_logits"], tp.numpy(), rtol=1e-5, atol=1e-5) and
                                np.allclose(saved["test_member_logits"], tm.numpy(), rtol=1e-5, atol=1e-5),
                                f"Fresh prediction replay mismatch: {key}")
                    require(abs(va - row["valid_accuracy"]) <= 1e-7 and
                            abs(vc - row["valid_ce"]) <= 1e-5 and
                            abs(ta - row["test_accuracy"]) <= 1e-7 and
                            abs(tc - row["test_ce"]) <= 1e-5,
                            f"Fresh metric replay mismatch: {key}")
                    report["scores"][key] = {"test_accuracy": ta, "test_ce": tc,
                                             "score_sha256": sha(score_dir / "score.json"),
                                             "predictions_sha256": sha(predictions_path)}
    actual = {str(path.parent.relative_to(ROOT / "scores"))
              for path in (ROOT / "scores").rglob("score.json")}
    require(actual == expected_keys, "Undeclared/nondefault candidate test score exists")
    study.write_json(ROOT / "FINAL_SCORE_AUDIT.json", report)
    print(json.dumps({"fresh_replays": len(report["scores"]),
                      "audit_sha256": sha(ROOT / "FINAL_SCORE_AUDIT.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit-and-lock", "audit-scores"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "audit-and-lock":
        audit_and_lock(torch.device(args.device))
    else:
        audit_scores(torch.device(args.device))


if __name__ == "__main__":
    main()
