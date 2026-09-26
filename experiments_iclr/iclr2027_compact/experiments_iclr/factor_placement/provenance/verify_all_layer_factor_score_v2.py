"""Independent, validation-first audit for the post hoc all-layer factor arm.

The lock step never loads test masks or labels. Test replay is a separate
command, gated by the complete immutable validation lock.
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

import all_layer_factor_study as arm
import tuning as primary


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "all_layer_factor_results"
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01),
              (0.001, 0.0), (0.001, 0.01),
              (0.003, 0.0), (0.003, 0.01))
DEFAULT = (0.001, 0.0)
EPOCHS = 1000
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now")
REPLAY_LOGIT_MAX_ABS_TOL = 1e-4
PRIMARY_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
CONTROL_DIR = ROOT / "same_runtime_tied36_results"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu().numpy()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.shape).encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_sha(tensor):
    value = np.ascontiguousarray(tensor.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def key(dataset, lr, wd, seed):
    return f"{dataset}/lr{lr:g}_wd{wd:g}/seed{seed}"


def cell_path(dataset, lr, wd, seed):
    return OUT / "results" / key(dataset, lr, wd, seed)


@torch.no_grad()
def metrics(model, bundle, indices, labels):
    model.eval()
    member = torch.stack([model(bundle.graph, bundle.x, tabm_seed=m)
                          .index_select(0, indices).detach().cpu()
                          for m in range(4)])
    pooled = member.mean(0)
    labels = labels.detach().cpu()
    return (float((pooled.argmax(-1) == labels).float().mean().item()),
            float(F.cross_entropy(pooled, labels).item()), pooled, member)


def audit_cell(dataset, lr, wd, seed, freeze_sha, bundle, device):
    identity = key(dataset, lr, wd, seed)
    folder = cell_path(dataset, lr, wd, seed)
    result_path = folder / "result.json"
    trace_path = folder / "validation_trace.csv"
    checkpoint_path = folder / "checkpoint.pt"
    predictions_path = folder / "validation_predictions.npz"
    require(result_path.is_file() and trace_path.is_file(), f"Missing cell: {identity}")
    result = json.loads(result_path.read_text())
    require(result["protocol"] == "all_layer_factor_placement_posthoc_v1" and
            result["freeze_sha256"] == freeze_sha and
            result["dataset"] == dataset and result["arm"] == "all_layer" and
            result["lr"] == lr and result["weight_decay"] == wd and
            result["seed"] == seed and result["epochs_required"] == EPOCHS and
            result["validation_trace_sha256"] == sha(trace_path),
            f"Cell identity/hash differs: {identity}")
    with trace_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == TRACE_FIELDS,
                f"Trace schema differs: {identity}")
        trace = list(reader)
    require(len(trace) == result["epochs_completed"] and len(trace) <= EPOCHS,
            f"Trace length differs: {identity}")
    best_acc, best_ce, best_epoch = -math.inf, math.inf, 0
    for epoch, record in enumerate(trace, 1):
        require(int(record["epoch"]) == epoch, f"Trace epoch differs: {identity}")
        acc, ce = float(record["valid_accuracy"]), float(record["valid_ce"])
        require(math.isfinite(acc) and 0 <= acc <= 1 and
                math.isfinite(ce) and ce >= 0, f"Invalid trace metric: {identity}")
        better = acc > best_acc or (acc == best_acc and ce < best_ce)
        require(int(record["selected_now"]) == int(better),
                f"Selection flag differs: {identity}")
        if better:
            best_acc, best_ce, best_epoch = acc, ce, epoch
    record = {"result_sha256": sha(result_path), "trace_sha256": sha(trace_path),
              "failure": result["failure"]}
    if result["failure"] is not None:
        require(len(trace) < EPOCHS and not checkpoint_path.exists() and
                not predictions_path.exists(),
                f"Failed cell has complete trace/checkpoint: {identity}")
        return record
    require(len(trace) == EPOCHS and checkpoint_path.is_file() and
            predictions_path.is_file() and
            result["selected_epoch"] == best_epoch and
            result["selected_valid_accuracy"] == best_acc and
            result["selected_valid_ce"] == best_ce and
            result["checkpoint_sha256"] == sha(checkpoint_path) and
            result["validation_predictions_sha256"] == sha(predictions_path),
            f"Selected checkpoint disagrees with validation trace: {identity}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["dataset"] == dataset and checkpoint["arm"] == "all_layer" and
            checkpoint["lr"] == lr and checkpoint["weight_decay"] == wd and
            checkpoint["seed"] == seed and checkpoint["epoch"] == best_epoch and
            checkpoint["freeze_sha256"] == freeze_sha and
            state_sha(checkpoint["state_dict"]) == result["selected_state_sha256"],
            f"Checkpoint identity/state differs: {identity}")
    primary.seed_all(seed)
    model = arm.make_model(bundle, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    va, vc, pooled, member = metrics(model, bundle, bundle.valid_idx, bundle.valid_y)
    with np.load(predictions_path, allow_pickle=False) as saved:
        require(set(saved.files) == {"pooled_logits", "member_logits",
                                     "valid_indices", "valid_labels"},
                f"Validation predictions schema differs: {identity}")
        recorded_pooled = saved["pooled_logits"]
        recorded_member = saved["member_logits"]
        require(np.array_equal(saved["valid_indices"], bundle.valid_idx.cpu().numpy()) and
                np.array_equal(saved["valid_labels"], bundle.valid_y.cpu().numpy()) and
                recorded_pooled.shape == tuple(pooled.shape) and
                recorded_member.shape == tuple(member.shape) and
                np.isfinite(recorded_pooled).all() and np.isfinite(recorded_member).all() and
                tensor_sha(torch.from_numpy(recorded_pooled)) ==
                result["selected_valid_pooled_logits_sha256"],
                f"Recorded validation predictions differ: {identity}")
        pooled_delta = float(np.max(np.abs(recorded_pooled - pooled.numpy())))
        member_delta = float(np.max(np.abs(recorded_member - member.numpy())))
        require(pooled_delta <= REPLAY_LOGIT_MAX_ABS_TOL and
                member_delta <= REPLAY_LOGIT_MAX_ABS_TOL and
                np.array_equal(recorded_pooled.argmax(-1), pooled.argmax(-1).numpy()) and
                np.array_equal(recorded_member.argmax(-1), member.argmax(-1).numpy()),
                f"Independent validation logit/decision replay differs: {identity}")
    require(abs(va - best_acc) <= 1e-7 and abs(vc - best_ce) <= 1e-5,
            f"Independent validation replay differs: {identity}")
    record.update({"checkpoint_sha256": sha(checkpoint_path),
                   "validation_predictions_sha256": sha(predictions_path),
                   "selected_epoch": best_epoch,
                   "selected_valid_accuracy": best_acc,
                   "selected_valid_ce": best_ce,
                   "replay_pooled_logit_max_abs_diff": pooled_delta,
                   "replay_member_logit_max_abs_diff": member_delta})
    return record


def candidate_choice(rows):
    eligible = [row for row in rows if row["valid"]]
    require(eligible, "No finite three-seed candidate")
    return max(eligible, key=lambda r: (r["mean_valid_accuracy"],
                                        -r["mean_valid_ce"], -r["lr"],
                                        -r["weight_decay"]))


def audit_and_lock(device):
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    require(not lock_path.exists(), "Refusing to replace all-layer selection lock")
    freeze_sha = arm.check_freeze()
    frozen = json.loads(arm.FREEZE.read_text())
    matrix = frozen["matrix"]
    require(frozen["posthoc_choice_after_primary_outcomes"] is True and
            tuple(matrix["datasets"]) == DATASETS and
            tuple(matrix["seeds"]) == SEEDS and
            tuple((x["lr"], x["weight_decay"]) for x in matrix["candidates"]) == CANDIDATES and
            tuple(matrix["default_candidate"]) == DEFAULT and
            matrix["epochs"] == EPOCHS and matrix["members"] == 4,
            "Frozen all-layer matrix differs from independent constants")
    primary_lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    primary_lock = json.loads(primary_lock_path.read_text())
    require(sha(primary_lock_path) == PRIMARY_LOCK_SHA and
            frozen["primary_selection_lock_sha256"] == PRIMARY_LOCK_SHA and
            len(primary_lock["cells"]) == 432 and
            primary_lock["freeze_sha256"] == primary.check_freeze(),
            "Primary global validation lock is incomplete")
    cells, selections = {}, {}
    for dataset in DATASETS:
        bundle, _ = primary.load_graph(dataset, device, include_test=False)
        rows = []
        for lr, wd in CANDIDATES:
            seeds = []
            for seed in SEEDS:
                identity = key(dataset, lr, wd, seed)
                cell = audit_cell(dataset, lr, wd, seed, freeze_sha, bundle, device)
                cells[identity] = cell
                seeds.append(cell)
            valid = all(s["failure"] is None for s in seeds)
            rows.append({"lr": lr, "weight_decay": wd, "valid": valid,
                         "mean_valid_accuracy": sum(s["selected_valid_accuracy"] for s in seeds) / 3 if valid else None,
                         "mean_valid_ce": sum(s["selected_valid_ce"] for s in seeds) / 3 if valid else None,
                         "seed_valid_accuracy": [s["selected_valid_accuracy"] for s in seeds] if valid else None,
                         "seed_selected_epoch": [s["selected_epoch"] for s in seeds] if valid else None})
        chosen = candidate_choice(rows)
        selections[dataset] = {"candidate_table": rows,
                               "selected_candidate": [chosen["lr"], chosen["weight_decay"]],
                               "selected_mean_valid_accuracy": chosen["mean_valid_accuracy"],
                               "selected_mean_valid_ce": chosen["mean_valid_ce"],
                               "selected_seed_valid_accuracy": chosen["seed_valid_accuracy"],
                               "selected_seed_epoch": chosen["seed_selected_epoch"]}
    require(len(cells) == 72 and len(selections) == 4, "Incomplete all-layer matrix")
    lock = {"protocol": "all_layer_factor_placement_posthoc_v1",
            "freeze_sha256": freeze_sha, "primary_selection_lock_sha256": sha(primary_lock_path),
            "cells": cells, "selections": selections,
            "selection_uses_test_labels": False,
            "test_scoring_allowlist": "selected candidate and predeclared (0.001,0) default only"}
    primary.write_json(lock_path, lock)
    print(json.dumps({"validation_lock_sha256": sha(lock_path),
                      "cells": len(cells), "selections": selections}), flush=True)


def audit_scores(device):
    freeze_sha = arm.check_freeze()
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    primary_lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    control_lock_path = CONTROL_DIR / "VALIDATION_SELECTION_LOCK.json"
    control_freeze_path = CONTROL_DIR / "FROZEN_STUDY.json"
    require(control_lock_path.is_file() and control_freeze_path.is_file(),
            "Same-runtime TIED36 selection lock missing")
    control_lock = json.loads(control_lock_path.read_text())
    require(lock["freeze_sha256"] == freeze_sha and len(lock["cells"]) == 72 and
            lock["primary_selection_lock_sha256"] == sha(primary_lock_path) ==
            PRIMARY_LOCK_SHA and
            control_lock["protocol"] == "same_runtime_tied36_v1" and
            len(control_lock["cells"]) == 36 and
            control_lock["freeze_sha256"] == sha(control_freeze_path) and
            control_lock["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA,
            "Validation lock or primary lock differs")
    expected = set()
    report = {"protocol": "all_layer_factor_placement_posthoc_v1",
              "freeze_sha256": freeze_sha,
              "validation_selection_lock_sha256": sha(lock_path), "scores": {}}
    for dataset in DATASETS:
        bundle, _ = primary.load_graph(dataset, device, include_test=True)
        selected = tuple(lock["selections"][dataset]["selected_candidate"])
        for lr, wd in dict.fromkeys((selected, DEFAULT)):
            for seed in SEEDS:
                identity = key(dataset, lr, wd, seed)
                expected.add(identity)
                folder = cell_path(dataset, lr, wd, seed)
                score_dir = OUT / "scores" / identity
                score_path = score_dir / "score.json"
                predictions_path = score_dir / "predictions.npz"
                require(score_path.is_file() and predictions_path.is_file(),
                        f"Missing score: {identity}")
                row = json.loads(score_path.read_text())
                locked = lock["cells"][identity]
                require(sha(folder / "result.json") == locked["result_sha256"] and
                        sha(folder / "checkpoint.pt") == locked["checkpoint_sha256"] and
                        row["freeze_sha256"] == freeze_sha and
                        row["validation_selection_lock_sha256"] == sha(lock_path) and
                        row["primary_selection_lock_sha256"] == sha(primary_lock_path) and
                        row["same_runtime_tied36_validation_lock_sha256"] == sha(control_lock_path) and
                        row["dataset"] == dataset and row["arm"] == "all_layer" and
                        row["lr"] == lr and row["weight_decay"] == wd and
                        row["seed"] == seed and
                        row["selected_candidate"] == ((lr, wd) == selected) and
                        row["predeclared_default"] == ((lr, wd) == DEFAULT) and
                        row["checkpoint_sha256"] == locked["checkpoint_sha256"] and
                        row["test_predictions_sha256"] == sha(predictions_path),
                        f"Score identity/hash differs: {identity}")
                primary.seed_all(seed)
                model = arm.make_model(bundle, device)
                checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                va, vc, vp, _ = metrics(model, bundle, bundle.valid_idx, bundle.valid_y)
                ta, tc, tp, tm = metrics(model, bundle, bundle.test_idx, bundle.test_y)
                with np.load(predictions_path, allow_pickle=False) as saved:
                    require(set(saved.files) == {"valid_pooled_logits", "test_pooled_logits", "test_member_logits"} and
                            np.allclose(saved["valid_pooled_logits"], vp.numpy(), rtol=1e-5, atol=1e-5) and
                            np.allclose(saved["test_pooled_logits"], tp.numpy(), rtol=1e-5, atol=1e-5) and
                            np.allclose(saved["test_member_logits"], tm.numpy(), rtol=1e-5, atol=1e-5) and
                            np.array_equal(saved["test_pooled_logits"].argmax(-1), tp.argmax(-1).numpy()) and
                            np.array_equal(saved["test_member_logits"].argmax(-1), tm.argmax(-1).numpy()),
                            f"Independent exact decision replay differs: {identity}")
                require(abs(va - row["valid_accuracy"]) <= 1e-7 and
                        abs(vc - row["valid_ce"]) <= 1e-5 and
                        abs(ta - row["test_accuracy"]) <= 1e-7 and
                        abs(tc - row["test_ce"]) <= 1e-5,
                        f"Independent metric replay differs: {identity}")
                report["scores"][identity] = {"score_sha256": sha(score_path),
                                              "predictions_sha256": sha(predictions_path),
                                              "test_accuracy": ta, "test_ce": tc}
    actual = {str(path.parent.relative_to(OUT / "scores"))
              for path in (OUT / "scores").rglob("score.json")}
    require(actual == expected, "Unauthorized or missing test score")
    audit_path = OUT / "FINAL_SCORE_AUDIT.json"
    require(not audit_path.exists(), "Refusing overwrite of final score audit")
    primary.write_json(audit_path, report)
    print(json.dumps({"score_audit_sha256": sha(audit_path),
                      "fresh_replays": len(expected)}), flush=True)


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
