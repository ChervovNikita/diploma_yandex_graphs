"""Independent validation lock and allowed-score replay for the mechanism study."""
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

import mechanism as study
import tuning as base


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("tied", "untied", "sync")
SEEDS = (0, 1, 2)
DEFAULT = (0.001, 0.0)
SYNC_CANDIDATES = ((0.0003, 0.0), (0.0003, 0.01),
                   (0.001, 0.0), (0.001, 0.01),
                   (0.003, 0.0), (0.003, 0.01))
EPOCHS = 1000
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now", "graph_weights_equal")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu().numpy()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.shape).encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def cell_key(dataset, arm, lr, wd, seed):
    return f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"


def cell_path(dataset, arm, lr, wd, seed):
    return ROOT / "results" / cell_key(dataset, arm, lr, wd, seed)


def graph_stack_state_equal(state: dict) -> bool:
    first = "propagation_stacks.0."
    names = [name for name in state if name.startswith(first)]
    if not names:
        return False
    return all(all(torch.equal(state[name], state[f"propagation_stacks.{member}." + name[len(first):]])
                   for member in range(1, 4)) for name in names)


def expected_collapsed_state(state: dict) -> dict:
    require(graph_stack_state_equal(state), "SYNC graph copies differ in checkpoint")
    result = {}
    for name, tensor in state.items():
        if name.startswith("propagation_stacks.0."):
            result["residual_modules." + name[len("propagation_stacks.0."):]] = tensor
        elif name.startswith("propagation_stacks."):
            continue
        else:
            result[name] = tensor
    return result


def audit_one(dataset, arm, lr, wd, seed, freeze_sha):
    key = cell_key(dataset, arm, lr, wd, seed)
    folder = cell_path(dataset, arm, lr, wd, seed)
    result_path = folder / "result.json"
    trace_path = folder / "validation_trace.csv"
    require(result_path.is_file() and trace_path.is_file(), f"Missing mechanism cell {key}")
    result = json.loads(result_path.read_text())
    require(result["protocol"] == "optimizer_aggregation_mechanism_v1" and
            result["freeze_sha256"] == freeze_sha and
            result["dataset"] == dataset and result["arm"] == arm and
            result["seed"] == seed and result["learning_rate"] == lr and
            result["weight_decay"] == wd and result["epochs_required"] == EPOCHS,
            f"Cell identity/configuration mismatch: {key}")
    require(result["validation_trace_sha256"] == sha(trace_path), f"Trace digest changed: {key}")
    with trace_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == TRACE_FIELDS, f"Trace schema mismatch: {key}")
        rows = list(reader)
    require(len(rows) == result["epochs_completed"] and len(rows) <= EPOCHS,
            f"Trace length mismatch: {key}")
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    for epoch, row in enumerate(rows, 1):
        require(int(row["epoch"]) == epoch, f"Epoch sequence mismatch: {key}")
        acc, ce = float(row["valid_accuracy"]), float(row["valid_ce"])
        require(math.isfinite(acc) and math.isfinite(ce) and 0 <= acc <= 1 and ce >= 0,
                f"Invalid validation metric: {key}")
        better = acc > best_acc or (acc == best_acc and ce < best_ce)
        require(int(row["selected_now"]) == int(better), f"Checkpoint flag mismatch: {key}")
        require(row["graph_weights_equal"] == ("1" if arm == "sync" else ""),
                f"Graph weight equality gate not recorded: {key}")
        if better:
            best_acc, best_ce, best_epoch = acc, ce, epoch
    initial = result["initialization"]
    require(initial["canonical_projector_state_sha256"] is not None and
            initial["paired_initial_logits_max_abs_diff"] <= 1e-5,
            f"Matched initialization gate failed: {key}")
    cell = {"result_sha256": sha(result_path), "trace_sha256": sha(trace_path),
            "failure": result["failure"], "initialization": initial}
    checkpoint_path = folder / "checkpoint.pt"
    if result["failure"] is not None:
        require(len(rows) < EPOCHS and not checkpoint_path.exists(),
                f"Failed cell has a complete trace or checkpoint: {key}")
        return cell
    require(len(rows) == EPOCHS and checkpoint_path.is_file(), f"Incomplete finite cell: {key}")
    require(result["selected_epoch"] == best_epoch and
            result["selected_valid_accuracy"] == best_acc and
            result["selected_valid_ce"] == best_ce,
            f"Selected result differs from validation trace: {key}")
    require(result["checkpoint_sha256"] == sha(checkpoint_path),
            f"Checkpoint bytes changed: {key}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["dataset"] == dataset and checkpoint["arm"] == arm and
            checkpoint["seed"] == seed and checkpoint["learning_rate"] == lr and
            checkpoint["weight_decay"] == wd and checkpoint["epoch"] == best_epoch and
            checkpoint["freeze_sha256"] == freeze_sha,
            f"Checkpoint metadata mismatch: {key}")
    state = checkpoint["state_dict"]
    require(state_sha(state) == result["selected_state_sha256"],
            f"Checkpoint tensor state changed: {key}")
    cell.update({"checkpoint_sha256": sha(checkpoint_path),
                 "selected_epoch": best_epoch, "selected_valid_accuracy": best_acc,
                 "selected_valid_ce": best_ce})
    if arm == "sync":
        moments = result["first_step_private_moments"]
        require(moments is not None and moments["separate_first_moment_tensors"] > 0 and
                moments["separate_first_moment_tensors"] == moments["separate_second_moment_tensors"] and
                (moments["first_moment_max_abs_member_difference"] > 0 or
                 moments["second_moment_max_abs_member_difference"] > 0),
                f"Private Adam state gate failed: {key}")
        expected = expected_collapsed_state(state)
        collapsed_path = folder / "collapsed_checkpoint.pt"
        require(collapsed_path.is_file() and
                result["collapsed_checkpoint_sha256"] == sha(collapsed_path),
                f"Missing or changed collapsed checkpoint: {key}")
        collapsed = torch.load(collapsed_path, map_location="cpu", weights_only=True)
        require(collapsed["source_checkpoint_sha256"] == sha(checkpoint_path) and
                collapsed["freeze_sha256"] == freeze_sha and
                collapsed["dataset"] == dataset and collapsed["seed"] == seed and
                set(collapsed["state_dict"]) == set(expected) and
                all(torch.equal(collapsed["state_dict"][name], tensor)
                    for name, tensor in expected.items()),
                f"Collapsed checkpoint differs from equal-weight state: {key}")
        require(result["collapse_validation"]["collapsed_state_sha256"] == state_sha(expected) and
                result["collapse_validation"]["max_abs_validation_member_logit_difference"] <= 1e-5 and
                result["collapse_validation"]["pooled_validation_decision_mismatches"] == 0,
                f"Collapsed validation gate failed: {key}")
        cell["collapsed_checkpoint_sha256"] = sha(collapsed_path)
    return cell


def candidate_choice(table):
    valid = [row for row in table if row["valid"]]
    require(valid, "All six SYNC candidates invalid")
    return max(valid, key=lambda row: (row["mean_valid_accuracy"],
                                       -row["mean_valid_ce"], -row["lr"], -row["weight_decay"]))


def audit_and_lock():
    lock_path = ROOT / "MECHANISM_VALIDATION_LOCK.json"
    require(not lock_path.exists(), "Mechanism lock exists; refusing overwrite")
    freeze_sha = study.check_freeze()
    frozen = json.loads((ROOT / "MECHANISM_FREEZE.json").read_text())
    matrix = frozen["matrix"]
    require(tuple(matrix["datasets"]) == DATASETS and tuple(matrix["arms"]) == ARMS and
            tuple(matrix["seeds"]) == SEEDS and matrix["epochs"] == EPOCHS and
            matrix["planned_cells"] == 96 and
            tuple((row["lr"], row["weight_decay"]) for row in matrix["sync_candidates"]) == SYNC_CANDIDATES and
            tuple((row["lr"], row["weight_decay"]) for row in matrix["control_candidates"]) == (DEFAULT,),
            "Mechanism freeze matrix differs from independent constants")
    cells, sync_selections = {}, {}
    for dataset in DATASETS:
        sync_selections[dataset] = {}
        for arm in ARMS:
            candidates = SYNC_CANDIDATES if arm == "sync" else (DEFAULT,)
            table = []
            for lr, wd in candidates:
                seed_rows = []
                for seed in SEEDS:
                    key = cell_key(dataset, arm, lr, wd, seed)
                    row = audit_one(dataset, arm, lr, wd, seed, freeze_sha)
                    cells[key] = row
                    seed_rows.append(row)
                valid = all(row["failure"] is None for row in seed_rows)
                table.append({"lr": lr, "weight_decay": wd, "valid": valid,
                              "mean_valid_accuracy": (sum(row["selected_valid_accuracy"] for row in seed_rows) / 3
                                                      if valid else None),
                              "mean_valid_ce": (sum(row["selected_valid_ce"] for row in seed_rows) / 3
                                                if valid else None),
                              "seed_valid_accuracy": ([row["selected_valid_accuracy"] for row in seed_rows]
                                                      if valid else None)})
            if arm != "sync":
                require(table[0]["valid"], f"Default control incomplete: {dataset}/{arm}")
            else:
                chosen = candidate_choice(table)
                sync_selections[dataset] = {
                    "candidate_table": table,
                    "selected_candidate": [chosen["lr"], chosen["weight_decay"]],
                    "selected_mean_valid_accuracy": chosen["mean_valid_accuracy"],
                    "selected_mean_valid_ce": chosen["mean_valid_ce"],
                    "selected_seed_valid_accuracy": chosen["seed_valid_accuracy"],
                }
        for seed in SEEDS:
            reference = cells[cell_key(dataset, "tied", *DEFAULT, seed)]["initialization"]
            for arm in ARMS:
                for lr, wd in (SYNC_CANDIDATES if arm == "sync" else (DEFAULT,)):
                    initial = cells[cell_key(dataset, arm, lr, wd, seed)]["initialization"]
                    for field in ("canonical_projector_state_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
                        require(initial[field] == reference[field],
                                f"Canonical initialization/RNG differs: {dataset}/{arm}/{seed}/{field}")
    require(len(cells) == 96 and len(sync_selections) == 4,
            "Incomplete 96-cell mechanism validation matrix")
    lock = {"protocol": "optimizer_aggregation_mechanism_v1", "freeze_sha256": freeze_sha,
            "base_432_freeze_sha256": sha(ROOT / "FROZEN_STUDY.json"),
            "cells": cells, "sync_selections": sync_selections,
            "test_scoring_allowlist": "default TIED/UNTIED and selected/default SYNC only",
            "selection_uses_test_labels": False}
    study.write_json(lock_path, lock)
    print(json.dumps({"mechanism_lock_sha256": sha(lock_path),
                      "validated_cells": len(cells), "sync_graph_selections": 4}), flush=True)


@torch.no_grad()
def fresh_metrics(model, arm, bundle, idx, labels):
    model.eval()
    members = torch.stack([value.index_select(0, idx) for value in
                           base.member_logits(model, study.arm_for_base(arm), bundle)]).detach().cpu()
    pooled = members.mean(0)
    labels = labels.detach().cpu()
    return (float((pooled.argmax(-1) == labels).float().mean().item()),
            float(F.cross_entropy(pooled, labels).item()), pooled, members)


def audit_scores(device):
    freeze_sha = study.check_freeze()
    lock_path = ROOT / "MECHANISM_VALIDATION_LOCK.json"
    base_lock_path = ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    base_lock = json.loads(base_lock_path.read_text())
    require(lock["freeze_sha256"] == freeze_sha and len(lock["cells"]) == 96 and
            base_lock["freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            len(base_lock["cells"]) == 432,
            "Both complete validation locks required")
    expected, report = set(), {"freeze_sha256": freeze_sha,
                               "mechanism_validation_lock_sha256": sha(lock_path),
                               "original_432_validation_lock_sha256": sha(base_lock_path),
                               "scores": {}}
    for dataset in DATASETS:
        bundle, _ = base.load_graph(dataset, device, include_test=True)
        selected = tuple(lock["sync_selections"][dataset]["selected_candidate"])
        for arm in ARMS:
            candidates = dict.fromkeys((selected, DEFAULT)) if arm == "sync" else (DEFAULT,)
            for lr, wd in candidates:
                for seed in SEEDS:
                    key = cell_key(dataset, arm, lr, wd, seed)
                    expected.add(key)
                    folder = cell_path(dataset, arm, lr, wd, seed)
                    score_dir = ROOT / "scores" / key
                    scored = json.loads((score_dir / "score.json").read_text())
                    frozen_cell = lock["cells"][key]
                    require(sha(folder / "result.json") == frozen_cell["result_sha256"] and
                            sha(folder / "checkpoint.pt") == frozen_cell["checkpoint_sha256"] and
                            sha(score_dir / "predictions.npz") == scored["predictions_sha256"],
                            f"Score/cell hash changed: {key}")
                    require(scored["freeze_sha256"] == freeze_sha and
                            scored["mechanism_validation_lock_sha256"] == sha(lock_path) and
                            scored["original_432_validation_lock_sha256"] == sha(base_lock_path) and
                            scored["dataset"] == dataset and scored["arm"] == arm and
                            scored["seed"] == seed and scored["learning_rate"] == lr and
                            scored["weight_decay"] == wd and
                            scored["checkpoint_sha256"] == frozen_cell["checkpoint_sha256"],
                            f"Score metadata mismatch: {key}")
                    base.seed_all(seed)
                    model, _ = study.make_model(arm, bundle, device)
                    payload = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                    model.load_state_dict(payload["state_dict"], strict=True)
                    va, vc, vp, _ = fresh_metrics(model, arm, bundle, bundle.valid_idx, bundle.valid_y)
                    ta, tc, tp, tm = fresh_metrics(model, arm, bundle, bundle.test_idx, bundle.test_y)
                    if arm == "sync":
                        compact, _ = base.make_model("tied", bundle, device)
                        compact.load_state_dict(expected_collapsed_state(payload["state_dict"]), strict=True)
                        _, _, compact_pooled, compact_members = fresh_metrics(
                            compact, "tied", bundle, bundle.test_idx, bundle.test_y)
                        collapse_diff = float((tm - compact_members).abs().max().item())
                        collapse_decisions = int((tp.argmax(-1) != compact_pooled.argmax(-1)).sum().item())
                        require(collapse_diff <= 1e-5 and collapse_decisions == 0 and
                                scored["collapsed_test_replay"]["max_abs_member_logit_difference"] <= 1e-5 and
                                scored["collapsed_test_replay"]["decision_mismatches"] == 0,
                                f"Independent collapsed test replay failed: {key}")
                    else:
                        require(scored["collapsed_test_replay"] is None,
                                f"Unexpected collapsed score for control: {key}")
                    with np.load(score_dir / "predictions.npz", allow_pickle=False) as saved:
                        old_valid, old_test, old_members = (saved["valid_pooled_logits"],
                                                            saved["test_pooled_logits"],
                                                            saved["test_member_logits"])
                    require(np.allclose(old_valid, vp.numpy(), atol=1e-5, rtol=1e-5) and
                            np.allclose(old_test, tp.numpy(), atol=1e-5, rtol=1e-5) and
                            np.allclose(old_members, tm.numpy(), atol=1e-5, rtol=1e-5) and
                            np.array_equal(old_valid.argmax(-1), vp.numpy().argmax(-1)) and
                            np.array_equal(old_test.argmax(-1), tp.numpy().argmax(-1)) and
                            abs(va - scored["valid_accuracy"]) <= 1e-7 and
                            abs(vc - scored["valid_ce"]) <= 1e-5 and
                            abs(ta - scored["test_accuracy"]) <= 1e-7 and
                            abs(tc - scored["test_ce"]) <= 1e-5,
                            f"Fresh score/decision replay failed: {key}")
                    report["scores"][key] = {"test_accuracy": ta, "test_ce": tc,
                                              "score_sha256": sha(score_dir / "score.json"),
                                              "predictions_sha256": sha(score_dir / "predictions.npz")}
    actual = {str(path.parent.relative_to(ROOT / "scores"))
              for path in (ROOT / "scores").rglob("score.json")}
    require(actual == expected, "Test scores exist outside selected/default allowlist")
    study.write_json(ROOT / "MECHANISM_FINAL_SCORE_AUDIT.json", report)
    print(json.dumps({"fresh_replayed_scores": len(report["scores"]),
                      "report_sha256": sha(ROOT / "MECHANISM_FINAL_SCORE_AUDIT.json")}), flush=True)


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
