"""Frozen optimizer-aggregation diagnostic with fresh TIED/UNTIED controls.

SYNC uses equal graph weights at every forward but separate per-member AdamW
moments. Training never receives test indices or labels. Test scoring requires
both this study's complete 96-cell lock and the original 432-cell lock.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import tuning as base


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("tied", "untied", "sync")
SEEDS = (0, 1, 2)
MEMBERS = 4
EPOCHS = 1000
DEFAULT = (0.001, 0.0)
CANDIDATES = base.CANDIDATES
SGD_STEPS = 5
SGD_PARAM_TOL = 1e-4
SGD_LOGIT_TOL = 1e-4
COLLAPSE_LOGIT_TOL = 1e-5
SOURCE_FILES = ("mechanism.py", "verify_mechanism.py", "MECHANISM_PROTOCOL.md")
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now", "graph_weights_equal")


def sha(path: Path) -> str:
    return base.sha256_file(path)


def write_json(path: Path, value: dict) -> None:
    base.write_json(path, value)


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def configurations(arm: str):
    require(arm in ARMS, f"Unknown arm {arm}")
    return CANDIDATES if arm == "sync" else (DEFAULT,)


def cell_dir(dataset: str, arm: str, lr: float, wd: float, seed: int) -> Path:
    require((lr, wd) in configurations(arm), f"Undeclared candidate: {arm}, {lr}, {wd}")
    return ROOT / "results" / dataset / arm / base.candidate_name(lr, wd) / f"seed{seed}"


def arm_for_base(arm: str) -> str:
    require(arm in ARMS, f"Unknown arm {arm}")
    return "untied" if arm == "sync" else arm


def make_model(arm: str, bundle, device):
    return base.make_model(arm_for_base(arm), bundle, device)


def graph_stack_parameters(model):
    require(hasattr(model, "propagation_stacks") and
            len(model.propagation_stacks) == MEMBERS, "Expected four private graph stacks")
    groups = [list(stack.parameters()) for stack in model.propagation_stacks]
    require(groups and len(groups[0]) > 0 and all(len(x) == len(groups[0]) for x in groups),
            "Mismatched private graph stack structure")
    require(all(len(list(stack.buffers())) == 0 for stack in model.propagation_stacks),
            "Unexpected mutable graph stack buffer")
    return groups


def assert_equal_graph_weights(model) -> None:
    groups = graph_stack_parameters(model)
    flattened = [torch.cat([parameter.detach().reshape(-1) for parameter in group])
                 for group in groups]
    require(all(torch.equal(flattened[0], other) for other in flattened[1:]),
            "SYNC graph weights differ at a forward boundary")


@torch.no_grad()
def synchronize_graph_weights(model) -> None:
    groups = graph_stack_parameters(model)
    flattened = [torch.cat([parameter.detach().reshape(-1) for parameter in group])
                 for group in groups]
    average = torch.stack(flattened, dim=0).mean(0)
    if not bool(torch.isfinite(average).all()):
        raise RuntimeError("Nonfinite graph parameter after optimizer step")
    offset = 0
    for params in zip(*groups):
        size = params[0].numel()
        value = average[offset:offset + size].view_as(params[0])
        for parameter in params:
            parameter.copy_(value)
        offset += size
    require(offset == average.numel(), "SYNC flattening accounting changed")
    assert_equal_graph_weights(model)


def scale_private_graph_gradients(model) -> None:
    for group in graph_stack_parameters(model):
        for parameter in group:
            require(parameter.grad is not None, "Missing private graph gradient")
            parameter.grad.mul_(MEMBERS)


@torch.no_grad()
def moment_audit(model, optimizer) -> dict:
    groups = graph_stack_parameters(model)
    storage_first, storage_second = [], []
    difference_first, difference_second = 0.0, 0.0
    for params in zip(*groups):
        first = [optimizer.state[p]["exp_avg"] for p in params]
        second = [optimizer.state[p]["exp_avg_sq"] for p in params]
        storage_first.extend(v.untyped_storage().data_ptr() for v in first)
        storage_second.extend(v.untyped_storage().data_ptr() for v in second)
        difference_first = max(difference_first, *(
            float((first[0] - v).abs().max().item()) for v in first[1:]))
        difference_second = max(difference_second, *(
            float((second[0] - v).abs().max().item()) for v in second[1:]))
    require(len(storage_first) == len(set(storage_first)) and
            len(storage_second) == len(set(storage_second)),
            "SYNC optimizer moments share storage")
    require(difference_first > 0 or difference_second > 0,
            "Separate AdamW moments did not diverge")
    return {"first_moment_max_abs_member_difference": difference_first,
            "second_moment_max_abs_member_difference": difference_second,
            "separate_first_moment_tensors": len(storage_first),
            "separate_second_moment_tensors": len(storage_second)}


def collapsed_state(model) -> dict:
    assert_equal_graph_weights(model)
    source = model.state_dict()
    result = {}
    for name, value in source.items():
        if name.startswith("propagation_stacks.0."):
            result["residual_modules." + name[len("propagation_stacks.0."):]] = value
        elif name.startswith("propagation_stacks."):
            continue
        else:
            result[name] = value
    return result


def collapse_model(model, bundle, device):
    collapsed, _ = base.make_model("tied", bundle, device)
    collapsed.load_state_dict(collapsed_state(model), strict=True)
    return collapsed


@torch.no_grad()
def collapse_validation_audit(model, bundle, device) -> dict:
    assert_equal_graph_weights(model)
    collapsed = collapse_model(model, bundle, device)
    model.eval()
    collapsed.eval()
    original = torch.stack(base.member_logits(model, "untied", bundle))[:, bundle.valid_idx]
    compact = torch.stack(base.member_logits(collapsed, "tied", bundle))[:, bundle.valid_idx]
    max_difference = float((original - compact).abs().max().item())
    decision_mismatches = int((original.mean(0).argmax(-1) !=
                               compact.mean(0).argmax(-1)).sum().item())
    require(max_difference <= COLLAPSE_LOGIT_TOL and decision_mismatches == 0,
            "Collapsed graph stack changes validation logits or decisions")
    return {"max_abs_validation_member_logit_difference": max_difference,
            "pooled_validation_decision_mismatches": decision_mismatches,
            "collapsed_parameter_count": sum(p.numel() for p in collapsed.parameters()),
            "collapsed_state_sha256": base.state_sha(collapsed.state_dict())}


def one_update(model, arm: str, bundle, optimizer):
    if arm == "sync":
        assert_equal_graph_weights(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for logits in base.member_logits(model, arm_for_base(arm), bundle):
        loss = F.cross_entropy(logits.index_select(0, bundle.train_idx), bundle.train_y)
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        (loss / MEMBERS).backward()
    if arm == "sync":
        scale_private_graph_gradients(model)
    optimizer.step()
    if arm == "sync":
        synchronize_graph_weights(model)


def rng_snapshot(device):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None)


def rng_restore(value, device):
    random.setstate(value[0])
    np.random.set_state(value[1])
    torch.set_rng_state(value[2])
    if value[3] is not None:
        torch.cuda.set_rng_state(value[3], device)


def rng_equal(a, b) -> bool:
    return (a[0] == b[0] and a[1][0] == b[1][0] and
            np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:] and
            torch.equal(a[2], b[2]) and
            ((a[3] is None and b[3] is None) or
             (a[3] is not None and b[3] is not None and torch.equal(a[3], b[3]))))


def paired_max_parameter_difference(tied, sync):
    compact = collapsed_state(sync)
    baseline = tied.state_dict()
    require(set(compact) == set(baseline), "Collapsed state keys differ from TIED")
    return max(float((baseline[key] - compact[key]).abs().max().item())
               for key in baseline)


@torch.no_grad()
def paired_max_logit_difference(tied, sync, bundle):
    tied.eval()
    sync.eval()
    a = torch.stack(base.member_logits(tied, "tied", bundle))
    b = torch.stack(base.member_logits(sync, "untied", bundle))
    return float((a - b).abs().max().item())


def sgd_equivalence_smoke(bundle, device) -> dict:
    base.seed_all(73)
    tied, tied_sha = base.make_model("tied", bundle, device)
    tied_rng = rng_snapshot(device)
    base.seed_all(73)
    sync, sync_sha = base.make_model("untied", bundle, device)
    sync_rng = rng_snapshot(device)
    require(tied_sha == sync_sha and rng_equal(tied_rng, sync_rng),
            "SGD smoke constructors differ in canonical state or RNG")
    require(paired_max_parameter_difference(tied, sync) == 0.0,
            "SGD smoke initial states differ")
    opt_tied = torch.optim.SGD(tied.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    opt_sync = torch.optim.SGD(sync.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    parameter_diffs, logit_diffs = [], []
    for step in range(1, SGD_STEPS + 1):
        before = rng_snapshot(device)
        one_update(tied, "tied", bundle, opt_tied)
        tied_after = rng_snapshot(device)
        rng_restore(before, device)
        one_update(sync, "sync", bundle, opt_sync)
        sync_after = rng_snapshot(device)
        require(rng_equal(tied_after, sync_after),
                f"Dropout RNG differs at SGD smoke step {step}")
        parameter_diff = paired_max_parameter_difference(tied, sync)
        logit_diff = paired_max_logit_difference(tied, sync, bundle)
        require(parameter_diff <= SGD_PARAM_TOL and logit_diff <= SGD_LOGIT_TOL,
                f"SGD identity failed at step {step}: {parameter_diff}, {logit_diff}")
        parameter_diffs.append(parameter_diff)
        logit_diffs.append(logit_diff)
        rng_restore(tied_after, device)
    return {"steps": SGD_STEPS, "parameter_max_abs_differences": parameter_diffs,
            "member_logit_max_abs_differences": logit_diffs,
            "parameter_tolerance": SGD_PARAM_TOL, "logit_tolerance": SGD_LOGIT_TOL}


def adam_moment_smoke(bundle, device) -> dict:
    base.seed_all(91)
    tied, tied_sha = base.make_model("tied", bundle, device)
    tied_rng = rng_snapshot(device)
    base.seed_all(91)
    sync, sync_sha = base.make_model("untied", bundle, device)
    sync_rng = rng_snapshot(device)
    require(tied_sha == sync_sha and rng_equal(tied_rng, sync_rng),
            "Adam smoke constructors differ")
    opt_tied = torch.optim.AdamW(tied.parameters(), lr=DEFAULT[0], weight_decay=DEFAULT[1])
    opt_sync = torch.optim.AdamW(sync.parameters(), lr=DEFAULT[0], weight_decay=DEFAULT[1])
    one_update(tied, "tied", bundle, opt_tied)
    rng_restore(tied_rng, device)
    one_update(sync, "sync", bundle, opt_sync)
    moments = moment_audit(sync, opt_sync)
    moments["tied_vs_sync_graph_and_boundary_state_max_abs_difference_after_one_step"] = (
        paired_max_parameter_difference(tied, sync))
    return moments


def missing_gradient_scale_negative_control(bundle, device) -> dict:
    """A deliberately wrong SGD step must fail the tied-update identity."""
    base.seed_all(109)
    tied, tied_sha = base.make_model("tied", bundle, device)
    tied_rng = rng_snapshot(device)
    base.seed_all(109)
    wrong, wrong_sha = base.make_model("untied", bundle, device)
    require(tied_sha == wrong_sha and rng_equal(tied_rng, rng_snapshot(device)),
            "Negative-control constructors differ")
    opt_tied = torch.optim.SGD(tied.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    opt_wrong = torch.optim.SGD(wrong.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    one_update(tied, "tied", bundle, opt_tied)
    rng_restore(tied_rng, device)
    assert_equal_graph_weights(wrong)
    wrong.train()
    opt_wrong.zero_grad(set_to_none=True)
    for logits in base.member_logits(wrong, "untied", bundle):
        loss = F.cross_entropy(logits.index_select(0, bundle.train_idx), bundle.train_y)
        (loss / MEMBERS).backward()
    # Deliberately omit the ×4 graph-gradient correction here only.
    opt_wrong.step()
    synchronize_graph_weights(wrong)
    difference = paired_max_parameter_difference(tied, wrong)
    require(difference > SGD_PARAM_TOL,
            "Deliberately missing ×4 correction was not detected by SGD gate")
    return {"deliberately_omitted_scale": MEMBERS,
            "tied_vs_wrong_one_step_parameter_max_abs_difference": difference,
            "minimum_required_difference": SGD_PARAM_TOL}


def expected_freeze() -> dict:
    base_sha = base.check_freeze()
    original = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    return {
        "protocol": "optimizer_aggregation_mechanism_v1",
        "base_validation_grid_freeze_sha256": base_sha,
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "graphs": original["graphs"],
        "matrix": {"datasets": DATASETS, "arms": ARMS, "seeds": SEEDS,
                   "control_candidates": [{"lr": DEFAULT[0], "weight_decay": DEFAULT[1]}],
                   "sync_candidates": [{"lr": lr, "weight_decay": wd} for lr, wd in CANDIDATES],
                   "planned_cells": 96,
                   "epochs": EPOCHS, "optimizer": "AdamW",
                   "objective": "mean of four member cross-entropies",
                   "sync_graph_gradient_scale": MEMBERS,
                   "sync_rule": "each graph copy AdamW on member gradient, then mean/copy corresponding graph parameters after every update",
                   "shared_boundary_and_output_rule": "mean-loss gradients, one shared optimizer state",
                   "checkpoint_rule": "highest pooled validation accuracy; lowest pooled CE; earliest epoch",
                   "sync_candidate_rule": "highest three-seed mean selected validation accuracy; lowest mean CE; lower LR; lower decay",
                   "nonfinite_rule": "any nonfinite seed invalidates its three-seed SYNC candidate; failed control or all failed SYNC candidates makes comparison incomplete",
                   "test_rule": "all 96 cells locked AND separate complete 432-cell lock; then default TIED/UNTIED and selected/default SYNC only",
                   "sgd_smoke_steps": SGD_STEPS,
                   "sgd_parameter_tolerance": SGD_PARAM_TOL,
                   "sgd_logit_tolerance": SGD_LOGIT_TOL,
                   "initial_member_logit_tolerance": base.INITIAL_LOGIT_TOL,
                   "collapse_member_logit_tolerance": COLLAPSE_LOGIT_TOL,
                   "complete_study_audit_cutoff_utc": "2026-09-26T06:00:00Z"},
    }


def check_freeze() -> str:
    path = ROOT / "MECHANISM_FREEZE.json"
    require(path.is_file(), "Missing prospective mechanism freeze")
    expected = json.loads(json.dumps(expected_freeze(), sort_keys=True, allow_nan=False))
    require(json.loads(path.read_text()) == expected,
            "Mechanism source, original graph/data, or matrix differs from freeze")
    return sha(path)


def preflight(device):
    freeze_sha = check_freeze()
    bundle, _ = base.load_graph("cora", device, include_test=False)
    initial = {}
    for arm in ARMS:
        base.seed_all(0)
        model, canonical = make_model(arm, bundle, device)
        initial[arm] = base.initial_audit(model, arm_for_base(arm), bundle, 0, device, canonical)
    require(len({initial[a]["canonical_projector_state_sha256"] for a in ARMS}) == 1,
            "Preflight canonical initial states differ")
    require(initial["untied"]["parameter_count"] == initial["sync"]["parameter_count"],
            "SYNC training parameter count differs from UNTIED")
    sgd = sgd_equivalence_smoke(bundle, device)
    wrong_scale = missing_gradient_scale_negative_control(bundle, device)
    adam = adam_moment_smoke(bundle, device)
    report = {"freeze_sha256": freeze_sha, "device": str(device),
              "initial": initial, "sgd_equivalence": sgd,
              "missing_scale_negative_control": wrong_scale,
              "adam_private_moments": adam,
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    write_json(ROOT / f"PREFLIGHT_{str(device).replace(':', '_').upper()}.json", report)
    print(json.dumps({"preflight": "passed", "device": str(device),
                      "freeze_sha256": freeze_sha,
                      "sgd_max_parameter_difference": max(sgd["parameter_max_abs_differences"]),
                      "sgd_max_logit_difference": max(sgd["member_logit_max_abs_differences"])}), flush=True)


def train_one(dataset, bundle, freeze_sha, arm, lr, wd, seed, device):
    final = cell_dir(dataset, arm, lr, wd, seed)
    if final.exists():
        result_path = final / "result.json"
        trace_path = final / "validation_trace.csv"
        require(result_path.is_file() and trace_path.is_file(), f"Incomplete existing cell {final}")
        old = json.loads(result_path.read_text())
        require(old["freeze_sha256"] == freeze_sha and old["dataset"] == dataset and
                old["arm"] == arm and old["seed"] == seed and
                old["learning_rate"] == lr and old["weight_decay"] == wd and
                old["validation_trace_sha256"] == sha(trace_path),
                f"Existing cell fails hash/config gate {final}")
        if old["failure"] is None:
            require(old["epochs_completed"] == EPOCHS and
                    old["checkpoint_sha256"] == sha(final / "checkpoint.pt"),
                    f"Existing finite cell has incomplete checkpoint: {final}")
        return "skipped"
    work = final.with_name(final.name + ".inprogress")
    require(not work.exists(), f"Interrupted cell needs inspection: {work}")
    work.mkdir(parents=True)
    base.seed_all(seed)
    model, canonical = make_model(arm, bundle, device)
    initial = base.initial_audit(model, arm_for_base(arm), bundle, seed, device, canonical)
    if arm == "sync":
        assert_equal_graph_weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    best_state = None
    trace = []
    first_moments = None
    failure = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        if arm == "sync":
            assert_equal_graph_weights(model)
        try:
            one_update(model, arm, bundle, optimizer)
        except RuntimeError as error:
            if str(error) not in ("Nonfinite training loss", "Nonfinite graph parameter after optimizer step"):
                raise
            failure = f"{str(error).lower()} at epoch {epoch}"
            break
        if arm == "sync" and epoch == 1:
            first_moments = moment_audit(model, optimizer)
        valid_acc, valid_ce, _, _ = base.evaluate(
            model, arm_for_base(arm), bundle, bundle.valid_idx, bundle.valid_y)
        if not math.isfinite(valid_acc) or not math.isfinite(valid_ce):
            failure = f"nonfinite validation metric at epoch {epoch}"
            break
        improved = valid_acc > best_acc or (valid_acc == best_acc and valid_ce < best_ce)
        if improved:
            best_acc, best_ce, best_epoch = valid_acc, valid_ce, epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        trace.append((epoch, valid_acc, valid_ce, int(improved),
                      int(arm == "sync") if arm == "sync" else ""))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    with (work / "validation_trace.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRACE_FIELDS)
        writer.writerows(trace)
    result = {"protocol": "optimizer_aggregation_mechanism_v1",
              "freeze_sha256": freeze_sha, "dataset": dataset, "arm": arm, "seed": seed,
              "learning_rate": lr, "weight_decay": wd,
              "epochs_required": EPOCHS, "epochs_completed": len(trace),
              "training_seconds": seconds, "initialization": initial,
              "first_step_private_moments": first_moments,
              "validation_trace_sha256": sha(work / "validation_trace.csv"),
              "failure": failure}
    if failure is None:
        require(best_state is not None and len(trace) == EPOCHS, "Incomplete finite mechanism training")
        checkpoint = {"state_dict": best_state, "dataset": dataset, "arm": arm,
                      "learning_rate": lr, "weight_decay": wd,
                      "seed": seed, "epoch": best_epoch, "freeze_sha256": freeze_sha}
        torch.save(checkpoint, work / "checkpoint.pt")
        model.load_state_dict(best_state, strict=True)
        val_acc, val_ce, val_pooled, _ = base.evaluate(
            model, arm_for_base(arm), bundle, bundle.valid_idx, bundle.valid_y)
        require(abs(val_acc - best_acc) <= 1e-7 and abs(val_ce - best_ce) <= 1e-6,
                "Selected mechanism validation checkpoint did not replay")
        result.update({"selected_epoch": best_epoch, "selected_valid_accuracy": best_acc,
                       "selected_valid_ce": best_ce,
                       "selected_valid_pooled_logits_sha256": base.tensor_sha(val_pooled),
                       "selected_state_sha256": base.state_sha(best_state),
                       "checkpoint_sha256": sha(work / "checkpoint.pt")})
        if arm == "sync":
            collapse = collapse_validation_audit(model, bundle, device)
            result["collapse_validation"] = collapse
            torch.save({"state_dict": collapsed_state(model), "dataset": dataset,
                        "seed": seed, "freeze_sha256": freeze_sha,
                        "source_checkpoint_sha256": result["checkpoint_sha256"]},
                       work / "collapsed_checkpoint.pt")
            result["collapsed_checkpoint_sha256"] = sha(work / "collapsed_checkpoint.pt")
    write_json(work / "result.json", result)
    work.rename(final)
    print(json.dumps({"completed": str(final.relative_to(ROOT)), "seconds": seconds,
                      "failure": failure}), flush=True)
    return failure or "complete"


def run_dataset(dataset, device):
    freeze_sha = check_freeze()
    bundle, _ = base.load_graph(dataset, device, include_test=False)
    for arm in ARMS:
        for lr, wd in configurations(arm):
            for seed in SEEDS:
                train_one(dataset, bundle, freeze_sha, arm, lr, wd, seed, device)


def score_dataset(dataset, device):
    freeze_sha = check_freeze()
    mechanism_lock_path = ROOT / "MECHANISM_VALIDATION_LOCK.json"
    original_lock_path = ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"
    require(mechanism_lock_path.is_file() and original_lock_path.is_file(),
            "Both complete validation locks are required before test inference")
    mechanism_lock = json.loads(mechanism_lock_path.read_text())
    original_lock = json.loads(original_lock_path.read_text())
    require(mechanism_lock["freeze_sha256"] == freeze_sha and
            len(mechanism_lock["cells"]) == 96 and
            original_lock["freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            len(original_lock["cells"]) == 432,
            "Validation locks do not match complete frozen studies")
    bundle, _ = base.load_graph(dataset, device, include_test=True)
    for arm in ARMS:
        selected = tuple(mechanism_lock["sync_selections"][dataset]["selected_candidate"])
        allowed = dict.fromkeys((selected, DEFAULT)) if arm == "sync" else (DEFAULT,)
        for lr, wd in allowed:
          for seed in SEEDS:
            key = f"{dataset}/{arm}/{base.candidate_name(lr, wd)}/seed{seed}"
            folder = cell_dir(dataset, arm, lr, wd, seed)
            locked = mechanism_lock["cells"][key]
            require(sha(folder / "result.json") == locked["result_sha256"] and
                    sha(folder / "checkpoint.pt") == locked["checkpoint_sha256"],
                    f"Mechanism cell changed after lock: {key}")
            result = json.loads((folder / "result.json").read_text())
            base.seed_all(seed)
            model, _ = make_model(arm, bundle, device)
            payload = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(payload["state_dict"], strict=True)
            va, vc, vp, _ = base.evaluate(model, arm_for_base(arm), bundle,
                                          bundle.valid_idx, bundle.valid_y)
            require(abs(va - result["selected_valid_accuracy"]) <= 1e-7 and
                    abs(vc - result["selected_valid_ce"]) <= 1e-6,
                    f"Validation replay failed before test inference: {key}")
            ta, tc, tp, tm = base.evaluate(model, arm_for_base(arm), bundle,
                                           bundle.test_idx, bundle.test_y)
            collapse_test = None
            if arm == "sync":
                compact = collapse_model(model, bundle, device)
                _, _, compact_test, compact_test_members = base.evaluate(
                    compact, "tied", bundle, bundle.test_idx, bundle.test_y)
                difference = float((tm - compact_test_members).abs().max().item())
                mismatches = int((tp.argmax(-1) != compact_test.argmax(-1)).sum().item())
                require(difference <= COLLAPSE_LOGIT_TOL and mismatches == 0,
                        f"Collapsed inference differs on test nodes: {key}")
                collapse_test = {"max_abs_member_logit_difference": difference,
                                 "decision_mismatches": mismatches}
            out = ROOT / "scores" / dataset / arm / base.candidate_name(lr, wd) / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(out / "predictions.npz",
                                valid_pooled_logits=vp.numpy(),
                                test_pooled_logits=tp.numpy(),
                                test_member_logits=tm.numpy())
            write_json(out / "score.json", {
                "protocol": "optimizer_aggregation_mechanism_v1",
                "freeze_sha256": freeze_sha,
                "mechanism_validation_lock_sha256": sha(mechanism_lock_path),
                "original_432_validation_lock_sha256": sha(original_lock_path),
                "dataset": dataset, "arm": arm, "seed": seed,
                "learning_rate": lr, "weight_decay": wd,
                "selected_sync_candidate": arm == "sync" and (lr, wd) == selected,
                "predeclared_default": (lr, wd) == DEFAULT,
                "checkpoint_sha256": locked["checkpoint_sha256"],
                "valid_accuracy": va, "valid_ce": vc,
                "test_accuracy": ta, "test_ce": tc,
                "collapsed_test_replay": collapse_test,
                "predictions_sha256": sha(out / "predictions.npz")})
            print(json.dumps({"scored": key, "test_accuracy": ta}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("check-freeze")
    sub.add_parser("preflight").add_argument("--device", default="cpu")
    run = sub.add_parser("run")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--device", default="cuda")
    score = sub.add_parser("score")
    score.add_argument("--dataset", choices=DATASETS, required=True)
    score.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "freeze":
        path = ROOT / "MECHANISM_FREEZE.json"
        require(not path.exists(), "Refusing to overwrite prospective mechanism freeze")
        write_json(path, expected_freeze())
        print(json.dumps({"mechanism_freeze_sha256": sha(path)}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"mechanism_freeze_sha256": check_freeze()}), flush=True)
    elif args.command == "preflight":
        preflight(torch.device(args.device))
    elif args.command == "run":
        run_dataset(args.dataset, torch.device(args.device))
    else:
        score_dataset(args.dataset, torch.device(args.device))


if __name__ == "__main__":
    main()
