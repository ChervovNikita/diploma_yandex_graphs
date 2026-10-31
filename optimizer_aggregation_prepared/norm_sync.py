"""Prospective 12-cell global-norm ablation for synchronized graph weights."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mechanism as mech
import tuning as base


ROOT = Path(__file__).resolve().parent
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
EPOCHS = 1000
ETA = 0.001
DECAY = 0.0
BETAS = (0.9, 0.999)
EPS = 1e-8
NORM_REL_TOL = 1e-5
NORM_ABS_FLOOR = 1e-8
SOURCE_FILES = ("norm_sync.py", "verify_norm_sync.py", "NORM_MATCHED_SYNC_PROTOCOL.md")
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now",
                "graph_weights_equal", "reference_graph_update_norm",
                "candidate_graph_update_norm", "applied_graph_update_norm",
                "relative_norm_error", "norm_scale", "zero_case")


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return base.sha256_file(path)


def write_json(path: Path, value: dict) -> None:
    base.write_json(path, value)


def expected_freeze() -> dict:
    design = json.loads((ROOT / "NORM_SYNC_DESIGN_FREEZE.json").read_text())
    require(design["protocol"] == "norm_matched_sync_design_v1" and
            design["source_sha256"]["NORM_MATCHED_SYNC_PROTOCOL.md"] ==
            sha(ROOT / "NORM_MATCHED_SYNC_PROTOCOL.md") and
            design["source_sha256"]["MECHANISM_FREEZE.json"] ==
            sha(ROOT / "MECHANISM_FREEZE.json") and
            design["source_sha256"]["FROZEN_STUDY.json"] ==
            sha(ROOT / "FROZEN_STUDY.json") and
            design["planned_cells"] == 12 and
            tuple(design["datasets"]) == DATASETS and
            tuple(design["seeds"]) == SEEDS,
            "NORM-SYNC implementation differs from prospective design record")
    return {
        "protocol": "norm_matched_sync_v1",
        "design_freeze_sha256": sha(ROOT / "NORM_SYNC_DESIGN_FREEZE.json"),
        "base_freeze_sha256": base.check_freeze(),
        "mechanism_freeze_sha256": mech.check_freeze(),
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "matrix": {"datasets": DATASETS, "seeds": SEEDS, "planned_cells": 12,
                   "epochs": EPOCHS, "learning_rate": ETA, "weight_decay": DECAY,
                   "betas": BETAS, "epsilon": EPS,
                   "norm_relative_tolerance": NORM_REL_TOL,
                   "norm_absolute_floor": NORM_ABS_FLOOR,
                   "reference_rule": "one virtual AdamW moment pair per graph parameter, fed mean full-member gradient on current NORM-SYNC graph weights",
                   "candidate_rule": "average four private AdamW graph updates after ×4 gradient correction",
                   "applied_rule": "candidate graph direction scaled to virtual reference global L2 norm; shared boundary and output parameters use mean-loss AdamW",
                   "zero_rule": "zero reference => zero applied; positive reference with zero candidate => invalid cell; nonfinite => invalid cell",
                   "checkpoint_rule": "highest pooled validation accuracy; lowest pooled CE; earliest epoch",
                   "test_rule": "complete 12-cell, 96-cell, and 432-cell validation locks before 12 default-cell test scores",
                   "complete_study_audit_cutoff_utc": "2026-09-26T06:00:00Z"},
    }


def check_freeze() -> str:
    path = ROOT / "NORM_SYNC_FREEZE.json"
    require(path.is_file(), "Missing prospective NORM-SYNC source freeze")
    expected = json.loads(json.dumps(expected_freeze(), sort_keys=True, allow_nan=False))
    require(json.loads(path.read_text()) == expected,
            "NORM-SYNC source, original graphs, or fixed matrix differs from freeze")
    return sha(path)


def cell_dir(dataset: str, seed: int) -> Path:
    require(dataset in DATASETS and seed in SEEDS, "Undeclared NORM-SYNC cell")
    return ROOT / "results_norm_sync" / dataset / f"seed{seed}"


def new_model_and_optimizers(bundle, seed: int, device: torch.device):
    base.seed_all(seed)
    model, canonical = mech.make_model("sync", bundle, device)
    initial = base.initial_audit(model, "untied", bundle, seed, device, canonical)
    mech.assert_equal_graph_weights(model)
    groups = mech.graph_stack_parameters(model)
    reference = [torch.nn.Parameter(p.detach().clone()) for p in groups[0]]
    actual_opt = torch.optim.AdamW(model.parameters(), lr=ETA, betas=BETAS,
                                   eps=EPS, weight_decay=DECAY)
    reference_opt = torch.optim.AdamW(reference, lr=ETA, betas=BETAS,
                                      eps=EPS, weight_decay=DECAY)
    require(len(reference) == len(groups[0]) and len(reference) > 0,
            "Virtual reference graph stack differs")
    return model, actual_opt, reference, reference_opt, initial


def flat(values) -> torch.Tensor:
    return torch.cat([value.detach().reshape(-1) for value in values])


@torch.no_grad()
def copy_flat_to_graph_copies(model, flat_state: torch.Tensor) -> None:
    groups = mech.graph_stack_parameters(model)
    offset = 0
    for corresponding in zip(*groups):
        size = corresponding[0].numel()
        value = flat_state[offset:offset + size].view_as(corresponding[0])
        for parameter in corresponding:
            parameter.copy_(value)
        offset += size
    require(offset == flat_state.numel(), "Graph flattening changed")
    mech.assert_equal_graph_weights(model)


def one_norm_matched_update(model, actual_opt, reference, reference_opt, bundle) -> dict:
    mech.assert_equal_graph_weights(model)
    groups = mech.graph_stack_parameters(model)
    before_parts = [p.detach().clone() for p in groups[0]]
    before = flat(before_parts)
    model.train()
    actual_opt.zero_grad(set_to_none=True)
    for logits in base.member_logits(model, "untied", bundle):
        loss = F.cross_entropy(logits.index_select(0, bundle.train_idx), bundle.train_y)
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        (loss / 4).backward()
    mech.scale_private_graph_gradients(model)

    reference_opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for ref, initial, corresponding in zip(reference, before_parts, zip(*groups)):
            ref.copy_(initial)
            ref.grad = torch.stack([p.grad.detach() for p in corresponding], dim=0).mean(0)
            require(bool(torch.isfinite(ref.grad).all()), "Nonfinite mean graph gradient")
    reference_opt.step()
    reference_delta = flat(reference) - before
    reference_norm = float(torch.linalg.vector_norm(reference_delta.double()).item())

    actual_opt.step()
    mech.synchronize_graph_weights(model)
    candidate_delta = flat(groups[0]) - before
    candidate_norm = float(torch.linalg.vector_norm(candidate_delta.double()).item())
    require(math.isfinite(reference_norm) and math.isfinite(candidate_norm),
            "Nonfinite graph update norm")
    if reference_norm == 0.0:
        next_state, scale = before, 0.0
        zero_case = "both_zero" if candidate_norm == 0.0 else "zero_reference"
    else:
        require(candidate_norm > 0.0, "zero sync update with positive reference norm")
        scale = reference_norm / candidate_norm
        require(math.isfinite(scale), "Nonfinite norm matching scale")
        next_state = before + candidate_delta * scale
        zero_case = "none"
    require(bool(torch.isfinite(next_state).all()), "Nonfinite applied graph weights")
    copy_flat_to_graph_copies(model, next_state)
    applied_delta = flat(groups[0]) - before
    applied_norm = float(torch.linalg.vector_norm(applied_delta.double()).item())
    allowed = max(NORM_ABS_FLOOR, NORM_REL_TOL * reference_norm)
    norm_error = abs(applied_norm - reference_norm)
    require(norm_error <= allowed,
            f"NORM-SYNC applied/reference graph update norm mismatch: {norm_error}, {allowed}")
    require(all("exp_avg" in reference_opt.state[p] and
                "exp_avg_sq" in reference_opt.state[p] for p in reference),
            "Virtual reference AdamW moments missing")
    return {"reference_graph_update_norm": reference_norm,
            "candidate_graph_update_norm": candidate_norm,
            "applied_graph_update_norm": applied_norm,
            "relative_norm_error": norm_error / max(reference_norm, NORM_ABS_FLOOR),
            "zero_case": zero_case,
            "norm_scale": scale,
            "reference_moment_tensor_count": len(reference)}


def preflight(device: torch.device) -> None:
    freeze_sha = check_freeze()
    bundle, _ = base.load_graph("cora", device, include_test=False)
    base.seed_all(73)
    tied, tied_canonical = base.make_model("tied", bundle, device)
    tied_rng = mech.rng_snapshot(device)
    tied_before = flat(list(tied.residual_modules.parameters())).clone()
    model, actual_opt, reference, reference_opt, initial = new_model_and_optimizers(
        bundle, 73, device)
    require(tied_canonical == initial["canonical_projector_state_sha256"] and
            mech.rng_equal(tied_rng, mech.rng_snapshot(device)),
            "NORM-SYNC/TIED preflight initial model or RNG differs")
    tied_opt = torch.optim.AdamW(tied.parameters(), lr=ETA, betas=BETAS,
                                 eps=EPS, weight_decay=DECAY)
    mech.rng_restore(tied_rng, device)
    mech.one_update(tied, "tied", bundle, tied_opt)
    tied_after = mech.rng_snapshot(device)
    tied_delta = flat(list(tied.residual_modules.parameters())) - tied_before
    tied_first_norm = float(torch.linalg.vector_norm(tied_delta.double()).item())
    mech.rng_restore(tied_rng, device)
    steps = []
    norm_after_first = None
    virtual_first_delta = None
    for step_index in range(5):
        row = one_norm_matched_update(model, actual_opt, reference, reference_opt, bundle)
        steps.append(row)
        if step_index == 0:
            norm_after_first = mech.rng_snapshot(device)
            virtual_first_delta = flat(reference) - tied_before
    require(norm_after_first is not None and mech.rng_equal(tied_after, norm_after_first) and
            abs(tied_first_norm - steps[0]["reference_graph_update_norm"]) <=
            max(NORM_ABS_FLOOR, NORM_REL_TOL * tied_first_norm),
            "Virtual first-step mean-gradient norm differs from actual matched TIED")
    require(virtual_first_delta is not None and
            torch.linalg.vector_norm((virtual_first_delta - tied_delta).double()).item() <=
            max(NORM_ABS_FLOOR, NORM_REL_TOL * tied_first_norm),
            "Virtual first-step graph update vector differs from actual matched TIED")
    moments = mech.moment_audit(model, actual_opt)
    require(len(reference_opt.state) == len(reference) and
            all(float(reference_opt.state[p]["step"].item()) == 5 for p in reference) and
            any(step["candidate_graph_update_norm"] != step["reference_graph_update_norm"]
                for step in steps),
            "Virtual reference or norm intervention inactive")
    report = {"protocol": "norm_matched_sync_v1", "freeze_sha256": freeze_sha,
              "device": str(device), "initialization": initial,
              "five_step_norm_audit": steps, "separate_member_moments": moments,
              "actual_tied_first_graph_update_norm": tied_first_norm,
              "virtual_tied_first_graph_update_vector_l2_error":
              float(torch.linalg.vector_norm((virtual_first_delta - tied_delta).double()).item()),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    write_json(ROOT / f"NORM_SYNC_PREFLIGHT_{str(device).replace(':', '_').upper()}.json",
               report)
    print(json.dumps({"preflight": "passed", "device": str(device),
                      "freeze_sha256": freeze_sha,
                      "max_relative_norm_error": max(x["relative_norm_error"] for x in steps)}),
          flush=True)


def train_one(dataset: str, seed: int, bundle, device, freeze_sha: str) -> None:
    final = cell_dir(dataset, seed)
    if final.exists():
        result = json.loads((final / "result.json").read_text())
        require(result["freeze_sha256"] == freeze_sha and
                result["dataset"] == dataset and result["seed"] == seed and
                result["validation_trace_sha256"] == sha(final / "validation_trace.csv"),
                f"Existing NORM-SYNC cell failed identity gate: {dataset}/{seed}")
        if result["failure"] is None:
            require(result["checkpoint_sha256"] == sha(final / "checkpoint.pt"),
                    f"Existing NORM-SYNC checkpoint changed: {dataset}/{seed}")
        else:
            require(not (final / "checkpoint.pt").exists(),
                    f"Failed NORM-SYNC cell has checkpoint: {dataset}/{seed}")
        return
    work = final.with_name(final.name + ".inprogress")
    require(not work.exists(), f"Interrupted NORM-SYNC cell needs inspection: {work}")
    work.mkdir(parents=True)
    model, actual_opt, reference, reference_opt, initial = new_model_and_optimizers(
        bundle, seed, device)
    best_acc, best_ce, best_epoch = -float("inf"), float("inf"), 0
    best_state, first_moments, traces = None, None, []
    max_norm_error, min_scale, max_scale = 0.0, float("inf"), 0.0
    failure = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        try:
            step = one_norm_matched_update(model, actual_opt, reference, reference_opt, bundle)
        except RuntimeError as error:
            if str(error) not in (
                "Nonfinite training loss", "Nonfinite mean graph gradient",
                "Nonfinite graph parameter after optimizer step",
                "Nonfinite graph update norm",
                "zero sync update with positive reference norm",
                "Nonfinite norm matching scale", "Nonfinite applied graph weights"):
                raise
            failure = f"{str(error)} at epoch {epoch}"
            break
        if epoch == 1:
            first_moments = mech.moment_audit(model, actual_opt)
        max_norm_error = max(max_norm_error, step["relative_norm_error"])
        min_scale = min(min_scale, step["norm_scale"])
        max_scale = max(max_scale, step["norm_scale"])
        valid_acc, valid_ce, _, _ = base.evaluate(
            model, "untied", bundle, bundle.valid_idx, bundle.valid_y)
        if not math.isfinite(valid_acc) or not math.isfinite(valid_ce):
            failure = f"Nonfinite NORM-SYNC validation metric at epoch {epoch}"
            break
        improved = valid_acc > best_acc or (valid_acc == best_acc and valid_ce < best_ce)
        if improved:
            best_acc, best_ce, best_epoch = valid_acc, valid_ce, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        traces.append((epoch, valid_acc, valid_ce, int(improved), 1,
                       step["reference_graph_update_norm"],
                       step["candidate_graph_update_norm"],
                       step["applied_graph_update_norm"],
                       step["relative_norm_error"], step["norm_scale"],
                       step["zero_case"]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    with (work / "validation_trace.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(TRACE_FIELDS)
        writer.writerows(traces)
    if failure is not None:
        result = {"protocol": "norm_matched_sync_v1", "freeze_sha256": freeze_sha,
                  "dataset": dataset, "seed": seed, "epochs_required": EPOCHS,
                  "epochs_completed": len(traces), "training_seconds": seconds,
                  "initialization": initial, "first_step_private_moments": first_moments,
                  "reference_moment_tensor_count": len(reference),
                  "validation_trace_sha256": sha(work / "validation_trace.csv"),
                  "failure": failure}
        write_json(work / "result.json", result)
        work.rename(final)
        print(json.dumps({"completed": [dataset, seed], "failure": failure}), flush=True)
        return
    require(best_state is not None and len(traces) == EPOCHS,
            "Incomplete NORM-SYNC finite run")
    torch.save({"state_dict": best_state, "protocol": "norm_matched_sync_v1",
                "dataset": dataset, "seed": seed, "epoch": best_epoch,
                "freeze_sha256": freeze_sha}, work / "checkpoint.pt")
    model.load_state_dict(best_state, strict=True)
    valid_acc, valid_ce, pooled, _ = base.evaluate(
        model, "untied", bundle, bundle.valid_idx, bundle.valid_y)
    require(abs(valid_acc - best_acc) <= 1e-7 and abs(valid_ce - best_ce) <= 1e-6,
            "NORM-SYNC selected checkpoint failed validation replay")
    collapse = mech.collapse_validation_audit(model, bundle, device)
    torch.save({"state_dict": mech.collapsed_state(model), "dataset": dataset,
                "seed": seed, "freeze_sha256": freeze_sha,
                "source_checkpoint_sha256": sha(work / "checkpoint.pt")},
               work / "collapsed_checkpoint.pt")
    result = {"protocol": "norm_matched_sync_v1", "freeze_sha256": freeze_sha,
              "dataset": dataset, "seed": seed, "epochs_required": EPOCHS,
              "epochs_completed": EPOCHS, "training_seconds": seconds,
              "initialization": initial, "first_step_private_moments": first_moments,
              "reference_moment_tensor_count": len(reference),
              "selected_epoch": best_epoch, "selected_valid_accuracy": best_acc,
              "selected_valid_ce": best_ce,
              "selected_valid_pooled_logits_sha256": base.tensor_sha(pooled),
              "selected_state_sha256": base.state_sha(best_state),
              "validation_trace_sha256": sha(work / "validation_trace.csv"),
              "checkpoint_sha256": sha(work / "checkpoint.pt"),
              "collapsed_checkpoint_sha256": sha(work / "collapsed_checkpoint.pt"),
              "collapse_validation": collapse,
              "max_relative_norm_error": max_norm_error,
              "min_norm_scale": min_scale, "max_norm_scale": max_scale,
              "failure": None}
    write_json(work / "result.json", result)
    work.rename(final)
    print(json.dumps({"completed": [dataset, seed], "seconds": seconds,
                      "max_relative_norm_error": max_norm_error}), flush=True)


def run_dataset(dataset: str, device: torch.device) -> None:
    freeze_sha = check_freeze()
    bundle, _ = base.load_graph(dataset, device, include_test=False)
    for seed in SEEDS:
        train_one(dataset, seed, bundle, device, freeze_sha)


def score_dataset(dataset: str, device: torch.device) -> None:
    freeze_sha = check_freeze()
    norm_lock_path = ROOT / "NORM_SYNC_VALIDATION_LOCK.json"
    mech_lock_path = ROOT / "MECHANISM_VALIDATION_LOCK.json"
    base_lock_path = ROOT / "ORIGINAL_432_VALIDATION_SELECTION_LOCK.json"
    require(all(p.is_file() for p in (norm_lock_path, mech_lock_path, base_lock_path)),
            "Complete 12-, 96-, and 432-cell validation locks required")
    norm_lock = json.loads(norm_lock_path.read_text())
    mech_lock = json.loads(mech_lock_path.read_text())
    base_lock = json.loads(base_lock_path.read_text())
    require(norm_lock["freeze_sha256"] == freeze_sha and len(norm_lock["cells"]) == 12 and
            mech_lock["freeze_sha256"] == sha(ROOT / "MECHANISM_FREEZE.json") and
            len(mech_lock["cells"]) == 96 and
            base_lock["freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            len(base_lock["cells"]) == 432,
            "One or more complete validation locks mismatches source freeze")
    bundle, _ = base.load_graph(dataset, device, include_test=True)
    for seed in SEEDS:
        key = f"{dataset}/seed{seed}"
        cell = cell_dir(dataset, seed)
        locked = norm_lock["cells"][key]
        require(sha(cell / "result.json") == locked["result_sha256"] and
                sha(cell / "checkpoint.pt") == locked["checkpoint_sha256"],
                f"Locked NORM-SYNC cell changed: {key}")
        base.seed_all(seed)
        model, _ = mech.make_model("sync", bundle, device)
        checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        valid_acc, valid_ce, valid_pooled, _ = base.evaluate(
            model, "untied", bundle, bundle.valid_idx, bundle.valid_y)
        result = json.loads((cell / "result.json").read_text())
        require(abs(valid_acc - result["selected_valid_accuracy"]) <= 1e-7 and
                abs(valid_ce - result["selected_valid_ce"]) <= 1e-6,
                f"Validation replay failed before NORM-SYNC test inference: {key}")
        test_acc, test_ce, test_pooled, test_members = base.evaluate(
            model, "untied", bundle, bundle.test_idx, bundle.test_y)
        collapsed = mech.collapse_model(model, bundle, device)
        _, _, collapsed_pooled, collapsed_members = base.evaluate(
            collapsed, "tied", bundle, bundle.test_idx, bundle.test_y)
        collapse_diff = float((test_members - collapsed_members).abs().max().item())
        collapse_decisions = int((test_pooled.argmax(-1) !=
                                  collapsed_pooled.argmax(-1)).sum().item())
        require(collapse_diff <= mech.COLLAPSE_LOGIT_TOL and collapse_decisions == 0,
                f"Collapsed NORM-SYNC test inference differs: {key}")
        out = ROOT / "scores_norm_sync" / dataset / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(out / "predictions.npz",
                            valid_pooled_logits=valid_pooled.numpy(),
                            test_pooled_logits=test_pooled.numpy(),
                            test_member_logits=test_members.numpy())
        write_json(out / "score.json", {
            "protocol": "norm_matched_sync_v1", "freeze_sha256": freeze_sha,
            "norm_validation_lock_sha256": sha(norm_lock_path),
            "mechanism_validation_lock_sha256": sha(mech_lock_path),
            "original_432_validation_lock_sha256": sha(base_lock_path),
            "dataset": dataset, "seed": seed,
            "checkpoint_sha256": locked["checkpoint_sha256"],
            "valid_accuracy": valid_acc, "valid_ce": valid_ce,
            "test_accuracy": test_acc, "test_ce": test_ce,
            "collapsed_test_replay": {"max_abs_member_logit_difference": collapse_diff,
                                      "decision_mismatches": collapse_decisions},
            "predictions_sha256": sha(out / "predictions.npz")})
        print(json.dumps({"scored": key, "test_accuracy": test_acc}), flush=True)


def main() -> None:
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
        path = ROOT / "NORM_SYNC_FREEZE.json"
        require(not path.exists(), "Refusing to overwrite prospective NORM-SYNC freeze")
        write_json(path, expected_freeze())
        print(json.dumps({"norm_sync_freeze_sha256": sha(path)}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"norm_sync_freeze_sha256": check_freeze()}), flush=True)
    elif args.command == "preflight":
        preflight(torch.device(args.device))
    elif args.command == "run":
        run_dataset(args.dataset, torch.device(args.device))
    else:
        score_dataset(args.dataset, torch.device(args.device))


if __name__ == "__main__":
    main()
