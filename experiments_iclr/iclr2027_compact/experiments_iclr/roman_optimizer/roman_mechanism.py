"""Fresh Roman mask-0 optimizer-history intervention with a test firewall."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric

import mechanism as mech
import norm_sync_v2 as norm
import tuning as base

ROOT = Path(__file__).resolve().parent
PROTOCOL = "roman_optimizer_history_mask0_depth2_5_v3"
DEPTHS = (2, 5)
SEEDS = (0, 1, 2)
ARMS = ("tied", "untied", "sync", "norm_sync")
EPOCHS = 1000
LR = 0.001
WD = 0.0
MEMBERS = 4
GRAD_REL_TOL = 1e-6
GRAD_ABS_FLOOR = 1e-8
GRAD_LINF_TOL = 1e-7
FORMULA_COORD_TOL = 2 * torch.finfo(torch.float32).eps
SOURCE_FILES = {"roman_mechanism.py", "verify_roman_mechanism.py",
                "tuning.py", "ROMAN_MECHANISM_PROTOCOL.md",
                "ROMAN_MECHANISM_DESIGN_FREEZE.json",
                "roman_multimask.py", "models.py", "mechanism.py",
                "norm_sync_v2.py", "data/roman_empire.npz"}
TRACE_FIELDS = ("epoch", "valid_accuracy", "valid_ce", "selected_now",
                "graph_weights_equal", "reference_graph_update_norm",
                "candidate_graph_update_norm", "applied_graph_update_norm",
                "relative_norm_error", "nominal_norm_scale", "applied_norm_scale",
                "relative_scale_correction", "rounding_correction_iterations",
                "direct_cast_abs_norm_error", "zero_case")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def check_freeze() -> str:
    path = ROOT / "ROMAN_MECHANISM_SOURCE_FREEZE.json"
    frozen = json.loads(path.read_text())
    assert frozen["protocol"] == PROTOCOL
    assert frozen["depths"] == list(DEPTHS)
    assert frozen["seeds"] == list(SEEDS)
    assert frozen["arms"] == list(ARMS)
    assert frozen["expected_cells"] == 24
    assert frozen["configuration"] == {
        "official_mask": 0, "depths": list(DEPTHS),
        "width": base.WIDTH, "members": MEMBERS,
        "training_edges": 65854, "added_self_loops": False,
        "epochs": EPOCHS, "learning_rate": LR, "weight_decay": WD,
        "checkpoint_rule": "highest pooled validation accuracy; lowest pooled CE; earliest epoch",
        "initial_member_logit_tolerance": base.INITIAL_LOGIT_TOL,
        "sgd_parameter_tolerance": mech.SGD_PARAM_TOL,
        "sgd_logit_tolerance": mech.SGD_LOGIT_TOL,
        "collapse_logit_tolerance": mech.COLLAPSE_LOGIT_TOL,
        "norm_relative_tolerance": norm.NORM_REL_TOL,
        "norm_absolute_floor": norm.NORM_ABS_FLOOR,
        "rounding_relative_radius": norm.ROUNDING_RADIUS,
        "rounding_bisection_steps": norm.ROUNDING_BISECTIONS,
        "first_step_gradient_relative_tolerance": GRAD_REL_TOL,
        "first_step_gradient_absolute_floor": GRAD_ABS_FLOOR,
        "first_step_gradient_linf_tolerance": GRAD_LINF_TOL,
        "first_step_adam_formula_coordinate_tolerance": FORMULA_COORD_TOL,
        "first_step_graph_dtype": "float32",
        "first_step_initial_graph_weight_max_abs_bound": 1.0,
        "test_rule": "complete 24-cell validation lock before any test score"}
    assert frozen["design_freeze_sha256"] == sha(ROOT / "ROMAN_MECHANISM_DESIGN_FREEZE.json")
    assert set(frozen["source_sha256"]) == SOURCE_FILES
    assert frozen["runtime"] == {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__, "numpy": np.__version__}
    for rel, digest in frozen["source_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    return sha(path)


def run_dir(depth: int, seed: int, arm: str) -> Path:
    assert depth in DEPTHS and seed in SEEDS and arm in ARMS
    return ROOT / "results/roman" / f"depth{depth}" / f"seed{seed}" / arm


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=LR,
                             betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=WD)


def make_arm(arm: str, bundle, device: torch.device):
    if arm in ("tied", "untied", "sync", "norm_sync"):
        model_arm = "tied" if arm == "tied" else "untied"
        return base.make_model(model_arm, bundle, device)
    raise ValueError(arm)


def eval_arm(arm: str) -> str:
    return "tied" if arm == "tied" else "untied"


def initial_four_arm_gate(bundle, device, depth: int, seed: int) -> dict:
    rows = {}
    logits = {}
    snapshots = {}
    for arm in ARMS:
        base.seed_all(seed)
        model, canonical = make_arm(arm, bundle, device)
        snapshots[arm] = mech.rng_snapshot(device)
        row = base.initial_audit(model, eval_arm(arm), bundle, seed, device, canonical)
        rows[arm] = row
        model.eval()
        with torch.no_grad():
            logits[arm] = torch.stack(base.member_logits(model, eval_arm(arm), bundle)).cpu().numpy()
        if arm in ("sync", "norm_sync"):
            mech.assert_equal_graph_weights(model)
        del model
    anchor = rows["tied"]
    for arm in ARMS[1:]:
        for field in ("canonical_projector_state_sha256", "python_rng_sha256",
                      "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
            assert rows[arm][field] == anchor[field], (depth, seed, arm, field)
        assert mech.rng_equal(snapshots[arm], snapshots["tied"])
        diff = float(np.max(np.abs(logits[arm] - logits["tied"])))
        assert diff <= base.INITIAL_LOGIT_TOL, (depth, seed, arm, diff)
        assert np.array_equal(logits[arm].argmax(-1), logits["tied"].argmax(-1))
        rows[arm]["tied_initial_member_logits_max_abs_difference"] = diff
    rows["tied"]["tied_initial_member_logits_max_abs_difference"] = 0.0
    return {"depth": depth, "seed": seed, "arms": rows}


def sgd_gate(bundle, device, depth: int, seed: int) -> dict:
    base.seed_all(seed)
    tied, tied_sha = base.make_model("tied", bundle, device)
    rng = mech.rng_snapshot(device)
    base.seed_all(seed)
    sync, sync_sha = base.make_model("untied", bundle, device)
    assert tied_sha == sync_sha and mech.rng_equal(rng, mech.rng_snapshot(device))
    assert mech.paired_max_parameter_difference(tied, sync) == 0.0
    opt_tied = torch.optim.SGD(tied.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    opt_sync = torch.optim.SGD(sync.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0)
    differences = []
    for step in range(1, 6):
        before = mech.rng_snapshot(device)
        mech.one_update(tied, "tied", bundle, opt_tied)
        tied_after = mech.rng_snapshot(device)
        mech.rng_restore(before, device)
        mech.one_update(sync, "sync", bundle, opt_sync)
        assert mech.rng_equal(tied_after, mech.rng_snapshot(device))
        parameter = mech.paired_max_parameter_difference(tied, sync)
        logit = mech.paired_max_logit_difference(tied, sync, bundle)
        assert parameter <= mech.SGD_PARAM_TOL and logit <= mech.SGD_LOGIT_TOL
        differences.append({"step": step, "parameter_max_abs": parameter,
                            "member_logit_max_abs": logit})
        mech.rng_restore(tied_after, device)
    return {"depth": depth, "seed": seed, "steps": differences}


def norm_gate(bundle, device, depth: int, seed: int) -> dict:
    base.seed_all(seed)
    tied, tied_canonical = base.make_model("tied", bundle, device)
    matched_rng = mech.rng_snapshot(device)
    tied_before = norm.flat(list(tied.residual_modules.parameters())).clone()
    assert tied_before.dtype == torch.float32
    assert float(tied_before.abs().max().item()) <= 1.0
    tied_opt = optimizer_for(tied)
    base.seed_all(seed)
    model, canonical = base.make_model("untied", bundle, device)
    assert tied_canonical == canonical
    assert mech.rng_equal(matched_rng, mech.rng_snapshot(device))
    groups = mech.graph_stack_parameters(model)
    reference = [torch.nn.Parameter(p.detach().clone()) for p in groups[0]]
    actual_opt = optimizer_for(model)
    reference_opt = torch.optim.AdamW(reference, lr=LR, betas=(0.9, 0.999),
                                      eps=1e-8, weight_decay=WD)
    rows = []
    mech.rng_restore(matched_rng, device)
    mech.one_update(tied, "tied", bundle, tied_opt)
    tied_after = mech.rng_snapshot(device)
    tied_delta = norm.flat(list(tied.residual_modules.parameters())) - tied_before
    mech.rng_restore(matched_rng, device)
    for step in range(1, 6):
        row = norm.one_norm_matched_update(model, actual_opt, reference,
                                            reference_opt, bundle)
        if step == 1:
            assert mech.rng_equal(tied_after, mech.rng_snapshot(device))
            virtual_delta = norm.flat(reference) - tied_before
            vector_difference = (virtual_delta - tied_delta).double()
            first_step_vector_error = float(torch.linalg.vector_norm(
                vector_difference).item())
            first_step_vector_linf_error = float(
                vector_difference.abs().max().item())
            tied_norm = float(torch.linalg.vector_norm(tied_delta.double()).item())
            virtual_norm = float(torch.linalg.vector_norm(
                virtual_delta.double()).item())
            tied_gradient = norm.flat(
                [p.grad for p in tied.residual_modules.parameters()])
            explicit_mean_parts = [
                torch.stack([p.grad.detach() for p in corresponding], dim=0).mean(0)
                for corresponding in zip(*groups)]
            assert all(torch.equal(ref.grad, mean) for ref, mean in
                       zip(reference, explicit_mean_parts))
            mean_gradient = norm.flat(explicit_mean_parts)
            assert tied_gradient.dtype == mean_gradient.dtype == torch.float32
            gradient_difference = (tied_gradient - mean_gradient).double()
            gradient_l2_error = float(torch.linalg.vector_norm(
                gradient_difference).item())
            gradient_linf_error = float(gradient_difference.abs().max().item())
            tied_gradient_norm = float(torch.linalg.vector_norm(
                tied_gradient.double()).item())
            mean_gradient_norm = float(torch.linalg.vector_norm(
                mean_gradient.double()).item())
            gradient_l2_allowed = (GRAD_REL_TOL * max(
                tied_gradient_norm, mean_gradient_norm) + GRAD_ABS_FLOOR)
            assert gradient_l2_error <= gradient_l2_allowed
            assert gradient_linf_error <= GRAD_LINF_TOL
            for optimizer in (tied_opt, reference_opt):
                group = optimizer.param_groups[0]
                assert group["lr"] == LR and group["weight_decay"] == WD == 0.0
                assert group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
            before64 = tied_before.double()
            tied_g64, mean_g64 = tied_gradient.double(), mean_gradient.double()
            tied_expected = (before64 - tied_opt.param_groups[0]["lr"] * tied_g64 /
                             (tied_g64.abs() + tied_opt.param_groups[0]["eps"])).to(
                                 tied_before.dtype)
            virtual_expected = (before64 - reference_opt.param_groups[0]["lr"] * mean_g64 /
                                (mean_g64.abs() +
                                 reference_opt.param_groups[0]["eps"])).to(
                                     tied_before.dtype)
            tied_formula_linf_error = float((
                norm.flat(list(tied.residual_modules.parameters())) -
                tied_expected).abs().max().item())
            virtual_formula_linf_error = float((
                norm.flat(reference) - virtual_expected).abs().max().item())
            assert tied_formula_linf_error <= FORMULA_COORD_TOL
            assert virtual_formula_linf_error <= FORMULA_COORD_TOL
        assert row["relative_norm_error"] <= norm.NORM_REL_TOL or (
            row["reference_graph_update_norm"] < norm.NORM_ABS_FLOOR and
            abs(row["applied_graph_update_norm"] - row["reference_graph_update_norm"]) <=
            norm.NORM_ABS_FLOOR)
        assert abs(row["relative_scale_correction"]) <= norm.ROUNDING_RADIUS + 1e-12
        assert row["rounding_correction_iterations"] in (0, norm.ROUNDING_BISECTIONS)
        allowed = max(norm.NORM_ABS_FLOOR,
                      norm.NORM_REL_TOL * row["reference_graph_update_norm"])
        assert ((row["rounding_correction_iterations"] == 0 and
                 row["direct_cast_abs_norm_error"] <= allowed) or
                (row["rounding_correction_iterations"] == norm.ROUNDING_BISECTIONS and
                 row["direct_cast_abs_norm_error"] > allowed))
        mech.assert_equal_graph_weights(model)
        rows.append({"step": step, **row})
    moments = mech.moment_audit(model, actual_opt)
    assert len(reference_opt.state) == len(reference)
    assert all(float(reference_opt.state[p]["step"].item()) == 5 for p in reference)
    collapse = mech.collapse_validation_audit(model, bundle, device)
    return {"depth": depth, "seed": seed, "steps": rows,
            "separate_moments": moments, "collapse": collapse,
            "canonical_initial_sha256": canonical,
            "first_step_virtual_vs_tied_vector_l2_error": first_step_vector_error,
            "first_step_virtual_vs_tied_vector_linf_error": first_step_vector_linf_error,
            "first_step_tied_graph_update_norm": tied_norm,
            "first_step_virtual_graph_update_norm": virtual_norm,
            "first_step_tied_gradient_norm": tied_gradient_norm,
            "first_step_explicit_mean_gradient_norm": mean_gradient_norm,
            "first_step_tied_vs_mean_gradient_l2_error": gradient_l2_error,
            "first_step_tied_vs_mean_gradient_linf_error": gradient_linf_error,
            "first_step_gradient_l2_allowed": gradient_l2_allowed,
            "first_step_reference_equals_explicit_mean_exact": True,
            "first_step_tied_adam_formula_linf_error": tied_formula_linf_error,
            "first_step_virtual_adam_formula_linf_error": virtual_formula_linf_error,
            "first_step_optimizer_settings": {
                "learning_rate": tied_opt.param_groups[0]["lr"],
                "betas": tied_opt.param_groups[0]["betas"],
                "epsilon": tied_opt.param_groups[0]["eps"],
                "weight_decay": tied_opt.param_groups[0]["weight_decay"]},
            "first_step_initial_graph_weight_max_abs": float(tied_before.abs().max().item())}


def preflight(device: torch.device) -> None:
    frozen_sha = check_freeze()
    records = []
    descriptors = {}
    for depth in DEPTHS:
        base.configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=False)
        assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
        descriptors[str(depth)] = descriptor
        for seed in SEEDS:
            records.append({"depth": depth, "seed": seed,
                            "initial": initial_four_arm_gate(bundle, device, depth, seed),
                            "sgd": sgd_gate(bundle, device, depth, seed),
                            "norm": norm_gate(bundle, device, depth, seed)})
            print("PREFLIGHT_PASS", depth, seed, flush=True)
    assert len(records) == 6
    write_json(ROOT / "preflight_cuda.json",
               {"status": "ALL_SEED_DEPTH_CUDA_PREFLIGHT_PASS",
                "source_freeze_sha256": frozen_sha,
                "records": records, "data_descriptors": descriptors,
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda})
    print("ALL_SEED_DEPTH_CUDA_PREFLIGHT_PASS", len(records), flush=True)


def train_one(depth: int, seed: int, arm: str, bundle,
              device: torch.device, frozen_sha: str) -> None:
    final = run_dir(depth, seed, arm)
    if final.exists():
        old = json.loads((final / "result.json").read_text())
        assert old["source_freeze_sha256"] == frozen_sha
        assert old["depth"] == depth and old["seed"] == seed and old["arm"] == arm
        for filename, digest in old["artifact_sha256"].items():
            assert sha(final / filename) == digest
        print("SKIP_COMPLETE", depth, seed, arm, flush=True)
        return
    work = final.with_name(arm + ".inprogress")
    assert not work.exists()
    work.mkdir(parents=True)
    base.seed_all(seed)
    model, canonical = make_arm(arm, bundle, device)
    initial = base.initial_audit(model, eval_arm(arm), bundle, seed, device, canonical)
    model.eval()
    with torch.no_grad():
        initial_logits = torch.stack(base.member_logits(model, eval_arm(arm), bundle)).cpu().numpy()
    assert initial_logits.shape == (MEMBERS, 22662, bundle.classes)
    np.save(work / "initial_logits.npy", initial_logits)
    initial["saved_initial_logits_sha256"] = sha(work / "initial_logits.npy")
    if arm in ("sync", "norm_sync"):
        mech.assert_equal_graph_weights(model)
    opt = optimizer_for(model)
    reference = reference_opt = None
    if arm == "norm_sync":
        groups = mech.graph_stack_parameters(model)
        reference = [torch.nn.Parameter(p.detach().clone()) for p in groups[0]]
        reference_opt = torch.optim.AdamW(reference, lr=LR, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=WD)
    best_acc, best_ce, best_epoch = -1.0, math.inf, 0
    trace = []
    first_moments = None
    start = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        step_data = {}
        if arm == "norm_sync":
            step_data = norm.one_norm_matched_update(model, opt, reference,
                                                      reference_opt, bundle)
        else:
            mech.one_update(model, arm, bundle, opt)
        if epoch == 1 and arm in ("sync", "norm_sync"):
            first_moments = mech.moment_audit(model, opt)
        if arm in ("sync", "norm_sync"):
            mech.assert_equal_graph_weights(model)
        acc, ce, _, _ = base.evaluate(model, eval_arm(arm), bundle,
                                      bundle.valid_idx, bundle.valid_y)
        if not (math.isfinite(acc) and math.isfinite(ce)):
            raise RuntimeError(f"Nonfinite validation at {depth}/{seed}/{arm}/{epoch}")
        improved = acc > best_acc or (acc == best_acc and ce < best_ce)
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, epoch
            torch.save({"state_dict": model.state_dict(), "depth": depth,
                        "seed": seed, "arm": arm, "epoch": epoch},
                       work / "checkpoint.pt")
        trace.append({"epoch": epoch, "valid_accuracy": acc, "valid_ce": ce,
                      "selected_now": int(improved),
                      "graph_weights_equal": int(arm in ("sync", "norm_sync")),
                      **{name: step_data.get(name, "") for name in
                         ("reference_graph_update_norm",
                         "candidate_graph_update_norm", "applied_graph_update_norm",
                          "relative_norm_error", "nominal_norm_scale",
                          "applied_norm_scale", "relative_scale_correction",
                          "rounding_correction_iterations",
                          "direct_cast_abs_norm_error", "zero_case")}})
        if epoch == 1 or epoch % 100 == 0:
            print("TRAIN_PROGRESS", depth, seed, arm, epoch, flush=True)
    with (work / "validation_trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(trace)
    checkpoint = torch.load(work / "checkpoint.pt", map_location=device, weights_only=True)
    assert checkpoint["depth"] == depth and checkpoint["seed"] == seed
    assert checkpoint["arm"] == arm and checkpoint["epoch"] == best_epoch
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    valid_acc, valid_ce, valid_logits, _ = base.evaluate(
        model, eval_arm(arm), bundle, bundle.valid_idx, bundle.valid_y)
    assert abs(valid_acc - best_acc) < 1e-7 and abs(valid_ce - best_ce) < 1e-6
    collapse = (mech.collapse_validation_audit(model, bundle, device)
                if arm in ("sync", "norm_sync") else None)
    np.savez_compressed(work / "selected_validation.npz",
                        member_logits=valid_logits.detach().cpu().numpy(),
                        node_index=bundle.valid_idx.cpu().numpy(),
                        y_true=bundle.valid_y.cpu().numpy())
    artifacts = ("checkpoint.pt", "validation_trace.csv", "selected_validation.npz",
                 "initial_logits.npy")
    row = {"protocol": PROTOCOL, "source_freeze_sha256": frozen_sha,
           "dataset": "roman", "official_mask": 0, "depth": depth,
           "seed": seed, "arm": arm, "epochs_completed": EPOCHS,
           "selected_epoch": best_epoch, "valid_accuracy": valid_acc,
           "valid_ce": valid_ce, "parameter_count": initial["parameter_count"],
           "initialization": initial, "first_step_moments": first_moments,
           "selected_collapse": collapse,
           "train_seconds": time.monotonic() - start,
           "artifact_sha256": {name: sha(work / name) for name in artifacts}}
    write_json(work / "result.json", row)
    work.rename(final)
    print("ARM_VALIDATION_COMPLETE", depth, seed, arm, flush=True)
    del model, opt
    torch.cuda.empty_cache()


def train(device: torch.device) -> None:
    frozen_sha = check_freeze()
    gate = json.loads((ROOT / "preflight_cuda.json").read_text())
    assert gate["status"] == "ALL_SEED_DEPTH_CUDA_PREFLIGHT_PASS"
    assert gate["source_freeze_sha256"] == frozen_sha and len(gate["records"]) == 6
    for depth in DEPTHS:
        base.configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=False)
        assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
        assert descriptor == gate["data_descriptors"][str(depth)]
        for seed in SEEDS:
            for arm in ARMS:
                train_one(depth, seed, arm, bundle, device, frozen_sha)
    print("ALL_24_VALIDATION_RUNS_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "train"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA required")
    preflight(device) if args.stage == "preflight" else train(device)


if __name__ == "__main__":
    main()
