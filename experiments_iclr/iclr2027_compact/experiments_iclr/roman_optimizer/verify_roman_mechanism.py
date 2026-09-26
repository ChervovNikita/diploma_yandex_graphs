"""Independent validation lock, post-lock test score, and full replay audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mechanism as mech
import norm_sync_v2 as norm
import roman_mechanism as study
import tuning as base

ROOT = Path(__file__).resolve().parent
REPLAY_TOL = 1e-4


def read(path: Path):
    return json.loads(path.read_text())


def trace_check(path: Path, row: dict, arm: str) -> None:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == study.EPOCHS
    best_acc, best_ce, best_epoch = -1.0, math.inf, 0
    for epoch, r in enumerate(rows, 1):
        assert int(r["epoch"]) == epoch
        acc, ce = float(r["valid_accuracy"]), float(r["valid_ce"])
        assert math.isfinite(acc) and math.isfinite(ce) and 0 <= acc <= 1
        improved = acc > best_acc or (acc == best_acc and ce < best_ce)
        assert int(r["selected_now"]) == int(improved)
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, epoch
        assert int(r["graph_weights_equal"]) == int(arm in ("sync", "norm_sync"))
        if arm == "norm_sync":
            reference = float(r["reference_graph_update_norm"])
            candidate = float(r["candidate_graph_update_norm"])
            applied = float(r["applied_graph_update_norm"])
            error = float(r["relative_norm_error"])
            nominal = float(r["nominal_norm_scale"])
            scale = float(r["applied_norm_scale"])
            correction = float(r["relative_scale_correction"])
            iterations = int(r["rounding_correction_iterations"])
            direct_error = float(r["direct_cast_abs_norm_error"])
            assert all(math.isfinite(v) and v >= 0 for v in
                       (reference, candidate, applied, error, nominal, scale,
                        direct_error))
            assert math.isfinite(correction)
            allowed = max(1e-8, 1e-5 * reference)
            assert abs(applied - reference) <= allowed
            assert abs(error - abs(applied - reference) /
                       max(reference, norm.NORM_ABS_FLOOR)) <= 1e-7
            assert abs(correction) <= norm.ROUNDING_RADIUS + 1e-12
            assert iterations in (0, norm.ROUNDING_BISECTIONS)
            if reference == 0.0:
                assert nominal == scale == correction == direct_error == 0.0
                assert iterations == 0
                assert r["zero_case"] == (
                    "both_zero" if candidate == 0.0 else "zero_reference")
            else:
                assert candidate > 0.0
                assert r["zero_case"] == "none"
                assert abs(nominal - reference / candidate) <= 1e-6 * max(
                    1.0, reference / candidate)
                assert abs(scale / nominal - 1.0 - correction) <= 1e-10
                assert ((iterations == 0 and direct_error <= allowed and
                         correction == 0.0) or
                        (iterations == norm.ROUNDING_BISECTIONS and
                         direct_error > allowed))
        else:
            assert all(r[k] == "" for k in
                       ("reference_graph_update_norm", "candidate_graph_update_norm",
                        "applied_graph_update_norm", "relative_norm_error",
                        "nominal_norm_scale", "applied_norm_scale",
                        "relative_scale_correction", "rounding_correction_iterations",
                        "direct_cast_abs_norm_error", "zero_case"))
    assert best_epoch == row["selected_epoch"]
    assert abs(best_acc - row["valid_accuracy"]) < 1e-7
    assert abs(best_ce - row["valid_ce"]) < 1e-6


def check_preflight(source_sha: str) -> dict:
    gate = read(ROOT / "preflight_cuda.json")
    assert gate["status"] == "ALL_SEED_DEPTH_CUDA_PREFLIGHT_PASS"
    assert gate["source_freeze_sha256"] == source_sha
    assert {(r["depth"], r["seed"]) for r in gate["records"]} == {
        (d, s) for d in study.DEPTHS for s in study.SEEDS}
    for record in gate["records"]:
        assert len(record["sgd"]["steps"]) == 5
        assert len(record["norm"]["steps"]) == 5
        for step in record["sgd"]["steps"]:
            parameter = float(step["parameter_max_abs"])
            logit = float(step["member_logit_max_abs"])
            assert math.isfinite(parameter) and 0 <= parameter <= mech.SGD_PARAM_TOL
            assert math.isfinite(logit) and 0 <= logit <= mech.SGD_LOGIT_TOL
        for step in record["norm"]["steps"]:
            reference = float(step["reference_graph_update_norm"])
            candidate = float(step["candidate_graph_update_norm"])
            applied = float(step["applied_graph_update_norm"])
            error = float(step["relative_norm_error"])
            nominal = float(step["nominal_norm_scale"])
            scale = float(step["applied_norm_scale"])
            correction = float(step["relative_scale_correction"])
            iterations = int(step["rounding_correction_iterations"])
            direct_error = float(step["direct_cast_abs_norm_error"])
            assert all(math.isfinite(x) and x >= 0 for x in
                       (reference, candidate, applied, error, nominal, scale,
                        direct_error))
            assert math.isfinite(correction)
            allowed = max(norm.NORM_ABS_FLOOR, norm.NORM_REL_TOL * reference)
            assert abs(applied - reference) <= allowed
            assert abs(correction) <= norm.ROUNDING_RADIUS + 1e-12
            assert iterations in (0, norm.ROUNDING_BISECTIONS)
            if reference == 0.0:
                assert nominal == scale == correction == direct_error == 0.0
                assert iterations == 0
                assert step["zero_case"] == (
                    "both_zero" if candidate == 0.0 else "zero_reference")
            else:
                assert candidate > 0.0
                assert step["zero_case"] == "none"
                assert abs(nominal - reference / candidate) <= 1e-6 * max(
                    1.0, reference / candidate)
                assert abs(scale / nominal - 1.0 - correction) <= 1e-10
                assert ((iterations == 0 and direct_error <= allowed and
                         correction == 0.0) or
                        (iterations == norm.ROUNDING_BISECTIONS and
                         direct_error > allowed))
        collapse = record["norm"]["collapse"]
        assert collapse["pooled_validation_decision_mismatches"] == 0
        assert math.isfinite(collapse["max_abs_validation_member_logit_difference"])
        assert 0 <= collapse["max_abs_validation_member_logit_difference"] <= mech.COLLAPSE_LOGIT_TOL
        moments = record["norm"]["separate_moments"]
        assert moments["separate_first_moment_tensors"] > 0
        assert moments["separate_second_moment_tensors"] > 0
        assert (moments["first_moment_max_abs_member_difference"] > 0 or
                moments["second_moment_max_abs_member_difference"] > 0)
        numeric = record["norm"]
        assert numeric["first_step_reference_equals_explicit_mean_exact"] is True
        assert numeric["first_step_optimizer_settings"] == {
            "learning_rate": 0.001, "betas": [0.9, 0.999],
            "epsilon": 1e-8, "weight_decay": 0.0}
        assert 0 <= numeric["first_step_initial_graph_weight_max_abs"] <= 1.0
        tied_g = float(numeric["first_step_tied_gradient_norm"])
        mean_g = float(numeric["first_step_explicit_mean_gradient_norm"])
        g_l2 = float(numeric["first_step_tied_vs_mean_gradient_l2_error"])
        g_linf = float(numeric["first_step_tied_vs_mean_gradient_linf_error"])
        allowed_g_l2 = study.GRAD_REL_TOL * max(tied_g, mean_g) + study.GRAD_ABS_FLOOR
        assert all(math.isfinite(v) and v >= 0 for v in
                   (tied_g, mean_g, g_l2, g_linf, allowed_g_l2))
        assert abs(float(numeric["first_step_gradient_l2_allowed"]) -
                   allowed_g_l2) <= 1e-12
        assert g_l2 <= allowed_g_l2 and g_linf <= study.GRAD_LINF_TOL
        for field in ("first_step_tied_adam_formula_linf_error",
                      "first_step_virtual_adam_formula_linf_error"):
            value = float(numeric[field])
            assert math.isfinite(value) and 0 <= value <= study.FORMULA_COORD_TOL
        for field in ("first_step_virtual_vs_tied_vector_l2_error",
                      "first_step_virtual_vs_tied_vector_linf_error",
                      "first_step_tied_graph_update_norm",
                      "first_step_virtual_graph_update_norm"):
            value = float(numeric[field])
            assert math.isfinite(value) and value >= 0
        assert set(record["initial"]["arms"]) == set(study.ARMS)
    return gate


def expected_run_dirs() -> set[Path]:
    return {study.run_dir(depth, seed, arm) for depth in study.DEPTHS
            for seed in study.SEEDS for arm in study.ARMS}


def validate(device: torch.device) -> None:
    source_sha = study.check_freeze()
    gate = check_preflight(source_sha)
    assert not (ROOT / "validation_lock.json").exists(), "Existing lock requires review"
    assert not (ROOT / "test_scores").exists(), "Test score files predate validation lock"
    expected = expected_run_dirs()
    found = {p for p in (ROOT / "results/roman").glob("depth*/seed*/*") if p.is_dir()}
    assert found == expected, f"Expected exactly 24 run directories: {len(found)}"
    records = []
    for depth in study.DEPTHS:
        base.configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=False)
        assert descriptor == gate["data_descriptors"][str(depth)]
        assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
        for seed in study.SEEDS:
            initial_logits = {}
            initials = {}
            for arm in study.ARMS:
                folder = study.run_dir(depth, seed, arm)
                row = read(folder / "result.json")
                assert row["protocol"] == study.PROTOCOL
                assert row["source_freeze_sha256"] == source_sha
                assert (row["depth"], row["seed"], row["arm"]) == (depth, seed, arm)
                assert row["official_mask"] == 0 and row["epochs_completed"] == 1000
                assert not any(key.startswith("test") for key in row)
                assert set(row["artifact_sha256"]) == {
                    "checkpoint.pt", "validation_trace.csv",
                    "selected_validation.npz", "initial_logits.npy"}
                for rel, digest in row["artifact_sha256"].items():
                    assert study.sha(folder / rel) == digest
                trace_check(folder / "validation_trace.csv", row, arm)
                base.seed_all(seed)
                model, canonical = study.make_arm(arm, bundle, device)
                fresh = base.initial_audit(model, study.eval_arm(arm), bundle,
                                           seed, device, canonical)
                stored = row["initialization"]
                for field in ("canonical_projector_state_sha256", "python_rng_sha256",
                              "numpy_rng_sha256", "cpu_rng_sha256",
                              "cuda_rng_sha256", "parameter_count"):
                    assert fresh[field] == stored[field], (depth, seed, arm, field)
                archived_initial = np.load(folder / "initial_logits.npy", allow_pickle=False)
                assert archived_initial.shape == (4, 22662, bundle.classes)
                assert np.isfinite(archived_initial).all()
                assert study.sha(folder / "initial_logits.npy") == stored["saved_initial_logits_sha256"]
                model.eval()
                with torch.no_grad():
                    replay_initial = torch.stack(base.member_logits(
                        model, study.eval_arm(arm), bundle)).cpu().numpy()
                assert float(np.max(np.abs(archived_initial - replay_initial))) <= base.INITIAL_LOGIT_TOL
                assert np.array_equal(archived_initial.argmax(-1), replay_initial.argmax(-1))
                initial_logits[arm] = archived_initial
                initials[arm] = stored
                checkpoint = torch.load(folder / "checkpoint.pt", map_location=device,
                                        weights_only=True)
                assert checkpoint["depth"] == depth and checkpoint["seed"] == seed
                assert checkpoint["arm"] == arm and checkpoint["epoch"] == row["selected_epoch"]
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                if arm in ("sync", "norm_sync"):
                    mech.assert_equal_graph_weights(model)
                acc, ce, logits, pooled = base.evaluate(
                    model, study.eval_arm(arm), bundle,
                    bundle.valid_idx, bundle.valid_y)
                assert abs(acc - row["valid_accuracy"]) < 1e-7
                assert abs(ce - row["valid_ce"]) < 1e-4
                with np.load(folder / "selected_validation.npz", allow_pickle=False) as p:
                    assert np.array_equal(p["node_index"], bundle.valid_idx.cpu().numpy())
                    assert np.array_equal(p["y_true"], bundle.valid_y.cpu().numpy())
                    saved = p["member_logits"]
                replay = logits.detach().cpu().numpy()
                assert saved.shape == replay.shape
                assert float(np.max(np.abs(saved - replay))) <= REPLAY_TOL
                assert np.array_equal(saved.argmax(-1), replay.argmax(-1))
                assert np.array_equal(saved.mean(0).argmax(-1), replay.mean(0).argmax(-1))
                if arm in ("sync", "norm_sync"):
                    collapse = mech.collapse_validation_audit(model, bundle, device)
                    assert collapse["pooled_validation_decision_mismatches"] == 0
                    assert collapse["max_abs_validation_member_logit_difference"] <= 1e-5
                    assert row["selected_collapse"]["collapsed_state_sha256"] == collapse["collapsed_state_sha256"]
                records.append({"depth": depth, "seed": seed, "arm": arm,
                                "result_sha256": study.sha(folder / "result.json"),
                                "checkpoint_sha256": row["artifact_sha256"]["checkpoint.pt"],
                                "selected_epoch": row["selected_epoch"],
                                "valid_accuracy": acc, "valid_ce": ce,
                                "parameter_count": row["parameter_count"]})
                del model
            anchor = initials["tied"]
            for arm in study.ARMS[1:]:
                for field in ("canonical_projector_state_sha256", "python_rng_sha256",
                              "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
                    assert initials[arm][field] == anchor[field]
                assert float(np.max(np.abs(initial_logits[arm] - initial_logits["tied"]))) <= base.INITIAL_LOGIT_TOL
    assert len(records) == 24
    study.write_json(ROOT / "validation_lock.json",
                     {"status": "COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK",
                      "source_freeze_sha256": source_sha,
                      "preflight_sha256": study.sha(ROOT / "preflight_cuda.json"),
                      "records": records,
                      "test_scored": False})
    print("COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK", flush=True)


def score(device: torch.device) -> None:
    source_sha = study.check_freeze()
    gate = check_preflight(source_sha)
    lock_path = ROOT / "validation_lock.json"
    lock = read(lock_path)
    assert lock["status"] == "COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK"
    assert lock["source_freeze_sha256"] == source_sha
    assert lock["preflight_sha256"] == study.sha(ROOT / "preflight_cuda.json")
    assert lock["test_scored"] is False and len(lock["records"]) == 24
    keyed = {(r["depth"], r["seed"], r["arm"]): r for r in lock["records"]}
    assert len(keyed) == 24
    for depth in study.DEPTHS:
        base.configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=True)
        assert descriptor == gate["data_descriptors"][str(depth)]
        assert hasattr(bundle, "test_idx_cpu") and hasattr(bundle, "test_y_cpu")
        idx = bundle.test_idx_cpu.to(device)
        y = bundle.test_y_cpu.to(device)
        for seed in study.SEEDS:
            for arm in study.ARMS:
                folder = study.run_dir(depth, seed, arm)
                row = read(folder / "result.json")
                assert study.sha(folder / "result.json") == keyed[depth, seed, arm]["result_sha256"]
                assert study.sha(folder / "checkpoint.pt") == keyed[depth, seed, arm]["checkpoint_sha256"]
                target = ROOT / "test_scores" / f"depth{depth}" / f"seed{seed}" / arm
                assert not target.exists(), f"Test cell already scored: {target}"
                target.mkdir(parents=True)
                base.seed_all(seed)
                model, _ = study.make_arm(arm, bundle, device)
                checkpoint = torch.load(folder / "checkpoint.pt", map_location=device,
                                        weights_only=True)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                acc, ce, logits, _ = base.evaluate(model, study.eval_arm(arm), bundle, idx, y)
                assert math.isfinite(acc) and math.isfinite(ce)
                np.savez_compressed(target / "test_predictions.npz",
                                    member_logits=logits.detach().cpu().numpy(),
                                    node_index=bundle.test_idx_cpu.numpy(),
                                    y_true=bundle.test_y_cpu.numpy())
                study.write_json(target / "test_result.json",
                                 {"protocol": study.PROTOCOL,
                                  "source_freeze_sha256": source_sha,
                                  "validation_lock_sha256": study.sha(lock_path),
                                  "depth": depth, "seed": seed, "arm": arm,
                                  "selected_checkpoint_sha256": study.sha(folder / "checkpoint.pt"),
                                  "test_accuracy": acc, "test_ce": ce,
                                  "test_predictions_sha256": study.sha(target / "test_predictions.npz")})
                print("TEST_SCORED", depth, seed, arm, flush=True)
                del model
    print("ALL_24_TEST_CELLS_SCORED", flush=True)


def audit_test(device: torch.device) -> None:
    source_sha = study.check_freeze()
    lock_path = ROOT / "validation_lock.json"
    lock = read(lock_path)
    assert lock["status"] == "COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK"
    assert lock["source_freeze_sha256"] == source_sha
    assert len(lock["records"]) == 24
    locked = {(r["depth"], r["seed"], r["arm"]): r for r in lock["records"]}
    assert len(locked) == 24
    expected = {(d, s, a) for d in study.DEPTHS for s in study.SEEDS for a in study.ARMS}
    expected_dirs = {ROOT / "test_scores" / f"depth{d}" / f"seed{s}" / a
                     for d, s, a in expected}
    actual_dirs = {p for p in (ROOT / "test_scores").glob("depth*/seed*/*") if p.is_dir()}
    assert actual_dirs == expected_dirs, "Missing or extra test-score directory"
    found = set()
    rows = []
    for depth in study.DEPTHS:
        base.configure(depth)
        bundle, _ = base.load_graph("roman", device, include_test=True)
        idx = bundle.test_idx_cpu.to(device)
        y = bundle.test_y_cpu.to(device)
        for seed in study.SEEDS:
            for arm in study.ARMS:
                key = depth, seed, arm
                folder = study.run_dir(depth, seed, arm)
                target = ROOT / "test_scores" / f"depth{depth}" / f"seed{seed}" / arm
                report = read(target / "test_result.json")
                assert report["protocol"] == study.PROTOCOL
                assert report["source_freeze_sha256"] == source_sha
                assert report["validation_lock_sha256"] == study.sha(lock_path)
                assert (report["depth"], report["seed"], report["arm"]) == key
                assert study.sha(folder / "result.json") == locked[key]["result_sha256"]
                assert study.sha(folder / "checkpoint.pt") == locked[key]["checkpoint_sha256"]
                assert report["selected_checkpoint_sha256"] == study.sha(folder / "checkpoint.pt")
                assert report["test_predictions_sha256"] == study.sha(target / "test_predictions.npz")
                base.seed_all(seed)
                model, _ = study.make_arm(arm, bundle, device)
                checkpoint = torch.load(folder / "checkpoint.pt", map_location=device,
                                        weights_only=True)
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                if arm in ("sync", "norm_sync"):
                    mech.assert_equal_graph_weights(model)
                acc, ce, logits, _ = base.evaluate(model, study.eval_arm(arm), bundle, idx, y)
                assert abs(acc - report["test_accuracy"]) < 1e-7
                assert abs(ce - report["test_ce"]) < 1e-4
                with np.load(target / "test_predictions.npz", allow_pickle=False) as p:
                    assert np.array_equal(p["node_index"], bundle.test_idx_cpu.numpy())
                    assert np.array_equal(p["y_true"], bundle.test_y_cpu.numpy())
                    saved = p["member_logits"]
                replay = logits.detach().cpu().numpy()
                assert saved.shape == replay.shape
                assert float(np.max(np.abs(saved - replay))) <= REPLAY_TOL
                assert np.array_equal(saved.argmax(-1), replay.argmax(-1))
                assert np.array_equal(saved.mean(0).argmax(-1), replay.mean(0).argmax(-1))
                rows.append({"depth": depth, "seed": seed, "arm": arm,
                             "test_accuracy": acc, "test_ce": ce,
                             "test_result_sha256": study.sha(target / "test_result.json")})
                found.add(key)
                del model
    assert found == expected and len(rows) == 24
    keyed = {(r["depth"], r["seed"], r["arm"]): r["test_accuracy"] for r in rows}
    paired = [{"depth": d, "seed": s,
               "tied_minus_untied_pp": 100 * (keyed[d, s, "tied"] - keyed[d, s, "untied"]),
               "tied_minus_sync_pp": 100 * (keyed[d, s, "tied"] - keyed[d, s, "sync"]),
               "tied_minus_norm_sync_pp": 100 * (keyed[d, s, "tied"] - keyed[d, s, "norm_sync"]),
               "sync_minus_norm_sync_pp": 100 * (keyed[d, s, "sync"] - keyed[d, s, "norm_sync"])}
              for d in study.DEPTHS for s in study.SEEDS]
    study.write_json(ROOT / "complete_score_audit.json",
                     {"status": "COMPLETE_24_CELL_CUDA_TEST_REPLAY_PASS",
                      "protocol": study.PROTOCOL,
                      "source_freeze_sha256": source_sha,
                      "validation_lock_sha256": study.sha(lock_path),
                      "records": rows, "paired": paired,
                      "scope": "post hoc one Roman graph, official mask 0, two depths and three optimization seeds"})
    print("COMPLETE_24_CELL_CUDA_TEST_REPLAY_PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "score", "audit-test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    cli = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA required")
    if cli.stage == "validate":
        validate(device)
    elif cli.stage == "score":
        score(device)
    else:
        audit_test(device)


if __name__ == "__main__":
    main()
