"""Independent local score and provenance audit of the compact Roman24 stage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEPTHS = (2, 5)
SEEDS = (0, 1, 2)
ARMS = ("tied", "untied", "sync", "norm_sync")
SOURCE = {"roman_mechanism.py", "verify_roman_mechanism.py", "tuning.py",
          "ROMAN_MECHANISM_PROTOCOL.md", "ROMAN_MECHANISM_DESIGN_FREEZE.json",
          "roman_multimask.py", "models.py", "mechanism.py", "norm_sync_v2.py"}
PROTOCOL = "roman_optimizer_history_mask0_depth2_5_v3"
# Integer-decision accuracy is exact; the original CUDA audit stores float32
# accuracy. A two-arm percentage-point contrast can differ by a few 1e-6 pp.
PAIRED_REPORTING_TOL_PP = 1e-5


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def trace_check(path: Path, row: dict, arm: str) -> None:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1000
    best_acc, best_ce, best_epoch = -1.0, math.inf, 0
    for epoch, item in enumerate(rows, 1):
        assert int(item["epoch"]) == epoch
        acc, ce = float(item["valid_accuracy"]), float(item["valid_ce"])
        assert math.isfinite(acc) and math.isfinite(ce) and 0 <= acc <= 1
        improved = acc > best_acc or (acc == best_acc and ce < best_ce)
        assert int(item["selected_now"]) == int(improved)
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, epoch
        assert int(item["graph_weights_equal"]) == int(arm in ("sync", "norm_sync"))
        if arm == "norm_sync":
            ref = float(item["reference_graph_update_norm"])
            candidate = float(item["candidate_graph_update_norm"])
            applied = float(item["applied_graph_update_norm"])
            error = float(item["relative_norm_error"])
            nominal = float(item["nominal_norm_scale"])
            scale = float(item["applied_norm_scale"])
            correction = float(item["relative_scale_correction"])
            iterations = int(item["rounding_correction_iterations"])
            direct = float(item["direct_cast_abs_norm_error"])
            assert all(math.isfinite(x) for x in
                       (ref, candidate, applied, error, nominal, scale,
                        correction, direct))
            assert min(ref, candidate, applied, error, nominal, scale, direct) >= 0
            allowed = max(1e-8, 1e-5 * ref)
            assert abs(applied - ref) <= allowed
            assert abs(error - abs(applied - ref) / max(ref, 1e-8)) <= 1e-7
            assert abs(correction) <= 0.001 + 1e-12
            assert iterations in (0, 48)
            if ref == 0:
                assert nominal == scale == correction == direct == 0
                assert iterations == 0
                assert item["zero_case"] == (
                    "both_zero" if candidate == 0 else "zero_reference")
            else:
                assert candidate > 0 and item["zero_case"] == "none"
                assert abs(nominal - ref / candidate) <= 1e-6 * max(
                    1, ref / candidate)
                assert abs(scale / nominal - 1 - correction) <= 1e-10
                assert ((iterations == 0 and direct <= allowed and correction == 0) or
                        (iterations == 48 and direct > allowed))
        else:
            assert all(item[k] == "" for k in
                       ("reference_graph_update_norm", "candidate_graph_update_norm",
                        "applied_graph_update_norm", "relative_norm_error",
                        "nominal_norm_scale", "applied_norm_scale",
                        "relative_scale_correction", "rounding_correction_iterations",
                        "direct_cast_abs_norm_error", "zero_case"))
    assert best_epoch == row["selected_epoch"]
    assert abs(best_acc - row["valid_accuracy"]) < 1e-7
    assert abs(best_ce - row["valid_ce"]) < 1e-6


def preflight_check(gate: dict) -> None:
    assert gate["status"] == "ALL_SEED_DEPTH_CUDA_PREFLIGHT_PASS"
    assert {(r["depth"], r["seed"]) for r in gate["records"]} == {
        (d, s) for d in DEPTHS for s in SEEDS}
    for record in gate["records"]:
        assert len(record["sgd"]["steps"]) == 5
        assert len(record["norm"]["steps"]) == 5
        for row in record["sgd"]["steps"]:
            assert 0 <= row["parameter_max_abs"] <= 1e-4
            assert 0 <= row["member_logit_max_abs"] <= 1e-4
        numeric = record["norm"]
        assert numeric["first_step_reference_equals_explicit_mean_exact"] is True
        assert numeric["first_step_optimizer_settings"] == {
            "learning_rate": 0.001, "betas": [0.9, 0.999],
            "epsilon": 1e-8, "weight_decay": 0.0}
        assert 0 <= numeric["first_step_initial_graph_weight_max_abs"] <= 1
        norms = (float(numeric["first_step_tied_gradient_norm"]),
                 float(numeric["first_step_explicit_mean_gradient_norm"]))
        assert all(math.isfinite(x) and x >= 0 for x in norms)
        allowed = 1e-6 * max(norms) + 1e-8
        assert abs(float(numeric["first_step_gradient_l2_allowed"]) - allowed) <= 1e-12
        assert 0 <= numeric["first_step_tied_vs_mean_gradient_l2_error"] <= allowed
        assert 0 <= numeric["first_step_tied_vs_mean_gradient_linf_error"] <= 1e-7
        assert 0 <= numeric["first_step_tied_adam_formula_linf_error"] <= 2.384185791015625e-7
        assert 0 <= numeric["first_step_virtual_adam_formula_linf_error"] <= 2.384185791015625e-7
        assert numeric["collapse"]["pooled_validation_decision_mismatches"] == 0
        assert numeric["collapse"]["max_abs_validation_member_logit_difference"] <= 1e-5
        assert numeric["separate_moments"]["separate_first_moment_tensors"] > 0
        assert numeric["separate_moments"]["separate_second_moment_tensors"] > 0
        for step in numeric["steps"]:
            ref = float(step["reference_graph_update_norm"])
            applied = float(step["applied_graph_update_norm"])
            assert math.isfinite(ref) and math.isfinite(applied)
            assert abs(applied - ref) <= max(1e-8, 1e-5 * ref)


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--public-npz", type=Path)
    args = cli.parse_args()
    freeze_path = ROOT / "ROMAN_MECHANISM_SOURCE_FREEZE.json"
    freeze = read(freeze_path)
    assert freeze["protocol"] == PROTOCOL
    assert freeze["depths"] == list(DEPTHS) and freeze["seeds"] == list(SEEDS)
    assert freeze["arms"] == list(ARMS) and freeze["expected_cells"] == 24
    assert set(freeze["source_sha256"]) == SOURCE | {"data/roman_empire.npz"}
    assert not (ROOT / "data/roman_empire.npz").exists()
    for name, digest in freeze["source_sha256"].items():
        if name != "data/roman_empire.npz":
            assert sha(ROOT / name) == digest, name
    design = read(ROOT / "ROMAN_MECHANISM_DESIGN_FREEZE.json")
    assert sha(ROOT / "numeric_revision_evidence" /
               "ROMAN_MECHANISM_V2_FIRST_STEP_DIAG_DEPTH2_SEED1.json") == \
        design["v2_failing_first_step_diagnostic_sha256"]
    assert sha(ROOT / "numeric_revision_evidence" /
               "roman_mechanism_v2_all6_first_step_diag.log") == \
        design["all_six_training_only_calibration_log_sha256"]
    gate, lock, audit = (read(ROOT / name) for name in
                         ("preflight_cuda.json", "validation_lock.json",
                          "complete_score_audit.json"))
    source_sha = sha(freeze_path)
    assert gate["source_freeze_sha256"] == source_sha
    preflight_check(gate)
    assert lock["status"] == "COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK"
    assert lock["source_freeze_sha256"] == source_sha
    assert lock["preflight_sha256"] == sha(ROOT / "preflight_cuda.json")
    assert lock["test_scored"] is False and len(lock["records"]) == 24
    amendment_root = ROOT / "verification_amendment"
    amendment = read(amendment_root / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    amendment_source = amendment_root / "roman_mechanism_v3_verification_amendment.py"
    diagnosis = amendment_root / "ROMAN_MECHANISM_V3_COLLAPSE_DIAG_DEPTH2_SEED2_SYNC.json"
    transport = amendment_root / "TRANSPORT_INTERRUPT_NOTE.json"
    assert amendment["status"] == "PROSPECTIVE_VALIDATION_ONLY_VERIFICATION_AMENDMENT_V1"
    assert amendment["original_v3_source_freeze_sha256"] == source_sha
    assert amendment["original_preflight_sha256"] == sha(ROOT / "preflight_cuda.json")
    assert amendment["amendment_source_sha256"] == sha(amendment_source)
    assert amendment["diagnosis_sha256"] == sha(diagnosis)
    assert lock["verification_amendment_sha256"] == sha(
        amendment_root / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    assert lock["amendment_source_sha256"] == sha(amendment_source)
    assert len(amendment["prior_complete_result_sha256"]) == 10
    assert read(transport)["amendment_freeze_sha256"] == sha(
        amendment_root / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    for item in lock["records"]:
        collapse = item["collapse_replay"]
        if item["arm"] in ("sync", "norm_sync"):
            assert collapse["cpu_member_logits_bitwise_equal"] is True
            assert collapse["cpu_collapsed_state_exactly_mapped"] is True
            assert collapse["gpu_collapsed_state_exactly_mapped"] is True
            assert collapse["member_validation_decision_mismatches"] == 0
            assert collapse["pooled_validation_decision_mismatches"] == 0
            assert collapse["max_abs_validation_member_logit_difference"] <= 1e-4
        else:
            assert collapse is None
    assert audit["status"] == "COMPLETE_24_CELL_CUDA_TEST_REPLAY_PASS"
    assert audit["source_freeze_sha256"] == source_sha
    assert audit["validation_lock_sha256"] == sha(ROOT / "validation_lock.json")
    test_collapse = read(amendment_root / "post_score_test_collapse_audit.json")
    test_collapse_source = amendment_root / "roman_v3_postscore_test_collapse_audit.py"
    assert test_collapse["status"] == "COMPLETE_12_CELL_POST_SCORE_TEST_COLLAPSE_AUDIT_PASS"
    assert test_collapse["source_freeze_sha256"] == source_sha
    assert test_collapse["verification_amendment_sha256"] == sha(
        amendment_root / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    assert test_collapse["validation_lock_sha256"] == sha(ROOT / "validation_lock.json")
    assert test_collapse["complete_score_audit_sha256"] == sha(ROOT / "complete_score_audit.json")
    assert test_collapse["audit_source_sha256"] == sha(test_collapse_source)
    assert test_collapse["gpu_diagnostic_tolerance"] == 1e-4
    assert len(test_collapse["records"]) == 12
    collapse_keys = {(r["depth"], r["seed"], r["arm"]) for r in test_collapse["records"]}
    assert collapse_keys == {(d, s, a) for d in DEPTHS for s in SEEDS
                             for a in ("sync", "norm_sync")}
    for item in test_collapse["records"]:
        assert item["cpu_member_logits_bitwise_equal"] is True
        assert item["cpu_state_exactly_mapped"] is True
        assert item["gpu_state_exactly_mapped"] is True
        assert item["gpu_member_decision_mismatches"] == 0
        assert item["gpu_pooled_decision_mismatches"] == 0
        assert item["gpu_max_abs_member_logit_difference"] <= 1e-4
        run = ROOT / "results/roman" / f"depth{item['depth']}" / f"seed{item['seed']}" / item["arm"]
        test = ROOT / "test_scores" / f"depth{item['depth']}" / f"seed{item['seed']}" / item["arm"]
        assert read(run / "result.json")["artifact_sha256"]["checkpoint.pt"] == item["selected_checkpoint_sha256"]
        assert sha(test / "test_result.json") == item["test_result_sha256"]
    manifest = read(ROOT / "decision_derivation_manifest.json")
    assert manifest["status"] == "COMPLETE_24_CELL_DECISION_DERIVATION_PASS"
    assert manifest["source_freeze_sha256"] == source_sha
    assert manifest["preflight_sha256"] == sha(ROOT / "preflight_cuda.json")
    assert manifest["validation_lock_sha256"] == sha(ROOT / "validation_lock.json")
    assert manifest["complete_score_audit_sha256"] == sha(ROOT / "complete_score_audit.json")
    assert manifest["original_public_npz_sha256"] == freeze["source_sha256"]["data/roman_empire.npz"]
    assert manifest["derivation_script_sha256"] == sha(ROOT / "prepare_roman_mechanism_compact.py")
    assert manifest["verification_amendment_freeze_sha256"] == sha(
        amendment_root / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json")
    assert manifest["verification_amendment_source_sha256"] == sha(amendment_source)
    assert manifest["verification_diagnosis_sha256"] == sha(diagnosis)
    assert manifest["transport_interrupt_note_sha256"] == sha(transport)
    assert manifest["post_score_test_collapse_audit_sha256"] == sha(
        amendment_root / "post_score_test_collapse_audit.json")
    assert manifest["post_score_test_collapse_source_sha256"] == sha(test_collapse_source)
    anchor = ROOT / "official_labels_mask0.npz"
    assert manifest["official_label_anchor_sha256"] == sha(anchor)
    with np.load(anchor, allow_pickle=False) as a:
        labels = a["node_labels"].copy()
        masks = {part: a[f"{part}_mask"].copy()
                 for part in ("train", "val", "test")}
    assert labels.shape == (22662,)
    assert tuple(int(masks[k].sum()) for k in ("train", "val", "test")) == (
        11331, 5665, 5666)
    assert not np.any(masks["train"] & masks["val"])
    assert not np.any(masks["train"] & masks["test"])
    assert not np.any(masks["val"] & masks["test"])
    if args.public_npz:
        assert sha(args.public_npz) == freeze["source_sha256"]["data/roman_empire.npz"]
        with np.load(args.public_npz, allow_pickle=False) as p:
            assert np.array_equal(labels, p["node_labels"])
            for part in ("train", "val", "test"):
                assert np.array_equal(masks[part], p[f"{part}_masks"][0])
    assert len(manifest["records"]) == 24
    derived = {(r["depth"], r["seed"], r["arm"]): r for r in manifest["records"]}
    locked = {(r["depth"], r["seed"], r["arm"]): r for r in lock["records"]}
    audited = {(r["depth"], r["seed"], r["arm"]): r for r in audit["records"]}
    assert len(derived) == len(locked) == len(audited) == 24
    scores = {}
    for depth in DEPTHS:
        for seed in SEEDS:
            initials = {}
            for arm in ARMS:
                key = depth, seed, arm
                run = ROOT / "results/roman" / f"depth{depth}" / f"seed{seed}" / arm
                test = ROOT / "test_scores" / f"depth{depth}" / f"seed{seed}" / arm
                row, score = read(run / "result.json"), read(test / "test_result.json")
                assert row["protocol"] == score["protocol"] == PROTOCOL
                assert row["source_freeze_sha256"] == score["source_freeze_sha256"] == source_sha
                assert (row["depth"], row["seed"], row["arm"]) == key
                assert (score["depth"], score["seed"], score["arm"]) == key
                assert row["official_mask"] == 0 and row["epochs_completed"] == 1000
                if key == (2, 2, "sync"):
                    assert row["recovery"]["training_weights_or_trace_recomputed"] is False
                    for name, digest in amendment["failed_cell"]["inprogress_artifact_sha256"].items():
                        assert row["artifact_sha256"][name] == digest
                prior_key = f"{depth}/{seed}/{arm}"
                if prior_key in amendment["prior_complete_result_sha256"]:
                    assert sha(run / "result.json") == amendment["prior_complete_result_sha256"][prior_key]
                assert score["validation_lock_sha256"] == sha(ROOT / "validation_lock.json")
                assert sha(run / "result.json") == derived[key]["result_sha256"] == locked[key]["result_sha256"]
                assert sha(test / "test_result.json") == derived[key]["test_result_sha256"] == audited[key]["test_result_sha256"]
                assert row["artifact_sha256"]["selected_validation.npz"] == derived[key]["original_validation_sha256"]
                assert score["test_predictions_sha256"] == derived[key]["original_test_sha256"]
                assert score["selected_checkpoint_sha256"] == locked[key]["checkpoint_sha256"]
                assert sha(run / "validation_trace.csv") == row["artifact_sha256"]["validation_trace.csv"]
                trace_check(run / "validation_trace.csv", row, arm)
                initials[arm] = row["initialization"]
                decisions = run / "selected_decisions.npz"
                assert sha(decisions) == derived[key]["derived_decisions_sha256"]
                with np.load(decisions, allow_pickle=False) as p:
                    for part in ("valid", "test"):
                        idx = p[f"{part}_indices"]
                        y = p[f"{part}_labels"]
                        member = p[f"{part}_member_pred"]
                        pred = p[f"{part}_pool_pred"]
                        official = np.flatnonzero(masks["val" if part == "valid" else "test"])
                        assert np.array_equal(idx, official)
                        assert np.array_equal(y, labels[idx])
                        assert member.shape == (4, len(idx)) and pred.shape == (len(idx),)
                        assert np.issubdtype(member.dtype, np.integer)
                        assert np.issubdtype(pred.dtype, np.integer)
                        assert np.all((member >= 0) & (member < 18))
                        assert np.all((pred >= 0) & (pred < 18))
                        acc = float(np.mean(pred == y))
                        target = row if part == "valid" else score
                        assert abs(acc - target[f"{part}_accuracy"]) < 1e-6
                        if part == "test":
                            assert abs(acc - audited[key]["test_accuracy"]) < 1e-6
                            scores[key] = acc
            anchor_initial = initials["tied"]
            for arm in ARMS[1:]:
                for name in ("canonical_projector_state_sha256", "python_rng_sha256",
                             "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
                    assert initials[arm][name] == anchor_initial[name]
    assert len(scores) == 24
    paired = []
    max_pair_reporting_difference_pp = 0.0
    for depth in DEPTHS:
        for seed in SEEDS:
            a = lambda arm: scores[depth, seed, arm]
            item = {"depth": depth, "seed": seed,
                    "tied_minus_untied_pp": 100 * (a("tied") - a("untied")),
                    "tied_minus_sync_pp": 100 * (a("tied") - a("sync")),
                    "tied_minus_norm_sync_pp": 100 * (a("tied") - a("norm_sync")),
                    "sync_minus_norm_sync_pp": 100 * (a("sync") - a("norm_sync"))}
            original = next(r for r in audit["paired"] if
                            r["depth"] == depth and r["seed"] == seed)
            differences = [abs(item[k] - original[k]) for k in item
                           if k not in ("depth", "seed")]
            max_pair_reporting_difference_pp = max(
                max_pair_reporting_difference_pp, *differences)
            assert all(value <= PAIRED_REPORTING_TOL_PP for value in differences)
            paired.append(item)
    out = {"status": "COMPLETE_24_CELL_LOCAL_SCORE_AUDIT_PASS",
           "scope": "post hoc one Roman graph, official mask0, two depths and three seeds",
           "source_freeze_sha256": source_sha,
           "validation_lock_sha256": sha(ROOT / "validation_lock.json"),
           "complete_score_audit_sha256": sha(ROOT / "complete_score_audit.json"),
           "public_npz_checked": args.public_npz is not None,
           "max_pair_float32_reporting_difference_pp": max_pair_reporting_difference_pp,
           "pair_float32_reporting_tolerance_pp": PAIRED_REPORTING_TOL_PP,
           "paired": paired}
    (ROOT / "local_compact_score_audit.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("COMPLETE_24_CELL_LOCAL_SCORE_AUDIT_PASS", len(scores),
          "paired", len(paired))


if __name__ == "__main__":
    main()
