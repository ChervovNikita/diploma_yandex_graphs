"""Prospective, validation-only numerical verification amendment to frozen Roman V3.

The frozen V3 sources, optimizer, checkpoint selection, training schedule, and
NORM-SYNC applied-update gate are never edited. The original 1e-5 selected
GPU-collapse gate failed after a complete depth-2/seed-2/SYNC training trace.
Five validation-only GPU replays showed 1.14e-5 max drift, comparable to
same-model GPU replay drift, with equal decisions; a CPU replay was bitwise
equal. This overlay uses a predeclared 1e-4 GPU diagnostic bound (the V3
five-step SGD logit bound) plus exact CPU member-logit/state equality and
equal GPU decisions. No held-out test outcome informed this amendment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE / "roman_mechanism_v3_prepared"
sys.path.insert(0, str(STUDY))
import mechanism as mech  # noqa: E402
import roman_mechanism as study  # noqa: E402
import tuning as base  # noqa: E402
import verify_roman_mechanism as original_verify  # noqa: E402

MANIFEST = STUDY / "ROMAN_V3_VERIFICATION_AMENDMENT_FREEZE.json"
DIAG = HERE / "ROMAN_MECHANISM_V3_COLLAPSE_DIAG_DEPTH2_SEED2_SYNC.json"
ORIGINAL_FREEZE_SHA = "489451d58772c073e7be04bd1d1f7859644f56a6626d9b4a14cd6f752acc0088"
PREFLIGHT_SHA = "b8ef89127561627096fb3e1d66bba2f36a6aec628b247be3e1f37ea1d2cfa75c"
FAILED_KEY = (2, 2, "sync")
GPU_DIAGNOSTIC_TOL = 1e-4
PRIOR_COMPLETE = {(2, seed, arm) for seed in (0, 1)
                  for arm in study.ARMS} | {(2, 2, arm) for arm in ("tied", "untied")}
FAILURE_FILES = ("checkpoint.pt", "initial_logits.npy", "validation_trace.csv")


def amended_collapse(model, bundle, device: torch.device) -> dict:
    """Check literal state mapping, CPU functional identity, GPU decision identity."""
    assert device.type == "cuda"
    assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
    mech.assert_equal_graph_weights(model)
    collapsed = mech.collapse_model(model, bundle, device)
    mapped = mech.collapsed_state(model)
    assert set(mapped) == set(collapsed.state_dict())
    assert all(torch.equal(value, collapsed.state_dict()[key])
               for key, value in mapped.items())
    model.eval()
    collapsed.eval()
    with torch.no_grad():
        original = torch.stack(base.member_logits(model, "untied", bundle))[:, bundle.valid_idx]
        compact = torch.stack(base.member_logits(collapsed, "tied", bundle))[:, bundle.valid_idx]
    gpu_max = float((original - compact).abs().max().item())
    pooled_mismatch = int((original.mean(0).argmax(-1) !=
                           compact.mean(0).argmax(-1)).sum().item())
    member_mismatch = int((original.argmax(-1) != compact.argmax(-1)).sum().item())
    assert math.isfinite(gpu_max) and gpu_max <= GPU_DIAGNOSTIC_TOL
    assert pooled_mismatch == member_mismatch == 0

    cpu = torch.device("cpu")
    cpu_bundle, _ = base.load_graph("roman", cpu, include_test=False)
    assert not hasattr(cpu_bundle, "test_idx_cpu") and not hasattr(cpu_bundle, "test_y_cpu")
    cpu_model, _ = base.make_model("untied", cpu_bundle, cpu)
    cpu_model.load_state_dict(model.state_dict(), strict=True)
    mech.assert_equal_graph_weights(cpu_model)
    cpu_compact = mech.collapse_model(cpu_model, cpu_bundle, cpu)
    cpu_mapped = mech.collapsed_state(cpu_model)
    assert set(cpu_mapped) == set(cpu_compact.state_dict())
    assert all(torch.equal(value, cpu_compact.state_dict()[key])
               for key, value in cpu_mapped.items())
    cpu_model.eval()
    cpu_compact.eval()
    with torch.no_grad():
        cpu_original = torch.stack(base.member_logits(
            cpu_model, "untied", cpu_bundle))[:, cpu_bundle.valid_idx]
        cpu_collapsed = torch.stack(base.member_logits(
            cpu_compact, "tied", cpu_bundle))[:, cpu_bundle.valid_idx]
    assert torch.equal(cpu_original, cpu_collapsed)
    return {"max_abs_validation_member_logit_difference": gpu_max,
            "pooled_validation_decision_mismatches": pooled_mismatch,
            "member_validation_decision_mismatches": member_mismatch,
            "collapsed_parameter_count": sum(p.numel() for p in collapsed.parameters()),
            "collapsed_state_sha256": base.state_sha(collapsed.state_dict()),
            "gpu_diagnostic_tolerance": GPU_DIAGNOSTIC_TOL,
            "cpu_member_logits_bitwise_equal": True,
            "cpu_collapsed_state_exactly_mapped": True,
            "gpu_collapsed_state_exactly_mapped": True,
            "amendment_source_sha256": study.sha(Path(__file__))}


def prepare() -> None:
    assert not MANIFEST.exists()
    assert not (STUDY / "validation_lock.json").exists()
    assert not (STUDY / "test_scores").exists()
    assert study.check_freeze() == ORIGINAL_FREEZE_SHA
    assert study.sha(STUDY / "preflight_cuda.json") == PREFLIGHT_SHA
    diagnosis = json.loads(DIAG.read_text())
    assert diagnosis["status"] == "VALIDATION_ONLY_COLLAPSE_DIAGNOSIS"
    assert diagnosis["source_freeze_sha256"] == ORIGINAL_FREEZE_SHA
    assert (diagnosis["depth"], diagnosis["seed"], diagnosis["arm"]) == FAILED_KEY
    assert all(row["pooled_decision_mismatches"] == 0 for row in diagnosis["paired_replays"])
    assert diagnosis["cpu_paired_replay"]["member_logit_max_abs_difference"] == 0
    assert max(row["member_logit_max_abs_difference"] for row in diagnosis["paired_replays"]) > mech.COLLAPSE_LOGIT_TOL
    assert max(row["member_logit_max_abs_difference"] for row in diagnosis["paired_replays"]) < GPU_DIAGNOSTIC_TOL
    completed = {}
    for depth, seed, arm in sorted(PRIOR_COMPLETE):
        folder = study.run_dir(depth, seed, arm)
        row = json.loads((folder / "result.json").read_text())
        assert row["source_freeze_sha256"] == ORIGINAL_FREEZE_SHA
        assert row["selected_collapse"] is None or row["selected_collapse"][
            "max_abs_validation_member_logit_difference"] <= mech.COLLAPSE_LOGIT_TOL
        completed[f"{depth}/{seed}/{arm}"] = study.sha(folder / "result.json")
    found = {(d, s, a) for d in study.DEPTHS for s in study.SEEDS for a in study.ARMS
             if (study.run_dir(d, s, a) / "result.json").exists()}
    assert found == PRIOR_COMPLETE
    failed = study.run_dir(*FAILED_KEY).with_name("sync.inprogress")
    assert failed.is_dir() and set(p.name for p in failed.iterdir()) == set(FAILURE_FILES)
    failure_hashes = {name: study.sha(failed / name) for name in FAILURE_FILES}
    assert failure_hashes["checkpoint.pt"] == diagnosis["checkpoint_sha256"]
    log = STUDY / "train_v3.log"
    assert "Collapsed graph stack changes validation logits or decisions" in log.read_text()
    study.write_json(MANIFEST, {
        "status": "PROSPECTIVE_VALIDATION_ONLY_VERIFICATION_AMENDMENT_V1",
        "original_v3_source_freeze_sha256": ORIGINAL_FREEZE_SHA,
        "original_preflight_sha256": PREFLIGHT_SHA,
        "amendment_source_sha256": study.sha(Path(__file__)),
        "diagnosis_sha256": study.sha(DIAG),
        "original_failure_log_sha256": study.sha(log),
        "prior_complete_result_sha256": completed,
        "failed_cell": {"depth": 2, "seed": 2, "arm": "sync",
                        "inprogress_artifact_sha256": failure_hashes},
        "gpu_diagnostic_tolerance": GPU_DIAGNOSTIC_TOL,
        "original_gpu_collapse_tolerance": mech.COLLAPSE_LOGIT_TOL,
        "required_cpu_member_logits_bitwise_equal": True,
        "required_gpu_member_and_pooled_decisions_equal": True,
        "training_and_selection_changed": False,
        "norm_applied_update_gate_changed": False,
        "test_scores_opened_at_freeze": False,
        "failure_artifacts_preserved": True})
    print("VERIFICATION_AMENDMENT_FROZEN", study.sha(MANIFEST), flush=True)


def check_manifest(pretest: bool = True) -> dict:
    source_sha = study.check_freeze()
    assert source_sha == ORIGINAL_FREEZE_SHA
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == "PROSPECTIVE_VALIDATION_ONLY_VERIFICATION_AMENDMENT_V1"
    assert manifest["original_v3_source_freeze_sha256"] == source_sha
    assert manifest["original_preflight_sha256"] == study.sha(STUDY / "preflight_cuda.json")
    assert manifest["amendment_source_sha256"] == study.sha(Path(__file__))
    assert manifest["diagnosis_sha256"] == study.sha(DIAG)
    assert manifest["original_failure_log_sha256"] == study.sha(STUDY / "train_v3.log")
    assert manifest["gpu_diagnostic_tolerance"] == GPU_DIAGNOSTIC_TOL
    for key, digest in manifest["prior_complete_result_sha256"].items():
        depth, seed, arm = key.split("/")
        assert study.sha(study.run_dir(int(depth), int(seed), arm) / "result.json") == digest
    failed = study.run_dir(*FAILED_KEY).with_name("sync.inprogress")
    for name, digest in manifest["failed_cell"]["inprogress_artifact_sha256"].items():
        assert study.sha(failed / name) == digest
    if pretest:
        assert not (STUDY / "test_scores").exists()
    return manifest


def recover_failed(device: torch.device) -> None:
    manifest = check_manifest()
    final = study.run_dir(*FAILED_KEY)
    work = final.with_name("sync.inprogress")
    assert not final.exists()
    base.configure(2)
    bundle, _ = base.load_graph("roman", device, include_test=False)
    assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
    base.seed_all(2)
    model, canonical = study.make_arm("sync", bundle, device)
    initial = base.initial_audit(model, "untied", bundle, 2, device, canonical)
    initial["saved_initial_logits_sha256"] = study.sha(work / "initial_logits.npy")
    archived_initial = np.load(work / "initial_logits.npy", allow_pickle=False)
    assert archived_initial.shape == (4, 22662, bundle.classes)
    model.eval()
    with torch.no_grad():
        replay_initial = torch.stack(base.member_logits(model, "untied", bundle)).cpu().numpy()
    assert float(np.max(np.abs(archived_initial - replay_initial))) <= base.INITIAL_LOGIT_TOL
    assert np.array_equal(archived_initial.argmax(-1), replay_initial.argmax(-1))

    # Recreate the original first step from the same canonical initialization
    # and seed; the original completed training trace/checkpoint are preserved.
    optimizer = study.optimizer_for(model)
    mech.one_update(model, "sync", bundle, optimizer)
    first_moments = mech.moment_audit(model, optimizer)
    mech.assert_equal_graph_weights(model)
    checkpoint = torch.load(work / "checkpoint.pt", map_location=device, weights_only=True)
    assert (checkpoint["depth"], checkpoint["seed"], checkpoint["arm"]) == FAILED_KEY
    with (work / "validation_trace.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == study.EPOCHS
    best_acc, best_ce, best_epoch = -1.0, math.inf, 0
    for epoch, row in enumerate(rows, 1):
        assert int(row["epoch"]) == epoch
        acc, ce = float(row["valid_accuracy"]), float(row["valid_ce"])
        assert math.isfinite(acc) and math.isfinite(ce)
        improved = acc > best_acc or (acc == best_acc and ce < best_ce)
        assert int(row["selected_now"]) == int(improved)
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, epoch
    assert checkpoint["epoch"] == best_epoch
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    acc, ce, logits, _ = base.evaluate(model, "untied", bundle,
                                       bundle.valid_idx, bundle.valid_y)
    assert abs(acc - best_acc) < 1e-7 and abs(ce - best_ce) < 1e-6
    collapse = amended_collapse(model, bundle, device)
    temporary = final.with_name("sync.amended-finalizing")
    assert not temporary.exists()
    shutil.copytree(work, temporary)
    np.savez_compressed(temporary / "selected_validation.npz",
                        member_logits=logits.detach().cpu().numpy(),
                        node_index=bundle.valid_idx.cpu().numpy(),
                        y_true=bundle.valid_y.cpu().numpy())
    artifacts = (*FAILURE_FILES, "selected_validation.npz")
    result = {"protocol": study.PROTOCOL,
              "source_freeze_sha256": ORIGINAL_FREEZE_SHA,
              "dataset": "roman", "official_mask": 0,
              "depth": 2, "seed": 2, "arm": "sync",
              "epochs_completed": study.EPOCHS,
              "selected_epoch": best_epoch,
              "valid_accuracy": acc, "valid_ce": ce,
              "parameter_count": initial["parameter_count"],
              "initialization": initial,
              "first_step_moments": first_moments,
              "selected_collapse": collapse,
              "train_seconds": None,
              "recovery": {"status": "COMPLETED_TRAINING_RECOVERED_AFTER_NUMERICAL_VALIDATION_GATE",
                           "amendment_freeze_sha256": study.sha(MANIFEST),
                           "original_failure_log_sha256": manifest["original_failure_log_sha256"],
                           "original_inprogress_artifact_sha256": manifest["failed_cell"][
                               "inprogress_artifact_sha256"],
                           "first_step_moments_recomputed_from_frozen_initialization": True,
                           "training_weights_or_trace_recomputed": False},
              "artifact_sha256": {name: study.sha(temporary / name) for name in artifacts}}
    study.write_json(temporary / "result.json", result)
    temporary.rename(final)
    print("FAILED_CELL_VALIDATION_RECOVERED", best_epoch, acc, flush=True)


def train_remaining(device: torch.device) -> None:
    check_manifest()
    assert study.run_dir(*FAILED_KEY).is_dir(), "Run recover first"
    original = mech.collapse_validation_audit
    try:
        mech.collapse_validation_audit = amended_collapse
        study.train(device)
    finally:
        mech.collapse_validation_audit = original


def validate(device: torch.device) -> None:
    manifest = check_manifest()
    gate = original_verify.check_preflight(ORIGINAL_FREEZE_SHA)
    assert not (STUDY / "validation_lock.json").exists()
    expected = original_verify.expected_run_dirs()
    failed_work = study.run_dir(*FAILED_KEY).with_name("sync.inprogress")
    found = {p for p in (STUDY / "results/roman").glob("depth*/seed*/*")
             if p.is_dir() and p != failed_work}
    assert found == expected
    records = []
    for depth in study.DEPTHS:
        base.configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=False)
        assert descriptor == gate["data_descriptors"][str(depth)]
        assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
        for seed in study.SEEDS:
            initial_logits, initials = {}, {}
            for arm in study.ARMS:
                folder = study.run_dir(depth, seed, arm)
                row = json.loads((folder / "result.json").read_text())
                assert row["protocol"] == study.PROTOCOL
                assert row["source_freeze_sha256"] == ORIGINAL_FREEZE_SHA
                assert (row["depth"], row["seed"], row["arm"]) == (depth, seed, arm)
                assert row["official_mask"] == 0 and row["epochs_completed"] == study.EPOCHS
                assert set(row["artifact_sha256"]) == set(FAILURE_FILES) | {"selected_validation.npz"}
                for name, digest in row["artifact_sha256"].items():
                    assert study.sha(folder / name) == digest
                original_verify.trace_check(folder / "validation_trace.csv", row, arm)
                base.seed_all(seed)
                model, canonical = study.make_arm(arm, bundle, device)
                fresh = base.initial_audit(model, study.eval_arm(arm), bundle,
                                           seed, device, canonical)
                stored = row["initialization"]
                for field in ("canonical_projector_state_sha256", "python_rng_sha256",
                              "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256",
                              "parameter_count"):
                    assert fresh[field] == stored[field], (depth, seed, arm, field)
                archived = np.load(folder / "initial_logits.npy", allow_pickle=False)
                assert archived.shape == (4, 22662, bundle.classes)
                assert np.isfinite(archived).all()
                assert study.sha(folder / "initial_logits.npy") == stored["saved_initial_logits_sha256"]
                model.eval()
                with torch.no_grad():
                    replay_initial = torch.stack(base.member_logits(
                        model, study.eval_arm(arm), bundle)).cpu().numpy()
                assert float(np.max(np.abs(archived - replay_initial))) <= base.INITIAL_LOGIT_TOL
                assert np.array_equal(archived.argmax(-1), replay_initial.argmax(-1))
                initial_logits[arm] = archived
                initials[arm] = stored
                checkpoint = torch.load(folder / "checkpoint.pt", map_location=device,
                                        weights_only=True)
                assert (checkpoint["depth"], checkpoint["seed"], checkpoint["arm"],
                        checkpoint["epoch"]) == (depth, seed, arm, row["selected_epoch"])
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                if arm in ("sync", "norm_sync"):
                    mech.assert_equal_graph_weights(model)
                acc, ce, logits, _ = base.evaluate(model, study.eval_arm(arm), bundle,
                                                   bundle.valid_idx, bundle.valid_y)
                assert abs(acc - row["valid_accuracy"]) < 1e-7
                assert abs(ce - row["valid_ce"]) < 1e-4
                with np.load(folder / "selected_validation.npz", allow_pickle=False) as archive:
                    assert np.array_equal(archive["node_index"], bundle.valid_idx.cpu().numpy())
                    assert np.array_equal(archive["y_true"], bundle.valid_y.cpu().numpy())
                    saved = archive["member_logits"]
                replay = logits.detach().cpu().numpy()
                assert saved.shape == replay.shape
                assert float(np.max(np.abs(saved - replay))) <= original_verify.REPLAY_TOL
                assert np.array_equal(saved.argmax(-1), replay.argmax(-1))
                assert np.array_equal(saved.mean(0).argmax(-1), replay.mean(0).argmax(-1))
                collapse = None
                if arm in ("sync", "norm_sync"):
                    collapse = amended_collapse(model, bundle, device)
                    assert collapse["collapsed_state_sha256"] == row["selected_collapse"][
                        "collapsed_state_sha256"]
                    original_limit = mech.COLLAPSE_LOGIT_TOL if (depth, seed, arm) in PRIOR_COMPLETE else GPU_DIAGNOSTIC_TOL
                    assert row["selected_collapse"][
                        "max_abs_validation_member_logit_difference"] <= original_limit
                records.append({"depth": depth, "seed": seed, "arm": arm,
                                "result_sha256": study.sha(folder / "result.json"),
                                "checkpoint_sha256": row["artifact_sha256"]["checkpoint.pt"],
                                "selected_epoch": row["selected_epoch"],
                                "valid_accuracy": acc, "valid_ce": ce,
                                "parameter_count": row["parameter_count"],
                                "collapse_replay": collapse})
            anchor = initials["tied"]
            for arm in study.ARMS[1:]:
                for field in ("canonical_projector_state_sha256", "python_rng_sha256",
                              "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256"):
                    assert initials[arm][field] == anchor[field]
                assert float(np.max(np.abs(initial_logits[arm] -
                                           initial_logits["tied"]))) <= base.INITIAL_LOGIT_TOL
    assert len(records) == 24
    study.write_json(STUDY / "validation_lock.json", {
        "status": "COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK",
        "source_freeze_sha256": ORIGINAL_FREEZE_SHA,
        "preflight_sha256": PREFLIGHT_SHA,
        "verification_amendment_sha256": study.sha(MANIFEST),
        "amendment_source_sha256": study.sha(Path(__file__)),
        "original_failure_artifacts_preserved": True,
        "records": records, "test_scored": False})
    print("COMPLETE_24_CELL_VALIDATION_REPLAY_LOCK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "recover", "train", "validate",
                                            "score", "audit-test"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
        return
    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    if args.stage == "recover":
        recover_failed(device)
    elif args.stage == "train":
        train_remaining(device)
    elif args.stage == "validate":
        validate(device)
    elif args.stage == "score":
        check_manifest(pretest=False)
        original_verify.score(device)
    else:
        check_manifest(pretest=False)
        original_verify.audit_test(device)


if __name__ == "__main__":
    main()
