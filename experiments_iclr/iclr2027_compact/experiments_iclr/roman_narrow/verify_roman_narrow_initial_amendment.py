"""Frozen numerical overlay for narrow-control initial member decisions only.

The original independent validator stopped before test scoring on one member
argmax at a 2.38e-7 margin. The same-model GPU forward flips that decision;
all pooled decisions agree. Training, checkpoint selection, selected
validation replay, once-only test scoring, and test replay are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "roman_mechanism_v3_prepared"
sys.path.insert(0, str(ORIGINAL))
import tuning as base  # noqa: E402
import verify_roman_mechanism as original_trace  # noqa: E402
import roman_narrow_untied as narrow  # noqa: E402
import verify_roman_narrow_untied as original_verify  # noqa: E402

MANIFEST = narrow.ROOT / "INITIAL_DECISION_AMENDMENT_FREEZE.json"
DIAG = narrow.ROOT / "initial_replay_all6_diagnostic.json"
FAILURE_LOG = narrow.ROOT / "validate.log"
REPLAY_TOL = original_verify.REPLAY_TOL


def read(path: Path):
    return json.loads(path.read_text())


def prepare() -> None:
    assert not MANIFEST.exists()
    assert not (narrow.ROOT / "validation_lock.json").exists()
    assert not (narrow.ROOT / "test_scores").exists()
    freeze_sha = narrow.check_freeze()
    diagnosis = read(DIAG)
    assert diagnosis["status"] == "ALL_6_INITIAL_REPLAY_VALIDATION_ONLY_DIAGNOSTIC"
    assert diagnosis["source_freeze_sha256"] == freeze_sha
    assert diagnosis["test_ids_or_labels_accessed"] is False
    indexed = {(r["depth"], r["seed"]): r for r in diagnosis["records"]}
    assert set(indexed) == {(d, s) for d in narrow.DEPTHS for s in narrow.SEEDS}
    for item in indexed.values():
        assert item["reconstructed_untied_initial_state_bitwise_equal"] is True
        assert item["cpu_model_state_bitwise_equal_to_gpu_state"] is True
        assert all(r["max_abs_member_logit_difference"] <= base.INITIAL_LOGIT_TOL
                   and r["pooled_decision_mismatch_count"] == 0
                   for r in item["gpu_vs_archive"])
        assert all(r["max_abs_member_logit_difference"] == 0 and
                   r["member_decision_mismatch_count"] == 0
                   for r in item["cpu_vs_cpu"])
    assert "assert np.array_equal(archived.argmax(-1), initial_replay.argmax(-1))" in (
        HERE / "verify_roman_narrow_untied.py").read_text()
    assert "AssertionError" in FAILURE_LOG.read_text()
    results = {f"{d}/{s}": narrow.sha(narrow.run_dir(d, s) / "result.json")
               for d in narrow.DEPTHS for s in narrow.SEEDS}
    narrow.write_json(MANIFEST, {
        "status": "PROSPECTIVE_NARROW_INITIAL_DECISION_VERIFICATION_AMENDMENT",
        "scope": "initial member decisions only; selected validation/test gates unchanged",
        "original_narrow_source_freeze_sha256": freeze_sha,
        "original_verifier_sha256": narrow.sha(HERE / "verify_roman_narrow_untied.py"),
        "amended_verifier_sha256": narrow.sha(Path(__file__)),
        "all_six_validation_only_diagnosis_sha256": narrow.sha(DIAG),
        "original_validation_failure_log_sha256": narrow.sha(FAILURE_LOG),
        "six_trained_result_sha256": results,
        "initial_member_logit_tolerance": base.INITIAL_LOGIT_TOL,
        "initial_member_flip_rule": "both archived and replay top-two margins <= 2*observed max absolute member-logit drift",
        "exact_pooled_initial_decisions_required": True,
        "repeated_cpu_member_logits_bitwise_equal_required": True,
        "selected_validation_decision_gate_changed": False,
        "test_decision_gate_changed": False,
        "training_or_selection_changed": False,
        "test_scores_opened_at_freeze": False})
    print("NARROW_INITIAL_AMENDMENT_FROZEN", narrow.sha(MANIFEST), flush=True)


def check_manifest(pretest: bool = True) -> dict:
    freeze_sha = narrow.check_freeze()
    m = read(MANIFEST)
    assert m["status"] == "PROSPECTIVE_NARROW_INITIAL_DECISION_VERIFICATION_AMENDMENT"
    assert m["original_narrow_source_freeze_sha256"] == freeze_sha
    assert m["original_verifier_sha256"] == narrow.sha(
        HERE / "verify_roman_narrow_untied.py")
    assert m["amended_verifier_sha256"] == narrow.sha(Path(__file__))
    assert m["all_six_validation_only_diagnosis_sha256"] == narrow.sha(DIAG)
    assert m["original_validation_failure_log_sha256"] == narrow.sha(FAILURE_LOG)
    assert m["initial_member_logit_tolerance"] == base.INITIAL_LOGIT_TOL
    for key, digest in m["six_trained_result_sha256"].items():
        depth, seed = map(int, key.split("/"))
        assert narrow.sha(narrow.run_dir(depth, seed) / "result.json") == digest
    if pretest:
        assert not (narrow.ROOT / "test_scores").exists()
    return m


def initial_guard(archived: np.ndarray, replay: np.ndarray) -> dict:
    assert archived.shape == replay.shape
    delta = float(np.max(np.abs(archived - replay)))
    assert math.isfinite(delta) and delta <= base.INITIAL_LOGIT_TOL
    old_winner, new_winner = archived.argmax(-1), replay.argmax(-1)
    assert np.array_equal(archived.mean(0).argmax(-1),
                          replay.mean(0).argmax(-1))
    changed = np.argwhere(old_winner != new_winner)
    flips = []
    for member, node in changed:
        old_top = np.argsort(archived[member, node])[-2:][::-1]
        new_top = np.argsort(replay[member, node])[-2:][::-1]
        old_margin = float(archived[member, node, old_top[0]] -
                           archived[member, node, old_top[1]])
        new_margin = float(replay[member, node, new_top[0]] -
                           replay[member, node, new_top[1]])
        assert old_margin <= 2 * delta + 1e-12
        assert new_margin <= 2 * delta + 1e-12
        flips.append({"member": int(member), "node": int(node),
                      "archived_winner": int(old_winner[member, node]),
                      "replay_winner": int(new_winner[member, node]),
                      "archived_top2_class": old_top.tolist(),
                      "replay_top2_class": new_top.tolist(),
                      "archived_top2_margin": old_margin,
                      "replay_top2_margin": new_margin})
    return {"max_abs_initial_member_logit_difference": delta,
            "pooled_initial_decision_mismatches": 0,
            "member_initial_decision_mismatches": len(flips),
            "member_flips": flips,
            "all_flip_margins_within_twice_observed_drift": True}


def validate(device: torch.device) -> None:
    amendment = check_manifest()
    freeze_sha = narrow.check_freeze()
    gate = original_verify.gate_check(freeze_sha)
    assert not (narrow.ROOT / "validation_lock.json").exists()
    expected = {narrow.run_dir(d, s) for d in narrow.DEPTHS for s in narrow.SEEDS}
    found = {p for p in (narrow.ROOT / "results/roman").glob("depth*/seed*/*")
             if p.is_dir()}
    assert found == expected
    records = []
    cpu = torch.device("cpu")
    for depth in narrow.DEPTHS:
        narrow.configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=False)
        cpu_bundle, cpu_descriptor = base.load_graph("roman", cpu, include_test=False)
        assert descriptor == cpu_descriptor == gate["data_descriptors"][str(depth)]
        assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
        assert not hasattr(cpu_bundle, "test_idx_cpu") and not hasattr(cpu_bundle, "test_y_cpu")
        for seed in narrow.SEEDS:
            folder = narrow.run_dir(depth, seed)
            row = read(folder / "result.json")
            assert row["protocol"] == narrow.PROTOCOL
            assert row["source_freeze_sha256"] == freeze_sha
            assert (row["depth"], row["seed"], row["arm"], row["width"]) == (
                depth, seed, "untied", narrow.WIDTHS[depth])
            assert row["official_mask"] == 0 and row["epochs_completed"] == narrow.EPOCHS
            assert row["parameter_count"] == narrow.PARAMETERS[depth]
            assert set(row["artifact_sha256"]) == {
                "checkpoint.pt", "validation_trace.csv",
                "initial_logits.npy", "selected_validation.npz"}
            for name, digest in row["artifact_sha256"].items():
                assert narrow.sha(folder / name) == digest
            original_trace.trace_check(folder / "validation_trace.csv", row, "untied")
            base.seed_all(seed)
            model, canonical = narrow.make_model(bundle, device)
            fresh = base.initial_audit(model, "untied", bundle,
                                       seed, device, canonical)
            stored = row["initialization"]
            for field in ("canonical_projector_state_sha256", "python_rng_sha256",
                          "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256",
                          "parameter_count"):
                assert fresh[field] == stored[field]
            full_initial_hash = base.state_sha(model.state_dict())
            diagnosis = next(r for r in read(DIAG)["records"]
                             if r["depth"] == depth and r["seed"] == seed)
            assert canonical == diagnosis["canonical_full_tied_initial_state_sha256"]
            assert full_initial_hash == diagnosis["untied_initial_state_sha256"]
            archived = np.load(folder / "initial_logits.npy", allow_pickle=False)
            assert archived.shape == (4, 22662, bundle.classes)
            assert np.isfinite(archived).all()
            assert narrow.sha(folder / "initial_logits.npy") == stored[
                "saved_initial_logits_sha256"]
            model.eval()
            with torch.no_grad():
                replay_initial = torch.stack(base.member_logits(
                    model, "untied", bundle)).cpu().numpy()
            initial_replay = initial_guard(archived, replay_initial)
            cpu_model, _ = base.make_model("untied", cpu_bundle, cpu)
            cpu_model.load_state_dict(model.state_dict(), strict=True)
            assert all(torch.equal(value.cpu(), cpu_model.state_dict()[name])
                       for name, value in model.state_dict().items())
            cpu_model.eval()
            with torch.no_grad():
                cpu_one = torch.stack(base.member_logits(cpu_model, "untied", cpu_bundle))
                cpu_two = torch.stack(base.member_logits(cpu_model, "untied", cpu_bundle))
            assert torch.equal(cpu_one, cpu_two)
            initial_replay["repeated_cpu_member_logits_bitwise_equal"] = True
            initial_replay["canonical_full_state_sha256"] = canonical
            initial_replay["untied_full_state_sha256"] = full_initial_hash
            checkpoint = torch.load(folder / "checkpoint.pt", map_location=device,
                                    weights_only=True)
            assert (checkpoint["depth"], checkpoint["seed"], checkpoint["arm"],
                    checkpoint["width"], checkpoint["epoch"]) == (
                        depth, seed, "untied", narrow.WIDTHS[depth],
                        row["selected_epoch"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            acc, ce, logits, _ = base.evaluate(model, "untied", bundle,
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
            records.append({"depth": depth, "seed": seed,
                            "width": narrow.WIDTHS[depth],
                            "parameter_count": row["parameter_count"],
                            "selected_epoch": row["selected_epoch"],
                            "valid_accuracy": acc, "valid_ce": ce,
                            "result_sha256": narrow.sha(folder / "result.json"),
                            "checkpoint_sha256": row["artifact_sha256"]["checkpoint.pt"],
                            "initial_replay": initial_replay})
            print("NARROW_AMENDED_VALIDATION_PASS", depth, seed, flush=True)
    assert len(records) == 6
    freeze = read(narrow.ROOT / "SOURCE_FREEZE.json")
    narrow.write_json(narrow.ROOT / "validation_lock.json", {
        "status": "COMPLETE_6_CELL_NARROW_VALIDATION_REPLAY_LOCK",
        "source_freeze_sha256": freeze_sha,
        "preflight_sha256": narrow.sha(narrow.ROOT / "preflight.json"),
        "width_lock_sha256": narrow.WIDTH_LOCK_SHA,
        "original_validation_lock_sha256": freeze["original_validation_lock_sha256"],
        "baseline_tied128_artifact_sha256": freeze["baseline_tied128_artifact_sha256"],
        "initial_decision_amendment_sha256": narrow.sha(MANIFEST),
        "initial_diagnosis_sha256": amendment["all_six_validation_only_diagnosis_sha256"],
        "records": records, "test_scored": False})
    print("COMPLETE_6_CELL_NARROW_VALIDATION_REPLAY_LOCK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "validate", "score", "audit-test"),
                        required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
        return
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    if args.stage == "validate":
        validate(device)
    elif args.stage == "score":
        check_manifest(pretest=False)
        original_verify.score(device)
    else:
        check_manifest(pretest=False)
        original_verify.audit_test(device)


if __name__ == "__main__":
    main()
