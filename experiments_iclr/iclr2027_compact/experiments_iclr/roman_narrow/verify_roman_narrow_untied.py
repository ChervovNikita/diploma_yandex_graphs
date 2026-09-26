"""Independent six-cell validation lock, once-only test score, and replay."""
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
import verify_roman_mechanism as original_verify  # noqa: E402
import roman_narrow_untied as narrow  # noqa: E402

REPLAY_TOL = 1e-4


def read(path: Path):
    return json.loads(path.read_text())


def gate_check(freeze_sha: str) -> dict:
    gate_path = narrow.ROOT / "preflight.json"
    gate = read(gate_path)
    assert gate["status"] == "ALL_6_NARROW_CUDA_PREFLIGHT_PASS"
    assert gate["source_freeze_sha256"] == freeze_sha
    assert {(r["depth"], r["seed"]) for r in gate["records"]} == {
        (d, s) for d in narrow.DEPTHS for s in narrow.SEEDS}
    for row in gate["records"]:
        assert row["width"] == narrow.WIDTHS[row["depth"]]
        assert row["parameter_count"] == narrow.PARAMETERS[row["depth"]]
        assert 0 <= row["repeat_initial_max_abs_logit_difference"] <= base.INITIAL_LOGIT_TOL
    return gate


def validate(device: torch.device) -> None:
    freeze_sha = narrow.check_freeze()
    gate = gate_check(freeze_sha)
    assert not (narrow.ROOT / "validation_lock.json").exists()
    assert not (narrow.ROOT / "test_scores").exists()
    expected = {narrow.run_dir(d, s) for d in narrow.DEPTHS for s in narrow.SEEDS}
    found = {p for p in (narrow.ROOT / "results/roman").glob("depth*/seed*/*")
             if p.is_dir()}
    assert found == expected
    records = []
    for depth in narrow.DEPTHS:
        narrow.configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=False)
        assert descriptor == gate["data_descriptors"][str(depth)]
        assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
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
            original_verify.trace_check(folder / "validation_trace.csv", row, "untied")
            base.seed_all(seed)
            model, canonical = narrow.make_model(bundle, device)
            fresh = base.initial_audit(model, "untied", bundle,
                                       seed, device, canonical)
            stored = row["initialization"]
            for field in ("canonical_projector_state_sha256", "python_rng_sha256",
                          "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256",
                          "parameter_count"):
                assert fresh[field] == stored[field]
            archived = np.load(folder / "initial_logits.npy", allow_pickle=False)
            assert archived.shape == (4, 22662, bundle.classes)
            assert np.isfinite(archived).all()
            assert narrow.sha(folder / "initial_logits.npy") == stored[
                "saved_initial_logits_sha256"]
            model.eval()
            with torch.no_grad():
                initial_replay = torch.stack(base.member_logits(
                    model, "untied", bundle)).cpu().numpy()
            assert float(np.max(np.abs(archived - initial_replay))) <= base.INITIAL_LOGIT_TOL
            assert np.array_equal(archived.argmax(-1), initial_replay.argmax(-1))
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
            records.append({"depth": depth, "seed": seed, "width": narrow.WIDTHS[depth],
                            "parameter_count": row["parameter_count"],
                            "selected_epoch": row["selected_epoch"],
                            "valid_accuracy": acc, "valid_ce": ce,
                            "result_sha256": narrow.sha(folder / "result.json"),
                            "checkpoint_sha256": row["artifact_sha256"]["checkpoint.pt"]})
            print("NARROW_VALIDATION_REPLAY_PASS", depth, seed, flush=True)
    assert len(records) == 6
    freeze = read(narrow.ROOT / "SOURCE_FREEZE.json")
    narrow.write_json(narrow.ROOT / "validation_lock.json", {
        "status": "COMPLETE_6_CELL_NARROW_VALIDATION_REPLAY_LOCK",
        "source_freeze_sha256": freeze_sha,
        "preflight_sha256": narrow.sha(narrow.ROOT / "preflight.json"),
        "width_lock_sha256": narrow.WIDTH_LOCK_SHA,
        "original_validation_lock_sha256": freeze[
            "original_validation_lock_sha256"],
        "baseline_tied128_artifact_sha256": freeze[
            "baseline_tied128_artifact_sha256"],
        "records": records, "test_scored": False})
    print("COMPLETE_6_CELL_NARROW_VALIDATION_REPLAY_LOCK", flush=True)


def score(device: torch.device) -> None:
    freeze_sha = narrow.check_freeze()
    gate = gate_check(freeze_sha)
    lock_path = narrow.ROOT / "validation_lock.json"
    lock = read(lock_path)
    assert lock["status"] == "COMPLETE_6_CELL_NARROW_VALIDATION_REPLAY_LOCK"
    assert lock["source_freeze_sha256"] == freeze_sha
    assert lock["preflight_sha256"] == narrow.sha(narrow.ROOT / "preflight.json")
    assert lock["test_scored"] is False and len(lock["records"]) == 6
    assert not (narrow.ROOT / "test_scores").exists()
    keyed = {(r["depth"], r["seed"]): r for r in lock["records"]}
    assert len(keyed) == 6
    for depth in narrow.DEPTHS:
        narrow.configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=True)
        assert descriptor == gate["data_descriptors"][str(depth)]
        idx, y = bundle.test_idx_cpu.to(device), bundle.test_y_cpu.to(device)
        for seed in narrow.SEEDS:
            folder = narrow.run_dir(depth, seed)
            assert narrow.sha(folder / "result.json") == keyed[depth, seed]["result_sha256"]
            assert narrow.sha(folder / "checkpoint.pt") == keyed[depth, seed]["checkpoint_sha256"]
            target = narrow.ROOT / "test_scores" / f"depth{depth}" / f"seed{seed}" / "untied"
            assert not target.exists()
            target.mkdir(parents=True)
            base.seed_all(seed)
            model, _ = narrow.make_model(bundle, device)
            checkpoint = torch.load(folder / "checkpoint.pt", map_location=device,
                                    weights_only=True)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            acc, ce, logits, _ = base.evaluate(model, "untied", bundle, idx, y)
            assert math.isfinite(acc) and math.isfinite(ce)
            np.savez_compressed(target / "test_predictions.npz",
                                member_logits=logits.detach().cpu().numpy(),
                                node_index=bundle.test_idx_cpu.numpy(),
                                y_true=bundle.test_y_cpu.numpy())
            narrow.write_json(target / "test_result.json", {
                "protocol": narrow.PROTOCOL,
                "source_freeze_sha256": freeze_sha,
                "validation_lock_sha256": narrow.sha(lock_path),
                "depth": depth, "seed": seed, "arm": "untied",
                "width": narrow.WIDTHS[depth],
                "selected_checkpoint_sha256": narrow.sha(folder / "checkpoint.pt"),
                "test_accuracy": acc, "test_ce": ce,
                "test_predictions_sha256": narrow.sha(target / "test_predictions.npz")})
            print("NARROW_TEST_SCORED", depth, seed, flush=True)
    print("ALL_6_NARROW_TEST_CELLS_SCORED", flush=True)


def audit_test(device: torch.device) -> None:
    freeze_sha = narrow.check_freeze()
    lock_path = narrow.ROOT / "validation_lock.json"
    lock = read(lock_path)
    assert lock["status"] == "COMPLETE_6_CELL_NARROW_VALIDATION_REPLAY_LOCK"
    assert lock["source_freeze_sha256"] == freeze_sha
    locked = {(r["depth"], r["seed"]): r for r in lock["records"]}
    assert len(locked) == 6
    expected = {narrow.ROOT / "test_scores" / f"depth{d}" / f"seed{s}" / "untied"
                for d in narrow.DEPTHS for s in narrow.SEEDS}
    found = {p for p in (narrow.ROOT / "test_scores").glob("depth*/seed*/*") if p.is_dir()}
    assert found == expected
    freeze = read(narrow.ROOT / "SOURCE_FREEZE.json")
    rows, contrasts = [], []
    for depth in narrow.DEPTHS:
        narrow.configure(depth)
        bundle, _ = base.load_graph("roman", device, include_test=True)
        idx, y = bundle.test_idx_cpu.to(device), bundle.test_y_cpu.to(device)
        for seed in narrow.SEEDS:
            folder = narrow.run_dir(depth, seed)
            target = narrow.ROOT / "test_scores" / f"depth{depth}" / f"seed{seed}" / "untied"
            report = read(target / "test_result.json")
            assert report["protocol"] == narrow.PROTOCOL
            assert report["source_freeze_sha256"] == freeze_sha
            assert report["validation_lock_sha256"] == narrow.sha(lock_path)
            assert (report["depth"], report["seed"], report["arm"],
                    report["width"]) == (depth, seed, "untied", narrow.WIDTHS[depth])
            assert narrow.sha(folder / "result.json") == locked[depth, seed]["result_sha256"]
            assert narrow.sha(folder / "checkpoint.pt") == locked[depth, seed]["checkpoint_sha256"]
            assert report["selected_checkpoint_sha256"] == locked[depth, seed]["checkpoint_sha256"]
            assert report["test_predictions_sha256"] == narrow.sha(target / "test_predictions.npz")
            base.seed_all(seed)
            model, _ = narrow.make_model(bundle, device)
            checkpoint = torch.load(folder / "checkpoint.pt", map_location=device,
                                    weights_only=True)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            acc, ce, logits, _ = base.evaluate(model, "untied", bundle, idx, y)
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
            key = f"{depth}/{seed}"
            baseline = freeze["baseline_tied128_artifact_sha256"][key]
            tied_result = narrow.ORIGINAL / "test_scores" / f"depth{depth}" / f"seed{seed}" / "tied" / "test_result.json"
            assert narrow.sha(tied_result) == baseline["test_result_sha256"]
            tied_acc = read(tied_result)["test_accuracy"]
            rows.append({"depth": depth, "seed": seed,
                         "narrow_width": narrow.WIDTHS[depth],
                         "narrow_parameters": narrow.PARAMETERS[depth],
                         "tied128_parameters": narrow.TIED_PARAMETERS[depth],
                         "narrow_test_accuracy": acc,
                         "tied128_test_accuracy": tied_acc,
                         "narrow_test_result_sha256": narrow.sha(target / "test_result.json"),
                         "baseline_tied128_test_result_sha256": baseline["test_result_sha256"]})
            contrasts.append({"depth": depth, "seed": seed,
                              "tied128_minus_narrow_untied_pp": 100 * (tied_acc - acc)})
            print("NARROW_TEST_REPLAY_PASS", depth, seed, flush=True)
    assert len(rows) == len(contrasts) == 6
    narrow.write_json(narrow.ROOT / "complete_score_audit.json", {
        "status": "COMPLETE_6_CELL_NARROW_CUDA_TEST_REPLAY_PASS",
        "protocol": narrow.PROTOCOL,
        "source_freeze_sha256": freeze_sha,
        "validation_lock_sha256": narrow.sha(lock_path),
        "original_v3_source_freeze_sha256": narrow.ORIGINAL_SOURCE_SHA,
        "width_lock_sha256": narrow.WIDTH_LOCK_SHA,
        "interpretation_limit": "parameter-matched width sensitivity on one graph/mask; width changes initial function and architecture",
        "records": rows, "paired": contrasts})
    print("COMPLETE_6_CELL_NARROW_CUDA_TEST_REPLAY_PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "score", "audit-test"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    if args.stage == "validate":
        validate(device)
    elif args.stage == "score":
        score(device)
    else:
        audit_test(device)


if __name__ == "__main__":
    main()
