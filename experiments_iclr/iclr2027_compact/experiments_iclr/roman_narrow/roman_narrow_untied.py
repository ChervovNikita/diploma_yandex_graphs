"""Prospectively parameter-matched Roman UNTIED width sensitivity control.

The six widths/cells are fixed by a parameter-only lock. This control changes
width and therefore neither matches TIED128's initial function nor isolates a
single architecture cause. The frozen Roman24 data, model class, loss,
optimizer, schedule, and checkpoint rule are reused without tuning.
"""
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

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "roman_mechanism_v3_prepared"
ROOT = HERE / "roman_narrow_untied_prepared"
sys.path.insert(0, str(ORIGINAL))
import mechanism as mech  # noqa: E402
import roman_mechanism as original  # noqa: E402
import roman_multimask as roman  # noqa: E402
import tuning as base  # noqa: E402

PROTOCOL = "roman_mask0_parameter_matched_narrow_untied_v1"
DEPTHS = (2, 5)
SEEDS = (0, 1, 2)
EPOCHS = 1000
WIDTHS = {2: 68, 5: 65}
PARAMETERS = {2: 211600, 5: 451924}
TIED_PARAMETERS = {2: 208960, 5: 456640}
WIDTH_LOCK_SHA = "49df5c4706694914250f08513ffbb30bee5a78d6ae1368a2f083383d461ad9e9"
ORIGINAL_SOURCE_SHA = "489451d58772c073e7be04bd1d1f7859644f56a6626d9b4a14cd6f752acc0088"
TRACE_FIELDS = original.TRACE_FIELDS


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def run_dir(depth: int, seed: int) -> Path:
    assert depth in DEPTHS and seed in SEEDS
    return ROOT / "results/roman" / f"depth{depth}" / f"seed{seed}" / "untied"


def configure(depth: int) -> None:
    base.configure(depth)
    roman.WIDTH = WIDTHS[depth]
    assert roman.MASK == 0 and roman.DEPTH == depth
    assert roman.WIDTH == WIDTHS[depth]


def make_model(bundle, device):
    model, canonical = original.make_arm("untied", bundle, device)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[roman.DEPTH]
    return model, canonical


def baseline_hashes() -> dict:
    record = {}
    for depth in DEPTHS:
        for seed in SEEDS:
            folder = original.run_dir(depth, seed, "tied")
            score = ORIGINAL / "test_scores" / f"depth{depth}" / f"seed{seed}" / "tied" / "test_result.json"
            row = json.loads((folder / "result.json").read_text())
            assert row["parameter_count"] == TIED_PARAMETERS[depth]
            record[f"{depth}/{seed}"] = {
                "result_sha256": sha(folder / "result.json"),
                "checkpoint_sha256": sha(folder / "checkpoint.pt"),
                "test_result_sha256": sha(score)}
    return record


def freeze() -> None:
    assert not ROOT.exists()
    assert original.check_freeze() == ORIGINAL_SOURCE_SHA
    assert sha(HERE / "ROMAN_NARROW_UNTIED_WIDTH_LOCK.json") == WIDTH_LOCK_SHA
    width_lock = json.loads((HERE / "ROMAN_NARROW_UNTIED_WIDTH_LOCK.json").read_text())
    assert width_lock["status"] == "PARAMETER_COUNT_ONLY_WIDTH_LOCK_BEFORE_NARROW_TRAINING"
    assert width_lock["width_lock_source_sha256"] == sha(
        HERE / "roman_narrow_untied_width_lock.py")
    assert [(r["depth"], r["selected_untied_width"],
             r["selected_untied_parameter_count"], r["tied_parameter_count"])
            for r in width_lock["records"]] == [
                (depth, WIDTHS[depth], PARAMETERS[depth], TIED_PARAMETERS[depth])
                for depth in DEPTHS]
    for item in width_lock["records"]:
        target = item["tied_parameter_count"]
        best = min(item["candidates"],
                   key=lambda row: (abs(row["untied_parameter_count"] - target),
                                    row["width"]))
        assert best["width"] == item["selected_untied_width"]
        assert best["untied_parameter_count"] == item["selected_untied_parameter_count"]
    baseline = baseline_hashes()
    assert len(baseline) == 6
    ROOT.mkdir(parents=True)
    write_json(ROOT / "SOURCE_FREEZE.json", {
        "status": "PROSPECTIVE_6_CELL_NARROW_UNTIED_SOURCE_FREEZE",
        "protocol": PROTOCOL, "depths": list(DEPTHS), "seeds": list(SEEDS),
        "widths": WIDTHS, "parameter_counts": PARAMETERS,
        "tied128_parameter_counts": TIED_PARAMETERS,
        "epochs": EPOCHS, "official_mask": 0,
        "width_selection": "parameter count only, integer 32..128, smaller width on ties",
        "width_lock_sha256": WIDTH_LOCK_SHA,
        "original_v3_source_freeze_sha256": ORIGINAL_SOURCE_SHA,
        "original_validation_lock_sha256": sha(ORIGINAL / "validation_lock.json"),
        "original_test_audit_sha256": sha(ORIGINAL / "complete_score_audit.json"),
        "baseline_tied128_artifact_sha256": baseline,
        "source_sha256": {"roman_narrow_untied.py": sha(Path(__file__)),
                          "verify_roman_narrow_untied.py": sha(
                              HERE / "verify_roman_narrow_untied.py"),
                          "ROMAN_NARROW_UNTIED_PROTOCOL.md": sha(
                              HERE / "ROMAN_NARROW_UNTIED_PROTOCOL.md")},
        "runtime": {"python": ".".join(map(str, sys.version_info[:3])),
                    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    "torch_geometric": torch_geometric.__version__,
                    "numpy": np.__version__},
        "test_scores_used_for_width_selection": False,
        "initial_function_matched_to_tied128": False,
        "pure_architecture_causal_test": False,
        "test_rule": "all six selected validation checkpoints independently locked before test scoring"})
    print("NARROW_SOURCE_FROZEN", sha(ROOT / "SOURCE_FREEZE.json"), flush=True)


def check_freeze() -> str:
    assert original.check_freeze() == ORIGINAL_SOURCE_SHA
    assert sha(HERE / "ROMAN_NARROW_UNTIED_WIDTH_LOCK.json") == WIDTH_LOCK_SHA
    path = ROOT / "SOURCE_FREEZE.json"
    value = json.loads(path.read_text())
    assert value["status"] == "PROSPECTIVE_6_CELL_NARROW_UNTIED_SOURCE_FREEZE"
    assert value["protocol"] == PROTOCOL
    assert value["depths"] == list(DEPTHS) and value["seeds"] == list(SEEDS)
    assert value["widths"] == {str(k): v for k, v in WIDTHS.items()}
    assert value["parameter_counts"] == {str(k): v for k, v in PARAMETERS.items()}
    assert value["tied128_parameter_counts"] == {str(k): v for k, v in TIED_PARAMETERS.items()}
    assert value["width_lock_sha256"] == WIDTH_LOCK_SHA
    assert value["original_v3_source_freeze_sha256"] == ORIGINAL_SOURCE_SHA
    assert value["original_validation_lock_sha256"] == sha(ORIGINAL / "validation_lock.json")
    assert value["original_test_audit_sha256"] == sha(ORIGINAL / "complete_score_audit.json")
    assert value["baseline_tied128_artifact_sha256"] == baseline_hashes()
    for name, digest in value["source_sha256"].items():
        assert sha(HERE / name) == digest
    assert value["runtime"] == {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "torch_geometric": torch_geometric.__version__, "numpy": np.__version__}
    return sha(path)


def preflight(device: torch.device) -> None:
    freeze_sha = check_freeze()
    assert not (ROOT / "preflight.json").exists()
    records = []
    descriptors = {}
    for depth in DEPTHS:
        configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=False)
        descriptors[str(depth)] = descriptor
        assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
        for seed in SEEDS:
            outputs = []
            for _ in range(2):
                base.seed_all(seed)
                model, canonical = make_model(bundle, device)
                initial = base.initial_audit(model, "untied", bundle, seed,
                                             device, canonical)
                model.eval()
                with torch.no_grad():
                    logits = torch.stack(base.member_logits(model, "untied", bundle))
                assert bool(torch.isfinite(logits).all())
                outputs.append((initial, logits))
            left, right = outputs
            for field in ("canonical_projector_state_sha256", "python_rng_sha256",
                          "numpy_rng_sha256", "cpu_rng_sha256", "cuda_rng_sha256",
                          "parameter_count"):
                assert left[0][field] == right[0][field]
            drift = float((left[1] - right[1]).abs().max().item())
            assert drift <= base.INITIAL_LOGIT_TOL
            assert torch.equal(left[1].argmax(-1), right[1].argmax(-1))
            records.append({"depth": depth, "seed": seed,
                            "width": WIDTHS[depth],
                            "parameter_count": left[0]["parameter_count"],
                            "repeat_initial_max_abs_logit_difference": drift,
                            "canonical_initial_sha256": canonical})
            print("NARROW_PREFLIGHT_PASS", depth, seed, flush=True)
    assert len(records) == 6
    write_json(ROOT / "preflight.json", {
        "status": "ALL_6_NARROW_CUDA_PREFLIGHT_PASS",
        "source_freeze_sha256": freeze_sha,
        "records": records, "data_descriptors": descriptors})


def train_one(depth: int, seed: int, bundle, device,
              source_sha: str) -> None:
    final = run_dir(depth, seed)
    if final.exists():
        old = json.loads((final / "result.json").read_text())
        assert old["source_freeze_sha256"] == source_sha
        for name, digest in old["artifact_sha256"].items():
            assert sha(final / name) == digest
        print("SKIP_COMPLETE", depth, seed, flush=True)
        return
    work = final.with_name("untied.inprogress")
    assert not work.exists()
    work.mkdir(parents=True)
    base.seed_all(seed)
    model, canonical = make_model(bundle, device)
    initial = base.initial_audit(model, "untied", bundle, seed,
                                 device, canonical)
    model.eval()
    with torch.no_grad():
        initial_logits = torch.stack(base.member_logits(model, "untied", bundle)).cpu().numpy()
    assert initial_logits.shape == (4, 22662, bundle.classes)
    np.save(work / "initial_logits.npy", initial_logits)
    initial["saved_initial_logits_sha256"] = sha(work / "initial_logits.npy")
    optimizer = original.optimizer_for(model)
    best_acc, best_ce, best_epoch = -1.0, math.inf, 0
    trace = []
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        mech.one_update(model, "untied", bundle, optimizer)
        acc, ce, _, _ = base.evaluate(model, "untied", bundle,
                                      bundle.valid_idx, bundle.valid_y)
        assert math.isfinite(acc) and math.isfinite(ce)
        improved = acc > best_acc or (acc == best_acc and ce < best_ce)
        if improved:
            best_acc, best_ce, best_epoch = acc, ce, epoch
            torch.save({"state_dict": model.state_dict(), "depth": depth,
                        "seed": seed, "arm": "untied", "width": WIDTHS[depth],
                        "epoch": epoch}, work / "checkpoint.pt")
        trace.append({"epoch": epoch, "valid_accuracy": acc,
                      "valid_ce": ce, "selected_now": int(improved),
                      "graph_weights_equal": 0,
                      **{name: "" for name in TRACE_FIELDS[5:]}})
        if epoch == 1 or epoch % 100 == 0:
            print("NARROW_TRAIN_PROGRESS", depth, seed, epoch, flush=True)
    with (work / "validation_trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(trace)
    checkpoint = torch.load(work / "checkpoint.pt", map_location=device,
                            weights_only=True)
    assert (checkpoint["depth"], checkpoint["seed"], checkpoint["arm"],
            checkpoint["width"], checkpoint["epoch"]) == (
                depth, seed, "untied", WIDTHS[depth], best_epoch)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    acc, ce, logits, _ = base.evaluate(model, "untied", bundle,
                                       bundle.valid_idx, bundle.valid_y)
    assert abs(acc - best_acc) < 1e-7 and abs(ce - best_ce) < 1e-6
    np.savez_compressed(work / "selected_validation.npz",
                        member_logits=logits.detach().cpu().numpy(),
                        node_index=bundle.valid_idx.cpu().numpy(),
                        y_true=bundle.valid_y.cpu().numpy())
    artifacts = ("checkpoint.pt", "validation_trace.csv",
                 "initial_logits.npy", "selected_validation.npz")
    write_json(work / "result.json", {
        "protocol": PROTOCOL, "source_freeze_sha256": source_sha,
        "dataset": "roman", "official_mask": 0,
        "depth": depth, "seed": seed, "arm": "untied",
        "width": WIDTHS[depth], "parameter_count": initial["parameter_count"],
        "epochs_completed": EPOCHS, "selected_epoch": best_epoch,
        "valid_accuracy": acc, "valid_ce": ce,
        "initialization": initial,
        "train_seconds": time.monotonic() - started,
        "artifact_sha256": {name: sha(work / name) for name in artifacts}})
    work.rename(final)
    print("NARROW_VALIDATION_COMPLETE", depth, seed, flush=True)


def train(device: torch.device) -> None:
    source_sha = check_freeze()
    gate = json.loads((ROOT / "preflight.json").read_text())
    assert gate["status"] == "ALL_6_NARROW_CUDA_PREFLIGHT_PASS"
    assert gate["source_freeze_sha256"] == source_sha and len(gate["records"]) == 6
    for depth in DEPTHS:
        configure(depth)
        bundle, descriptor = base.load_graph("roman", device, include_test=False)
        assert descriptor == gate["data_descriptors"][str(depth)]
        assert not hasattr(bundle, "test_idx_cpu") and not hasattr(bundle, "test_y_cpu")
        for seed in SEEDS:
            train_one(depth, seed, bundle, device, source_sha)
    print("ALL_6_NARROW_VALIDATION_RUNS_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "preflight", "train"), required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "freeze":
        freeze()
        return
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.cuda.is_available()
    preflight(device) if args.stage == "preflight" else train(device)


if __name__ == "__main__":
    main()
