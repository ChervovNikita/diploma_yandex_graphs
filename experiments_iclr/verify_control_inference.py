"""Recompute selected Roman Empire SAGE test logits from saved checkpoints.

This read-only audit complements verify_control_artifacts.py, which checks
arithmetic within saved prediction files. Here each model is reconstructed,
its selected state dict is loaded strictly, and the pinned official graph is
passed through the model in evaluation mode on CPU. No test label is used in
model construction or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from projector_controls import make_model, model_output
from datasets import load_dataset
from verify_control_artifacts import (
    EXPECTED_VARIANTS,
    digest_file,
    digest_row,
    safe_file,
)


def selected_paths(root: Path, row: dict[str, str], audit: dict | None) -> tuple[Path, Path]:
    variant = row["variant"]
    split = int(row["split"])
    stem = f"roman-empire_SAGE_{variant}_split{split}"
    adopted_fresh = (
        variant == "gnnm" and split == 0
        and audit is not None and audit.get("adopted") == "fresh"
    )
    artifact_root = (
        "experiments_iclr/gnnm_split0_verification"
        if adopted_fresh else "experiments_iclr/results"
    )
    checkpoint = safe_file(root, row["checkpoint"],
                           artifact_root + "/checkpoints", ".pt")
    prediction = safe_file(root, row["prediction_file"],
                           artifact_root + "/predictions", ".npz")
    if checkpoint.stem != stem or prediction.stem != stem:
        raise AssertionError(f"Artifact identity mismatch: {stem}")
    if variant == "gnnm" and split == 0 and audit is not None:
        label = "fresh" if adopted_fresh else "pilot"
        if audit.get("adopted") not in {"fresh", "pilot"}:
            raise AssertionError("Unknown split-0 adoption decision")
        if digest_row(row) != audit[f"{label}_row_sha256"]:
            raise AssertionError("Selected split-0 CSV row differs from adoption audit")
        if digest_file(checkpoint) != audit[f"{label}_checkpoint_sha256"]:
            raise AssertionError("Selected split-0 checkpoint differs from adoption audit")
        if digest_file(prediction) != audit[f"{label}_prediction_sha256"]:
            raise AssertionError("Selected split-0 prediction differs from adoption audit")
    return checkpoint, prediction


def validate_protocol(rows: list[dict[str, str]], require_complete: bool) -> None:
    allowed = {
        ("roman-empire", "SAGE", variant, split)
        for variant in EXPECTED_VARIANTS for split in range(5)
    }
    keys = [
        (row["dataset"], row["model"], row["variant"], int(row["split"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate selected control row")
    if not set(keys).issubset(allowed):
        raise AssertionError("Unexpected dataset, model, variant, or split")
    if require_complete and set(keys) != allowed:
        raise AssertionError(f"Incomplete matrix: {sorted(allowed - set(keys))}")
    for row in rows:
        variant = row["variant"]
        split = int(row["split"])
        m = 1 if variant in {"base", "gnnm_m1"} else 4
        if (int(row["seed"]) != split
                or int(row["num_layers"]) != 5
                or int(row["hidden_dim"]) != 512
                or not math.isclose(float(row["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12)
                or int(row["m"]) != m
                or int(row["num_steps"]) != 5000
                or not 1 <= int(row["best_step"]) <= 5000):
            raise AssertionError(f"Mixed protocol: {variant} split {split}")


def compare_logits(stem: str, actual: np.ndarray, saved: np.ndarray,
                   atol: float, rtol: float) -> tuple[float, float]:
    if actual.shape != saved.shape:
        raise AssertionError(f"{stem}: fresh/saved logit dimensions differ")
    if not np.isfinite(actual).all() or not np.isfinite(saved).all():
        raise AssertionError(f"{stem}: nonfinite logits")
    difference = np.abs(actual.astype(np.float64) - saved.astype(np.float64))
    allowed = atol + rtol * np.abs(saved.astype(np.float64))
    bad = difference > allowed
    max_abs = float(difference.max(initial=0.0))
    max_ratio = float((difference / allowed).max(initial=0.0))
    if bad.any():
        coordinate = tuple(int(i) for i in np.unravel_index(
            int(np.argmax(difference / allowed)), difference.shape))
        raise AssertionError(
            f"{stem}: {int(bad.sum())} logits exceed atol={atol:g}, "
            f"rtol={rtol:g}; worst {coordinate}: fresh={actual[coordinate]:.9g}, "
            f"saved={saved[coordinate]:.9g}, abs_diff={difference[coordinate]:.3g}, "
            f"allowed={allowed[coordinate]:.3g}"
        )
    return max_abs, max_ratio


def audit_one(root: Path, row: dict[str, str], audit: dict | None,
              graph, test_masks: torch.Tensor, atol: float, rtol: float) -> dict:
    variant = row["variant"]
    split = int(row["split"])
    stem = f"roman-empire_SAGE_{variant}_split{split}"
    checkpoint, prediction = selected_paths(root, row, audit)
    official_nodes = test_masks[:, split].nonzero(as_tuple=False).flatten().numpy()
    official_labels = graph.y[official_nodes].numpy()
    with np.load(prediction, allow_pickle=False) as archive:
        needed = {"node_index", "y_true", "member_logits", "member_pred"}
        if not needed.issubset(archive.files):
            raise AssertionError(f"{stem}: missing saved prediction arrays")
        nodes = np.asarray(archive["node_index"])
        labels = np.asarray(archive["y_true"])
        saved = np.asarray(archive["member_logits"])
        saved_preds = np.asarray(archive["member_pred"])
    m = int(row["m"])
    expected_shape = (m, official_nodes.size, 18)
    if saved.shape != expected_shape or saved_preds.shape != expected_shape[:2]:
        raise AssertionError(f"{stem}: saved member dimensions differ from protocol")
    if not np.array_equal(nodes, official_nodes):
        raise AssertionError(f"{stem}: saved nodes differ from official test mask")
    if not np.array_equal(labels, official_labels):
        raise AssertionError(f"{stem}: saved labels differ from official graph labels")
    if int(row["n_test"]) != official_nodes.size:
        raise AssertionError(f"{stem}: CSV test count differs from official mask")
    if not np.array_equal(saved.argmax(axis=-1), saved_preds):
        raise AssertionError(f"{stem}: saved predictions differ from saved logits")

    args = SimpleNamespace(model="SAGE", num_layers=5, hidden_dim=512, m=m)
    torch.manual_seed(0)
    model = make_model(args, variant, graph.x.shape[1], 18, torch.device("cpu"))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params != int(row["num_params"]):
        raise AssertionError(f"{stem}: model parameter count differs from CSV")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        actual = np.stack([
            model_output(model, variant, graph, member)[official_nodes]
            .detach().cpu().numpy()
            for member in range(m)
        ], axis=0)
    seconds = time.monotonic() - started
    max_abs, max_ratio = compare_logits(stem, actual, saved, atol, rtol)
    fresh_preds = actual.argmax(axis=-1)
    changed = fresh_preds != saved_preds
    # CPU and CUDA can reverse an almost exact tie even when every raw logit
    # agrees within the declared numerical tolerance. Report such cases,
    # and fail if a reversed decision exceeds the two-logit error allowance.
    max_saved_flip_margin = 0.0
    if changed.any():
        member, node = np.nonzero(changed)
        saved_top = saved_preds[changed]
        fresh_top = fresh_preds[changed]
        saved_margin = (saved[member, node, saved_top]
                        - saved[member, node, fresh_top])
        pair_allowance = (2 * atol + rtol
                          * (np.abs(saved[member, node, saved_top])
                             + np.abs(saved[member, node, fresh_top])))
        max_saved_flip_margin = float(saved_margin.max())
        if np.any(saved_margin > pair_allowance):
            raise AssertionError(f"{stem}: member decision change exceeds numerical tolerance")
    return {
        "variant": variant,
        "split": split,
        "members": m,
        "test_nodes": int(official_nodes.size),
        "max_logit_abs_error": max_abs,
        "max_tolerance_fraction": max_ratio,
        "decision_changes_within_tolerance": int(changed.sum()),
        "max_saved_flip_margin": max_saved_flip_margin,
        "inference_seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--variant", action="append", choices=sorted(EXPECTED_VARIANTS))
    parser.add_argument("--split", action="append", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.threads < 1 or args.atol <= 0 or args.rtol < 0:
        parser.error("threads and atol must be positive; rtol must be nonnegative")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("max-rows must be positive")
    if args.split is not None and any(s < 0 or s > 4 for s in args.split):
        parser.error("splits must be in 0..4")

    root = args.repo.resolve()
    matrix = root / "experiments_iclr/results/projector_controls.csv"
    with matrix.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate_protocol(rows, args.require_complete)
    matrix_rows_at_start = len(rows)
    audit_path = root / "experiments_iclr/results/gnnm_split0_verification_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    if args.require_complete and audit is None:
        raise AssertionError("The split-0 adoption audit is missing")
    if args.variant:
        rows = [row for row in rows if row["variant"] in args.variant]
    if args.split:
        rows = [row for row in rows if int(row["split"]) in args.split]
    if args.max_rows:
        rows = rows[:args.max_rows]
    if not rows:
        raise AssertionError("No selected rows match the requested filter")

    source = root / "data/roman_empire.npz"
    manifest = json.loads((root / "experiments_iclr/data_manifest.json").read_text())
    expected_digest = manifest["files"]["roman_empire.npz"]["sha256"]
    actual_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise AssertionError("Official Roman Empire source differs from pinned SHA256")
    torch.set_num_threads(args.threads)
    graph, _, _, test_masks, classes, targets, binary = load_dataset(
        "roman-empire", add_self_loops=True, device=torch.device("cpu"),
        data_dir=str(root / "data")
    )
    if classes != 18 or targets != 18 or binary or test_masks.shape[1] < 5:
        raise AssertionError("Unexpected official Roman Empire graph or masks")
    started = time.monotonic()
    outputs = []
    for row in rows:
        result = audit_one(root, row, audit, graph, test_masks, args.atol, args.rtol)
        outputs.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({
        "verified_rows": len(outputs),
        "matrix_rows_at_start": matrix_rows_at_start,
        "decision_changes_within_tolerance": sum(
            out["decision_changes_within_tolerance"] for out in outputs),
        "max_logit_abs_error": max(out["max_logit_abs_error"] for out in outputs),
        "max_tolerance_fraction": max(out["max_tolerance_fraction"] for out in outputs),
        "wall_seconds": time.monotonic() - started,
        "split0_selection": None if audit is None else audit.get("adopted"),
        "pinned_graph_sha256": actual_digest,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
