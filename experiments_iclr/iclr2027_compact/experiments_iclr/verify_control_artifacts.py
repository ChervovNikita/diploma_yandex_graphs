"""Verify selected Roman Empire SAGE control artifacts without rerunning training.

This reads the selected CSV rows and prediction archives after the queued runs
finish. It checks their identities, shapes, and numerical consistency. It does
not use test results to choose or replace a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


EXPECTED_VARIANTS = {
    "gnnm", "independent_projectors", "heads_only", "input_only",
    "output_only", "base", "gnnm_m1", "untied_backbone",
    "freeze_output_factors", "ens_pooled",
}


def close(name: str, actual: float, recorded: str, atol: float = 1e-5) -> None:
    expected = float(recorded)
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise AssertionError(f"{name}: nonfinite value")
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=atol):
        raise AssertionError(f"{name}: recomputed {actual}, CSV {expected}")


def softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(centered)
    return exp / exp.sum(axis=-1, keepdims=True)


def logsumexp(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=-1)
    return maximum + np.log(np.exp(logits - maximum[..., None]).sum(axis=-1))


def safe_file(root: Path, relative: str, expected_dir: str, suffix: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != suffix:
        raise AssertionError(f"Unsafe artifact path: {relative}")
    full = (root / path).resolve()
    allowed = (root / expected_dir).resolve()
    if not full.is_relative_to(allowed) or not full.is_file():
        raise AssertionError(f"Missing or external artifact: {relative}")
    return full


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_row(row: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--check-checkpoints", action="store_true",
                        help="Load every selected state dict and check finite tensors")
    args = parser.parse_args()
    root = args.repo.resolve()
    matrix = root / "experiments_iclr/results/projector_controls.csv"
    audit_path = root / "experiments_iclr/results/gnnm_split0_verification_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    if args.require_complete and audit is None:
        raise AssertionError("The split-0 checkpoint adoption audit is missing")
    with matrix.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    keys = [(row["dataset"], row["model"], row["variant"], int(row["split"]))
            for row in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate selected control row")
    allowed = {("roman-empire", "SAGE", variant, split)
               for variant in EXPECTED_VARIANTS for split in range(5)}
    if not set(keys).issubset(allowed):
        raise AssertionError("Unexpected dataset, backbone, variant, or mask")
    if args.require_complete and set(keys) != allowed:
        missing = sorted(allowed - set(keys))
        raise AssertionError(f"Incomplete 50-row matrix; missing {missing}")

    masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    max_error = 0.0
    for row in rows:
        variant = row["variant"]
        split = int(row["split"])
        stem = f"roman-empire_SAGE_{variant}_split{split}"
        expected_m = 1 if variant in {"base", "gnnm_m1"} else 4
        if (int(row["seed"]) != split
                or int(row["num_layers"]) != 5
                or int(row["hidden_dim"]) != 512
                or not math.isclose(float(row["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12)
                or int(row["m"]) != expected_m
                or int(row["num_steps"]) != 5000
                or not 1 <= int(row["best_step"]) <= 5000
                or not math.isfinite(float(row["val_metric"]))
                or not math.isfinite(float(row["test_metric"]))):
            raise AssertionError(f"Selected row violates the fixed protocol: {stem}")
        adopted_fresh = variant == "gnnm" and split == 0 and bool(audit) and audit.get("adopted") == "fresh"
        if adopted_fresh and digest_row(row) != audit["fresh_row_sha256"]:
            raise AssertionError("Selected split-0 row differs from audited fresh row")
        if variant == "gnnm" and split == 0 and audit and audit.get("adopted") == "pilot":
            if digest_row(row) != audit["pilot_row_sha256"]:
                raise AssertionError("Selected split-0 row differs from audited pilot row")
        artifact_root = ("experiments_iclr/gnnm_split0_verification"
                         if adopted_fresh else "experiments_iclr/results")
        checkpoint = safe_file(root, row["checkpoint"],
                               artifact_root + "/checkpoints", ".pt")
        prediction = safe_file(root, row["prediction_file"],
                               artifact_root + "/predictions", ".npz")
        if checkpoint.stem != stem or prediction.stem != stem:
            raise AssertionError(f"Artifact identity mismatch for {stem}")
        if variant == "gnnm" and split == 0 and audit:
            label = "fresh" if adopted_fresh else "pilot"
            if (digest_file(checkpoint) != audit[f"{label}_checkpoint_sha256"]
                    or digest_file(prediction) != audit[f"{label}_prediction_sha256"]):
                raise AssertionError("Selected split-0 artifacts differ from audited hashes")
        if args.check_checkpoints:
            import torch

            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not state:
                raise AssertionError(f"Empty or invalid selected checkpoint: {stem}")
            for name, tensor in state.items():
                if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                    raise AssertionError(f"Invalid checkpoint tensor {name}: {stem}")
            del state
        with np.load(prediction, allow_pickle=False) as z:
            required = {"node_index", "y_true", "ensemble_pred", "member_pred",
                        "member_logits", "ensemble_prob", "confidence"}
            if not required.issubset(z.files):
                raise AssertionError(f"Missing arrays in {stem}: {required-set(z.files)}")
            nodes = np.asarray(z["node_index"])
            y = np.asarray(z["y_true"])
            pooled_saved = np.asarray(z["ensemble_pred"])
            members_saved = np.asarray(z["member_pred"])
            logits = np.asarray(z["member_logits"], dtype=np.float64)
            probs_saved = np.asarray(z["ensemble_prob"], dtype=np.float64)
            confidence_saved = np.asarray(z["confidence"], dtype=np.float64)

        m, n, c = logits.shape
        if m != expected_m or n != int(row["n_test"]) or c != 18:
            raise AssertionError(f"Logit dimensions do not match protocol: {stem}")
        if (nodes.shape != (n,) or y.shape != (n,)
                or pooled_saved.shape != (n,) or members_saved.shape != (m, n)
                or probs_saved.shape != (n, c)
                or confidence_saved.shape != (n,)):
            raise AssertionError(f"Saved array dimensions disagree: {stem}")
        if not np.issubdtype(nodes.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
            raise AssertionError(f"Node indices or labels are not integer: {stem}")
        if not np.array_equal(nodes, np.unique(nodes)) or np.any(nodes < 0):
            raise AssertionError(f"Node indices repeat or are negative: {stem}")
        if np.any(y < 0) or np.any(y >= c) or not np.isfinite(logits).all():
            raise AssertionError(f"Invalid labels or logits: {stem}")
        if split in masks:
            old_nodes, old_y = masks[split]
            if not np.array_equal(nodes, old_nodes) or not np.array_equal(y, old_y):
                raise AssertionError(f"Different held-out nodes or labels on split {split}")
        else:
            masks[split] = (nodes, y)

        member_pred = logits.argmax(axis=-1)
        pool_logits = logits.mean(axis=0)
        pool_pred = pool_logits.argmax(axis=-1)
        probs = softmax(pool_logits)
        if not np.array_equal(member_pred, members_saved):
            raise AssertionError(f"Member predictions do not match logits: {stem}")
        if not np.array_equal(pool_pred, pooled_saved):
            raise AssertionError(f"Pooled predictions do not match logits: {stem}")
        err = float(np.max(np.abs(probs - probs_saved)))
        max_error = max(max_error, err)
        if err > 2e-6 or np.max(np.abs(probs.max(axis=-1) - confidence_saved)) > 2e-6:
            raise AssertionError(f"Saved probabilities/confidence do not match logits: {stem}")

        correct = member_pred == y[None, :]
        pool_correct = pool_pred == y
        if np.any(correct.all(axis=0) & ~pool_correct):
            raise AssertionError(f"Pool harms unanimous correct members: {stem}")
        pair_disagreement = (np.mean([(member_pred[i] != member_pred[j]).mean()
                                      for i, j in combinations(range(m), 2)])
                             if m > 1 else 0.0)
        close(f"{stem} test_acc", float(pool_correct.mean()), row["test_acc"])
        close(f"{stem} test_metric", float(pool_correct.mean()), row["test_metric"])
        close(f"{stem} mean_member_acc", float(correct.mean()), row["mean_member_acc"])
        close(f"{stem} pair_disagreement", float(pair_disagreement),
              row["pair_disagreement"])
        close(f"{stem} at_least_one_member_correct_rate",
              float(correct.any(axis=0).mean()),
              row["at_least_one_member_correct_rate"])
        ce = logsumexp(pool_logits) - pool_logits[np.arange(n), y]
        close(f"{stem} test_loss", float(ce.mean()), row["test_loss"], atol=1e-4)

    print(f"Verified {len(rows)} selected rows, {len(masks)} official masks; "
          f"max probability error {max_error:.3g}")


if __name__ == "__main__":
    main()
