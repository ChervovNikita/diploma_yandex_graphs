"""Read-only audit of the post hoc Roman Empire GAT tying diagnostic.

The matching launcher records source and dataset hashes before the first run.
This verifier accepts an ordered prefix during a restart and requires exactly
ten complete rows with --complete. It never writes files or invokes CUDA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch_geometric


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments_iclr" / "gat_failure_pair_results"
CSV = ROOT / "projector_controls.csv"
VARIANTS = ("gnnm", "untied_backbone")
DEPTH = {0: 5, 1: 5, 2: 4, 3: 4, 4: 4}
QUEUE_LOG = REPO / "experiments_iclr" / "logs" / "run_after_sage.log"
QUEUE_PID = REPO / "experiments_iclr" / "logs" / "run_after_sage_queue.pid"
QUEUE_MARKERS = {
    "ENSEMBLE_INFERENCE_PROFILE_COMPLETE",
    "PARAM_MATCHED_ENS_SKIPPED_AFTER_06_UTC",
}
SOURCE_FILES = (
    "experiments_iclr/projector_controls.py",
    "experiments_iclr/run_gat_failure_pair.sh",
    "experiments_iclr/verify_gat_failure_pair.py",
    "experiments_iclr/run_after_sage.sh",
    "ablation/models_ablation.py",
    "models.py",
    "datasets.py",
    "run_common.py",
    "data/roman_empire.npz",
)
FIELDS = (
    "dataset", "model", "variant", "split", "seed", "num_layers",
    "hidden_dim", "lr", "m", "num_steps", "best_step", "num_params",
    "train_seconds", "val_metric", "test_metric", "test_acc", "test_loss",
    "n_test", "mean_member_acc", "at_least_one_member_correct_rate",
    "pair_disagreement", "error_jaccard", "mean_entropy", "nll", "brier",
    "ece10", "checkpoint", "prediction_file",
)
EXPECTED_KEYS = tuple((split, variant) for split in DEPTH for variant in VARIANTS)
GNNM_PARAMS = {4: 3331392, 5: 4121408}
UNTIED_PARAMS = {4: 12811584, 5: 15971648}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def queue_processes() -> list[str]:
    """Find the actual queue process even if its saved PID was superseded."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True,
    )
    matches = []
    pattern = re.compile(r"^(?:/usr/bin/)?bash experiments_iclr/run_after_sage\.sh(?:\s|$)")
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and pattern.match(parts[1]):
            matches.append(parts[0])
    return matches


def queue_terminal_marker() -> str:
    if not QUEUE_PID.is_file() or not QUEUE_LOG.is_file():
        raise RuntimeError("Original queue PID file or log is missing")
    pid_text = QUEUE_PID.read_text().strip()
    if not pid_text.isdecimal() or int(pid_text) <= 0:
        raise RuntimeError("Original queue PID file is invalid")
    if queue_processes():
        raise RuntimeError("Original GPU queue is still running")
    lines = [line.strip() for line in QUEUE_LOG.read_text().splitlines()
             if line.strip()]
    if not lines or lines[-1] not in QUEUE_MARKERS:
        raise RuntimeError("Original GPU queue lacks its final success marker")
    return lines[-1]


def expected_manifest() -> dict:
    marker = queue_terminal_marker()
    return {
        "schema": 1,
        "purpose": "Post hoc Roman Empire GAT whole-residual-stack tying diagnostic",
        "dataset": "roman-empire",
        "model": "GAT",
        "variants": list(VARIANTS),
        "depth_by_official_mask": {str(k): v for k, v in DEPTH.items()},
        "hidden_dim": 512,
        "learning_rate": 3e-5,
        "members": 4,
        "max_steps": 5000,
        "device": "cuda:0",
        "source_sha256": {relative: sha256(REPO / relative)
                          for relative in SOURCE_FILES},
        "queue_log_sha256": sha256(QUEUE_LOG),
        "queue_success_marker": marker,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
    }


def required_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {field} in {row['variant']} mask {row['split']}")
    return value


def artifact_path(row: dict[str, str], field: str, suffix: str) -> Path:
    stem = f"roman-empire_GAT_{row['variant']}_split{row['split']}"
    directory = "checkpoints" if field == "checkpoint" else "predictions"
    expected = ROOT / directory / f"{stem}{suffix}"
    recorded = Path(row[field])
    if recorded.is_absolute() or ".." in recorded.parts:
        raise ValueError(f"Unsafe {field} path in mask {row['split']}")
    path = REPO / recorded
    if path != expected or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Unexpected {field} path in mask {row['split']}: {recorded}")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {field}: {path}")
    return path


def check_checkpoint(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Invalid checkpoint dictionary: {path}")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Non-tensor checkpoint entry {name}: {path}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Nonfinite checkpoint tensor {name}: {path}")


def check_prediction(path: Path, row: dict[str, str]) -> None:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "node_index", "y_true", "ensemble_pred", "member_pred",
            "member_logits", "ensemble_prob", "confidence", "degree",
            "local_homophily",
        }
        if set(data.files) != required:
            raise ValueError(f"Prediction fields differ from protocol: {path}")
        n = int(row["n_test"])
        y = data["y_true"]
        selected = data["ensemble_pred"]
        members = data["member_pred"]
        logits = data["member_logits"]
        if n != 5666 or y.shape != (n,) or selected.shape != (n,):
            raise ValueError(f"Unexpected Roman Empire test shape: {path}")
        if members.shape != (4, n) or logits.ndim != 3 or logits.shape[:2] != (4, n):
            raise ValueError(f"Unexpected member shape: {path}")
        c = logits.shape[2]
        if c < 2 or data["ensemble_prob"].shape != (n, c):
            raise ValueError(f"Unexpected class dimension: {path}")
        for field in ("node_index", "confidence", "degree", "local_homophily"):
            if data[field].shape != (n,):
                raise ValueError(f"Unexpected {field} shape: {path}")
        if not np.isfinite(logits).all() or not np.isfinite(data["ensemble_prob"]).all():
            raise ValueError(f"Nonfinite prediction arrays: {path}")
        if not np.array_equal(members, logits.argmax(axis=-1)):
            raise ValueError(f"Member predictions differ from logits: {path}")
        if not np.array_equal(selected, logits.mean(axis=0).argmax(axis=-1)):
            raise ValueError(f"Pooled predictions differ from logits: {path}")
        accuracy = float(np.mean(selected == y))
        if abs(accuracy - required_float(row, "test_metric")) > 1e-6:
            raise ValueError(f"Recorded test metric differs from saved predictions: {path}")
        if abs(accuracy - required_float(row, "test_acc")) > 1e-6:
            raise ValueError(f"Recorded test accuracy differs from saved predictions: {path}")
        member_accuracy = float(np.mean(members == y[None, :]))
        if abs(member_accuracy - required_float(row, "mean_member_acc")) > 1e-6:
            raise ValueError(f"Recorded member accuracy differs from predictions: {path}")


def check_row(row: dict[str, str]) -> tuple[int, str]:
    if any(value is None for value in row.values()) or None in row:
        raise ValueError("Truncated or extended CSV row")
    if row["dataset"] != "roman-empire" or row["model"] != "GAT":
        raise ValueError("Wrong dataset or backbone in GAT result root")
    split = int(row["split"])
    variant = row["variant"]
    if split not in DEPTH or variant not in VARIANTS:
        raise ValueError(f"Unexpected key: {variant} mask {split}")
    if (int(row["seed"]) != split or int(row["num_layers"]) != DEPTH[split]
            or int(row["hidden_dim"]) != 512
            or not math.isclose(float(row["lr"]), 3e-5, rel_tol=0, abs_tol=1e-12)
            or int(row["m"]) != 4 or int(row["num_steps"]) != 5000):
        raise ValueError(f"Mixed protocol in {variant} mask {split}")
    step = int(row["best_step"])
    if not 1 <= step <= 5000 or (step != 1 and step % 10):
        raise ValueError(f"Invalid selected step in {variant} mask {split}")
    count = int(row["num_params"])
    expected_count = (GNNM_PARAMS if variant == "gnnm" else UNTIED_PARAMS)[DEPTH[split]]
    if count != expected_count:
        raise ValueError(f"Unexpected parameter count in {variant} mask {split}")
    if required_float(row, "train_seconds") <= 0:
        raise ValueError(f"Invalid training duration in {variant} mask {split}")
    for field in ("val_metric", "test_metric", "test_acc", "mean_member_acc",
                  "at_least_one_member_correct_rate", "pair_disagreement",
                  "error_jaccard", "ece10"):
        value = required_float(row, field)
        if not 0 <= value <= 1:
            raise ValueError(f"Out-of-range {field} in {variant} mask {split}")
    for field in ("test_loss", "mean_entropy", "nll", "brier"):
        if required_float(row, field) < 0:
            raise ValueError(f"Negative {field} in {variant} mask {split}")
    checkpoint = artifact_path(row, "checkpoint", ".pt")
    prediction = artifact_path(row, "prediction_file", ".npz")
    check_checkpoint(checkpoint)
    check_prediction(prediction, row)
    return split, variant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--require-splits", nargs="*", type=int, default=[])
    args = parser.parse_args()
    if any(split not in DEPTH for split in args.require_splits):
        parser.error("Required splits must be among official masks 0–4")
    manifest_path = ROOT / "protocol.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen protocol is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest != expected_manifest():
        raise RuntimeError("Frozen source, data, queue, environment, or protocol changed")
    if CSV.exists():
        with CSV.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError("Unexpected results CSV header")
            rows = list(reader)
    else:
        rows = []
    keys = [check_row(row) for row in rows]
    if keys != list(EXPECTED_KEYS[:len(keys)]):
        raise ValueError(f"Results are not an ordered unique prefix: {keys}")
    required = ({(split, variant) for split in args.require_splits
                 for variant in VARIANTS} |
                (set(EXPECTED_KEYS) if args.complete else set()))
    if not required.issubset(keys):
        raise ValueError(f"Missing required rows: {sorted(required - set(keys))}")
    if args.complete and len(keys) != len(EXPECTED_KEYS):
        raise ValueError(f"Expected ten rows, found {len(keys)}")
    print(json.dumps({"verified_rows": len(keys), "complete": bool(args.complete),
                      "keys": keys}, sort_keys=True))


if __name__ == "__main__":
    main()
