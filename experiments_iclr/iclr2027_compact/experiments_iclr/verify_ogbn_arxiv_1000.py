"""Fail-closed complete gate for the fixed 1,000-epoch ogbn-arxiv repeat.

This reuses the pinned, independently tested 100/300 artifact verifier without
changing it. No score is printed until all nine canonical result directories
and the complete summary have passed. ``--preflight`` checks only frozen
sources and the already audited 300-epoch provenance, with no result access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

import verify_ogbn_arxiv_results as base


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiments_iclr/ogbn_arxiv_1000_results"
LOCK_PATH = REPO / "experiments_iclr/ogbn_arxiv_1000_source_lock.json"
DATA_ROOT = REPO / "experiments_iclr/data"
EXPECTED_300_CONFIG_SHA256 = "ee26cd93647d77c964a554960ab4893fdc95e5bd7d73fd9479e0a8c32a1bde97"
EXPECTED_DATA_SHA256 = "6ac7f655985f90184d86b27a934b8a4d5e9a0f50e7a846e188ac3a3cf2033c8b"
PINNED_SOURCE = {
    "experiments_iclr/ogbn_arxiv_pilot.py": "7abf975fcf900f52a01a51652605b2d2519aaee02918948732ee5401952dd89b",
    "models.py": "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303",
    "experiments_iclr/verify_ogbn_arxiv_results.py": "3e9b2157db7cc2ee5f16d9259fdca4a4d35daf94cc8de0bd4454092fee384b6d",
}
CONFIG = {**base.EXPECTED_CONFIG_300, "max_epochs": 1000, "min_epochs": 1000}
SUMMARY_COLUMNS = (
    "seed", "variant", "selected_epoch", "valid_accuracy", "valid_ce",
    "test_accuracy", "test_ce", "parameter_count", "trainable_parameter_count",
    "epochs_run", "train_step_seconds_total", "validation_seconds_total",
    "total_seconds", "mean_train_step_ms", "median_train_step_ms",
    "peak_allocated_mib", "checkpoint_sha256",
)
PAIR_NAMES = tuple((seed, variant) for seed in base.SEEDS for variant in base.VARIANTS)


def fail(message: str) -> None:
    raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        fail(f"Missing, empty, or linked file: {path}")


def check_source_lock() -> dict[str, str]:
    regular_file(LOCK_PATH)
    lock = json.loads(LOCK_PATH.read_text())
    if not isinstance(lock, dict) or set(lock) != {"schema", "files"} or lock["schema"] != 1:
        fail("Invalid 1,000-epoch source lock schema")
    files = lock["files"]
    expected_names = set(PINNED_SOURCE) | {
        "experiments_iclr/ogbn_arxiv_300_results/run_config.json",
        "experiments_iclr/ogbn_arxiv_300_results/dataset_manifest.json",
        "experiments_iclr/ogbn_arxiv_1000_protocol.md",
        "experiments_iclr/verify_ogbn_arxiv_1000.py",
        "experiments_iclr/run_ogbn_arxiv_1000_once.sh",
        "experiments_iclr/test_verify_ogbn_arxiv_1000.py",
    }
    if not isinstance(files, dict) or set(files) != expected_names:
        fail("Incomplete or unexpected 1,000-epoch source-lock file set")
    if files["experiments_iclr/ogbn_arxiv_300_results/run_config.json"] != EXPECTED_300_CONFIG_SHA256:
        fail("The 300-epoch run-config hash differs from the frozen protocol")
    if files["experiments_iclr/ogbn_arxiv_300_results/dataset_manifest.json"] != EXPECTED_DATA_SHA256:
        fail("The official OGB manifest hash differs from the frozen protocol")
    for name, pinned_hash in PINNED_SOURCE.items():
        if files[name] != pinned_hash:
            fail(f"Frozen source hash differs from the 100/300 protocol: {name}")
    for name, expected_hash in files.items():
        if (not isinstance(expected_hash, str) or len(expected_hash) != 64
                or any(char not in "0123456789abcdef" for char in expected_hash)):
            fail(f"Invalid lock digest: {name}")
        path = REPO / name
        regular_file(path)
        if sha256_file(path) != expected_hash:
            fail(f"Source/provenance hash mismatch: {name}")
    return files


def check_complete_layout(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        fail(f"Missing or linked 1,000-epoch result root: {root}")
    expected_top = {
        "dataset_manifest.json", "run_config.json", "summary.csv", "summary.json",
        *(f"seed_{seed}" for seed in base.SEEDS),
    }
    if {path.name for path in root.iterdir()} != expected_top:
        fail("1,000-epoch result root has missing or extra top-level entries")
    for name in ("dataset_manifest.json", "run_config.json", "summary.csv", "summary.json"):
        regular_file(root / name)
    for seed in base.SEEDS:
        seed_dir = root / f"seed_{seed}"
        if seed_dir.is_symlink() or not seed_dir.is_dir():
            fail(f"Missing or linked seed directory: {seed_dir}")
        if {path.name for path in seed_dir.iterdir()} != set(base.VARIANTS):
            fail(f"Incomplete or unexpected variant directory for seed {seed}")
        for variant in base.VARIANTS:
            pair_dir = seed_dir / variant
            if pair_dir.is_symlink() or not pair_dir.is_dir():
                fail(f"Missing or linked pair directory: {pair_dir}")
            expected_pair = {
                "selected.json", "epochs.csv", "selected_checkpoint.pt",
                "selected_predictions.npz",
            }
            if {path.name for path in pair_dir.iterdir()} != expected_pair:
                fail(f"Incomplete or unexpected artifacts: {pair_dir}")
            for name in expected_pair:
                regular_file(pair_dir / name)


def close(actual: Any, expected: Any, label: str, tol: float = 1e-12) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        fail(f"Boolean value in {label}")
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {label}") from exc
    if not math.isfinite(a) or not math.isfinite(e) or abs(a - e) > tol:
        fail(f"{label} mismatch: {a!r} != {e!r}")


def selected_rows(root: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows = {}
    for seed, variant in PAIR_NAMES:
        row = json.loads((root / f"seed_{seed}" / variant / "selected.json").read_text())
        if row.get("seed") != seed or row.get("variant") != variant:
            fail(f"Selected identity mismatch: seed {seed}, {variant}")
        rows[(seed, variant)] = row
    return rows


def verify_summary(root: Path, selected: dict[tuple[int, str], dict[str, Any]]) -> None:
    with (root / "summary.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SUMMARY_COLUMNS:
            fail("summary.csv header differs from the frozen producer schema")
        rows = list(reader)
    if len(rows) != len(PAIR_NAMES):
        fail("summary.csv is not the complete nine-pair table")
    seen: set[tuple[int, str]] = set()
    for csv_row in rows:
        if None in csv_row or any(value is None for value in csv_row.values()):
            fail("Truncated or extended summary.csv row")
        try:
            key = (int(csv_row["seed"]), csv_row["variant"])
        except ValueError as exc:
            raise ValueError("Invalid summary.csv seed") from exc
        if key not in selected or key in seen:
            fail(f"Missing, unexpected, or duplicate summary row: {key}")
        seen.add(key)
        expected = selected[key]
        for field in SUMMARY_COLUMNS:
            value = csv_row[field]
            reference = expected[field]
            if reference is None:
                if value != "":
                    fail(f"summary.csv {field} should be empty for {key}")
            elif field in {"variant", "checkpoint_sha256"}:
                if value != reference:
                    fail(f"summary.csv {field} mismatch for {key}")
            elif field in {"seed", "selected_epoch", "parameter_count",
                           "trainable_parameter_count", "epochs_run"}:
                if value != str(reference):
                    fail(f"summary.csv {field} mismatch for {key}")
            else:
                close(value, reference, f"summary.csv {field} for {key}")
    if seen != set(PAIR_NAMES):
        fail("summary.csv omitted a canonical pair")

    summary = json.loads((root / "summary.json").read_text())
    if not isinstance(summary, dict) or set(summary) != {
        "unit_of_replication", "accuracy_units", "variants", "paired_differences",
    }:
        fail("summary.json schema differs from the frozen producer")
    if summary["unit_of_replication"] != "optimization seed on one official temporal graph split":
        fail("summary.json replication unit changed")
    if summary["accuracy_units"] != "fraction correct; multiply by 100 for percentage points":
        fail("summary.json units changed")
    if set(summary["variants"]) != set(base.VARIANTS):
        fail("summary.json variant set is incomplete")
    for variant in base.VARIANTS:
        values = np.asarray([selected[(seed, variant)]["test_accuracy"] for seed in base.SEEDS], dtype=float)
        entry = summary["variants"][variant]
        if not isinstance(entry, dict) or set(entry) != {
            "num_seeds", "seeds", "test_accuracy_mean", "test_accuracy_sample_sd",
        } or entry["num_seeds"] != 3 or entry["seeds"] != list(base.SEEDS):
            fail(f"summary.json {variant} structure differs from nine-pair protocol")
        close(entry["test_accuracy_mean"], float(values.mean()), f"{variant} test mean")
        close(entry["test_accuracy_sample_sd"], float(values.std(ddof=1)), f"{variant} test SD")
    expected_pairs = (("gnnm", "ens"), ("gnnm", "base"), ("ens", "base"))
    if set(summary["paired_differences"]) != {f"{left}_minus_{right}" for left, right in expected_pairs}:
        fail("summary.json paired-comparison set differs from protocol")
    for left, right in expected_pairs:
        key = f"{left}_minus_{right}"
        differences = np.asarray([
            selected[(seed, left)]["test_accuracy"] - selected[(seed, right)]["test_accuracy"]
            for seed in base.SEEDS
        ], dtype=float)
        entry = summary["paired_differences"][key]
        if not isinstance(entry, dict) or set(entry) != {"seeds", "differences", "mean", "sample_sd"}:
            fail(f"summary.json {key} structure changed")
        if entry["seeds"] != list(base.SEEDS) or len(entry["differences"]) != 3:
            fail(f"summary.json {key} seed set changed")
        for seed, actual, expected in zip(base.SEEDS, entry["differences"], differences):
            close(actual, expected, f"{key} seed {seed}")
        close(entry["mean"], float(differences.mean()), f"{key} mean")
        close(entry["sample_sd"], float(differences.std(ddof=1)), f"{key} SD")


def verify_complete() -> dict[str, Any]:
    check_source_lock()
    check_complete_layout(RESULT_ROOT)
    if sha256_file(RESULT_ROOT / "dataset_manifest.json") != EXPECTED_DATA_SHA256:
        fail("1,000-epoch OGB manifest differs from the pinned 100/300 data")
    report = base.verify_tree(RESULT_ROOT, DATA_ROOT, CONFIG)
    if report["status"] != "complete" or report["verified_runs"] != 9 or report["missing"]:
        fail("The 1,000-epoch result is not a complete nine-pair run")
    selected = selected_rows(RESULT_ROOT)
    for row in selected.values():
        if row["epochs_run"] != 1000:
            fail("A fixed-budget arm does not contain exactly 1,000 epochs")
    verify_summary(RESULT_ROOT, selected)
    report["profile"] = "fixed-1000"
    report["result_root"] = str(RESULT_ROOT.relative_to(REPO))
    report["source_lock_sha256"] = sha256_file(LOCK_PATH)
    report["summary_sha256"] = {
        "summary.csv": sha256_file(RESULT_ROOT / "summary.csv"),
        "summary.json": sha256_file(RESULT_ROOT / "summary.json"),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="check frozen sources without reading future results")
    args = parser.parse_args()
    if args.preflight:
        files = check_source_lock()
        print(json.dumps({"status": "preflight_passed", "profile": "fixed-1000",
                          "source_lock_sha256": sha256_file(LOCK_PATH),
                          "checked_files": len(files)}, sort_keys=True))
        return 0
    print(json.dumps(verify_complete(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"fixed-1000 ogbn-arxiv verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
