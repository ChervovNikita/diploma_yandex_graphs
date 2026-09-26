"""CPU-only corruption checks for the fixed-1,000 OGB completion profile."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

import verify_ogbn_arxiv_results as base
from verify_ogbn_arxiv_1000 import (
    CONFIG, PAIR_NAMES, REPO, SUMMARY_COLUMNS, check_complete_layout,
    verify_summary,
)


def must_fail(action, phrase: str) -> None:
    try:
        action()
    except (OSError, ValueError) as exc:
        assert phrase in str(exc), (phrase, str(exc))
    else:
        raise AssertionError(f"Expected a failure containing {phrase!r}")


def write_epochs(path: Path, *, length: int = 1000, changed_flag: bool = False,
                 last_improves: bool = False) -> dict:
    rows = []
    for epoch in range(1, length + 1):
        improved = epoch == 1 or (last_improves and epoch == 1000)
        rows.append({
            "epoch": epoch,
            "train_mean_member_ce": 1.0,
            "valid_pooled_accuracy": 0.61 if last_improves and epoch == 1000 else 0.6,
            "valid_pooled_ce": 1.0,
            "train_step_seconds": 0.1,
            "validation_seconds": 0.01,
            "elapsed_seconds": 0.11 * epoch,
            "is_selected_so_far": int(improved or (changed_flag and epoch == 500)),
        })
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=base.EPOCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "epochs_run": length,
        "selected_epoch": 1000 if last_improves else 1,
        "valid_accuracy": 0.61 if last_improves else 0.6,
        "valid_ce": 1.0,
        "train_step_seconds_total": 0.1 * length,
        "validation_seconds_total": 0.01 * length,
        "mean_train_step_ms": 100.0,
        "median_train_step_ms": 100.0,
        "total_seconds": 0.12 * length,
    }


def sample_selected() -> dict[tuple[int, str], dict]:
    rows = {}
    for seed, variant in PAIR_NAMES:
        accuracy = 0.6 + 0.01 * seed + 0.001 * base.VARIANTS.index(variant)
        rows[(seed, variant)] = {
            "seed": seed,
            "variant": variant,
            "selected_epoch": 1000,
            "valid_accuracy": accuracy,
            "valid_ce": 1.0,
            "test_accuracy": accuracy - 0.01,
            "test_ce": 1.1,
            "parameter_count": 100,
            "trainable_parameter_count": 100,
            "epochs_run": 1000,
            "train_step_seconds_total": 100.0,
            "validation_seconds_total": 10.0,
            "total_seconds": 111.0,
            "mean_train_step_ms": 100.0,
            "median_train_step_ms": 100.0,
            "peak_allocated_mib": None,
            "checkpoint_sha256": "a" * 64,
        }
    return rows


def write_summary(root: Path, selected: dict[tuple[int, str], dict]) -> None:
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(selected.values())
    variants = {}
    for variant in base.VARIANTS:
        values = np.asarray([selected[(seed, variant)]["test_accuracy"] for seed in base.SEEDS])
        variants[variant] = {
            "num_seeds": 3, "seeds": list(base.SEEDS),
            "test_accuracy_mean": float(values.mean()),
            "test_accuracy_sample_sd": float(values.std(ddof=1)),
        }
    paired = {}
    for left, right in (("gnnm", "ens"), ("gnnm", "base"), ("ens", "base")):
        values = np.asarray([
            selected[(seed, left)]["test_accuracy"] - selected[(seed, right)]["test_accuracy"]
            for seed in base.SEEDS
        ])
        paired[f"{left}_minus_{right}"] = {
            "seeds": list(base.SEEDS), "differences": values.tolist(),
            "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
        }
    (root / "summary.json").write_text(json.dumps({
        "unit_of_replication": "optimization seed on one official temporal graph split",
        "accuracy_units": "fraction correct; multiply by 100 for percentage points",
        "variants": variants, "paired_differences": paired,
    }))


def populate_layout(root: Path) -> None:
    for name in ("dataset_manifest.json", "run_config.json", "summary.csv", "summary.json"):
        (root / name).write_text("x")
    for seed, variant in PAIR_NAMES:
        directory = root / f"seed_{seed}" / variant
        directory.mkdir(parents=True)
        for name in ("selected.json", "epochs.csv", "selected_checkpoint.pt",
                     "selected_predictions.npz"):
            (directory / name).write_text("x")


def main() -> None:
    assert CONFIG == {**base.EXPECTED_CONFIG, "max_epochs": 1000, "min_epochs": 1000}
    parent = REPO / "experiments_iclr/.tmp/ogbn_arxiv_1000_tests"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        root = Path(temporary)
        epoch_path = root / "epochs.csv"
        selected = write_epochs(epoch_path)
        base.verify_epochs(epoch_path, selected, CONFIG)
        selected = write_epochs(epoch_path, last_improves=True)
        base.verify_epochs(epoch_path, selected, CONFIG)
        selected = write_epochs(epoch_path, length=999)
        must_fail(lambda: base.verify_epochs(epoch_path, selected, CONFIG),
                  "stops before max epochs")
        selected = write_epochs(epoch_path, changed_flag=True)
        must_fail(lambda: base.verify_epochs(epoch_path, selected, CONFIG),
                  "Selected flag differs")

    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        root = Path(temporary)
        populate_layout(root)
        check_complete_layout(root)
        missing = root / "seed_2/gnnm/selected_predictions.npz"
        missing.unlink()
        must_fail(lambda: check_complete_layout(root), "Incomplete or unexpected artifacts")

    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        root = Path(temporary)
        selected = sample_selected()
        write_summary(root, selected)
        verify_summary(root, selected)
        summary = json.loads((root / "summary.json").read_text())
        summary["paired_differences"]["gnnm_minus_ens"]["differences"][0] += 0.01
        (root / "summary.json").write_text(json.dumps(summary))
        must_fail(lambda: verify_summary(root, selected), "gnnm_minus_ens seed 0")

    print("OGB1000_CPU_TEST_PASS: fixed history, late best, truncation, selection flag, layout, summary corruption")


if __name__ == "__main__":
    main()
