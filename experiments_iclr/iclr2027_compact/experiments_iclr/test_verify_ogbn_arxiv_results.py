"""Synthetic CPU checks for the read-only ogbn-arxiv result verifier.

The fixture lives briefly inside this repository and uses no OGB files or GPU.
"""

from __future__ import annotations

import csv
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from verify_ogbn_arxiv_results import (
    EPOCH_COLUMNS, PAIRS, SELECTION, repo_path, result_pairs,
    sha256_file, tensor_fingerprint, verify_pair,
)


def expect_failure(action, phrase: str) -> None:
    try:
        action()
    except (ValueError, OSError, RuntimeError) as exc:
        assert phrase in str(exc), (phrase, str(exc))
    else:
        raise AssertionError(f"Expected failure containing {phrase!r}")


def score(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    pooled = torch.from_numpy(logits).mean(dim=0)
    target = torch.from_numpy(labels)
    return (
        float((pooled.argmax(dim=-1) == target).float().mean().item()),
        float(F.cross_entropy(pooled, target).item()),
    )


def fixture_logits(members: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.array([
        [[3, 0], [0, 3]],
        [[2, 0], [0, 2]],
        [[-1, 4], [0, 2]],
        [[3, 0], [2, 1]],
    ], dtype=np.float32)[:members]
    test = np.array([
        [[3, 0], [0, 3]],
        [[0, 2], [0, 2]],
        [[2, 0], [3, 0]],
        [[3, 0], [0, 3]],
    ], dtype=np.float32)[:members]
    return valid, test


def write_pair(root: Path, seed: int, variant: str, manifest: dict,
               official_indices: dict[str, np.ndarray], labels: np.ndarray) -> None:
    directory = root / f"seed_{seed}" / variant
    directory.mkdir(parents=True)
    members = 1 if variant == "base" else 4
    valid_logits, test_logits = fixture_logits(members)
    valid_labels = labels[official_indices["valid"]]
    test_labels = labels[official_indices["test"]]
    valid_acc, valid_ce = score(valid_logits, valid_labels)
    test_acc, test_ce = score(test_logits, test_labels)
    np.savez_compressed(
        directory / "selected_predictions.npz",
        valid_member_logits=valid_logits, valid_labels=valid_labels,
        valid_indices=official_indices["valid"],
        test_member_logits=test_logits, test_labels=test_labels,
        test_indices=official_indices["test"],
    )
    checkpoint = directory / "selected_checkpoint.pt"
    torch.save({
        "state_dict": {"synthetic_weight": torch.tensor([1.0])},
        "epoch": 2, "variant": variant, "seed": seed,
    }, checkpoint)
    rows = [
        {
            "epoch": 1, "train_mean_member_ce": 1.0,
            "valid_pooled_accuracy": 0.5, "valid_pooled_ce": 1.0,
            "train_step_seconds": 0.1, "validation_seconds": 0.01,
            "elapsed_seconds": 0.11, "is_selected_so_far": 1,
        },
        {
            "epoch": 2, "train_mean_member_ce": 0.5,
            "valid_pooled_accuracy": valid_acc, "valid_pooled_ce": valid_ce,
            "train_step_seconds": 0.2, "validation_seconds": 0.02,
            "elapsed_seconds": 0.35, "is_selected_so_far": 1,
        },
    ]
    with (directory / "epochs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPOCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    selected = {
        "seed": seed, "variant": variant, "selected_epoch": 2,
        "validation_selection": SELECTION,
        "valid_accuracy": valid_acc, "valid_ce": valid_ce,
        "test_accuracy": test_acc, "test_ce": test_ce,
        "parameter_count": 1, "trainable_parameter_count": 1,
        "epochs_run": 2, "train_step_seconds_total": 0.3,
        "validation_seconds_total": 0.03, "total_seconds": 0.5,
        "mean_train_step_ms": 150.0, "median_train_step_ms": 150.0,
        "peak_allocated_mib": None,
        "checkpoint_sha256": sha256_file(checkpoint),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
    }
    (directory / "selected.json").write_text(json.dumps(selected))


def main() -> None:
    torch.set_num_threads(1)
    temp_parent = repo_path("experiments_iclr/.tmp/ogbn_arxiv_verifier_tests")
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as temporary:
        root = Path(temporary)
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        indices = {
            "valid": np.array([2, 3], dtype=np.int64),
            "test": np.array([4, 5], dtype=np.int64),
        }
        manifest = {
            "num_nodes": 6, "num_classes": 2,
            "split_sizes": {"valid": 2, "test": 2},
            "fingerprints_sha256": {
                "valid_index": tensor_fingerprint(indices["valid"]),
                "test_index": tensor_fingerprint(indices["test"]),
            },
        }
        config = {"configuration": {
            "max_epochs": 2, "min_epochs": 2, "patience": 2,
            "eval_every": 1,
        }}
        for seed, variant in PAIRS:
            write_pair(root, seed, variant, manifest, indices, labels)
        present, partial = result_pairs(root)
        assert len(present) == 9 and not partial
        reports = [verify_pair(root, seed, variant, manifest, indices, labels,
                               config, check_model=False)
                   for seed, variant in PAIRS]
        assert len(reports) == 9
        assert reports[0]["valid"]["accuracy"] == 1.0
        assert reports[0]["test"]["pairwise_prediction_disagreement"] is None
        assert reports[1]["test"]["pairwise_prediction_disagreement"] is not None
        assert sum(reports[1]["test"]["correct_member_counts"]) == 2

        selected_path = root / "seed_0" / "base" / "selected.json"
        original = selected_path.read_text()
        altered = json.loads(original)
        altered["selected_epoch"] = 1
        selected_path.write_text(json.dumps(altered))
        expect_failure(lambda: verify_pair(root, 0, "base", manifest, indices, labels,
                                           config, check_model=False), "validation-only replay")
        selected_path.write_text(original)

        prediction = selected_path.parent / "selected_predictions.npz"
        prediction.rename(prediction.with_suffix(".absent"))
        expect_failure(lambda: verify_pair(root, 0, "base", manifest, indices, labels,
                                           config, check_model=False), "Missing, empty")
        prediction.with_suffix(".absent").rename(prediction)

        checkpoint = selected_path.parent / "selected_checkpoint.pt"
        checkpoint_bytes = checkpoint.read_bytes()
        with checkpoint.open("ab") as stream:
            stream.write(b"corrupt")
        expect_failure(lambda: verify_pair(root, 0, "base", manifest, indices, labels,
                                           config, check_model=False), "Checkpoint SHA256")
        checkpoint.write_bytes(checkpoint_bytes)
        assert sha256_file(checkpoint) == json.loads(original)["checkpoint_sha256"]

        last_selected = root / "seed_2" / "gnnm" / "selected.json"
        last_selected.unlink()
        present, partial = result_pairs(root)
        assert len(present) == 8 and partial == ["seed_2/gnnm"]
    print("CPU synthetic verifier checks passed: nine pairs, validation selection, missing archive, changed checkpoint, partial run")


if __name__ == "__main__":
    main()
