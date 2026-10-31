"""Small CPU fixtures for the independent ogbn-arxiv untied verifier."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments_iclr import ogbn_arxiv_pilot as pilot
from experiments_iclr import ogbn_arxiv_untied_control as control
from experiments_iclr import verify_ogbn_arxiv_untied as verifier


def rejects(call, fragment: str) -> None:
    try:
        call()
    except ValueError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"Expected {fragment!r}, got {exc!r}") from exc
    else:
        raise AssertionError(f"Expected rejection containing {fragment!r}")


def test_complete_tree(root: Path) -> None:
    result = root / "results"
    for relative in verifier.expected_artifact_files():
        path = result / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    manifest = {
        "sha256": {
            relative: verifier.sha256_file(result / relative)
            for relative in sorted(verifier.expected_artifact_files())
        }
    }
    (result / "artifact_manifest.json").write_text(json.dumps(manifest))
    verifier.verify_tree_layout(result)
    verifier.verify_artifact_manifest(result)
    (result / "seed_0" / "tied" / "extra.txt").write_text("extra")
    rejects(lambda: verifier.verify_tree_layout(result), "extra entries")
    (result / "seed_0" / "tied" / "extra.txt").unlink()
    (result / "seed_0" / "tied" / "epochs.csv").write_text("changed")
    rejects(lambda: verifier.verify_artifact_manifest(result), "SHA256 differs")
    (result / "seed_0" / "tied" / "epochs.csv").unlink()
    rejects(lambda: verifier.verify_tree_layout(result), "missing or extra entries")


def test_initialization(root: Path) -> None:
    audit = control.cpu_smoke()["initialization"]
    path = root / "initialization.json"
    path.write_text(json.dumps(audit))
    verifier.verify_initialization(path, 37)
    audit["untied_stack_sha256"][0] = "0" * 64
    path.write_text(json.dumps(audit))
    rejects(lambda: verifier.verify_initialization(path, 37), "differs initially")


def test_epochs(root: Path) -> None:
    path = root / "epochs.csv"
    rows = []
    for epoch in range(1, 301):
        rows.append({
            "epoch": epoch,
            "train_mean_member_ce": 1.0,
            "valid_pooled_accuracy": 0.4 if epoch <= 2 else 0.3,
            "valid_pooled_ce": 0.8 if epoch != 2 else 0.7,
            "train_step_seconds": 0.01,
            "validation_seconds": 0.02,
            "elapsed_seconds": epoch * 0.03,
            "is_selected_so_far": int(epoch <= 2),
        })
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=verifier.EPOCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    selected = {
        "selected_epoch": 2, "valid_accuracy": 0.4, "valid_ce": 0.7,
        "train_step_seconds_total": 3.0, "validation_seconds_total": 6.0,
        "total_seconds": 9.1, "mean_train_step_ms": 10.0,
        "median_train_step_ms": 10.0,
    }
    verifier.verify_epochs(path, selected)
    wrong = {**selected, "selected_epoch": 3}
    rejects(lambda: verifier.verify_epochs(path, wrong), "validation-only replay")
    rows[-1]["is_selected_so_far"] = 1
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=verifier.EPOCH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    rejects(lambda: verifier.verify_epochs(path, selected), "Selected-so-far flag")


def test_checkpoint_and_logits(root: Path) -> None:
    device = torch.device("cpu")
    cfg = pilot.Configuration(
        backbone="SAGE", layers=2, hidden_dim=16, dropout=.2,
        max_epochs=1, min_epochs=1, patience=1, eval_every=1,
    )
    graph, x = control._fixed_synthetic_graph(5, device)
    official_labels = np.arange(9, dtype=np.int64) % 3
    indices = {
        "valid": np.array([0, 1, 2], dtype=np.int64),
        "test": np.array([3, 4], dtype=np.int64),
    }
    manifest = {
        "num_nodes": 9, "num_features": 5, "num_classes": 3,
        "split_sizes": {"valid": 3, "test": 2},
        "fingerprints_sha256": {
            "valid_index": verifier.tensor_fingerprint(indices["valid"]),
            "test_index": verifier.tensor_fingerprint(indices["test"]),
        },
    }
    for arm in verifier.ARMS:
        model = control.build_arm_model(arm, cfg, 5, 3, device)
        arm_dir = root / arm
        arm_dir.mkdir()
        checkpoint = arm_dir / "selected_checkpoint.pt"
        torch.save({
            "state_dict": model.state_dict(), "epoch": 1,
            "arm": arm, "seed": 0,
        }, checkpoint)
        selected = {
            "arm": arm, "seed": 0, "selected_epoch": 1,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "checkpoint_sha256": verifier.sha256_file(checkpoint),
        }
        loaded = verifier.verify_checkpoint(
            checkpoint, selected, {"configuration": asdict(cfg)}, manifest
        )
        logits = verifier.replay_member_logits(loaded, graph, x, manifest)
        arrays = {}
        for split in ("valid", "test"):
            split_logits = np.stack([
                full.index_select(0, torch.from_numpy(indices[split])).numpy()
                for full in logits
            ])
            target = torch.from_numpy(official_labels[indices[split]])
            pooled = torch.from_numpy(split_logits).mean(dim=0)
            selected[f"{split}_accuracy"] = float(
                (pooled.argmax(dim=-1) == target).float().mean().item()
            )
            selected[f"{split}_ce"] = float(F.cross_entropy(pooled, target).item())
            arrays[f"{split}_member_logits"] = split_logits
            arrays[f"{split}_indices"] = indices[split]
            arrays[f"{split}_labels"] = official_labels[indices[split]]
        predictions = arm_dir / "selected_predictions.npz"
        np.savez_compressed(predictions, **arrays)
        verifier.verify_predictions(
            predictions, selected, manifest, indices, official_labels,
            loaded, graph, x,
        )
        arrays["valid_member_logits"] = arrays["valid_member_logits"].copy()
        arrays["valid_member_logits"][0, 0, 0] += 0.01
        np.savez_compressed(predictions, **arrays)
        rejects(
            lambda: verifier.verify_predictions(
                predictions, selected, manifest, indices, official_labels,
                loaded, graph, x,
            ),
            "differ from checkpoint replay",
        )


def main() -> None:
    temp_parent = verifier.REPO / "experiments_iclr" / ".tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_parent) as name:
        root = Path(name)
        test_complete_tree(root)
        test_initialization(root)
        test_epochs(root)
        test_checkpoint_and_logits(root)
    print("untied verifier synthetic CPU checks: PASS")


if __name__ == "__main__":
    main()
