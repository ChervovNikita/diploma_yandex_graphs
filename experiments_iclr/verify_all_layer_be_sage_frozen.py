"""Read-only audit of the frozen all-layer SAGE run and selected checkpoints.

The GAT observation in protocol.json is a launch-time snapshot. A later GAT
rerun must not be substituted for it. This verifier never calls the runner's
dynamic GAT-state function and never writes to the result directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments_iclr"))
sys.dont_write_bytecode = True

import all_layer_be_sage as study  # noqa: E402
from datasets import load_dataset  # noqa: E402

RESULT_ROOT = ROOT / "experiments_iclr/all_layer_be_sage_results"
REPLAY_ATOL = 1e-4
REPLAY_RTOL = 1e-4
LAUNCH_LOG = ROOT / "experiments_iclr/logs/all_layer_launcher.log"
FAILED_GAT_WAITER_LOG = ROOT / "experiments_iclr/logs/gat_pair_launcher.log"
QUEUE_LOG = ROOT / "experiments_iclr/logs/run_after_sage.log"
HISTORICAL_GAT = {
    "gat_status": "incomplete_or_failed_after_launcher_exit",
    "gat_final_marker_present": False,
    "gat_verification_exit_code": None,
    "gat_rows_observed": None,
    "gat_protocol_sha256": None,
    "gat_results_sha256": None,
    "gat_run_log_sha256": None,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    require(path.is_file(), f"Missing file: {path}")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_file(path: Path) -> dict:
    require(path.is_file(), f"Missing file: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def verify_frozen_inputs() -> tuple[dict, str, datetime]:
    manifest = json_file(study.MANIFEST_PATH)
    require(set(manifest) == {
        "schema", "protocol", "upstream", "sage_comparators",
        "source_sha256", "python_version", "torch_version",
        "torch_geometric_version",
    }, "Unexpected frozen protocol fields")
    require(manifest["schema"] == 1, "Unexpected frozen protocol schema")
    require(manifest["protocol"] == study.PROTOCOL, "All-layer protocol changed")
    expected_sources = {
        name: digest(ROOT / name) for name in study.SOURCES
    }
    require(manifest["source_sha256"] == expected_sources,
            "Frozen source or Roman Empire data hash differs")
    require(manifest["python_version"] == sys.version.split()[0],
            "Python version differs from frozen run")
    require(manifest["torch_version"] == torch.__version__,
            "Torch version differs from frozen run")
    require(manifest["torch_geometric_version"] ==
            __import__("torch_geometric").__version__,
            "PyG version differs from frozen run")
    require(manifest["sage_comparators"] == study.completed_sage_comparators(),
            "Frozen selected SAGE comparator evidence changed")

    upstream = manifest["upstream"]
    require(isinstance(upstream, dict) and
            set(upstream) == set(HISTORICAL_GAT) |
            {"queue_log_sha256", "queue_success_marker"},
            "Unexpected launch-time upstream fields")
    for key, value in HISTORICAL_GAT.items():
        require(upstream[key] == value,
                f"Launch-time GAT observation differs: {key}")
    require(upstream["queue_success_marker"] ==
            "ENSEMBLE_INFERENCE_PROFILE_COMPLETE",
            "Unexpected original queue terminal marker")
    require(upstream["queue_log_sha256"] == digest(QUEUE_LOG),
            "Original queue log changed after manifest freeze")
    require(QUEUE_LOG.read_text().rstrip().endswith(
            upstream["queue_success_marker"]),
            "Original queue lacks the frozen terminal marker")

    require(LAUNCH_LOG.is_file(), "All-layer launcher log missing")
    lines = LAUNCH_LOG.read_text().splitlines()
    starts = [line for line in lines if line.startswith("ALL_LAYER_LAUNCH_START ")]
    configs = [line for line in lines if line.startswith("CONFIG ")]
    require(len(starts) == len(configs) == 1,
            "Expected one launch and one frozen CONFIG record")
    match = re.fullmatch(
        r"ALL_LAYER_LAUNCH_START (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)",
        starts[0])
    require(match is not None, "Invalid all-layer launch timestamp")
    launch = datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    require(json.loads(configs[0][7:]) == manifest,
            "Launch CONFIG differs from frozen protocol")
    require(study.MANIFEST_PATH.stat().st_mtime >= launch.timestamp() and
            study.MANIFEST_PATH.stat().st_mtime < launch.timestamp() + 600,
            "Frozen protocol timestamp is inconsistent with launch")
    require(FAILED_GAT_WAITER_LOG.is_file(),
            "Original failed GAT waiter log missing")
    waiter = FAILED_GAT_WAITER_LOG.read_text().strip()
    require(waiter == "Original queue exited without its final success marker",
            "Original GAT waiter failure evidence changed")
    require(FAILED_GAT_WAITER_LOG.stat().st_mtime <= launch.timestamp(),
            "GAT waiter failure log postdates all-layer launch")
    return manifest, digest(FAILED_GAT_WAITER_LOG), launch


def verify_selected_files(manifest: dict) -> list[dict[str, str]]:
    # This is a complete-run gate. Inspect the matrix before loading any large
    # checkpoint or replaying the graph, so incomplete runs fail promptly.
    require(study.RESULT_CSV.is_file(), "Incomplete all-layer run: result CSV missing")
    require(study.HASHES_PATH.is_file(),
            "Incomplete all-layer run: artifact hashes missing")
    with study.RESULT_CSV.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == study.controls.FIELDS,
                "Unexpected all-layer CSV columns")
        rows = list(reader)
    require(len(rows) == 5,
            f"Incomplete all-layer run: expected five masks, found {len(rows)}")
    require([int(row["split"]) for row in rows] == list(range(5)),
            "All-layer rows must be the five official masks in order")
    hashes = json_file(study.HASHES_PATH)
    require(set(hashes) == {str(split) for split in range(5)},
            "Artifact hash manifest must contain exactly five masks")
    for split in range(5):
        entry = hashes[str(split)]
        require(isinstance(entry, dict) and
                set(entry) == {"checkpoint", "prediction_file"},
                f"Unexpected artifact hash entries for mask {split}")
        require(all(isinstance(value, str) and
                    re.fullmatch(r"[0-9a-f]{64}", value) for value in entry.values()),
                f"Invalid artifact hash for mask {split}")
    identity = json_file(study.IDENTITY_PATH)
    require(identity.get("schema") == 1 and
            identity.get("source_sha256") == manifest["source_sha256"],
            "Initialization audit differs from frozen source")
    checks = identity.get("checks")
    require(isinstance(checks, list) and len(checks) == 5,
            "Initialization audit needs all five masks")
    for split, check in enumerate(checks):
        require(check == {
            "mask": split,
            "members_checked": 4,
            "initial_logit_max_abs_difference": 0.0,
            "post_construction_rng_matches": True,
            "base_trainable_parameters": study.BASE_PARAMS,
            "added_trainable_parameters": study.ADDED_PARAMS,
            "total_trainable_parameters": study.TOTAL_PARAMS,
        }, f"Invalid initialization audit on mask {split}")
    checked_rows, _ = study.validate_resume(require_complete=True)
    require(checked_rows == rows, "Selected row validation changed the matrix")

    lines = LAUNCH_LOG.read_text().splitlines()
    result_lines = [line[7:] for line in lines if line.startswith("RESULT ")]
    require(len(result_lines) == 5, "Launcher needs five result records")
    for split, (record, row) in enumerate(zip(result_lines, rows)):
        logged = json.loads(record)
        require(isinstance(logged, dict) and
                set(logged) == set(study.controls.FIELDS),
                f"Invalid logged result for mask {split}")
        require(all(str(logged[key]) == row[key] for key in row),
                f"Logged and selected CSV rows differ on mask {split}")
    require(sum(line == "ALL_LAYER_BE_SAGE_COMPLETE" for line in lines) == 1,
            "All-layer trainer has no complete marker")
    # The launcher writes its own complete marker only after invoking the
    # old dynamic GAT-state verifier. Its absence cannot invalidate a complete
    # frozen-manifest run; the selected trainer artifacts and replay decide.
    return rows


def compare_logits(split: int, fresh: np.ndarray, saved: np.ndarray,
                   atol: float, rtol: float) -> dict:
    require(fresh.shape == saved.shape == (4, 5666, 18),
            f"Unexpected replay logit shape on mask {split}")
    require(np.isfinite(fresh).all() and np.isfinite(saved).all(),
            f"Nonfinite replay logits on mask {split}")
    difference = np.abs(fresh.astype(np.float64) -
                        saved.astype(np.float64))
    allowance = atol + rtol * np.abs(saved.astype(np.float64))
    require(np.all(difference <= allowance),
            f"Checkpoint logits differ from saved logits on mask {split}: "
            f"max abs={difference.max():.6g}, max allowance={allowance.max():.6g}")
    fresh_pred = fresh.argmax(axis=-1)
    saved_pred = saved.argmax(axis=-1)
    changes = fresh_pred != saved_pred
    if changes.any():
        member, node = np.nonzero(changes)
        saved_top = saved_pred[changes]
        fresh_top = fresh_pred[changes]
        margin = (saved[member, node, saved_top] -
                  saved[member, node, fresh_top])
        pair_allowance = (2 * atol + rtol *
                          (np.abs(saved[member, node, saved_top]) +
                           np.abs(saved[member, node, fresh_top])))
        require(np.all(margin <= pair_allowance),
                f"Replay changes a member decision beyond tolerance on mask {split}")
    saved_pooled = saved.mean(axis=0)
    fresh_pooled = fresh.mean(axis=0)
    saved_pooled_pred = saved_pooled.argmax(axis=-1)
    fresh_pooled_pred = fresh_pooled.argmax(axis=-1)
    pooled_changes = saved_pooled_pred != fresh_pooled_pred
    if pooled_changes.any():
        node = np.nonzero(pooled_changes)[0]
        saved_top = saved_pooled_pred[pooled_changes]
        fresh_top = fresh_pooled_pred[pooled_changes]
        margin = (saved_pooled[node, saved_top] -
                  saved_pooled[node, fresh_top])
        pair_allowance = (2 * atol + rtol *
                          (np.abs(saved_pooled[node, saved_top]) +
                           np.abs(saved_pooled[node, fresh_top])))
        require(np.all(margin <= pair_allowance),
                f"Replay changes a pooled decision beyond tolerance on mask {split}")
    return {
        "split": split,
        "max_logit_abs_error": float(difference.max(initial=0.0)),
        "max_tolerance_fraction": float(
            (difference / allowance).max(initial=0.0)),
        "decision_changes_within_tolerance": int(changes.sum()),
        "pooled_decision_changes_within_tolerance": int(pooled_changes.sum()),
    }


def replay(rows: list[dict[str, str]], threads: int,
           atol: float, rtol: float) -> list[dict]:
    torch.set_num_threads(threads)
    graph, _, valid_masks, test_masks, _, classes, binary = load_dataset(
        "roman-empire", add_self_loops=True, device=torch.device("cpu"),
        data_dir=str(ROOT / "data"))
    require(graph.x.shape[1] == 300 and classes == 18 and not binary and
            test_masks.shape[1] >= 5, "Unexpected official graph or masks")
    reports = []
    for split, row in enumerate(rows):
        nodes = test_masks[:, split].nonzero(as_tuple=False).flatten()
        labels = graph.y[nodes].cpu().numpy()
        with np.load(study.expected_artifact(split, "prediction_file")[1],
                     allow_pickle=False) as archive:
            saved_nodes = np.asarray(archive["node_index"])
            saved_labels = np.asarray(archive["y_true"])
            saved = np.asarray(archive["member_logits"])
        require(np.array_equal(saved_nodes, nodes.numpy()) and
                np.array_equal(saved_labels, labels),
                f"Saved test nodes or labels differ from official mask {split}")
        checkpoint = study.expected_artifact(split, "checkpoint")[1]
        model = study.AllLayerBESAGEModel(
            study.fixed_args(), graph.x.shape[1], classes,
            torch.device("cpu"))
        require(sum(p.numel() for p in model.parameters() if p.requires_grad) ==
                study.TOTAL_PARAMS, f"Replay model size differs on mask {split}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        started = time.monotonic()
        with torch.inference_mode():
            full_logits = torch.stack([
                model(graph, graph.x, member) for member in range(4)
            ], dim=0)
        fresh = full_logits[:, nodes].cpu().numpy()
        result = compare_logits(split, fresh, saved, atol, rtol)
        pooled = full_logits.mean(0)
        valid = valid_masks[:, split]
        val_accuracy = float((pooled[valid].argmax(-1) ==
                              graph.y[valid]).float().mean())
        # The saved archive contains only test logits. CPU/GPU reduction
        # order can flip a near tie, so use a bounded validation discrepancy
        # and rely on the exact saved-test-logit replay as the primary check.
        require(math.isclose(val_accuracy, float(row["val_metric"]),
                             rel_tol=0, abs_tol=1e-3),
                f"Checkpoint validation accuracy differs on mask {split}")
        test_accuracy = float((pooled[nodes].argmax(-1) ==
                               graph.y[nodes]).float().mean())
        saved_test_accuracy = float(
            (saved.mean(axis=0).argmax(-1) == labels).mean())
        require(math.isclose(saved_test_accuracy, float(row["test_metric"]),
                             rel_tol=0, abs_tol=1e-6),
                f"Saved logits disagree with selected test accuracy on mask {split}")
        result["validation_accuracy"] = val_accuracy
        result["validation_accuracy_delta"] = (
            val_accuracy - float(row["val_metric"]))
        result["test_accuracy"] = test_accuracy
        result["test_accuracy_delta"] = (
            test_accuracy - float(row["test_metric"]))
        result["replay_seconds"] = time.monotonic() - started
        print(json.dumps(result, sort_keys=True), flush=True)
        reports.append(result)
        del model, state, full_logits, pooled
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    started = time.monotonic()
    manifest, waiter_hash, launch = verify_frozen_inputs()
    rows = verify_selected_files(manifest)
    reports = replay(rows, args.threads, REPLAY_ATOL, REPLAY_RTOL)
    print(json.dumps({
        "status": "verified",
        "verified_masks": len(reports),
        "replay_atol": REPLAY_ATOL,
        "replay_rtol": REPLAY_RTOL,
        "launch_utc": launch.astimezone(timezone.utc).isoformat(),
        "historical_gat_status": manifest["upstream"]["gat_status"],
        "launcher_complete_marker_present": any(
            line.startswith("ALL_LAYER_LAUNCH_COMPLETE ")
            for line in LAUNCH_LOG.read_text().splitlines()),
        "original_failed_gat_waiter_sha256": waiter_hash,
        "max_logit_abs_error": max(r["max_logit_abs_error"] for r in reports),
        "max_tolerance_fraction": max(
            r["max_tolerance_fraction"] for r in reports),
        "wall_seconds": time.monotonic() - started,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
