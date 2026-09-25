"""Recalculate selected scores from a compact external SAGE study bundle.

Run this script inside either external_sage/ or roman_bridge/. The full
checkpoint replay needs the omitted checkpoint files and CUDA; its recorded
completion audit is checked here as evidence of that earlier replay.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ("tied", "untied_propagation", "all_layer_be")
SEEDS = (0, 1, 2)


def require(value, message):
    if not value:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cross_entropy(logits, labels):
    values = logits.astype(np.float64)
    peak = values.max(axis=1)
    logsumexp = peak + np.log(np.exp(values - peak[:, None]).sum(axis=1))
    return float(np.mean(logsumexp - values[np.arange(len(labels)), labels]))


def verify_dataset(dataset):
    result_root = ROOT / "results" / dataset
    manifest_path = result_root / "source_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["dataset"] == dataset and manifest["split"] == 0,
            "Manifest graph/split mismatch")
    require(manifest["optimization_seeds"] == list(SEEDS) and
            manifest["arms"] == list(ARMS), "Manifest keys mismatch")
    for filename, expected in manifest["source_sha256"].items():
        path = ROOT / filename
        if path.is_file():
            require(sha(path) == expected, f"Source/data hash mismatch: {filename}")
        else:
            require(filename.startswith("data/"),
                    f"Missing code/protocol file: {filename}")
    records = {}
    for seed in SEEDS:
        starting = None
        for arm in ARMS:
            folder = result_root / f"seed{seed}" / arm
            row = read_json(folder / "result.json")
            require((row["dataset"], row["optimization_seed"], row["arm"])
                    == (dataset, seed, arm), f"Run identity mismatch: {folder}")
            require(row["source_manifest_sha256"] == sha(manifest_path),
                    f"Source manifest mismatch: {folder}")
            require(row["epochs_run"] == 300, f"Training budget mismatch: {folder}")
            for filename in ("initialization.json", "validation_trace.csv",
                             "selected_predictions.npz"):
                require(sha(folder / filename) == row["artifact_sha256"][filename],
                        f"Artifact hash mismatch: {folder / filename}")
            init = read_json(folder / "initialization.json")
            init_key = tuple(init[k] for k in
                             ("canonical_tied_state_sha256", "cpu_rng_sha256",
                              "cuda_rng_sha256"))
            if starting is None:
                starting = init_key
            else:
                require(init_key == starting, f"Initial state mismatch: {folder}")
            require(float(init["paired_initial_logits_max_abs_diff"]) <= 1e-5,
                    f"Initial function tolerance exceeded: {folder}")
            with (folder / "validation_trace.csv").open(newline="") as f:
                trace = list(csv.DictReader(f))
            require(len(trace) == 300 and
                    all(int(item["epoch"]) == i for i, item in enumerate(trace, 1)),
                    f"Validation trace incomplete: {folder}")
            selected = min(trace, key=lambda item: (
                -float(item["valid_accuracy"]), float(item["valid_ce"]),
                int(item["epoch"])))
            require(int(selected["epoch"]) == int(row["selected_epoch"]),
                    f"Checkpoint selection mismatch: {folder}")
            with np.load(folder / "selected_predictions.npz", allow_pickle=False) as f:
                pred = {key: f[key] for key in f.files}
            require(set(pred) == {f"{part}_{field}" for part in ("valid", "test")
                                   for field in ("indices", "labels", "member_logits")},
                    f"Prediction schema mismatch: {folder}")
            result = {}
            for part in ("valid", "test"):
                logits = pred[f"{part}_member_logits"]
                labels = pred[f"{part}_labels"].astype(np.int64)
                ids = pred[f"{part}_indices"]
                require(logits.shape == (4, len(labels), manifest["data_descriptor"]["num_classes"]),
                        f"Prediction shape mismatch: {folder}, {part}")
                require(len(ids) == len(labels) and np.isfinite(logits).all(),
                        f"Invalid predictions: {folder}, {part}")
                pooled = logits.astype(np.float32).mean(axis=0)
                acc = float(np.mean(pooled.argmax(axis=1) == labels))
                ce = cross_entropy(pooled, labels)
                require(abs(acc - float(row[f"{part}_accuracy"])) < 1e-7 and
                        abs(ce - float(row[f"{part}_ce"])) < 1e-5,
                        f"Recomputed metric mismatch: {folder}, {part}")
                result[f"{part}_accuracy"] = acc
                if part == "test":
                    result["member_mean"] = float(np.mean(
                        logits.argmax(axis=2) == labels[None, :]))
                    result["pool_gain"] = acc - result["member_mean"]
            records[(seed, arm)] = result
    gpu_audit = read_json(result_root / "completion_audit.json")
    require(gpu_audit["status"] == "COMPLETE_CUDA_REPLAY_PASS" and
            len(gpu_audit["records"]) == 9, "Full checkpoint replay record missing")
    require({(r["seed"], r["arm"]) for r in gpu_audit["records"]} == set(records),
            "GPU replay run keys mismatch")
    for r in gpu_audit["records"]:
        require(abs(r["test_accuracy"] - records[(r["seed"], r["arm"])]["test_accuracy"]) < 1e-7,
                "GPU replay metric mismatch")
        require(r["test_max_logit_drift"] < 1e-4 and
                r["valid_max_logit_drift"] < 1e-4,
                "GPU replay tolerance mismatch")
    values = {}
    for arm in ARMS:
        scores = [100 * records[(seed, arm)]["test_accuracy"] for seed in SEEDS]
        values[arm] = {"seed_test_accuracy_pct": scores,
                       "mean_pct": float(np.mean(scores)),
                       "sample_sd_pct": float(np.std(scores, ddof=1))}
    diffs = [values["tied"]["seed_test_accuracy_pct"][seed] -
             values["untied_propagation"]["seed_test_accuracy_pct"][seed]
             for seed in SEEDS]
    print(json.dumps({"dataset": dataset, "status": "COMPACT_METRIC_AUDIT_PASS",
                      "scores": values, "tied_minus_untied_pp": diffs}, indent=2))


def main():
    found = [p.name for p in (ROOT / "results").iterdir() if p.is_dir()]
    require(set(found) in ({"wikics", "actor"}, {"roman"}),
            "Unexpected study dataset set")
    for dataset in sorted(found):
        verify_dataset(dataset)


if __name__ == "__main__":
    main()
