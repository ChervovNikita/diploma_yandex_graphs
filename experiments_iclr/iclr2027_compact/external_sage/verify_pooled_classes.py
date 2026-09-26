"""Verify the 18-cell external SAGE pooled-logit/member-class public projection."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get("GNNM_EXTERNAL_SAGE_BASE", str(ROOT)))
ARMS = ("tied", "untied_propagation", "all_layer_be")
GRAPHS = ("wikics", "actor")
SEEDS = (0, 1, 2)


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    x = logits.astype(np.float64)
    shift = x.max(axis=1)
    return float(np.mean(shift + np.log(np.exp(x - shift[:, None]).sum(axis=1))
                         - x[np.arange(labels.size), labels]))


def verify_graph(graph: str, manifest: dict) -> dict:
    source_path = BASE / "results" / graph / "source_manifest.json"
    source = read(source_path)
    need(source["dataset"] == graph and source["split"] == 0 and
         source["optimization_seeds"] == list(SEEDS) and
         source["arms"] == list(ARMS), f"Source manifest {graph}")
    for name, digest in source["source_sha256"].items():
        path = BASE / name
        if path.exists():
            need(sha_file(path) == digest, f"Source hash {name}")
        else:
            need(name.startswith("data/"), f"Missing code/protocol file {name}")
    records = {}
    for seed in SEEDS:
        initial_key = None
        for arm in ARMS:
            key = f"{graph}/seed{seed}/{arm}"
            row = manifest["cells"][key]
            folder = BASE / "results" / key
            result = read(folder / "result.json")
            need((result["dataset"], result["optimization_seed"], result["arm"]) ==
                 (graph, seed, arm), f"Identity {key}")
            need(result["source_manifest_sha256"] == sha_file(source_path) and
                 result["epochs_run"] == 300, f"Protocol/budget {key}")
            need(sha_file(folder / "result.json") == row["source_result_sha256"] and
                 result["artifact_sha256"]["selected_predictions.npz"] ==
                 row["source_predictions_sha256"], f"Original prediction provenance {key}")
            for name in ("initialization.json", "validation_trace.csv"):
                need(sha_file(folder / name) == result["artifact_sha256"][name],
                     f"Source artifact {key}/{name}")
            init = read(folder / "initialization.json")
            this_initial = tuple(init[name] for name in
                                 ("canonical_tied_state_sha256", "cpu_rng_sha256", "cuda_rng_sha256"))
            if initial_key is None:
                initial_key = this_initial
            need(this_initial == initial_key and
                 float(init["paired_initial_logits_max_abs_diff"]) <= 1e-5,
                 f"Paired initial state {key}")
            with (folder / "validation_trace.csv").open(newline="") as stream:
                trace = list(csv.DictReader(stream))
            need(len(trace) == 300 and
                 all(int(t["epoch"]) == i for i, t in enumerate(trace, 1)),
                 f"Validation trace {key}")
            chosen = min(trace, key=lambda t: (-float(t["valid_accuracy"]),
                                                float(t["valid_ce"]), int(t["epoch"])))
            need(int(chosen["epoch"]) == result["selected_epoch"], f"Selected epoch {key}")
            projection_path = ROOT / row["projection_path"]
            need(sha_file(projection_path) == row["projection_sha256"] and
                 projection_path.stat().st_size == row["projection_bytes"],
                 f"Projection file {key}")
            with np.load(projection_path, allow_pickle=False) as pred:
                need(set(pred.files) == {f"{part}_{field}" for part in ("valid", "test")
                                         for field in ("pooled_logits", "member_class", "indices", "labels")},
                     f"Projection schema {key}")
                result_metrics = {}
                for part in ("valid", "test"):
                    pool = pred[f"{part}_pooled_logits"]
                    classes = pred[f"{part}_member_class"]
                    labels = pred[f"{part}_labels"]
                    ids = pred[f"{part}_indices"]
                    need(pool.dtype == np.float32 and classes.dtype == np.int16 and
                         labels.dtype == ids.dtype == np.int64 and
                         pool.shape == (len(labels), source["data_descriptor"]["num_classes"]) and
                         classes.shape == (4, len(labels)) and len(ids) == len(labels) and
                         np.isfinite(pool).all(), f"Projection shape/dtype {key}/{part}")
                    for field, array in (("pooled_logits", pool), ("member_class", classes),
                                         ("labels", labels), ("indices", ids)):
                        record = row["arrays"][f"{part}_{field}"]
                        need(str(array.dtype) == record["dtype"] and
                             list(array.shape) == record["shape"] and
                             sha_array(array) == record["bytes_sha256"],
                             f"Projection array hash {key}/{part}_{field}")
                    accuracy = float(np.mean(pool.argmax(axis=1) == labels))
                    cross_entropy = ce(pool, labels)
                    need(abs(accuracy - result[f"{part}_accuracy"]) < 1e-7 and
                         abs(cross_entropy - result[f"{part}_ce"]) < 1e-5,
                         f"Recorded score {key}/{part}")
                    result_metrics[part] = {"accuracy": accuracy, "ce": cross_entropy}
                    if part == "test":
                        result_metrics["member_mean"] = float(np.mean(classes == labels[None, :]))
                        result_metrics["pool_gain"] = accuracy - result_metrics["member_mean"]
            records[(seed, arm)] = result_metrics
    gpu = read(BASE / "results" / graph / "completion_audit.json")
    need(gpu["status"] == "COMPLETE_CUDA_REPLAY_PASS" and len(gpu["records"]) == 9,
         f"GPU replay audit {graph}")
    need({(r["seed"], r["arm"]) for r in gpu["records"]} == set(records),
         f"GPU replay keys {graph}")
    for original in gpu["records"]:
        metric = records[(original["seed"], original["arm"])]
        need(abs(metric["test"]["accuracy"] - original["test_accuracy"]) < 1e-7 and
             abs(metric["member_mean"] - original["mean_member_test_accuracy"]) < 1e-7 and
             abs(metric["pool_gain"] - original["pooling_gain"]) < 1e-7 and
             original["test_max_logit_drift"] < 1e-4 and
             original["valid_max_logit_drift"] < 1e-4,
             f"GPU replay values {graph}/{original['seed']}/{original['arm']}")
    return {arm: [100 * records[(seed, arm)]["test"]["accuracy"] for seed in SEEDS]
            for arm in ARMS}


def main() -> None:
    manifest = read(ROOT / "PROJECTION_MANIFEST.json")
    need(manifest["protocol"] == "external_sage_pooled_logits_member_classes_v1" and
         manifest["cell_count"] == len(manifest["cells"]) == 18, "Projection manifest")
    expected = {f"{graph}/seed{seed}/{arm}" for graph in GRAPHS for seed in SEEDS for arm in ARMS}
    need(set(manifest["cells"]) == expected, "Complete cell set")
    need(sum(r["source_predictions_bytes"] for r in manifest["cells"].values()) ==
         manifest["source_npz_total_bytes"] == 13_068_802 and
         sum(r["projection_bytes"] for r in manifest["cells"].values()) ==
         manifest["projection_npz_total_bytes"] == 3_631_396,
         "Projection bytes")
    scores = {graph: verify_graph(graph, manifest) for graph in GRAPHS}
    print(json.dumps({"status": "ALL_18_EXTERNAL_SAGE_PROJECTIONS_PASS", "cells": 18,
                      "source_to_projection_saving_bytes": 9_437_406,
                      "test_accuracy_pct_by_seed": scores}, sort_keys=True))


if __name__ == "__main__":
    main()
