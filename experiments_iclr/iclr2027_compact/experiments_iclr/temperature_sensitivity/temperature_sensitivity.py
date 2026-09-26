"""Two-stage, post hoc validation-only scalar temperature sensitivity.

`fit` reads validation-only logits/labels and writes all 72 betas. `score`
verifies the complete lock before it opens any test companion or label file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


GRAPHS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
METRICS = ("accuracy", "nll", "brier", "ece10")
LOW, HIGH, STEPS = 0.01, 100.0, 100
LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
PROTOCOL = Path(__file__).resolve().with_name("TEMPERATURE_PROTOCOL.md")
DESIGN_FREEZE = Path(__file__).resolve().with_name("TEMPERATURE_DESIGN_FREEZE.json")
DESIGN_AMENDMENT = Path(__file__).resolve().with_name("TEMPERATURE_DESIGN_AMENDMENT_V2.json")
ORIGINAL_FREEZE_SHA = "85ddd0cb187410929065f1a978f35c1baa6e60830354669c544398418afe3416"
ORIGINAL_SOURCE_SHA = "b27e284276de92d8014dacf7903167d1fa775921f3c4c8a3c9888ea22a8b9758"
ORIGINAL_EXTRACTOR_SHA = "c010e5a7a1a28e9694ba968533fd8f992bef5fbf12de0fb575defc33fbf7095a"


def need(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_design_provenance() -> tuple[str, str]:
    need(sha(DESIGN_FREEZE) == ORIGINAL_FREEZE_SHA, "Original pre-execution design freeze")
    first = read(DESIGN_FREEZE)
    need(first["status"] == "SOURCE_AND_PROTOCOL_FROZEN_BEFORE_VALIDATION_EXTRACTION_OR_TEMPERATURE_FIT" and
         first["fit_score_source_sha256"] == ORIGINAL_SOURCE_SHA and
         first["extractor_sha256"] == ORIGINAL_EXTRACTOR_SHA and
         first["protocol_sha256"] == sha(PROTOCOL), "Original design/source binding")
    amended = read(DESIGN_AMENDMENT)
    need(amended["status"] == "PRE_EXECUTION_SOURCE_HARDENING_V2" and
         amended["original_design_freeze_sha256"] == ORIGINAL_FREEZE_SHA and
         amended["original_fit_score_source_sha256"] == ORIGINAL_SOURCE_SHA and
         amended["original_extractor_sha256"] == ORIGINAL_EXTRACTOR_SHA and
         amended["amended_fit_score_source_sha256"] == sha(Path(__file__)) and
         amended["protocol_sha256"] == sha(PROTOCOL), "Versioned source amendment")
    return ORIGINAL_FREEZE_SHA, sha(DESIGN_AMENDMENT)


def keys(selection_lock: dict) -> list[str]:
    found = []
    for graph in GRAPHS:
        for arm in ARMS:
            lr, wd = selection_lock["selections"][graph][arm]["selected_candidate"]
            found.extend(f"{graph}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}" for seed in range(3))
    need(len(found) == len(set(found)) == 72, "Selected key cardinality")
    return found


def logits_probabilities(logits: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    scaled = logits.astype(np.float64) * beta
    peak = scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled - peak)
    denom = exp.sum(axis=1, keepdims=True)
    return exp / denom, peak[:, 0] + np.log(denom[:, 0])


def nll_derivative(logits: np.ndarray, labels: np.ndarray, beta: float) -> tuple[float, float]:
    probabilities, logsumexp = logits_probabilities(logits, beta)
    z = logits.astype(np.float64)
    truth = z[np.arange(len(labels)), labels]
    nll = float(np.mean(logsumexp - beta * truth))
    derivative = float(np.mean(np.sum(probabilities * z, axis=1) - truth))
    need(np.isfinite(nll) and np.isfinite(derivative), "Nonfinite temperature objective")
    return nll, derivative


def fit_beta(logits: np.ndarray, labels: np.ndarray) -> dict:
    nll_lo, derivative_lo = nll_derivative(logits, labels, LOW)
    nll_hi, derivative_hi = nll_derivative(logits, labels, HIGH)
    need(derivative_lo <= derivative_hi + 1e-9, "NLL derivative is not monotone")
    if derivative_lo >= 0:
        beta, branch = LOW, "lower_endpoint"
    elif derivative_hi <= 0:
        beta, branch = HIGH, "upper_endpoint"
    else:
        lo, hi = LOW, HIGH
        for _ in range(STEPS):
            midpoint = (lo + hi) / 2
            _, gradient = nll_derivative(logits, labels, midpoint)
            if gradient < 0:
                lo = midpoint
            else:
                hi = midpoint
        beta, branch = (lo + hi) / 2, "interior_bisection_100"
    before, _ = nll_derivative(logits, labels, 1.0)
    after, at_beta_derivative = nll_derivative(logits, labels, beta)
    need(after <= before + 1e-8,
         "Fitted validation NLL unexpectedly exceeds no scaling")
    return {"beta": beta, "temperature": 1.0 / beta, "branch": branch,
            "derivative_at_low": derivative_lo, "derivative_at_high": derivative_hi,
            "derivative_at_beta": at_beta_derivative,
            "nll_at_low": nll_lo, "nll_at_high": nll_hi,
            "validation_nll_before": before, "validation_nll_after": after}


def verified_validation_arrays(validation_stage: Path, study: Path) -> tuple[dict, list[str]]:
    selection_path = study / "VALIDATION_SELECTION_LOCK.json"
    need(sha(selection_path) == LOCK_SHA, "Original validation-selection lock")
    selection = read(selection_path)
    selected_keys = keys(selection)
    manifest = read(validation_stage / "VALIDATION_ONLY_MANIFEST.json")
    need(manifest["protocol"] == "selected_hpo72_validation_only_logits_transport_v1" and
         manifest["selection_lock_sha256"] == LOCK_SHA and
         manifest["cell_count"] == len(manifest["cells"]) == 72 and
         set(manifest["cells"]) == set(selected_keys), "Validation-only manifest coverage")
    frozen = read(study / "FROZEN_STUDY.json")
    need(sha(study / "FROZEN_STUDY.json") == manifest["frozen_study_sha256"],
         "Frozen study source")
    for graph in GRAPHS:
        reference = manifest["references"][graph]
        path = validation_stage / reference["path"]
        need(sha(path) == reference["sha256"], f"Validation reference {graph}")
        with np.load(path, allow_pickle=False) as loaded:
            need(set(loaded.files) == {"valid_indices", "valid_labels"},
                 f"Validation reference schema {graph}")
            indices, labels = loaded["valid_indices"], loaded["valid_labels"]
        need(indices.dtype == labels.dtype == np.int64 and
             len(labels) == reference["valid_nodes"] ==
             frozen["graphs"][graph]["descriptor"]["split_sizes"]["valid"],
             f"Validation reference values {graph}")
    return manifest, selected_keys


def validation_cell(validation_stage: Path, manifest: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    graph = key.split("/")[0]
    row = manifest["cells"][key]
    path = validation_stage / row["path"]
    need(sha(path) == row["sha256"], f"Validation logits hash {key}")
    with np.load(path, allow_pickle=False) as loaded:
        need(set(loaded.files) == {"valid_pooled_logits"}, f"Validation logits schema {key}")
        logits = loaded["valid_pooled_logits"]
    with np.load(validation_stage / manifest["references"][graph]["path"], allow_pickle=False) as ref:
        labels = ref["valid_labels"]
    need(logits.dtype == np.float32 and list(logits.shape) == row["shape"] and
         logits.shape[0] == len(labels) and np.isfinite(logits).all(),
         f"Validation logits values {key}")
    return logits, labels


def fit(args: argparse.Namespace) -> None:
    design_sha, amendment_sha = verify_design_provenance()
    need(not args.lock.exists(), "Refusing to overwrite a temperature lock")
    manifest, selected_keys = verified_validation_arrays(args.validation_stage, args.study)
    fitted = {}
    for key in selected_keys:
        logits, labels = validation_cell(args.validation_stage, manifest, key)
        fitted[key] = {**fit_beta(logits, labels),
                       "validation_logits_sha256": manifest["cells"][key]["sha256"]}
    lock = {
        "protocol": "posthoc_validation_scalar_temperature_all72_v1",
        "qualification": "Post hoc probability-quality sensitivity; validation reused after checkpoint/hyperparameter selection; test already inspected before design",
        "selection_lock_sha256": LOCK_SHA,
        "validation_only_manifest_sha256": sha(args.validation_stage / "VALIDATION_ONLY_MANIFEST.json"),
        "temperature_source_sha256": sha(Path(__file__)),
        "temperature_protocol_sha256": sha(PROTOCOL),
        "original_design_freeze_sha256": design_sha,
        "design_amendment_sha256": amendment_sha,
        "beta_bounds": [LOW, HIGH], "bisection_steps": STEPS,
        "cell_count": len(fitted), "cells": fitted,
    }
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "ALL72_VALIDATION_TEMPERATURE_LOCK_WRITTEN",
                      "cell_count": len(fitted), "lock_sha256": sha(args.lock)}, sort_keys=True))


def verify_lock(validation_stage: Path, study: Path, lock_path: Path) -> tuple[dict, list[str]]:
    design_sha, amendment_sha = verify_design_provenance()
    manifest, selected_keys = verified_validation_arrays(validation_stage, study)
    lock = read(lock_path)
    need(lock["protocol"] == "posthoc_validation_scalar_temperature_all72_v1" and
         lock["selection_lock_sha256"] == LOCK_SHA and
         lock["validation_only_manifest_sha256"] == sha(validation_stage / "VALIDATION_ONLY_MANIFEST.json") and
         lock["temperature_source_sha256"] == sha(Path(__file__)) and
         lock["temperature_protocol_sha256"] == sha(PROTOCOL) and
         lock["original_design_freeze_sha256"] == design_sha and
         lock["design_amendment_sha256"] == amendment_sha and
         lock["beta_bounds"] == [LOW, HIGH] and lock["bisection_steps"] == STEPS and
         lock["cell_count"] == len(lock["cells"]) == 72 and
         set(lock["cells"]) == set(selected_keys), "Temperature lock header/coverage")
    for key in selected_keys:
        logits, labels = validation_cell(validation_stage, manifest, key)
        expected = fit_beta(logits, labels)
        recorded = lock["cells"][key]
        need(recorded["validation_logits_sha256"] == manifest["cells"][key]["sha256"],
             f"Temperature source array {key}")
        for name, value in expected.items():
            if isinstance(value, str):
                need(recorded[name] == value, f"Temperature branch {key}")
            else:
                need(abs(recorded[name] - value) <= 1e-12 * max(1.0, abs(value)),
                     f"Temperature replay {key}/{name}")
    return lock, selected_keys


def probability_metrics(logits: np.ndarray, labels: np.ndarray, beta: float) -> dict:
    probabilities, logsumexp = logits_probabilities(logits, beta)
    truth = logits.astype(np.float64)[np.arange(len(labels)), labels]
    nll = float(np.mean(logsumexp - beta * truth))
    selected = probabilities[np.arange(len(labels)), labels]
    brier = float(np.mean(np.sum(probabilities ** 2, axis=1) - 2 * selected + 1))
    classes = probabilities.argmax(axis=1)
    correct = (classes == labels).astype(np.float64)
    confidence = probabilities.max(axis=1)
    bins = np.minimum(np.floor(confidence * 10).astype(np.int64), 9)
    ece = 0.0
    for b in range(10):
        mask = bins == b
        if mask.any():
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    return {"accuracy": float(correct.mean()), "nll": nll,
            "brier": brier, "ece10": ece}


def score(args: argparse.Namespace) -> None:
    lock, selected_keys = verify_lock(args.validation_stage, args.study, args.lock)
    need(not args.out.exists(), "Refusing to overwrite test sensitivity output")
    # No test companion or label file is opened before the complete lock replay.
    companion = read(args.test_companion / "MANIFEST.json")
    need(companion["cell_count"] == 72 and set(companion["cells"]) == set(selected_keys) and
         companion["selection_lock_sha256"] == LOCK_SHA,
         "Complete exact test companion")
    rows = []
    labels = {}
    hard = read(args.study / "HARD_DECISION_EXPORT_MANIFEST.json")
    for graph in GRAPHS:
        ref_path = args.study / hard["graph_references"][graph]["reference_path"]
        need(sha(ref_path) == hard["graph_references"][graph]["reference_sha256"],
             f"Test reference hash {graph}")
        with np.load(ref_path, allow_pickle=False) as ref:
            labels[graph] = ref["full_labels"][ref["test_indices"]]
    for key in selected_keys:
        graph, arm = key.split("/")[:2]
        seed = int(key.rsplit("seed", 1)[1])
        test_path = args.test_companion / "logits" / f"{key}.npz"
        need(sha(test_path) == companion["cells"][key]["derivative_npz_sha256"],
             f"Exact test companion hash {key}")
        with np.load(test_path, allow_pickle=False) as loaded:
            member = loaded["test_member_logits"]
        pooled = member.mean(axis=0)
        need(hashlib.sha256(pooled.tobytes(order="C")).hexdigest() ==
             companion["cells"][key]["test_pooled_logits_bytes_sha256"],
             f"Original pooled-logit bytes {key}")
        beta = lock["cells"][key]["beta"]
        original = probability_metrics(pooled, labels[graph], 1.0)
        scaled = probability_metrics(pooled, labels[graph], beta)
        need(abs(original["accuracy"] - scaled["accuracy"]) < 1e-12 and
             np.array_equal(pooled.argmax(axis=1),
                            (pooled.astype(np.float64) * beta).argmax(axis=1)),
             f"Positive scalar changed class decisions {key}")
        row = {"key": key, "graph": graph, "arm": arm, "seed": seed,
               "test_nodes": len(labels[graph]), "beta": beta,
               "temperature": 1.0 / beta}
        for metric in METRICS:
            row[f"before_{metric}"] = original[metric]
            row[f"after_{metric}"] = scaled[metric]
            row[f"delta_{metric}"] = scaled[metric] - original[metric]
        rows.append(row)
    need(len(rows) == 72, "All 72 test cells")
    args.out.mkdir(parents=True)
    csv_path = args.out / "TEMPERATURE_TEST_ALL72.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_key = {(r["graph"], r["arm"], r["seed"]): r for r in rows}
    summary = {}
    for graph in GRAPHS:
        summary[graph] = {}
        for arm in ARMS:
            summary[graph][arm] = {}
            for metric in METRICS:
                vals = np.array([by_key[(graph, arm, seed)][f"delta_{metric}"]
                                 for seed in range(3)], dtype=np.float64)
                summary[graph][arm][metric] = {
                    "paired_seed_deltas_after_minus_before": [float(v) for v in vals],
                    "mean_delta": float(vals.mean()),
                    "sample_sd_delta": float(vals.std(ddof=1)),
                }
    result = {"protocol": "posthoc_validation_scalar_temperature_all72_v1",
              "qualification": lock["qualification"],
              "temperature_validation_lock_sha256": sha(args.lock),
              "test_companion_manifest_sha256": sha(args.test_companion / "MANIFEST.json"),
              "source_sha256": sha(Path(__file__)),
              "protocol_sha256": sha(PROTOCOL),
              "cell_count": len(rows), "test_all72_csv_sha256": sha(csv_path),
              "summary": summary}
    (args.out / "TEMPERATURE_TEST_SUMMARY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "ALL72_POSTHOC_TEMPERATURE_SENSITIVITY_COMPLETE",
                      "cells": len(rows),
                      "summary_sha256": sha(args.out / "TEMPERATURE_TEST_SUMMARY.json")},
                     sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit", "verify-lock", "score"))
    parser.add_argument("--validation-stage", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--test-companion", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "fit":
        fit(args)
    elif args.command == "verify-lock":
        verify_lock(args.validation_stage, args.study, args.lock)
        print(json.dumps({"status": "ALL72_VALIDATION_TEMPERATURE_LOCK_REPLAY_PASS",
                          "lock_sha256": sha(args.lock)}, sort_keys=True))
    else:
        need(args.test_companion is not None and args.out is not None,
             "score requires --test-companion and --out")
        score(args)


if __name__ == "__main__":
    main()
