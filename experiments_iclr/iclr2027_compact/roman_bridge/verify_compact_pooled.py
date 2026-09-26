"""Public CPU audit of all nine Roman bridge pooled-logit rows.

The original member float logits and checkpoints are omitted. Their frozen
hashes and completed CUDA replay are retained as provenance, but cannot be
replayed from this smaller stage.
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


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def ce(logits: np.ndarray, labels: np.ndarray) -> float:
    values = logits.astype(np.float64)
    peak = values.max(1)
    return float(np.mean(peak + np.log(np.exp(values - peak[:, None]).sum(1))
                         - values[np.arange(len(labels)), labels]))


def main() -> None:
    derivation = read(ROOT / "POOLED_DERIVATION.json")
    source_path = ROOT / "results" / "roman" / "source_manifest.json"
    source = read(source_path)
    require(derivation["protocol"] == "roman_bridge_pooled_member_classes_v1" and
            derivation["source_manifest_sha256"] == sha(source_path) and
            derivation["source_verifier_sha256"] ==
            sha(ROOT / "verify_full_original.py") and
            derivation["public_verifier_sha256"] == sha(Path(__file__)),
            "Pooled derivation source/verifier link differs")
    require(source["dataset"] == "roman" and source["split"] == 0 and
            source["optimization_seeds"] == list(SEEDS) and
            source["arms"] == list(ARMS), "Frozen Roman matrix differs")
    for filename, digest in source["source_sha256"].items():
        path = ROOT / filename
        if path.is_file():
            require(sha(path) == digest, f"Frozen code/data changed: {filename}")
        else:
            require(filename.startswith("data/"),
                    f"Missing frozen source file: {filename}")
    claimed = {(r["seed"], r["arm"]): r for r in derivation["rows"]}
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    require(len(derivation["rows"]) == 9 and set(claimed) == expected,
            "Pooled derivation rows are incomplete or duplicated")
    observations = {}
    for seed in SEEDS:
        initial = None
        for arm in ARMS:
            folder = ROOT / "results" / "roman" / f"seed{seed}" / arm
            result_path = folder / "result.json"
            row = read(result_path)
            link = claimed[(seed, arm)]
            require((row["dataset"], row["optimization_seed"], row["arm"]) ==
                    ("roman", seed, arm) and row["epochs_run"] == 300 and
                    row["source_manifest_sha256"] == sha(source_path) and
                    link["result_sha256"] == sha(result_path) and
                    link["original_member_logit_npz_sha256"] ==
                    row["artifact_sha256"]["selected_predictions.npz"],
                    f"Source result or omitted original hash differs: {seed}/{arm}")
            for name in ("initialization.json", "validation_trace.csv"):
                require(sha(folder / name) == row["artifact_sha256"][name],
                        f"Frozen validation artifact differs: {seed}/{arm}/{name}")
            init = read(folder / "initialization.json")
            init_key = tuple(init[k] for k in
                             ("canonical_tied_state_sha256", "cpu_rng_sha256",
                              "cuda_rng_sha256"))
            if initial is None:
                initial = init_key
            else:
                require(initial == init_key,
                        f"Initial state differs across arms: {seed}")
            require(float(init["paired_initial_logits_max_abs_diff"]) <= 1e-5,
                    f"Initial function gate failed: {seed}/{arm}")
            with (folder / "validation_trace.csv").open(newline="") as stream:
                trace = list(csv.DictReader(stream))
            require(len(trace) == 300 and
                    all(int(r["epoch"]) == i for i, r in enumerate(trace, 1)),
                    f"Validation trace incomplete: {seed}/{arm}")
            selected = min(trace, key=lambda r:
                           (-float(r["valid_accuracy"]),
                            float(r["valid_ce"]), int(r["epoch"])))
            require(int(selected["epoch"]) == int(row["selected_epoch"]),
                    f"Checkpoint selection differs: {seed}/{arm}")
            compact_path = folder / "selected_pooled.npz"
            require(sha(compact_path) == link["pooled_hard_class_npz_sha256"],
                    f"Pooled/hard-class bytes changed: {seed}/{arm}")
            require(not (folder / "selected_predictions.npz").exists(),
                    f"Unlisted original member logits retained: {seed}/{arm}")
            with np.load(compact_path, allow_pickle=False) as saved:
                expected_arrays = {f"{part}_{field}" for part in ("valid", "test")
                                   for field in ("indices", "labels", "pooled_logits",
                                                 "member_classes")}
                require(set(saved.files) == expected_arrays,
                        f"Pooled schema differs: {seed}/{arm}")
                metrics = {}
                for part in ("valid", "test"):
                    logits = saved[f"{part}_pooled_logits"]
                    members = saved[f"{part}_member_classes"]
                    labels = saved[f"{part}_labels"]
                    indices = saved[f"{part}_indices"]
                    require(logits.dtype == np.float32 and logits.ndim == 2 and
                            logits.shape == (len(labels), 18) and
                            members.shape == (4, len(labels)) and
                            np.issubdtype(members.dtype, np.integer) and
                            indices.shape == labels.shape and
                            np.issubdtype(indices.dtype, np.integer) and
                            np.issubdtype(labels.dtype, np.integer) and
                            np.all(np.diff(indices) > 0) and
                            np.min(labels) >= 0 and np.max(labels) < 18 and
                            np.min(members) >= 0 and np.max(members) < 18 and
                            np.isfinite(logits).all(),
                            f"Pooled array invalid: {seed}/{arm}/{part}")
                    accuracy = float(np.mean(logits.argmax(1) == labels))
                    loss = ce(logits, labels)
                    require(abs(accuracy - float(row[f"{part}_accuracy"])) < 1e-7 and
                            abs(loss - float(row[f"{part}_ce"])) < 1e-5,
                            f"Pooled metric differs: {seed}/{arm}/{part}")
                    metrics[f"{part}_accuracy"] = accuracy
                    metrics[f"{part}_ce"] = loss
                    if part == "test":
                        metrics["member_mean"] = float(np.mean(
                            members == labels[None, :]))
                        metrics["pool_gain"] = accuracy - metrics["member_mean"]
                observations[(seed, arm)] = metrics
    audit = read(ROOT / "results" / "roman" / "completion_audit.json")
    require(audit["status"] == "COMPLETE_CUDA_REPLAY_PASS" and
            audit["source_manifest_sha256"] == sha(source_path) and
            len(audit["records"]) == 9 and
            {(r["seed"], r["arm"]) for r in audit["records"]} == expected,
            "Original full checkpoint replay record differs")
    for record in audit["records"]:
        key = (record["seed"], record["arm"])
        metrics = observations[key]
        result = read(ROOT / "results" / "roman" /
                      f"seed{key[0]}" / key[1] / "result.json")
        require(record["checkpoint_sha256"] ==
                result["artifact_sha256"]["checkpoint.pt"] and
                record["selected_epoch"] == result["selected_epoch"] and
                abs(record["valid_accuracy"] - metrics["valid_accuracy"]) < 1e-7 and
                abs(record["test_accuracy"] - metrics["test_accuracy"]) < 1e-7 and
                abs(record["test_ce"] - metrics["test_ce"]) < 1e-5 and
                abs(record["mean_member_test_accuracy"] - metrics["member_mean"]) < 1e-6 and
                abs(record["pooling_gain"] - metrics["pool_gain"]) < 1e-6 and
                record["valid_max_logit_drift"] < 1e-4 and
                record["test_max_logit_drift"] < 1e-4,
                f"Original CUDA replay link differs: {key}")
    print(json.dumps({"status": "POOLED_METRIC_AUDIT_PASS", "rows": 9,
                      "scope": "Recomputed pooled accuracy/CE and member hard-class accuracy; original member-logit pooling and checkpoint replay require omitted full artifacts"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
