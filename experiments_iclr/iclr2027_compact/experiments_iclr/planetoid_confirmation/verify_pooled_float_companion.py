"""Verify the post-score pooled float32 add-on against the compact base."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def metrics(logits: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    accuracy = float(np.mean(logits.argmax(axis=-1) == truth))
    values = logits.astype(np.float64)
    maximum = values.max(axis=1)
    lse = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    ce = float(np.mean(lse - values[np.arange(len(truth)), truth]))
    return accuracy, ce


def main() -> None:
    addon = ROOT / "pooled_float_companion"
    manifest_path = addon / "POOLED_FLOAT_COMPANION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    frozen = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    lock = json.loads((ROOT / "VALIDATION_SELECTION_LOCK.json").read_text())
    final = json.loads((ROOT / "FINAL_SCORE_AUDIT.json").read_text())
    strict = json.loads((ROOT / "STRICT_DECISION_AUDIT.json").read_text())
    hard = json.loads((ROOT / "HARD_DECISION_EXPORT_MANIFEST.json").read_text())
    require(manifest["protocol"] == "postscore_exact_float32_pooled_test_logit_transport_v1" and
            manifest["study_protocol"] == frozen["protocol"] and
            manifest["base_compact_manifest_sha256"] == sha(ROOT / "COMPACT_BUNDLE_MANIFEST.json") and
            manifest["frozen_study_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            manifest["selection_lock_sha256"] == sha(ROOT / "VALIDATION_SELECTION_LOCK.json") and
            manifest["final_score_audit_sha256"] == sha(ROOT / "FINAL_SCORE_AUDIT.json") and
            manifest["strict_decision_audit_sha256"] == sha(ROOT / "STRICT_DECISION_AUDIT.json") and
            manifest["hard_decision_export_manifest_sha256"] == sha(ROOT / "HARD_DECISION_EXPORT_MANIFEST.json") and
            manifest["verifier_sha256"] == sha(Path(__file__)) and
            manifest["protocol_sha256"] == sha(addon / "POOLED_FLOAT_COMPANION_PROTOCOL.md") and
            manifest["readme_sha256"] == sha(ROOT / "README.md") and
            len(lock["cells"]) == 216 and
            set(manifest["cells"]) == set(final["scores"]) == set(strict["per_checkpoint"]) ==
            set(hard["cells"]), "Companion/base audit chain differs")
    require(all(strict[name] == 0 for name in (
        "total_valid_decision_mismatches", "total_test_decision_mismatches",
        "total_test_member_decision_mismatches")), "Strict decision audit failed")
    expected_files = set()
    max_ce_error = 0.0
    for key, row in manifest["cells"].items():
        dataset = key.split("/", 1)[0]
        expected_rel = Path("pooled_test_logits") / f"{key}.npy"
        require(row["pooled_file"] == expected_rel.as_posix(), f"Unexpected companion path: {key}")
        path = addon / expected_rel
        expected_files.add(path.resolve())
        require(path.is_file() and sha(path) == row["pooled_file_sha256"],
                f"Companion file missing/changed: {key}")
        logits = np.load(path, allow_pickle=False)
        ref_record = hard["graph_references"][dataset]
        ref_path = ROOT / ref_record["reference_path"]
        require(sha(ref_path) == ref_record["reference_sha256"],
                f"Test reference changed: {dataset}")
        with np.load(ref_path, allow_pickle=False) as ref:
            labels, indices = ref["full_labels"], ref["test_indices"]
        descriptor = frozen["graphs"][dataset]["descriptor"]
        require(tensor_sha(labels) == descriptor["tensor_sha256"]["labels"] and
                tensor_sha(indices) == descriptor["tensor_sha256"]["test_indices"],
                f"Graph label/index fingerprint changed: {dataset}")
        truth = labels[indices]
        require(logits.dtype == np.float32 and
                list(logits.shape) == row["shape"] == [len(truth), descriptor["classes"]] and
                row["dtype"] == "float32" and np.isfinite(logits).all() and
                tensor_sha(logits) == row["pooled_tensor_sha256"] and
                row["source_float_predictions_sha256"] ==
                final["scores"][key]["predictions_sha256"] ==
                hard["cells"][key]["source_float_predictions_sha256"],
                f"Pooled float source/shape/dtype differs: {key}")
        score_path = ROOT / "scores" / key / "score.json"
        require(sha(score_path) == row["source_score_sha256"] ==
                final["scores"][key]["score_sha256"],
                f"Scored metadata changed: {key}")
        score = json.loads(score_path.read_text())
        hard_path = ROOT / hard["cells"][key]["hard_path"]
        require(sha(hard_path) == hard["cells"][key]["hard_sha256"],
                f"Hard classes changed: {key}")
        with np.load(hard_path, allow_pickle=False) as decisions:
            require(np.array_equal(logits.argmax(axis=-1), decisions["pooled_test_class"]),
                    f"Float/hard decision mismatch: {key}")
        accuracy, ce = metrics(logits, truth)
        ce_error = abs(ce - score["test_ce"])
        max_ce_error = max(max_ce_error, ce_error)
        require(abs(accuracy - score["test_accuracy"]) <= 1e-7 and
                abs(accuracy - final["scores"][key]["test_accuracy"]) <= 1e-7 and
                abs(accuracy - strict["per_checkpoint"][key]["test_accuracy"]) <= 1e-7 and
                ce_error <= 1e-5 and
                abs(ce - strict["per_checkpoint"][key]["test_ce"]) <= 1e-5 and
                abs(accuracy - row["recomputed_test_accuracy"]) <= 1e-12 and
                abs(ce - row["recomputed_test_ce_float64"]) <= 1e-12,
                f"Float32 pooled accuracy/CE differs: {key}")
    actual_files = {path.resolve() for path in (addon / "pooled_test_logits").rglob("*.npy")}
    require(actual_files == expected_files, "Extra or missing pooled float files")
    print(json.dumps({"status": "PASS", "pooled_float32_scores": len(expected_files),
                      "max_cross_entropy_replay_error": max_ce_error,
                      "scope": "exact pooled float32 logit transport and independent test accuracy/CE replay"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
