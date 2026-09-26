"""Selected partial family versus ENS on exactly paired published test nodes.

Read the frozen validation selection, compact hard decisions, official test
reference, and final score audit. This is descriptive accounting, not an
independent-node significance test. Run from any directory with NumPy.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDIES = {
    "validation_tuning": ("cora", "wikics", "actor", "chameleon_filtered"),
    "planetoid_confirmation": ("citeseer", "pubmed"),
    "cora_preprocessing": ("cora_raw", "cora_normalized"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def cell_key(graph: str, arm: str, candidate: list[float], seed: int) -> str:
    lr, wd = candidate
    return f"{graph}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"


def classes(stage: Path, manifest: dict, score_audit: dict, key: str,
            labels: np.ndarray, classes_count: int) -> tuple[np.ndarray, str]:
    meta = manifest["cells"][key]
    path = stage / meta["hard_path"]
    require(sha(path) == meta["hard_sha256"], f"Hard-decision hash: {key}")
    with np.load(path, allow_pickle=False) as data:
        predicted = data["pooled_test_class"].copy()
    require(predicted.shape == labels.shape and predicted.ndim == 1,
            f"Test shape: {key}")
    require(np.issubdtype(predicted.dtype, np.integer) and
            bool(np.all((0 <= predicted) & (predicted < classes_count))),
            f"Class range: {key}")
    measured = float(np.mean(predicted == labels))
    require(abs(measured - meta["test_accuracy"]) < 1e-12,
            f"Hard score: {key}")
    require(abs(measured - score_audit["scores"][key]["test_accuracy"]) < 1e-6,
            f"Frozen score: {key}")
    return predicted, meta["hard_sha256"]


def calculate() -> dict:
    rows = []
    summaries = []
    inputs = []
    for study, graphs in STUDIES.items():
        stage = ROOT / study
        lock_path = stage / "VALIDATION_SELECTION_LOCK.json"
        manifest_path = stage / "HARD_DECISION_EXPORT_MANIFEST.json"
        audit_path = stage / "FINAL_SCORE_AUDIT.json"
        lock = json.loads(lock_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        score_audit = json.loads(audit_path.read_text())
        require(manifest["selection_lock_sha256"] == sha(lock_path),
                f"Selection-lock hash: {study}")
        require(manifest["final_score_audit_sha256"] == sha(audit_path),
                f"Score-audit hash: {study}")
        inputs.append({"study": study, "validation_lock_sha256": sha(lock_path),
                       "hard_manifest_sha256": sha(manifest_path),
                       "final_score_audit_sha256": sha(audit_path)})
        for graph in graphs:
            reference = manifest["graph_references"][graph]
            refpath = stage / reference["reference_path"]
            require(sha(refpath) == reference["reference_sha256"],
                    f"Reference hash: {study}/{graph}")
            with np.load(refpath, allow_pickle=False) as data:
                indices = data["test_indices"].copy()
                labels = data["full_labels"][indices].copy()
            require(len(labels) == reference["test_nodes"] and
                    len(set(indices.tolist())) == len(indices),
                    f"Test reference: {study}/{graph}")
            arm = lock["partial_family_choice"][graph]["selected_arm"]
            require(arm in ("private_first", "private_last"),
                    f"Partial-family arm: {study}/{graph}")
            selected = lock["selections"][graph]
            partial_candidate = selected[arm]["selected_candidate"]
            ens_candidate = selected["ens"]["selected_candidate"]
            graph_rows = []
            for seed in (0, 1, 2):
                pkey = cell_key(graph, arm, partial_candidate, seed)
                ekey = cell_key(graph, "ens", ens_candidate, seed)
                require(pkey in score_audit["scores"] and
                        ekey in score_audit["scores"],
                        f"Final selected scores: {study}/{graph}/seed{seed}")
                partial, phash = classes(stage, manifest, score_audit, pkey,
                                          labels, reference["classes"])
                ens, ehash = classes(stage, manifest, score_audit, ekey,
                                      labels, reference["classes"])
                pc, ec = partial == labels, ens == labels
                both = int(np.sum(pc & ec))
                p_only = int(np.sum(pc & ~ec))
                e_only = int(np.sum(~pc & ec))
                neither = int(np.sum(~pc & ~ec))
                n = len(labels)
                require(both + p_only + e_only + neither == n,
                        f"Paired partition: {study}/{graph}/seed{seed}")
                row = {"study": study, "graph": graph, "seed": seed,
                       "partial_arm": arm, "partial_candidate": partial_candidate,
                       "ens_candidate": ens_candidate, "N": n,
                       "both_correct": both, "partial_only_correct": p_only,
                       "ens_only_correct": e_only, "neither_correct": neither,
                       "partial_accuracy": (both + p_only) / n,
                       "ens_accuracy": (both + e_only) / n,
                       "delta_accuracy": (p_only - e_only) / n,
                       "partial_cell": pkey, "ens_cell": ekey,
                       "partial_hard_sha256": phash, "ens_hard_sha256": ehash,
                       "test_reference_sha256": reference["reference_sha256"]}
                graph_rows.append(row)
                rows.append(row)
            total = {field: sum(row[field] for row in graph_rows)
                     for field in ("N", "both_correct", "partial_only_correct",
                                   "ens_only_correct", "neither_correct")}
            summaries.append({"study": study, "graph": graph, "partial_arm": arm,
                              **total,
                              "mean_accuracy_delta_pp": 100 *
                              (total["partial_only_correct"] -
                               total["ens_only_correct"]) / total["N"],
                              "seed_delta_pp": [100 * row["delta_accuracy"]
                                                for row in graph_rows]})
    require(len(rows) == 24 and len(summaries) == 8,
            "Expected 24 seed pairs and 8 settings")
    return {"description": "Selected partial family versus validation-selected ENS; descriptive paired test decisions",
            "scope": "Each row shares one test-node set within an optimizer seed. The same nodes recur across three seeds; raw/normalized Cora and the primary/raw repeat are related settings. No independent-node confidence intervals or p-values are implied.",
            "inputs": inputs, "seed_rows": rows, "setting_summaries": summaries}


def main() -> None:
    require(len(sys.argv) == 1 or sys.argv == [sys.argv[0], "--check"],
            "Usage: python analyze_paired_decisions.py [--check]")
    report = calculate()
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    fields = ("study", "graph", "seed", "partial_arm", "N", "both_correct",
              "partial_only_correct", "ens_only_correct", "neither_correct",
              "partial_accuracy", "ens_accuracy", "delta_accuracy",
              "partial_cell", "ens_cell", "partial_hard_sha256",
              "ens_hard_sha256", "test_reference_sha256")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: row[k] for k in fields} for row in report["seed_rows"])
    csv_text = stream.getvalue()
    for name, content in (("PAIRED_DECISIONS.json", json_text),
                          ("PAIRED_DECISIONS.csv", csv_text)):
        path = HERE / name
        if "--check" in sys.argv:
            require(path.read_bytes() == content.encode(),
                    f"Packaged output mismatch: {name}")
        else:
            path.write_text(content)
    print(json.dumps({"status": "PASS", "settings": 8,
                      "paired_seed_rows": 24}, sort_keys=True))


if __name__ == "__main__":
    main()
