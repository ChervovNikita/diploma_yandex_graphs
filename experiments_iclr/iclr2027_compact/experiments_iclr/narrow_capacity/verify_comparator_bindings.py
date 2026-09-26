"""Recompute the eight narrow-versus-TIED rows from audited public score records."""
from __future__ import annotations

import hashlib
import json
import statistics
import tarfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
DEFAULT = (0.001, 0.0)


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def score_key(graph, arm, candidate, seed):
    lr, wd = candidate
    return f"{graph}/{arm}/lr{lr:g}_wd{wd:g}/seed{seed}"


def main():
    summary = read(HERE / "narrow72/CAPACITY_SENSITIVITY_COMPARISON.json")
    narrow_manifest = read(HERE / "MANIFEST.json")
    narrow_freeze = read(HERE / "narrow72/FROZEN_NARROW_UNTIED72_STUDY.json")
    narrow_lock = read(HERE / "narrow72/VALIDATION_SELECTION_LOCK.json")
    narrow_audit = read(HERE / "narrow72/FINAL_SCORE_AUDIT.json")
    scan_path = HERE / "narrow72/NARROW_UNTIED72_WIDTH_SCAN.json"
    scan = read(scan_path)
    primary_root = EXPERIMENTS / "validation_tuning"
    primary_lock = read(primary_root / "VALIDATION_SELECTION_LOCK.json")
    primary_audit_path = primary_root / "FINAL_SCORE_AUDIT.json"
    primary_audit = read(primary_audit_path)
    tied_root = EXPERIMENTS / "factor_placement/tied36"
    tied_lock = read(tied_root / "VALIDATION_SELECTION_LOCK.json")
    tied_audit_path = tied_root / "FINAL_SCORE_AUDIT.json"
    tied_audit = read(tied_audit_path)
    factor_manifest = read(EXPERIMENTS / "factor_placement/MANIFEST.json")
    need(sha(scan_path) == narrow_freeze["width_scan_sha256"] ==
         narrow_lock["width_scan_sha256"] and
         sha(primary_root / "VALIDATION_SELECTION_LOCK.json") ==
         narrow_freeze["original_validation_lock_sha256"] ==
         narrow_lock["original_validation_lock_sha256"] and
         sha(tied_root / "VALIDATION_SELECTION_LOCK.json") ==
         narrow_freeze["same_runtime_tied36_validation_lock_sha256"] ==
         narrow_lock["same_runtime_tied36_validation_lock_sha256"] and
         sha(primary_audit_path) == summary["original_final_score_audit_sha256"] and
         sha(tied_audit_path) == summary["same_runtime_tied36_final_score_audit_sha256"] ==
         factor_manifest["tied36_final_audit_sha256"] and
         sha(HERE / "narrow72/FINAL_SCORE_AUDIT.json") ==
         summary["narrow_final_score_audit_sha256"],
         "Width scan or comparator lock/audit binding differs")
    archives = {}
    for label, path in (("narrow", HERE / "CELL_RECORDS.tar.xz"),
                        ("tied", EXPERIMENTS / "factor_placement/CELL_RECORDS.tar.xz")):
        with tarfile.open(path, "r:xz") as archive:
            archives[label] = {entry.name: archive.extractfile(entry).read()
                               for entry in archive if entry.isfile() and
                               "/scores/" in entry.name and entry.name.endswith("/score.json")}
    checked = []
    for row in summary["rows"]:
        graph, setting = row["dataset"], row["setting"]
        need(setting in ("selected", "default"), "Unknown setting")
        width = scan["graphs"][graph]
        need(row["width"] == width["selected_width"] and
             row["narrow_parameter_count"] == width["selected_untied_parameters"] and
             row["tied128_parameter_count"] == width["tied128_parameters"],
             f"Width/parameter count differs: {graph}")
        narrow_candidate = (tuple(narrow_lock["selections"][graph]["selected_candidate"])
                            if setting == "selected" else DEFAULT)
        if graph in ("cora", "wikics"):
            comparator = "same_runtime_tied36"
            tied_candidate = (tuple(tied_lock["selections"][graph]["selected_candidate"])
                              if setting == "selected" else DEFAULT)
        else:
            comparator = "original_same_runtime_tied"
            tied_candidate = (tuple(primary_lock["selections"][graph]["tied"]["selected_candidate"])
                              if setting == "selected" else DEFAULT)
        need(row["comparator"] == comparator and
             tuple(row["narrow_candidate"]) == narrow_candidate and
             tuple(row["tied_candidate"]) == tied_candidate,
             f"Comparator/candidate binding differs: {graph}/{setting}")
        narrow_values, tied_values = [], []
        for seed in (0, 1, 2):
            key = score_key(graph, "untied", narrow_candidate, seed)
            name = f"narrow72/scores/{key}/score.json"
            data = archives["narrow"][name]
            need(hashlib.sha256(data).hexdigest() == narrow_audit["scores"][key]["score_sha256"] and
                 narrow_manifest["scores"]["narrow72/" + key]["test_accuracy"] ==
                 json.loads(data)["test_accuracy"], f"Narrow audited score differs: {key}")
            narrow_values.append(json.loads(data)["test_accuracy"])
            key = score_key(graph, "tied", tied_candidate, seed)
            if graph in ("cora", "wikics"):
                name = f"tied36/scores/{key}/score.json"
                data = archives["tied"][name]
                audited = tied_audit["scores"][key]
                projected = factor_manifest["test_scores"]["tied36/" + key]
                need(projected["source_score_sha256"] == audited["score_sha256"] and
                     projected["test_accuracy"] == json.loads(data)["test_accuracy"],
                     f"TIED36 compact binding differs: {key}")
            else:
                data = (primary_root / "scores" / key / "score.json").read_bytes()
                audited = primary_audit["scores"][key]
            need(hashlib.sha256(data).hexdigest() == audited["score_sha256"],
                 f"TIED audited score differs: {key}")
            tied_values.append(json.loads(data)["test_accuracy"])
        differences = [a-b for a,b in zip(narrow_values, tied_values)]
        for field, value in (("narrow_seed_accuracy", narrow_values),
                             ("tied_seed_accuracy", tied_values)):
            need(row[field] == value, f"Stored seed scores differ: {graph}/{setting}/{field}")
        for field, value in (("narrow_mean", statistics.mean(narrow_values)),
                             ("tied_mean", statistics.mean(tied_values)),
                             ("paired_difference_mean", statistics.mean(differences)),
                             ("paired_difference_sample_sd", statistics.stdev(differences))):
            need(abs(row[field] - value) <= 1e-12,
                 f"Stored summary arithmetic differs: {graph}/{setting}/{field}")
        checked.append(f"{graph}/{setting}")
    need(len(checked) == 8 and len(set(checked)) == 8,
         "Eight-row comparator coverage differs")
    print(json.dumps({"status": "PASS", "rows": checked,
                      "narrow_summary_sha256": sha(HERE / "narrow72/CAPACITY_SENSITIVITY_COMPARISON.json")}))


if __name__ == "__main__":
    main()
