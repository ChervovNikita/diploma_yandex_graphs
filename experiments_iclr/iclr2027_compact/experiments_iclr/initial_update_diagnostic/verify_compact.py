"""Check the complete compact first-update evidence without raw gradient arrays."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GRAPHS = ("cora", "wikics", "actor", "chameleon_filtered")
EXPECTED = {(graph, seed) for graph in GRAPHS for seed in range(3)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def read(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def main() -> None:
    compact = read("COMPACT_STAGE_MANIFEST.json")
    listed = compact["files_sha256"]
    actual = {path.name for path in ROOT.iterdir()
              if path.is_file() and path.name != "COMPACT_STAGE_MANIFEST.json"
              and not path.name.startswith(".")}
    require(actual == set(listed), "Compact stage has missing or unlisted files")
    for name, digest in listed.items():
        require(sha(ROOT / name) == digest, f"Compact file hash differs: {name}")
    require(not any(ROOT.rglob("*.npz")) and not compact["raw_arrays_included"],
            "Compact stage unexpectedly contains raw gradient arrays")

    freeze = read("INITIAL_UPDATE_DIAGNOSTIC_FREEZE.json")
    result = read("INITIAL_UPDATE_DIAGNOSTIC_RESULTS.json")
    audit = read("INITIAL_UPDATE_ARRAY_AUDIT.json")
    raw_manifest = read("initial_update_raw_manifest.json")
    provenance = read("initial_update_diagnostic.provenance.json")
    require(freeze["protocol"] == result["protocol"] ==
            "initial_adam_update_order_train_only_v1", "Diagnostic protocol differs")
    require(result["freeze_sha256"] == sha(ROOT / "INITIAL_UPDATE_DIAGNOSTIC_FREEZE.json") ==
            audit["original_freeze_sha256"] and
            audit["original_result_sha256"] ==
            sha(ROOT / "INITIAL_UPDATE_DIAGNOSTIC_RESULTS.json"),
            "Diagnostic freeze/result/audit link differs")
    require(freeze["base_freeze_sha256"] == sha(ROOT / "FROZEN_STUDY.json") and
            freeze["mechanism_freeze_sha256"] == sha(ROOT / "MECHANISM_FREEZE.json") and
            all(sha(ROOT / name) == digest
                for name, digest in freeze["source_sha256"].items()),
            "Frozen source or base identity differs")
    require(raw_manifest["original_diagnostic_freeze_sha256"] ==
            sha(ROOT / "INITIAL_UPDATE_DIAGNOSTIC_FREEZE.json") and
            raw_manifest["original_diagnostic_result_sha256"] ==
            sha(ROOT / "INITIAL_UPDATE_DIAGNOSTIC_RESULTS.json") and
            raw_manifest["source_script_sha256"] ==
            sha(ROOT / "export_initial_update_gradients.py") and
            audit["raw_export_manifest_sha256"] ==
            compact["raw_array_manifest_sha256"] ==
            sha(ROOT / "initial_update_raw_manifest.json"),
            "Raw export manifest link differs")
    require(audit["status"] == "PASS" and
            audit["auditor_sha256"] == sha(ROOT / "audit_initial_update_arrays.py") and
            audit["sensitivity_status"].startswith("thresholds selected post hoc"),
            "Independent audit identity differs")

    def identity(rows):
        return {(row["dataset"], int(row["seed"])) for row in rows}

    diagnostic_rows = result["rows"]
    audit_rows = audit["rows"]
    raw_rows = raw_manifest["rows"]
    require(len(diagnostic_rows) == len(audit_rows) == len(raw_rows) == 12 and
            identity(diagnostic_rows) == identity(audit_rows) ==
            identity(raw_rows) == EXPECTED, "Expected all 12 fixed graph/seed rows")
    result_by_key = {(row["dataset"], row["seed"]): row
                     for row in diagnostic_rows}
    raw_by_key = {(row["dataset"], row["seed"]): row for row in raw_rows}
    for row in audit_rows:
        key = row["dataset"], row["seed"]
        original, raw = result_by_key[key], raw_by_key[key]
        require(row["source_array_sha256"] == raw["sha256"] and
                raw["shape"] == [4, 165120] and
                row["actual_tied_formula_max_abs_error"] < 1e-5 and
                row["actual_sync_formula_max_abs_error"] < 1e-5 and
                abs(row["update_cosine"] - original["update_cosine"]) < 1e-10 and
                abs(row["sync_over_tied_l2_norm_ratio"] -
                    original["sync_over_tied_l2_norm_ratio"]) < 1e-10,
                f"Audit/result row differs: {key}")

    with (ROOT / "initial_update_diagnostic.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 12 and identity(csv_rows) == EXPECTED,
            "Diagnostic CSV matrix differs")
    for row in csv_rows:
        original = result_by_key[row["dataset"], int(row["seed"])]
        require(abs(float(row["update_cosine"]) -
                    original["update_cosine"]) < 1e-12 and
                abs(float(row["sync_over_tied_l2_norm_ratio"]) -
                    original["sync_over_tied_l2_norm_ratio"]) < 1e-12,
                "Diagnostic CSV/result value differs")

    require(provenance["source_sha256"] ==
            sha(ROOT / "plot_initial_update_diagnostic.py") and
            provenance["audit_sha256"] ==
            sha(ROOT / "INITIAL_UPDATE_ARRAY_AUDIT.json") and
            provenance["figure_sha256"] ==
            sha(ROOT / "initial_update_diagnostic.png") and
            provenance["selection"] == "all 12 frozen graph/seed rows, none excluded",
            "Figure provenance differs")
    print("PASS: compact file hashes, frozen 12-case matrix, diagnostic and "
          "audit links, CSV values, and figure provenance. Raw-array "
          "recalculation is outside this compact stage.")


if __name__ == "__main__":
    main()
