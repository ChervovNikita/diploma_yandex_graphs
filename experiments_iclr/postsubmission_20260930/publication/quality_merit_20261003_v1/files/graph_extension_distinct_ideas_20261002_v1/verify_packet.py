"""Integrity only. No model, dataset, scientific output or runtime is imported."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_record_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def check_hash(path: str, expected: str) -> None:
    target = resolve_record_path(path)
    assert target.is_file(), f"Missing bound file: {path}"
    assert digest(target) == expected, f"Bound hash changed: {path}"


def main() -> None:
    objects = {}
    for path in sorted(ROOT.rglob("*.json")):
        objects[str(path.relative_to(ROOT))] = json.loads(path.read_text())
    jsonl_rows = 0
    for path in sorted(ROOT.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                jsonl_rows += 1
                if row.get("path") and row.get("sha256"):
                    check_hash(row["path"], row["sha256"])

    # Whitelisted saved reports/indices and the root timing receipt only.
    # Do not recurse into inherited receipt references to original run files.
    bindings = objects["INPUT_BINDINGS.json"]
    for item in bindings["inputs"]:
        check_hash(item["path"], item["sha256"])
    assert bindings["read_accounting"]["new_scoped_primary_methods"] == 4
    assert bindings["read_accounting"]["new_full_primary_reads"] == 0
    assert bindings["read_accounting"]["retained_primary_rereads"] == 0

    for name in ("READ_SCOPES.json", "PAPER_CONCLUSIONS.json", "QUALITY_PRIMARY_CONCLUSIONS.json"):
        papers = objects[name]["papers"]
        assert len(papers) == (2 if name.startswith("QUALITY_") else 4)
        for paper in papers:
            check_hash(paper["primary_html_path"], paper["primary_html_sha256"])
            check_hash(paper["blocks_path"], paper["blocks_sha256"])
            blocks = json.loads(resolve_record_path(paper["blocks_path"]).read_text())
            for lo, hi in paper["read_scope"]["paragraph_blocks_zero_based_inclusive"]:
                assert 0 <= lo <= hi < len(blocks)

    for item in objects["QUALITY_PRIMARY_PASSAGES.json"]["sources"]:
        check_hash(item["source_path"], item["source_sha256"])
        identity = item["canonical_id"].split(":", 1)[1]
        blocks = objects[f"primary/{identity}_html_blocks.json"]
        for group in item["paragraph_groups"]:
            lo, hi = group["range"]
            assert group["blocks"] == blocks[lo:hi + 1]

    size = objects["LARGE_DATASET_SIZE_BINDINGS.json"]
    check_hash(size["source_snapshot_path"], size["source_snapshot_sha256"])
    for item in size["sections"]:
        check_hash(item["path"], item["sha256"])
    resource = objects["QUALITY_RESOURCE_PLAN.json"]["source_bound_inherited_timing"]
    check_hash(resource["path"], resource["sha256"])
    assert resource["underlying_run_files_opened"] is False

    scope = objects["SCOPE_RECEIPT.json"]
    assert scope["methods_adopted"] == scope["pilots_adopted"] == 0
    assert scope["conditional_support_amendments_pending_root_review"] == 1
    candidates = objects["CANDIDATE_SPEC.json"]
    assert candidates["promoted_GNNM_methods"] == candidates["promoted_pilots"] == 0
    assert len(candidates["sources"]["new_scoped_primary_papers"]) == 4

    manifest = ROOT / "MANIFEST.sha256"
    if manifest.exists():
        listed = {}
        for line in manifest.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            path = ROOT / relative
            assert path.is_file(), f"Missing manifest file: {relative}"
            assert digest(path) == expected, f"Manifest mismatch: {relative}"
            listed[relative] = expected
        actual = {
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and path.name not in {"MANIFEST.sha256", "SEAL.json"}
        }
        assert set(listed) == actual, "Manifest inventory mismatch"
        if "SEAL.json" in objects:
            assert objects["SEAL.json"]["manifest_sha256"] == digest(manifest)
            assert objects["SEAL.json"]["files_sealed"] == len(listed)

    print(json.dumps({"integrity": "PASS", "JSON_files": len(objects), "JSONL_rows": jsonl_rows,
                      "saved_input_bindings": len(bindings["inputs"]), "scoped_primary_methods": 4,
                      "new_full_paper_reads": 0, "adopted_methods_or_pilots": 0,
                      "pending_support_question": 1, "scientific_correctness_certified": False}, indent=2))


if __name__ == "__main__":
    main()
