#!/usr/bin/env python3
"""Literature/source custody and JSON checks only; no ML runtime or fitting."""
import ast
from hashlib import sha256
import json
from pathlib import Path


def check_pin(path, row):
    assert path.stat().st_size == row.get("bytes", row.get("size")), str(path)
    assert sha256(path.read_bytes()).hexdigest() == row["sha256"], str(path)


def verify_manifest(root):
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        assert path.is_relative_to(root)
        check_pin(path, row)
    return len(manifest["files"])


def run():
    here = Path(__file__).resolve().parent
    research = here.parent
    parsed = []
    for path in sorted(here.rglob("*.json")):
        json.loads(path.read_text())
        parsed.append(str(path.relative_to(here)))
    source = (here / "static_check.py").read_text()
    ast.parse(source)
    compile(source, str(here / "static_check.py"), "exec")
    bindings = json.loads((here / "INPUT_BINDINGS.json").read_text())
    for row in bindings["files"]:
        path = (research / row["relative_path"]).resolve()
        assert path.is_relative_to(research)
        check_pin(path, row)
    reused = json.loads((here / "REUSED_CONCLUSIONS.json").read_text())
    papers = json.loads((here / "PAPER_CONCLUSIONS.json").read_text())
    scope = json.loads((here / "READ_SCOPES.json").read_text())
    assert len(reused["entries"]) == reused["records_reused"] == 16
    assert len(papers["reused_conclusion_assessments"]) == 16
    assert len(papers["new_primary_conclusions"]) == scope["new_scoped_primary_method_identity_reads"] == 1
    assert scope["new_full_paper_certifications"] == 0
    assert all(row["title"] and row["new_primary_read_in_this_packet"] is False
               for row in [dict(entry, title=papers["reused_conclusion_assessments"][i]["title"])
                           for i,entry in enumerate(reused["entries"])])
    for primary in scope["new_primary"]:
        for row in [primary["primary"],primary["mechanical_extraction"],*primary["visual_files"]]:
            check_pin(here / row["path"], row)
        assert primary["semantic_pdf_pages_read"] == primary["visual_pages_inspected"] == [1,2,3,4]
        assert primary["mechanical_pages_extracted"] == 18
    prior = research / "pooled_joint_native_empty_kernel_diagnostic_preparation_20261004_v1"
    assert sha256((prior / "MANIFEST.json").read_bytes()).hexdigest() == "38456c635b6e434b54882f3d6b0aab4394be4b5a7674bb2362ea121de2fac6b7"
    diagnostic_payloads = verify_manifest(prior)
    literature_payloads = verify_manifest(research / "literature_memory/index_v43")
    return {"status":"PASS_LITERATURE_SOURCE_CUSTODY_ONLY","JSON_files_parsed":parsed,
            "own_static_checker_AST_compile":True,"input_pins_verified":len(bindings["files"]),
            "reused_conclusions":16,"new_scoped_primary_method_reads":1,
            "new_full_paper_certifications":0,"prior_diagnostic_manifest_unchanged":True,
            "prior_diagnostic_payload_entries_verified":diagnostic_payloads,
            "index_v43_payload_entries_verified":literature_payloads,
            "primary_pdf_extraction_and_four_visual_page_pins_verified":True,
            "algebra_reviewed_symbolically_only":True,"numerical_or_native_or_GPU_execution":False,
            "project_model_scores_dataset_arrays_checkpoints_read":False,
            "scientific_fit_admitted":False,"novelty_or_predictive_utility_established":False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
