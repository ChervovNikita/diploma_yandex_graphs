#!/usr/bin/env python3
"""Stdlib static checks; no imported project/runtime/Torch or numerical work."""
import ast
from hashlib import sha256
import json
from pathlib import Path


def check_pin(path, row):
    assert path.stat().st_size == row.get("bytes", row.get("size")), str(path)
    assert sha256(path.read_bytes()).hexdigest() == row["sha256"], str(path)


def run():
    here = Path(__file__).resolve().parent
    research = here.parent
    parsed = []
    for path in sorted(here.glob("*.py")):
        source = path.read_text()
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
        parsed.append(path.name)
    for path in sorted(here.glob("*.json")):
        json.loads(path.read_text())
    bindings = json.loads((here / "SOURCE_BINDINGS.json").read_text())
    for row in bindings["files"]:
        path = (research / row["relative_path"]).resolve()
        assert path.is_relative_to(research)
        check_pin(path, row)
    original = json.loads((research / bindings["native_bindings_relative_path"]).read_text())
    for row in original["files"]:
        path = (research / row["relative_path"]).resolve()
        assert path.is_relative_to(research)
        check_pin(path, row)
    payloads = 0
    manifests = bindings["manifests"] + original["manifests"]
    for row in manifests:
        root = (research / row["packet"]).resolve()
        assert root.is_relative_to(research)
        manifest = root / "MANIFEST.json"
        assert sha256(manifest.read_bytes()).hexdigest() == row["sha256"]
        for entry in json.loads(manifest.read_text())["files"]:
            path = (root / entry["path"]).resolve()
            assert path.is_relative_to(root)
            check_pin(path, entry)
            payloads += 1
    plan = json.loads((here / "PLAN.json").read_text())
    release = json.loads((here / "ROOT_RELEASE.example.json").read_text())
    limits = json.loads((here / "LIMITS.json").read_text())
    diagnose = ast.parse((here / "diagnose.py").read_text())
    case_assignment = next(n for n in diagnose.body if isinstance(n, ast.Assign)
                           and any(isinstance(t,ast.Name) and t.id == "CASES" for t in n.targets))
    cases = ast.literal_eval(case_assignment.value)
    assert list(cases) == [row["case"] for row in plan["cases"]]
    assert list(cases) == [row["case"] for row in release["authorized_invocations"]]
    assert release["limits"] == limits and release["execution_enabled"] is False
    failure = json.loads((research / next(row["relative_path"] for row in bindings["files"]
                                         if row["key"] == "failed_receipt")).read_text())
    assert release["cuda_visible_devices"] == failure["runtime_identity"]["profile"]["CUDA_VISIBLE_DEVICES"]
    assert failure["status"] == "FAIL" and failure["data_files_opened"] == [] and failure["fits"] == 0
    return {"status":"PASS_STATIC_ONLY","AST_and_compile_files":parsed,
            "diagnostic_source_and_failure_pins_verified":len(bindings["files"]),
            "original_source_pins_verified":len(original["files"]),
            "dependency_manifests_verified":len(manifests),
            "dependency_manifest_payload_entries_verified":payloads,
            "disabled_release_cases_and_bounds_checked":True,
            "project_or_Torch_imported":False,"native_or_GPU_executed":False,
            "dataset_arrays_opened":False,"old_checkpoints_or_science_outcomes_opened":False,
            "kernel_cause_determined":False,"native_qualification_pass":False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
