"""Audit exact disabled load_plan with scientific imports and omitted files denied."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    manifest = json.loads((root / args.manifest).read_text())
    plan = json.loads((root / manifest["plan"]["path"]).read_text())
    allowed = {row["path"] for row in manifest["files"]}
    excluded = {row["phase_relative_mirror_receipt"]["path"] for row in
                manifest["provider_provenance_receipts_retained_without_optional_copy"]
                if not row["required_by_other_source_or_runtime_binding"]}
    assert allowed.isdisjoint(excluded)
    forbidden_imports = {"torch", "numpy", "scipy", "sklearn", "torch_geometric",
                         "torch_scatter", "torch_sparse", "ogb", "pickle", "_pickle",
                         "cloudpickle", "joblib", "dill", "h5py", "pandas"}
    assert not (forbidden_imports & {name.split(".")[0] for name in sys.modules})
    opened, parsed, imported, absent_bytecode = Counter(), Counter(), set(), set()
    enabled = {"value": True}
    original_loads = json.loads

    def audited_loads(raw, *positional, **keywords):
        value = original_loads(raw, *positional, **keywords)
        if enabled["value"]:
            if isinstance(value, dict) and "studies" in value and "groups" in value:
                parsed["master_plan"] += 1
            elif isinstance(value, dict) and "study_id" in value and "runs" in value:
                parsed["study_spec"] += 1
            else:
                raise AssertionError("load_plan decoded unexpected JSON input")
        return value

    def audit(event, arguments):
        if not enabled["value"]:
            return
        if event == "import":
            name = arguments[0]
            imported.add(name)
            if name.split(".")[0] in forbidden_imports:
                raise AssertionError("Scientific/tensor deserializer import denied: " + name)
        if event == "open":
            name, mode, flags = arguments
            if isinstance(name, (str, bytes)):
                text = os.fsdecode(name)
                if not os.path.isabs(text):
                    text = os.path.abspath(text)
                try:
                    relative = Path(text).relative_to(root).as_posix()
                except ValueError:
                    return  # Interpreter/stdlib files lie outside the research phase.
                if "__pycache__" in Path(relative).parts and relative.endswith(".pyc") and not Path(text).exists():
                    absent_bytecode.add(relative)
                    return  # Missing import cache: no bytes were read or written.
                if relative in excluded:
                    raise AssertionError("Optional provider cache access denied: " + relative)
                if relative not in allowed:
                    raise AssertionError("Undeclared phase input denied: " + relative)
                if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                        flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                    raise AssertionError("Phase writes denied during load_plan")
                opened[relative] += 1
        if event in {"subprocess.Popen", "os.system", "os.exec", "socket.connect"}:
            raise AssertionError("Process/network operation denied during load_plan")

    json.loads = audited_loads
    sys.addaudithook(audit)
    sys.path.insert(0, str(root / plan["runtime_package"]))
    from launcher import load_plan
    assert Path(sys.modules["launcher"].__file__).resolve() == root / plan["runtime_launcher"]
    started = time.monotonic()
    actual, digest, studies, runs, placements, dependencies = load_plan(root, root / manifest["plan"]["path"])
    elapsed = time.monotonic() - started
    enabled["value"] = False
    json.loads = original_loads
    assert actual == plan and digest == manifest["plan_hash"]
    assert len(studies) == 15 and len(runs) == 26770
    assert set(runs) == set(placements) == set(dependencies)
    required_sources = {row["path"] for study in studies.values() for row in study["spec"]["source"]["files"]}
    assert required_sources <= set(opened)
    assert parsed == {"master_plan": 1, "study_spec": 15}
    assert not (forbidden_imports & {name.split(".")[0] for name in sys.modules})
    assert set(opened) <= allowed and set(opened).isdisjoint(excluded)
    assert all(placements[parent]["worker"].split("/")[0] == placements[key]["worker"].split("/")[0]
               for key, parents in dependencies.items() for parent in parents)
    model_groups = sum(row["run"]["kind"] == "model_fit" for row in runs.values())
    assert model_groups == 11581
    result = {"schema": "actual-disabled-launcher-source-access-audit-v1", "plan_hash": digest,
              "actual_launcher_path": plan["runtime_launcher"],
              "actual_launcher_sha256": hashlib.sha256((root / plan["runtime_launcher"]).read_bytes()).hexdigest(),
              "loaded_studies": len(studies), "registered_records": len(runs), "model_groups": model_groups,
              "wall_seconds": elapsed, "JSON_metadata_decoded": dict(parsed),
              "phase_file_open_counts": dict(sorted(opened.items())),
              "absent_interpreter_bytecode_lookups": sorted(absent_bytecode),
              "required_source_files_opened": len(required_sources),
              "source_hash_validation_executed_by_actual_controller": True,
              "all_phase_reads_within_manifest_union": True,
              "optional_provider_paths_denied_and_unused": sorted(excluded),
              "optional_provider_cache_necessary": False,
              "scientific_and_tensor_deserializer_imports_denied": sorted(forbidden_imports),
              "scientific_modules_loaded": [], "phase_writes_or_child_processes": 0,
              "tensor_arrays_or_targets_deserialized": 0, "CUDA_initialized": False,
              "scientific_fits": 0, "GO_created": False}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
