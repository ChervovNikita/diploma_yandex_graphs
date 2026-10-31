"""Audit exact disabled launcher loading; arrays, processes and undeclared reads denied."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

from sync_common import allocation_guard, bound_master, fingerprint, read_json, safe, verify_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--master-hash", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--output", required=True, help="Fresh project-relative JSON audit receipt")
    args = parser.parse_args()
    allocation_guard()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("Preparation requires hidden CUDA and bytecode writes disabled")
    root = Path(args.project_root).absolute()
    manifest, manifest_receipt = read_json(root, args.manifest, sha256=args.manifest_sha256)
    plan, summary, plan_receipt, summary_receipt = bound_master(
        root, args.master, args.master_hash, args.summary, args.summary_sha256)
    if manifest["plan"] != plan_receipt or manifest["plan_hash"] != args.master_hash or manifest["exact_summary_binding"] != summary_receipt:
        raise ValueError("Manifest differs from explicit final master/summary bindings")
    allowed = {row["path"] for row in manifest["files"]}
    if len(allowed) != len(manifest["files"]):
        raise ValueError("Repeated manifest path")
    excluded = {row["phase_relative_mirror_receipt"]["path"] for row in
                manifest.get("provider_provenance_receipts_retained_without_optional_copy", [])
                if not row["required_by_other_source_or_runtime_binding"]}
    if not allowed.isdisjoint(excluded):
        raise ValueError("Manifest optional provider union differs")
    forbidden = {"torch", "numpy", "scipy", "sklearn", "torch_geometric", "torch_scatter", "torch_sparse",
                 "torch_cluster", "torch_spline_conv", "torchdiffeq", "torch_householder", "torch_householder_cpp",
                 "torchdata", "dgl", "ogb", "pickle", "_pickle", "cloudpickle", "joblib", "dill", "h5py", "pandas"}
    if forbidden & {name.split(".")[0] for name in sys.modules}:
        raise ValueError("Scientific module imported before metadata audit")
    opened, parsed, imported, bytecode = Counter(), Counter(), set(), set()
    enabled, strict_json = {"value": True}, {"value": True}
    original_loads = json.loads
    def loads(raw, *positional, **keywords):
        value = original_loads(raw, *positional, **keywords)
        if strict_json["value"]:
            if isinstance(value, dict) and "studies" in value and "groups" in value:
                parsed["master_plan"] += 1
            elif isinstance(value, dict) and "study_id" in value and "runs" in value:
                parsed["study_spec"] += 1
            else:
                raise AssertionError("Unexpected JSON decoded by actual load_plan")
        return value
    def audit(event, arguments):
        if not enabled["value"]:
            return
        if event == "import":
            name = arguments[0]
            imported.add(name)
            if name.split(".")[0] in forbidden:
                raise AssertionError("Scientific/tensor provider import denied: " + name)
        if event == "open":
            name, mode, flags = arguments
            if isinstance(name, (str, bytes)):
                absolute = Path(os.path.abspath(os.fsdecode(name)))
                try:
                    relative = absolute.relative_to(root).as_posix()
                except ValueError:
                    return  # CPython and stdlib files are outside the phase.
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
                if writing:
                    raise AssertionError("Phase writes denied during load_plan")
                if "__pycache__" in absolute.parts and absolute.suffix == ".pyc" and not absolute.exists():
                    bytecode.add(relative)
                    return
                if relative in excluded:
                    raise AssertionError("Omitted provenance cache accessed: " + relative)
                if relative not in allowed:
                    raise AssertionError("Undeclared phase input accessed: " + relative)
                opened[relative] += 1
        if event in {"subprocess.Popen", "os.system", "os.exec", "socket.connect"}:
            raise AssertionError("Process/network operation denied during metadata load")
    json.loads = loads
    sys.addaudithook(audit)
    controller = safe(root, plan["runtime_package"], file=False)
    sys.path.insert(0, str(controller))
    launcher = importlib.import_module("launcher")
    if Path(launcher.__file__).resolve() != (root / plan["runtime_launcher"]).resolve():
        raise ValueError("Wrong launcher imported")
    started = time.monotonic()
    actual, digest, studies, runs, placements, dependencies = launcher.load_plan(root, root / args.master)
    seconds = time.monotonic() - started
    strict_json["value"] = False
    json.loads = original_loads
    compiler = importlib.import_module("compile_master_plan")
    counts = verify_counts(plan, summary, [row["spec"] for row in studies.values()], compiler)
    if actual != plan or digest != args.master_hash or counts != manifest["expected_counts"]:
        raise ValueError("Actual launcher/typed counts differ from explicit binding")
    if set(runs) != set(placements) or set(runs) != set(dependencies):
        raise ValueError("Incomplete placement/dependency union")
    if parsed != {"master_plan": 1, "study_spec": len(plan["studies"])}:
        raise ValueError("Metadata JSON parse counts differ")
    required_sources = {row["path"] for study in studies.values() for row in study["spec"]["source"]["files"]}
    if not required_sources <= set(opened):
        raise ValueError("Actual controller did not read every bound study source")
    if forbidden & {name.split(".")[0] for name in sys.modules}:
        raise ValueError("Scientific modules loaded")
    if not set(opened) <= allowed or not set(opened).isdisjoint(excluded):
        raise ValueError("Read union differs")
    if not all(placements[p]["worker"].split("/")[0] == placements[k]["worker"].split("/")[0]
               for k, parents in dependencies.items() for p in parents):
        raise ValueError("Cross-host dependency")
    enabled["value"] = False
    result = {"schema": "generic-final-master-source-loading-audit-v1", "plan_hash": digest,
              "manifest": manifest_receipt, "master": plan_receipt, "summary": summary_receipt,
              "actual_launcher_path": plan["runtime_launcher"],
              "actual_launcher_sha256": hashlib.sha256((root / plan["runtime_launcher"]).read_bytes()).hexdigest(),
              "expected_counts": counts, "loaded_studies": len(studies), "registered_records": len(runs),
              "wall_seconds": seconds, "JSON_metadata_decoded": dict(parsed),
              "phase_file_open_counts": dict(sorted(opened.items())), "required_source_files_opened": len(required_sources),
              "absent_interpreter_bytecode_lookups": sorted(bytecode),
              "actual_controller_source_hash_validation": True, "all_phase_reads_within_manifest_union": True,
              "optional_provider_paths_denied_and_unused": sorted(excluded),
              "scientific_and_tensor_deserializer_imports_denied": sorted(forbidden),
              "phase_writes_or_child_processes": 0, "tensor_arrays_or_targets_deserialized": 0,
              "CUDA_initialized": False, "scientific_fits": 0, "GO_created": False}
    from sync_common import fresh
    fresh(safe(root, args.output, file=False), result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"expected_counts", "phase_file_open_counts", "absent_interpreter_bytecode_lookups"}}, sort_keys=True))


if __name__ == "__main__":
    main()
