"""Seal new source closure only. Emits no root/runtime/scientific approval."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    bindings, manifest = HERE / "SOURCE_BINDINGS.json", HERE / "SOURCE_MANIFEST_v1.json"
    if bindings.exists() or manifest.exists():
        raise ValueError("Preserve sealed packets; create a new explicit generation")
    static_path = HERE / "STATIC_SOURCE_CHECK_v1.json"
    static = json.loads(static_path.read_text())
    if static.get("status") != "source_checks_passed_no_numerical_execution":
        raise ValueError("Required stdlib source checks missing")
    for name, expected in static["python_source_sha256"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Source changed after static checks")
    closure = {}
    for packet, manifest_name, expected in (
        ("shared_message_quality_v3", "SOURCE_MANIFEST_v1.json", "607c487f83696a21d5792c3f22b408491f9fb55c8f7e0b1dd1527b8f68653dcc"),
        ("propagation_predictive_v1", "SOURCE_MANIFEST_v1.json", "38030c08357a28e2e24959317562fce92208607092ed785c314a1a2db9821a35"),
    ):
        directory = PHASE / packet
        upstream = directory / manifest_name
        if sha(upstream) != expected:
            raise ValueError("Exact preserved dependency packet differs")
        record = json.loads(upstream.read_text())
        closure[str(upstream.relative_to(PHASE))] = expected
        for row in record["files"]:
            path = directory / row["relative_path"]
            if path.is_symlink() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("Preserved upstream source changed")
            closure[str(path.relative_to(PHASE))] = row["sha256"]
        upstream_binding = directory / "SOURCE_BINDINGS.json"
        for relative, expected_sha in json.loads(upstream_binding.read_text())["sha256"].items():
            path = PHASE / relative
            if path.is_symlink() or sha(path) != expected_sha:
                raise ValueError("Dependency closure bytes differ")
            closure[relative] = expected_sha
    closure["protocols/bounded_run_v1.py"] = sha(PHASE / "protocols/bounded_run_v1.py")
    if closure["protocols/bounded_run_v1.py"] != "b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23":
        raise ValueError("Exact independently audited whole-process deadline source changed")
    generated_path = HERE / "GENERATED_MODULE_POLICY_v1.json"
    if sha(generated_path) != sha(PHASE / "shared_message_quality_v3/GENERATED_MODULE_POLICY_v1.json"):
        raise ValueError("Scientific guard must use the exact captured quality-v3 generation policy")
    for relative in ("shared_message_study_design_v1/PLAN_ROOT_before_validation_v2.json",
                     "shared_message_study_design_v1/RNG_MEMBER_ORDER_ROOT_before_validation_v1.json"):
        closure[relative] = sha(PHASE / relative)
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name not in (bindings.name, manifest.name):
            if path.is_symlink():
                raise ValueError("Source symlink forbidden")
            closure[str(path.relative_to(PHASE))] = sha(path)
    package_roots = [HERE.name, "shared_message_quality_v3", "propagation_cost_engineering_v1", "propagation_cost_impl_v1"]
    importable = {p: h for p, h in closure.items() if Path(p).parts[0] in package_roots and p.endswith(".py")}
    package_dirs = sorted(str(Path(p).parent) for p in importable if Path(p).name == "__init__.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("shared_message_scientific_seal_guard", HERE / "import_guard.py")
    guard = importlib.util.module_from_spec(spec)
    exec(compile((HERE / "import_guard.py").read_bytes(), str(HERE / "import_guard.py"), "exec", dont_inherit=True), guard.__dict__)
    trees = {packet: guard.inspect_tree(PHASE / packet, [p for p in importable if Path(p).parts[0] == packet], package_dirs)
             for packet in package_roots}
    policy_path = HERE / "RUNTIME_IMPORT_POLICY_v1.json"
    with policy_path.open("x") as stream:
        json.dump(dict(schema="gnnm-shared-message-predictive-exact-import-policy-v2", package_roots=package_roots,
            package_directories=package_dirs, importable_sha256=importable,
            generated_module_policy_sha256=sha(generated_path), exact_source_importable_tree=trees,
            read_or_write_bytecode=False, unrelated_phase_paths_on_sys_path=False,
            legacy_bridge_bound_but_not_required_by_scientific_entrypoints=True), stream, indent=2, sort_keys=True)
        stream.write("\n")
    closure[str(policy_path.relative_to(PHASE))] = sha(policy_path)
    with bindings.open("x") as stream:
        json.dump({"schema": "gnnm-shared-message-predictive-source-bindings-v2", "sha256": closure,
                   "actual_data_runtime_resource_admission_established": False,
                   "scientific_training_validation_export_or_cost_launch_authorized": False}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    files = [{"relative_path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p != manifest]
    with manifest.open("x") as stream:
        json.dump({"schema": "gnnm-shared-message-fixed30-scientific-source-manifest-v2", "files": files,
                   "status": "source_only_sealed_launch_and_validation_closed", "source_bindings_sha256": sha(bindings),
                   "static_check_sha256": sha(static_path), "dependency_closure_entries": len(closure),
                   "Torch_labels_datasets_GPU_remote_weights_or_prior_scores_accessed_by_author": False,
                   "source_model_numerical_verification_performed": False, "scientific_fits": 0,
                   "test_exporter_or_evaluator_present": False,
                   "older_source_protocol_seed_guards_or_thresholds_modified": False,
                   "novelty_acceptance_quality_or_convergence_established": False}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"manifest_sha256": sha(manifest), "bindings_sha256": sha(bindings), "closure_entries": len(closure), "launch_closed": True}))


if __name__ == "__main__":
    main()
