"""One immutable source generation; execution and fits remain unadmitted."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bindings_path, manifest_path = HERE / "SOURCE_BINDINGS.json", HERE / "SOURCE_MANIFEST_v1.json"
    if bindings_path.exists() or manifest_path.exists():
        raise ValueError("Preserve sealed generation; use a new explicit packet for repairs")
    check = json.loads((HERE / "STDLIB_SOURCE_CHECK_v1.json").read_text())
    if check["status"] != "passed_stdlib_source_only":
        raise ValueError("Missing stdlib syntax/formula/source closure check")
    for name, wanted in check["python_source_sha256"].items():
        if sha(HERE / name) != wanted:
            raise ValueError("Source changed after stdlib checks")
    generated_path = HERE / "GENERATED_MODULE_POLICY_v1.json"
    generated = json.loads(generated_path.read_text())
    if generated.get("schema") != "gnnm-shared-message-pinned-PyG-generated-source-policy-v1" or not generated["generated_modules"]:
        raise ValueError("Exact actual captured PyG generation metadata required")
    captured = generated["capture_receipt"]
    if sha(PHASE / captured["path"]) != captured["sha256"]:
        raise ValueError("Root captured PyG metadata receipt changed")
    closure = {}
    closure[captured["path"]] = captured["sha256"]
    closure[generated["capture_root_audit"]["path"]] = generated["capture_root_audit"]["sha256"]
    for packet, wanted in (("propagation_cost_engineering_v1", "139cf316ff00bde3b1dad7adeb6154cdc660a322262e1ee72afdb6502943c96d"),
                           ("propagation_cost_impl_v1", "b9f76667aaf2479ee5cd8816877ebc650c99380189c4869545519e753fdf63b9")):
        path = PHASE / packet / "SOURCE_MANIFEST_local_run01.json"
        if sha(path) != wanted:
            raise ValueError("Upstream generation manifest changed")
        closure[str(path.relative_to(PHASE))] = wanted
        upstream = json.loads(path.read_text())
        for row in upstream["files"]:
            file = path.parent / row["relative_path"]
            if file.stat().st_size != row["bytes"] or sha(file) != row["sha256"]:
                raise ValueError("Upstream sealed source changed")
            closure[str(file.relative_to(PHASE))] = row["sha256"]
        for row in upstream.get("context_bindings", []):
            file = PHASE / row["relative_to_project"]
            if file.stat().st_size != row["bytes"] or sha(file) != row["sha256"]:
                raise ValueError("Inherited source context changed")
            closure[str(file.relative_to(PHASE))] = row["sha256"]
    for name in ("protocols/run_authorized_v2.py", "protocols/run_logged.py", "protocols/repo_env.sh",
                 "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json", "protocols/bounded_run_v1.py",
                 "propagation_cost_engineering_v1/engineering_entry_bridge_v2.py"):
        closure[name] = sha(PHASE / name)
    if closure["protocols/bounded_run_v1.py"] != "b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23":
        raise ValueError("Independent-audited external deadline generation changed")
    if closure["propagation_cost_engineering_v1/engineering_entry_bridge_v2.py"] != "2c0383a4af90381fdc1b8bacde758b5d02b85ae69b2f7054464eb832286bcf24":
        raise ValueError("Legitimate legacy bridge source changed")
    for file in HERE.iterdir():
        if file.is_file():
            if file.is_symlink():
                raise ValueError("Source symlink forbidden")
            closure[str(file.relative_to(PHASE))] = sha(file)
    package_roots = [HERE.name, "propagation_cost_engineering_v1", "propagation_cost_impl_v1"]
    importable = {p: h for p, h in closure.items() if Path(p).parts[0] in package_roots and p.endswith(".py")}
    package_dirs = sorted(str(Path(p).parent) for p in importable if Path(p).name == "__init__.py")
    # Execute only the stdlib guard, never model/source payload modules.
    import importlib.util
    spec = importlib.util.spec_from_file_location("shared_message_seal_guard", HERE / "import_guard.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    captured_report = guard.verify_generated_capture(generated)
    closure[generated["capture_outer_terminal"]["path"]] = generated["capture_outer_terminal"]["sha256"]
    for row in captured_report["installed_sources"] + captured_report["generated_sources"]:
        closure[row["copy_relative_to_phase"]] = row["sha256"]
    closure[captured_report["request"]["path"]] = captured_report["request"]["sha256"]
    captured_terminal = PHASE / generated["capture_outer_terminal"]["path"]
    capture_start = captured_terminal.parent / "START.json"
    closure[str(capture_start.relative_to(PHASE))] = captured_report["outer_START_sha256"]
    capture_supervisor = PHASE / captured_report["supervisor_relative_to_phase"]
    for file in capture_supervisor.iterdir():
        if file.name in ("authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log"):
            closure[str(file.relative_to(PHASE))] = sha(file)
    trees = {packet: guard.inspect_tree(PHASE / packet, [p for p in importable if Path(p).parts[0] == packet], package_dirs)
             for packet in package_roots}
    policy_path = HERE / "RUNTIME_IMPORT_POLICY_v1.json"
    with policy_path.open("x") as stream:
        json.dump(dict(schema="gnnm-shared-message-exact-import-policy-v1", package_roots=package_roots,
            package_directories=package_dirs, importable_sha256=importable,
            generated_module_policy_sha256=sha(generated_path),
            exact_source_importable_tree=trees, read_or_write_bytecode=False,
            unrelated_phase_paths_on_sys_path=False,
            legacy_bridge_bound_but_not_required_by_qualification=True), stream, indent=2, sort_keys=True)
        stream.write("\n")
    closure[str(policy_path.relative_to(PHASE))] = sha(policy_path)
    with bindings_path.open("x") as stream:
        json.dump(dict(schema="gnnm-shared-message-source-bindings-v3", sha256=closure,
            scientific_training_authorized=False, numerical_qualification_executed=False), stream, indent=2, sort_keys=True)
        stream.write("\n")
    files = [dict(relative_path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(HERE.iterdir()) if p.is_file()]
    with manifest_path.open("x") as stream:
        json.dump(dict(schema="gnnm-shared-message-source-manifest-v3", status="source_only_sealed_qualification_unexecuted",
            files=files, source_bindings_sha256=sha(bindings_path), stdlib_check_sha256=sha(HERE / "STDLIB_SOURCE_CHECK_v1.json"),
            Torch_GPU_provider_data_or_numerical_execution_by_author=False,
            scientific_fits=0, heldout_labels_accessed=False, existing_scores_modified=False,
            original_thirty_fit_protocol_or_thresholds_adopted=False,
            scientific_protocol_and_runtime_resource_admission_required=True,
            novelty_or_acceptance_established=False), stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(manifest_sha256=sha(manifest_path), source_bindings_sha256=sha(bindings_path), qualification_unexecuted=True)))


if __name__ == "__main__":
    main()
