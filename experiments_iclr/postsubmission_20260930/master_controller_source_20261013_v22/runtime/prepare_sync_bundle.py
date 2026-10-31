"""Plan-bound portable artifact manifest; opaque byte hashing, no science I/O.

No tensor loader, Git history, credentials, scientific outputs or remote
transport is used. Author absolute runtime bindings remain allocation-only.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import sqlite3
import ssl
import subprocess
import sys
import tarfile
import tempfile

ALLOCATION_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
ORIGINAL_FIVE = {"romanempire", "amazonratings", "minesweeper", "questions", "tolokers"}
FORBIDDEN_PARTS = {".git", ".ssh", "__pycache__", "execution", "logs", "checkpoints", "checkpoint", "quarantine", "history_dump"}
PACKAGES = ("torch", "numpy", "scipy", "scikit-learn", "torch-geometric", "torch-scatter", "torch-sparse", "ogb")
STDLIB_MODULES = ("fcntl", "sqlite3", "ssl", "gzip", "zlib", "tarfile", "subprocess")


def stdlib_capabilities():
    available = {}
    for module in STDLIB_MODULES:
        try: importlib.import_module(module); available[module] = True
        except (ImportError, OSError): available[module] = False
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE compatibility_probe (id INTEGER PRIMARY KEY, state TEXT)")
        connection.execute("INSERT INTO compatibility_probe VALUES (1, 'ready')")
        connection.commit()
        available["sqlite_transactions"] = connection.execute("SELECT state FROM compatibility_probe WHERE id=1").fetchone() == ("ready",)
    available["ssl_context"] = isinstance(ssl.create_default_context(), ssl.SSLContext)
    return available


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def safe(root, relative, *, file=True, forbid=True):
    root = Path(root).absolute()
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("Project root must not traverse symlinks")
    text = str(relative); rel = PurePosixPath(text)
    if (not text or rel.is_absolute() or ".." in rel.parts or "\\" in text
            or str(rel) != text or rel == PurePosixPath(".")):
        raise ValueError("Canonical confined POSIX relative path required")
    if forbid and (set(rel.parts) & FORBIDDEN_PARTS or rel.suffix.lower() in {".log", ".pem", ".key"}
            or rel.name in {"id_rsa", "id_ed25519", "APPROVAL.json", ".env", ".netrc"} or "private_key" in rel.name.lower()):
        raise ValueError("Excluded credential/history/output artifact: " + text)
    path = root.joinpath(*rel.parts)
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("Artifacts may not traverse symlinks: " + text)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes phase layout")
    if file and not path.is_file():
        raise ValueError("Missing required regular artifact: " + text)
    return path


def receipt(root, relative):
    path = safe(root, relative); digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(relative), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked_receipt(row):
    if (not isinstance(row, dict) or not {"path", "bytes", "sha256"} <= set(row)
            or type(row["bytes"]) is not int or row["bytes"] < 0
            or not isinstance(row["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
        raise ValueError("Exact path/bytes/SHA256 receipt required")
    return {key: row[key] for key in ("path", "bytes", "sha256")}


def nested_receipts(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            yield checked_receipt(value)
        else:
            for item in value.values():
                yield from nested_receipts(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_receipts(item)


def runtime_requirements():
    return {"schema": "sync-runtime-requirements-v1", "implementation": platform.python_implementation(),
        "python_version": platform.python_version(), "python_major_minor": list(sys.version_info[:2]),
        "platform_system": platform.system(), "machine": platform.machine(), "libc": list(platform.libc_ver()),
        "stdlib": {"sqlite": sqlite3.sqlite_version, "OpenSSL": ssl.OPENSSL_VERSION,
                   "required_modules": list(STDLIB_MODULES)}, "stdlib_capabilities": stdlib_capabilities(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "binary_import_modules": ["torch", "numpy", "scipy", "sklearn", "torch_geometric", "torch_scatter", "torch_sparse", "ogb"],
        "environment_recreation_required": True, "venv_and_system_libraries_bundled": False,
        "compiled_extension_ABI": "exact_Python_major_minor_and_Torch_PyG_scatter_sparse_build_versions",
        "CUDA_runtime": "Torch_wheel_build_and_host_driver_must_be_qualified;no_CUDA_pilot_in_preparation",
        "micro_and_system_library_versions_are_provenance_not_equality_gates": True,
        "numerical_equivalence_or_score_claim": False}


def compatibility_requirements(expected):
    if not all(package in expected.get("packages", {}) for package in PACKAGES):
        raise ValueError("Every mandatory numerical package lock must be declared")
    return {"implementation": "CPython", "python_major_minor": [3, 11], "platform_system": "Linux", "machine": "x86_64",
        "numerical_package_versions": {package: expected["packages"][package] for package in PACKAGES},
        "stdlib_capabilities": [*STDLIB_MODULES, "sqlite_transactions", "ssl_context"],
        "compiled_ABI": "CP311_Torch2.1_CUDA11.8_exact_Torch_PyG_scatter_sparse_build_pins",
        "native_binary_qualification": "CPU_import_and_native_scatter_sparse_operation_probe_without_CUDA_initialization",
        "provenance_only_version_fields": ["python_version", "libc", "stdlib.sqlite", "stdlib.OpenSSL"],
        "GPU_resource_and_driver_qualification": "separate_root_gate_before_scientific_GO"}


def read_json(root, relative):
    path = safe(root, relative)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if str(relative).endswith(".gz") else raw)


def manifest(root, plan_relative, environment, tooling_receipts=()):
    plan = read_json(root, plan_relative)
    if plan.get("scientific_launch_enabled") is not False or plan.get("human_GO_required") is not True:
        raise ValueError("Only disabled human-GO master plans may be synchronized")
    if not plan.get("runtime_package") or not plan.get("runtime_launcher"):
        raise ValueError("A consolidated frozen master controller is required")
    files, reasons, specs, author, empty, aliases = {}, {}, [], [], set(), {}
    def add(row, reason):
        row = checked_receipt(row)
        if row["path"] in files:
            if files[row["path"]] != row: raise ValueError("Conflicting union receipts: " + row["path"])
            reasons[row["path"]].add(reason); return
        safe(root, row["path"])
        files[row["path"]] = row; reasons.setdefault(row["path"], set()).add(reason)
    def add_path(path, reason):
        if path in files: reasons[path].add(reason)
        else: add(receipt(root, path), reason)
    add(receipt(root, plan_relative), "master_plan")
    for row in tooling_receipts: add(row, "synchronization_tooling")
    runtime = plan["runtime_package"]
    if plan["runtime_launcher"] != runtime + "/launcher.py":
        raise ValueError("Frozen launcher path differs from its runtime package")
    add(receipt(root, plan["runtime_launcher"]), "controller_entrypoint")
    add(receipt(root, runtime + "/start_massive.sh"), "existing_startup_requires_Git_checkout")
    add(receipt(root, "AGENTS.md"), "required_phase_marker")
    rendezvous = plan.get("phase_rendezvous", {})
    if (not rendezvous.get("remote_name") or not re.fullmatch("[0-9a-f]{64}", rendezvous.get("remote_url_sha256", ""))):
        raise ValueError("Plan-bound existing Git receipt remote digest required")
    run_keys, author_keys = set(), set(); assigned = {}
    for entry in plan["studies"]:
        value = read_json(root, entry["path"])
        if fingerprint(value) != entry["spec_hash"]:
            raise ValueError("Master study changed")
        normalize = lambda s: "".join(c for c in str(s).lower() if c.isalnum())
        dataset = value["dataset"]
        scope = value["data"].get("dataset_scope", [])
        if normalize(dataset) in ORIGINAL_FIVE or any(normalize(s) in ORIGINAL_FIVE for s in scope):
            raise ValueError("Original five paper datasets are hard excluded")
        add(receipt(root, entry["path"]), "master_study_spec")
        specs.append({"study_id": value["study_id"], "dataset": dataset, "path": entry["path"], "spec_hash": entry["spec_hash"],
                      "run_count": len(value["runs"])})
        for row in nested_receipts(value["source"]): add(row, "study_source_binding")
        evidence = value["data"].get("evidence", {})
        for row in nested_receipts(evidence.get("provider_files", []) if isinstance(evidence, dict) else []):
            if PurePosixPath(row["path"]).is_absolute():
                original = dict(row)
                try: relative = Path(row["path"]).relative_to(Path(root).absolute()).as_posix()
                except ValueError: raise ValueError("Absolute data provenance lies outside authorized phase; not accessed")
            else:
                original, relative = dict(row), row["path"]
            safe(root, relative, file=False)
            aliases[original["path"]] = {"original_provenance_receipt": original, "phase_relative_mirror_receipt": dict(row, path=relative),
                "spec_rewritten": False, "execution_uses_absolute_provenance_path": False,
                "provenance_only_optional_copy_excluded": True, "reason": "data.evidence.provider_files_is_metadata_not_loader_input"}
        required_data = {key: item for key, item in value["data"].items() if key != "evidence"}
        required_data["evidence"] = ({key: item for key, item in evidence.items() if key != "provider_files"}
                                     if isinstance(evidence, dict) else evidence)
        for row in nested_receipts(required_data): add(row, "required_study_data_receipt")
        for run in value["runs"]:
            key = value["study_id"] + "/" + run["run_id"]
            if key in run_keys: raise ValueError("Repeated study/run identity")
            run_keys.add(key)
            for row in nested_receipts(run.get("config", {})): add(row, "run_input_receipt")
            if run["kind"] != "author_cli_group":
                adapter = run["adapter"].split(":", 1)[0]
                add_path(adapter, "frozen_factory_entrypoint")
                # Some standardized contracts use relative strings for role files.
                for field in ("node_inputs", "endpoint_inputs"):
                    for val in run["config"].get(field, {}).values():
                        if isinstance(val, str): add_path(val, "relative_role_input")
                continue
            author_keys.add(key); cli = run["cli"]
            if "--dataset" in cli.get("argv", []):
                index = cli["argv"].index("--dataset")
                if index + 1 >= len(cli["argv"]) or normalize(cli["argv"][index + 1]) in ORIGINAL_FIVE:
                    raise ValueError("Original five author datasets are hard excluded")
            for row in cli["source_files"] + cli["raw_files"]: add(row, "author_all_host_hash_validation_only")
            empty.update((cli["basis_dir"], cli["processed_dir"]))
            for path in cli["quarantine"].values():
                safe(root, path, file=False, forbid=False)
                if (Path(root) / path).exists(): raise ValueError("Author quarantine already exists; never include or repeat")
            author.append({"run_key": key, "execution_host": "allocation", "absolute_execution_repository": cli["execution_repository"],
                "absolute_python_executable": cli["python_executable"], "absolute_environment": cli["environment"],
                "runtime_versions": cli["runtime_versions"], "private_dependency_dir": cli["private_dependency_dir"],
                "source_and_raw_files_must_exist_on_all_hosts_for_hash_validation": True,
                "execute_or_rewrite_author_contract_on_server77": False})
    for group in plan["groups"]:
        for key in group["run_keys"]:
            if key not in run_keys or key in assigned: raise ValueError("Unknown/repeated placement")
            assigned[key] = group["gpu_assignment"]
    if set(assigned) != run_keys: raise ValueError("Missing master placement")
    if any(not assigned[k].startswith("allocation/") for k in author_keys):
        raise ValueError("Absolute author CLI contract must remain allocated to allocation")
    for path in empty:
        directory = safe(root, path, file=False)
        if not directory.is_dir() or any(directory.iterdir()): raise ValueError("Author initial empty workdir changed")
    for row in files.values():
        if receipt(root, row["path"]) != row: raise ValueError("Required artifact bytes changed: " + row["path"])
    sizes = Counter()
    for path, row in files.items(): sizes[PurePosixPath(path).parts[0]] += row["bytes"]
    return {"schema": "plan-bound-sync-manifest-v1", "plan": files[plan_relative], "plan_hash": fingerprint(plan),
        "phase_relative_layout_preserved": True, "study_ids_and_run_ids_preserved": True,
        "files": [dict(row, reasons=sorted(reasons[path])) for path, row in sorted(files.items())],
        "empty_directories": sorted(empty), "studies": specs, "runtime_requirements": environment,
        "runtime_compatibility_policy": compatibility_requirements(environment),
        "allocation_only_author_contracts": author, "host_bindings": plan.get("host_bindings", {}),
        "provider_provenance_receipts_retained_without_optional_copy": [dict(aliases[name],
            required_by_other_source_or_runtime_binding=aliases[name]["phase_relative_mirror_receipt"]["path"] in files)
            for name in sorted(aliases)],
        "Git_overlay_contract": {"existing_or_later_authorized_cloned_checkout_required": True,
            "git_directory_bundled": False, "remote_name": rendezvous["remote_name"],
            "remote_url_sha256": rendezvous["remote_url_sha256"], "source_phase_relative_to_repository": "experiments_iclr/postsubmission_20260930",
            "artifact_layout_relative_to_destination_phase_must_match": True,
            "remote_URL_or_credentials_in_manifest": False},
        "statistics": {"files": len(files), "opaque_bytes": sum(r["bytes"] for r in files.values()),
                       "bytes_by_top_directory": dict(sorted(sizes.items())), "studies": len(specs), "registered_runs": len(run_keys),
                       "author_groups": len(author), "provider_provenance_receipts_retained": len(aliases),
                       "provenance_only_files_not_bundled": sum(a["phase_relative_mirror_receipt"]["path"] not in files for a in aliases.values()),
                       "provenance_only_bytes_not_bundled": sum(a["original_provenance_receipt"]["bytes"] for a in aliases.values()
                           if a["phase_relative_mirror_receipt"]["path"] not in files)},
        "all_study_bindings_needed_on_every_host": True, "tensor_deserialization_performed": False,
        "environment_ready_on_server77": False, "server77_access_authorized": False,
        "scientific_launch_enabled": False, "GO_created": False,
        "portability_requirements": ["destination_phase_relative_layout_exact", "exact_standardized_runtime_environment",
            "verified_plan_hostname_and_GPU_UUIDs", "all_manifest_hashes", "allocation_only_author_absolute_environment",
            "existing_Git_checkout_and_plan_bound_remote_digest_for_start_massive_and_phase_sidecar"],
        "time_estimates": "provisional_placeholder_fit_values_are_not_ETA;accurate_window_requires_later_resource_qualification"}


def verify_manifest(root, value):
    if value.get("schema") != "plan-bound-sync-manifest-v1": raise ValueError("Unsupported sync manifest")
    seen = set()
    for row in value["files"]:
        row = checked_receipt(row)
        if row["path"] in seen: raise ValueError("Duplicate manifest member")
        seen.add(row["path"])
        if receipt(root, row["path"]) != row: raise ValueError("Manifest artifact changed: " + row["path"])
    plan = read_json(root, value["plan"]["path"])
    if fingerprint(plan) != value["plan_hash"]: raise ValueError("Manifest plan hash differs")
    for row in value["studies"]:
        if fingerprint(read_json(root, row["path"])) != row["spec_hash"]: raise ValueError("Manifest spec changed")
    return plan


def make_bundle(root, value, path):
    verify_manifest(root, value)
    path = Path(path)
    if path.exists(): raise ValueError("Use a fresh immutable tar output")
    with tarfile.open(path, "x", format=tarfile.PAX_FORMAT) as archive:
        for row in value["files"]:
            source = safe(root, row["path"]); info = archive.gettarinfo(str(source), arcname=row["path"])
            if not info.isfile(): raise ValueError("Only regular files may enter bundle")
            info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
            info.mode = 0o555 if os.access(source, os.X_OK) else 0o444
            with source.open("rb") as stream: archive.addfile(info, stream)
        for relative in value["empty_directories"]:
            info = tarfile.TarInfo(relative + "/"); info.type = tarfile.DIRTYPE; info.mode = 0o755; archive.addfile(info)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    path.chmod(0o444)
    verify_bundle(value, path)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
            "archive_format": "tar_PAX_regular_files_and_declared_empty_dirs_only"}


def verify_bundle(value, path):
    expected = {r["path"]: checked_receipt(r) for r in value["files"]}
    empty = set(value["empty_directories"]); seen, directories = set(), set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if member.isdir():
                name = member.name.rstrip("/")
                if name not in empty or name in directories: raise ValueError("Unexpected/duplicate archive directory")
                directories.add(name); continue
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected/linked/duplicate archive member")
            row = expected[member.name]
            if member.size != row["bytes"]: raise ValueError("Archive member size differs")
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
            if digest.hexdigest() != row["sha256"]: raise ValueError("Archive member hash differs")
            seen.add(member.name)
    if seen != set(expected) or directories != empty: raise ValueError("Archive omitted declared files/empty directories")
    return {"files": len(seen), "opaque_hashes_verified": True, "tensor_deserialization_performed": False}


def install_bundle(root, value, path):
    """Overlay regular opaque bytes, retaining identical files and never clobbering."""
    verify_bundle(value, path)
    expected = {r["path"]: checked_receipt(r) for r in value["files"]}
    for row in expected.values():
        target = safe(root, row["path"], file=False)
        if target.exists() and receipt(root, row["path"]) != row:
            raise ValueError("Existing destination artifact conflicts; preserve it: " + row["path"])
    for relative in value["empty_directories"]:
        target = safe(root, relative, file=False)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ValueError("Existing author initial directory is not empty")
    installed = 0
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if member.isdir(): continue
            target = safe(root, member.name, file=False)
            if target.exists(): continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".sync-partial-", delete=False) as output:
                    temporary = Path(output.name)
                    with archive.extractfile(member) as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""): output.write(block)
                    output.flush(); os.fsync(output.fileno())
                temporary.chmod(member.mode & 0o555)
                try: os.link(temporary, target); installed += 1
                except FileExistsError:
                    if receipt(root, member.name) != expected[member.name]: raise ValueError("Destination changed during no-clobber install")
            finally:
                if temporary is not None: temporary.unlink(missing_ok=True)
    for relative in value["empty_directories"]: safe(root, relative, file=False).mkdir(parents=True, exist_ok=True)
    verify_manifest(root, value)
    return {"new_files_installed": installed, "identical_existing_files_retained": len(expected) - installed,
            "all_hashes_verified": True, "source_or_target_deserialization": False}


def host_verify(host, bindings, *, observed=None):
    if observed is None:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=15)
        observed = {"hostname": socket.gethostname(), "gpu_uuids": output.splitlines()}
    if host == "allocation": expected = {"hostname": "anogena-2-0", "gpu_uuids": [ALLOCATION_UUID]}
    elif host == "server77":
        bound = bindings.get(host, {})
        if bound.get("status") != "verified": raise ValueError("Server77 requires later reauthorization and frozen verified binding")
        expected = {key: bound[key] for key in ("hostname", "gpu_uuids")}
    else: raise ValueError("Explicit allocation/server77 host required")
    if observed != expected: raise ValueError("Hostname/GPU UUID preflight failed")
    return observed


def environment_verify(expected, observed=None):
    observed = observed or runtime_requirements()
    return runtime_compatibility_report(expected, observed)


def runtime_compatibility_report(expected, observed, *, require_packages=True):
    requirements = compatibility_requirements(expected)
    for key in ("implementation", "python_major_minor", "platform_system", "machine"):
        if observed.get(key) != requirements[key]: raise ValueError("Required runtime compatibility failed: " + key)
    capabilities = observed.get("stdlib_capabilities", {})
    missing = [capability for capability in requirements["stdlib_capabilities"] if capabilities.get(capability) is not True]
    if missing: raise ValueError("Required stdlib capability absent: " + ", ".join(missing))
    if require_packages:
        mismatches = {package: {"required": version, "observed": observed.get("packages", {}).get(package)}
            for package, version in requirements["numerical_package_versions"].items() if observed.get("packages", {}).get(package) != version}
        if mismatches: raise ValueError("Mandatory numerical/compiled ABI package locks differ: " + canonical(mismatches))
    def field(value, dotted):
        for part in dotted.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return value
    differences = {key: {"source": field(expected, key), "observed": field(observed, key)}
        for key in requirements["provenance_only_version_fields"] if field(expected, key) != field(observed, key)}
    return {"status": "required_metadata_and_stdlib_compatibility_passed", "required_compatibility": requirements,
        "observed_environment": observed, "observed_provenance_differences": differences,
        "numerical_package_locks_checked": require_packages, "native_binary_compatibility_verified": False,
        "bitwise_environment_replication_claimed": False, "GPU_execution_qualification_performed": False}


def binary_compatibility_probe(expected_versions=None):
    expected_versions = expected_versions or {package: importlib.metadata.version(package) for package in PACKAGES}
    code = """import importlib,json,sys
expected=json.loads(sys.argv[1])
mapping={'torch':'torch','numpy':'numpy','scipy':'scipy','scikit-learn':'sklearn','torch-geometric':'torch_geometric','torch-scatter':'torch_scatter','torch-sparse':'torch_sparse','ogb':'ogb'}
observed={}
for package,module in mapping.items():
 value=importlib.import_module(module).__version__;observed[package]=value
 # Sparse wheel metadata carries the mandatory build suffix; module source
 # sometimes exposes only its public version. Real native ops verify the ABI.
 assert value==expected[package] or (package in {'torch-scatter','torch-sparse'} and value==expected[package].split('+')[0])
import torch,torch_scatter,torch_sparse
assert not torch.cuda.is_initialized()
source=torch.tensor([1.,2.,3.]); index=torch.tensor([0,1,0])
assert torch.equal(torch_scatter.scatter_add(source,index,dim=0,dim_size=2),torch.tensor([4.,2.]))
edges=torch.tensor([[0,1],[1,0]]); values=torch.tensor([2.,3.]); dense=torch.tensor([[5.],[7.]])
assert torch.equal(torch_sparse.spmm(edges,values,2,2,dense),torch.tensor([[14.],[15.]]))
assert not torch.cuda.is_initialized()
print(json.dumps({'native_CPU_imports_and_operations_passed':True,'runtime_module_versions':observed,'CUDA_initialized':False,'scientific_fits':0}))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, canonical(expected_versions)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=60)
    if result.returncode: raise ValueError("Pinned native binary import/operation compatibility probe failed")
    return json.loads(result.stdout.strip())


def allocation_guard():
    return host_verify("allocation", {})


def Git_overlay_verify(root, value):
    contract = value["Git_overlay_contract"]
    def output(argv):
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
        if result.returncode: raise ValueError("Existing Git checkout/remote preflight failed")
        return result.stdout.strip()
    repository = Path(output(["git", "-C", str(root), "rev-parse", "--show-toplevel"]).decode()).resolve()
    if not Path(root).resolve().is_relative_to(repository):
        raise ValueError("Destination phase must be inside its existing Git checkout")
    remote = output(["git", "-C", str(repository), "remote", "get-url", contract["remote_name"]])
    if hashlib.sha256(remote).hexdigest() != contract["remote_url_sha256"]:
        raise ValueError("Existing Git remote differs from plan-bound private URL digest")
    return {"existing_checkout": True, "phase_relative_to_repository": Path(root).resolve().relative_to(repository).as_posix(),
            "remote_name": contract["remote_name"],
            "remote_URL_digest_matches_plan": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    for name in ("project-root", "plan", "environment", "output"): build.add_argument("--" + name, required=True)
    build.add_argument("--tar", action="store_true")
    check = sub.add_parser("preflight")
    for name in ("project-root", "manifest", "host"): check.add_argument("--" + name, required=True)
    check.add_argument("--binary-import-check", action="store_true")
    check.add_argument("--manifest-hash", required=True)
    check.add_argument("--expected-hostname")
    check.add_argument("--expected-gpu-uuids", nargs="+")
    install = sub.add_parser("install")
    for name in ("project-root", "manifest", "manifest-hash", "host", "archive"):
        install.add_argument("--" + name, required=True)
    install.add_argument("--expected-hostname"); install.add_argument("--expected-gpu-uuids", nargs="+")
    env = sub.add_parser("environment"); env.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "environment":
        allocation_guard(); path = Path(args.output)
        if path.exists(): raise ValueError("Fresh exact environment record required")
        path.write_text(json.dumps(runtime_requirements(), indent=2) + "\n"); path.chmod(0o444); return
    if args.command == "prepare":
        allocation_guard(); root = Path(args.project_root).absolute(); out = safe(root, args.output, file=False)
        if out.exists(): raise ValueError("Use a fresh immutable sync output")
        expected = read_json(root, args.environment); environment_verify(expected)
        tool = Path(__file__).absolute()
        if not tool.is_relative_to(root): raise ValueError("Synchronization tooling must be frozen inside phase")
        value = manifest(root, args.plan, expected, [receipt(root, str(tool.relative_to(root))), receipt(root, args.environment)])
        out.mkdir(parents=True); path = out / "MANIFEST.json"; path.write_text(json.dumps(value, indent=2) + "\n"); path.chmod(0o444)
        summary = {"manifest_hash": fingerprint(value), "plan_hash": value["plan_hash"], "statistics": value["statistics"],
                   "environment_hash": fingerprint(expected), "scientific_launch_enabled": False, "GO_created": False}
        if args.tar: summary["bundle"] = make_bundle(root, value, out / "ARTIFACTS.tar")
        (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n"); (out / "SUMMARY.json").chmod(0o444)
        print(canonical(summary)); return
    if args.host == "allocation": observed = allocation_guard()
    elif args.host == "server77":
        if not args.expected_hostname or not args.expected_gpu_uuids:
            raise ValueError("Later authorized server77 preflight needs explicit frozen hostname/GPU UUID arguments before any project read")
        observed = host_verify(args.host, {"server77": {"status": "verified", "hostname": args.expected_hostname,
                                                       "gpu_uuids": args.expected_gpu_uuids}})
    else: raise ValueError("Explicit allocation/server77 host required")
    root = Path(args.project_root).absolute(); value = read_json(root, args.manifest)
    if fingerprint(value) != args.manifest_hash: raise ValueError("Synchronization manifest differs from its pinned hash")
    host_verify(args.host, value["host_bindings"], observed=observed)
    if args.command == "install":
        Git_overlay_verify(root, value)
        print(canonical(install_bundle(root, value, safe(root, args.archive)))); return
    plan = verify_manifest(root, value); runtime = environment_verify(value["runtime_requirements"])
    Git = Git_overlay_verify(root, value)
    if args.binary_import_check:
        runtime["native_binary_compatibility_probe"] = binary_compatibility_probe(value["runtime_requirements"]["packages"])
        runtime["native_binary_compatibility_verified"] = True
    print(canonical({"status": "environment_metadata_and_all_manifest_hashes_ready", "plan_hash": fingerprint(plan), "host": args.host,
                     "physical_host": observed, "runtime": runtime, "Git_overlay": Git, "scientific_training_started": False, "GO_created": False,
                     "binary_imports_checked_without_CUDA_initialization": args.binary_import_check,
                     "GPU_execution_or_resource_qualification_performed": False,
                     "author_execution_allowed_on_this_host": args.host == "allocation"}))


if __name__ == "__main__": main()
