"""Plan-bound portable artifact manifest; opaque byte hashing, no science I/O.

No tensor loader, Git history, credentials, scientific outputs or remote
transport is used. Author absolute runtime bindings remain allocation-only.
"""
import argparse
from collections import Counter
import gzip
import hashlib
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
                   "required_modules": ["fcntl", "sqlite3", "ssl", "gzip", "zlib", "tarfile", "subprocess"]},
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "binary_import_modules": ["torch", "numpy", "scipy", "sklearn", "torch_geometric", "torch_scatter", "torch_sparse", "ogb"],
        "environment_recreation_required": True, "venv_and_system_libraries_bundled": False,
        "compiled_extension_ABI": "exact_Python_major_minor_and_Torch_PyG_scatter_sparse_build_versions",
        "CUDA_runtime": "Torch_wheel_build_and_host_driver_must_be_qualified;no_CUDA_pilot_in_preparation",
        "numerical_equivalence_or_score_claim": False}


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
        row = checked_receipt(row); safe(root, row["path"])
        if row["path"] in files and files[row["path"]] != row:
            raise ValueError("Conflicting union receipts: " + row["path"])
        files[row["path"]] = row; reasons.setdefault(row["path"], set()).add(reason)
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
        for row in nested_receipts(value["data"]):
            if PurePosixPath(row["path"]).is_absolute():
                original = dict(row)
                try: relative = Path(row["path"]).relative_to(Path(root).absolute()).as_posix()
                except ValueError: raise ValueError("Absolute data provenance lies outside authorized phase; not accessed")
                row = dict(row, path=relative)
                aliases[original["path"]] = {"original_provenance_receipt": original, "phase_relative_mirror_receipt": row,
                    "spec_rewritten": False, "execution_uses_absolute_provenance_path": False}
                add(row, "historical_absolute_provider_provenance_phase_relative_mirror")
            else:
                add(row, "study_data_receipt")
        for run in value["runs"]:
            key = value["study_id"] + "/" + run["run_id"]
            if key in run_keys: raise ValueError("Repeated study/run identity")
            run_keys.add(key)
            for row in nested_receipts(run.get("config", {})): add(row, "run_input_receipt")
            if run["kind"] != "author_cli_group":
                adapter = run["adapter"].split(":", 1)[0]
                add(receipt(root, adapter), "frozen_factory_entrypoint")
                # Some standardized contracts use relative strings for role files.
                for field in ("node_inputs", "endpoint_inputs"):
                    for val in run["config"].get(field, {}).values():
                        if isinstance(val, str): add(receipt(root, val), "relative_role_input")
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
        "allocation_only_author_contracts": author, "host_bindings": plan.get("host_bindings", {}),
        "absolute_data_provenance_aliases": [aliases[name] for name in sorted(aliases)],
        "Git_overlay_contract": {"existing_or_later_authorized_cloned_checkout_required": True,
            "git_directory_bundled": False, "remote_name": rendezvous["remote_name"],
            "remote_url_sha256": rendezvous["remote_url_sha256"], "source_phase_relative_to_repository": "experiments_iclr/postsubmission_20260930",
            "artifact_layout_relative_to_destination_phase_must_match": True,
            "remote_URL_or_credentials_in_manifest": False},
        "statistics": {"files": len(files), "opaque_bytes": sum(r["bytes"] for r in files.values()),
                       "bytes_by_top_directory": dict(sorted(sizes.items())), "studies": len(specs), "registered_runs": len(run_keys),
                       "author_groups": len(author), "historical_absolute_provider_aliases": len(aliases)},
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
    keys = ("implementation", "python_version", "python_major_minor", "platform_system", "machine", "libc", "stdlib", "packages")
    if any(key not in observed or key not in expected or observed[key] != expected[key] for key in keys):
        raise ValueError("Exact runtime environment is not ready")
    return {key: observed[key] for key in keys}


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
        code = "import importlib,sys;[importlib.import_module(n) for n in sys.argv[1:]];import torch;assert not torch.cuda.is_initialized()"
        subprocess.run([sys.executable, "-B", "-c", code, *value["runtime_requirements"]["binary_import_modules"]],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), check=True, timeout=60)
    print(canonical({"status": "environment_metadata_and_all_manifest_hashes_ready", "plan_hash": fingerprint(plan), "host": args.host,
                     "physical_host": observed, "runtime": runtime, "Git_overlay": Git, "scientific_training_started": False, "GO_created": False,
                     "binary_imports_checked_without_CUDA_initialization": args.binary_import_check,
                     "GPU_execution_or_resource_qualification_performed": False,
                     "author_execution_allowed_on_this_host": args.host == "allocation"}))


if __name__ == "__main__": main()
