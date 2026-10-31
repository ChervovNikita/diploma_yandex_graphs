"""Capture exact runtime dependency closure and recreate a fresh isolated venv.

Capture is allocation-only metadata. Dry-run never downloads or installs.
Recreation is a future explicit action after host reauthorization; it does
not touch the running allocation venv, use sudo, or start scientific workers.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys

from prepare_sync_bundle import (PACKAGES, allocation_guard, canonical, fingerprint, host_verify,
                                 read_json, safe, compatibility_requirements, runtime_compatibility_report)
from prepare_sync_bundle import environment_verify, binary_compatibility_probe

TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
PYG_INDEX = "https://data.pyg.org/whl/torch-2.1.0+cu118.html"
PYPI_INDEX = "https://pypi.org/simple"
COMPILED = {"torch-scatter", "torch-sparse"}


def verify_runtime_compatibility(expected_environment):
    """Root startup API; plain receipt using this process's Python, no CUDA init.

Caller owns physical host/plan/source/approval checks. This API checks mandatory
numerical metadata, required stdlib capabilities, and actual CPU native kernels.
It neither writes files nor installs packages, registers jobs or starts workers.
"""
    report = environment_verify(expected_environment)
    probe = binary_compatibility_probe(expected_environment["packages"])
    return {"schema": "runtime-compatibility-receipt-v1", "status": "required_runtime_compatibility_passed",
        "source_environment_hash": fingerprint(expected_environment), "runtime_executable": str(Path(sys.executable).resolve()),
        "observed_environment": report["observed_environment"], "observed_environment_hash": fingerprint(report["observed_environment"]),
        "required_compatibility": report["required_compatibility"],
        "observed_provenance_differences": report["observed_provenance_differences"],
        "mandatory_numerical_pins_verified": True, "required_stdlib_capabilities_verified": True,
        "CPU_native_extension_probe": probe, "native_binary_compatibility_verified": True,
        "CUDA_initialized": False, "GPU_resource_qualification_performed": False,
        "scientific_fits": 0, "workers_started": 0, "bitwise_environment_replication_claimed": False}


def name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def dependency_closure(roots=PACKAGES):
    try:
        from packaging.requirements import Requirement
        from packaging.markers import default_environment
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.markers import default_environment
    marker_environment = dict(default_environment(), extra="")
    queue, locked, edges = list(roots), {}, []
    while queue:
        package = name(queue.pop())
        if package in locked: continue
        distribution = metadata.distribution(package); version = distribution.version
        if not re.fullmatch(r"[A-Za-z0-9.+!-]+", version): raise ValueError("Non-literal installed version")
        locked[package] = version
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate(marker_environment): continue
            if requirement.url: raise ValueError("Direct-URL dependency needs a separately frozen wheel receipt")
            dependency = name(requirement.name); observed = metadata.version(dependency)
            if observed not in requirement.specifier:
                raise ValueError("Source metadata dependency is inconsistent: " + package + " -> " + dependency)
            queue.append(dependency); edges.append({"parent": package, "dependency": dependency,
                                                  "source_requirement": text, "observed_version": observed})
    return dict(sorted(locked.items())), sorted(edges, key=canonical)


def capture(environment):
    locked, edges = dependency_closure()
    if any(locked[name(p)] != version for p, version in environment["packages"].items()):
        raise ValueError("Allocation numerical versions changed after environment snapshot")
    return {"schema": "compatible-isolated-sync-environment-recipe-v2", "environment_hash": fingerprint(environment),
        "source_environment": environment, "packages": locked, "dependency_edges": edges,
        "roots": list(PACKAGES), "indices": {"torch": TORCH_INDEX, "compiled_pyg": PYG_INDEX, "other": PYPI_INDEX},
        "base_interpreter": "CPython3.11_Linux_x86_64_with_required_stdlib_capabilities;micro_and_system_library_versions_recorded_as_provenance",
        "required_compatibility": compatibility_requirements(environment),
        "wheel_policy": "binary_only_no_source_build_no_dependencies_unpinned",
        "author_scope": "allocation_author_absolute_environment_unchanged_not_recreated_here",
        "CUDA_environment_qualification": "pending_on_server77;CPU_import_check_is_not_GPU_qualification",
        "no_sudo": True, "running_allocation_environment_modified": False,
        "downloads_or_installation_performed": False, "scientific_launch_enabled": False}


def recipe_commands(root, recipe, base_python, venv_relative, wheels_relative, lock_relative):
    venv = safe(root, venv_relative, file=False); wheels = safe(root, wheels_relative, file=False)
    lock = safe(root, lock_relative, file=False)
    if venv.exists() or wheels.exists(): raise ValueError("Fresh isolated venv and wheelhouse required; never alter an existing environment")
    if venv == Path(sys.prefix).resolve() or venv.name == ".venv": raise ValueError("Running/default allocation venv is forbidden")
    packages = recipe["packages"]
    if packages.get("torch") != "2.1.2+cu118" or any("pt21cu118" not in packages.get(p, "") for p in COMPILED):
        raise ValueError("This recipe explicitly targets the observed Torch2.1/CUDA11.8 compiled ABI")
    python = str(venv / "bin/python")
    commands = [[str(base_python), "-B", "-m", "venv", str(venv)],
        [python, "-B", "-m", "pip", "download", "--only-binary=:all:", "--no-deps", "--index-url", TORCH_INDEX,
         "--dest", str(wheels), "torch==" + packages["torch"]],
        [python, "-B", "-m", "pip", "download", "--only-binary=:all:", "--no-deps", "--no-index", "--find-links", PYG_INDEX,
         "--dest", str(wheels), *[p + "==" + packages[p] for p in sorted(COMPILED)]],
        [python, "-B", "-m", "pip", "download", "--only-binary=:all:", "--no-deps", "--index-url", PYPI_INDEX,
         "--dest", str(wheels), *[p + "==" + v for p, v in packages.items() if p != "torch" and p not in COMPILED]],
        [python, "-B", "-m", "pip", "install", "--only-binary=:all:", "--no-index", "--find-links", str(wheels), "-r", str(lock)],
        [python, "-B", "-m", "pip", "check"]]
    return commands


def base_check(base_python, expected):
    code = """import importlib,json,platform,sqlite3,ssl,sys
capabilities={}
for name in ['fcntl','sqlite3','ssl','gzip','zlib','tarfile','subprocess']:
 try: importlib.import_module(name); capabilities[name]=True
 except (ImportError,OSError): capabilities[name]=False
with sqlite3.connect(':memory:') as connection:
 connection.execute('CREATE TABLE probe (id INTEGER PRIMARY KEY,state TEXT)')
 connection.execute("INSERT INTO probe VALUES (1,'ready')");connection.commit()
 capabilities['sqlite_transactions']=connection.execute('SELECT state FROM probe WHERE id=1').fetchone()==('ready',)
capabilities['ssl_context']=isinstance(ssl.create_default_context(),ssl.SSLContext)
print(json.dumps({'implementation':platform.python_implementation(),'python_version':platform.python_version(),'python_major_minor':list(sys.version_info[:2]),'platform_system':platform.system(),'machine':platform.machine(),'libc':list(platform.libc_ver()),'stdlib':{'sqlite':sqlite3.sqlite_version,'OpenSSL':ssl.OPENSSL_VERSION},'stdlib_capabilities':capabilities}))
"""
    raw = subprocess.check_output([str(base_python), "-B", "-c", code], text=True, timeout=30)
    value = json.loads(raw)
    return runtime_compatibility_report(expected, value, require_packages=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "dry-run", "recreate"])
    parser.add_argument("--project-root", required=True); parser.add_argument("--recipe", required=True)
    parser.add_argument("--environment"); parser.add_argument("--recipe-hash")
    parser.add_argument("--base-python"); parser.add_argument("--venv", default="isolated_environments/gnnm_sync_v1")
    parser.add_argument("--wheelhouse", default="isolated_environments/gnnm_sync_v1_wheels")
    parser.add_argument("--lock", default="isolated_environments/gnnm_sync_v1_lock.txt")
    parser.add_argument("--manifest"); parser.add_argument("--expected-hostname"); parser.add_argument("--expected-gpu-uuids", nargs="+")
    args = parser.parse_args()
    if args.mode == "capture": allocation_guard()
    elif args.mode == "recreate":
        if not args.expected_hostname or not args.expected_gpu_uuids:
            raise ValueError("Later reauthorized server77 setup requires explicit expected hostname/UUIDs")
        host_verify("server77", {"server77": {"status": "verified", "hostname": args.expected_hostname,
                                            "gpu_uuids": args.expected_gpu_uuids}})
    root = Path(args.project_root).absolute()
    if args.mode == "capture":
        if not args.environment: raise ValueError("Exact source environment snapshot required")
        output = safe(root, args.recipe, file=False)
        if output.exists(): raise ValueError("Preserve earlier recipe; use fresh output")
        value = capture(read_json(root, args.environment)); output.write_text(json.dumps(value, indent=2) + "\n"); output.chmod(0o444)
        print(canonical({"recipe_hash": fingerprint(value), "environment_hash": value["environment_hash"], "pinned_distributions": len(value["packages"]),
                         "dependency_edges": len(value["dependency_edges"]), "downloads_or_installs": 0})); return
    recipe = read_json(root, args.recipe)
    if recipe.get("schema") != "compatible-isolated-sync-environment-recipe-v2" or fingerprint(recipe) != args.recipe_hash:
        raise ValueError("Pinned compatibility recipe hash required")
    if not args.base_python: raise ValueError("Supply a qualified CPython3.11 Linux x86_64 base interpreter")
    commands = recipe_commands(root, recipe, args.base_python, args.venv, args.wheelhouse, args.lock)
    if args.mode == "dry-run":
        print(canonical({"recipe_hash": fingerprint(recipe), "commands": commands, "shell_commands": [shlex.join(c) for c in commands],
                         "base_interpreter_prerequisite_verified": False, "downloads_or_installs": 0,
                         "stdlib_or_GPU_environment_replicated": False})); return
    if not args.manifest: raise ValueError("Later final plan-bound sync manifest required before target setup")
    value = read_json(root, args.manifest); host_verify("server77", value["host_bindings"])
    if fingerprint(value["runtime_requirements"]) != recipe["environment_hash"]:
        raise ValueError("Recipe belongs to another final master environment")
    base_report = base_check(args.base_python, recipe["source_environment"])
    lock = safe(root, args.lock, file=False); lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists(): raise ValueError("Fresh exact lock file required")
    lock.write_text("\n".join(p + "==" + v for p, v in recipe["packages"].items()) + "\n")
    lock.with_name(lock.name + ".BASE_COMPATIBILITY.json").write_text(json.dumps(base_report, indent=2) + "\n")
    wheels = safe(root, args.wheelhouse, file=False); wheels.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    for command in commands: subprocess.run(command, env=env, check=True)
    wheel_receipts = []
    for path in sorted(wheels.iterdir()):
        if path.suffix != ".whl" or path.is_symlink(): raise ValueError("Wheelhouse contains a non-wheel or link")
        wheel_receipts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (wheels / "WHEEL_RECEIPTS.json").write_text(json.dumps(wheel_receipts, indent=2) + "\n")
    print(canonical({"status": "fresh_isolated_packages_installed_requires_exact_metadata_CPU_import_and_resource_preflight",
                     "recipe_hash": fingerprint(recipe), "wheel_count": len(wheel_receipts), "scientific_workers_started": 0}))


if __name__ == "__main__": main()
