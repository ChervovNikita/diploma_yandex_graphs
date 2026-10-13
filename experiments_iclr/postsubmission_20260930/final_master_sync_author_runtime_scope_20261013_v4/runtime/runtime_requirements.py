"""Metadata-only runtime and provider closure; no scientific imports or installs.

AST inventory is conservative: it records imports in source bytes, including
optional branches, separately from declared runtime environments. It does not
claim every source import is executed or that a compiled provider is portable.
"""
import ast
from collections import defaultdict
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import sys

from sync_common import nested_receipts, safe, receipt, checked


MODULE_DISTRIBUTIONS = {
    "torch": "torch", "numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn",
    "torch_geometric": "torch-geometric", "torch_scatter": "torch-scatter",
    "torch_sparse": "torch-sparse", "torch_cluster": "torch-cluster", "torch_spline_conv": "torch-spline-conv",
    "torchdiffeq": "torchdiffeq", "torch_householder": "torch-householder", "torchdata": "torchdata",
    "dgl": "dgl", "ogb": "ogb", "networkx": "networkx", "yaml": "PyYAML",
    "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn", "tqdm": "tqdm",
    "einops": "einops", "numba": "numba", "datasketch": "datasketch", "yacs": "yacs",
    "ninja": "ninja", "setuptools": "setuptools", "psutil": "psutil", "packaging": "packaging",
}


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def metadata_environment(search_paths, required):
    """Use overlay metadata before the base interpreter, never import providers."""
    available = {}
    for directory in search_paths:
        for dist in metadata.distributions(path=[directory]):
            name = dist.metadata.get("Name")
            if name:
                available.setdefault(normalized(name), dist)
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            available.setdefault(normalized(name), dist)
    pending = list(required)
    observed, missing = {}, []
    # Marker evaluation is standard package metadata logic, not a model import.
    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None
    while pending:
        name = pending.pop()
        key = normalized(name)
        if key in observed or key in missing:
            continue
        dist = available.get(key)
        if dist is None:
            missing.append(key)
            continue
        requirements = list(dist.requires or [])
        active = []
        if Requirement is not None:
            for raw in requirements:
                requirement = Requirement(raw)
                if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                    active.append({"name": requirement.name, "specifier": str(requirement.specifier)})
                    pending.append(requirement.name)
        else:
            # Keep a conservative closure on a stdlib-only preparation host.
            # Unevaluated markers must never be presented as a qualified lock.
            for raw in requirements:
                match = re.match(r"\s*([A-Za-z0-9_.-]+)", raw)
                if match:
                    active.append({"name": match.group(1), "specifier": "", "marker_unevaluated": True})
                    pending.append(match.group(1))
        observed[key] = {"name": dist.metadata["Name"], "version": dist.version,
                         "metadata_directory": str(dist._path), "Requires-Dist": requirements,
                         "active_required_distributions": active}
    mismatches = []
    if Requirement is not None:
        for row in observed.values():
            for dep in row["active_required_distributions"]:
                got = observed.get(normalized(dep["name"]))
                if got and dep["specifier"] and not Requirement(dep["name"] + dep["specifier"]).specifier.contains(got["version"], prereleases=True):
                    mismatches.append({"requiring": row["name"], "required": dep, "observed": got["version"]})
    return {"search_paths": search_paths, "distributions": dict(sorted(observed.items())),
            "missing_distribution_metadata": sorted(missing), "dependency_version_mismatches": mismatches,
            "marker_evaluation_available": Requirement is not None,
            "marker_policy": "evaluate_installed_metadata" if Requirement else "conservative_all_declared_dependencies_included",
            "scientific_imports": 0, "CUDA_initialized": False, "package_installs": 0}


def source_imports(root, files):
    imports, dynamic, failed = defaultdict(set), defaultdict(set), []
    local = {Path(row["path"]).stem for row in files if row["path"].endswith(".py")}
    for row in files:
        path = row["path"]
        if not path.endswith(".py") or "/dependencies/" in path:
            continue  # Vendored toolchains are tracked through distribution metadata.
        try:
            tree = ast.parse(safe(root, path).read_text(), filename=path)
        except (SyntaxError, UnicodeDecodeError) as error:
            failed.append({"path": path, "error": str(error)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.name.split(".")[0]].add(path)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                imports[node.module.split(".")[0]].add(path)
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                function = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if function in ("import_module", "__import__"):
                    dynamic[node.args[0].value.split(".")[0]].add(path)
    for name, paths in dynamic.items():
        imports[name].update(paths)
    stdlib = set(sys.stdlib_module_names) | set(sys.builtin_module_names)
    external = {name: sorted(paths) for name, paths in imports.items() if name not in stdlib and name not in local}
    mapping = metadata.packages_distributions()
    distributions = {name: sorted(set(mapping.get(name, []) + ([MODULE_DISTRIBUTIONS[name]] if name in MODULE_DISTRIBUTIONS else [])))
                     for name in external}
    return {"scope": "conservative_AST_inventory_not_dynamic_execution_proof",
            "external_module_source_paths": dict(sorted(external.items())),
            "module_distribution_candidates": distributions,
            "unresolved_external_module_names": sorted(name for name, dist in distributions.items() if not dist),
            "AST_parse_failures": failed, "tensor_deserialization": False}


def contracts(root, plan, studies, files, receipt_function=None):
    if receipt_function is None:receipt_function = receipt
    source = source_imports(root, files)
    base = metadata_environment([], plan["runtime_requirements"]["packages"])
    providers = {}
    for study in studies:
        for run in study["runs"]:
            if run["kind"] == "author_cli_group":
                cli = run["cli"]
                private = cli.get("private_dependency_dir")
                key = "author:" + (private or "base") + ":" + json.dumps(cli["environment"], sort_keys=True)
                if key not in providers:
                    overlay = str(safe(root, private, file=False)) if private else None
                    env = cli["environment"]
                    if overlay and env.get("PYTHONPATH") and Path(env["PYTHONPATH"]).resolve() != Path(overlay).resolve():
                        raise ValueError("Author PYTHONPATH differs from frozen overlay")
                    observation = metadata_environment([overlay] if overlay else [], cli["runtime_versions"])
                    for name, version in cli["runtime_versions"].items():
                        got = observation["distributions"].get(normalized(name))
                        if got is None or got["version"] != version:
                            raise ValueError("Author metadata lock changed: " + name)
                    providers[key] = {"kind": "isolated_author_child_environment" if overlay else "base_author_child_environment", "run_keys": [],
                        "environment": env, "private_dependency_dir": private,
                        "python_executable": cli["python_executable"], "declared_versions": cli["runtime_versions"],
                        "observed_metadata_closure": observation, "execution_host": "allocation",
                        "environment_applied_in_child_only": True,
                        "source_receipts": cli["source_files"], "runtime_provider_receipts": cli.get("runtime_provider_files", []),
                        "raw_data_receipts": cli["raw_files"],
                        "native_import_or_ABI_probe_performed": False}
                providers[key]["run_keys"].append(study["study_id"] + "/" + run["run_id"])
            config = run.get("config", {})
            bindings = {name: config[name] for name in ("bsnn_sources", "bayesian_bsnn_sources") if config.get(name)}
            if len(bindings) == 2 and bindings["bsnn_sources"] != bindings["bayesian_bsnn_sources"]:
                raise ValueError("Conflicting legacy/new Bayesian BSNN provider bindings")
            binding = next(iter(bindings.values()), None)
            if binding:
                dependency = binding["householder_provider_relative_root"]
                key = "BSNN:" + dependency + ":" + json.dumps(binding, sort_keys=True)
                if key not in providers:
                    directory = safe(root, dependency, file=False)
                    overlay_paths = [str(directory / "toolchain_setuptools68"), str(directory)]
                    observation = metadata_environment(overlay_paths, ["torch-householder", "setuptools", "ninja", "torch"])
                    compiled = [row for row in binding["provider"]["files"] if row["path"].endswith(".so")]
                    if not compiled:
                        raise ValueError("BSNN source binding omits its compiled provider")
                    providers[key] = {"kind": "isolated_BSNN_native_provider", "run_keys": [], "source_binding_keys": [],
                        "source_binding": binding, "compiled_provider_receipts": compiled,
                        "observed_metadata_closure": observation,
                        "scoped_sys_path": [dependency + "/toolchain_setuptools68", dependency, binding["author_directory"]],
                        "scoped_environment": {"TORCH_EXTENSIONS_DIR": dependency + "/extension_build", "MAX_JOBS": "1",
                                               "PATH_prefix": dependency + "/ninja/data/bin"},
                        "scope": "adapter_import_context_restores_parent_sys_path_environment_and_modules",
                        "ABI_contract": {"python_major_minor": plan["runtime_requirements"]["python_major_minor"],
                                         "torch": plan["runtime_requirements"]["packages"]["torch"],
                                         "platform_system": plan["runtime_requirements"]["platform_system"],
                                         "machine": plan["runtime_requirements"]["machine"]},
                        "native_import_or_ABI_probe_performed": False,
                        "target_host_native_provider_qualification_required": True}
                providers[key]["source_binding_keys"] = sorted(set(providers[key]["source_binding_keys"]) | set(bindings))
                # Many endpoint configs repeat the same immutable binding; keep finite owners only.
                if run["kind"] == "model_fit":
                    providers[key]["run_keys"].append(study["study_id"] + "/" + run["run_id"])
            buddy = config.get("buddy_sources")
            if buddy:
                cache_builder = checked(buddy["cache_builder"])
                packet = Path(cache_builder["path"]).parent.parent.as_posix()
                declaration = packet + "/PROVIDER_MANIFEST.json"
                declared_receipt = receipt_function(root, declaration)
                value = json.loads(safe(root, declaration).read_text())
                version = re.fullmatch(r"datasketch([0-9]+(?:\.[0-9]+)+)", value.get("distribution", ""))
                if version is None or "cached fitting does not import datasketch" not in value.get("runtime_loader", ""):
                    raise ValueError("Exact BUDDY preparation-only datasketch declaration required")
                dependency = packet + "/providers/datasketch"
                key = "BUDDY-cache:" + dependency
                if key not in providers:
                    rows = [checked(row) for row in value["files"]]
                    if not rows or any(not row["path"].startswith(dependency + "/") for row in rows):
                        raise ValueError("BUDDY provider declaration escapes exact isolated directory")
                    overlay = str(safe(root, dependency, file=False))
                    observation = metadata_environment([overlay], ["datasketch"])
                    found = observation["distributions"].get("datasketch")
                    if found is None or found["version"] != version.group(1):
                        raise ValueError("BUDDY isolated datasketch metadata pin differs")
                    providers[key] = {"kind": "isolated_BUDDY_cache_regeneration_provider", "run_keys": [],
                        "private_dependency_dir": dependency, "declared_versions": {"datasketch": version.group(1)},
                        "observed_metadata_closure": observation, "scoped_sys_path": [dependency, packet + "/source"],
                        "scope": "source_cache_preparation_and_exact_regeneration_only; cached_scientific_fitting_does_not_import_datasketch",
                        "scientific_fit_runtime_dependency": False, "source_provider_manifest": declared_receipt,
                        "provider_receipts": rows, "native_import_or_ABI_probe_performed": False,
                        "provider_imports": 0, "package_installs": 0, "active_parent_environment_modified": False,
                        "cache_builder_source": cache_builder, "optional_regeneration_not_executed": True}
                if run["kind"] == "model_fit":providers[key]["run_keys"].append(study["study_id"] + "/" + run["run_id"])
    candidates = sorted({name for rows in source["module_distribution_candidates"].values() for name in rows})
    native_inventory = metadata_environment([], candidates)
    for name, version in plan["runtime_requirements"]["packages"].items():
        observed = base["distributions"].get(normalized(name))
        if observed is None or observed["version"] != version:
            raise ValueError("Base numerical runtime metadata changed: " + name)
    return {"schema": "final-master-transitive-runtime-requirements-v1",
            "base_plan_environment": plan["runtime_requirements"], "base_metadata_closure": base,
            "provider_contracts": list(providers.values()), "source_import_inventory": source,
            "conservative_native_source_distribution_inventory": native_inventory,
            "unresolved_or_optional_source_imports_are_not_claimed_runtime_ready": True,
            "target_host_base_and_provider_native_ABI_preflight_required": True,
            "scientific_fits": 0, "tensor_deserialization": False, "GO_created": False}
