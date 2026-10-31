"""Stdlib source closure and restricted imports for the exact runtime copy.

Only phase-owned source is hashed here. No tensor/data payload is opened.
The runtime's installed third-party environment is separately version-frozen.
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
IMPORT_SUFFIXES = {".py", ".pyw", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}
GENERATED_NAMES = {"torch_geometric.nn.conv.sage_conv_SAGEConv_propagate",
                   "torch_geometric.nn.conv.sage_conv_SAGEConv_edge_updater"}
GENERATOR_SOURCES = {"torch_geometric/template.py", "torch_geometric/inspector.py",
                     "torch_geometric/nn/conv/message_passing.py", "torch_geometric/nn/conv/sage_conv.py",
                     "torch_geometric/nn/conv/propagate.jinja", "torch_geometric/nn/conv/edge_updater.jinja",
                     "torch_geometric/nn/conv/collect.jinja"}


def source_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confined(path, root=PHASE):
    root, path = Path(root).resolve(), Path(os.path.abspath(path))
    if path == root or not path.is_relative_to(root):
        raise ValueError("Source must remain inside the declared phase")
    part = root
    for name in path.relative_to(root).parts:
        part /= name
        if part.is_symlink():
            raise ValueError("Source/import tree symlinks are forbidden")
    return path.resolve()


def inspect_tree(root, allowed_files, package_directories):
    """Reject bytecode/extensions/extras, shadow directories and all symlinks."""
    root = confined(root)
    allowed = set(allowed_files)
    packages = set(package_directories)
    actual = set()
    module_stems = {str(Path(p).with_suffix("")) for p in allowed if p.endswith(".py")}
    for directory, dirs, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in dirs:
            path = parent / name
            relative = str(path.relative_to(PHASE))
            if path.is_symlink() or name == "__pycache__" or (relative in module_stems and relative not in packages):
                raise ValueError("Unsealed shadow/cache/symlink directory: " + relative)
        for name in files:
            path = parent / name
            relative = str(path.relative_to(PHASE))
            if path.is_symlink():
                raise ValueError("Import/evidence tree symlink: " + relative)
            if path.suffix.lower() in IMPORT_SUFFIXES:
                actual.add(relative)
                if relative not in allowed:
                    raise ValueError("Unsealed importable source/bytecode/extension: " + relative)
    if actual != allowed:
        raise ValueError("Exact runtime source tree is missing sealed importables")
    return sorted(actual)


def verify_generated_capture(generated):
    if (generated.get("schema") != "gnnm-shared-message-pinned-PyG-generated-source-policy-v1"
            or generated.get("distribution") != "torch-geometric" or generated.get("runtime_version") != "2.7.0"
            or not generated.get("generated_modules") or set(generated["generated_modules"]) - GENERATED_NAMES
            or set(generated.get("installed_source_sha256", {})) != GENERATOR_SOURCES):
        raise ValueError("Exact installed PyG generator/template and generated names required")
    capture = generated["capture_receipt"]
    capture_path = confined(PHASE / capture["path"])
    if str(capture_path.relative_to(PHASE)) != capture["path"] or source_sha(capture_path) != capture["sha256"]:
        raise ValueError("Captured installed/template/generated metadata receipt changed")
    audit_ref = generated["capture_root_audit"]
    audit_path = confined(PHASE / audit_ref["path"])
    if source_sha(audit_path) != audit_ref["sha256"]:
        raise ValueError("Root provenance source-only audit changed")
    report = json.loads(capture_path.read_text())
    if (report.get("schema") != "gnnm-pyg-generated-source-metadata-v1" or report.get("package") != "torch-geometric"
            or report.get("package_version") != "2.7.0" or report.get("TMPDIR_relative_to_phase") != "tmp"
            or report.get("Torch_or_PyG_imported") is not False or report.get("dataset_or_target_read") is not False
            or report.get("scientific_or_numerical_run") is not False):
        raise ValueError("Completed exact source-only PyG provenance capture required")
    installed = {r["package_relative_path"]: r["sha256"] for r in report["installed_sources"]}
    if installed != generated["installed_source_sha256"] or len(installed) != len(report["installed_sources"]):
        raise ValueError("Policy differs from actual captured installed source hashes")
    modules = {}
    for row in report["generated_sources"]:
        name = row["module_name"]
        if name not in GENERATED_NAMES or (name in modules and modules[name] != row["sha256"]):
            raise ValueError("Unknown or ambiguous captured generated source")
        modules[name] = row["sha256"]
    expected = {n: dict(sha256=h, basename_prefix=n + "_") for n, h in modules.items()}
    if generated["generated_modules"] != expected:
        raise ValueError("Generated name/hash/prefix policy differs from actual capture")
    for row in report["installed_sources"] + report["generated_sources"]:
        file = confined(PHASE / row["copy_relative_to_phase"])
        if file.stat().st_size != row["bytes"] or source_sha(file) != row["sha256"]:
            raise ValueError("Captured source copy changed")
    terminal_ref = generated["capture_outer_terminal"]
    terminal_path = confined(PHASE / terminal_ref["path"])
    if source_sha(terminal_path) != terminal_ref["sha256"]:
        raise ValueError("Captured source-only outer terminal changed")
    terminal = json.loads(terminal_path.read_text())
    if (terminal.get("schema") != "gnnm-whole-process-bound-terminal-v1" or terminal.get("complete") is not True
            or terminal.get("child_exit_code") != 0 or terminal.get("timed_out") is not False
            or terminal.get("interruption") is not None or terminal.get("root_request_unchanged") is not True
            or terminal.get("within_whole_cap") is not True or terminal.get("START_sha256") != report["outer_START_sha256"]):
        raise ValueError("Completed source-only metadata outer envelope required")
    start_path = confined(terminal_path.parent / "START.json")
    if source_sha(start_path) != report["outer_START_sha256"]:
        raise ValueError("Captured metadata outer START changed")
    start = json.loads(start_path.read_text())
    request_path = confined(PHASE / report["request"]["path"])
    if (source_sha(request_path) != report["request"]["sha256"] or start.get("root_request") != report["request"]
            or start.get("research_script_sha256") != report["entrypoint_sha256"]):
        raise ValueError("Captured source-only request/entrypoint chain changed")
    supervisor = confined(PHASE / report["supervisor_relative_to_phase"])
    names = {"authorization.json", "completion.json", "command.json", "environment.json", "stdout_stderr.log"}
    if set(terminal.get("inner_evidence", {})) != names:
        raise ValueError("Captured metadata inner receipt set changed")
    for name in names:
        file = confined(supervisor / name)
        if terminal["inner_evidence"][name] != dict(sha256=source_sha(file), bytes=file.stat().st_size):
            raise ValueError("Captured metadata inner evidence changed")
    return report


def verify_packet(wanted_manifest_sha):
    manifest_path = confined(HERE / "SOURCE_MANIFEST_v1.json")
    if source_sha(manifest_path) != wanted_manifest_sha:
        raise ValueError("Request-bound runtime source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "gnnm-shared-message-fixed30-scientific-source-manifest-v2" or manifest.get("status") != "source_only_sealed_launch_and_validation_closed":
        raise ValueError("Exact sealed scientific v2 packet required")
    names = set()
    for row in manifest["files"]:
        relative = row["relative_path"]
        file = confined(HERE / relative, HERE)
        if relative in names or str(file.relative_to(HERE)) != relative:
            raise ValueError("Duplicate or noncanonical source manifest path")
        names.add(relative)
        if file.stat().st_size != row["bytes"] or source_sha(file) != row["sha256"]:
            raise ValueError("Runtime sealed packet source changed: " + relative)
    bindings_path = HERE / "SOURCE_BINDINGS.json"
    if source_sha(bindings_path) != manifest["source_bindings_sha256"]:
        raise ValueError("Runtime dependency binding changed")
    bindings = json.loads(bindings_path.read_text())
    if bindings.get("schema") != "gnnm-shared-message-predictive-source-bindings-v2":
        raise ValueError("Exact scientific v2 dependency closure required")
    for relative, wanted in bindings["sha256"].items():
        file = confined(PHASE / relative)
        if str(file.relative_to(PHASE)) != relative or source_sha(file) != wanted:
            raise ValueError("Runtime inherited source changed: " + relative)
    policy_path = HERE / "RUNTIME_IMPORT_POLICY_v1.json"
    policy = json.loads(policy_path.read_text())
    if policy.get("schema") != "gnnm-shared-message-predictive-exact-import-policy-v2":
        raise ValueError("Missing exact phase import policy")
    generated_path = HERE / "GENERATED_MODULE_POLICY_v1.json"
    if source_sha(generated_path) != policy["generated_module_policy_sha256"]:
        raise ValueError("Generated PyG source policy changed")
    generated = json.loads(generated_path.read_text())
    if generated.get("schema") != "gnnm-shared-message-pinned-PyG-generated-source-policy-v1":
        raise ValueError("Exact captured generated-source policy required")
    verify_generated_capture(generated)
    policy["generated_source_policy"] = generated
    for relative, wanted in policy["importable_sha256"].items():
        if bindings["sha256"].get(relative) != wanted:
            raise ValueError("Import policy differs from dependency seal")
    tree = {}
    for package in policy["package_roots"]:
        allowed = [p for p in policy["importable_sha256"] if Path(p).parts[0] == package]
        tree[package] = inspect_tree(PHASE / package, allowed, policy["package_directories"])
    return policy, tree


def verify_generator_installation(policy):
    """Verify installed PyG version and only its captured generator/templates."""
    generated = policy["generated_source_policy"]
    distribution = importlib.metadata.distribution(generated["distribution"])
    if distribution.version != generated["runtime_version"]:
        raise ValueError("PyG generated-code runtime version changed")
    installed = []
    for relative, wanted in sorted(generated["installed_source_sha256"].items()):
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not relative.startswith("torch_geometric/"):
            raise ValueError("Noncanonical installed generator/template path")
        file = Path(distribution.locate_file(relative))
        # The installed package has its own source root; reject every symlink
        # segment, rather than treating the project's tmp as installed code.
        base = Path(distribution.locate_file("")).absolute()
        lexical = file.absolute()
        if not lexical.is_relative_to(base):
            raise ValueError("Installed generator source escaped its distribution")
        current = base
        if current.is_symlink():
            raise ValueError("Installed source root symlink forbidden")
        for part in lexical.relative_to(base).parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Installed generator/template source symlink forbidden")
        if source_sha(file) != wanted:
            raise ValueError("Pinned installed PyG generator/template changed: " + relative)
        installed.append(dict(relative_to_distribution=relative, path=str(file.resolve()), sha256=wanted))
    if not installed:
        raise ValueError("Missing installed PyG generator/template source pins")
    return dict(distribution=generated["distribution"], runtime_version=distribution.version, sources=installed)


def generated_module_receipt(name, raw_path, policy):
    """Allow only a captured full name, exact source hash and confined tmp .py."""
    generated = policy["generated_source_policy"]
    expected = generated["generated_modules"].get(name)
    if expected is None:
        raise ValueError("Unsealed generated phase module name: " + name)
    tmp = confined(PHASE / "tmp")
    file = confined(raw_path, tmp)
    if file.suffix != ".py" or not file.name.startswith(expected["basename_prefix"]):
        raise ValueError("Generated module is not its exact captured tmp source kind")
    actual = source_sha(file)
    if actual != expected["sha256"]:
        raise ValueError("Generated module source differs from the captured template output: " + name)
    return dict(module=name, relative_to_phase=str(file.relative_to(PHASE)),
                sha256=actual, source_kind="pinned_PyG_generated_tmp_source",
                generated_module_policy_sha256=policy["generated_module_policy_sha256"])


def verify_imported_receipts(rows, policy, *, require_generated=False):
    """Audit saved exact phase and generated-source module evidence."""
    if not rows:
        raise ValueError("Missing imported source module receipts")
    seen = set()
    generated = policy["generated_source_policy"]
    for row in rows:
        name, relative = row["module"], row["relative_to_phase"]
        if name in seen:
            raise ValueError("Duplicate imported module receipt")
        seen.add(name)
        if row.get("source_kind") == "pinned_PyG_generated_tmp_source":
            expected = generated["generated_modules"].get(name)
            lexical = Path(relative)
            if (expected is None or lexical.is_absolute() or ".." in lexical.parts
                    or lexical.parts[:1] != ("tmp",) or lexical.suffix != ".py"
                    or not lexical.name.startswith(expected["basename_prefix"])
                    or row.get("sha256") != expected["sha256"]
                    or row.get("generated_module_policy_sha256") != policy["generated_module_policy_sha256"]):
                raise ValueError("Saved generated module receipt differs from exact policy")
        elif row.get("sha256") != policy["importable_sha256"].get(relative):
            raise ValueError("Saved sealed phase module receipt differs")
    # Model-bearing evidence requires the captured SAGE output; validation
    # exporters/family audits may legitimately load only other phase sources.
    if require_generated and not set(generated["generated_modules"]).issubset(seen):
        raise ValueError("Captured generated module evidence is incomplete")


class SealedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, wanted):
        super().__init__(fullname, str(path))
        self.wanted = wanted

    def get_code(self, fullname):
        file = confined(self.path)
        raw = file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.wanted:
            raise ImportError("Sealed source changed before import: " + fullname)
        # Compile exact source; never read or write a bytecode cache.
        return compile(raw, str(file), "exec", dont_inherit=True)


class PhaseFinder(importlib.abc.MetaPathFinder):
    def __init__(self, policy):
        self.policy = policy
        self.modules = {}
        for relative, wanted in policy["importable_sha256"].items():
            path = Path(relative)
            if path.name == "__init__.py":
                fullname = ".".join(path.parent.parts)
                package = True
            else:
                fullname = ".".join(path.with_suffix("").parts)
                package = False
            self.modules[fullname] = (PHASE / relative, wanted, package)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] not in self.policy["package_roots"]:
            return None
        if fullname not in self.modules:
            raise ModuleNotFoundError("Unsealed phase module refused: " + fullname)
        file, wanted, package = self.modules[fullname]
        loader = SealedSourceLoader(fullname, file, wanted)
        return importlib.util.spec_from_file_location(fullname, file, loader=loader,
            submodule_search_locations=[str(file.parent)] if package else None)


def installed_path_only():
    """Remove cwd/script/PYTHONPATH sources before any external import."""
    roots = tuple(Path(p).resolve() for p in {sys.prefix, sys.base_prefix})
    clean = []
    for item in sys.path:
        if item and Path(item).is_absolute() and any(Path(item).resolve().is_relative_to(root) for root in roots):
            clean.append(item)
    if not clean:
        raise ValueError("Missing interpreter-owned stdlib/site-package paths")
    sys.path[:] = clean
    sys.dont_write_bytecode = True


def imported_modules(policy):
    verify_generator_installation(policy)
    rows = []
    for name, module in sorted(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        if name in policy["generated_source_policy"]["generated_modules"]:
            rows.append(generated_module_receipt(name, raw, policy))
            continue
        lexical = Path(os.path.abspath(raw))
        file = Path(raw).resolve()
        if not lexical.is_relative_to(PHASE) and not file.is_relative_to(PHASE):
            continue
        file = confined(raw)
        relative = str(file.relative_to(PHASE))
        wanted = policy["importable_sha256"].get(relative)
        if wanted is None:
            rows.append(generated_module_receipt(name, raw, policy))
            continue
        if source_sha(confined(raw)) != wanted:
            raise ValueError("Unsealed phase module already imported: " + name)
        rows.append(dict(module=name, relative_to_phase=relative, sha256=wanted))
    return rows


def install(manifest_sha):
    policy, tree = verify_packet(manifest_sha)
    installed_path_only()
    verify_generator_installation(policy)
    imported_modules(policy)
    sys.meta_path.insert(0, PhaseFinder(policy))
    return policy, tree
