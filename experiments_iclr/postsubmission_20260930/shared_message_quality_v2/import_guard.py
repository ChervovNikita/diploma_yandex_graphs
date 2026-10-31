"""Stdlib source closure and restricted imports for the exact runtime copy.

Only phase-owned source is hashed here. No tensor/data payload is opened.
The runtime's installed third-party environment is separately version-frozen.
"""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
IMPORT_SUFFIXES = {".py", ".pyw", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib"}


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


def verify_packet(wanted_manifest_sha):
    manifest_path = confined(HERE / "SOURCE_MANIFEST_v1.json")
    if source_sha(manifest_path) != wanted_manifest_sha:
        raise ValueError("Request-bound runtime source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "gnnm-shared-message-source-manifest-v2" or manifest.get("status") != "source_only_sealed_qualification_unexecuted":
        raise ValueError("Exact sealed v2 packet required")
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
    if bindings.get("schema") != "gnnm-shared-message-source-bindings-v2":
        raise ValueError("Exact v2 dependency closure required")
    for relative, wanted in bindings["sha256"].items():
        file = confined(PHASE / relative)
        if str(file.relative_to(PHASE)) != relative or source_sha(file) != wanted:
            raise ValueError("Runtime inherited source changed: " + relative)
    policy_path = HERE / "RUNTIME_IMPORT_POLICY_v1.json"
    policy = json.loads(policy_path.read_text())
    if policy.get("schema") != "gnnm-shared-message-exact-import-policy-v1":
        raise ValueError("Missing exact phase import policy")
    for relative, wanted in policy["importable_sha256"].items():
        if bindings["sha256"].get(relative) != wanted:
            raise ValueError("Import policy differs from dependency seal")
    tree = {}
    for package in policy["package_roots"]:
        allowed = [p for p in policy["importable_sha256"] if Path(p).parts[0] == package]
        tree[package] = inspect_tree(PHASE / package, allowed, policy["package_directories"])
    return policy, tree


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
    rows = []
    for name, module in sorted(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        file = Path(raw).resolve()
        if not file.is_relative_to(PHASE):
            continue
        relative = str(file.relative_to(PHASE))
        wanted = policy["importable_sha256"].get(relative)
        if wanted is None or source_sha(confined(file)) != wanted:
            raise ValueError("Unsealed phase module already imported: " + name)
        rows.append(dict(module=name, relative_to_phase=relative, sha256=wanted))
    return rows


def install(manifest_sha):
    policy, tree = verify_packet(manifest_sha)
    installed_path_only()
    imported_modules(policy)
    sys.meta_path.insert(0, PhaseFinder(policy))
    return policy, tree
