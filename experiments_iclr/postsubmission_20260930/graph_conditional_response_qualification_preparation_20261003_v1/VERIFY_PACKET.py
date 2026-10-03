"""Permitted source-only/custody checks. Never execute a numerical entry point."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import ast
import importlib
import importlib.abc
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
NUMERICAL_ROOTS = {"torch", "numpy", "scipy", "jax", "tensorflow", "cupy"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


class ImportBoundary(ast.NodeVisitor):
    def __init__(self):
        self.depth = 0

    def visit_FunctionDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split(".")[0] in NUMERICAL_ROOTS and self.depth == 0:
                raise AssertionError("Top-level numerical import found")

    def visit_ImportFrom(self, node):
        if node.module and node.module.split(".")[0] in NUMERICAL_ROOTS and self.depth == 0:
            raise AssertionError("Top-level numerical import found")


class NoNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in NUMERICAL_ROOTS:
            raise AssertionError(f"Source verification attempted numerical import: {fullname}")
        return None


def source_checks():
    if any(name.split(".")[0] in NUMERICAL_ROOTS for name in sys.modules):
        raise AssertionError("Use a fresh stdlib-only interpreter for source verification")
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        ImportBoundary().visit(tree)
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text())
    blocker = NoNumericalImports()
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, blocker)
    sys.path.insert(0, str(ROOT))
    try:
        importlib.import_module("core")
        spec = importlib.util.spec_from_file_location("qualification_prepared_witnesses", ROOT / "prepared_witnesses.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if len(module.WITNESS_NAMES) != 10:
            raise AssertionError("Expected ten prepared numerical witness families")
    finally:
        sys.path.pop(0)
        sys.meta_path.remove(blocker)
        sys.dont_write_bytecode = previous_bytecode
    numeric = sorted(n for n in sys.modules if n.split(".")[0] in NUMERICAL_ROOTS)
    if numeric:
        raise AssertionError("Numerical package loaded during source-only verification")
    return {"python_files_ast_parsed": len(python_files),
            "core_and_witness_inventory_stdlib_import_passed": True,
            "prepared_witness_families": len(module.WITNESS_NAMES),
            "numerical_packages_imported": numeric,
            "numerical_witnesses_executed": 0, "synthetic_tensor_operations": 0,
            "data_training_or_live_outcomes_accessed": False}


def verify_external():
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    inputs = provenance["external_bindings"]
    if len({row["path"] for row in inputs}) != len(inputs):
        raise AssertionError("Duplicate external binding")
    for row in inputs:
        path = (PROJECT / row["path"]).resolve()
        path.relative_to(PROJECT)
        if digest(path) != row["sha256"] or path.stat().st_size != row["size"]:
            raise AssertionError(f"External input custody mismatch: {row['path']}")
    return len(inputs)


def verify_manifest():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and p.name != "MANIFEST.json"}
    declared = {row["path"] for row in manifest["files"]}
    if actual != declared or len(declared) != manifest["payload_file_count"]:
        raise AssertionError("Payload file list/count mismatch")
    for row in manifest["files"]:
        path = (ROOT / row["path"]).resolve()
        path.relative_to(ROOT)
        if digest(path) != row["sha256"] or path.stat().st_size != row["size"]:
            raise AssertionError(f"Payload custody mismatch: {row['path']}")
    if any("__pycache__" in p or p.endswith(".pyc") for p in actual):
        raise AssertionError("Runtime bytecode artifact in source preparation")
    return len(declared)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--record-source-verification", action="store_true")
    args = parser.parse_args()
    if args.record_source_verification and not args.source_only:
        parser.error("Recording is allowed only before sealing with --source-only")
    result = {"schema": "source-only-qualification-preparation-check-v1",
              "UTC": datetime.now(timezone.utc).isoformat(), "status": "source_checks_passed",
              **source_checks(), "external_bindings_verified": verify_external(),
              "numerical_qualification_passed": False, "pilot_frozen_or_launched": False}
    if not args.source_only:
        result["sealed_payload_files_verified"] = verify_manifest()
    if args.record_source_verification:
        (ROOT / "SOURCE_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
