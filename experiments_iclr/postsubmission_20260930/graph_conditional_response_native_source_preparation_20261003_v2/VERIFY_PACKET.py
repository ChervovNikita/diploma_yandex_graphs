"""Source-only AST/import/custody verification; no native/numerical calls."""
from datetime import datetime, timezone
from hashlib import sha1, sha256
from pathlib import Path
import argparse
import ast
import importlib
import importlib.abc
import json
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RESEARCH = PROJECT / "postsubmission_research_20260930"
PREVIOUS = RESEARCH / "graph_conditional_response_qualification_preparation_20261003_v1"
PREVIOUS_MANIFEST_SHA256 = "a25174ad0c4ae8258dde9f2a3f28333f3a0c4c7618f1aad0bcd699c0685b65a9"
MODERN = RESEARCH / "continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3"
AUTHOR = RESEARCH / "coordinate_source_independent_review_v1/strong_backbones_v1"
PORTS = RESEARCH / "efficient_graph_ensemble_ports_preparation_v1"
NUMERICAL_ROOTS = {"torch", "torch_geometric", "torch_scatter", "torch_sparse", "numpy", "scipy",
                   "jax", "tensorflow", "cupy", "pandas", "sklearn", "matplotlib", "dgl"}
RAW_NATIVE = {"native_source/native_polyformer.py", "native_source/native_polyformer_outer.py",
              "native_source/native_preprocess.py"}
ENTRY_MODULES = ("core", "native_source", "resolved_groups", "masks", "train_roles", "prepared_native_checks")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def parse(path):
    return ast.parse(path.read_text(), filename=str(path))


class ImportBoundary(ast.NodeVisitor):
    def __init__(self):
        self.depth = 0

    def visit_FunctionDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node):
        if self.depth == 0 and any(a.name.split(".")[0] in NUMERICAL_ROOTS for a in node.names):
            raise AssertionError("Numerical import outside a deferred function")

    def visit_ImportFrom(self, node):
        if self.depth == 0 and node.module and node.module.split(".")[0] in NUMERICAL_ROOTS:
            raise AssertionError("Numerical import outside a deferred function")


class NoNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in NUMERICAL_ROOTS:
            raise AssertionError(f"Forbidden numerical import during source-only verification: {fullname}")
        return None


def verify_external():
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    rows = provenance["external_bindings"]
    if len({r["path"] for r in rows}) != len(rows):
        raise AssertionError("Duplicate external binding")
    for row in rows:
        path = (PROJECT / row["path"]).resolve()
        path.relative_to(PROJECT)
        if digest(path) != row["sha256"] or path.stat().st_size != row["size"]:
            raise AssertionError(f"External custody mismatch: {row['path']}")
    if digest(PREVIOUS / "MANIFEST.json") != PREVIOUS_MANIFEST_SHA256:
        raise AssertionError("Old sealed preparation changed")
    previous = json.loads((PREVIOUS / "MANIFEST.json").read_text())
    inherited = json.loads((PREVIOUS / "PROVENANCE.json").read_text())["external_bindings"]
    expected = {(PREVIOUS / "MANIFEST.json").relative_to(PROJECT).as_posix(): PREVIOUS_MANIFEST_SHA256}
    expected.update({(PREVIOUS / r["path"]).relative_to(PROJECT).as_posix(): r["sha256"] for r in previous["files"]})
    expected.update({r["path"]: r["sha256"] for r in inherited})
    declared = {r["path"]: r["sha256"] for r in rows}
    if len(inherited) != 19 or any(declared.get(path) != value for path, value in expected.items()):
        raise AssertionError("Old complete payload or inherited custody was omitted")
    return rows


def declaration(path, name, kind):
    found = [n for n in parse(path).body if isinstance(n, kind) and n.name == name]
    if len(found) != 1:
        raise AssertionError(f"Expected one pinned declaration {name}")
    return ast.dump(found[0], include_attributes=False)


def verify_source_copies():
    previous = {r["path"]: r for r in json.loads((PREVIOUS / "MANIFEST.json").read_text())["files"]}
    core_files = ("__init__.py", "groups.py", "response.py", "guards.py", "transaction.py")
    for name in core_files:
        relative = "core/" + name
        if digest(ROOT / relative) != previous[relative]["sha256"] or (ROOT / relative).read_bytes() != (PREVIOUS / relative).read_bytes():
            raise AssertionError("Copied generic core bytes changed")
    if (ROOT / "native_source/native_polyformer.py").read_bytes() != (PORTS / "native_polyformer.py").read_bytes():
        raise AssertionError("Native layer copy changed")
    outer = (PORTS / "native_polyformer_outer.py").read_text()
    expected_outer = outer.replace("from native_polyformer import PolyFormerBlock", "from .native_polyformer import PolyFormerBlock")
    if outer == expected_outer or (ROOT / "native_source/native_polyformer_outer.py").read_text() != expected_outer:
        raise AssertionError("Outer copy has changes beyond declared package import glue")
    if (ROOT / "native_source/native_preprocess.py").read_bytes() != (MODERN / "prototype/native_polyformer_preprocess.py").read_bytes():
        raise AssertionError("Native preprocessing copy changed")
    source = AUTHOR / "sources/polyformer_code/node_classification"
    for name in ("PolyAttn", "FFNNetwork", "FFN", "PolyFormerBlock"):
        if declaration(ROOT / "native_source/native_polyformer.py", name, ast.ClassDef) != declaration(source / "layers/PolyFormerBlock.py", name, ast.ClassDef):
            raise AssertionError(f"Native author class AST differs: {name}")
    if declaration(ROOT / "native_source/native_polyformer_outer.py", "PolyFormer", ast.ClassDef) != declaration(source / "mymodels.py", "PolyFormer", ast.ClassDef):
        raise AssertionError("Native PolyFormer class AST differs")
    for name in ("sparse_mx_to_torch_sparse_tensor", "mono_base"):
        if declaration(ROOT / "native_source/native_preprocess.py", name, ast.FunctionDef) != declaration(source / "utils.py", name, ast.FunctionDef):
            raise AssertionError(f"Native preprocessing AST differs: {name}")

    author_rows = json.loads((MODERN / "AUTHOR_SOURCE_BINDINGS.json").read_text())["bindings"]
    rows = [r for r in author_rows if r["author_repository"] == "air029/PolyFormer"]
    if len(rows) != 6:
        raise AssertionError("Pinned PolyFormer author scope changed")
    for row in rows:
        if row["commit"] != "d390f39e88d0eaac80318fdc7704bd3bf3cf8b13" or not row["match"]:
            raise AssertionError("Unqualified native author source pin")
        for path in (MODERN / row["local_path"], AUTHOR / row["path"]):
            body = path.read_bytes()
            blob = sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
            if digest(path) != row["sha256"] or len(body) != row["bytes"] or blob != row["git_blob_expected"]:
                raise AssertionError("Pinned native author byte/Git-blob custody differs")
    fetch = json.loads((ROOT / "PYG_CODE_FETCH_RECEIPT.json").read_text())["requests"]
    if len(fetch) != 5:
        raise AssertionError("Expected five pinned primary-library code files")
    for row in fetch:
        path = (ROOT / row["path"]).resolve()
        path.relative_to(ROOT / "pyg_source_pins")
        if (row["commit"] != "76ff9c2ce18c8cebf52122b57e2aeadce9793d10" or
                row["commit"] not in row["url"] or digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]):
            raise AssertionError("Pinned PyG source bytes differ")
    return {"copied_core_files_byte_identical": len(core_files), "native_author_class_AST_equalities": 5,
            "native_preprocessing_function_AST_equalities": 2,
            "pinned_native_author_files_byte_and_git_blob_verified": len(rows),
            "new_pinned_PyG_files_verified_not_imported": len(fetch),
            "outer_changes_limited_to_relative_import": True}


def source_checks(external_rows):
    if any(n.split(".")[0] in NUMERICAL_ROOTS for n in sys.modules):
        raise AssertionError("Use a fresh stdlib-only interpreter")
    files = sorted(ROOT.rglob("*.py"))
    for path in files:
        tree = parse(path)
        relative = path.relative_to(ROOT).as_posix()
        if relative not in RAW_NATIVE and not relative.startswith("pyg_source_pins/"):
            ImportBoundary().visit(tree)
    external_python = [(PROJECT / r["path"]) for r in external_rows if r["path"].endswith(".py")]
    for path in external_python:
        parse(path)  # AST only; never import foreign native/library code.
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text())
    blocker = NoNumericalImports()
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, blocker)
    sys.path.insert(0, str(ROOT))
    try:
        modules = {name: importlib.import_module(name) for name in ENTRY_MODULES}
        for name, module in tuple(sys.modules.items()):
            if name.split(".")[0] in ENTRY_MODULES:
                Path(module.__file__).resolve().relative_to(ROOT)
        if len(modules["prepared_native_checks"].CHECK_NAMES) != 3:
            raise AssertionError("Expected three prepared native check families")
        if any("native_source." + Path(r).stem in sys.modules for r in RAW_NATIVE):
            raise AssertionError("Raw native numerical source was imported")
        copies = verify_source_copies()
    finally:
        sys.path.pop(0)
        sys.meta_path.remove(blocker)
        sys.dont_write_bytecode = old_bytecode
    numeric = sorted(n for n in sys.modules if n.split(".")[0] in NUMERICAL_ROOTS)
    if numeric:
        raise AssertionError("Numerical package appeared during source-only verification")
    return {"payload_python_files_AST_parsed": len(files), "external_python_files_AST_parsed": len(external_python),
            "stdlib_inventory_modules_imported": list(ENTRY_MODULES),
            "deferred_numerical_import_boundary_passed": True, **copies,
            "prepared_native_check_families": 3, "numerical_packages_imported": numeric,
            "numerical_native_checks_executed": 0, "generic_witnesses_executed_by_this_successor": 0,
            "model_constructions_tensor_operations_or_data_calls": 0,
            "role_mask_sampling_or_feasibility_calls": 0,
            "source_review_limits": "AST/body/custody/import checks; no numerical or behavioral certification"}


def payload_bindings():
    return [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p), "size": p.stat().st_size}
            for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name not in ("MANIFEST.json", "SOURCE_VERIFICATION.json")]


def verify_manifest():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
    rows = manifest["files"]
    declared = {r["path"] for r in rows}
    if actual != declared or len(rows) != len(declared) or len(rows) != manifest["payload_file_count"]:
        raise AssertionError("Sealed payload inventory/count differs")
    if any("__pycache__" in p or p.endswith(".pyc") for p in actual):
        raise AssertionError("Runtime artifact in source-only preparation")
    for row in rows:
        path = (ROOT / row["path"]).resolve()
        path.relative_to(ROOT)
        if digest(path) != row["sha256"] or path.stat().st_size != row["size"]:
            raise AssertionError(f"Sealed payload custody differs: {row['path']}")
    receipt = json.loads((ROOT / "SOURCE_VERIFICATION.json").read_text())
    if receipt["verified_payload_bindings"] != payload_bindings():
        raise AssertionError("Recorded source verification is stale")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--record-source-verification", action="store_true")
    args = parser.parse_args()
    if args.record_source_verification and (not args.source_only or (ROOT / "MANIFEST.json").exists()):
        parser.error("Record source-only checks before sealing; sealed packets are immutable")
    external = verify_external()
    result = {"schema": "native-source-only-verification-v2", "UTC": datetime.now(timezone.utc).isoformat(),
              "status": "source_checks_passed", **source_checks(external),
              "external_bindings_verified": len(external), "native_numerical_qualification_passed": False,
              "control_recipe_qualification_passed": False, "pilot_frozen_or_launched": False}
    if not args.source_only:
        result["sealed_payload_files_verified"] = verify_manifest()
    if args.record_source_verification:
        result["verified_payload_bindings"] = payload_bindings()
        (ROOT / "SOURCE_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    shown = {k: v for k, v in result.items() if k != "verified_payload_bindings"}
    print(json.dumps(shown, indent=2))
