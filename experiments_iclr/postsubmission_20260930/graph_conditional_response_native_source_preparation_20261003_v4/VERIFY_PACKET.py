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
PREVIOUS = RESEARCH / "graph_conditional_response_native_source_preparation_20261003_v3"
PREVIOUS_MANIFEST_SHA256 = "ee06207b22ff3c073f70bcea60d21be88bb9dcf7c63e040d969fb74a4e8d1bd2"
MODERN = RESEARCH / "continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3"
AUTHOR = RESEARCH / "coordinate_source_independent_review_v1/strong_backbones_v1"
PORTS = RESEARCH / "efficient_graph_ensemble_ports_preparation_v1"
NUMERICAL_ROOTS = {"torch", "torch_geometric", "torch_scatter", "torch_sparse", "numpy", "scipy",
                   "jax", "tensorflow", "cupy", "pandas", "sklearn", "matplotlib", "dgl"}
RAW_NATIVE = {"native_source/native_polyformer.py", "native_source/native_polyformer_outer.py",
              "native_source/native_preprocess.py"}
ENTRY_MODULES = ("core", "native_source", "resolved_groups", "masks", "train_roles", "prepared_native_checks",
                 "prepared_native_reference_checks", "prepared_native_view_checks",
                 "prepared_native_transaction_checks", "prepared_native_adjoint_checks", "engineering_receipts")


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
    if len(inherited) != 92 or any(declared.get(path) != value for path, value in expected.items()):
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
    protected = ("EXPLICIT_SUCCESSOR_RULES.json", "MASK_ALTERNATIVES.json", "resolved_groups.py", "masks.py",
                 "native_source/tokens.py")
    for relative in protected:
        if (ROOT / relative).read_bytes() != (PREVIOUS / relative).read_bytes():
            raise AssertionError(f"Adopted decisions or unresolved scientific implementation changed: {relative}")
    if (ROOT / "native_source/adapter.py").read_bytes() != (PREVIOUS / "native_source/adapter.py").read_bytes():
        raise AssertionError("V4 changed the sealed v3 native adapter")
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
            "inherited_pinned_PyG_files_verified_not_imported": len(fetch),
            "outer_changes_limited_to_relative_import": True,
            "adopted_decision_and_unresolved_scientific_implementation_files_byte_identical": len(protected),
            "native_adapter_byte_identical_to_sealed_v3": True,
            **verify_split_attribution()}


def verify_split_attribution():
    source = AUTHOR / "sources/polyformer_code/node_classification"
    training = parse(source / "training.py")
    standard = {"roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}
    filtered = {"chameleon_filtered", "squirrel_filtered"}
    branches = {}
    for node in ast.walk(training):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.comparators) == 1:
            values = node.test.comparators[0]
            if isinstance(values, ast.List) and all(isinstance(v, ast.Constant) for v in values.elts):
                datasets = frozenset(v.value for v in values.elts)
                calls = {n.func.id for stmt in node.body for n in ast.walk(stmt)
                         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
                branches[datasets] = calls
    if "heter_fixed_splits" not in branches.get(frozenset(standard), ()) or "hetergraph_fixed_split" not in branches.get(frozenset(filtered), ()):
        raise AssertionError("Pinned dataset-to-split helper attribution changed")
    helpers = {n.name: n for n in parse(source / "utils.py").body if isinstance(n, ast.FunctionDef)}
    for name, expression in (("heter_fixed_splits", "dataset[0].train_mask.permute(1,0)[idx_run]"),
                             ("hetergraph_fixed_split", "dataset[0].train_mask[idx_run]")):
        actual = [n.value for n in helpers[name].body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == "train_mask" for t in n.targets)]
        if len(actual) != 1 or ast.dump(actual[0], include_attributes=False) != ast.dump(ast.parse(expression, mode="eval").body, include_attributes=False):
            raise AssertionError("Pinned split-helper expression changed")
    process = [n for n in ast.walk(parse(source / "dataloader/dataset_loader.py"))
               if isinstance(n, ast.FunctionDef) and n.name == "process"]
    assignments = [n.value for n in process[0].body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "train_mask" for t in n.targets)]
    expected = ast.parse("torch.tensor(data['train_masks']).to(torch.bool)", mode="eval").body
    if len(assignments) != 1 or ast.dump(assignments[0], include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise AssertionError("Filtered loader's untransposed raw-mask attribution changed")
    return {"Amazon_standard_and_filtered_split_call_paths_AST_verified": True,
            "author_Amazon_column_selection_consistent_with_successor": True}


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
        if len(modules["prepared_native_checks"].CHECK_NAMES) != 8:
            raise AssertionError("Expected eight prepared native check families")
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
            "prepared_native_check_families": 8, "numerical_packages_imported": numeric,
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
    result = {"schema": "native-source-only-verification-v4", "UTC": datetime.now(timezone.utc).isoformat(),
              "status": "source_checks_passed", "stdlib_interpreter": sys.version,
              "stdlib_interpreter_executable": sys.executable, **source_checks(external),
              "external_bindings_verified": len(external), "native_numerical_qualification_passed": False,
              "control_recipe_qualification_passed": False, "pilot_frozen_or_launched": False}
    if not args.source_only:
        result["sealed_payload_files_verified"] = verify_manifest()
    if args.record_source_verification:
        result["verified_payload_bindings"] = payload_bindings()
        (ROOT / "SOURCE_VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    shown = {k: v for k, v in result.items() if k != "verified_payload_bindings"}
    print(json.dumps(shown, indent=2))
