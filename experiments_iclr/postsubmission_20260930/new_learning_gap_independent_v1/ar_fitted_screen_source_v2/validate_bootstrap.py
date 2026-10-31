"""Stdlib-only AST/compile qualification; execute only artificial fixtures."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

sys.dont_write_bytecode = True
if sys.version_info < (3, 10):
    raise RuntimeError("Qualification requires Python >=3.10, matching the existing ast.MatchAs checker")
BASE = Path(__file__).resolve().parent
RESEARCH = BASE.parents[1]
V1 = BASE.parent / "ar_fitted_screen_source_v1"
EXPECTED_FREEZE = "210ba66129493a973f8826e207ba0841826b63040bc670701a18004b8f7e6833"
SPEC = importlib.util.spec_from_file_location("ar_bootstrap_qualification_diagnostics", BASE / "diagnostics.py")
DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTICS)
INVENTORY = []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def transformed(source, filename="artificial_fixture.py"):
    original = ast.parse(source, filename=filename)
    DIAGNOSTICS.reject_reserved_bindings(original)
    transform = DIAGNOSTICS.Transform(filename)
    tree = ast.fix_missing_locations(transform.visit(copy.deepcopy(original)))
    restored = DIAGNOSTICS.Strip().visit(copy.deepcopy(tree))
    if DIAGNOSTICS.digest(restored) != DIAGNOSTICS.digest(original):
        raise AssertionError("AST round trip changed original source")
    return original, tree, restored, transform.sites


class MemoryMeter(DIAGNOSTICS.Meter):
    """Exercise the actual meter with persistence confined to memory."""
    def __init__(self, cap=1e6):
        super().__init__(cap, BASE)
        self.snapshots = []

    def save(self, final=False):
        self.last_save = time.perf_counter()
        self.snapshots.append((final, self.depth, self.charged))


class BootstrapQualification(unittest.TestCase):
    def test_all_authored_and_pinned_python_roundtrip_and_compile(self):
        old_manifest = json.loads((V1 / "AUTHORING_MANIFEST.json").read_text())
        self.assertEqual(len(old_manifest["files"]), 12)
        authored = []
        for row in old_manifest["files"]:
            self.assertEqual(sha((V1 / row["path"]).read_bytes()), row["sha256"])
            path = BASE / row["path"]
            self.assertTrue(path.is_file())
            if path.suffix == ".py":
                authored.append(path)
            elif path.suffix == ".json":
                json.loads(path.read_text())
        authored.append(Path(__file__).resolve())
        self.assertEqual(set(BASE.glob("*.py")), set(authored))
        self.assertFalse((BASE / "runs").exists())
        pins = json.loads((BASE / "SOURCE_PINS.json").read_text())
        self.assertEqual(pins["scientific_freeze_sha256"], EXPECTED_FREEZE)
        dependencies = []
        for row in pins["files"]:
            path = (RESEARCH / row["research_relative_path"]).resolve()
            self.assertTrue(path.is_relative_to(RESEARCH))
            raw = path.read_bytes()
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(sha(raw), row["sha256"])
            if path.suffix == ".py":
                dependencies.append(path)
        for kind, paths in (("authored", authored), ("pinned_dependency", dependencies)):
            for path in paths:
                with self.subTest(path=path.relative_to(RESEARCH)):
                    raw = path.read_bytes()
                    relative = str(path.relative_to(RESEARCH))
                    original, tree, restored, sites = transformed(raw.decode("utf-8"), relative)
                    compile(raw, relative, "exec", dont_inherit=True)
                    compile(original, relative, "exec", dont_inherit=True)
                    compile(tree, relative, "exec", dont_inherit=True)
                    compile(restored, relative, "exec", dont_inherit=True)
                    INVENTORY.append({"kind": kind, "path": relative, "source_sha256": sha(raw),
                        "original_ast_sha256": DIAGNOSTICS.digest(original),
                        "transformed_ast_sha256": DIAGNOSTICS.digest(tree),
                        "stripped_ast_sha256": DIAGNOSTICS.digest(restored), "injected_sites": len(sites),
                        "raw_original_transformed_restored_compiled": True, "project_code_executed": False})
        self.assertFalse(any(name == "torch" or name.startswith(("torch.", "torch_geometric"))
                             or name == "models" or name.startswith("new_learning_gap_independent_v1")
                             for name in sys.modules))

    def test_original_meter_block_check_budget_and_invoke_preserved(self):
        source = '''with __ar_diagnostic_meter__.block("original"):
    __ar_diagnostic_meter__.check_budget()
    require(__ar_diagnostic_meter__.invoke("original", lambda: True), "ok")
__ar_diagnostic_meter__.invoke(label="keyword", function=lambda: 1)
'''
        original, tree, restored, _ = transformed(source)
        self.assertEqual(DIAGNOSTICS.digest(DIAGNOSTICS.Strip().visit(copy.deepcopy(original))),
                         DIAGNOSTICS.digest(original))
        original_calls = [ast.dump(node, include_attributes=False) for node in ast.walk(original)
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                          and isinstance(node.func.value, ast.Name) and node.func.value.id == DIAGNOSTICS.METER_NAME]
        surviving_calls = [ast.dump(node, include_attributes=False) for node in ast.walk(restored)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and isinstance(node.func.value, ast.Name) and node.func.value.id == DIAGNOSTICS.METER_NAME]
        self.assertEqual(original_calls, surviving_calls)
        self.assertEqual(sum(hasattr(node, "_ar_injected_guard") for node in ast.walk(tree)), 1)

    def test_nested_wrappers_survive_deepcopy_and_strip_recursively(self):
        original, tree, restored, sites = transformed('require(_require(True, "inner"), "outer")')
        calls = [node for node in ast.walk(tree) if getattr(node, "_ar_injected_guard", False) is True]
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(sites), 2)
        copied = copy.deepcopy(tree)
        self.assertEqual(sum(getattr(node, "_ar_injected_guard", False) is True for node in ast.walk(copied)), 2)
        self.assertEqual(DIAGNOSTICS.digest(restored), DIAGNOSTICS.digest(original))
        self.assertFalse(any(hasattr(node, "_ar_injected_guard") for node in ast.walk(restored)))

    def test_artificial_fixture_return_exception_and_evaluation_order(self):
        source = '''def mark(name, value):
    events.append(name)
    if name == "outer-message" and case == "argument-error":
        raise sentinel
    return value
def require(condition, message):
    events.append(("require", condition, message))
    if not condition:
        raise sentinel
    return token
_require = require
def scenario():
    with __ar_diagnostic_meter__.block("original-block"):
        __ar_diagnostic_meter__.check_budget()
        first = __ar_diagnostic_meter__.invoke("original-invoke", lambda: mark("original", token))
        second = require(mark("outer-condition", _require(mark("inner-condition", True), mark("inner-message", "inner")) is token and case != "require-error"), mark("outer-message", "outer"))
        if mark("guard", case == "guard-error"):
            raise sentinel
        assert mark("assert", case != "assert-error"), mark("assert-message", "assert failure")
        return mark("return", (first, second))
'''
        original, tree, _, _ = transformed(source)
        for case in ("return", "argument-error", "require-error", "guard-error", "assert-error"):
            with self.subTest(case=case):
                token, sentinel = object(), RuntimeError("source exception")
                outcomes = []
                for fixture in (original, tree):
                    events, meter = [], MemoryMeter()
                    namespace = {"events": events, "case": case, "token": token, "sentinel": sentinel,
                                 DIAGNOSTICS.METER_NAME: meter}
                    exec(compile(fixture, "artificial_fixture.py", "exec"), namespace)
                    try:
                        value = namespace["scenario"]()
                        self.assertIs(value[0], token)
                        self.assertIs(value[1], token)
                        outcomes.append(("return", value, list(events)))
                    except BaseException as error:
                        if case != "assert-error":
                            self.assertIs(error, sentinel)
                        outcomes.append(("raise", type(error), error.args, list(events)))
                    self.assertEqual(meter.depth, 0)
                    self.assertEqual(meter.stats["original-block"]["calls"], 1)
                    self.assertEqual(meter.stats["original-invoke"]["calls"], 1)
                self.assertEqual(outcomes[0], outcomes[1])
                self.assertNotIn("assert-message", outcomes[1][-1] if case != "assert-error" else [])

    def test_original_predicate_exception_precedes_deferred_budget_check(self):
        sentinel = RuntimeError("source exception")
        for source, expected in (("if True:\n    raise sentinel\n", RuntimeError),
                                 ('assert False, "source assert"', AssertionError)):
            with self.subTest(source=source):
                _, tree, _, _ = transformed(source)
                meter = MemoryMeter(cap=-1)
                with self.assertRaises(expected) as caught:
                    exec(compile(tree, "artificial_predicate.py", "exec"), {"sentinel": sentinel, DIAGNOSTICS.METER_NAME: meter})
                if expected is RuntimeError:
                    self.assertIs(caught.exception, sentinel)
                else:
                    self.assertEqual(caught.exception.args, ("source assert",))
                self.assertEqual(meter.depth, 0)

    def test_malformed_marked_wrappers_rejected(self):
        expressions = [
            '__ar_diagnostic_meter__.block("x", lambda: 1)',
            '__ar_diagnostic_meter__.check_budget()',
            '__ar_diagnostic_meter__.invoke("x")',
            '__ar_diagnostic_meter__.invoke("x", 1)',
            '__ar_diagnostic_meter__.invoke(1, lambda: 1)',
            '__ar_diagnostic_meter__.invoke("x", lambda value: value)',
            '__ar_diagnostic_meter__.invoke("x", lambda *values: values)',
            '__ar_diagnostic_meter__.invoke("x", lambda **values: values)',
            '__ar_diagnostic_meter__.invoke("x", lambda *, value=1: value)',
            '__ar_diagnostic_meter__.invoke("x", lambda: 1, extra=True)',
            'other.invoke("x", lambda: 1)',
        ]
        for source in expressions:
            with self.subTest(source=source):
                call = ast.parse(source, mode="eval").body
                call._ar_injected_guard = True
                with self.assertRaises(ValueError):
                    DIAGNOSTICS.Strip().visit(call)
        call = ast.parse('__ar_diagnostic_meter__.invoke("x", lambda: 1)', mode="eval").body
        call._ar_injected_guard = False
        with self.assertRaises(ValueError):
            DIAGNOSTICS.Strip().visit(call)

    def test_original_reserved_bindings_rejected_reads_allowed(self):
        patterns = [
            "{name} = 1", "del {name}", "def f({name}):\n    pass", "def {name}():\n    pass",
            "async def {name}():\n    pass", "class {name}:\n    pass", "import math as {name}",
            "from math import pi as {name}", "global {name}", "def f():\n    nonlocal {name}",
            "try:\n    pass\nexcept Exception as {name}:\n    pass",
            "match value:\n    case {name}:\n        pass", "match value:\n    case [*{name}]:\n        pass",
            "match value:\n    case {{**{name}}}:\n        pass",
        ]
        for name in (DIAGNOSTICS.METER_NAME, "__ar_instrumenter__"):
            for pattern in patterns:
                with self.subTest(name=name, pattern=pattern):
                    with self.assertRaises(ValueError):
                        DIAGNOSTICS.reject_reserved_bindings(ast.parse(pattern.format(name=name)))
        DIAGNOSTICS.reject_reserved_bindings(ast.parse('__ar_diagnostic_meter__.check_budget()\n__ar_instrumenter__.load_file("x", path)'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result.resolve()
    if not result_path.is_relative_to(RESEARCH):
        raise ValueError("Qualification output must remain within the research directory")
    started = time.perf_counter_ns()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BootstrapQualification)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    record = {"schema": "ar-bootstrap-ast-compile-qualification-v1", "success": result.wasSuccessful(),
        "python_version": sys.version, "python_executable": sys.executable,
        "test_count": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "test_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
        "scientific_freeze_sha256": EXPECTED_FREEZE,
        "validation_source_sha256": sha(Path(__file__).read_bytes()),
        "diagnostics_source_sha256": sha((BASE / "diagnostics.py").read_bytes()),
        "bytecode_writes_disabled": sys.dont_write_bytecode,
        "meter_tooling_imported_and_exercised": True,
        "only_artificial_fixtures_executed": True,
        "scientific_modules_imported": False, "transformed_project_code_executed": False,
        "scientific_draws_or_fits_performed": False,
        "authored_python_count": sum(row["kind"] == "authored" for row in INVENTORY),
        "pinned_python_count": sum(row["kind"] == "pinned_dependency" for row in INVENTORY),
        "compilation_excludes_non_python": True, "inventory": INVENTORY}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "inventory"}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
