"""Synchronous validation meter and in-memory AST instrumentation.

This stdlib tooling is imported by the v2 AST/compile qualification. Scientific
project code is compiled for qualification but is not imported or executed.

Files stay unchanged. Only pure raise guards, require/_require calls and assert
tests are wrapped. Mathematical and masked-law branches are not transformed.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import time


METER_NAME = "__ar_diagnostic_meter__"


class DiagnosticBudgetExceeded(RuntimeError):
    pass


class Meter:
    def __init__(self, cap, output):
        self.cap, self.output = cap, output
        self.depth, self.charged = 0, 0.0
        self.stats, self.inventory = {}, []
        self.phase_source_setup_seconds = 0.0
        self.last_save = 0.0

    @contextmanager
    def block(self, label, predicate_only=False):
        start = time.perf_counter_ns()
        outer = self.depth == 0
        self.depth += 1
        success = False
        try:
            yield
            success = True
        finally:
            self.depth -= 1
            elapsed = (time.perf_counter_ns() - start) / 1e9
            row = self.stats.setdefault(label, {"calls": 0, "inclusive_seconds": 0.0})
            row["calls"] += 1
            row["inclusive_seconds"] += elapsed
            if outer:
                self.charged += (time.perf_counter_ns() - start) / 1e9
            if outer and not predicate_only and time.perf_counter() - self.last_save >= 1:
                try:
                    self.save()
                except BaseException:
                    if success:
                        raise
                    # Retention failure must not replace a source exception.
            if outer and success and not predicate_only:
                self.check_budget()

    def invoke(self, label, function):
        # Exactly one synchronous call; return original value, no bool/coercion.
        # Original exceptions propagate through the timing context unchanged.
        # A test-only wrapper returns before the original if/assert may raise.
        # Defer budget/persistence there so the upcoming source raise is primary.
        predicate_only = label.endswith((":pure_if_raise_predicate", ":assert_predicate"))
        with self.block(label, predicate_only=predicate_only):
            return function()

    def check_budget(self):
        if self.charged > self.cap:
            self.save()
            raise DiagnosticBudgetExceeded("Frozen diagnostic cap exhausted; retain incomplete screen")

    def save(self, final=False):
        self.last_save = time.perf_counter()
        data = {"charged_union_seconds": self.charged, "cap_seconds": self.cap,
                "final_snapshot": bool(final and self.depth == 0), "active_depth": self.depth,
                "nested_inclusive_times_not_summed": self.stats, "transformation_inventory": self.inventory,
                "phase_source_setup_seconds": self.phase_source_setup_seconds,
                "enforcement": "Post-call successful outer require/full-block checks and existing chunk/stage boundaries; test-only if/assert checks defer polling to preserve an upcoming source raise. Observed overrun is retained.",
                "snapshot_scope": "Completed guard/block union only; interrupted outer blocks absent from a hard-kill snapshot are unmeasured, so a stale snapshot is a lower bound",
                "scope": "Guard predicates/calls and explicit diagnostic blocks; constructor preprocessing remains phase algorithm work"}
        path = self.output / "diagnostics.json"
        temp = self.output / "diagnostics.json.tmp"
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temp.replace(path)


class Transform(ast.NodeTransformer):
    def __init__(self, filename):
        self.filename, self.sites = filename, []

    def wrapped(self, expression, node, kind):
        hazards = (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)
        if any(isinstance(child, hazards) for child in ast.walk(expression)):
            raise ValueError("Unsupported binding/yield/await in validation site")
        label = self.filename + ":" + str(node.lineno) + ":" + str(node.col_offset) + ":" + kind
        self.sites.append({"line": node.lineno, "column": node.col_offset, "kind": kind})
        thunk = ast.Lambda(args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=expression)
        call = ast.Call(func=ast.Attribute(value=ast.Name(id=METER_NAME, ctx=ast.Load()), attr="invoke", ctx=ast.Load()),
                        args=[ast.Constant(label), thunk], keywords=[])
        call._ar_injected_guard = True
        return ast.copy_location(call, expression)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Raise):
            node.test = self.wrapped(node.test, node, "pure_if_raise_predicate")
        return node

    def visit_FunctionDef(self, node):
        # Read-only manifest/source/admission verification is phase setup,
        # explicitly classified by root; it performs no numerical validation.
        if node.name == "verify_contract":
            return node
        return self.generic_visit(node)

    def visit_Assert(self, node):
        node = self.generic_visit(node)
        node.test = self.wrapped(node.test, node, "assert_predicate")
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ("require", "_require"):
            # Whole call lives inside thunk: condition and message arguments
            # retain their original evaluation order and exception behavior.
            return self.wrapped(node, node, "require_call")
        return node


class Strip(ast.NodeTransformer):
    def visit_Call(self, node):
        if hasattr(node, "_ar_injected_guard"):
            if node._ar_injected_guard is not True:
                raise ValueError("Unexpected instrumentation marker")
            if not (isinstance(node.func, ast.Attribute) and isinstance(node.func.ctx, ast.Load) and
                    isinstance(node.func.value, ast.Name) and isinstance(node.func.value.ctx, ast.Load) and
                    node.func.value.id == METER_NAME and node.func.attr == "invoke" and
                    len(node.args) == 2 and not node.keywords and
                    isinstance(node.args[0], ast.Constant) and type(node.args[0].value) is str and
                    isinstance(node.args[1], ast.Lambda)):
                raise ValueError("Unexpected instrumentation call")
            thunk = node.args[1]
            arguments = thunk.args
            if (arguments.posonlyargs or arguments.args or arguments.vararg is not None or
                    arguments.kwonlyargs or arguments.kw_defaults or arguments.kwarg is not None or arguments.defaults):
                raise ValueError("Unexpected instrumentation lambda arguments")
            return self.visit(thunk.body)
        return self.generic_visit(node)


def digest(tree):
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def reject_reserved_bindings(tree):
    reserved = {METER_NAME, "__ar_instrumenter__"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names = [node.id]
        elif isinstance(node, ast.arg):
            names = [node.arg]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.alias):
            names = [node.asname or node.name.split(".")[0]]
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names = node.names
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names = [node.name]
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names = [node.name]
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names = [node.rest]
        if reserved.intersection(names):
            raise ValueError("Original AST binds reserved diagnostic injection name")


class Loader(importlib.abc.Loader):
    def __init__(self, owner, path):
        self.owner, self.path = owner, path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raw = self.path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if self.owner.hashes.get(str(self.path.resolve())) != actual:
            raise ValueError("AST loader source hash mismatch")
        tree = ast.parse(raw.decode("utf-8"), filename=str(self.path))
        reject_reserved_bindings(tree)
        original = digest(tree)
        transformer = Transform(str(self.path))
        transformed = ast.fix_missing_locations(transformer.visit(tree))
        restored = digest(Strip().visit(copy.deepcopy(transformed)))
        if restored != original:
            raise ValueError("Stripped instrumentation did not restore original AST")
        self.owner.meter.inventory.append({"path": str(self.path), "source_sha256": actual,
            "original_ast_sha256": original, "transformed_ast_sha256": digest(transformed),
            "stripped_ast_sha256": restored, "sites": transformer.sites})
        module.__dict__[METER_NAME] = self.owner.meter
        module.__dict__["__ar_instrumenter__"] = self.owner
        # This future compilation/execution happens only after root admission.
        # AST/compile qualification never invokes this loader or project code.
        exec(compile(transformed, str(self.path), "exec"), module.__dict__)


class Instrumenter(importlib.abc.MetaPathFinder):
    def __init__(self, research, hashes, meter):
        self.research, self.hashes, self.meter = research, hashes, meter

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("new_learning_gap_independent_v1."):
            return None
        candidate = self.research / (fullname.replace(".", "/") + ".py")
        package = self.research / fullname.replace(".", "/") / "__init__.py"
        selected = candidate if str(candidate.resolve()) in self.hashes else package
        if str(selected.resolve()) not in self.hashes:
            return None
        return importlib.util.spec_from_file_location(fullname, selected, loader=Loader(self, selected),
            submodule_search_locations=[str(selected.parent)] if selected.name == "__init__.py" else None)

    def load_file(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path, loader=Loader(self, path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
