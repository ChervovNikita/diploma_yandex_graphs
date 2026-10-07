"""Run stdlib-only AST/source-binding checks; never import the prototype."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def function(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def call_names(node):
    return [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]


def main():
    parsed = {}
    for path in sorted(ROOT.glob("*.py")):
        source = path.read_text()
        parsed[path.name] = ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")  # Compile only; no exec/import/pycache.
    prototype = parsed["prototype.py"]
    assignments = {n.targets[0].id: n.value for n in prototype.body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert assignments["RUNTIME_ENABLED"].value is False
    runtime = function(prototype, "build_runtime")
    guard = next(n for n in runtime.body if not isinstance(n, ast.Expr))
    assert isinstance(guard, ast.If)
    assert "RUNTIME_ENABLED is not True" == ast.unparse(guard.test)
    assert any(isinstance(n, ast.Raise) for n in ast.walk(guard))
    for node in prototype.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "torch" not in ast.unparse(node)
    run = function(parsed["runtime_fixture.py"], "run")
    assert ast.unparse(run.body[0]).startswith("api = build_runtime()")

    gat = function(prototype, "gat")
    calls = call_names(gat)
    assert calls.index("native.edge_updater") < calls.index("native.propagate")
    propagate = next(n for n in ast.walk(gat) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) == "native.propagate")
    assert [k.arg for k in propagate.keywords] == ["x", "alpha", "size"]
    assert "effective_alpha = alpha * gate.unsqueeze(-1)" in ast.unparse(gat)
    assert not any("softmax" in name for name in calls)
    gates = function(prototype, "edge_gates")
    assert "torch.where(u == v, torch.ones_like(external), external)" in ast.unparse(gates)
    assert "receive.index_select(0, v) * send.index_select(0, u)" in ast.unparse(gates)
    forward = function(prototype, "forward")
    text = ast.unparse(forward)
    assert "neighbor + native.lins[layer](x)" in text
    assert "x_local = x_local + x" in text
    assert "native.global_attn(native.ln(x_local))" in text
    assert "ForwardRecord(logits, tuple(records))" in text
    assert "policies.members[member].sample" in text
    assert not any(name.rsplit(".", 1)[-1].endswith("_")
                   and not name.rsplit(".", 1)[-1].startswith("__")
                   for name in call_names(runtime))
    assert not any("reset_parameters" in name or "install_factors" in name
                   for name in call_names(runtime))
    for node in ast.walk(runtime):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assert not any(ast.unparse(t).startswith("native.") for t in targets)
            assert not any(ast.unparse(t).startswith("self.current_") for t in targets)
    st = ast.unparse(function(prototype, "st_keep"))
    assert "hard - soft.detach() + soft" in st
    assert "straight_through[:, 0]" in st

    bindings = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())
    for record in bindings["files"]:
        raw = Path(record["path"]).read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    report = {"schema": "inactive-external-message-static-checks-v1",
              "AST_and_compile_only": True, "owned_python_files": list(parsed),
              "source_bindings_checked": len(bindings["files"]),
              "runtime_guard_precedes_numerical_imports": True,
              "native_parameter_assignment_or_reset": False,
              "module_gate_cache": False, "original_propagate_signature": True,
              "model_import_or_runtime_fixture_execution": False,
              "numerical_identity_or_gradient_qualification": "pending"}
    (ROOT / "STATIC_CHECKS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
