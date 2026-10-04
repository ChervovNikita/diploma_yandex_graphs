"""Stdlib-only source checks; do not import the adapter or scientific modules."""
import ast
from hashlib import sha256
import json
from pathlib import Path


def check():
    here = Path(__file__).resolve().parent
    records = json.loads((here / "SOURCE_BINDINGS.json").read_text())["records"]
    for record in records:
        path = here.parent / record["path"]
        data = path.read_bytes()
        if len(data) != record["bytes"] or sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Source binding changed: " + record["path"])
    source = (here / "single_control.py").read_text()
    tree = ast.parse(source)
    compile(source, str(here / "single_control.py"), "exec")
    component_source = (here / "component_check.py").read_text()
    ast.parse(component_source)
    compile(component_source, str(here / "component_check.py"), "exec")
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    methods = {node.name: node for node in classes["SingleEndpointPattern"].body
               if isinstance(node, ast.FunctionDef)}
    forward = ast.get_source_segment(source, methods["query_forward"])
    decode = ast.get_source_segment(source, methods["_decode_features"])
    depth_zero = ast.get_source_segment(source, methods["_depth_zero"])
    initializer = ast.get_source_segment(source, methods["__init__"])
    assert "self.decoder.lin[8](phi)" in decode
    assert "for index in range(8)" in decode
    assert "nn.Linear(last.in_features, 4" in initializer and "fork_rng" in initializer
    assert "self.auxiliary_emission(phi).T" in depth_zero
    assert "left_score.detach()" in forward and "right_score.detach()" in forward
    assert forward.index("neighbors.right, len(queries), right_weight") < forward.index("neighbors.left, len(queries), left_weight")
    assert "outer = h + self.decoder.xlin(h)" in forward and "outer.detach" not in source
    assert forward.index("left_score, eta_left") < forward.index("right_score, eta_right") < forward.index("scalar, _ =")
    assert all("teacher" not in argument.arg and "count" not in argument.arg
               for name in ("encode", "_decode_features", "_depth_zero", "query_forward")
               for argument in methods[name].args.args)
    assert not any(isinstance(node, ast.Name) and node.id in ("FactorLinear", "PatternTwin", "CompletionTwin")
                   for node in ast.walk(tree))
    return {"status": "PASS_SOURCE_ONLY", "bound_files": len(records),
            "syntax_and_static_structure": True, "project_modules_imported": False,
            "numerical_or_native_parity_checks_executed": False,
            "bounded_component_check_source_compiles_but_is_unexecuted": True,
            "runtime_qualification": "PENDING", "execution_release": None}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
