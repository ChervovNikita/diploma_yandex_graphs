"""Parse and inspect source using stdlib only; no numerical/module imports."""

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def check():
    parsed = {str(p.relative_to(ROOT)): ast.parse(p.read_text(), filename=str(p))
              for p in ROOT.rglob("*.py")}
    bindings = json.loads((ROOT / "SOURCE_BINDINGS.json").read_text())
    for record in bindings["copies"]:
        blob = (ROOT / record["packet_path"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == record["sha256"]
    pinned = (ROOT / "pinned/hlgnn_layer.py").read_bytes()
    assert hashlib.sha1(b"blob " + str(len(pinned)).encode() + b"\0" + pinned).hexdigest() == "14b9facd720fd571bba2672b61460f6f7fcd5e4f"
    module = parsed["member_filter.py"]
    encoder = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "SharedPowerHLGNN")
    methods = {n.name:n for n in encoder.body if isinstance(n, ast.FunctionDef)}
    init = methods["__init__"]
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],
                        [ast.literal_eval(d) for d in init.args.defaults]))
    assert defaults == {"in_channels":512,"hidden_channels":512,"K":15,"dropout":0.3,
                        "alpha":0.5,"members":1,"private_alpha":False}
    text = ast.unparse(methods["factored"])
    assert "z.new_ones" in text and "matmul(P, current" in text
    assert "self.temp[m]" in text and "storage == 'powers'" in text and "storage == 'aggregates'" in text
    assert "constant * effective_bias" in ast.unparse(methods["affine_after_filter"])
    forward = ast.unparse(methods["forward"])
    assert "if self.members == 1" in forward and "self.lin1(z)" in forward
    direct = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "direct_reference")
    assert "for member in range(model.members)" in ast.unparse(direct)
    assert "matmul(P, h" in ast.unparse(direct)
    plan = parsed["planned_equivalence.py"]
    plan_text = ast.unparse(plan)
    assert "torch.autograd.grad" in plan_text and "torch.testing.assert_close" in plan_text
    assert "native.state_dict()" in plan_text and "mask" in plan_text and "lin1.bias" in plan_text
    for name in ("member_filter.py", "planned_equivalence.py"):
        assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) in
                       ("torch.load", "torch.save", "subprocess.run", "os.system")
                       for n in ast.walk(parsed[name]))
        assert not any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)
                       for n in parsed[name].body)
    return {"status":"passed","scope":"stdlib AST/source only",
            "parsed_python_files":len(parsed),"pinned_layer_git_blob_verified":True,
            "native_defaults_checked":True,"constant_channel_and_two_storage_paths_present":True,
            "private_free_coefficients_and_direct_reference_present":True,
            "planned_output_input_parameter_alpha_gradient_check_present":True,
            "numerical_imports":False,"equivalence_executed":False,"training_or_scoring":False,
            "limits":"Syntax/source inspection does not prove numerical equivalence, runtime compatibility or memory behavior."}


if __name__ == "__main__":
    result=check()
    (ROOT / "SOURCE_CHECKS.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
