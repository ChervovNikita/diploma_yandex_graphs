"""Static preparation checks only; never imports the runtime being checked."""
from pathlib import Path
import ast
import hashlib
import json


def inspect_files(root):
    files = ("__init__.py", "boundary_core.py", "training_rules.py", "README.md", "metadata_check.py")
    records, trees = {}, {}
    for name in files:
        path = root / name
        raw = path.read_bytes()
        records[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if path.suffix == ".py":
            trees[name] = ast.parse(raw.decode(), filename=str(path))
    imported = []
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
    if any(name.split(".")[0] in {"torch", "numpy", "scipy", "torch_geometric", "models", "datasets"}
           for name in imported):
        raise ValueError("Numerical/model/data imports are forbidden in this core")
    for name in ("boundary_core.py", "training_rules.py"):
        for node in trees[name].body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                raise ValueError("No top-level runtime calls allowed")
    top_functions = {name: {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
                     for name, tree in trees.items()}
    expected = {"step_global_half_pool", "step_private_half_pool", "step_private_responsibility", "step_own_supcon"}
    actual = {name for name in top_functions["training_rules.py"] if name.startswith("step_")}
    if actual != expected:
        raise ValueError("Exactly four training-rule entrypoints required")
    rules = trees["training_rules.py"]
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in rules.body if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)}
    if {key: constants[key] for key in ("MEMBERS", "POOL_LAMBDA", "SUPCON_ALPHA", "SUPCON_TEMPERATURE", "SUPCON_EPSILON")} != {
            "MEMBERS": 4, "POOL_LAMBDA": 0.5, "SUPCON_ALPHA": 0.05,
            "SUPCON_TEMPERATURE": 0.2, "SUPCON_EPSILON": 1e-12}:
        raise ValueError("Fixed method constants changed")
    step_calls = [node for node in ast.walk(rules) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "step"]
    if len(step_calls) != 1:
        raise ValueError("Exactly one optimizer transition callsite required")
    core = trees["boundary_core.py"]
    replace_calls = [node for node in ast.walk(core) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "setattr"]
    if replace_calls:
        raise ValueError("Native modules must not be replaced")
    hook_methods = {node.func.attr for node in ast.walk(core) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr.startswith("register_forward")}
    if hook_methods != {"register_forward_pre_hook", "register_forward_hook"}:
        raise ValueError("Explicit native pre/post hooks required")
    return {"schema": "boundary-core-static-preparation-check-v1", "status": "static_checks_passed",
            "files": records, "imports": sorted(set(imported)), "training_rule_entrypoints": sorted(expected),
            "fixed_constants": constants, "optimizer_step_callsites": len(step_calls),
            "native_module_replacements": 0, "runtime_modules_imported_for_check": 0,
            "numerical_runtime_imports": 0, "models_constructed": 0, "numerical_probes": 0,
            "fits": 0, "remote_access": False, "source_recipe_adapters_qualified": False,
            "data_providers_implemented": False, "numeric_equivalence_verified": False,
            "remaining_ports": ["exact per-preset boundary/capture mapping", "native construction/settings frontend",
                                "native optimizer/state/RNG/normalization frontend", "native task loss/data-role provider",
                                "source checkpoint/selection/serving integration"]}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    receipt = inspect_files(root)
    (root / "STATIC_CHECKS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "files": len(receipt["files"]),
                      "training_methods": len(receipt["training_rule_entrypoints"]),
                      "runtime_imports": 0, "models": 0, "fits": 0, "adapters_qualified": False}))
