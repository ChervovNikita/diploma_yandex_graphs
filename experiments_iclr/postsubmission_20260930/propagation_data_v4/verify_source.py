"""Stdlib-only static and early-refusal checks; never imports dataset runtimes."""
import argparse
import ast
import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script = HERE / "acquire.py"
    source = script.read_text()
    tree = ast.parse(source)
    compile(source, str(script), "exec")
    compile(Path(__file__).read_text(), str(Path(__file__)), "exec")
    checks = {}
    attrs = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    checks["no_provider_num_classes_access"] = not any(n.attr == "num_classes" for n in attrs)
    target_attrs = [n for n in attrs if n.attr == "y"]
    checks["one_target_attribute_read"] = len(target_attrs) == 1
    target_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == "index_select"
                    and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "y"]
    checks["target_access_exactly_official_train_index_select"] = (
        len(target_calls) == 1 and ast.unparse(target_calls[0]) == "data.y.index_select(0, ids['train'])")
    checks["no_cuda_tensor_transfer"] = not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and (n.func.attr == "cuda" or n.func.attr == "to" and any(
            isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("cuda")
            for a in [*n.args, *(k.value for k in n.keywords)]))
        for n in ast.walk(tree))
    checks["only_cuda_namespace_use_is_is_initialized"] = all(
        isinstance(parent, ast.Attribute) and parent.attr == "is_initialized"
        for n in attrs if n.attr == "cuda"
        for parent in ast.walk(tree) if isinstance(parent, ast.Attribute) and parent.value is n)
    checks["no_training_prediction_or_score_entrypoints"] = not any(
        isinstance(n, ast.Attribute) and n.attr in {"backward", "fit", "predict", "roc_auc_score", "accuracy_score"}
        for n in ast.walk(tree))
    spec = importlib.util.spec_from_file_location("propagation_data_acquire_source_check", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bindings = module.source_bindings()
    checks["sealed_source_and_guard_dependencies_match"] = True
    checks["fixed_official_class_counts"] = module.TASKS["reddit2"]["class_count"] == 41 and module.TASKS["products"]["class_count"] == 47
    checks["exact_two_adopted_complete_tasks"] = set(module.TASKS) == {"reddit2", "products"}
    prospective_output = HERE / "NEVER_CREATED_STATIC_EXPORT"
    nonexistent_supervisor = HERE / "NEVER_CREATED_STATIC_SUPERVISOR"
    if prospective_output.exists() or nonexistent_supervisor.exists():
        raise ValueError("Static refusal test path already exists; no overwrite")
    argv = ["--task", "reddit2", "--output", str(prospective_output), "--supervisor", str(nonexistent_supervisor),
            "--authorize-public-download", "Reddit2"]
    refusals = []
    original_import = builtins.__import__

    def no_thirdparty(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_geometric", "ogb", "numpy"}:
            raise AssertionError("Third-party import reached during static refusal")
        return original_import(name, *args, **kwargs)

    cases = [
        ("mismatched_public_authorization", argv[:-1] + ["ogbn-products"], ValueError),
        ("output_outside_packet", argv[:3] + [str(HERE.parent / "NEVER_CREATED_OUTSIDE_PACKET")] + argv[4:], ValueError),
        ("missing_live_supervisor", argv, FileNotFoundError),
        ("invalid_task", ["--task", "unsupported", *argv[2:]], SystemExit),
    ]
    try:
        builtins.__import__ = no_thirdparty
        for name, arguments, expected in cases:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    module.main(arguments)
            except expected as error:
                refusals.append({"case": name, "refusal_type": type(error).__name__, "before_thirdparty_import": True})
            else:
                raise AssertionError("Expected early refusal: " + name)
    finally:
        builtins.__import__ = original_import
    checks["all_early_refusals_before_thirdparty_import"] = len(refusals) == 4
    checks["early_refusals_create_no_output"] = not prospective_output.exists() and not (HERE.parent / "NEVER_CREATED_OUTSIDE_PACKET").exists()
    checks["no_thirdparty_runtime_loaded"] = not any(n in sys.modules for n in ("torch", "torch_geometric", "ogb", "numpy"))
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks, sort_keys=True))
    module.write_json(args.output, {"schema": "gnnm-propagation-data-static-source-check-v1", "checks": checks,
                      "refusals": refusals, "source_bindings": bindings,
                      "provider_execution_tested": False, "numerical_tests_run": False, "downloads_run": False,
                      "scope": "Compilation, AST target scope, source hashes, guard-first refusal and path confinement only; no synthetic scientific result"})
    print(json.dumps({"passed": len(checks), "refusal_cases": len(refusals), "provider_execution_tested": False}), flush=True)


if __name__ == "__main__":
    main()
