"""Stdlib-only static checks; never represented as numerical qualification."""
import ast
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from guards import implementation_hashes

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    sources = implementation_hashes()
    pins = json.loads((HERE / "SOURCE_PINS.json").read_text())
    for name, checksum in pins["files"].items():
        source = HERE / "vendor" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum, name
        ast.parse(source.read_text())
    for source in HERE.glob("*.py"):
        ast.parse(source.read_text())
    builder = ast.parse((HERE / "cache_builder.py").read_text())
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get_edge_split" for n in ast.walk(builder))
    functions_by_name = {n.name: n for n in builder.body if isinstance(n, ast.FunctionDef)}
    build_splits = [n.args[1].value for n in ast.walk(functions_by_name["build"]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "official_split_file"]
    finalize_splits = [n.args[1].value for n in ast.walk(functions_by_name["finalize"]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "official_split_file"]
    assert build_splits == ["train", "valid"] and finalize_splits == ["test"]
    split_reader = functions_by_name["official_split_file"]
    source_check_line = next(n.lineno for n in ast.walk(split_reader) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "checked_split_loader_source")
    test_load_line = next(n.lineno for n in ast.walk(split_reader) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch" and n.func.attr == "load")
    assert source_check_line < test_load_line
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "validate_selected_checkpoint" for n in ast.walk(functions_by_name["verify_family_lock"]))
    config = json.loads((HERE / "CONFIG.json").read_text())
    assert config["seeds"] == [0, 1, 2]
    assert len(config["arms"]) == 5 and config["epochs"] == 100
    assert config["sign_k"] == 0 and config["year"] == 0
    assert config["graph_policy"] == "training_only_all_splits"
    # Evaluate only the two pure integer parameter functions, not model code.
    tree = ast.parse((HERE / "models.py").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"parameter_count", "matched_width"}]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "integer_parameter_arithmetic", "exec"), namespace)
    count = namespace["parameter_count"]
    width = namespace["matched_width"](count(256, factorized=True))
    assert width == 266
    expected = {"native1024": 1185091, "single256": 99907, "factorized4": 106349, "independent4": 399628, "matched_single": 106457}
    assert count(1024) == expected["native1024"] and count(256) == expected["single256"]
    assert count(256, factorized=True) == expected["factorized4"] and count(256, members=4) == expected["independent4"]
    assert count(width) == expected["matched_single"]
    factorized = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FactorizedBUDDY")
    forward = next(n for n in factorized.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "features" and n.func.attr == "to" and ast.unparse(n.args[0]) == "torch.float" for n in ast.walk(forward))
    result = {"status": "source_checks_pass", "implementation_hashes": sources, "python_files_parsed": len(list(HERE.glob("*.py"))), "source_snapshots_verified": len(pins["files"]), "no_combined_split_accessor_called": True, "build_reads_only_split_files": build_splits, "finalize_reads_split_files": finalize_splits, "loader_identity_checked_before_split_read": True, "checkpoint_validator_required_by_lock_consumer": True, "native_feature_cast_present": True, "family_cells": 15, "optimizer_fits": 24, "parameters_from_formula": expected, "matched_width": width, "torch_present": importlib.util.find_spec("torch") is not None, "numerical_qualification": "PENDING; requires test_cpu.py on a CPU runtime with the declared dependencies", "dataset_access": False, "gpu_execution": False, "remote_actions": False}
    if args.output:
        with Path(args.output).open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
