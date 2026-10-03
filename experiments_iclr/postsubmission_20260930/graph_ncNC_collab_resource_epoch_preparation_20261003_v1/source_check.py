"""Stdlib-only source/metadata checks; never import a numerical module."""
import sys
sys.dont_write_bytecode = True
import argparse
import ast
import hashlib
import json
from pathlib import Path
from guards import HERE, read_json, require, verify_preparation, verify_external_sources, file_sha, write_json

MODULES = {"guards.py", "train_only_data.py", "runtime.py", "gpu_parity.py", "resource_epoch.py", "run.py", "source_check.py"}
FORBIDDEN_IMPORTS = {"subprocess", "socket", "requests", "urllib", "paramiko", "ogb", "sklearn"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-seal", action="store_true")
    args = parser.parse_args()
    checks = []
    actual = {path.name for path in HERE.glob("*.py")}
    require(actual == MODULES, "Prepared module inventory differs")
    trees, hashes = {}, {}
    for name in sorted(MODULES):
        source = (HERE / name).read_text()
        tree = ast.parse(source, filename=name)
        compile(tree, name, "exec")  # Syntax compilation only, never exec/import.
        trees[name], hashes[name] = tree, file_sha(HERE / name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            require(not names & FORBIDDEN_IMPORTS, "Forbidden acquisition/heldout/remote import: " + name)
    checks.append("all_seven_modules_parse_compile_without_numerical_import")
    top_imports = []
    for node in trees["run.py"].body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module)
    require(set(top_imports) <= {"argparse", "datetime", "sys", "time", "guards"}, "Entry point imports numerical module before admission")
    main_tree = next(node for node in trees["run.py"].body if isinstance(node, ast.FunctionDef) and node.name == "main")
    admission_line = min(node.lineno for node in ast.walk(main_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "preflight")
    numerical_line = min(node.lineno for node in ast.walk(main_tree) if isinstance(node, ast.ImportFrom) and node.module == "runtime")
    require(admission_line < numerical_line, "Numerical imports precede standard-library admission")
    checks.append("stdlib_root_admission_precedes_Torch_models_and_data")
    for name, tree in trees.items():
        if name == "source_check.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                require(node.attr not in {"get_edge_split", "download", "urlopen", "system", "Popen"}, "Forbidden payload/remote API: " + name)
    load_references = [(name, node) for name, tree in trees.items() for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr == "load"]
    require(len(load_references) == 1 and load_references[0][0] == "train_only_data.py", "Dataset torch.load scope differs")
    data_source = (HERE / "train_only_data.py").read_text()
    require('root / "split/time/train.pt"' in data_source and 'weights_only=True' in data_source and 'map_location="cpu"' in data_source, "Authenticated direct TRAIN loader changed")
    require('np.repeat(raw_pairs, 2, axis=1)' in data_source, "Original reciprocal interleaving missing")
    require('sampler, data["raw_edge_index"], len(data["x"])' in data_source, "Native raw/default sampler call changed")
    require('native_utils.PermIterator' in data_source, "Native dropped-tail iterator missing")
    checks.append("exact_TRAIN_raw_feature_graph_allowlist_and_native_stream_route")
    bindings = read_json(HERE / "BINDINGS.json")
    metadata = read_json(HERE / "custody_metadata/EXISTING_DATA_QUALIFICATION.json")
    expected_allowed = {"split/time/train.pt", "raw/node-feat.csv.gz", "raw/edge.csv.gz"}
    require(set(bindings["allowed_data_files"]) == expected_allowed, "Custody allowlist differs")
    staged = {entry["archive_member"].removeprefix("collab/"): entry for entry in metadata["staged_members"]}
    for name in expected_allowed:
        for key, value in bindings["allowed_data_files"][name].items():
            require(staged[name][key] == value, "Custody metadata file digest differs")
    require(bindings["data_tensor_digests"] == {"train_records": metadata["train_topology"]["train_positive_sha256"],
        "raw_features": metadata["public_graph"]["raw_features_tensor_sha256"], "ordered_raw_graph": metadata["public_graph"]["ordered_graph_edge_sha256"]}, "Tensor metadata authority differs")
    for pin in bindings["metadata_recovery"]["decoded_only_exact_metadata"]:
        require(file_sha(HERE / pin["path"]) == pin["sha256"], "Decoded exact metadata bytes differ")
    prototype = verify_external_sources(args.research_root, bindings)
    checks.append("immutable_prototype_native_review_CPU_QA_and_analytic_oracles_verified")
    records, batch = bindings["train_positive_records"], bindings["native_train_batch_size"]
    require(records == 1179052 and batch == 65536 and records // batch == 17 and records % batch == 64940, "Native batch/tail arithmetic differs")
    work_node = next(node for node in trees["resource_epoch.py"].body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "EXPECTED_BATCH_WORK" for target in node.targets))
    work = ast.literal_eval(work_node.value)
    require(work == {"encoder_calls": 1, "outer_enumeration_calls": 2, "recursive_enumeration_calls": 16,
        "outer_full_node_xlin_calls": 8, "recursive_full_node_xlin_calls": 16,
        "outer_full_node_xlin_linear_map_calls": 16, "recursive_full_node_xlin_linear_map_calls": 32,
        "recursive_score_calls": 16, "completion_clamp_calls": 16, "outer_aggregation_calls": 24,
        "recursive_aggregation_calls": 16, "outer_nonlinear_decode_calls": 8, "recursive_nonlinear_decode_calls": 16}, "Complete member schedule work declaration changed")
    require('for training in (False, True)' in (HERE / "gpu_parity.py").read_text(), "GPU evaluation/training profiles missing")
    require('128 * torch.finfo(torch.float32).eps' in (HERE / "gpu_parity.py").read_text(), "Inherited arithmetic rule changed")
    checks.append("native17_batches_tail64940_and_complete_four_member_work_map_bound")
    require(not {"torch", "numpy", "pandas", "torch_sparse", "torch_scatter"} & set(sys.modules), "Source check imported numerical runtime")
    manifest_sha = None
    if args.require_seal or (HERE / "MANIFEST.json").exists():
        manifest_sha = verify_preparation()
        checks.append("new_preparation_manifest_verified")
    result = {"schema": "ncnc-collab-TRAIN-resource-stdlib-source-check-v1", "status": "SOURCE_METADATA_CHECKS_PASSED",
        "checks": checks, "public_module_sha256": hashes, "preparation_manifest_sha256": manifest_sha,
        "prototype_manifest_sha256": bindings["prototype_manifest_sha256"], "prototype_directory": str(prototype),
        "numerical_runtime_imported": False, "actual_data_tensor_or_labels_accessed": False,
        "GPU_or_remote_execution": False, "GPU_parity_or_resource_feasibility_claim": False}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
