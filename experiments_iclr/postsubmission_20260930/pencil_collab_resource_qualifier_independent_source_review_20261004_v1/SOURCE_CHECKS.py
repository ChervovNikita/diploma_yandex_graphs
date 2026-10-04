"""Independent local source/metadata checks. No target code or array imports."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat

REVIEW = Path(__file__).resolve().parent
PHASE = REVIEW.parent
SOURCE = PHASE / "pencil_collab_resource_qualifier_preparation_20261004_v1"
EXPECTED_MANIFEST = "340a0d46e533d62099508bf40d0c0367f8871ef4d5947e6fd7d33c88ea2f079f"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def check_row(root, row):
    path = root / row["path"]
    assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
    data = path.read_bytes()
    assert len(data) == row["bytes"] and digest(data) == row["sha256"], str(path)
    return {"path": row["path"], "bytes": len(data), "sha256": digest(data)}


def functions(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = call_name(node.value)
        return prefix + "." + node.attr if prefix else node.attr
    return ""


def main():
    manifest_bytes = (SOURCE / "MANIFEST.json").read_bytes()
    assert digest(manifest_bytes) == EXPECTED_MANIFEST
    manifest = json.loads(manifest_bytes)
    payload = [check_row(SOURCE, row) for row in manifest["files"]]
    assert len(payload) == manifest["payload_files"] == 59
    assert sum(r["bytes"] for r in payload) == manifest["payload_bytes"] == 622349
    actual_files = {str(p.relative_to(SOURCE)) for p in SOURCE.rglob("*") if p.is_file()}
    assert actual_files == {r["path"] for r in payload} | {"MANIFEST.json", "SEAL.json"}
    seal = json.loads((SOURCE / "SEAL.json").read_text())
    assert seal["manifest_sha256"] == EXPECTED_MANIFEST
    source_modes = {str(p.relative_to(SOURCE)): oct(stat.S_IMODE(p.stat().st_mode)) for p in SOURCE.rglob("*")}
    assert stat.S_IMODE(SOURCE.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(p.stat().st_mode) == (0o555 if p.is_dir() else 0o444) for p in SOURCE.rglob("*"))
    py_files = sorted(SOURCE.rglob("*.py"))
    for path in py_files:
        ast.parse(path.read_text(), filename=str(path))
    native_rows = json.loads((SOURCE / "NATIVE_SOURCE_BINDINGS.json").read_text())["files"]
    native = []
    for row in native_rows:
        observed = check_row(SOURCE, row)
        data = (SOURCE / row["path"]).read_bytes()
        git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert git_blob == row["git_blob_sha1"]
        origin = PHASE / row["origin"]
        assert origin.resolve().is_relative_to(PHASE.resolve())
        assert origin.read_bytes() == data
        native.append({**observed, "git_blob_sha1": git_blob, "origin": row["origin"], "origin_byte_identical": True})
    inputs = [check_row(PHASE, row) for row in json.loads((SOURCE / "INPUT_BINDINGS.json").read_text())["inputs"]]
    inherited = functions(PHASE / "graph_count_conditioned_pattern_minimal_gradient_preparation_20261004_v4/supervise.py")
    candidate = functions(SOURCE / "supervise.py")
    names = ("fsync_directory", "durable_write", "process_identity", "members_of_session", "held_identity", "kill_owned", "read_json", "inventory")
    for name in names:
        assert ast.dump(inherited[name], include_attributes=False) == ast.dump(candidate[name], include_attributes=False)
    tree = ast.parse((SOURCE / "worker.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    constructors = []
    helper_calls = {}
    for node in calls:
        name = call_name(node.func)
        if name == "ShaDowKHopSeqFromEdgesMapDataset":
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if isinstance(kw.value, ast.Constant)}
            constructors.append({"line": node.lineno, **args})
        if name in ("run_lp.train_loop", "run_lp.evaluate_loop", "run_lp.build_loaders", "run_lp.get_model"):
            helper_calls[name] = {"line": node.lineno, "keywords": {kw.arg: ast.unparse(kw.value) for kw in node.keywords}}
        assert name not in ("run_lp.main", "run_lp.save_checkpoint", "torch.save")
        assert not name.endswith(".get_edge_split")
    assert [r["data_split"] for r in constructors] == ["train", "valid"]
    assert helper_calls["run_lp.build_loaders"]["keywords"]["test_dataset_raw"] == "None"
    assert helper_calls["run_lp.train_loop"]["keywords"]["max_num_samples"] == "-1"
    assert helper_calls["run_lp.train_loop"]["keywords"]["gradient_accumulation_steps"] == "8"
    for key, value in {"evaluator": "None", "compute_loss": "False", "check_sequential_indices": "True", "show_progress": "False"}.items():
        assert helper_calls["run_lp.evaluate_loop"]["keywords"][key] == value
    install = json.loads((SOURCE / "DEPENDENCY_INSTALL_PLAN.json").read_text())
    requirements = (SOURCE / "dependency_requirements.txt").read_bytes()
    assert digest(requirements) == install["requirements_sha256"]
    assert install["new_packages_count"] == len(install["new_packages"]) == len(requirements.splitlines()) == 16
    assert {"--no-deps", "--require-hashes", "--no-compile", "--no-user"} <= set(install["proposed_install_argv"])
    assert "--upgrade" not in install["proposed_install_argv"]
    forbidden = {"torch", "numpy", "torch-geometric", "torch-sparse", "torch-scatter"}
    for row in install["new_packages"]:
        name = row["name"].lower().replace("_", "-")
        assert name not in forbidden and not name.startswith(("nvidia-", "cuda"))
    for name in ("ROOT_RELEASE_TEMPLATE.json", "DEPENDENCY_ADMISSION_TEMPLATE.json"):
        assert json.loads((SOURCE / name).read_text())["status"] not in ("APPROVED", "ROOT_ADMITTED_FOR_RESOURCE_ONLY")
    result = {
        "schema": "independent-pencil-source-checks-v1",
        "UTC": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_STATIC_INTEGRITY_AND_SELECTED_CALL_CONTRACTS_ONLY",
        "candidate_manifest_sha256": EXPECTED_MANIFEST,
        "payload_files": payload,
        "payload_file_count": len(payload),
        "payload_bytes": sum(r["bytes"] for r in payload),
        "exact_file_set": True,
        "sealed_source_modes": source_modes,
        "AST_parsed_files": len(py_files),
        "native_files": native,
        "bound_external_input_files": inputs,
        "unchanged_supervisor_function_ASTs": list(names),
        "native_dataset_constructors": constructors,
        "native_helper_call_contracts": helper_calls,
        "hash_pinned_new_wheel_count": 16,
        "candidate_templates_disabled": True,
        "target_code_executed": False,
        "numerical_modules_imported": False,
        "graph_or_split_arrays_opened": False,
        "network_requests": False,
        "candidate_modified": False,
        "limitation": "Static hash/AST checks do not resolve the separate-session torchrun ownership blocker or establish runtime/numerical compatibility."
    }
    (REVIEW / "SOURCE_CHECKS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "candidate_manifest_sha256", "payload_file_count", "payload_bytes", "AST_parsed_files")}, sort_keys=True))


if __name__ == "__main__":
    main()
