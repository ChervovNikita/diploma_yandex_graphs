"""Stdlib negative boundary checks; never imports runners or graph libraries."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prospective_shareable_guard", ROOT / "shareable_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def rejects(value, dataset):
    try:
        guard.require_shareable_manifest(value, dataset)
    except ValueError:
        return
    raise AssertionError("Expected rejection")


def main():
    phase = ROOT.parent
    checks = []
    for dataset in ("molhiv", "mnist_superpixels"):
        path = phase / "data_freezes" / dataset / "frozen_shareable_manifest_v2.json"
        original = json.loads(path.read_text())
        guard.require_shareable_manifest(original, dataset)
        for key in ("scoring_performed", "confirmation_data_accessed"):
            changed = copy.deepcopy(original); changed[key] = True; rejects(changed, dataset)
            changed = copy.deepcopy(original); changed.pop(key); rejects(changed, dataset)
        for key in ("class_counts", "label_summary", "targets"):
            changed = copy.deepcopy(original); changed["split_summaries"]["test"][key] = {}; rejects(changed, dataset)
        changed = copy.deepcopy(original)
        changed["split_summaries"]["test"]["graph_tensor_fingerprints"][0]["y"] = {"sha256": "0" * 64}
        rejects(changed, dataset)
        for key, value in (("individual_test_target_hashes_removed", 0), ("input_data_changed", True),
                           ("predictive_scores_read", True), ("parent_manifest_sha256", "invalid")):
            changed = copy.deepcopy(original); changed["manifest_amendment"][key] = value; rejects(changed, dataset)
        changed = copy.deepcopy(original)
        changed["split_summaries"]["test"]["canonical_split_sha256"] = "0" * 64
        rejects(changed, dataset)
        checks.append(dataset + ": accepted existing shareable; rejected13 boundary mutations")
    for filename in ("molhiv_pilot.py", "mnist_pilot.py", "verify_molhiv_pilot.py", "shareable_guard.py"):
        path = ROOT / filename; tree = ast.parse(path.read_text())
        if filename.endswith("pilot.py") and not filename.startswith("verify"):
            verify = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_data")
            assert "require_shareable_manifest" in ast.unparse(verify)
            source = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "source_paths")
            assert "shareable_guard.py" in ast.unparse(source)
        checks.append(filename + ": syntax/source binding inspected")
    print(json.dumps({"status": "passed", "checks": checks,
                      "guard_sha256": hashlib.sha256((ROOT / "shareable_guard.py").read_bytes()).hexdigest(),
                      "datasets_loaded": False, "graph_libraries_imported": False,
                      "server_accessed": False, "CUDA_context_created": False, "predictive_scores_computed": False}, indent=2))


if __name__ == "__main__":
    main()
