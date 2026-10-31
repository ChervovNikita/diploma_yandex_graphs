"""Stdlib syntax/structure/refusal audit only. Never imports Torch or data."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def main():
    output = HERE / "STATIC_SOURCE_CHECK_v1.json"
    if output.exists():
        raise ValueError("Preserve previous audit; use a new explicit generation")
    checks, source_sha = [], {}
    trees = {}
    for path in sorted(HERE.glob("*.py")):
        raw = path.read_bytes()
        compile(raw, str(path), "exec", dont_inherit=True)
        trees[path.name] = ast.parse(raw)
        source_sha[path.name] = hashlib.sha256(raw).hexdigest()
    checks.append({"check": "all_python_syntax_without_execution_or_bytecode", "passed": True})
    for name in ("contract.py", "core.py", "metrics.py", "evaluate.py", "artifact_plan.py"):
        for node in trees[name].body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                if module and module.split(".")[0] in ("torch", "numpy", "torch_geometric", "ogb"):
                    raise ValueError("Third-party import before admission: " + name)
    checks.append({"check": "scientific_modules_no_top_level_third_party_imports", "passed": True})
    source = (HERE / "core.py").read_text()
    if "from contract import" in source or "from labels import" in source or "from metrics import" in source:
        raise ValueError("Unqualified scientific module imports can collide")
    checks.append({"check": "scientific_imports_package_qualified", "passed": True})
    assert "inference_validation_work" not in source
    assert "split_forward(model, family, features, validation_ids" in source
    checks.append({"check": "one_validation_forward_and_no_profiler_proxy_duplicate", "passed": True})
    assert "optimizer = torch.optim.AdamW" in source and "if optimizer.state:" in source
    assert "range(1, 101)" in source and "PCG64_schedule_final_state" in source
    assert "training_layout_state" in source and "canonical_state" in source
    checks.append({"check": "fresh_optimizer_all100_epoch_state_and_rng_evidence", "passed": True})
    evaluator = (HERE / "evaluate.py").read_text()
    assert evaluator.count("measured = full_split(member, targets,") == 1
    assert "raw_sample(model, family, features" in evaluator
    assert "PropagateTimer" not in evaluator
    assert "PropagationTimer" in evaluator and "register_propagate_forward_pre_hook" in evaluator
    checks.append({"check": "one_final_cell_test_metric_and_separate_component_instrumentation", "passed": True})
    assert "all100_stored_metrics_replayed" in source and "collect_choices" in source
    assert "test_label_request_template" in source and "family_lock_payload_sha256" in source
    checks.append({"check": "complete30_lock_cycle_free_requests_and_replay_prerequisites_present", "passed": True})
    spec = importlib.util.spec_from_file_location("prospective_scientific_stdlib_contract", HERE / "contract.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    refusals = []
    def refuses(name, call):
        try:
            call()
        except (ValueError, RuntimeError, FileNotFoundError):
            refusals.append({"check": name, "passed": True})
            return
        raise ValueError("Expected refusal did not occur: " + name)
    refuses("outside_phase_path_rejected_without_file_access", lambda: contract.within(PHASE.parent / "not_in_phase"))
    refuses("unlisted_cell_rejected", lambda: contract.cell_id({"task": "Reddit2", "seed": 3, "arm": "candidate"}))
    refuses("noncanonical_reference_rejected_before_hash", lambda: contract.checked(
        {"path": "propagation_predictive_v1/../propagation_predictive_v1/contract.py", "sha256": "0" * 64}))
    with patch.object(contract.json, "loads", return_value={"schema": "STATIC_NEGATIVE_NOT_AN_ADMISSION"}):
        refuses("unqualified_method_schema_rejected_before_dependency_or_runtime_import", lambda: contract.binding(HERE / "contract.py"))
    with patch.object(contract.os.environ, "get", return_value="STATIC_NEGATIVE_UNFROZEN"):
        refuses("unfrozen_runtime_rejected_before_Torch", lambda: contract.runtime_before_import(
            {"runtime_policy": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True}}))
    source_path = HERE / "contract.py"
    first = contract.sha(source_path)
    assert contract.sha(source_path) == first and contract.hash_cache_report()["unchanged_signature_reuses"] >= 1
    stat = source_path.stat()
    original_stat = Path.stat
    changed = SimpleNamespace(st_dev=stat.st_dev, st_ino=stat.st_ino, st_size=stat.st_size,
                             st_mtime_ns=stat.st_mtime_ns + 1, st_ctime_ns=stat.st_ctime_ns)
    def simulated_stat(self, *args, **kwargs):
        return changed if self == source_path else original_stat(self, *args, **kwargs)
    with patch.object(Path, "stat", simulated_stat):
        refuses("changed_verified_file_signature_rejected_without_mutating_source", lambda: contract.sha(source_path))
    if any(name == "torch" or name.startswith(("torch.", "numpy", "ogb", "torch_geometric")) for name in sys.modules):
        raise ValueError("Static audit imported a forbidden third-party scientific package")
    with output.open("x") as stream:
        json.dump({"schema": "gnnm-propagation-scientific-source-stdlib-check-v1",
            "status": "source_checks_passed_no_numerical_execution", "checks": checks + refusals,
            "check_count": len(checks) + len(refusals), "python_source_sha256": source_sha,
            "Torch_CUDA_or_remote_executed": False, "dataset_targets_predictions_or_existing_results_read": False,
            "numerical_runtime_verified": False, "scientific_admission_issued": False}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"static_check": str(output), "checks": len(checks) + len(refusals)}))


if __name__ == "__main__":
    main()
