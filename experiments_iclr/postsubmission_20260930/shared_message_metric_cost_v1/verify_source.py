"""Stdlib author checks only; never execute Torch or calibration kernels."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def main():
    output = HERE / "SOURCE_CHECK_v1.json"
    if output.exists():
        raise ValueError("Preserve finalized source check")
    checks, hashes = [], {}
    for path in sorted(HERE.glob("*.py")):
        raw = path.read_bytes()
        compile(raw, str(path), "exec", dont_inherit=True)
        tree = ast.parse(raw)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        if path.name == "metric_cost.py":
            assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                ((getattr(n, "module", "") or "").split(".")[0] in {"torch", "numpy", "torch_geometric", "ogb"}
                 or any(a.name.split(".")[0] in {"torch", "numpy", "torch_geometric", "ogb"} for a in getattr(n, "names", [])))
                for n in tree.body)
    checks.append("syntax_compiles_no_top_level_scientific_import")
    path = HERE / "metric_cost.py"
    spec = importlib.util.spec_from_file_location("source_only_metric_cost_contract", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    namespace, extraction = module.extract_functions()
    assert set(r["function"] for r in extraction) == {"full_split", "from_statistics", "label_free_reductions"}
    assert all(callable(namespace[n]) for n in ("full_split", "from_statistics", "label_free_reductions"))
    checks.append("three_exact_source_whole_file_segment_and_AST_pins_compile_unchanged_definitions")
    assert module.SHAPES == {"Reddit2": dict(validation_nodes=23699, classes=41, member_counts=[1,4]),
                             "ogbn-products": dict(validation_nodes=39323, classes=47, member_counts=[1,4])}
    assert module.RUNTIME["CPU_threads"] == 4 and module.RUNTIME["torch_version"] == "2.1.2+cu118"
    checks.append("literal_complete_validation_shapes_member_counts_runtime_and_CPUthreads")
    with tempfile.TemporaryDirectory(prefix="STDLIB_REFUSAL_", dir=HERE) as temporary:
        request = Path(temporary) / "request.json"
        request.write_text(json.dumps(dict(schema="gnnm-shared-message-artificial-CPU-metric-request-v1",
                                          root_authorized_artificial_CPU_calibration=False)))
        try:
            module.admission(request, HERE / "not_created")
        except ValueError:
            pass
        else:
            raise AssertionError("Unapproved request was not refused")
    checks.append("unauthorized_request_refused_before_external_evidence_or_Torch")
    bounds = {t:{str(m):dict(worst_full_split_seconds=2.0) for m in [1,4]} for t in module.SHAPES}
    rows = [dict(task=t,arm=a,three_CPU_label_free_reduction_seconds=[1.0,0.9,0.8])
            for t in module.SHAPES for a in module.ARMS]
    charge = module.cpu_surcharge(bounds, rows)
    assert charge["total_seconds"] == 10*3*64*1.0*1.25
    for row in rows:
        row["three_CPU_label_free_reduction_seconds"] = [3.0,3.0,3.0]
    assert module.cpu_surcharge(bounds, rows)["total_seconds"] == 0
    checks.append("all10_cells_3seeds_64calls_125percent_excess_surcharge_negative_excess_zero")
    draft = json.loads((HERE / "DRAFT_REQUEST_v1.json").read_text())["request_payload"]
    assert draft["root_authorized_artificial_CPU_calibration"] is False and draft["source_manifest"] is None
    assert all(draft[k] is False for k in ("scientific_training_authorized", "real_data_or_heldout_targets_authorized", "GPU_operations_authorized", "quality_scores_or_retained_tensors_authorized"))
    assert draft["plan"] == module.PLAN and draft["helper_sha256"] == module.sha(path)
    checks.append("draft_closed_exact_helper_plan_and_false_scientific_heldout_GPU_permissions")
    text = path.read_text()
    assert text.index("sys.path[:]") < text.index("import argparse")
    assert "import torch" not in text[:text.index("def run(")]
    assert 'seed = plan["synthetic_seed"]' in text and "for _ in range(2)" in text and "for repetition in range(5)" in text
    assert '"agreeing_argmaxes", "different_argmaxes"' in text
    assert "validate_artificial(functions, member, pooled, targets, pattern)" in text
    assert "functions[\"from_statistics\"](counts)" in text
    assert 'outer.get("root_request") != reference(path)' in text and '"--require-idle-gpu"' in text
    assert "full_wall_and_chain_qualification_requires_successful_outer_terminal=True" in text
    assert "torch.save" not in text and "torch.load" not in text
    checks.append("exact_patterns_repetitions_CPU_synthetic_validation_no_tensor_IO_and_external_chain_gates")
    assert not any(n == "torch" or n.startswith(("torch.","numpy","torch_geometric","ogb")) for n in sys.modules)
    checks.append("author_executed_stdlib_only_no_calibration_or_scientific_workload")
    with output.open("x") as stream:
        json.dump(dict(schema="gnnm-metric-cost-stdlib-source-check-v1", status="passed_source_only_calibration_unexecuted",
            checks=checks, check_count=len(checks), python_source_sha256=hashes,
            Torch_data_labels_predictions_tensors_GPU_remote_or_calibration_accessed=False,
            original_source_function_bodies_modified=False, scientific_or_calibration_launch_authorized=False), stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(dict(check_count=len(checks), source_only=True, numerical_calibration_executed=False)))


if __name__ == "__main__":
    main()
