"""Bounded AST/source checks; imports no model library and runs no native work."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.with_name("runtime_v3")
PRODUCERS = ("run_fit.py", "adapters.py", "bookkeeping.py", "common.py", "controllers.py", "source_rng.py")


def tree(name, directory=HERE):
    return ast.parse((directory / name).read_text())


def dump(value):
    return ast.dump(value, include_attributes=False)


def statement(source):
    return ast.parse(source).body[0]


def remove_exact(body, expected):
    matching = [i for i, node in enumerate(body) if dump(node) == dump(expected)]
    assert len(matching) == 1, ast.unparse(expected)
    return body.pop(matching[0])


def method(module, cls, name):
    klass = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == cls)
    return next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)


def function(module, name):
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def assignment(node, name):
    return isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name


def main():
    checks = {}
    parent = json.loads((HERE / "PARENT_SOURCE.json").read_text())
    assert hashlib.sha256((PARENT / "MANIFEST.json").read_bytes()).hexdigest() == parent["manifest_sha256"]
    for pin in parent["files"]:
        data = (PARENT / pin["path"]).read_bytes()
        assert len(data) == pin["bytes"] and hashlib.sha256(data).hexdigest() == pin["sha256"]
    checks["parent_source_receipts_exact"] = True
    unchanged = [name for name in PRODUCERS if (HERE / name).read_bytes() == (PARENT / name).read_bytes()]
    assert set(unchanged) == {"bookkeeping.py", "common.py", "controllers.py", "source_rng.py"}
    checks["unchanged_producer_files_byte_identical"] = unchanged
    for name in PRODUCERS:
        compile((HERE / name).read_text(), str(HERE / name), "exec")
    checks["six_producer_files_compile_without_execution"] = True

    old, new = tree("run_fit.py", PARENT), tree("run_fit.py")
    original_new = copy.deepcopy(new)
    old_run, new_run = function(old, "run"), function(new, "run")
    old_try = next(n for n in old_run.body if isinstance(n, ast.Try))
    new_try = next(n for n in new_run.body if isinstance(n, ast.Try))
    old_loop = next(n for n in old_try.body if isinstance(n, ast.While))
    new_loop = next(n for n in new_try.body if isinstance(n, ast.While))
    report = statement("result.update(completed_epochs=selector.completed_epoch, selected_epoch=selector.selected_epoch, best_validation_metric=selector.best_metric, training_epochs_this_invocation=selector.completed_epoch-completed_at_invocation_start)")
    i = next(i for i, n in enumerate(new_loop.body) if dump(n) == dump(report))
    assert assignment(new_loop.body[i - 1], "decision") and assignment(new_loop.body[i + 1], "payload")
    remove_exact(old_loop.body, report)
    remove_exact(new_loop.body, report)
    checks["reporting_update_exact_original_statement_moved_immediately_after_selector_commit"] = True
    exporter_index = next(i for i, n in enumerate(new_try.body) if assignment(n, "mean_exporter"))
    exporter_assignment, exporter_if = new_try.body[exporter_index:exporter_index + 2]
    assert dump(exporter_assignment) == dump(statement('mean_exporter = getattr(adapter, "selected_mean_export", None)'))
    expected_exporter = statement('''if mean_exporter is not None:
    mean_export = mean_exporter(final_valid, job_hash=job_hash,
        selected_epoch=selected["selector"]["selected_epoch"],
        selected_checkpoint=export["selected_checkpoint"], runtime_binding=runtime_binding)
    if mean_export is not None:
        export["same_bank_mean_VALID"] = mean_export
''')
    assert dump(exporter_if) == dump(expected_exporter)
    assert assignment(new_try.body[exporter_index - 1], "export")
    del new_try.body[exporter_index:exporter_index + 2]
    assert dump(old) == dump(new)
    checks["run_fit_entire_remaining_AST_identical_after_two_declared_changes"] = True
    finalizer = next(n for n in function(original_new, "run").body if isinstance(n, ast.Try)).body
    sequence = [ast.unparse(n) for n in finalizer]
    load_index = next(i for i, s in enumerate(sequence) if s == "adapter.load_state(selected['adapter'])")
    validate_index = next(i for i, n in enumerate(finalizer) if assignment(n, "final_valid"))
    export_index = next(i for i, n in enumerate(finalizer) if assignment(n, "mean_exporter"))
    assert load_index < validate_index < export_index
    checks["selected_mean_export_occurs_after_own_selected_checkpoint_load_and_actual_VALID"] = True

    old, new = tree("adapters.py", PARENT), tree("adapters.py")
    original = copy.deepcopy(new)
    for import_node in new.body:
        if isinstance(import_node, ast.ImportFrom) and import_node.module == "common":
            import_node.names = [n for n in import_node.names if n.name not in ("cpu_clone", "digest")]
    base_init = method(new, "BaseAdapter", "__init__")
    for line in ("self._job_authority = copy.deepcopy(job)", "self._validation_generation = self._validation_serial = 0", "self._same_bank_mean_cache = None"):
        remove_exact(base_init.body, statement(line))
    remove_exact(method(new, "BaseAdapter", "load_state").body, statement("self._invalidate_validation()"))
    node_init = method(new, "NodeAdapter", "__init__")
    remove_exact(node_init.body, statement("original_job_authority = copy.deepcopy(job)"))
    remove_exact(node_init.body, statement("self._job_authority = original_job_authority"))
    for cls in ("NodeAdapter", "OGBAdapter"):
        for name in ("train_epoch", "train_diagnostic"):
            body = method(new, cls, name).body
            first = 1 if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str) else 0
            assert dump(body[first]) == dump(statement("self._invalidate_validation()"))
            body.pop(first)
        body = method(new, cls, "validate").body
        assert dump(body[0]) == dump(statement("self._same_bank_mean_cache = None"))
        body.pop(0)
        native_calls = [n for n in ast.walk(method(original, cls, "validate")) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.session.validate"]
        assert len(native_calls) == 1
        assert dump(body[-2]) == dump(statement("self._capture_same_call_mean(result, value)"))
        assert dump(body[-1]) == dump(statement("return value"))
        assert assignment(body[-3], "value") and isinstance(body[-3].value, ast.Dict)
        body[-3:] = [ast.Return(value=body[-3].value)]
    helpers = ("_invalidate_validation", "_tensor_identity", "_capture_same_call_mean", "selected_mean_export")
    for name in helpers:
        helper = method(original, "BaseAdapter", name)
        for node in ast.walk(helper):
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                assert not target.startswith(("self.session.", "self.model", "self.optimizers", "torch.cuda")), target
                assert not any(term in target for term in ("manual_seed", "get_rng_state", "set_rng_state", ".backward", ".step")), target
    for cls in new.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "BaseAdapter":
            cls.body = [n for n in cls.body if not isinstance(n, ast.FunctionDef) or n.name not in helpers]
    assert dump(old) == dump(new)
    checks["adapter_entire_remaining_AST_identical_after_declared_cache_and_export_additions"] = True
    checks["TRAIN_and_TRAIN_diagnostic_AST_identical_after_leading_cache_invalidation"] = True
    checks["each_Node_OGB_validator_calls_native_validate_once"] = True
    checks["four_new_helpers_have_no_native_session_model_optimizer_or_RNG_calls"] = True
    checks["nonzero_node_split_keeps_original_immutable_JOB_authority"] = True
    result = {"schema": "runtime-v4-static-contract-checks-v1", "status": "passed", "checks": checks,
        "native_constructions": 0, "native_forward_calls": 0, "optimizer_steps": 0, "TEST_accessed": False,
        "native_v4_lifecycle_qualification": "pending; static/source checks do not qualify stop/resume/finalization"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
