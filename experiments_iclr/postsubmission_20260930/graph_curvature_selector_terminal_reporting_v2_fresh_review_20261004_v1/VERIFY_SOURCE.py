"""Independent stdlib-only source custody and AST checks; no numerical imports."""
from pathlib import Path
import ast
from datetime import datetime, timezone
import hashlib
import json

PHASE = Path(__file__).resolve().parent.parent
SOURCE = PHASE / "graph_curvature_selector_terminal_reporting_source_20261004_v2"
EXPECTED_MANIFEST = "faace8c4ba59e3233a92809c3052c3dc02b37ddf77159c43e3573360818fbdf1"

def observed(path):
    data = path.read_bytes()
    return dict(path=str(path.relative_to(PHASE)), bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(), permissions=oct(path.stat().st_mode & 0o777))

def verify(base, record):
    path = base / record["path"]
    result = observed(path)
    assert result["bytes"] == record["bytes"] and result["sha256"] == record["sha256"], str(path)
    return result

manifest_observed = observed(SOURCE / "MANIFEST.json")
assert manifest_observed["sha256"] == EXPECTED_MANIFEST
manifest = json.loads((SOURCE / "MANIFEST.json").read_text())
seal = json.loads((SOURCE / "SEAL.json").read_text())
assert seal["manifest_sha256"] == EXPECTED_MANIFEST
payloads = [verify(SOURCE, record) for record in manifest["files"]]
bindings = json.loads((SOURCE / "SOURCE_BINDINGS.json").read_text())
records = [bindings["original_integration"], bindings["reporting_integration"],
    *bindings["frozen_source_records"], bindings["failure_review"],
    bindings["preserved_v1_manifest"], bindings["preserved_v1_seal"]]
bound_records = [verify(PHASE, record) for record in records]
v3 = json.loads((PHASE / "graph_curvature_selector_source_preparation_20261004_v3/SOURCE_BINDINGS.json").read_text())
indirect_runtime = [verify(PHASE, record) for record in [v3["native_adapter"],
    *v3["native_source_dependencies"], *v3["runtime_modules"].values()]]
canonical = ast.parse((PHASE / bindings["original_integration"]["path"]).read_text())
reporting = ast.parse((PHASE / bindings["reporting_integration"]["path"]).read_text())
fn = next(n for n in reporting.body if isinstance(n, ast.FunctionDef) and n.name == "continuation")
assert [arg.arg for arg in fn.args.kwonlyargs] == ["terminal_observer", "terminal_context"]
assert all(isinstance(n, ast.Constant) and n.value is None for n in fn.args.kw_defaults)
fn.args.kwonlyargs, fn.args.kw_defaults = [], []
hooks = [n for n in fn.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
    and isinstance(n.test.left, ast.Name) and n.test.left.id == "terminal_observer"]
assert len(hooks) == 1
hook = hooks[0]
following = fn.body[fn.body.index(hook)+1]
assert isinstance(following, ast.Expr) and isinstance(following.value, ast.Call)
assert ast.unparse(following.value.func) == "raw.load_state_dict"
fn.body.remove(hook)
assert ast.dump(canonical, include_attributes=False) == ast.dump(reporting, include_attributes=False)
reporter_ast = ast.parse((SOURCE / "terminal_reporting.py").read_text())
assert not any(isinstance(n, ast.Call) and (isinstance(n.func, ast.Name) and n.func.id in {"evaluate", "train_update"}
    or isinstance(n.func, ast.Attribute) and n.func.attr == "forward") for n in ast.walk(hook))
assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "terminal_observer"
    for n in ast.walk(hook)) == 1
result = dict(schema="graph-curvature-terminal-reporting-independent-source-checks-v1",
    checked_UTC=datetime.now(timezone.utc).isoformat(), source_manifest=manifest_observed,
    source_seal=observed(SOURCE / "SEAL.json"), source_directory_permissions=oct(SOURCE.stat().st_mode & 0o777),
    payloads_verified=payloads, directly_bound_records_verified=bound_records,
    indirect_native_runtime_source_records_verified=indirect_runtime,
    complete_canonical_module_AST_identical_after_optional_args_and_single_hook_removal=True,
    hook_immediately_before_selected_model_load=True,
    hook_observer_call_count=1, hook_direct_model_evaluate_train_forward_calls=0,
    numerical_packages_imported=False, prepared_modules_imported_or_executed=False,
    numerical_runtime_tests_performed=False, arrays_checkpoints_results_opened=False,
    servers_accessed=False, canonical_or_reporting_source_edited=False)
(Path(__file__).resolve().parent / "SOURCE_CHECKS.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
print(json.dumps(dict(source_manifest_sha256=EXPECTED_MANIFEST, payloads_verified=len(payloads),
    directly_bound_records_verified=len(bound_records), indirect_native_runtime_sources_verified=len(indirect_runtime),
    canonical_AST_equality=True, numerical_runtime_tests_performed=False), indent=2))
