"""Stdlib-only preparation checks; never import or execute prepared modules."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
bindings=json.loads((HERE/"SOURCE_BINDINGS.json").read_text())
verified=[]
for record in bindings.values():
    if isinstance(record,dict) and {"path","bytes","sha256"} <= record.keys():
        path=PHASE/record["path"]
        data=path.read_bytes()
        assert len(data)==record["bytes"] and hashlib.sha256(data).hexdigest()==record["sha256"], str(path)
        verified.append(record)
source=HERE/"boundary_qualification.py"
tree=ast.parse(source.read_text(),filename=str(source))
compile(tree,str(source),"exec")
imports=[]
for node in tree.body:
    if isinstance(node,ast.Import):
        imports.extend(alias.name for alias in node.names)
    elif isinstance(node,ast.ImportFrom):
        imports.append(node.module)
assert not any(name.split(".")[0] in {"numpy","torch","scipy","torch_geometric"} for name in imports)
assert not any(isinstance(node,ast.If) and any(isinstance(child,ast.Call) and isinstance(child.func,ast.Name)
    and child.func.id=="run_once" for child in ast.walk(node)) for node in tree.body)
run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="run_once")
primary_calls=[node for node in ast.walk(run) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
    and node.func.attr=="call" and len(node.args)>1 and isinstance(node.args[1],ast.Attribute)
    and node.args[1].attr=="continuation"]
assert len(primary_calls)==1
assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {"warm_native","train_update"}
    for node in ast.walk(tree))
canonical=ast.parse((PHASE/bindings["canonical_integration"]["path"]).read_bytes())
reporting=ast.parse((PHASE/bindings["reporting_integration"]["path"]).read_bytes())
fn=next(node for node in reporting.body if isinstance(node,ast.FunctionDef) and node.name=="continuation")
hook=next(node for node in fn.body if isinstance(node,ast.If) and ast.unparse(node.test)=="terminal_observer is not None")
tail=fn.body[fn.body.index(hook)+1:]
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="train_update"
    for statement in [hook,*tail] for n in ast.walk(statement))
assert ast.unparse(tail[0].value.func)=="raw.load_state_dict"
assert [arg.arg for arg in fn.args.kwonlyargs]==["terminal_observer","terminal_context"]
fn.args.kwonlyargs,fn.args.kw_defaults=[],[]
fn.body.remove(hook)
assert ast.dump(canonical,include_attributes=False)==ast.dump(reporting,include_attributes=False)
result=dict(schema="terminal-reporting-boundary-qualification-preparation-checks-v1",
    checked_UTC=datetime.now(timezone.utc).isoformat(), verified_bound_source_and_review_records=verified,
    prepared_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    prepared_source_bytes=len(source.read_bytes()), syntax_compile_only=True,
    numerical_imports_at_module_scope=False, source_module_imported_or_executed=False,
    native_runtime_or_exception_tests_performed=False, single_primary_continuation_call_in_run_once=True,
    isolated_hook_and_selected_tail_contain_no_training_update=True,
    canonical_AST_preserved_outside_hook=True,
    isolated_failure_cases_declared=15, additional_native_evaluations_declared=4,
    additional_training_updates_declared=0, exact_captured_preprocessing_required=True,
    arrays_checkpoints_results_opened=False, servers_accessed=False,
    canonical_or_reporting_source_modified=False, scientific_result=False,
    limits="AST/source checks cannot validate state copying, tracing, device runtime, exception paths or metric arithmetic")
(HERE/"PREPARATION_CHECKS.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps({key:result[key] for key in ["syntax_compile_only","source_module_imported_or_executed",
    "native_runtime_or_exception_tests_performed","single_primary_continuation_call_in_run_once",
    "isolated_hook_and_selected_tail_contain_no_training_update","exact_captured_preprocessing_required"]},indent=2))
