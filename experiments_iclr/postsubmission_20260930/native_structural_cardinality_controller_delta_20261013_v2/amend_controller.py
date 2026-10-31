"""Narrow metadata/fit-clock/source validation amendment; no parent overwrite."""
import ast,copy,hashlib,json
from pathlib import Path

def function_hashes(raw):
 return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}

def amend(path):
 path=Path(path);proof={}
 edits={
 'compile_master_plan.py':("    if 'private_response_source_contract' in config:","    if any(k in config for k in ('structural_source_contract','cardinality_source_contract','cardinality_control_contract','conditional_source_contract')):\n        from structural_cardinality_semantics import validate_run\n        return validate_run(run)\n    if 'private_response_source_contract' in config:",'fit_semantics'),
 'spec.py':('            budget = run["budget"]','            if any(k in run["config"] for k in ("structural_source_contract","cardinality_source_contract","cardinality_control_contract","conditional_source_contract")):\n                from structural_cardinality_semantics import validate_run\n                validate_run(run)\n            budget = run["budget"]','validate'),
 'native_stopping.py':("def require_completion(run):\n","def require_completion(run):\n if any(k in run['config'] for k in ('structural_source_contract','cardinality_source_contract','cardinality_control_contract','conditional_source_contract')):\n  from structural_cardinality_semantics import validate_run\n  validate_run(run)\n",'require_completion')}
 for name,(old,new,allowed) in edits.items():
  q=path/name;before=q.read_text()
  if before.count(old)!=1:raise ValueError('Unique actual controller amendment anchor required: '+name)
  after=before.replace(old,new)
  if name=='compile_master_plan.py':
   line=next(line for line in after.splitlines(keepends=True) if line.startswith('CONTROLLER_FILES = '));after=after.replace(line,line+"CONTROLLER_FILES = CONTROLLER_FILES + ('structural_cardinality_semantics.py',)\n")
  prior,current=function_hashes(before),function_hashes(after)
  changed={k for k in prior if prior[k]!=current[k]}
  if changed!={allowed}:raise ValueError('Only one requested function changed: '+name)
  proof[name]={'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest(),'changed_functions':sorted(changed),'unrelated_functions_unchanged':True};q.chmod(q.stat().st_mode | 0o200);q.write_text(after)
 return {'schema':'source-structural-cardinality-controller-AST-delta-v1','files':proof,'parent_modified':False,'scientific_fits':0}

if __name__=='__main__':
 import sys
 print(json.dumps(amend(sys.argv[1]),sort_keys=True))
