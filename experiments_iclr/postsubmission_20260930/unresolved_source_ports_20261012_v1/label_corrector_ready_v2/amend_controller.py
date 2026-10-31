"""Narrow cumulative compiler amendment; existing engine/native_stopping remain untouched.

Use amend_compiler(text) for composition and semantic_modules() for sibling module
bytes, or install(runtime_dir). Only fit_semantics,graph_check,typed_counts may
change AST. The source-specific contracts are embedded to avoid adapter namespace
imports in the root's cumulative controller.
"""
import ast,copy,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
BRANCHES=(('private_response_source_contract','response_semantics','validate_response_run'),('label_corrector_source_contract','label_corrector_semantics','validate_label_corrector_run'))
def _functions(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def amend_compiler(text):
 tree=ast.parse(text);f={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)};before=_functions(text)
 fit=f['fit_semantics'];constants={n.value for n in ast.walk(fit) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
 at=next(i for i,n in enumerate(fit.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='family' for t in n.targets))
 for key,module,name in BRANCHES:
  if key not in constants:
   fit.body[at:at]=ast.parse(f"if {key!r} in config:\n from {module} import {name}\n return {name}(run)\n").body;at+=1
 graph=f['graph_check'];fitblock=next(n for n in ast.walk(graph) if isinstance(n,ast.If) and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paid' for t in x.targets) for x in n.body))
 constants={n.value for n in ast.walk(fitblock) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
 if 'private_response_source_contract' not in constants:
  at=next(i for i,n in enumerate(fitblock.body) if isinstance(n,ast.If) and 'prefixes' in ast.unparse(n.test))
  n=fitblock.body[at]
  if not isinstance(n.test,ast.BoolOp) or not isinstance(n.test.op,ast.Or):raise ValueError('Expected typed prefix/paid ancestry graph gate')
  exclude=ast.parse("'private_response_source_contract' not in run['config']",mode='eval').body
  n.test.values[0]=ast.BoolOp(op=ast.And(),values=[n.test.values[0],exclude]);fitblock.body[at:at]=ast.parse("if 'private_response_source_contract' in run['config']:\n from response_semantics import validate_response_run\n validate_response_run(run)\n").body
 for n in ast.walk(f['typed_counts']):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='startswith' and n.args and any(isinstance(x,ast.Constant) and x.value=='known_sehgnn_' for x in ast.walk(n.args[0])):
   old=n.args[0];values=list(old.elts) if isinstance(old,ast.Tuple) else [old]
   for key in ('known_response_','known_label_corrector_'):
    if not any(isinstance(x,ast.Constant) and x.value==key for x in values):values.append(ast.Constant(value=key))
   n.args[0]=ast.Tuple(elts=values,ctx=ast.Load())
 lines=text.splitlines(keepends=True)
 for name in sorted(('fit_semantics','graph_check','typed_counts'),key=lambda k:f[k].lineno,reverse=True):
  n=f[name];lines[n.lineno-1:n.end_lineno]=[ast.unparse(ast.fix_missing_locations(n))+'\n']
 result=''.join(lines);after=_functions(result);changed={k for k in before if before[k]!=after[k]}
 if not changed<={'fit_semantics','graph_check','typed_counts'}:raise ValueError('Unexpected cumulative controller function change')
 return result

def _literal(path,name):
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets));return ast.literal_eval(node.value)
def semantic_modules():
 private=HERE.parent/'private_response_ready_v1/runtime';label=HERE/'runtime';out={}
 for name,root,adapter in [('response_semantics.py',private,'private_response.py'),('label_corrector_semantics.py',label,'label_corrector.py')]:
  code=(root/name).read_text();contract=_literal(root/'adapters'/adapter,'CONTRACT');code=code.replace('from adapters.private_response import CONTRACT','CONTRACT='+repr(contract));
  if adapter=='label_corrector.py':code=code.replace('from adapters.label_corrector import CONTRACT,WORK','CONTRACT='+repr(contract)+'\nWORK='+repr(_literal(root/'adapters'/adapter,'WORK')))
  ast.parse(code);out[name]=code
 return out

def install(runtime_dir):
 root=Path(runtime_dir);p=root/'compile_master_plan.py';old=p.read_text();new=amend_compiler(old);p.write_text(new)
 for name,code in semantic_modules().items():(root/name).write_text(code)
 return {'changed_AST_functions':sorted(k for k,v in _functions(old).items() if v!=_functions(new)[k]),'compiler_sha256':hashlib.sha256(new.encode()).hexdigest(),'new_modules':list(semantic_modules()),'engine_or_native_stopping_changed':False}
