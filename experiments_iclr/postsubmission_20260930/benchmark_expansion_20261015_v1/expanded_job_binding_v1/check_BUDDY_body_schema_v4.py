"""Reproduce the single metadata regression; imports no native/model library."""
import ast
import copy
import json
from pathlib import Path
E=Path(__file__).resolve().parent.parent
old=ast.parse((E/'buddy_native_provider_v1/provider.py').read_text())
new=ast.parse((E/'buddy_native_provider_v2/provider.py').read_text())
node=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='canonical_shared_body')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-body-schema-helper','exec'),namespace)
job={'body':None};before=copy.deepcopy(job)
try:int(job['body'])
except TypeError:pass
else:raise AssertionError('Original int(None) production metadata failure no longer reproduced')
assert namespace['canonical_shared_body'](job['body'])==namespace['canonical_shared_body'](0)==0 and job==before
for body in(True,False,1,-1,'0'):
    try:namespace['canonical_shared_body'](body)
    except ValueError:pass
    else:raise AssertionError('Unsupported shared-body label accepted')
a=next(n for n in old.body if isinstance(n,ast.ClassDef)and n.name=='NativeProviderSession')
b=next(n for n in new.body if isinstance(n,ast.ClassDef)and n.name=='NativeProviderSession')
for before,after in zip(a.body,b.body):
    if isinstance(before,ast.FunctionDef)and before.name!='__init__':assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
print(json.dumps({'status':'PASS_metadata_only','check_count':1,'old_TypeError_preserved':True,'null_and_0_canonical_metadata_0':True,'JOB_mutated':False,'training_provider_methods_AST_unchanged':True,'models':0,'forwards':0,'optimizer_updates':0,'Torch_imported':False}))
