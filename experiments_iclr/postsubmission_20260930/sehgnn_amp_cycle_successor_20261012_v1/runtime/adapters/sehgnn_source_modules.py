"""Explicit prospective source bindings; operator bodies remain source code."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import types

try:from ..source_utils import bound,load
except ImportError:from source_utils import bound,load


def prospective_module(root,row,tag,lr,engineering_cpu,aliases=()):
    tree=ast.parse(bound(root,row).read_bytes());changes=[]
    if tag in ('bank','joint_training'):
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and type(node.value) is float and node.value==.001:
                node.value=float(lr);changes.append('declared_Adam_LR')
            if engineering_cpu and isinstance(node,ast.Compare) and isinstance(node.left,ast.Attribute) and node.left.attr=='type' and len(node.comparators)==1:
                other=node.comparators[0]
                if isinstance(other,ast.Constant) and other.value=='cuda':
                    node.ops=[ast.In()];node.comparators=[ast.Tuple(elts=[ast.Constant('cpu'),ast.Constant('cuda')],ctx=ast.Load())];changes.append('synthetic_CPU_constructor_guard')
            if engineering_cpu and isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and len(node.args)>1:
                message=node.args[1]
                if isinstance(message,ast.Constant) and message.value=='Caller-owned enabled native CUDA AMP master scaler':
                    # Actual scientific source retains the original enabled-CUDA check.
                    node.args[0]=ast.BoolOp(op=ast.Or(),values=[node.args[0],ast.Compare(left=ast.Attribute(value=ast.Subscript(value=ast.Name('rt',ast.Load()),slice=ast.Constant('device'),ctx=ast.Load()),attr='type',ctx=ast.Load()),ops=[ast.Eq()],comparators=[ast.Constant('cpu')])]);changes.append('synthetic_CPU_disabled_GradScaler')
        expected=3 if tag=='bank' else 4
        if changes.count('declared_Adam_LR')!=expected:raise ValueError('Exact original Adam LR anchors changed')
    name='gnnm_sehgnn_prospective_'+tag+'_'+row['sha256'][:12]+'_'+str(lr).replace('.','p')+('_CPU' if engineering_cpu else '_CUDA')
    m=types.ModuleType(name);m.__file__=str(bound(root,row));sys.modules[name]=m
    previous={k:sys.modules.get(k) for k,v in aliases}
    for k,v in aliases:sys.modules[k]=v
    try:exec(compile(ast.fix_missing_locations(tree),m.__file__, 'exec'),m.__dict__)
    finally:
        for k,v in previous.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    m.prospective_source_binding={'source_receipt':row,'explicit_changes':changes,'bound_AST_sha256':hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        'historical_source_admission_reused':False,'engineering_CPU_constructor_binding':bool(engineering_cpu)}
    return m


def modules(root,sources,lr,engineering_cpu):
    rows=sources['modules'];factor=load(root,rows['factors'],'factors');helper=load(root,rows['supply'],'supply');views=load(root,rows['views'],'views')
    bank=prospective_module(root,rows['bank'],'bank',lr,engineering_cpu)
    contract=load(root,rows['joint_contracts'],'joint_contracts')
    additive=prospective_module(root,rows['joint_additive'],'joint_additive',lr,engineering_cpu,(('joint_contracts',contract),))
    joint=prospective_module(root,rows['joint_training'],'joint_training',lr,engineering_cpu,(('joint_contracts',contract),('joint_additive_adapter',additive)))
    policy=load(root,rows['policy'],'policy')
    # Source seals are byte identities here, not scientific authorization or
    # a claim that historical native resource/quality qualification is inherited.
    old_bank=__import__('json').loads(bound(root,sources['metadata']['bank_sources']).read_text())
    old_joint=__import__('json').loads(bound(root,sources['metadata']['joint_sources']).read_text())
    bank.source_gate=lambda:copy.deepcopy(old_bank)
    joint.source_gate=lambda:copy.deepcopy(old_joint)
    return {'factor':factor,'helper':helper,'views':views,'bank':bank,'contract':contract,'additive':additive,'joint':joint,'policy':policy,
        'source_bindings':{k:v.prospective_source_binding for k,v in (('bank',bank),('additive',additive),('joint',joint))}}


def bind_constructor(rt,engine,dropout):
    tree=ast.parse(Path(engine.__file__).read_bytes())
    node=next(copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_model')
    calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and isinstance(n.func.slice,ast.Constant) and n.func.slice.value=='model_class']
    if len(calls)!=1 or not isinstance(calls[0].args[7],ast.Constant) or calls[0].args[7].value!=.5:raise ValueError('Original complete constructor dropout anchor changed')
    calls[0].args[7].value=dropout
    ns=dict(engine.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),engine.__file__+':declared_dropout','exec'),ns)
    engine.make_model=ns['make_model']
    return {'native_constructor':'source full512/512/classes5/25+12/2projection/4tasklayers/residualFalse','source_file':engine.__file__,
        'explicit_provider_binding':{'dropout':dropout},'bound_AST_sha256':hashlib.sha256(ast.dump(node).encode()).hexdigest()}
