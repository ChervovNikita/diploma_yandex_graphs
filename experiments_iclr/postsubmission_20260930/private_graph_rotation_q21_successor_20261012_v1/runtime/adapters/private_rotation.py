"""Literal rotation Family methods on the existing prospective role provider.

New procedures retain source private stream custody. Existing Q21 procedures
remain at their own frozen factories and preserve their actual RNG recipe.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

try:
    from .classifier import ClassifierAdapter
    from .context_exclusion import import_source,seed_function
    from ..spec import confined,normalized_id,ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from adapters.context_exclusion import import_source,seed_function
    from spec import confined,normalized_id,ORIGINAL_PAPER_DATASETS

SOURCES={
 'family':('common_wrapper_paired_graph_native_family_source_20261010_v1/run_family.py','7af8048fe4f209922232bdf486b1b03c35f0df623842e377e2854d3ed8b111da'),
 'corrections':('common_wrapper_paired_graph_native_family_source_20261010_v1/corrections.py','58dc7ea7f15f3c0effd286e995503abb2606dab42222522985d102eceff63cc7'),
 'native':('common_wrapper_native_source_20261010_v1/models.py','07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303'),
 'routes':('shared_fast_graph_model_interface_20261010_v1/common_routes.py','84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773'),
 'factors':('portable_internal_be_public_interface_20261007_v2/core/factors.py','9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3'),
 'seedlaw':('common_wrapper_native_source_20261010_v1/run_common.py','e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc'),
 'base_train':('common_wrapper_native_source_20261010_v1/run_base.py','ca78469a7c1dcc9e98fba054ae1df714f6b19c3e91690ee7b48538005b4e52b1')}
ARMS={'shared4_paired_local':('paired_local',4),'paired_local_M1':('paired_local',1),
      'rademacher_allmap_M1':('rademacher',1)}
I4={'paired_local_I4':'paired_local_M1','rademacher_allmap_I4':'rademacher_allmap_M1'}
CONTRACT={'schema':'literal-private-rotation-prospective-v1','source_architecture':'native Model GAT/SAGE;depth2/H128/FFN1/heads8/LayerNorm',
 'loss':'bank:flattened4-route mean ownCE,private gradients averaged by4;M1:literal base_train boolean-mask ownCE',
 'optimizer':'one source AdamW group;lr;weight_decay0;default betas(.9,.999)/eps1e-8;no gradient/private-LR rescaling',
 'initialization':'paired_local equal-valued separate u1/u2;all diagonal factors1;rademacher control first input r only source signs',
 'private_RNG':'literal persistent fork_rng per route;M1 uses source.scope(stream0);source global/private states captured',
 'correction_placement':'SAGE FFN.linear1 on2H->H;GAT FFN.linear2 H->H;edges unchanged',
 'correction_I4_seed_adaptation':'offset5000081+1000003*body;body0 source anchor;matches corresponding shared-route direction seed',
 'source_protocol':'WikiCS11701x300,C10,TRAIN580/VALID5274,max1000/patience300;history preserved',
 'prospective_protocol':'existing WikiCS20/Photo10 roles;existing paired Q21 seeds;full5000,firstSELECT,no checkpoint0,common CAL',
 'existing_Q21_records_modified':False,'legacy_full_trajectory_or_failure_equality':False,
 'new_method_count':1,'additional_controls_not_new_algorithms':True}


def receipt(root,path):
    path=confined(Path(root).resolve(),str(path));raw=path.read_bytes()
    return {'path':str(path.relative_to(Path(root).resolve())),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def bound(root,row):
    if not isinstance(row,dict) or set(row)!={'path','bytes','sha256'} or receipt(root,row['path'])!=row:
        raise ValueError('Exact immutable source/input bytes required')
    return confined(Path(root).resolve(),row['path'])


def source_build(root,bindings,features,classes):
    if set(bindings)!=set(SOURCES) or any(bindings[k]['sha256']!=v[1] for k,v in SOURCES.items()):
        raise ValueError('Complete literal rotation source pins required')
    paths={k:bound(root,v) for k,v in bindings.items()}
    tree=ast.parse(paths['family'].read_bytes());family=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Family')
    methods=[copy.deepcopy(next(n for n in family.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in ('seeds','make','scope','logits')]
    original_hashes={n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in methods};changes=[]
    class Provider(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if ast.unparse(n.func)=='self.Model':
                if len(n.args)!=9 or ast.literal_eval(n.args[2])!=300 or ast.literal_eval(n.args[4])!=10:
                    raise ValueError('Literal Model provider anchors changed')
                n.args[2]=ast.Constant(features);n.args[4]=ast.Constant(classes);changes.extend(['features','classes'])
            if ast.unparse(n.func)=='torch.randint':
                if ast.literal_eval(n.args[2])!=(1,300):raise ValueError('Literal first-factor sign shape changed')
                n.args[2]=ast.Tuple(elts=[ast.Constant(1),ast.Constant(features)],ctx=ast.Load());changes.append('stem_sign_features')
            return n
    methods=[Provider().visit(n) for n in methods]
    if changes!=['features','classes','stem_sign_features'] or features<1 or classes<2:
        raise ValueError('Only three explicit provider sites may change')
    # Copy the complete source bank/single unit, including its original finite
    # loss boundary and the single-body persistent stream scope. No fit loop,
    # historical selector, NPZ reader, metrics or owner is imported/executed.
    fit=next(n for n in family.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    loop=next(n for n in ast.walk(fit) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='step')
    branch=loop.body[0];finite=loop.body[1]
    if not isinstance(branch,ast.If) or ast.unparse(branch.test)!='bank' or not isinstance(finite,ast.If):
        raise ValueError('Literal complete TRAIN unit structure changed')
    function=ast.parse('def source_train(self,model,optimizer,streams,members,bank):\n    torch=self.torch\n    return train_loss\n').body[0]
    function.body=[function.body[0],copy.deepcopy(branch),copy.deepcopy(finite),function.body[-1]]
    methods.append(function)
    code=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='LiteralFamily',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
    namespace={'math':math};exec(compile(code,'<literal_rotation_provider>','exec'),namespace)
    return namespace['LiteralFamily'](),paths,{'provider_changes':changes,'original_method_AST_hashes':original_hashes,
      'source_TRAIN_branch_AST_hash':hashlib.sha256(ast.dump(branch).encode()).hexdigest(),
      'copied_full_bank_or_scoped_single_unit':True,'source_legacy_loader_selector_owner_executed':False}


def base_train(path):
    tree=ast.parse(Path(path).read_bytes());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_step']
    if len(nodes)!=1:raise ValueError('One literal native TRAIN callback required')
    namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def make_adapter(*,torch,project_root,study,run,device):
    import numpy as np
    root=Path(project_root).resolve();c=run['config'];arm=c['rotation_arm']
    if normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS or arm not in ARMS:
        raise ValueError('Explicit permitted new-data rotation procedure required')
    if (study['dataset'] not in ('wikics','amazon-photo') or c['backbone'] not in ('GAT','SAGE') or c['width']!=128 or c['depth']!=2
            or c['learning_rate'] not in (.0005,.001,.002) or c['dropout'] not in (.1,.2,.4) or c['weight_decay']!=0
            or c['rotation_contract']!=CONTRACT or torch.get_default_dtype()!=torch.float32):
        raise ValueError('Frozen prospective source recipe/domain/dtype required')
    kind,members=ARMS[arm]
    if c['members']!=members or c['physical_slow_bodies']!=1 or c['source_optimizer_count']!=1:
        raise ValueError('One real source body/optimizer with declared routes required')
    data={k:torch.load(bound(root,row),map_location='cpu',weights_only=True) for k,row in c['node_inputs'].items()}
    graph,train,select=(data[k] for k in ('graph','train','select'))
    if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'} or set(select)!={'ids','y'}:
        raise ValueError('Separate target-free GRAPH and categorical TRAIN/SELECT required')
    x=graph['x'].to(device);edges=graph['edge_index'].to(device);ids=train['ids'].to(device);y=train['y'].to(device)
    if x.ndim!=2 or x.dtype!=torch.float32 or not torch.isfinite(x).all() or edges.dtype!=torch.long or edges.ndim!=2 or edges.shape[0]!=2:
        raise ValueError('Full finite source float32 features/int64 unweighted graph required')
    for role in (train,select):
        if (role['ids'].dtype!=torch.long or role['y'].dtype!=torch.long or role['ids'].ndim!=1 or role['y'].shape!=role['ids'].shape
                or len(role['ids'])==0 or len(torch.unique(role['ids']))!=len(role['ids']) or role['ids'].min()<0 or role['ids'].max()>=len(x)
                or role['y'].min()<0 or role['y'].max()>=c['classes']):raise ValueError('Unique nonempty categorical role authority required')
    if torch.isin(train['ids'],select['ids']).any():raise ValueError('TRAIN/SELECT overlap')
    source,paths,record=source_build(root,c['rotation_sources'],x.shape[1],c['classes'])
    source.torch=torch;source.F=torch.nn.functional
    source.cfg={'backbone':c['backbone'],'depth':2,'hidden':128,'dropout':c['dropout'],'learning_rate':c['learning_rate'],'rank':16,
                'correction_seed_offset':5000081+1000003*run['body']}
    source.device=torch.device(device);source.cuda=source.device.index if source.device.index is not None else 0
    source.Model=import_source(paths['native'],'rotation_native').Model
    source.routes=import_source(paths['routes'],'rotation_routes');source.factors=import_source(paths['factors'],'rotation_factors')
    source.corrections=import_source(paths['corrections'],'rotation_corrections');source.set_seed=seed_function(paths['seedlaw'],torch,np)
    namespace=base_train(paths['base_train']);namespace['F']=torch.nn.functional;source.train_step=namespace['train_step']
    labels=torch.zeros(len(x),dtype=torch.long,device=device);labels[ids]=y
    mask=torch.zeros(len(x),dtype=torch.bool,device=device);mask[ids]=True
    source.train_ids=ids;source.data=SimpleNamespace(x=x,edge_index=edges,y=labels,train_mask=mask)
    source.batch={'x':x,'edge_index':edges,'graph':SimpleNamespace(edge_index=edges)}
    model,optimizer,streams=source.make(run['seed'],run['body'],kind,True,members)
    if len(optimizer.param_groups)!=1 or optimizer.param_groups[0]['weight_decay']!=0 or optimizer.param_groups[0]['eps']!=1e-8 or optimizer.param_groups[0]['betas']!=(.9,.999):
        raise ValueError('Literal AdamW ownership/defaults changed')
    private={id(p) for m in model.modules() if isinstance(m,source.factors.FactorLinear) for p in (m.r,m.s)}
    private|={id(p) for n,p in model.named_parameters() if n.endswith(('.u1','.u2','.lora_a','.lora_b'))}
    roles={role:[n for n,p in model.named_parameters() if (id(p) in private)==(role=='private_fast')] for role in ('native_slow','private_fast')}
    def logits_fn(current,training,step):
        return source.logits(current,streams,members>1,torch.arange(len(x),device=device))
    def custom_train(adapter,step):
        own=source.source_train(model,optimizer,streams,members,members>1)
        corrected=len(model._correction_sites)*members
        return {'objective':own,'ce':own,'regularization':0.,'operation_counts':{'optimizer_updates':1,'model_forwards':members,
          'route_predictions':members,'backward_calls':1,'autograd_grad_VJP_calls':0,'supervised_examples':len(ids)*members,
          'native_graph_block_calls':2*members,'corrected_dense_map_calls':corrected,
          'Householder_reflection_calls':2*corrected if kind=='paired_local' else 0}}
    def capture():
        state=np.random.get_state()
        return {'streams':copy.deepcopy(streams),'numpy_rng':[state[0],state[1].tolist(),*state[2:]],
                'factor_members':{n:m.member for n,m in model.named_modules() if isinstance(m,source.factors.FactorLinear)},
                'training_modes':{n:m.training for n,m in model.named_modules()}}
    def restore(value):
        if len(value['streams'])!=members:raise ValueError('Source private stream roster changed')
        streams[:]=copy.deepcopy(value['streams']);state=value['numpy_rng'];np.random.set_state((state[0],np.array(state[1],dtype=np.uint32),*state[2:]))
        modules=dict(model.named_modules())
        for name,member in value['factor_members'].items():modules[name].member=member
        for name,mode in value['training_modes'].items():modules[name].training=mode
    adapter=ClassifierAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits_fn,train_ids=ids,train_y=y,
      select_ids=select['ids'].to(device),select_y=select['y'].to(device),regularizer=lambda current:x.new_zeros(()),
      fusion='mean_probabilities',selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5001,'save_every':1},
      custom_train=custom_train,external_capture=capture,external_restore=restore,
      provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'rotation_contract':CONTRACT,
        'source_provider_bindings':record,'actual_model_seed':source.seeds(run['seed'],run['body'])[0],
        'parameter_roles':roles,'legacy_trajectory_or_failure_equality':False})
    adapter.source_family=source;adapter.source_streams=streams;adapter.parameter_role_record=roles
    return adapter
