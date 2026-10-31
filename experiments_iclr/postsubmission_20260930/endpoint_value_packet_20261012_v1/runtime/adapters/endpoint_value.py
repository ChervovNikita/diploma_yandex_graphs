"""Source-native endpoint VALUE SAGE, with explicit provider-only shape bindings.

The source Family's constructor/seed/make methods, FactorLinear, native Model
and exact E/N message operation are reused. This module imports no numerics or
data until make_adapter. Full source recipe/history and new roles stay separate.
"""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace,ModuleType
import sys

try:
    from .classifier import ClassifierAdapter
    from .node import source_module
    from ..spec import normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
    from adapters.classifier import ClassifierAdapter
    from adapters.node import source_module
    from spec import normalized_id, ORIGINAL_PAPER_DATASETS

SOURCES={
 'family':('native_SAGE_endpoint_value_transport_source_20261011_v2/run_family.py','dbe58a63e2976e865b475b3557ec8fad7568dcf31bfcd721de816772c51d55bd'),
 'installer':('native_SAGE_endpoint_value_transport_source_20261011_v2/value_installer.py','e92df73611dd2cf4c063dc7bee1c7182a1c07907a77777f10d2a748fd92bb23a'),
 'transport':('native_SAGE_endpoint_value_transport_source_20261011_v2/value_transport.py','cf97376779782482148ed90b57ea47d58fe18fd5554aab543d41cf3f0e9cf263'),
 'routes':('shared_fast_graph_model_interface_20261010_v1/common_routes.py','84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773'),
 'factors':('portable_internal_be_public_interface_20261007_v2/core/factors.py','9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3'),
 'native':('common_wrapper_native_source_20261010_v1/models.py','07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303'),
 'driver':('common_wrapper_native_source_20261010_v1/run_common.py','e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc'),
}
ARMS=('BASE','E4','N4','equipped_M1','ordinary_M1','original_gnnm')
I4={'equipped_I4':'equipped_M1','ordinary_I4':'ordinary_M1'}
CONTRACT={'schema':'endpoint-value-standardized-source-port-v1','native':'exact source SAGE depth2/hidden128/LayerNorm/GELU residual FFN',
 'operator':'F_target.T @ F_source on four-channel stalks before factual mean; original self VALUE unchanged',
 'N4':'same source builders and parameter opportunity, receiver [x_i,silu(x_i)] context, per-node map gathered before mean',
 'factors':'all-map FactorLinear; retained native shared W/b; source first-stem Rademacher, later unit factors; added builder W/b zero with constructor RNG preserved',
 'objective':'source packed mean own CE, one AdamW WD0 per source logical update; unscaled own CE for M1/I4',
 'serving':'mean probabilities for source E/N/BASE/M1/I4; original GNNM mean logits',
 'source_controller':{'maximum_updates':1000,'patience_updates':300,'metric':'accuracy','ties':'first','cadence':1,'checkpoint0':False},
 'source_exact_recipe':False,'adaptations':['new GRAPH/TRAIN/SELECT/CAL/evaluate providers and dimensions','nine-point useful LR/dropout search','same-role native M1/own I4/original GNNM references','common after-pool500 scalar CAL'],
 'history':'closed WikiCS five-arm15-bank/24-body source comparison; no proof of new-data accuracy or generic conditional-factor utility'}

def receipt(root,path):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root) or path.is_symlink():raise ValueError('Confined source required')
    raw=path.read_bytes();return {'path':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def bound(root,row):
    rel=Path(row['path']);path=Path(root).resolve()/rel
    if rel.is_absolute() or '..' in rel.parts or path.is_symlink() or receipt(root,path)!=row:raise ValueError('Exact endpoint source/input changed')
    return path

def load_module(root,row):
    path=bound(root,row);name='endpoint_source_'+row['sha256'][:20]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]

def render_sources(root,bindings,nodes,features,classes):
    """Bind only the source's explicit provider-size anchors; retain all laws."""
    if set(bindings)!=set(SOURCES) or any(bindings[k]['sha256']!=v[1] for k,v in SOURCES.items()):raise ValueError('Exact endpoint source roster required')
    paths={k:bound(root,v) for k,v in bindings.items()}
    tree=ast.parse(paths['family'].read_text());family=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Family')
    names=('seeds','native_body','make','scope','training_outputs')
    methods=[copy.deepcopy(next(n for n in family.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in names]
    changes=[]
    class Dimensions(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='self' and node.func.attr=='Model':
                if len(node.args)!=9 or not isinstance(node.args[2],ast.Constant) or node.args[2].value!=300 or not isinstance(node.args[4],ast.Constant) or node.args[4].value!=10:raise ValueError('Source native Model anchors changed')
                node.args[2]=ast.copy_location(ast.Constant(features),node.args[2]);node.args[4]=ast.copy_location(ast.Constant(classes),node.args[4]);changes.extend(['native_input_dim','native_classes'])
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr=='randint':
                value=node.args[2]
                if not isinstance(value,ast.Tuple) or [getattr(v,'value',None) for v in value.elts]!=[1,300]:raise ValueError('Source first-stem Rademacher shape changed')
                value.elts[1]=ast.copy_location(ast.Constant(features),value.elts[1]);changes.append('stem_factor_input_dim')
            return node
    methods=[Dimensions().visit(n) for n in methods]
    if changes!=['native_input_dim','native_classes','stem_factor_input_dim']:raise ValueError('Unexpected source provider binding')
    construct=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='SourceBuild',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
    raw=paths['installer'].read_text()
    if raw.count('11701')!=4:raise ValueError('Source installer node anchors changed')
    installer=raw.replace('11701',str(nodes))
    return {'build':ast.unparse(construct)+'\n','installer':installer,'binding_changes':changes+['installer_full_graph_nodes']*4,
      'source_method_AST_hashes':{n.name:hashlib.sha256(ast.dump(next(v for v in family.body if isinstance(v,ast.FunctionDef) and v.name==n.name)).encode()).hexdigest() for n in methods}}

def construct_source(torch,root,bindings,*,nodes,features,classes,seed,body,arm,dropout,learning_rate,device,edge_metadata):
    rendered=render_sources(root,bindings,nodes,features,classes)
    ns={'__name__':'endpoint_bound_builder'};exec(compile(rendered['build'],'<pinned_endpoint_family_provider_binding>','exec'),ns)
    module_name='endpoint_bound_installer_'+hashlib.sha256(rendered['installer'].encode()).hexdigest()[:20]
    if module_name not in sys.modules:
        installed=ModuleType(module_name);installed.__file__=str(bound(root,bindings['installer']));sys.modules[module_name]=installed
        exec(compile(rendered['installer'],installed.__file__+'::provider_nodes','exec'),installed.__dict__)
    ins=vars(sys.modules[module_name])
    source=ns['SourceBuild']();source.torch=torch;source.F=torch.nn.functional;source.device=torch.device(device);source.cuda=source.device.index or 0
    source.cfg={'depth':2,'hidden':128,'dropout':dropout,'learning_rate':learning_rate,'rank':16}
    source.Model=load_module(root,bindings['native']).Model;source.set_seed=load_module(root,bindings['driver']).set_seed
    source.routes=load_module(root,bindings['routes']);source.factors=load_module(root,bindings['factors']);source.value_helper=load_module(root,bindings['transport'])
    source.value_installer=SimpleNamespace(**ins);source._operation={'E4':'endpoint','N4':'node','equipped_M1':'endpoint'}.get(arm,'none')
    source.edge_metadata=dict(edge_metadata);source._single_adapter=None
    if arm in ('BASE','E4','N4','equipped_M1'):
        members=4 if arm in ('BASE','E4','N4') else 1
        model,optimizer,streams=source.make(seed,body,'baseline',True,members,'shared' if members==4 else 'single')
    elif arm in ('ordinary_M1','original_gnnm'):
        native_seed=source.seeds(seed,body)[0];source.set_seed(native_seed)
        native=load_module(root,bindings['native']);args=('SAGE',2,features,128,classes,1,8,'LayerNorm',dropout)
        model=(native.TABMModel(*args,tabm_inits=4,device=device) if arm=='original_gnnm' else native.Model(*args)).to(device)
        members=4 if arm=='original_gnnm' else 1
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=0)
        streams=[]
        for row in range(members):
            source.set_seed(source.seeds(seed,body+row)[2]);streams.append({'cpu':torch.get_rng_state().clone(),**({'cuda':torch.cuda.get_rng_state(source.device).clone()} if source.device.type=='cuda' else {})})
    else:raise ValueError('Exact source/reference arm required')
    return source,model,optimizer,streams,rendered

def make_adapter(*,torch,project_root,study,run,device):
    c=run['config'];arm=c['endpoint_arm']
    if normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS or arm not in ARMS:raise ValueError('Declared new-data endpoint source arm required')
    if c['width']!=128 or c['depth']!=2 or c['classes']<2 or c['weight_decay']!=0 or c['learning_rate'] not in (.0005,.001,.002) or c['dropout'] not in (.1,.2,.4):raise ValueError('Fixed source architecture/useful search domain required')
    expected=4 if arm in ('BASE','E4','N4','original_gnnm') else 1
    if c['members']!=expected or c['source_optimizer_count']!=1:raise ValueError('Complete source ownership required')
    def load(row):return torch.load(bound(project_root,row),map_location='cpu',weights_only=True)
    graph=load(c['node_inputs']['graph']);train=load(c['node_inputs']['train']);select=load(c['node_inputs']['select'])
    if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'} or set(select)!={'ids','y'}:raise ValueError('Exactly GRAPH/TRAIN/SELECT required')
    if torch.isin(train['ids'],select['ids']).any() or not torch.equal(train['ids'],train['ids'].sort().values):raise ValueError('Sorted TRAIN and disjoint SELECT required by source mask ordering')
    x=graph['x'].to(device);edges=graph['edge_index'].to(device);train_ids=train['ids'].to(device);select_ids=select['ids'].to(device)
    train_y=train['y'].to(device);select_y=select['y'].to(device)
    if x.dtype!=torch.float32 or edges.dtype!=torch.long or edges.ndim!=2 or edges.shape[0]!=2:raise ValueError('Factual float32 feature/int64 COO graph required')
    metadata={'nodes':len(x),'edges':edges.shape[1],'self_edges':int(edges[0].eq(edges[1]).sum())}
    source,model,optimizer,streams,rendered=construct_source(torch,project_root,c['endpoint_sources'],nodes=len(x),features=x.shape[1],classes=c['classes'],
      seed=run['seed'],body=run['body'],arm=arm,dropout=c['dropout'],learning_rate=c['learning_rate'],device=device,edge_metadata=metadata)
    factual=SimpleNamespace(edge_index=edges);source.batch={'x':x,'edge_index':edges,'graph':factual};source.train_ids=train_ids
    labels=torch.zeros(len(x),dtype=torch.long,device=device);labels[train_ids]=train_y;mask=torch.zeros(len(x),dtype=torch.bool,device=device);mask[train_ids]=True
    source.data=SimpleNamespace(x=x,edge_index=edges,y=labels,train_mask=mask)
    def logits_fn(current,training,step):
        if arm in ('BASE','E4','N4'):
            return current(source.batch,route_scope=lambda member:source.scope(streams,member))[0]
        if arm=='equipped_M1':
            with source.scope(streams,0),source.factors.member_context(current,0):return current(factual,x).unsqueeze(0)
        outputs=[]
        for member in range(expected):
            with source.scope(streams,member):outputs.append(current(factual,x,tabm_seed=member) if arm=='original_gnnm' else current(factual,x))
        return torch.stack(outputs)
    def custom_train(adapter,step):
        adapter.optimizer.zero_grad(set_to_none=True)
        if arm in ('BASE','E4','N4','equipped_M1'):
            logits,_=source.training_outputs(model,streams,expected>1)
        else:logits=logits_fn(model,True,step)[:,train_ids]
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),train_y.repeat(expected))
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite source VALUE gradient')
        adapter.optimizer.step()
        return {'objective':float(loss.detach()),'ce':float(loss.detach()),'regularization':0.,
          'operation_counts':{'optimizer_updates':1,'backward_calls':1,'model_forwards':expected,'route_predictions':expected,'supervised_examples':len(train_ids)}}
    def capture():
        counters={name:{k:copy.deepcopy(getattr(module,k)) for k in ('value_counts_train','value_counts_eval','value_last_unit')}
          for name,module in model.named_modules() if hasattr(module,'value_counts_train')}
        return {'streams':copy.deepcopy(streams),'counters':counters,'training_modes':{name:module.training for name,module in model.named_modules()},
          'factor_members':{name:module.member for name,module in model.named_modules() if isinstance(module,source.factors.FactorLinear)}}
    def restore(value):
        streams[:]=copy.deepcopy(value['streams']);modules=dict(model.named_modules())
        for name,counters in value['counters'].items():
            for key,field in counters.items():setattr(modules[name],key,copy.deepcopy(field))
        for name,mode in value['training_modes'].items():modules[name].training=mode
        for name,member in value['factor_members'].items():modules[name].member=member
    selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':300,'save_every':1}
    return ClassifierAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits_fn,train_ids=train_ids,train_y=train_y,select_ids=select_ids,select_y=select_y,
      regularizer=lambda current:x.new_zeros(()),fusion='mean_logits' if arm=='original_gnnm' else 'mean_probabilities',selector=selector,
      custom_train=custom_train,external_capture=capture,external_restore=restore,
      provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'actual_model_seed':source.seeds(run['seed'],run['body'])[0],
        'source_contract':CONTRACT,'explicit_provider_binding_changes':rendered['binding_changes'],'source_method_AST_hashes':rendered['source_method_AST_hashes']})
