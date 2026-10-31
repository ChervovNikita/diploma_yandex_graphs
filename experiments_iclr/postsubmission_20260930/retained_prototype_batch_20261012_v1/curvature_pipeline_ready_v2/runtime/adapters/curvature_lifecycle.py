"""Exact curvature warm/selector/continuation API for a future typed root owner.

Not a generic model_fit adapter. No data loader, launcher, receipt granting GO,
resource admission, or historical donor/checkpoint reader is supplied here.
"""
import ast,copy,hashlib,importlib.util,json,sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

SEEDS=(88101,88103,88201,88203,88207)
PHOTO_DOMAIN=tuple({'learning_rate':lr,'dropout':drop} for lr in (.0005,.001,.002) for drop in (.35,.7,.85))

def bound(root,row):
 p=Path(root)/row['path'];raw=p.read_bytes()
 if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Exact source binding changed')
 return p

@contextmanager
def aliases(modules):
 prior={k:sys.modules.get(k) for k in modules};sys.modules.update(modules)
 try:yield
 finally:
  for k,v in prior.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v

def load(root,row):
 p=bound(root,row);name='retained_curvature_'+row['sha256'][:20]
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;exec(compile(p.read_bytes(),str(p),'exec'),module.__dict__)
 return sys.modules[name]

def compile_namespace(module,path,transform,extra):
 tree=transform(copy.deepcopy(ast.parse(path.read_text())));space=dict(vars(module),**extra);exec(compile(ast.fix_missing_locations(tree),'<prospective_curvature_typed_binding>','exec'),space)
 return SimpleNamespace(**space),hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()

def modules(root,bindings,point):
 if point not in PHOTO_DOMAIN:raise ValueError('Explicit complete native Photo HPO point required')
 dependencies={}
 for key in ('native_polyformer','native_polyformer_outer','native_polyformer_preprocess','native_polynormer','backbone_boundary_adapter','graph_band_route_initializer','graph_init_training_adapter','selector'):
  with aliases(dependencies):dependencies[key]=load(root,bindings[key])
 # Imports are literal source dependencies supplied in a bounded namespace.
 with aliases(dependencies):native=load(root,bindings['modern_teacher_adapter'])
 def native_bind(tree):
  for n in tree.body:
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SEEDS','NATIVE') for t in n.targets):
    key=next(t.id for t in n.targets if isinstance(t,ast.Name));n.value=ast.Name(id='_new_seeds' if key=='SEEDS' else '_native_binding',ctx=ast.Load())
  return tree
 native_options=copy.deepcopy(native.NATIVE);native_options['polynormer_r'].update(lr=point['learning_rate'],dropout=point['dropout'],global_dropout=point['dropout'])
 native_seeds=tuple(seed+100003*body for seed in SEEDS for body in range(4))
 with aliases(dependencies):api,native_AST=compile_namespace(native,bound(root,bindings['modern_teacher_adapter']),native_bind,{'_new_seeds':native_seeds,'_native_binding':native_options})
 def selector_bind(tree):
  count=0
  for n in ast.walk(tree):
   if isinstance(n,ast.Compare) and ast.unparse(n)=='seed in (17, 29, 43)':n.comparators[0]=ast.Name(id='_new_seeds',ctx=ast.Load());count+=1
  if count!=1:raise ValueError('Unique retained selector seed guard changed')
  return tree
 selector,selector_AST=compile_namespace(dependencies['selector'],bound(root,bindings['selector']),selector_bind,{'_new_seeds':SEEDS})
 def continuation_bind(tree):
  count=0
  for node in tree.body:
   if isinstance(node,ast.FunctionDef) and node.name=='continuation':
    for n in ast.walk(node):
     if isinstance(n,ast.Tuple) and ast.unparse(n)=='(950, None)':n.elts[0].value=4750;count+=1
  if count!=1:raise ValueError('Unique Photo950 continuation cap changed')
  return tree
 integration,integration_AST=compile_namespace(dependencies['graph_init_training_adapter'],bound(root,bindings['graph_init_training_adapter']),continuation_bind,{})
 dependencies.update(native_api=api,selector=selector,integration=integration)
 dependencies['binding_receipt']={'native_HPO_seed_AST':native_AST,'selector_seed_AST':selector_AST,'continuation4750_AST':integration_AST,'source_bytes_changed':False,'Photo_only':True,'continuation_epoch0_eligible_source_law_preserved':True,'selector_metric':'minimum pooled raw-logit NLL; earliest strict tie'}
 return dependencies

def acquire(sources,point,seed,split,graph,train,select,canonical_edges,constants,trace,save_local):
 """Root typed source acquisition:250 warm updates; TRAIN-only finite trials."""
 if seed not in SEEDS or not 0<=split<10 or graph.teacher_backbone!='polynormer_r':raise ValueError('Complete Photo10/new seed source custody required')
 adapter=sources['integration'];native_api=sources['native_api'];boundary=sources['backbone_boundary_adapter'];initializer=sources['graph_band_route_initializer'];selector=sources['selector']
 with aliases(sources):
  spec=native_api.specification('polynormer_r','single_author',0,seed)
  native,opt,warm=adapter.warm_native(native_api,spec,graph,train,select,trace,save_local)
  rng=warm['rng']
  try:
   snapshot=adapter.named_optimizer_snapshot(native,opt);copy_native=copy.deepcopy(native);k1=adapter.clone_boundary(copy_native,boundary,1);k4=adapter.clone_boundary(copy_native,boundary,4)
   k1.eval();identity=adapter.identity_logits_audit(copy_native,k1,k4,graph);k4.train(native.training)
   transported,transport=adapter.transport_optimizer(copy_native,snapshot,k4);raw_snapshot=adapter.named_optimizer_snapshot(k4,transported);args=adapter.raw_arguments(graph,'polynormer_r')
   theta0,fn,binding=selector.bind_head_only(k1,'polynormer_r',args);S=initializer.symmetric_normalized_adjacency(len(graph.teacher_input),canonical_edges,theta0.dtype,theta0.device)
   import torch
   nodes=torch.arange(len(graph.teacher_input),device=theta0.device,dtype=torch.int64)
   states,receipt=selector.select_initializations(k4,raw_snapshot,fn,theta0,S,nodes,train.nodes,train.labels,'polynormer_r',args,rng,seed,constants,initializer,adapter)
   receipt.update(source_warm=warm['metadata'],source_optimizer_transport=transport,source_identity_logits=identity,source_head_binding=binding,new_binding=sources['binding_receipt'],split=split,point=point,historical_donor=False,typed_root_accounting_required=True)
   return states,receipt,warm
  finally:adapter.rng_restore(rng)

def continue_arm(sources,state,graph,train,select,trace,save_logits):
 """Uses only returned pretrial model/named Adam and original continuation RNG."""
 adapter=sources['integration'];adapter.rng_restore(state['rng'])
 return adapter.continuation(state['model'],state['optimizer'],graph,train,select,trace,save_logits)
