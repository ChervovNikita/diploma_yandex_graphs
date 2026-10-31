"""Q21 exact private-bootstrap helpers/native maps under declared new roles.

Only source provider dimensions/globals are rebound. Fixed float64 TRAIN
weights and the source's two block VJPs remain exact; no scalar substitute.
"""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

try:
 from .classifier import ClassifierAdapter
 from .context_exclusion import import_source, seed_function, tensor_hash
 from ..spec import confined, normalized_id, ORIGINAL_PAPER_DATASETS
except ImportError:
 from adapters.classifier import ClassifierAdapter
 from adapters.context_exclusion import import_source, seed_function, tensor_hash
 from spec import confined, normalized_id, ORIGINAL_PAPER_DATASETS

SOURCES={
 'family':('common_wrapper_private_bootstrap_native_family_source_20261010_v1/run_family.py','eeed1e92ab79e9a9fcd5a52097ede1ff08c8cb4557af43ae5f5d200d48eaba6d'),
 'bootstrap':('common_wrapper_private_bootstrap_native_family_source_20261010_v1/bootstrap.py','40ca31357f252f6dbea25ba1ba82c6b263b243e16f18beaed23d2cf03a6c1674'),
 'corrections':('common_wrapper_paired_graph_native_family_source_20261010_v1/corrections.py','58dc7ea7f15f3c0effd286e995503abb2606dab42222522985d102eceff63cc7'),
 'native':('common_wrapper_native_source_20261010_v1/models.py','07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303'),
 'routes':('shared_fast_graph_model_interface_20261010_v1/common_routes.py','84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773'),
 'factors':('portable_internal_be_public_interface_20261007_v2/core/factors.py','9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3'),
 'seedlaw':('common_wrapper_native_source_20261010_v1/run_common.py','e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc'),
 'base_train':('common_wrapper_native_source_20261010_v1/run_base.py','ca78469a7c1dcc9e98fba054ae1df714f6b19c3e91690ee7b48538005b4e52b1'),
 'original_train':('source_snapshot/run_tabm.py','b1d748e50c854802b93f4d4fa7485f096daa1cd36d3dd3bd1c89fbfa63cbd124'),
}
# arm: (native equipment, factorized, routes, weighted-private-credit)
ARMS={
 'shared4_coherent_bootstrap':('coherent',True,4,True),
 'shared4_paired_graph_bootstrap':('paired_graph',True,4,True),
 'shared4_rank1_lora_graph_bootstrap':('rank1_lora_graph',True,4,True),
 'shared4_coherent':('coherent',True,4,False),
 'shared4_paired_graph':('paired_graph',True,4,False),
 'shared4_rank1_lora_graph':('rank1_lora_graph',True,4,False),
 'shared4_unchanged':('rademacher',True,4,False),
 'ordinary_M1':('coherent',False,1,False),
 'factorized_allmap_M1':('coherent',True,1,False),
 'paired_graph_M1':('paired_graph',True,1,False),
 'rank1_lora_graph_M1':('rank1_lora_graph',True,1,False),
 'original_gnnm':('original',False,4,False),
}
I4={'ordinary_genuine_I4':'ordinary_M1','factorized_allmap_genuine_I4':'factorized_allmap_M1',
 'paired_graph_genuine_I4':'paired_graph_M1','rank1_lora_graph_genuine_I4':'rank1_lora_graph_M1'}
CONTRACT={'weight_law':'np.default_rng(seed+6000119);Exp(1)float64[4,TRAIN];class mean1;no clipping/redraw;one fixed table shared across weighted arms/backbones',
 'credit':'shared=partial mean unweighted ownCE;private r/s and correction banks=partial mean(weight*ownCE);same four-route forward;two VJPs;one AdamW',
 'optimizer':'source AdamW lr;zero_decay;default_betas_epsilon;no private-rate normalization',
 'initialization':'source Model;coherent unit factors;unchanged reference first-r Rademacher;paired equal Householder normals;LoRA b0',
 'native':'GAT/SAGE depth2,width128,FFNmultiplier1,heads8,LayerNorm;full factual unweighted graph',
 'serving':'unweighted probability mean;original GNNM mean raw logits;no validation/bootstrap resampling',
 'source_protocol':'WikiCS TRAIN580 VALID5274;seeds7301/7403/7507;max1000/patience300/first accuracy every update',
 'new_protocol':'prepared development GRAPH/TRAIN/SELECT/CAL/evaluate;all masks;new paired seeds;full5000/first SELECT accuracy/no checkpoint0',
 'source_exact_recipe':False,'historical_negative_closure_preserved':True}
CONTRACT['matched_corrected_I4_seed_adaptation']='new corrected M1/I4 controls add1000003*body to correction_seed_offset;body0 retains source anchor;native/dropout identities already source-private'


def receipt(root,path):
 path=confined(Path(root).resolve(),str(path));raw=path.read_bytes()
 return {'path':str(path.relative_to(Path(root).resolve())),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(root,row):
 if not isinstance(row,dict) or set(row)!={'path','bytes','sha256'} or receipt(root,row['path'])!=row:raise ValueError('Exact immutable source/input binding changed')
 return confined(Path(root).resolve(),row['path'])
def source_paths(root,bindings):
 if set(bindings)!=set(SOURCES) or any(bindings[k]['sha256']!=v[1] for k,v in SOURCES.items()):raise ValueError('Complete pinned Q21 source roster required')
 return {k:bound(root,v) for k,v in bindings.items()}

def bootstrap_module(root,bindings,train_rows,classes):
 path=bound(root,bindings['bootstrap']);tree=ast.parse(path.read_bytes());changed=0
 for n in tree.body:
  if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='(MEMBERS, TRAIN_NODES, CLASSES)':
   if ast.literal_eval(n.value)!=(4,580,10):raise ValueError('Source bootstrap dimension anchors changed')
   n.value=ast.Tuple(elts=[ast.Constant(4),ast.Constant(train_rows),ast.Constant(classes)],ctx=ast.Load());changed+=1
 if changed!=1 or train_rows<classes or classes<2:raise ValueError('One explicit TRAIN/class provider binding required')
 ns={};exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
 return SimpleNamespace(**ns)

def source_function(path,name,namespace):
 tree=ast.parse(Path(path).read_bytes());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
 if len(nodes)!=1:raise ValueError('Exact source training callback required')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]

def source_build(root,bindings,features,classes):
 paths=source_paths(root,bindings);tree=ast.parse(paths['family'].read_bytes());family=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Family')
 methods=[copy.deepcopy(next(n for n in family.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in ('seeds','make','scope','logits')]
 changes=[];hashes={n.name:hashlib.sha256(ast.dump(n).encode()).hexdigest() for n in methods}
 class Dimensions(ast.NodeTransformer):
  def visit_Call(self,n):
   self.generic_visit(n)
   if ast.unparse(n.func)=='self.Model':
    if len(n.args)!=9 or ast.literal_eval(n.args[2])!=300 or ast.literal_eval(n.args[4])!=10:raise ValueError('Source native Model anchors differ')
    n.args[2]=ast.Constant(features);n.args[4]=ast.Constant(classes);changes.extend(['native_features','native_classes'])
   if ast.unparse(n.func)=='torch.randint':
    if ast.literal_eval(n.args[2])!=(1,300):raise ValueError('Source stem sign shape differs')
    n.args[2]=ast.Tuple(elts=[ast.Constant(1),ast.Constant(features)],ctx=ast.Load());changes.append('stem_factor_features')
   return n
 methods=[Dimensions().visit(n) for n in methods]
 if changes!=['native_features','native_classes','stem_factor_features']:raise ValueError('Only three declared source provider sites may change')
 tree=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='SourceBuild',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
 ns={};exec(compile(tree,'<Q21_source_provider_binding>','exec'),ns)
 return ns['SourceBuild'](),paths,{'provider_changes':changes+['bootstrap.TRAIN_NODES','bootstrap.CLASSES'],'original_method_AST_hashes':hashes}

def make_adapter(*,torch,project_root,study,run,device):
 import numpy as np
 c=run['config'];arm=c['bootstrap_arm'];root=Path(project_root).resolve()
 if normalized_id(study['dataset']) in ORIGINAL_PAPER_DATASETS or arm not in ARMS:raise ValueError('Explicit new-data Q21 source arm required')
 if (c['backbone'] not in ('GAT','SAGE') or c['width']!=128 or c['depth']!=2 or c['weight_decay']!=0
     or c['learning_rate'] not in (.0005,.001,.002) or c['dropout'] not in (.1,.2,.4) or c['source_contract']!=CONTRACT):raise ValueError('Frozen source recipe/useful finite domain required')
 kind,factorized,members,weighted=ARMS[arm]
 if c['members']!=members or c['physical_slow_bodies']!=1 or c['source_optimizer_count']!=1 or torch.get_default_dtype()!=torch.float32:raise ValueError('Exact source body/owner/dtype required')
 data={k:torch.load(bound(root,row),map_location='cpu',weights_only=True) for k,row in c['node_inputs'].items()};graph,train,select=(data[k] for k in ('graph','train','select'))
 if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'} or set(select)!={'ids','y'}:raise ValueError('Separate GRAPH/TRAIN/SELECT required')
 x=graph['x'].to(device);edges=graph['edge_index'].to(device);ids=train['ids'].to(device);y=train['y'].to(device)
 if x.ndim!=2 or x.dtype!=torch.float32 or not torch.isfinite(x).all() or edges.dtype!=torch.long or edges.ndim!=2 or edges.shape[0]!=2:raise ValueError('Native finite feature/int64 COO graph required')
 if len(torch.unique(ids))!=len(ids) or y.shape!=ids.shape or y.dtype!=torch.long or torch.isin(train['ids'],select['ids']).any():raise ValueError('Unique disjoint categorical TRAIN/SELECT required')
 if set(y.cpu().tolist())!=set(range(c['classes'])):raise ValueError('Every declared TRAIN class required by fixed bootstrap')
 source,paths,rendered=source_build(root,c['bootstrap_sources'],x.shape[1],c['classes'])
 source.torch=torch;source.cfg={'backbone':c['backbone'],'depth':2,'hidden':128,'dropout':c['dropout'],'learning_rate':c['learning_rate'],'rank':16,
  'correction_seed_offset':5000081+1000003*run['body']}
 source.device=torch.device(device);source.cuda=source.device.index or 0;source.Model=import_source(paths['native'],'Q21_native').Model
 source.routes=import_source(paths['routes'],'Q21_routes');source.factors=import_source(paths['factors'],'Q21_factors');source.corrections=import_source(paths['corrections'],'Q21_corrections')
 source.set_seed=seed_function(paths['seedlaw'],torch,np)
 source.bootstrap=bootstrap_module(root,c['bootstrap_sources'],len(ids),c['classes'])
 labels=torch.zeros(len(x),dtype=torch.long,device=device);labels[ids]=y;mask=torch.zeros(len(x),dtype=torch.bool,device=device);mask[ids]=True
 source.data=SimpleNamespace(x=x,edge_index=edges,y=labels,train_mask=mask);source.batch={'x':x,'edge_index':edges,'graph':SimpleNamespace(edge_index=edges)}
 source.train_step=source_function(paths['base_train'],'train_step',{'F':torch.nn.functional})
 if arm=='original_gnnm':
  native=import_source(paths['native'],'Q21_native');source.set_seed(source.seeds(run['seed'],run['body'])[0])
  model=native.TABMModel(c['backbone'],2,x.shape[1],128,c['classes'],1,8,'LayerNorm',c['dropout'],tabm_inits=4,device='cpu').to(device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=0);streams=[]
  for row in range(4):
   source.set_seed(source.seeds(run['seed'],run['body']+row)[2]);streams.append({'cpu':torch.get_rng_state().clone(),**({'cuda':torch.cuda.get_rng_state(source.device).clone()} if source.device.type=='cuda' else {})})
 else:model,optimizer,streams=source.make(run['seed'],run['body'],kind,factorized,members)
 weights=None;weight_hash=None;weight_version=None
 if weighted:
  payload=torch.load(bound(root,c['bootstrap_weights']),map_location='cpu',weights_only=True)
  if (payload['schema']!='private-bootstrap-fixed-TRAIN-v1' or payload['seed']!=run['seed'] or payload['train']!=c['node_inputs']['train']
      or payload['bootstrap_source']!=c['bootstrap_sources']['bootstrap'] or not torch.equal(payload['ids'],train['ids']) or not torch.equal(payload['y'],train['y'])):raise ValueError('Exact saved weight/TRAIN/source/seed ancestry required')
  weights=payload['weights'].to(device);weight_hash=tensor_hash(weights);weight_version=weights._version
  if weights.dtype!=torch.float64 or weights.requires_grad or weights.shape!=(4,len(ids)) or weight_hash!=payload['weights_tensor_sha256']:raise ValueError('Untrained resident float64 weight table differs')
  roles,shared,private=source.bootstrap.parameter_roles(torch,source.factors,model)
 else:roles={'gradient_law':'unchanged scalar ownCE source callback'};shared=private=None
 model._source_operator_record=rendered
 def guard():
  if weights is not None and (weights._version!=weight_version or tensor_hash(weights)!=weight_hash):raise ValueError('Immutable fixed TRAIN weights changed')
 def logits_fn(current,training,step):
  guard()
  if arm=='original_gnnm':
   return torch.stack([current(source.data,x,tabm_seed=row) for row in range(4)])
  return source.logits(current,streams,members>1,torch.arange(len(x),device=device))
 original_train=source_function(paths['original_train'],'train_step_tabm',{'F':torch.nn.functional,'TABM_INITS':4})
 def custom_train(adapter,step):
  guard();optimizer.zero_grad()
  if weighted:
   logits=source.logits(model,streams,True,ids)
   own,credit,unconnected,member_losses=source.bootstrap.bank_gradients(torch,torch.nn.functional,logits,y,weights,shared,private)
   optimizer.step();bw=0;vjps=2;components={'weighted_private_ce':credit,'member_losses':member_losses,'unconnected':unconnected}
  elif arm=='original_gnnm':own=original_train(model,source.data,optimizer,False);bw=4;vjps=0;components={}
  elif members==1:own=source.train_step(model,optimizer,source.data,False);bw=1;vjps=0;components={}
  else:
   logits=source.logits(model,streams,True,ids);loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y.repeat(4));loss.backward();optimizer.step();own=float(loss.detach());bw=1;vjps=0;components={}
  guard()
  if any(not torch.isfinite(p).all() or (p.grad is not None and not torch.isfinite(p.grad).all()) for p in model.parameters()):raise FloatingPointError('Nonfinite native/private parameter or gradient')
  return {'objective':own,'ce':own,'regularization':0.,'objective_components':components,
   'operation_counts':{'optimizer_updates':1,'backward_calls':bw,'autograd_grad_VJP_calls':vjps,'model_forwards':members,'route_predictions':members,'supervised_examples':len(ids)}}
 def capture():
  guard();nr=np.random.get_state()
  return {'streams':copy.deepcopy(streams),'numpy_rng':[nr[0],nr[1].tolist(),*nr[2:]],'weight_sha256':weight_hash,
   'factor_members':{n:m.member for n,m in model.named_modules() if isinstance(m,source.factors.FactorLinear)},
   'training_modes':{n:m.training for n,m in model.named_modules()}}
 def restore(value):
  guard()
  if value['weight_sha256']!=weight_hash or len(value['streams'])!=members:raise ValueError('Selected fixed weight/RNG ancestry differs')
  streams[:]=copy.deepcopy(value['streams']);nr=value['numpy_rng'];np.random.set_state((nr[0],np.array(nr[1],dtype=np.uint32),*nr[2:]))
  modules=dict(model.named_modules())
  for name,member in value['factor_members'].items():modules[name].member=member
  for name,mode in value['training_modes'].items():modules[name].training=mode
 adapter=ClassifierAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits_fn,train_ids=ids,train_y=y,
  select_ids=select['ids'].to(device),select_y=select['y'].to(device),regularizer=lambda current:x.new_zeros(()),
  fusion='mean_logits' if arm=='original_gnnm' else 'mean_probabilities',selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':5001,'save_every':1},
  custom_train=custom_train,external_capture=capture,external_restore=restore,
  provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'source_provider_bindings':rendered,
   'source_contract':CONTRACT,'parameter_roles':roles,'weight_sha256':weight_hash,'actual_model_seed':source.seeds(run['seed'],run['body'])[0]})
 adapter.source_family=source;adapter.source_streams=streams;adapter.bootstrap_weights=weights;adapter.parameter_role_record=roles
 return adapter
