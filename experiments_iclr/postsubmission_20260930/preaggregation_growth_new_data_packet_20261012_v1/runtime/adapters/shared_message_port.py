"""Full source D64 meanSAGE/CSR shared-message laws on fresh role providers."""
import ast
import copy
import hashlib
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace,FunctionType

try:
 from .classifier import ClassifierAdapter
 from .native_label_families import bound
except ImportError:
 from adapters.classifier import ClassifierAdapter
 from adapters.native_label_families import bound


def import_owned(root,c):
 for row in c['shared_source_imports']:bound(root,row)
 directory=Path(root)/c['shared_source_root'];sys.path.insert(0,str(directory))
 module=importlib.import_module('shared_message_quality_v3.models')
 complete=importlib.import_module('propagation_cost_engineering_v1.complete_models')
 drop=importlib.import_module('propagation_cost_impl_v1.dropout')
 for key,value in list(sys.modules.items()):
  if key.startswith(('shared_message_quality_v3','propagation_cost_engineering_v1','propagation_cost_impl_v1')) and getattr(value,'__file__',None):
   if not Path(value.__file__).resolve().is_relative_to(directory.resolve()):raise ValueError('No historical or foreign source import permitted')
 return module,complete,drop


def prepare_operator(torch,x,edges):
 """Exact incoming record mean from factual COO; forward and explicit adjoint."""
 n=len(x);src,dst=edges[0],edges[1];degree=torch.bincount(dst,minlength=n).to(torch.float32)
 if (degree==0).any():raise ValueError('Prepared self-covered factual graph required')
 values=degree[dst].reciprocal();coo=torch.sparse_coo_tensor(torch.stack((dst,src)),values,(n,n)).coalesce()
 return coo.to_sparse_csr(),coo.transpose(0,1).coalesce().to_sparse_csr()


def make_adapter(*,torch,project_root,study,run,device):
 root=Path(project_root).resolve();c=run['config'];arm=c['family_arm']
 if study['dataset'] not in ('wikics','amazon-photo') or c['backbone']!='SOURCE_MEAN_SAGE' or c['width']!=64 or c['depth']!=2 or c['weight_decay']!=.0001:
  raise ValueError('Only source D64/L2/M4 meanSAGE on declared homogeneous providers')
 models,complete,drop=import_owned(root,c)
 data={k:torch.load(bound(root,row),map_location='cpu',weights_only=True) for k,row in c['node_inputs'].items()};graph,train,select=[data[k] for k in ('graph','train','select')]
 if set(graph)!={'x','edge_index'} or set(train)!={'ids','y'} or set(select)!={'ids','y'}:raise ValueError('Exact separate role authorities')
 x=graph['x'].to(device);edges=graph['edge_index'].to(device);ids=train['ids'].to(device);y=train['y'].to(device);members=c['members']
 if x.dtype!=torch.float32 or x.ndim!=2 or edges.dtype!=torch.long or not torch.isfinite(x).all() or torch.isin(train['ids'],select['ids']).any():raise ValueError('Source float32 facts/disjoint categorical roles')
 actual_seed=run['seed']+1000003*run['body'];family=None;ledger=None;streams=[]
 if arm=='original_gnnm':
  # The literal author TABMModel and original four-backward callback are the
  # actual GNNM reference; the optimized source exact-member path stays separate.
  native=importlib.import_module('propagation_cost_impl_v1._pinned.frozen_models')
  with torch.random.fork_rng(devices=[]):
   torch.random.default_generator.manual_seed(actual_seed)
   model=native.TABMModel('SAGE',2,x.shape[1],64,c['classes'],1,1,'LayerNorm',c['dropout'],tabm_inits=4,device='cpu').to(device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=.0001,betas=(.9,.999),eps=1e-8)
  tree=ast.parse(bound(root,c['shared_original_train']).read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_step_tabm')
  ns={'F':torch.nn.functional,'TABM_INITS':4};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_GNNM::train','exec'),ns)
  labels=torch.zeros(len(x),dtype=torch.long,device=device);mask=torch.zeros(len(x),dtype=torch.bool,device=device);labels[ids]=y;mask[ids]=True
  source_data=SimpleNamespace(x=x,edge_index=edges,y=labels,train_mask=mask)
  g=torch.Generator(device='cpu').manual_seed(actual_seed+3000007);streams=[{'cpu':g.get_state()}]
  if str(device).startswith('cuda'):streams[0]['cuda']=torch.Generator(device=device).manual_seed(actual_seed+3000007).get_state()
  from contextlib import contextmanager
  @contextmanager
  def scope():
   dev=torch.device(device);with_devices=[] if dev.type=='cpu' else [dev.index or 0]
   with torch.random.fork_rng(devices=with_devices):
    torch.set_rng_state(streams[0]['cpu'])
    if with_devices:torch.cuda.set_rng_state(streams[0]['cuda'],dev)
    try:yield
    finally:
     streams[0]['cpu']=torch.get_rng_state()
     if with_devices:streams[0]['cuda']=torch.cuda.get_rng_state(dev)
  def logits_fn(current,training,step):
   with scope():return torch.stack([current(source_data,x,tabm_seed=row) for row in range(4)])
  def train_unit(step):
   with scope():return ns['train_step_tabm'](model,source_data,optimizer,False)
  constructor_count=1
 else:
  # Source construct permits a dimensions dictionary. Its old seed allowlist
  # is replaced by the explicit new paired/body identities; no learned code changes.
  env=dict(models.construct.__globals__);env['SEEDS']=tuple(c['shared_seed_roster'])
  construct=FunctionType(models.construct.__code__,env,'construct',models.construct.__defaults__,models.construct.__closure__)
  model=construct({'features':x.shape[1],'classes':c['classes']},arm,seed=actual_seed,checkpoint_blocks=False)
  # Source p is a plain forward recipe field, not a parameter initializer.
  # Rebind every source dropout field to the declared prospective neighbor.
  model.p=c['dropout']
  for module in model.modules():
   if isinstance(module,torch.nn.Dropout):module.p=c['dropout']
  model=model.to(device);recipe={**models.RECIPE,'lr':c['learning_rate'],'dropout':c['dropout']}
  env=dict(models.make_optimizer.__globals__);env['RECIPE']=recipe
  make_opt=FunctionType(models.make_optimizer.__code__,env,'make_optimizer',models.make_optimizer.__defaults__,models.make_optimizer.__closure__);optimizer=make_opt(model)
  payload=torch.load(bound(root,c['mean_operator']),map_location='cpu',weights_only=True)
  if payload['source_graph']!=c['node_inputs']['graph'] or payload['policy']!='incoming_record_mean_float32;factualCOO;no new loops':raise ValueError('Exact source operator preprocessing ancestry')
  ledger=complete.OperationLedger();forward=payload['forward'].to(device)
  op=(complete.NoAdjointCSR('full',forward,ledger,len(x)) if arm=='detached_shared_mean' else complete.CompleteCSR('full',forward,payload['adjoint'].to(device),ledger,len(x)))
  family=complete.CompleteFamily({'full':op},{'nodes':len(x),'canonical_edges':int(forward._nnz()),'P_edge_only_fingerprint':c['mean_operator']['sha256']},ledger,device)
  def logits_fn(current,training,step):
   return models.forward_logits(current,family,x,execution='packed',dropout_key=drop.DropoutKey(c['shared_task_index'],actual_seed,step) if training else None)
  def train_unit(step):
   return float(models.train_step(model,family,x,ids,y,optimizer,execution='packed',dropout_key=drop.DropoutKey(c['shared_task_index'],actual_seed,step),return_loss=True))
  constructor_count=2
 def custom_train(adapter,step):
  if ledger is not None:ledger.clear()
  value=train_unit(step);ops=[] if ledger is None else copy.deepcopy(ledger.sparse)
  return {'objective':value,'ce':value,'regularization':0.,'operation_counts':{'optimizer_updates':1,'model_forwards':4 if arm=='original_gnnm' else 1,
   'route_predictions':members,'logical_native_route_trajectories':members,'backward_calls':4 if arm=='original_gnnm' else 1,
   'autograd_grad_VJP_calls':0,'source_sparse_forward_calls':sum(r['direction']=='forward' for r in ops),
   'source_sparse_adjoint_calls':sum(r['direction']=='backward' for r in ops),'source_sparse_edge_channels':sum(r['edge_channels'] for r in ops),'supervised_examples':len(ids)*members}}
 def capture():return {'streams':copy.deepcopy(streams),'training_modes':{n:m.training for n,m in model.named_modules()}}
 def restore(v):
  streams[:]=copy.deepcopy(v['streams']);modules=dict(model.named_modules())
  for n,mode in v['training_modes'].items():modules[n].training=mode
  if ledger is not None:ledger.clear()
 adapter=ClassifierAdapter(torch,torch.nn.functional,model,optimizer,logits_fn=logits_fn,train_ids=ids,train_y=y,
  select_ids=select['ids'].to(device),select_y=select['y'].to(device),regularizer=lambda current:x.new_zeros(()),fusion='mean_logits',
  selector={'metric':'pooled_nll','direction':'min','ties':'first','patience_steps':5001,'save_every':1},custom_train=custom_train,external_capture=capture,external_restore=restore,
  provenance={'run_id':run['run_id'],'data':study['data'],'source':study['source'],'config':c,'actual_model_seed':actual_seed,
   'source_constructor_count':constructor_count,'source_optimizer_constructions':1,'source_final_primary_selected_NLL_secondary':True,
   'new_graph_role_seed_dropout_LR_interfaces':True,'learned_forward_and_TRAIN_function_code_unchanged':True,'historical_outcomes_or_authorization_inherited':False})
 adapter.source_family=family;adapter.source_models=models
 return adapter
