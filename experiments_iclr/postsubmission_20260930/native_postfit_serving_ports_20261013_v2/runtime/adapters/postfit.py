"""Frozen native head refits and learned operators, using literal source kernels."""
import ast,copy,hashlib,math,time,types
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from .common_native_wrapper import bound,load_module
from . import common_native_wrapper,endpoint_value,label_corrector
try:
 from ..postfit_semantics import CONTRACT,validate_postfit_run,validate_lifecycle
 from ..engine import restore_checkpoint
except ImportError:
 from postfit_semantics import CONTRACT,validate_postfit_run,validate_lifecycle
 from engine import restore_checkpoint

def require(v,message):
 if not v:raise ValueError(message)

def literal(root,row,names,env=None,select_rows=None):
 nodes=[copy.deepcopy(n) for n in ast.parse(bound(root,row).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 require({n.name for n in nodes}==set(names),'Exact literal source definitions required')
 if select_rows is not None:
  class Shape(ast.NodeTransformer):
   def visit_Constant(self,n):return ast.copy_location(ast.Constant(select_rows),n) if n.value==5274 else n
  nodes=[Shape().visit(n) for n in nodes]
 ns={'require':require,**(env or {})};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),row['path']+'::source_AST','exec'),ns);return SimpleNamespace(**ns),ns

def receipt(root,path):
 b=Path(path).read_bytes();return {'path':str(Path(path).resolve().relative_to(Path(root).resolve())),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def attach_feature_finalizer(adapter,torch,root,study,run,graph,train,select,head_getter,full_forward):
 c=run['config'];adapter.provenance=getattr(adapter,'provenance',{'config':c,'actual_model_seed':c['initialization_seed']})
 def finalize(self,destination,result):
  start=time.perf_counter();chosen=result['selected_checkpoint'];restore_checkpoint(torch,destination/'selected.pt',self,expected_fit_id=run['run_id']);self.model.eval()
  captured=[];head=head_getter();hook=None
  if head is not None:hook=head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().clone()))
  try:
   with torch.no_grad():Z=full_forward().detach().cpu().clone()
  finally:
   if hook is not None:hook.remove()
  require(Z.dtype==torch.float32 and Z.shape==(c['members'],11701,10) and bool(torch.isfinite(Z).all()),'One complete selected FP32 source output collection')
  bundle={'schema':'paid-frozen-native-fullgraph-v1','cohort_id':c['postfit_cohort_id'],'donor_fit_id':run['run_id'],'selected_checkpoint':chosen,'config':c,'Z':Z,'train_ids':train['ids'].cpu(),'train_y':train['y'].cpu(),'select_ids':select['ids'].cpu(),'select_y':select['y'].cpu(),'native_fusion':c['fusion']}
  if head is not None:
   require(len(captured)==c['members'] and all(v.shape==(11701,c['width']) and v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in captured),'Exactly one full native pre-head representation per source route')
   bundle['H']=torch.stack(captured);bundle['head']={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};bundle['head_members']=c['members'];bundle['head_factorized']=hasattr(head,'r')
  if c['postfit_family']=='serving':
   ops,_=literal(root,c['postfit_sources']['neighbor'],('neighbor_operator',))
   P,isolate=ops.neighbor_operator(torch,graph['edge_index'],len(graph['x']));p=Z.double().log_softmax(-1).exp();flat=p.permute(1,0,2).reshape(len(graph['x']),-1);bundle['neighbor']=torch.sparse.mm(P,flat).reshape(len(graph['x']),c['members'],10).permute(1,0,2).contiguous();bundle['isolates']=isolate
  q=destination/'frozen_features.pt';require(not q.exists(),'No feature acquisition overwrite');torch.save(bundle,q);f=receipt(root,q)
  counts={'optimizer_updates':0,'model_forwards':c['members'],'route_predictions':c['members'],'backward_calls':0,'supervised_examples':0,'postfit_selected_restore_calls':1,'postfit_feature_files':1,'postfit_feature_bytes':f['bytes'],'postfit_native_member_feature_forwards':c['members']}
  for k,v in counts.items():result['operation_counts'][k]=result['operation_counts'].get(k,0)+v
  result['costs']['postfit_feature_collection_seconds']=time.perf_counter()-start;result['postfit_features']={'receipt':f,'selected_checkpoint':chosen,'cohort_id':c['postfit_cohort_id'],'label_roles':['train','select'],'fullgraph_features_label_free':True,'operation_counts':counts};return result
 adapter.finalize_result=types.MethodType(finalize,adapter);return adapter

def native_adapter(torch,root,study,run,device):
 c=run['config'];graph,train,select=[torch.load(bound(root,c['node_inputs'][role]),map_location='cpu',weights_only=True) for role in ('graph','train','select')]
 if c['postfit_family']=='direct':
  arm=c['postfit_native_arm'];B=c['label_corrector_sources']
  if arm=='be_unit_contrastive':
   common=load_module(root,B['integration_common']);_,_,driver,_,_=common.dependencies();public=load_module(root,c['postfit_sources']['public']);n=public.Session('wikics',arm,c['initialization_seed'],device,bound(root,B['native']))
   class Coherent(torch.nn.Module):
    def __init__(self):super().__init__();self.native=n.model
   state=SimpleNamespace(common=common,integration=None,driver=driver,native=n,core=None,boundary=None,model=Coherent(),optimizer=label_corrector.OptimizerFacade(n,None))
  else:state=label_corrector.construct(torch,root,B,arm,c['initialization_seed'],device,graph,train)
  adapter=label_corrector.Adapter(torch,state,graph,train,select,arm,{'study':study['study_id'],'run_id':run['run_id'],'actual_model_seed':c['initialization_seed'],'config':c,'contract':CONTRACT})
  if arm=='be_unit_contrastive':
   def step(self,epoch):
    n=self.native;require(epoch==n.steps+1,'Literal complete epoch order');transition=0
    if epoch==101:self.local_transition();transition=1
    v=n.train_step(self.train_batch,self.labels);self.parameter_epoch=epoch
    counts={'optimizer_updates':1,'model_forwards':8,'route_predictions':8,'backward_calls':1,'supervised_examples':8*len(self.labels),'source_native_TRAIN_forwards':8,'source_native_TRAIN_backwards':1,'source_native_Adam_maps':1,'source_coherent_local_restores':transition}
    for k,val in counts.items():self.counters[k]=self.counters.get(k,0)+val
    return {'objective':float(v['loss']),'ce':float(v['own_mean']),'regularization':float(v['loss'])-float(v['own_mean']),'operation_counts':counts}
   adapter.train_step=types.MethodType(step,adapter)
  n=adapter.native
  def head():
   if arm=='original_GNNM':return None
   body=n.model.models[0].body;return body.pred_global if body._global else body.pred_local
  def forward():
   old=state.common.native_rng_state(n)
   with torch.no_grad():z,_=n.forward(adapter.full_batch)
   state.common.require_same_native_rng(n,old,'selected feature collection');return z
 else:
  adapter=(endpoint_value if c['postfit_family']=='affine' and not c.get('is_original_gnnm') else common_native_wrapper).make_adapter(torch=torch,project_root=root,study=study,run=run,device=device);adapter.selector['patience_steps']=300
  def head():
   if c['postfit_family']=='serving' or c.get('is_original_gnnm'):return None
   rows=[m for name,m in adapter.model.named_modules() if name.split('.')[-1]=='output_linear'];require(len(rows)==1,'One literal native affine head owner');return rows[0]
  def forward():return adapter.logits_fn(adapter.model,False,0)
 return attach_feature_finalizer(adapter,torch,root,study,run,graph,train,select,head,forward)

def load_data(torch,run,context):
 c=run['config'];v=c['lifecycle'];parents=context['paid_parent_checkpoints'];require(set(parents)==set(v['parent_fit_ids']),'All native feature cohort receipts required');parts=[]
 for i in v['donor_fit_ids']:
  p=parents[i];a=torch.load(p['features'],map_location='cpu',weights_only=True)
  require(a['cohort_id']==c['postfit_cohort_id'] and a['donor_fit_id']==i and a['selected_checkpoint']==p['receipt'] and a['config']==p['contract']['config'],'Immutable own selected feature authority required');parts.append(a)
 require(all(torch.equal(a['train_ids'],parts[0]['train_ids']) and torch.equal(a['train_y'],parts[0]['train_y']) and torch.equal(a['select_ids'],parts[0]['select_ids']) and torch.equal(a['select_y'],parts[0]['select_y']) for a in parts),'Exact ordered TRAIN/SELECT across all own-bank donors')
 data={k:parts[0][k] for k in ('train_ids','train_y','select_ids','select_y')}
 for k in ('H','Z','neighbor'):
  if k in parts[0]:data[k]=torch.cat([a[k] for a in parts])
 data['parts']=parts;return data

class HeadAdapter:
 def __init__(self,torch,model,optimizer,data,run,provenance):
  self.t,self.model,self.optimizer,self.data,self.run,self.provenance=torch,model,optimizer,data,run,provenance;self.best=None;self.best_step=None;self.stale=0;self.completed=0;self.last_receipt={}
 def capture_state(self):return {'provenance':copy.deepcopy(self.provenance),'best':self.best,'best_step':self.best_step,'stale':self.stale,'completed':self.completed,'receipt':copy.deepcopy(self.last_receipt),'torch_cpu':self.t.get_rng_state().clone()}
 def restore_state(self,v):
  require(v['provenance']==self.provenance,'Exact selected source feature ancestry required');self.best,self.best_step,self.stale,self.completed,self.last_receipt=v['best'],v['best_step'],v['stale'],v['completed'],copy.deepcopy(v['receipt']);self.t.set_rng_state(v['torch_cpu'])
 def observe_validation(self,step,v):
  fixed=self.run['config']['postfit_family'] in ('direct','serving');score=v['selection_metric'];improved=self.best is None or score>self.best
  if improved:self.best,self.best_step,self.stale=score,step,0
  else:self.stale+=1
  return {'save_selected':improved,'stop_requested':not fixed and self.stale>=300,'checkpoint_eligible':True,'best_saved_metric':self.best,'best_saved_step':self.best_step,'stale_steps':self.stale}

class DirectHead(HeadAdapter):
 def __init__(self,torch,root,data,run,provenance):
  import numpy as np
  src,env=literal(root,run['config']['postfit_sources']['direct'],('train_rms','prepare','parameters','effective','logits','objective','fit'),select_rows=len(data['select_ids']))
  H=data['H'];part=data['parts'];a={'H_train':H[:,data['train_ids']].numpy(),'H_dev':H[:,data['select_ids']].numpy(),'y_train':data['train_y'].numpy(),'y_dev':data['select_y'].numpy(),'train_ids':data['train_ids'].numpy(),'dev_ids':data['select_ids'].numpy()}
  if part[0]['head_factorized']:
   head=part[0]['head'];a.update(W=head['weight'].numpy(),r=head['r'].numpy(),s=head['s'].numpy(),bias=head['bias'].numpy())
  else:a.update(A=torch.stack([p['head']['weight'] for p in part]).numpy(),bias=torch.stack([p['head']['bias'] for p in part]).numpy())
  self.prepared=src.prepare(np,a);p=src.parameters(torch,self.prepared,run['config']['postfit_condition'])
  class Owned(torch.nn.Module):
   def __init__(self):
    super().__init__()
    for k,v in p.items():self.register_parameter(k,torch.nn.Parameter(v))
   def ordered(self):return OrderedDict(self.named_parameters())
  model=Owned();optimizer=torch.optim.LBFGS(list(model.parameters()),lr=1.,max_iter=1000,max_eval=50000,tolerance_grad=1e-6,tolerance_change=0.,history_size=10,line_search_fn='strong_wolfe')
  env['parameters']=lambda t,a,condition:model.ordered();self.src,self.np=src,np
  class Optim:
   def LBFGS(_,params,**kwargs):
    require([id(p) for p in params]==[id(p) for p in model.parameters()] and kwargs==dict(lr=1.,max_iter=1000,max_eval=50000,tolerance_grad=1e-6,tolerance_change=0.,history_size=10,line_search_fn='strong_wolfe'),'Literal solver args and registered source parameter order');return optimizer
  class Proxy:
   optim=Optim()
   def __getattr__(_,name):return getattr(torch,name)
  self.proxy=Proxy();super().__init__(torch,model,optimizer,data,run,provenance)
 def train_step(self,step):
  require(step==1 and self.completed==0,'Exactly one source LBFGS trajectory');endpoint,r=self.src.fit(self.proxy,self.np,self.prepared,self.run['config']['postfit_condition'],time.monotonic()+3590);self.last_receipt=r;self.completed=step;calls=r['actual_closure_calls_including_final_check'];m=self.run['config']['members']
  return {'objective':r['TRAIN_objective'],'ce':r['TRAIN_CE'],'regularization':r['effective_L2_penalty'],'operation_counts':{'optimizer_updates':1,'model_forwards':m*(calls+2),'route_predictions':m*(calls+2),'backward_calls':calls+1,'supervised_examples':m*580*(calls+2),'source_LBFGS_iterations':r['LBFGS_iterations'],'source_LBFGS_function_evaluations':r['optimizer_function_evaluations'],'source_LBFGS_closures_including_final_check':calls,'postfit_backbone_updates':0}}
 def predict_rows(self,ids,step=1):
  p=self.model.ordered();H=self.data['H'][:,ids].double()/self.t.from_numpy(self.prepared['scale']);return self.src.logits(self.t,H,self.src.effective(p),p['bias'])
 def validate(self,step):
  r=self.last_receipt;return {'objective':r['TRAIN_objective'],'ce':r['TRAIN_CE'],'regularization':r['effective_L2_penalty'],'selection_metric':float(step),'training_completion':step,'source_solver_receipt':r,'operation_counts':{'optimizer_updates':0,'model_forwards':0,'route_predictions':0,'backward_calls':0,'supervised_examples':0}}

class AffineHead(HeadAdapter):
 def __init__(self,torch,root,data,run,provenance):
  factors=load_module(root,run['config']['postfit_sources']['factors']);heads=[]
  for p in data['parts']:
   with torch.random.fork_rng(devices=[]):h=factors.FactorLinear(torch.nn.Linear(128,10,bias=True),p['head_members'])
   h.load_state_dict(p['head'],strict=True);heads.append(h)
  src,_=literal(root,run['config']['postfit_sources']['affine'],('delta_head',),{'copy':copy});model=src.delta_head(torch,heads,len(data['H']),run['config']['postfit_condition']);params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=0)
  with torch.no_grad():
   for ids in (data['train_ids'],data['select_ids']):require(torch.count_nonzero(model(data['H'][:,ids])).item()==0 and torch.equal(data['Z'][:,ids]+model(data['H'][:,ids]),data['Z'][:,ids]),'Literal source exact zero delta and native anchors')
  super().__init__(torch,model,optimizer,data,run,provenance)
 def predict_rows(self,ids,step=0):return self.data['Z'][:,ids]+self.model(self.data['H'][:,ids])
 def train_step(self,step):
  require(step==self.completed+1,'One source head AdamW map per update');self.optimizer.zero_grad(set_to_none=True);z=self.predict_rows(self.data['train_ids']);loss=self.t.nn.functional.cross_entropy(z.flatten(0,1),self.data['train_y'].repeat(len(z)));require(bool(self.t.isfinite(loss)),'Finite source own CE');loss.backward();require(all(p.grad is not None and bool(self.t.isfinite(p.grad).all()) for p in self.model.parameters() if p.requires_grad),'Finite exact head gradients');self.optimizer.step();self.completed=step;m=len(z);f=2*m if self.run['config']['postfit_condition']=='BE_head_continuation' else m
  return {'objective':float(loss.detach()),'ce':float(loss.detach()),'regularization':0.,'operation_counts':{'optimizer_updates':1,'model_forwards':f,'route_predictions':m,'backward_calls':1,'supervised_examples':len(self.data['train_y'])*m,'postfit_backbone_updates':0}}
 def validate(self,step):
  with self.t.no_grad():
   z=self.predict_rows(self.data['select_ids']);p=z.softmax(-1).mean(0);accuracy=float(p.argmax(-1).eq(self.data['select_y']).float().mean());ce=float(self.t.nn.functional.cross_entropy(z.flatten(0,1),self.data['select_y'].repeat(len(z))))
  m=len(z);return {'objective':ce,'ce':ce,'regularization':0.,'selection_metric':accuracy,'accuracy':accuracy,'operation_counts':{'optimizer_updates':0,'model_forwards':2*m if self.run['config']['postfit_condition']=='BE_head_continuation' else m,'route_predictions':m,'backward_calls':0,'supervised_examples':0}}

def fold_assignment(torch,select_ids,query_ids=None):
 gen=torch.Generator(device='cpu').manual_seed(11709);perm=torch.randperm(len(select_ids),generator=gen);folds=torch.empty(len(select_ids),dtype=torch.long);folds[perm]=torch.arange(len(select_ids))%5
 if query_ids is None:return folds
 assigned=(query_ids.cpu()+11709)%5;lookup={int(i):int(k) for i,k in zip(select_ids,folds)}
 return torch.tensor([lookup.get(int(i),int(k)) for i,k in zip(query_ids,assigned)],dtype=torch.long)

class ServingHead(HeadAdapter):
 def __init__(self,torch,root,data,run,provenance):
  src,_=literal(root,run['config']['postfit_sources']['operators'],('features','make_operator'),{'math':math});rule=run['config']['postfit_rule'];log=data['Z'].double().log_softmax(-1);p=log.exp();context=src.features(torch,p,data['neighbor'] if rule=='reliability_graph' else p);model=src.make_operator(torch,rule,len(p),10);optimizer=torch.optim.Adam(model.parameters(),lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0);self.log,self.prob,self.context=log,p,context;self.folds=fold_assignment(torch,data['select_ids']);self.fit_ids=data['select_ids'][self.folds!=run['config']['postfit_fold']];self.fit_y=data['select_y'][self.folds!=run['config']['postfit_fold']];super().__init__(torch,model,optimizer,data,run,provenance)
 def train_step(self,step):
  require(step==self.completed+1,'Exactly500 fixed source operator maps');self.optimizer.zero_grad(set_to_none=True);q,penalty=self.model(self.log[:,self.fit_ids],self.prob[:,self.fit_ids],self.context[:,self.fit_ids]);ce=self.t.nn.functional.nll_loss(q,self.fit_y);loss=ce+penalty;require(bool(self.t.isfinite(loss)),'Finite source aggregator loss');loss.backward();require(all(p.grad is not None and bool(self.t.isfinite(p.grad).all()) for p in self.model.parameters()),'Finite source aggregator gradients');self.optimizer.step();self.completed=step
  return {'objective':float(loss.detach()),'ce':float(ce.detach()),'regularization':float(penalty.detach()),'operation_counts':{'optimizer_updates':1,'model_forwards':1,'route_predictions':len(self.prob),'backward_calls':1,'supervised_examples':len(self.fit_ids),'postfit_backbone_updates':0}}
 def predict_rows(self,ids,step=500):return self.model(self.log[:,ids],self.prob[:,ids],self.context[:,ids])[0].unsqueeze(0)
 def validate(self,step):
  require(step==500,'Held fold cannot select epoch');return {'objective':0.,'ce':0.,'regularization':0.,'selection_metric':float(step),'training_completion':step,'operation_counts':{'optimizer_updates':0,'model_forwards':0,'route_predictions':0,'backward_calls':0,'supervised_examples':0}}

def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
 validate_postfit_run(run)
 if run['config']['postfit_phase']=='native':return native_adapter(torch,project_root,study,run,device)
 require(execution_context is not None,'Paid frozen source context required');data=load_data(torch,run,execution_context);provenance={'study_id':study['study_id'],'run_id':run['run_id'],'config':run['config'],'source_feature_receipts':{i:p['feature_receipt'] for i,p in execution_context['paid_parent_checkpoints'].items()},'contract':CONTRACT}
 torch.set_num_threads(1)
 return {'direct':DirectHead,'affine':AffineHead,'serving':ServingHead}[run['config']['postfit_family']](torch,project_root,data,run,provenance)
