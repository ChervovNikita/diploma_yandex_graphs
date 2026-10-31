"""Owned exact NCN growth lifecycle; models/targets are opened only at fit admission."""
import ast
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace

try:
 from .native_label_families import bound,load_module
 from ..engine import AdapterStepError
except ImportError:
 from adapters.native_label_families import bound,load_module
 from engine import AdapterStepError

ARMS=('no_growth','graph_growth','unfiltered_growth','capable_single_rank8','independent_graph_growth4')


def definitions(root,row,names,env):
 tree=ast.parse(bound(root,row).read_bytes());nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 if {n.name for n in nodes}!=set(names):raise ValueError('Exact source function roster changed')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(row['path'])+'::owned_source','exec'),env)
 return SimpleNamespace(**env)


def sources(root,bindings):
 for r in bindings.values():bound(root,r)
 directory=bound(root,bindings['native']).parents[2];sys.path.insert(0,str(directory))
 native=importlib.import_module('baseline_models.NCN.model')
 if Path(native.__file__).resolve()!=bound(root,bindings['native']):raise ValueError('Owned author NCN import only')
 # The source transfer module imports only its own private_adam module.
 row=bindings['private_adam'];path=bound(root,row)
 if 'private_adam' in sys.modules and Path(sys.modules['private_adam'].__file__).resolve()!=path:raise ValueError('Foreign private_adam import')
 if 'private_adam' not in sys.modules:
  spec=importlib.util.spec_from_file_location('private_adam',path);m=importlib.util.module_from_spec(spec);sys.modules['private_adam']=m;spec.loader.exec_module(m)
 values={k:load_module(root,bindings[k]) for k in ('heads','models','steps','geometry','cycle','initializer','growth')}
 custody=definitions(root,bindings['custody_functions'],('make_pair','queries','support_tensor'),{})
 warm=definitions(root,bindings['warm_train'],('own_loss','ordinary_step'),{})
 post=definitions(root,bindings['growth_train'],('own_loss','ordinary_step'),{})
 return SimpleNamespace(native=native,custody=custody,warm=warm,post=post,**values)


class MultiAdam:
 """Four genuine source Adam objects behind engine's serializable owner view."""
 def __init__(self,optimizers):
  self.optimizers=list(optimizers);self.param_groups=[g for o in optimizers for g in o.param_groups]
 @property
 def state(self):return {p:v for o in self.optimizers for p,v in o.state.items()}
 def state_dict(self):return {'source_optimizers':[o.state_dict() for o in self.optimizers]}
 def load_state_dict(self,state):
  if len(state['source_optimizers'])!=len(self.optimizers):raise ValueError('Source Adam roster changed')
  for o,s in zip(self.optimizers,state['source_optimizers']):o.load_state_dict(s)
  self.param_groups=[g for o in self.optimizers for g in o.param_groups]


def optimizer(torch,model,lr):
 return torch.optim.Adam(list(model.parameters()),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)


def original_boundary(torch,root,s,seed,factor_seed,device):
 """Literal author BEBlock at native input/output boundaries, complete NCN interiors."""
 model,_,_,partition=s.models.build(s.native,s.heads,arm='capable_single',seed=seed,factor_seed=factor_seed,device=device)
 env={'torch':torch,'nn':torch.nn};BE=definitions(root,s.bindings['boundary'],('BEBlock',),env).BEBlock
 class Boundary(torch.nn.Module):
  def __init__(self,inputs,outputs,first):
   super().__init__();self.block=BE(inputs,outputs,first,'cpu',4);self.member=0
  def forward(self,x):return self.block(x,self.member)
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(factor_seed)
  stem=model.encoder.xemb[1];last=model.predictor.lin[-1]
  if not isinstance(stem,torch.nn.Linear) or not isinstance(last,torch.nn.Linear) or (stem.in_features,stem.out_features,last.in_features,last.out_features)!=(3703,256,256,1):raise ValueError('Exact source NCN boundary sites required')
  prior={n:p for n,p in model.named_parameters() if not n.startswith(('encoder.xemb.1.','predictor.lin.'+str(len(model.predictor.lin)-1)+'.'))}
  model.encoder.xemb[1]=Boundary(3703,256,True);model.predictor.lin[-1]=Boundary(256,1,False)
 if any(dict(model.named_parameters())[n] is not p for n,p in prior.items()):raise ValueError('Every existing interior Parameter retained')
 model.to(device)
 class Bank(torch.nn.Module):
  member_count=4
  def __init__(self):super().__init__();self.native=model
  def forward(self,x,support,positive,negative,route=None):
   selected=range(4) if route is None else [route];out=[]
   for member in selected:
    if type(member) is not int or member not in range(4):raise ValueError('Original source BE route')
    a,b=model.encoder.xemb[1],model.predictor.lin[-1];old=a.member,b.member;a.member=b.member=member
    try:out.append(model(x,support,positive,negative,route=0))
    finally:a.member,b.member=old
   return torch.cat([p for p,n in out],1),torch.cat([n for p,n in out],1)
 return Bank().to(device),{'boundary_sites':['encoder.xemb.1','predictor.lin.last'],'literal_original_BEBlock':s.bindings['boundary'],'complete_native_interior_retained':True,'source_classification_to_link_BCE_adaptation':True}


def boundary_episode(torch,s,model,opt,x,support,inner,outer,streams):
 """Original own BCE/member, four backward calls then one Adam at each pass."""
 values=[]
 for index in range(3):
  opt.zero_grad(set_to_none=True);parts=[]
  for member in range(4):
   q=outer if index==1 else inner[member]
   p,n=s.steps.functional_forward(torch,model,dict(model.named_parameters()),x,support,q,training=index!=1,
      stream=(streams,member) if index!=1 else None,route=member,advance=index==2)
   value=s.post.own_loss(torch,p,n);(value/4).backward();parts.append(float(value.detach()))
  if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('Original boundary whole-body gradient invalid')
  opt.step();values.append(sum(parts)/4)
 return values


class Adapter:
 def __init__(self,torch,s,c,run,model,optimizers,x,train,select,initialization):
  self.t,self.s,self.config,self.run=torch,s,c,run;self.model=model;self.optimizers=optimizers;self.optimizer=optimizers[0] if len(optimizers)==1 else MultiAdam(optimizers)
  self.x,self.train,self.select=x,train,select;self.device=x.device;self.initialization=initialization
  self.warm=c['growth_stage']=='warm';self.independent=c['growth_arm']=='independent_graph_growth4';self.original=c['growth_arm']=='original_gnnm';self._best=None;self._best_step=None
  self.roster=list(model.routes) if self.independent else [model];self.route_seeds=[c['initialization_seed']+5*m for m in range(4)] if self.independent else [c['initialization_seed']]
  self.streams=[s.steps.DropoutStreams(torch,x.device,seed if self.warm else seed+200000) for seed in self.route_seeds]
  self.endpoint=[s.geometry.route_streams(seed,4) for seed in self.route_seeds];self.random=[s.geometry.route_streams(seed,4,control=True) for seed in self.route_seeds]
  self.full_support=s.custody.support_tensor(torch,train,list(range(len(train))),len(x),x.device)
  self.counters={'complete_source_cycles':0,'source_episodes':0,'optimizer_updates':0,'backward_calls':0}
  self.provenance={'run_id':run['run_id'],'config':copy.deepcopy(c),'initialization':copy.deepcopy(initialization),'actual_model_seed':c['initialization_seed'],'source_unit':'one complete61episode source outer cycle;183Adam transitions per optimizer owner','history_inherited':False}

 def train_step(self,step):
  self.model.train();tick={'optimizer_updates':0,'backward_calls':0,'model_forwards':0,'supervised_examples':0,'source_episodes':0};values=[]
  try:
   for route,opt,seed,drop,e,r in zip(self.roster,self.optimizers,self.route_seeds,self.streams,self.endpoint,self.random):
    negative,order,episodes=self.s.custody.make_pair(self.t,self.s.geometry,self.s.cycle,self.train,len(self.x),seed,step-1,64,256,e,r)
    if len(order)!=3870 or len(episodes)!=61:raise ValueError('Full source3870positive TRAIN61episode tail coverage required')
    for endpoint,_,kept,_ in episodes:
     support=self.s.custody.support_tensor(self.t,self.train,kept,len(self.x),self.device)
     inner,outer=self.s.custody.queries(self.t,self.train,negative,endpoint,self.device)
     result=boundary_episode(self.t,self.s,route,opt,self.x,support,inner,outer,drop) if self.original else (self.s.warm if self.warm else self.s.post).ordinary_step(self.t,self.s.steps,route,opt,self.x,support,inner,outer,drop)
     values.extend(result);tick['optimizer_updates']+=3;tick['source_episodes']+=1;tick['backward_calls']+=12 if self.original else 3
     tick['model_forwards']+=12 if self.original else 3 if route.member_count==1 else 9
     tick['supervised_examples']+=2*sum(q[0].shape[1]+q[1].shape[1] for q in inner)+(outer[0].shape[1]+outer[1].shape[1])*(4 if self.original or route.member_count==4 else 1)
     del support,inner,outer
   for k in ('source_episodes','optimizer_updates','backward_calls'):self.counters[k]+=tick[k]
   self.counters['complete_source_cycles']+=1
  except BaseException:
   # Failed in-episode transitions have unknown partial cost; the engine preserves this honestly.
   self._partial_complete_episode_counts=tick.copy();raise
  mean=sum(values)/len(values);return {'objective':mean,'own_source_pass_objective':mean,'operation_counts':tick,'source_cycles_completed':step,'selector_values_read':False}

 def validate(self,step):
  if self.warm:
   # Monotone completion is a terminal TRAIN-only custody selector, no logits or heldout values.
   return {'selection_metric':float(step),'training_completion':step,'SELECT_values_read':False,'terminal_warm_custody':True}
  self.model.eval();p,n=self.score(self.select['positive'],self.select['negative'])
  rank=1+.5*((n.mean(2)>=p.mean(1)[:,None]).sum(1)+(n.mean(2)>p.mean(1)[:,None]).sum(1))
  mrr=round(float(rank.float().reciprocal().mean()),4)
  return {'selection_metric':mrr,'MRR':mrr,'queries':len(p),'negative_pool_width':n.shape[1],'SELECT_values_read':True,'source_metric_round4':True}

 def observe_validation(self,step,metrics):
  score=metrics['selection_metric'];above=self._best is None or score>self._best
  if above:self._best,self._best_step=score,step
  return {'save_selected':above,'stop_requested':False,'best_saved_metric':self._best,'best_saved_step':self._best_step,'checkpoint_eligible':True,'stale_steps':0}

 def score(self,positive,negative,chunk=256):
  torch=self.t;outp=[];outn=[]
  with torch.no_grad():
   for start in range(0,len(positive),chunk):
    p=positive[start:start+chunk].t().to(self.device);pool=negative[start:start+chunk];n=pool.permute(2,0,1).reshape(2,-1).to(self.device)
    a,b=self.s.steps.functional_forward(torch,self.model,dict(self.model.named_parameters()),self.x,self.full_support,(p,n),training=False)
    outp.append(a.cpu());outn.append(b.reshape(len(p[0]),pool.shape[1],self.model.member_count).cpu())
  return torch.cat(outp),torch.cat(outn)

 def capture_state(self):
  import numpy as np
  state=np.random.get_state()
  return {'streams':[s.state_dict() for s in self.streams],'endpoint':[[{k:v.getstate() for k,v in row.items()} for row in rows] for rows in self.endpoint],
    'random':[[{k:v.getstate() for k,v in row.items()} for row in rows] for rows in self.random],'counters':copy.deepcopy(self.counters),'best':self._best,'best_step':self._best_step,
    'global_torch_cpu':self.t.get_rng_state(),'global_python':random.getstate(),'numpy_rng':[state[0],state[1].tolist(),*state[2:]],
    'global_torch_cuda':self.t.cuda.get_rng_state_all() if self.t.cuda.is_initialized() else [],'initialization':copy.deepcopy(self.initialization),'provenance':copy.deepcopy(self.provenance),
    'boundary_members':{n:m.member for n,m in self.model.named_modules() if hasattr(m,'member')}}

 def restore_state(self,v):
  import numpy as np
  try:from engine import _equal_tree
  except ImportError:from ..engine import _equal_tree
  if not _equal_tree(self.t,v['provenance'],self.provenance) or not _equal_tree(self.t,v['initialization'],self.initialization):raise ValueError('Exact paid lifecycle/source ancestry required')
  for stream,state in zip(self.streams,v['streams']):
   stream.rows=[{k:(x.clone() if self.t.is_tensor(x) else x) for k,x in state[str(i)].items()} for i in range(len(state))]
  for targets,states in ((self.endpoint,v['endpoint']),(self.random,v['random'])):
   for rows,savedrows in zip(targets,states):
    for row,saved in zip(rows,savedrows):
     for k in row:row[k].setstate(saved[k])
  self.counters=copy.deepcopy(v['counters']);self._best,self._best_step=v['best'],v['best_step'];self.t.set_rng_state(v['global_torch_cpu']);random.setstate(v['global_python'])
  nr=v['numpy_rng'];np.random.set_state((nr[0],np.array(nr[1],dtype=np.uint32),*nr[2:]))
  if v['global_torch_cuda']:self.t.cuda.set_rng_state_all(v['global_torch_cuda'])
  modules=dict(self.model.named_modules())
  for n,member in v['boundary_members'].items():modules[n].member=member


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
 root=Path(project_root).resolve();c=run['config'];stage=c['growth_stage'];arm=c['growth_arm'];s=sources(root,c['growth_sources']);s.bindings=c['growth_sources'];dev=torch.device(device)
 if study['dataset']!='citeseer-heart' or stage not in ('warm','post','cold_control') or (stage=='post' and arm not in ARMS):raise ValueError('Exact independent CiteSeer source lifecycle')
 graph=torch.load(bound(root,c['link_inputs']['graph']),map_location='cpu',weights_only=True);trainpack=torch.load(bound(root,c['link_inputs']['train']),map_location='cpu',weights_only=True)
 if set(graph)!={'x','train_support_edge_index'} or set(trainpack)!={'positive'} or graph['x'].shape!=(3327,3703) or trainpack['positive'].shape!=(3870,2):raise ValueError('Exact source GRAPH/TRAIN geometry')
 x=graph['x'].to(dev);train=trainpack['positive'];select=None if stage=='warm' else torch.load(bound(root,c['link_inputs']['select']),map_location='cpu',weights_only=True)
 seed=c['initialization_seed'];factor_seed=c['factor_seed'];initialization={'stage':stage,'arm':arm,'fresh_constructor':True,'source_initialization_calls':0}
 if stage=='post':
  parents=(execution_context or {}).get('paid_parent_checkpoints',{});ids=c['lifecycle']['parent_fit_ids']
  if len(ids)!=1 or set(parents)!=set(ids):raise ValueError('Own completed paid native warm parent required')
  parent=parents[ids[0]];pc=parent['contract'];receipt=parent['receipt']
  if pc['config']['growth_stage']!='warm' or pc['budget']['maximum']!=20 or receipt['step']!=20 or pc['config']['initialization_seed']!=seed or pc['config']['learning_rate']!=c['learning_rate']:
   raise ValueError('Terminal20 same-source/same-seed/same-LR native acquisition required')
  warm,_,_,_=s.models.build(s.native,s.heads,arm='capable_single',seed=seed,factor_seed=factor_seed,device=dev)
  payload=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
  if payload['fit_id']!=ids[0] or payload['step']!=20:raise ValueError('Source terminal20 checkpoint authority')
  warm.load_state_dict(payload['model_state'],strict=True);basis={};calibration={};basisrecord={}
  restored=(execution_context or {}).get('growth_restored_initialization')
  if arm!='no_growth':
   if restored is not None:
    basis={k:v.to(dev) for k,v in restored['initial_basis'].items()};calibration=restored['calibration'];basisrecord=restored['basis_record']
   else:
    draw=torch.load(bound(root,c['calibration_draw']),map_location='cpu',weights_only=False)
    if draw['seed']!=seed or draw['cycle_index']!=0 or draw['GRAPH']!=c['link_inputs']['graph'] or draw['TRAIN']!=c['link_inputs']['train']:raise ValueError('Fixed new TRAIN-only calibration draw authority')
    negative,episodes=draw['negative'],draw['episodes'];endpoint,_,kept,desc=episodes[0];support=s.custody.support_tensor(torch,train,kept,len(x),dev);_,queries=s.custody.queries(torch,train,negative,endpoint,dev)
    h,g,calibration=s.growth.calibrate(torch,warm,x,support,queries)
    required='single' if arm=='capable_single_rank8' else 'unfiltered' if arm=='unfiltered_growth' else 'graph'
    basis,basisrecord=s.growth.bases(torch,warm.encoder,h,g,support,required=required);del h,g,support,queries,negative,episodes
  model,partition=s.growth.make(torch,s.native,s.heads,s.models,s.initializer,warm,arm,seed,factor_seed,dev,basis)
  initialization.update(parent_fit=ids[0],parent_checkpoint=receipt,parent_budget=pc['budget'],parent_TERMINAL20=True,
    calibration=calibration,basis_record=basisrecord,initial_basis={k:v.detach().cpu() for k,v in basis.items()},partition=partition,
    source_initialization_calls={'donor_reconstruction_bodies':1,'new_source_bodies':4 if arm=='independent_graph_growth4' else 1,'TRAIN_gradient_calibrations':int(arm!='no_growth' and restored is None),'SVD_count':len(basisrecord.get('records',[])) if restored is None else 0,'heldout_basis_guidance':False})
  del warm
 elif arm=='original_gnnm':model,proof=original_boundary(torch,root,s,seed,factor_seed,dev);initialization.update(boundary=proof)
 else:model,_,_,partition=s.models.build(s.native,s.heads,arm='capable_single',seed=seed,factor_seed=factor_seed,device=dev);initialization.update(partition=partition)
 if stage=='post' and restored is not None:initialization=copy.deepcopy(restored)
 roster=list(model.routes) if arm=='independent_graph_growth4' else [model];opts=[optimizer(torch,m,c['learning_rate']) for m in roster]
 return Adapter(torch,s,c,run,model,opts,x,train,select,initialization)
