"""Exact full-General native NSD methods on independent binary role providers."""
import copy,json,random
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from .nsd_sources import environment,extract
from .native_label_families import bound
from .classifier import ClassifierAdapter

ARMS=('assigned_fixed','assigned_resampled','layer_shared_incidence','node_capacity_right','joint_multi_transport_single','ordinary_M1','factorized_M1','original_gnnm')


def helpers(torch,root,B):
 import numpy as np
 return extract(root,B['baseline'],('seed_all','capture_rng','restore_rng','evaluation_rng'),{'random':random,'contextmanager':contextmanager})


def factory(torch,root,B,c,graph,device):
 e=environment(torch,root,B);p=json.loads(bound(root,B['live_policy']).read_bytes());args={**p['configuration']['native_args'],'input_dim':graph['x'].shape[1],'output_dim':2,'graph_size':len(graph['x']),'device':str(device)}
 args['dropout']=c.get('dropout',.2)
 f=e.placement.make_native_placed_factory(torch,e.adapter,graph['edge_index'],graph['edge_index'].to(device),args)
 return e,f,args


def original_bank(torch,root,B,e,f,seed):
 original=f();nn=torch.nn;BE=extract(root,B['boundary'],('BEBlock',),{'torch':torch,'nn':nn}).BEBlock
 class Boundary(nn.Module):
  def __init__(self,a,b,first):super().__init__();self.block=BE(a,b,first,'cpu',4);self.member=0
  def forward(self,x):return self.block(x,self.member)
 prior={n:p for n,p in original.named_parameters() if not n.startswith(('lin1.','lin2.'))}
 original.lin1=Boundary(original.input_dim,original.hidden_dim,True);original.lin2=Boundary(original.hidden_dim,2,False);original.to(original.device)
 if any(dict(original.named_parameters())[n] is not p for n,p in prior.items()):raise ValueError('Retain every actual native NSD interior Parameter')
 class Bank(nn.Module):
  members_count=4
  def __init__(self):super().__init__();self.body=original
  def forward_route(self,m,x):
   old=original.lin1.member,original.lin2.member;original.lin1.member=original.lin2.member=m;raw=[]
   hook=original.lin2.register_forward_hook(lambda module,inputs,output:raw.append(output))
   try:
    original(x)
    if len(raw)!=1:raise ValueError('One literal native raw head')
    return raw[0]
   finally:hook.remove();original.lin1.member,original.lin2.member=old
  def forward(self,x):return torch.stack([self.forward_route(m,x) for m in range(4)])
 return Bank().to(original.device)


def construct(torch,root,B,e,f,c,seed):
 a=c['nsd_arm'];torch.random.default_generator.manual_seed(seed)
 if torch.device(c['device']).type=='cuda':torch.cuda.manual_seed(seed)
 if a=='original_gnnm':return original_bank(torch,root,B,e,f,seed)
 if a in ('geometry_warm','geometry_branch','geometry_only_source'):return e.geometry.make_bank(torch,e.adapter,f)
 if a=='ordinary_M1':return f()
 kwargs={'members':1 if a=='factorized_M1' else 4}
 if a.startswith('assigned_'):kwargs['assignment']=a.removeprefix('assigned_')
 if a=='layer_shared_incidence':kwargs['layer_shared']=True
 if a=='node_capacity_right':kwargs['capacity_control']=True
 bank=e.adapter.SharedNativeBank(f,**kwargs)
 if a=='joint_multi_transport_single':return e.adapter.JointMultiTransportSingle(bank,classes=2,width=64)
 return bank


def groups(model,e,c):
 if c['nsd_arm']=='original_gnnm':return [{'params':[p for p in model.parameters() if p.requires_grad],'weight_decay':0.}]
 if c['nsd_arm']=='ordinary_M1':
  a,b=model.grouped_parameters();return [{'params':a,'weight_decay':.0005},{'params':b,'weight_decay':.0005}]
 return model.parameter_groups(.0005,.0005)


def optimizer(torch,model,e,c):
 gs=groups(model,e,c)
 return torch.optim.AdamW(gs,lr=c['learning_rate'],weight_decay=0.) if c['nsd_arm']=='original_gnnm' else torch.optim.Adam(gs,lr=c['learning_rate'])


def factors(model):
 return [v for member in model.members for learner in member.sheaf_learners for v in (learner.linear1.r,learner.linear1.s)]


def plain_capture(torch,model):
 return {n:{k:v.detach().cpu().clone() if torch.is_tensor(v) else None for k,v in vars(m).items() if k=='L'} for n,m in model.named_modules() if 'L' in vars(m)}


def plain_restore(model,state,device):
 modules=dict(model.named_modules())
 for n,attrs in state.items():
  for k,v in attrs.items():setattr(modules[n],k,None if v is None else v.to(device).clone())


class NSDAdapter(ClassifierAdapter):
 def probabilities(self,z):
  return z.exp().mean(0) if self.fusion=='mean_probabilities' else super().probabilities(z)
 def validate(self,step):
  import numpy as np
  from sklearn.metrics import roc_auc_score
  self.model.eval();epoch=self.selected_parent_epoch if step==0 and hasattr(self,'selected_parent_epoch') else step+(100 if self.phase=='geometry_branch' else 0)
  with self.h.evaluation_rng(np,self.t,self.seed,epoch),self.t.no_grad():
   z=self.logits_fn(self.model,False,step);p=self.probabilities(z)[self.select_ids];logp=p.log();v=self.e.baseline.metrics(self.t,roc_auc_score,logp,self.select_y)
  return {**v,'objective':v['nll'],'ce':v['nll'],'regularization':0.,'selection_metric':v['auroc'],'source_epoch':epoch,'fusion':self.fusion,'operation_counts':{'model_forwards':self.native_paths,'route_predictions':self.members,'optimizer_updates':0}}
 def observe_validation(self,step,v):
  key=(v['auroc'],-v['nll'],-v['source_epoch']);above=self._best_key is None or key>self._best_key
  if above:self._best_key=key;self._best_step=step
  return {'save_selected':above,'stop_requested':False,'best_saved_metric':None if self._best_key is None else self._best_key[0],'best_saved_step':self._best_step,'stale_steps':0,'checkpoint_eligible':True}
 def source_frozen_factor_ownership(self):
  if self.phase not in ('geometry_warm','geometry_branch'):return set()
  fs=factors(self.model);frozen=[p for p in fs if not p.requires_grad]
  if frozen:
   if len(frozen)!=len(fs) or any(p.grad is not None or p in self.optimizer.state or not bool(self.t.all(p==1)) for p in fs):raise ValueError('Exact source identity warm factors,frozen,no gradients/moments')
  return {id(p) for p in frozen}


def make_adapter(*,torch,project_root,study,run,device,execution_context=None):
 import numpy as np
 root=Path(project_root).resolve();c=copy.deepcopy(run['config']);c['device']=str(device);B=c['nsd_sources'];data={k:torch.load(bound(root,v),map_location='cpu',weights_only=True) for k,v in c['node_inputs'].items()};graph,train,select=[data[k] for k in ('graph','train','select')]
 if study['dataset'] not in ('twitch-de','twitch-de-musae-raw') or c['classes']!=2 or set(graph)!={'x','edge_index'} or torch.isin(train['ids'],select['ids']).any():raise ValueError('Exact public binary NSD roles')
 e,f,args=factory(torch,root,B,c,graph,device);h=helpers(torch,root,B);seed=c['initialization_seed'];h.seed_all(np,torch,seed);model=construct(torch,root,B,e,f,c,seed);phase=c['nsd_arm'];pending=None;parent=None
 if phase in ('layer_shared_incidence','node_capacity_right'):
  from .nsd_spd_power import bind_constant_incidence_adjoint
  bind_constant_incidence_adjoint(torch,model)

 if phase=='geometry_warm':
  for p in factors(model):p.requires_grad_(False)
 opt=optimizer(torch,model,e,c);x=graph['x'].to(device);ids,y=train['ids'].to(device),train['y'].to(device)
 if phase=='geometry_branch':
  context=execution_context or {};parents=context.get('paid_parent_checkpoints',{});pid=c['lifecycle']['parent_fit_ids'][0]
  if set(parents)!={pid}:raise ValueError('Own paid100 native geometry warm')
  parent=parents[pid];terminal=torch.load(parent['checkpoint'],map_location='cpu',weights_only=False)
  if terminal['step']!=100 or parent['contract']['config']['nsd_arm']!='geometry_warm':raise ValueError('Exact terminal100 source warm required')
  screen=context['registry'].run(study['study_id'],c['screen_parent'])
  if screen['state']!='complete':raise ValueError('Whole common17forward TRAIN-only source screen required')
  from engine import checkpoint_receipt
  import hashlib
  record=screen['result'];path=Path(context['output_root'])/study['study_id']/c['screen_parent']/record['screen_file']['path']
  raw=path.read_bytes()
  if record.get('warm_parent')!=pid or record.get('terminal_warm_checkpoint')!=parent['receipt'] or record['screen_file']!={'path':'SCREEN.pt','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}:raise ValueError('Same-source common TRAIN-only screen byte authority')
  screen_value=torch.load(path,map_location='cpu',weights_only=True)
  warm_best=Path(context['output_root'])/study['study_id']/pid/'selected.pt';best_bound=parent['result']['selected_checkpoint']
  if not 1<=best_bound['step']<=100 or checkpoint_receipt(warm_best,fit_id=pid,step=best_bound['step'])!=best_bound:raise ValueError('Paid positive-step warm-best eligibility byte authority')
  best=torch.load(warm_best,map_location='cpu',weights_only=False)
  model.load_state_dict(best['model_state']);opt.load_state_dict(best['optimizer_state'])
  for p in factors(model):p.requires_grad_(False)
  pending={'terminal_model':terminal['model_state'],'terminal_optimizer':terminal['optimizer_state'],'terminal_rng':terminal['adapter_state']['external']['source_rng'],'screen':screen_value,'branch':c['geometry_choice']}
  selected_parent_epoch=best['step']
 def logits(current,training,step):
  if phase in ('geometry_warm','geometry_branch','geometry_only_source'):return torch.stack([member(x) for member in current.members])
  z=current(x)
  if z.ndim==2:return z.unsqueeze(0)
  return z
 def activate(adapter):
  nonlocal pending
  if pending is None:return
  model.load_state_dict(pending['terminal_model']);opt.load_state_dict(pending['terminal_optimizer']);h.restore_rng(np,torch,pending['terminal_rng'])
  for p in factors(model):p.requires_grad_(True)
  chosen=pending['screen']['choices'][pending['branch']]
  if chosen is not None:
   for member,(index,sign) in enumerate(((chosen[0],1),(chosen[0],-1),(chosen[1],1),(chosen[1],-1))):
    target=model.members[member].sheaf_learners[0].linear1.r;direction=pending['screen']['directions_CPU'][index].to(target)
    with torch.no_grad():target.copy_(1+sign*.01*direction)
  pending=None
 def unit(adapter,step):
  activate(adapter);model.train();members=4 if phase in ('geometry_warm','geometry_branch','geometry_only_source') else c['members'];values=[]
  if phase in ('geometry_warm','geometry_branch','geometry_only_source'):
   counts={k:0 for k in ('train_forward_attempts','train_forwards_completed','backward_attempts','backwards_completed','Adam_attempts','Adam_steps_completed')};values=e.train.train_update(torch,model,opt,{'x':x,'train_index':ids,'train_y':y},counts);loss=sum(values)/4;backwards=4
  elif phase=='original_gnnm':
   opt.zero_grad(set_to_none=True)
   for m in range(4):
    z=model.forward_route(m,x);v=torch.nn.functional.cross_entropy(z[ids],y);(v/4).backward();values.append(float(v.detach()))
   opt.step();loss=sum(values)/4;backwards=4
  else:
   opt.zero_grad(set_to_none=True);z=logits(model,True,step)
   loss_tensor=e.adapter.JointMultiTransportSingle.own_F(z[0],ids,y) if phase=='joint_multi_transport_single' else e.adapter.SharedNativeBank.own_F(z,ids,y)
   loss_tensor.backward();opt.step();loss=float(loss_tensor.detach());backwards=1
  return {'objective':loss,'ce':loss,'regularization':0.,'operation_counts':{'model_forwards':4 if phase=='joint_multi_transport_single' else members,'route_predictions':c['members'],'backward_calls':backwards,'optimizer_updates':1,'supervised_examples':len(ids)*(4 if phase=='joint_multi_transport_single' else members)}}
 def capture():return {'source_rng':h.capture_rng(np,torch),'plain_L':plain_capture(torch,model),'pending_branch_start':copy.deepcopy(pending),'best_key':copy.deepcopy(adapter._best_key),'source_phase':phase,'source_native_args':args}
 def restore(v):
  nonlocal pending
  h.restore_rng(np,torch,v['source_rng']);plain_restore(model,v['plain_L'],device);pending=copy.deepcopy(v['pending_branch_start']);adapter._best_key=copy.deepcopy(v['best_key'])
 adapter=NSDAdapter(torch,torch.nn.functional,model,opt,logits_fn=logits,train_ids=ids,train_y=y,select_ids=select['ids'].to(device),select_y=select['y'].to(device),regularizer=lambda m:x.new_zeros(()),fusion='mean_logits' if phase=='original_gnnm' else 'mean_probabilities',
  selector={'metric':'auroc','direction':'max','ties':'first','patience_steps':501,'save_every':1},custom_train=unit,external_capture=capture,external_restore=restore,
  provenance={'run_id':run['run_id'],'config':c,'actual_model_seed':seed,'native_args':args,'native_source':'exact pinned fullGeneralSVD/blockdegree normalization/ELU/residual/NNL source;ordered incidence custody','historical_outcomes_or_authorization_inherited':False})
 adapter.e,adapter.h,adapter.seed,adapter.phase=e,h,seed,phase;adapter._best_key=None;adapter.native_paths=c['source_complete_native_paths'];adapter.members=c['members']
 if phase=='geometry_branch':adapter.selected_parent_epoch=selected_parent_epoch
 return adapter
