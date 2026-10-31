"""Root engineering-only source Adam moments and untrained first-action capacity."""
from pathlib import Path
import copy

def allocate_source_moments(torch,adapter):
 """No optimizer kernel;source-frozen warm factors retain no state."""
 count=0
 for group in adapter.optimizer.param_groups:
  for p in group['params']:
   if not p.requires_grad:
    if p in adapter.optimizer.state:raise ValueError('Source warm inactive factor must have no Adam moments')
    continue
   if p in adapter.optimizer.state:raise ValueError('Fresh untrained resource prototype required')
   step_device=p.device if group.get('capturable') or group.get('fused') else torch.device('cpu')
   state={'step':torch.zeros((),dtype=torch.float32,device=step_device),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
   if group.get('amsgrad'):state['max_exp_avg_sq']=torch.zeros_like(p)
   adapter.optimizer.state[p]=state;count+=1
 return {'live_parameter_moment_states':count,'optimizer_kernels':0,'source_inactive_factor_states':0}


def first_action_capacity_callback(*,torch,project_root,study,run,device):
 """Actual GRAPH/TRAIN-IDs only;no warm,rank,NLL,SELECT or optimizer."""
 from adapters.nsd import factory,construct,helpers
 from adapters.native_label_families import bound
 import numpy as np
 root=Path(project_root).resolve();c=copy.deepcopy(run['config']);c.update(nsd_arm='geometry_warm',device=str(device));B=c['nsd_sources'];graph=torch.load(bound(root,c['node_inputs']['graph']),map_location='cpu',weights_only=True);ids=torch.load(bound(root,c['node_inputs']['train']),map_location='cpu',weights_only=True)['ids'].to(device)
 e,f,args=factory(torch,root,B,c,graph,device);h=helpers(torch,root,B);before=h.capture_rng(np,torch);model=None
 try:
  model=construct(torch,root,B,e,f,c,c['initialization_seed']);member=model.members[0];member.eval();x=graph['x'].to(device);counts={k:0 for k in ('screen_native_builder_returns','extra_offnode_spmm_attempts','extra_offnode_spmm_completed','offnode_sparse_copy_bytes_total','screen_action_CPU_bytes_total')}
  with torch.no_grad(),e.init.FirstActionCapture(torch,member,ids,counts,None) as first:member(x)
  common=first.baseline_value
  with torch.no_grad(),e.init.FirstActionCapture(torch,member,ids,counts,common) as second:member(x)
  if not torch.isfinite(first.action).all() or not torch.isfinite(second.action).all():raise FloatingPointError('Full actual first-action shape prototype must be finite')
  return {'counts':counts,'common_value_shape':list(common.shape),'common_value_bytes':common.numel()*common.element_size(),'CPU_action_shape':list(second.action.shape),'native_forward_calls':2,'extra_offnode_spmm':2,'model_constructor_calls':1,'complete_native_modules':4,'optimizer_objects':0,'optimizer_updates':0,'trained100_warm_or_screen_eligibility_claim':False,'resource_capacity_only':True}
 finally:h.restore_rng(np,torch,before)
