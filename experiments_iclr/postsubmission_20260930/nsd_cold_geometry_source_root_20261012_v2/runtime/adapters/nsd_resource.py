"""Root-only untrained capacity prototype; exact source TRAIN callback,commit denied."""
from .nsd import make_adapter,factors

def make_resource_adapter(*,torch,project_root,study,run,device,execution_context=None):
 # Run is the target-free engineering clone; no actual SELECT targets are decoded.
 adapter=make_adapter(torch=torch,project_root=project_root,study=study,run=run,device=device,execution_context=execution_context)
 if run['config'].get('resource_active_geometry_prototype'):
  if adapter.phase!='geometry_warm':raise ValueError('Only explicit untrained source geometry capacity prototype')
  for p in factors(adapter.model):p.requires_grad_(True)
  adapter.phase='resource_active_geometry_prototype'
  adapter.provenance['resource_prototype']='untrained active factors=1;no paid warm/screen/eligibility/utility claim;exact four own NLL old-state source callback'
 return adapter

def run_source_callback(adapter,step,*,expected_denial):
 try:adapter.train_step(step)
 except expected_denial:return {'complete_source_gradient_callback_reached_optimizer':True,'optimizer_commit':False}
 raise RuntimeError('Root optimizer denial must intercept after the whole exact source callback')
