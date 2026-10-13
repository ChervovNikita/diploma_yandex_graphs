"""Literal structural18 fixed60 cells; current-role source custody adaptation."""
import copy
from . import private_response as prior
from .common_native_wrapper import load_module,bound
CONTRACT={'schema':'literal-structural-interaction18-fixed60-current-role-v1','source':'models.build/FIXED_RECIPE18/run.py ordinary_joint; no horizon extension/HPO/historical state','geometry':'endpoint inner256/outer64; source fixed seeds and structural600000+seed;61whole episodes incltail percycle; all paired draws made beforeupdates;exact unionmask','training':'source three committed Adam inner/outer/repeatedinner;.001 betas.9,.999 eps1e-8 WD0 foreachFalse fusedFalse; own mean except genuine native4 own sum and outer times4','selection':'full60cycles eval5 first strict maximum complete500negative MRRrounded4; no earlystop; no checkpoint0','serving':'committed state meanrawlogits; exact structural topology/witness enumeration; no caps/sampling/fallback','adaptations':['custodied current CiteSeer GRAPH/TRAIN/SELECT with CAL/evaluate query-only endpoints','fresh native single,I4 four independently selected whole bodies and originalGNNM boundary source3pass controls; no historical reference reuse','fixed60 sourcebudget is cycle rather than donor epoch'],'historical_state_inherited':False}

def construct(torch,root,B,arm,seed,factor,structural_seed,device):
 s=prior.sources(root,B['base']);dev=torch.device(device)
 for row in B['structural'].values():bound(root,row)
 base=load_module(root,B['structural']['models_base']);topology=load_module(root,B['structural']['topology'])
 with prior.aliases({'models_base':base,'topology':topology}):models=load_module(root,B['structural']['models'])
 if arm=='original_GNNM':model,partition=s.original.original_boundary(torch,root,s,seed,factor,dev);shared=tuple(n for n,p in model.named_parameters());private=()
 else:
  s.models=models
  model,shared,private,partition=models.build(s.native,s.heads,arm='capable_single' if arm=='native_M1' else arm,seed=seed,factor_seed=factor,device=dev,input_dim=3703,structural_seed=structural_seed,message_width=32,message_hidden=64,chunk_size=4096)
 optimizer=torch.optim.Adam(list(model.parameters()),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
 return s,model,optimizer,shared,private,partition

class Adapter(prior.Adapter):
 def observe_validation(self,step,v):
  better=self._best is None or v['selection_metric']>self._best
  if better:self._best,self._best_step=v['selection_metric'],step
  return {'save_selected':better,'stop_requested':False,'best_saved_metric':self._best,'best_saved_step':self._best_step,'full_budget':True}
 def episode(self,support,inner,outer):
  value,counts=super().episode(support,inner,outer)
  if hasattr(self.model,'drain_work'):
   value['structural_work']=self.model.drain_work()
   counts.update({'source_structural_'+k:v for k,v in value['structural_work'].items()})
  return value,counts
 def validate(self,step):
  value=super().validate(step)
  if hasattr(self.model,'drain_work'):value['structural_work']=self.model.drain_work()
  return value

def make_adapter(*,torch,project_root,study,run,device):
 from structural_cardinality_semantics import validate_structural_run
 validate_structural_run(run);c=run['config'];arm=c['structural_arm']
 def load(row):return torch.load(bound(project_root,row),map_location='cpu',weights_only=True)
 graph,train,select=[load(c['link_inputs'][k]) for k in ('graph','train','select')]
 if study['dataset']!='citeseer-heart' or set(graph)!={'x','train_support_edge_index'} or set(train)!={'positive'} or graph['x'].shape!=(3327,3703) or train['positive'].shape!=(3870,2) or select['negative'].shape!=(227,500,2):raise ValueError('Current complete source CiteSeer custody required')
 state=construct(torch,project_root,c['structural_sources'],arm,c['initialization_seed'],c['factor_seed'],c['structural_seed'],device)
 a=Adapter(torch,*state,graph,train,select,arm,c['initialization_seed'],run['seed'],{'study':study['study_id'],'run_id':run['run_id'],'config':c,'actual_model_seed':c['initialization_seed'],'contract':CONTRACT});a.draw_output=__import__('pathlib').Path(project_root)/c['source_draw_output']/run['run_id'];return a
