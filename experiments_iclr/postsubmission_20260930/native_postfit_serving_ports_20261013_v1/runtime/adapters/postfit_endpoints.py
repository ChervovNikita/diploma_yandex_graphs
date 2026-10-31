"""Label-free frozen banks and fixed fold routing; target jobs remain separate."""
import gc,hashlib,math
from pathlib import Path
from .postfit import require,receipt,fold_assignment
from .node_endpoints import scalar_calibration,evaluate_bank
try:
 from ..engine import restore_checkpoint
 from ..run_study import load_factory,construct_adapter,paid_parent_checkpoints
 from ..spec import confined
except ImportError:
 from engine import restore_checkpoint
 from run_study import load_factory,construct_adapter,paid_parent_checkpoints
 from spec import confined

def frozen_native_bank(torch,root,study,identifier,registry):
 row=registry.run(study['study_id'],identifier);v=row['result'] or {};f=v.get('postfit_features')
 require(row['state']=='complete' and isinstance(f,dict) and f['selected_checkpoint']==v.get('selected_checkpoint'),'Complete byte-bound source selected native output required')
 q=confined(root,f['receipt']['path']);require(receipt(root,q)==f['receipt'],'Immutable source selected feature bytes');a=torch.load(q,map_location='cpu',weights_only=True);require(a['donor_fit_id']==identifier and a['selected_checkpoint']==f['selected_checkpoint'],'Frozen native source identity');return a,f

def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];fits=c['fit_parents'];require(len(fits) in (1,4,5) and len(set(fits))==len(fits) and set(fits)<=set(run['parents']),'Exact declared source bank ancestry required')
 ids={role:torch.load(confined(project_root,c['endpoint_inputs'][role+'_ids']),map_location='cpu',weights_only=True) for role in ('cal','evaluate')}
 require(all(torch.is_tensor(v) and v.dtype==torch.long and v.ndim==1 and len(v)==len(torch.unique(v)) for v in ids.values()) and not torch.isin(ids['cal'],ids['evaluate']).any(),'Disjoint label-free target query IDs')
 outputs={k:[] for k in ids};ancestry=[];select_outputs=[];select_ids=None;folds=None
 for i in fits:
  p=registry.run(study['study_id'],i);r=p['contract'];require(p['state']=='complete' and r['kind']=='model_fit','Complete source fits required')
  if r['config']['postfit_phase']=='native':
   data,f=frozen_native_bank(torch,project_root,study,i,registry)
   for role,q in ids.items():outputs[role].append(data['Z'][:,q])
   select_ids=data['select_ids'];select_outputs.append(data['Z'][:,select_ids]);ancestry.append({'run_id':i,'selected_checkpoint':p['result']['selected_checkpoint'],'feature_receipt':f['receipt'],'native_uniform_reference':True});continue
  context={'registry':registry,'output_root':output_root,'paid_parent_checkpoints':paid_parent_checkpoints(project_root,study,r,registry,output_root)}
  factory=load_factory(project_root,r['adapter']);adapter=construct_adapter(factory,execution_context=context,torch=torch,project_root=project_root,study=study,run=r,device='cpu');checkpoint=Path(output_root)/study['study_id']/i/'selected.pt';restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=i);adapter.model.eval()
  with torch.no_grad():
   for role,q in ids.items():outputs[role].append(adapter.predict_rows(q,p['result']['selected']['step']).detach().cpu())
   select_ids=adapter.data['select_ids'];select_outputs.append(adapter.predict_rows(select_ids,p['result']['selected']['step']).detach().cpu())
  ancestry.append({'run_id':i,'selected_checkpoint':p['result']['selected_checkpoint'],'feature_receipts':{k:v['feature_receipt'] for k,v in context['paid_parent_checkpoints'].items()},'postfit_source_family':r['config']['postfit_family'],'rule':r['config'].get('postfit_rule'),'fold':r['config'].get('postfit_fold')});del adapter;gc.collect()
 if c.get('postfit_fold_bank'):
  require(len(fits)==5 and [a['fold'] for a in ancestry]==list(range(5)),'Complete exact fivefold operator roster');folds=fold_assignment(torch,select_ids)
  def routed(values,query):
   route=fold_assignment(torch,select_ids,query);stack=torch.cat(values,dim=0);require(stack.shape==(5,len(query),10),'Five finite fold endpoints');return stack[route,torch.arange(len(query))].unsqueeze(0)
  final={role:routed(outputs[role],q) for role,q in ids.items()};select_bank=routed(select_outputs,select_ids)
 else:
  final={role:torch.cat(values) for role,values in outputs.items()};select_bank=torch.cat(select_outputs)
 files={}
 for role,z in final.items():
  require(z.ndim==3 and z.shape[1:]==(len(ids[role]),10) and bool(torch.isfinite(z).all()),'Finite exact source served outputs');q=output_dir/(role+'_logits.pt');torch.save({'ids':ids[role],'logits':z},q);files[role]=receipt(project_root,q)
 q=output_dir/'SELECT_source_outputs.pt';torch.save({'ids':select_ids,'logits':select_bank,**({} if folds is None else {'folds':folds})},q)
 return {'status':'complete','files':files,'source_SELECT_outputs':receipt(project_root,q),'ancestry':ancestry,'fusion':c['fusion'],'independent_complete_learners':len(fits)==4,'source_fold_operator_bank':bool(c.get('postfit_fold_bank')),'fold_routing':None if folds is None else {'SELECT':'source torch CPU11709 randperm position modulo5','unseen_queries':'(public node ID+11709)%5','labels_used_for_assignment':False,'final_refit':False,'held_fold_checkpoint_selection':False},'CAL_targets_read':False,'evaluation_targets_read':False,'scientific_backbone_fits':0,'new_native_forwards':0}

def export_noeffect(*,project_root,study,run,output_root,output_dir,registry,device):
 """Source M1 reliability is the same uniform law, explicitly zero operator fits."""
 value=export_bank(project_root=project_root,study=study,run=run,output_root=output_root,output_dir=output_dir,registry=registry,device=device);value['structural_M1_no_weighting_effect_no_fit']=True;return value
