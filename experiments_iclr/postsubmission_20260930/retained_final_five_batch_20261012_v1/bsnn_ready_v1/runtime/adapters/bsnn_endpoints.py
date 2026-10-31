"""NSD selected native banks; complete paid warm/screen ancestry for geometry."""
try:
 from .label_family_endpoints import scalar_calibration,evaluate_bank
 from ..run_study import load_factory,paid_parent_checkpoints
except ImportError:
 from adapters.label_family_endpoints import scalar_calibration,evaluate_bank
 from run_study import load_factory,paid_parent_checkpoints
from pathlib import Path
import gc,hashlib


def receipt(root,p):
 b=p.read_bytes();return {'path':str(p.relative_to(root)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 from engine import restore_checkpoint
 root=Path(project_root).resolve();c=run['config'];inputs=c['endpoint_inputs'];ids={r:torch.load(root/inputs[r+'_ids'],map_location='cpu',weights_only=True) for r in ('cal','evaluate')};values={r:[] for r in ids};ancestry=[]
 for id in c['fit_parents']:
  row=registry.run(study['study_id'],id);fit=row['contract']
  if row['state']!='complete' or row['result']['completed_steps']!=fit['budget']['maximum']:raise ValueError('Full paid native own-selected fit')
  context={'registry':registry,'output_root':output_root,'paid_parent_checkpoints':paid_parent_checkpoints(root,study,fit,registry,output_root)}
  adapter=load_factory(root,fit['adapter'])(torch=torch,project_root=root,study=study,run=fit,device=device,execution_context=context)
  checkpoint=Path(output_root)/study['study_id']/id/'selected.pt';restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=id);adapter.model.eval();step=row['result']['selected']['step']
  import numpy as np
  epoch=adapter.selected_parent_epoch if step==0 and hasattr(adapter,'selected_parent_epoch') else step
  with adapter.h.evaluation_rng(np,torch,adapter.seed,epoch),torch.no_grad():
   z=adapter.logits_fn(adapter.model,False,step)
   if z.ndim!=3 or not torch.isfinite(z).all() or (len(c['fit_parents'])==4 and z.shape[0]!=1):raise ValueError('Whole native finite logits and genuine one-view I4 ancestry')
   for r in ids:values[r].append(z[:,ids[r].to(device)].cpu())
  ancestry.append({'run_id':id,'checkpoint':receipt(root,checkpoint),'selected_step':step,'selected_source_epoch':epoch,'actual_model_seed':adapter.seed,'paid_native_warm':fit['config'].get('lifecycle'),'warm_best_eligible':step==0})
  del adapter;gc.collect()
  if str(device).startswith('cuda'):torch.cuda.empty_cache()
 if len(c['fit_parents'])==4 and len({a['actual_model_seed'] for a in ancestry})!=4:raise ValueError('Genuine I4 independent source seeds')
 files={}
 for role in values:
  p=Path(output_dir)/(role+'_logits.pt');torch.save({'ids':ids[role],'logits':torch.cat(values[role])},p);files[role]=receipt(root,p)
 return {'status':'complete','files':files,'ancestry':ancestry,'fusion':c['fusion'],'scientific_backbone_fits':0,'CAL_targets_read':False,'evaluation_targets_read':False}
