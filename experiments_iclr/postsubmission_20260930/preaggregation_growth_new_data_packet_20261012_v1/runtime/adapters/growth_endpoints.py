"""Source-logit query-only export, balanced scalar CAL and complete grouped ranks."""
import copy,gc,hashlib
from pathlib import Path
try:
 from ..engine import restore_checkpoint
 from ..run_study import load_factory,paid_parent_checkpoints
 from ..spec import confined
except ImportError:
 from engine import restore_checkpoint
 from run_study import load_factory,paid_parent_checkpoints
 from spec import confined


def receipt(root,path):
 b=Path(path).read_bytes();return {'path':str(Path(path).resolve().relative_to(Path(root).resolve())),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def load(torch,root,row):
 p=confined(root,row['path'])
 if receipt(root,p)!=row:raise ValueError('Exact immutable endpoint/checkpoint authority')
 return torch.load(p,map_location='cpu',weights_only=True)


def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];fits=c['fit_parents'];queries={r:load(torch,project_root,c['endpoint_inputs'][r+'_queries']) for r in ('cal','evaluate')};banks={r:[] for r in queries};ancestry=[]
 for id in fits:
  row=registry.run(study['study_id'],id);fit=row['contract'];checkpoint=output_root/study['study_id']/id/'selected.pt'
  if row['state']!='complete' or row['result']['completed_steps']!=fit['budget']['maximum']:raise ValueError('Full completed own-selected fit required')
  payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
  context={'paid_parent_checkpoints':paid_parent_checkpoints(project_root,study,fit,registry,output_root)}
  if fit['config']['growth_stage']=='post':context['growth_restored_initialization']=payload['adapter_state']['initialization']
  adapter=load_factory(project_root,fit['adapter'])(torch=torch,project_root=project_root,study=study,run=fit,device=device,execution_context=context)
  restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=id);adapter.model.eval()
  for role,q in queries.items():
   if set(q)!={'ids','pairs'} or q['pairs'].shape!=(len(q['ids']),2):raise ValueError('Opaque query-only role')
   rows=[]
   with torch.no_grad():
    for start in range(0,len(q['ids']),256):
     pairs=q['pairs'][start:start+256].t().to(device)
     # Duplicate opaque query list avoids decoding positive/negative flags; both source head calls are charged.
     logits,_=adapter.s.steps.functional_forward(torch,adapter.model,dict(adapter.model.named_parameters()),adapter.x,adapter.full_support,(pairs,pairs),training=False)
     rows.append(logits.cpu())
   banks[role].append(torch.cat(rows).t())
  ancestry.append({'run_id':id,'selected_checkpoint':receipt(project_root,checkpoint),'selected_cycle':payload['step'],'actual_model_seed':adapter.provenance['actual_model_seed'],'native_parent':adapter.initialization.get('parent_checkpoint')})
  del adapter;gc.collect()
  if str(device).startswith('cuda'):torch.cuda.empty_cache()
 if len(fits)==4 and (len({a['actual_model_seed'] for a in ancestry})!=4 or any(z.shape[0]!=1 for z in banks['evaluate'])):raise ValueError('Genuine cold I4 requires4own-selected one-view whole bodies')
 files={}
 for role in banks:
  p=output_dir/(role+'_logits.pt');torch.save({'ids':queries[role]['ids'],'logits':torch.cat(banks[role])},p);files[role]=receipt(project_root,p)
 return {'status':'complete','files':files,'ancestry':ancestry,'fusion':'mean_logits','source_readout':'mean raw complete-native-link logits','CAL_or_evaluate_targets_read':False,'initialization_gradient_or_SVD_reexecuted':False,'scientific_fits':0,
  'export_setup_constructors':sum(5 if registry.run(study['study_id'],i)['contract']['config']['growth_arm']=='independent_graph_growth4' else 2 if registry.run(study['study_id'],i)['contract']['config']['growth_stage']=='post' else 1 for i in fits),
  'export_setup_optimizer_constructors':sum(4 if registry.run(study['study_id'],i)['contract']['config']['growth_arm']=='independent_graph_growth4' else 1 for i in fits)}


def balanced(torch,scores,target):
 p=target['is_positive'];return .5*(torch.nn.functional.softplus(-scores[p]).mean()+torch.nn.functional.softplus(scores[~p]).mean())


def scalar_calibration(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];export=registry.run(study['study_id'],c['export_parent'])['result'];bank=load(torch,project_root,export['files']['cal']);target=load(torch,project_root,c['endpoint_inputs']['cal_targets'])
 if not torch.equal(bank['ids'],target['ids']):raise ValueError('CAL target order')
 scores=bank['logits'].double().mean(0).detach();scale=torch.nn.Parameter(torch.zeros((),dtype=torch.float64));opt=torch.optim.Adam([scale],lr=.01)
 for _ in range(500):
  opt.zero_grad(set_to_none=True);loss=balanced(torch,scores/scale.exp(),target)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite source-balanced scalar CAL')
  loss.backward();opt.step()
  with torch.no_grad():scale.clamp_(-4,4)
 return {'status':'complete','temperature':float(scale.detach().exp()),'export_parent':c['export_parent'],'fusion':'mean_logits','scalar_optimizer_updates':500,'evaluate_targets_read':False,'scientific_fits':0}


def evaluate_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];export=registry.run(study['study_id'],c['export_parent'])['result'];cal=registry.run(study['study_id'],c['calibration_parent'])['result'];bank=load(torch,project_root,export['files']['evaluate']);target=load(torch,project_root,c['endpoint_inputs']['evaluate_targets'])
 if cal['export_parent']!=c['export_parent'] or not torch.equal(bank['ids'],target['ids']):raise ValueError('Exact endpoint ancestry')
 mask=target['is_positive'];P=int(mask.sum());K=target['negative_pool_width'];out={}
 for name,temp in (('raw',1.),('calibrated',cal['temperature'])):
  z=bank['logits'].double().mean(0)/temp;p=torch.zeros(P,dtype=z.dtype);n=torch.zeros(P*K,dtype=z.dtype)
  p[target['original_index'][mask]]=z[mask];n[target['original_index'][~mask]]=z[~mask];n=n.reshape(P,K)
  ranks=1+.5*((n>=p[:,None]).sum(1)+(n>p[:,None]).sum(1))
  out[name]={'MRR':float(ranks.double().reciprocal().mean()),'MRR_source_round4':round(float(ranks.double().reciprocal().mean()),4),'Hits10':float((ranks<=10).double().mean()),'balanced_binary_NLL':float(balanced(torch,z,target)),'positive_queries':P,'negative_pool_width':K}
 return {'status':'complete','metrics':out,'ancestry':export['ancestry'],'CAL_order':'after source mean raw-logit pool','ranking_temperature_monotonic_invariance':True,'scientific_fits':0}
