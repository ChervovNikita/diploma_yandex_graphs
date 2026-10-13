"""Byte-authorized selected HGT exports and FP32-native mean-logit serving."""
import gc,math
from pathlib import Path
from .node_endpoints import receipt,load_bound
from engine import restore_checkpoint
from run_study import load_factory
from spec import confined,fingerprint

def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];fits=c['fit_parents'];out=Path(output_root)
 if len(fits) not in (1,4) or len(set(fits))!=len(fits) or not set(fits).issubset(run['parents']):raise ValueError('One source group or four own selected M1 fits')
 ids={r:torch.load(confined(project_root,c['endpoint_inputs'][r+'_ids']),map_location='cpu',weights_only=True) for r in ('cal','evaluate')}
 if any(v.dtype!=torch.long or v.ndim!=1 for v in ids.values()) or torch.isin(ids['cal'],ids['evaluate']).any():raise ValueError('Disjoint ordered label-free endpoint IDs')
 values={r:[] for r in ids};ancestry=[];configs=[]
 for fit in fits:
  row=registry.run(study['study_id'],fit);actual=row['contract'];result=row['result']
  if row['state']!='complete' or actual['kind']!='model_fit' or result['status']!='complete' or not 31<=result['completed_steps']<=300:raise ValueError('Complete native stopped/capped source fit')
  adapter=load_factory(project_root,actual['adapter'])(torch=torch,project_root=project_root,study=study,run=actual,device=device)
  selected=result['selected'];authority=result['selected_checkpoint'];relative=Path(authority['path'])
  if relative.is_absolute() or '..' in relative.parts or authority['fit_id']!=fit or authority['step']!=selected['step']:raise ValueError('Exact selected checkpoint authority')
  checkpoint=out/study['study_id']/fit/relative;observed=receipt(checkpoint,project_root)
  if any(observed[k]!=authority[k] for k in ('bytes','sha256')):raise ValueError('Selected HGT checkpoint changed')
  saved=restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=fit)
  if saved['step']!=selected['step'] or adapter.scheduler.last_epoch!=selected['step']:raise ValueError('Selected source OneCycle epoch differs')
  adapter.model.eval()
  with torch.no_grad():z=adapter.model();replay=float(adapter.F.cross_entropy(z.mean(0)[adapter.select_ids],adapter.select_y))
  if abs(replay-selected['validation']['pooled_nll'])>1e-7:raise ValueError('Native selected pooled CE replay differs beyond source1e-7 tolerance')
  if z.ndim!=3 or not torch.isfinite(z).all() or len(fits)==4 and len(z)!=1:raise ValueError('Genuine I4 has four separate full single-route learners')
  for role,v in ids.items():values[role].append(z[:,v.to(z.device)].detach().cpu())
  ancestry.append({'run_id':fit,'checkpoint':observed,'selected_step':selected['step'],'completed_updates':result['completed_steps'],'actual_model_seed':adapter.provenance['actual_model_seed'],
   'config_hash':fingerprint(actual['config']),'selected_pooled_CE_replay_delta':replay-selected['validation']['pooled_nll'],'source_replay_absolute_tolerance':1e-7,'construction_counts':adapter.preparation_counts})
  configs.append(actual['config']);del adapter,z;gc.collect()
  if str(device).startswith('cuda'):torch.cuda.empty_cache()
 if len(fits)==4:
  if len({r['actual_model_seed'] for r in ancestry})!=4:raise ValueError('Four distinct own HGT model seeds')
  excluded={'initialization_seed','reservation_mib','estimated_seconds'};normalized=[{k:v for k,v in r.items() if k not in excluded} for r in configs]
  if any(r!=normalized[0] for r in normalized[1:]):raise ValueError('Own I4 source/data/optimizer/selector contracts differ')
 files={}
 for role,parts in values.items():
  logits=torch.cat(parts);path=output_dir/(role+'_logits.pt');torch.save({'ids':ids[role],'logits':logits,'native_FP32_mean_logits':logits.mean(0)},path);files[role]=receipt(path,project_root)
 return {'status':'complete','files':files,'ancestry':ancestry,'fusion':'mean_logits','independent_complete_learners':len(fits)==4,'CAL_targets_read':False,'evaluation_targets_read':False,
  'operation_counts':{'optimizer_updates':0,'source_complete_selected_HGT_paths':sum(r['members'] for r in configs),
   'source_export_complete_body_constructors':sum(r['construction_counts']['source_complete_body_constructors'] for r in ancestry),'source_export_unstepped_optimizer_constructors':len(fits),'source_export_OneCycle_constructors':len(fits)},'scientific_backbone_fits':0}

def scalar_calibration(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];parent=c['export_parent'];row=registry.run(study['study_id'],parent)
 if row['state']!='complete' or parent not in run['parents']:raise ValueError('Completed exact export authority')
 bank=load_bound(torch,project_root,row['result']['files']['cal']);targets=torch.load(confined(project_root,c['endpoint_inputs']['cal_targets']),map_location='cpu',weights_only=True)
 if set(targets)!={'ids','y'} or not torch.equal(bank['ids'],targets['ids']):raise ValueError('Exact CAL role ordering')
 if c['fusion']!='mean_logits' or c['placement']!='after_native_FP32_mean_logits' or c['steps']!=500 or c['learning_rate']!=.01:raise ValueError('Prospective common source-independent scalar CAL500 law')
 scores=bank['native_FP32_mean_logits'].double().detach();logT=torch.nn.Parameter(torch.zeros((),dtype=torch.float64));optimizer=torch.optim.Adam([logT],lr=.01)
 for step in range(500):
  optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(scores/logT.exp(),targets['y'])
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite scalar CAL loss')
  loss.backward();optimizer.step()
  with torch.no_grad():logT.clamp_(-4,4)
 return {'status':'complete','temperature':float(logT.detach().exp()),'placement':c['placement'],'fusion':'mean_logits','CAL_examples':len(targets['y']),
  'scalar_optimizer_updates':500,'export_parent':parent,'evaluation_targets_read':False,'scientific_backbone_fits':0,'source_native_HGT_training_CAL':False}

def evaluate_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];export_id,cal_id=c['export_parent'],c['calibration_parent']
 if not {export_id,cal_id}.issubset(run['parents']):raise ValueError('Exact export/CAL dependencies')
 er=registry.run(study['study_id'],export_id);cr=registry.run(study['study_id'],cal_id)
 if er['state']!='complete' or cr['state']!='complete':raise ValueError('Whole closed export/CAL required')
 exported,cal=er['result'],cr['result'];bank=load_bound(torch,project_root,exported['files']['evaluate']);targets=torch.load(confined(project_root,c['endpoint_inputs']['evaluate_targets']),map_location='cpu',weights_only=True)
 if cal['export_parent']!=export_id or cal['fusion']!='mean_logits' or set(targets)!={'ids','y'} or not torch.equal(bank['ids'],targets['ids']):raise ValueError('Exact bank/target/CAL ordering')
 scores=bank['native_FP32_mean_logits'].double();labels=targets['y'];metrics={}
 for name,T in (('raw',1.),('calibrated',cal['temperature'])):
  lp=(scores/T).log_softmax(-1);prob=lp.exp();correct=scores.argmax(-1).eq(labels);one_hot=torch.nn.functional.one_hot(labels,prob.shape[1])
  metrics[name]={'accuracy':float(correct.double().mean()),'correct':int(correct.sum()),'examples':len(labels),'nll':float(-lp.gather(1,labels[:,None]).mean()),'brier_sum_classes':float((prob-one_hot).square().sum(-1).mean())}
 return {'status':'complete','metrics':metrics,'fusion':'mean_logits','native_serving_arithmetic':'FP32 mean of complete source logits before float64 endpoint calculations',
  'ancestry':exported['ancestry'],'evaluation_scope':study['data']['prior_exposure'],'statistical_inference':'whole_seed_split_cohort_required','scientific_backbone_fits':0}
