"""Native selected export, member-temperature CAL500 and whole-bank scoring."""
import gc
import math
from pathlib import Path
from adapters.node_endpoints import receipt,load_bound
from engine import restore_checkpoint
from run_study import load_factory
from spec import confined,fingerprint

def export_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];fits=c['fit_parents'];out=Path(output_root)
 if len(fits) not in (1,4) or len(set(fits))!=len(fits) or not set(fits).issubset(run['parents']):raise ValueError('One own joint/tied learner or four own selected M1 learners')
 ids={r:torch.load(confined(project_root,c['endpoint_inputs'][r+'_ids']),map_location='cpu',weights_only=True) for r in ('cal','evaluate')}
 if any(v.dtype!=torch.long or v.ndim!=1 for v in ids.values()) or torch.isin(ids['cal'],ids['evaluate']).any():raise ValueError('Disjoint ordered label-free export IDs')
 values={r:[] for r in ids};ancestry=[];configs=[]
 for fit in fits:
  row=registry.run(study['study_id'],fit);actual=row['contract']
  if row['state']!='complete' or actual['kind']!='model_fit' or row['result']['status']!='complete':raise ValueError('Complete native stopped/capped selected own fit required')
  adapter=load_factory(project_root,actual['adapter'])(torch=torch,project_root=project_root,study=study,run=actual,device=device)
  selected=row['result']['selected'];authority=row['result']['selected_checkpoint'];relative=Path(authority['path'])
  if relative.is_absolute() or '..' in relative.parts or authority['fit_id']!=fit or authority['step']!=selected['step']:raise ValueError('Exact selected checkpoint authority')
  checkpoint=out/study['study_id']/fit/relative;observed=receipt(checkpoint,project_root)
  if any(observed[k]!=authority[k] for k in ('bytes','sha256')):raise ValueError('Selected checkpoint bytes changed')
  restored=restore_checkpoint(torch,checkpoint,adapter,expected_fit_id=fit)
  if restored['step']!=selected['step']:raise ValueError('Selected step differs')
  if not adapter.model.original and adapter.optimizer.state['tf1_global']['updates']!=selected['step']:raise ValueError('Selected TF1Adam global iteration differs')
  adapter.model.eval()
  with torch.no_grad():
   logits=adapter.model();replay=adapter.validate(selected['step'])
  saved=selected['validation']['objective']
  if not math.isclose(replay['objective'],saved,rel_tol=1e-5,abs_tol=1e-7):raise ValueError('Selected own-total replay differs beyond source reduction tolerance')
  if logits.ndim!=3 or not torch.isfinite(logits).all() or len(fits)==4 and len(logits)!=1:raise ValueError('Complete native logits; I4 has four own single-view fits')
  for role,v in ids.items():values[role].append(logits[:,v.to(logits.device)].detach().cpu())
  ancestry.append({'run_id':fit,'checkpoint':observed,'selected_step':selected['step'],'completed_updates':row['result']['completed_steps'],
   'actual_model_seed':adapter.provenance['actual_model_seed'],'config_hash':fingerprint(actual['config']),
   'SELECT_total_replay_delta':replay['objective']-saved,'SELECT_total_replay_tolerance':{'rel_tol':1e-5,'abs_tol':1e-7},
   'construction_counts':adapter.preparation_counts})
  configs.append(actual['config']);del adapter,logits;gc.collect()
  if str(device).startswith('cuda'):torch.cuda.empty_cache()
 if len(fits)==4:
  if len({r['actual_model_seed'] for r in ancestry})!=4:raise ValueError('Four distinct source body seeds')
  excluded={'initialization_seed','reservation_mib','estimated_seconds'}
  views=[{k:v for k,v in c.items() if k not in excluded} for c in configs]
  if any(v!=views[0] for v in views[1:]):raise ValueError('Own I4 architecture/objective/role contracts differ')
 files={}
 for role,parts in values.items():
  logits=torch.cat(parts);pack={'ids':ids[role],'logits':logits,'member_log_probs':logits.log_softmax(-1),
   'native_pool_probabilities':logits.softmax(-1).mean(0) if c['fusion']=='mean_probabilities' else logits.mean(0).softmax(-1)}
  path=output_dir/(role+'_logits.pt');torch.save(pack,path);files[role]=receipt(path,project_root)
 return {'status':'complete','files':files,'ancestry':ancestry,'fusion':c['fusion'],'independent_complete_learners':len(fits)==4,
  'CAL_targets_read':False,'evaluation_targets_read':False,'source_selected_replays':len(fits),'scientific_backbone_fits':0}

def log_pool(torch,member_log_probs,logT):
 return torch.logsumexp(torch.nn.functional.log_softmax(member_log_probs/logT.exp(),dim=-1),dim=0)-math.log(len(member_log_probs))

def fit_temperature(torch,log_probs,targets,*,fusion,mean_logits=None):
 """Literal native500/update latest-tie selector; synthetic QA can call it."""
 logT=torch.nn.Parameter(torch.zeros((),dtype=torch.float64));optimizer=torch.optim.Adam([logT],lr=.05,betas=(.9,.999),eps=1e-8,weight_decay=0)
 idx=torch.arange(len(targets));lp=log_probs.double()
 def objective():
  pooled=log_pool(torch,lp,logT) if fusion=='mean_probabilities' else (mean_logits.double()/logT.exp()).log_softmax(-1)
  return -pooled[idx,targets].mean()
 with torch.no_grad():best=float(objective());best_logT=0.;best_step=0
 if not math.isfinite(best):raise FloatingPointError('Nonfinite initial CAL NLL')
 for step in range(1,501):
  optimizer.zero_grad();loss=objective()
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native CAL loss')
  loss.backward()
  if not torch.isfinite(logT.grad):raise FloatingPointError('Nonfinite CAL gradient')
  optimizer.step()
  with torch.no_grad():
   logT.clamp_(-5.,5.);value=float(objective())
   if not math.isfinite(value):raise FloatingPointError('Nonfinite post-update CAL NLL')
   if value<=best:best,best_logT,best_step=value,float(logT),step
 return {'temperature':math.exp(best_logT),'logT':best_logT,'selected_step':best_step,'CAL_minimum_NLL':best,
  'objective_forward_evaluations':1001,'scalar_optimizer_updates':500,'endpoint':'latest_tied_minimum_step0_to500'}

def scalar_calibration(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];export_id=c['export_parent'];row=registry.run(study['study_id'],export_id)
 if row['state']!='complete' or export_id not in run['parents']:raise ValueError('Completed label-free bank export required')
 bank=load_bound(torch,project_root,row['result']['files']['cal']);targets=torch.load(confined(project_root,c['endpoint_inputs']['cal_targets']),map_location='cpu',weights_only=True)
 if set(targets)!={'ids','y'} or not torch.equal(bank['ids'],targets['ids']):raise ValueError('Exact CAL ordering')
 expected='before_member_probability_pool' if c['fusion']=='mean_probabilities' else 'after_mean_logits'
 if c['placement']!=expected or c['steps']!=500 or c['learning_rate']!=.05:raise ValueError('Source CAL500 placement/learning rate required')
 value=fit_temperature(torch,bank['member_log_probs'],targets['y'],fusion=c['fusion'],mean_logits=bank['logits'].double().mean(0))
 return {'status':'complete',**value,'placement':c['placement'],'fusion':c['fusion'],'export_parent':export_id,'CAL_examples':len(targets['y']),
  'evaluation_targets_read':False,'scientific_backbone_fits':0,'original_GNNM_CAL_adaptation':c['fusion']=='mean_logits'}

def evaluate_bank(*,project_root,study,run,output_root,output_dir,registry,device):
 import torch
 c=run['config'];export_id,cal_id=c['export_parent'],c['calibration_parent']
 if not {export_id,cal_id}.issubset(run['parents']):raise ValueError('Exact export/CAL dependencies')
 er=registry.run(study['study_id'],export_id);cr=registry.run(study['study_id'],cal_id)
 if er['state']!='complete' or cr['state']!='complete':raise ValueError('Both endpoints complete')
 exported,cal=er['result'],cr['result']
 if cal['export_parent']!=export_id or cal['fusion']!=c['fusion']:raise ValueError('CAL bank custody differs')
 bank=load_bound(torch,project_root,exported['files']['evaluate']);targets=torch.load(confined(project_root,c['endpoint_inputs']['evaluate_targets']),map_location='cpu',weights_only=True)
 if set(targets)!={'ids','y'} or not torch.equal(bank['ids'],targets['ids']):raise ValueError('Exact evaluation ordering')
 labels=targets['y'];metrics={}
 for name,T in (('raw',1.),('calibrated',cal['temperature'])):
  if name=='raw':p=bank['native_pool_probabilities'].double();lp=p.clamp_min(torch.finfo(p.dtype).tiny).log()
  elif c['fusion']=='mean_probabilities':lp=log_pool(torch,bank['member_log_probs'].double(),torch.tensor(math.log(T),dtype=torch.float64));p=lp.exp()
  else:lp=(bank['logits'].double().mean(0)/T).log_softmax(-1);p=lp.exp()
  correct=p.argmax(-1).eq(labels);one_hot=torch.nn.functional.one_hot(labels,p.shape[1])
  metrics[name]={'accuracy':float(correct.double().mean()),'correct':int(correct.sum()),'examples':len(labels),
   'nll':float(-lp.gather(1,labels[:,None]).mean()),'brier_sum_classes':float((p-one_hot).square().sum(-1).mean())}
 return {'status':'complete','metrics':metrics,'fusion':c['fusion'],'ancestry':exported['ancestry'],
  'evaluation_scope':study['data']['prior_exposure'],'statistical_inference':'whole_seed_split_cohort_required','scientific_backbone_fits':0}
