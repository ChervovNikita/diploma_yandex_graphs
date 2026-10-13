"""Bounded source postfit procedures and paid frozen feature custody; stdlib only."""
import hashlib,json
from pathlib import Path
CONTRACT={'schema':'literal-native-postfit-source-procedures-v1','roles':'current WikiCS mask0 TRAIN580 SELECT3505 CAL1769; factual graph unchanged','donors':'fresh own-selected native source states; no historical tensors','direct':'source four heads; shared TRAIN RMS; fixed FP64 LBFGS max_iter1000/max_eval50000; no SELECT choice','affine':'literal zero-delta native head/128x10 affine; raw FP32 anchors; AdamW.001/WD0; source1000/p300 strict-first SELECT','serving':'all five literal FP64 operators; CPU11709 SELECT randperm position modulo5; five500-update folds; no heldfold selection or finalrefit','extension':'CAL/evaluate label-free ID fold=(ID+11709)%5; SELECT source folds retained; learned originalGNNM extension separate from uniform native reference','feature_custody':'each paid donor restores own selected state once after closure and exports label-free fullgraph features/logits; all cohort receipts close before any head update','nonlinear':'no source architecture/trainer/budget; explicitly unimplemented'}
DIRECT=('BE_factor_refit','BE_full_refit','ordinary_full_refit','single_full_refit')
RULES=('temperature_global','temperature_member','reliability_self','linear_stacking','reliability_graph')
SCHEMA='paid-frozen-native-feature-cohort-v1'

def validate_lifecycle(run):
 c=run['config'];v=c.get('lifecycle')
 if v is None:return []
 if (set(v)!={'schema','parent_fit_ids','donor_fit_ids','cohort_id','new_optimizer','parameter_initialization'} or v.get('schema')!=SCHEMA or v.get('new_optimizer') is not True or v.get('parameter_initialization')!='source_selected_frozen_features_fresh_head_only'
     or not isinstance(v['parent_fit_ids'],list) or len(v['parent_fit_ids'])!=({'direct':18,'affine':18,'serving':90}.get(c.get('postfit_family'))) or len(set(v['parent_fit_ids']))!=len(v['parent_fit_ids']) or any(not isinstance(x,str) for x in v['parent_fit_ids'])
     or not isinstance(v['donor_fit_ids'],list) or len(v['donor_fit_ids']) not in (1,4) or not set(v['donor_fit_ids'])<=set(v['parent_fit_ids']) or v['cohort_id']!=c.get('postfit_cohort_id')):raise ValueError('All and only paid frozen native cohort ancestors, exact own donor and fresh head optimizer required')
 return v['parent_fit_ids']

def validate_postfit_run(run):
 c,b,s=run['config'],run['budget'],run['selection'];phase=c.get('postfit_phase');family=c.get('postfit_family');m=c.get('members');owners=c.get('source_optimizer_count');bodies=c.get('physical_slow_bodies')
 lr=.001 if phase=='native' or family=='affine' else 1. if family=='direct' else .01
 if (run.get('kind')!='model_fit' or c.get('postfit_source_contract')!=CONTRACT or family not in ('direct','affine','serving') or phase not in ('native','head') or c.get('historical_state_inherited') is not False or c.get('classes')!=10 or c.get('stage')!='comparative' or s.get('checkpoint0') is not False or s.get('ties')!='first' or owners!=1 or c.get('learning_rate')!=lr or c.get('weight_decay')!=0):raise ValueError('Exact registered source postfit family/role/optimizer contract required')
 if phase=='native':
  if validate_lifecycle(run) or bodies!=1 or m not in (1,4):raise ValueError('Native donors must be fresh complete cold bodies')
  cap=1100 if family=='direct' else 1000
  expected={'unit':'optimizer_updates','policy':'full_budget','maximum':1100} if family=='direct' else {'unit':'optimizer_updates','policy':'native_selector_stop','maximum':1000,'minimum':301,'patience':300}
  if b!=expected or s!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,**({} if family=='direct' else {'patience_steps':300})} or c.get('width')!=(512 if family=='direct' else 128):raise ValueError('Native source1100/local100 or1000/p300,strict-first every-update SELECT required')
  if family=='direct':
   if c.get('postfit_native_arm') not in ('native_single','be_unit_contrastive','original_GNNM') or run['seed'] not in (6101,6203,6307) or c.get('initialization_seed')!=run['seed']+1009*run['body']:raise ValueError('Literal public WikiCS Session donor source/seed required')
   fwd=8 if m==4 else 2;back=8 if c['postfit_native_arm']=='original_GNNM' else 1
   if m!=(1 if c['postfit_native_arm']=='native_single' else 4) or c.get('depth')!=7:raise ValueError('Literal direct native complete route/depth ownership required')
  else:
   if run['seed'] not in (7301,7403,7507) or c.get('initialization_seed')!=run['seed']+1000003*run['body'] or c.get('dropout')!=.2:raise ValueError('Literal common native fixed source anchor/seed required')
   fwd=m;back=4 if c.get('is_original_gnnm') else 1
   arms=('E4','equipped_M1','original_gnnm') if family=='affine' else ('ordinary_M1','factorized_allmap_M1','shared4_unchanged','original_gnnm')
   if c.get('postfit_native_arm') not in arms or m!=(4 if c['postfit_native_arm'] in ('E4','shared4_unchanged','original_gnnm') else 1) or c.get('depth')!=2 or c.get('backbone') not in (('SAGE',) if family=='affine' else ('GCN','GAT','SAGE')):raise ValueError('Literal native128/L2 source arm/backbone/route ownership required')
  base={'optimizer_objects':1,'physical_slow_bodies':1,'known_optimizer_step_calls':cap if family=='direct' else None,'budget_unit':'optimizer_updates','registered_budget_units':cap,'source_complete_body_constructors':1,'source_optimizer_constructors':1,'source_discarded_body_constructors':0,'known_postfit_feature_collection_member_forwards':m,'law':'literal_'+family+'_cold_native_source_donor','source_TRAIN_forward_calls_upper_bound':cap*fwd,'source_TRAIN_backward_calls_upper_bound':cap*back}
  if family=='direct':base.update(known_source_TRAIN_forward_calls=cap*fwd,known_source_TRAIN_backward_calls=cap*back)
  else:base.update(native_stopping_updates_minimum=301,native_stopping_updates_maximum=1000)
  return base
 ids=validate_lifecycle(run)
 if not ids or bodies!=0 or c.get('is_original_gnnm') is not False:raise ValueError('Frozen selected source inputs,zero native bodies,fresh head owner required')
 expected={'direct':{'name':'LBFGS','learning_rate':1.,'max_iter':1000,'max_eval':50000,'tolerance_grad':1e-6,'tolerance_change':0.,'history_size':10,'line_search_fn':'strong_wolfe'},'affine':{'name':'AdamW','learning_rate':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.},'serving':{'name':'Adam','learning_rate':.01,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.}}[family]
 if c.get('postfit_optimizer')!=expected or c.get('postfit_dtype')!=('float32' if family=='affine' else 'float64') or c.get('postfit_CPU_threads')!=1 or c.get('donor_learning_rate')!=.001:raise ValueError('Full literal source head optimizer/dtype and separate donor LR must be explicit')
 if family=='direct' and c.get('postfit_regularization')!={'lambda':.001,'half':True,'class_centered_effective_A':True,'member_reduction':'mean'}:raise ValueError('Exact source effective-A regularization metadata required')
 if family=='serving' and c.get('source_shrinkage')!=.01:raise ValueError('Exact source fixed operator shrinkage coefficient required')
 if family=='direct':
  if c.get('postfit_condition') not in DIRECT or b!={'unit':'optimizer_updates','policy':'full_budget','maximum':1} or s!={'metric':'training_completion','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False} or c.get('width')!=512 or m!=(1 if c['postfit_condition']=='single_full_refit' else 4):raise ValueError('One source joint LBFGS step;fixed endpoint,no development selector required')
  return {'optimizer_objects':1,'physical_slow_bodies':0,'known_optimizer_step_calls':1,'budget_unit':'optimizer_updates','registered_budget_units':1,'known_postfit_LBFGS_trajectories':1,'bounded_postfit_LBFGS_iterations_maximum':1000,'bounded_postfit_LBFGS_closures_maximum':50000,'known_postfit_backbone_updates':0,'law':'literal_direct_fixed_FP64_LBFGS'}
 if family=='affine':
  condition=c.get('postfit_condition');arm=c.get('postfit_bank_arm')
  valid={'E4':('BE_head_continuation','common_affine_residual','private_affine_residual'),'equipped_M1':('BE_head_continuation','affine_residual'),'equipped_I4':('BE_head_continuation','private_affine_residual')}
  if arm not in valid or condition not in valid[arm] or m!=(4 if arm=='E4' else 1) or c.get('width')!=128 or b!={'unit':'optimizer_updates','policy':'native_selector_stop','maximum':1000,'minimum':301,'patience':300} or s!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':300}:raise ValueError('Literal source seven affine-capacity cells and1000/p300 source head selectors required')
  return {'optimizer_objects':1,'physical_slow_bodies':0,'known_optimizer_step_calls':None,'budget_unit':'optimizer_updates','registered_budget_units':1000,'native_stopping_updates_minimum':301,'native_stopping_updates_maximum':1000,'source_complete_body_constructors':0,'source_discarded_body_constructors':0,'source_optimizer_constructors':1,'source_TRAIN_forward_calls_upper_bound':1000*(2*m if condition=='BE_head_continuation' else m),'source_TRAIN_backward_calls_upper_bound':1000,'known_postfit_backbone_updates':0,'law':'literal_affine_zero_delta_source_AdamW_strict_first1000p300'}
 if c.get('postfit_rule') not in RULES or c.get('postfit_fold') not in range(5) or m not in (1,4) or c.get('width')!=8 or b!={'unit':'optimizer_updates','policy':'full_budget','maximum':500} or s!={'metric':'training_completion','direction':'max','ties':'first','evaluation_interval':500,'checkpoint0':False}:raise ValueError('Literal fivefold fixed500 Adam operators;heldfold never selects epoch required')
 if m==1 and c['postfit_rule'] in ('temperature_member','reliability_self','reliability_graph'):raise ValueError('Source M1 duplicates/noeffect rules must be paid references,never new fits')
 return {'optimizer_objects':1,'physical_slow_bodies':0,'known_optimizer_step_calls':500,'budget_unit':'optimizer_updates','registered_budget_units':500,'known_postfit_operator_Adam_maps':500,'known_postfit_backbone_updates':0,'law':'literal_serving_fixed500_fold_'+c['postfit_rule']}

def require_cohort_parent(parent,run):
 validate_postfit_run(parent);validate_postfit_run(run)
 a,b=parent['config'],run['config']
 if (a['postfit_phase']!='native' or b['postfit_phase']!='head' or parent['run_id'] not in validate_lifecycle(run) or a.get('postfit_cohort_id')!=b.get('postfit_cohort_id') or a['postfit_family']!=b['postfit_family'] or a['node_inputs']!=b['node_inputs'] or a['postfit_sources']!=b['postfit_sources']):raise ValueError('Each postfit parent must be the exact paid source native feature cohort')

def validate_donor_binding(run,parents):
 c=run['config'];ids=c['lifecycle']['donor_fit_ids'];rows=[parents[i] for i in ids];family=c['postfit_family'];arm=c.get('postfit_bank_arm');condition=c.get('postfit_condition')
 if family=='direct':expected='be_unit_contrastive' if condition.startswith('BE_') else 'native_single';bodies=list(range(4)) if condition=='ordinary_full_refit' else [0]
 elif family=='affine':expected='E4' if arm=='E4' else 'equipped_M1';bodies=[run['body'] if arm=='equipped_I4' else 0]
 else:
  expected={'ordinary_M1':'ordinary_M1','ordinary_I4':'ordinary_M1','factorized_allmap_M1':'factorized_allmap_M1','factorized_allmap_I4':'factorized_allmap_M1','shared4_unchanged':'shared4_unchanged','original_gnnm':'original_gnnm'}.get(arm);bodies=list(range(4)) if arm in ('ordinary_I4','factorized_allmap_I4') else [0]
 if len(rows)!=len(bodies) or [p['body'] for p in rows]!=bodies or any(p['seed']!=run['seed'] or p['config'].get('postfit_native_arm')!=expected for p in rows):raise ValueError('Exact same-seed source native bank and own body identity required for this postfit procedure')
 if family=='serving' and any(p['config'].get('backbone')!=c.get('postfit_donor_backbone') for p in rows):raise ValueError('Serving donor backbone identity must match')

def paid_feature_parents(root,study,run,registry,output_root):
 from engine import checkpoint_receipt
 result={}
 ids=validate_lifecycle(run)
 validate_donor_binding(run,{i:registry.run(study['study_id'],i)['contract'] for i in ids})
 if set(ids)!={i for i in run['parents'] if registry.run(study['study_id'],i)['contract']['kind']=='model_fit'}:raise ValueError('Exact direct paid feature cohort dependencies required')
 for i in ids:
  p=registry.run(study['study_id'],i);require_cohort_parent(p['contract'],run);v=p['result'] or {};bound=v.get('selected_checkpoint');cap=p['contract']['budget']['maximum'];completed=v.get('completed_steps');stopped=p['contract']['budget']['policy']=='native_selector_stop'
  if (p['state']!='complete' or v.get('status')!='complete' or not isinstance(completed,int) or not 1<=completed<=cap or not stopped and completed!=cap or not isinstance(bound,dict) or bound.get('fit_id')!=i or bound.get('path')!='selected.pt' or bound.get('step')!=v.get('selected',{}).get('step') or not 1<=bound['step']<=completed):raise ValueError('Complete source donor selected byte authority required')
  q=Path(output_root)/study['study_id']/i/'selected.pt'
  if q.is_symlink() or not q.resolve().is_relative_to(Path(root).resolve()) or checkpoint_receipt(q,fit_id=i,step=bound['step'])!=bound:raise ValueError('Paid selected checkpoint changed')
  f=v.get('postfit_features');record=f.get('receipt') if isinstance(f,dict) else None
  if not isinstance(record,dict) or f.get('selected_checkpoint')!=bound or f.get('cohort_id')!=run['config']['postfit_cohort_id'] or f.get('label_roles')!=['train','select']:raise ValueError('Complete selected frozen feature receipt required before any head fit')
  path=Path(root)/record['path'];raw=path.read_bytes()
  if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()) or len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Frozen native feature bytes changed')
  result[i]={'checkpoint':q,'receipt':bound,'features':path,'feature_receipt':record,'contract':p['contract'],'result':v}
 return result

def validate_special_bank(row,runs):
 c=row['config'];fits=c['fit_parents'];parents=[runs.get(i) for i in fits]
 if any(p is None or p['kind']!='model_fit' or p['config'].get('postfit_source_contract')!=CONTRACT for p in parents) or not set(fits)<=set(row['parents']):raise ValueError('Postfit bank requires exact registered paid fits')
 for p in parents:validate_postfit_run(p)
 if c.get('postfit_fold_bank'):
  if len(fits)!=5 or [p['config'].get('postfit_fold') for p in parents]!=list(range(5)) or any(p['config'].get('postfit_phase')!='head' or p['config'].get('postfit_family')!='serving' for p in parents):raise ValueError('Five distinct exact source fold operator checkpoints required')
  fields=('postfit_rule','postfit_bank_arm','node_inputs','postfit_sources','members','lifecycle')
  if any(any(p['config'][k]!=parents[0]['config'][k] for k in fields) or p['seed']!=parents[0]['seed'] for p in parents):raise ValueError('All fold fits must share exact selected native bank/roles/rule')
  return {'run_id':row['run_id'],'method':row['method'],'split':row['split'],'seed':row['seed'],'fit_parents':fits,'routes':1,'genuine_I4':False,'independent_bodies':0,'views_per_body':1,'genuine_I16':False,'member0_reused':False,'postfit_fold_bank':True,'source_members':parents[0]['config']['members']}
 if len(fits)!=4 or any(p['config'].get('postfit_family')!='affine' or p['config'].get('postfit_phase')!='head' or p['config'].get('postfit_bank_arm') not in ('equipped_I4','equipped_M1') for p in parents) or [p['body'] for p in parents]!=[0,1,2,3] or len({p['config']['lifecycle']['donor_fit_ids'][0] for p in parents})!=4:raise ValueError('Four separately selected own-source I4 heads required')
 if parents[0]['config']['postfit_bank_arm']=='equipped_M1' and not c.get('exact_member0_reference'):raise ValueError('M1/I4 member0 source equivalence must be declared explicitly')
 conditions=[p['config']['postfit_condition'] for p in parents];compatible=conditions==['affine_residual']+['private_affine_residual']*3 and parents[0]['config']['postfit_bank_arm']=='equipped_M1' and c.get('exact_member0_reference')
 if any(p['seed']!=parents[0]['seed'] for p in parents) or not compatible and len(set(conditions))!=1:raise ValueError('Own-selected I4 head contract differs')
 return {'run_id':row['run_id'],'method':row['method'],'split':row['split'],'seed':row['seed'],'fit_parents':fits,'routes':4,'genuine_I4':True,'independent_bodies':4,'views_per_body':1,'genuine_I16':False,'member0_reused':bool(c.get('exact_member0_reference')),'paid_frozen_feature_heads':True}
