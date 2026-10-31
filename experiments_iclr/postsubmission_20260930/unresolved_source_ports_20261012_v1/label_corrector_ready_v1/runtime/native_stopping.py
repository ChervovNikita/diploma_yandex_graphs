"""Narrow native meanSAGE stopping and bounded-cost contract; stdlib only."""
import copy

NATIVE_BUDGET={'unit':'optimizer_updates','policy':'native_selector_stop','maximum':100000,'minimum':51,'patience':50}
NATIVE_SELECTION={'metric':'objective','direction':'min','ties':'last','evaluation_interval':1,'checkpoint0':False,'patience_steps':50}
ARMS={'peer':{'P1','F1','I4','S4','PA1','IA4','SA4','SM4','SX4','UX4','original_gnnm'},
      'tour':{'P1','F1','I4','S4','PV1','WI4','V1','SW4','C4','R4','original_gnnm'}}

def validate_native_run(run):
 c=run['config']
 if 'rank1_source_contract' in c:
  arm=c.get('rank1_arm');original=arm=='original_GNNM';members=1 if arm in ('point_M1','q_M1') else 4
  if (arm not in ('point4','point_M1','fixed_noise4','q4','q_M1','original_GNNM') or run.get('kind')!='model_fit' or run['budget']!={'unit':'optimizer_updates','policy':'native_selector_stop','maximum':1000,'minimum':301,'patience':300}
      or run['selection']!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':300}
      or c.get('members')!=members or c.get('physical_slow_bodies')!=1 or c.get('source_optimizer_count')!=1
      or c.get('learning_rate')!=.001 or c.get('dropout')!=.2 or c.get('weight_decay')!=0 or c.get('width')!=128 or c.get('depth')!=2 or c.get('rank')!=16 or c.get('classes')!=10
      or c.get('backbone')!=('SAGE' if original else 'source_endpoint_E_VALUE_SAGE') or c.get('is_original_gnnm')!=original or c.get('historical_state_inherited') is not False
      or c.get('initialization_seed')!=run['seed']+1000003*run['body'] or run['seed'] not in (7301,7403,7507) or run['body'] not in (range(4) if arm in ('point_M1','q_M1') else (0,))
      or c.get('fusion')!=('mean_logits' if original else 'mean_probabilities') or c.get('source_TRAIN_forwards_per_update')!=members or c.get('source_TRAIN_backwards_per_update')!=(4 if original else 1)
      or c.get('source_complete_body_constructors')!=1 or c.get('factor_coordinates_per_route')!=(0 if original else 3958)
      or c.get('posterior_law')!=('fixed_noise' if arm=='fixed_noise4' else 'learned_q' if arm in ('q4','q_M1') else 'point')
      or c.get('prior_std')!=.1 or c.get('initial_q_std')!=.1 or c.get('KL_TRAIN_divisor')!=580):raise ValueError('Exact adopted endpoint E Rank1 literal Gaussian/objective/cap1000/patience300 contract required')
  return {'optimizer_objects':1,'physical_slow_bodies':1,'known_optimizer_step_calls':None,'budget_unit':'optimizer_updates','registered_budget_units':1000,'native_stopping_updates_minimum':301,'native_stopping_updates_maximum':1000,'source_complete_body_constructors':1,'source_optimizer_constructors':1,'source_discarded_body_constructors':0,'source_TRAIN_forward_calls_upper_bound':1000*members,'source_TRAIN_backward_calls_upper_bound':1000*(4 if original else 1),'law':'literal_endpoint_E_Rank1_mean_selector_stop300_or_cap1000','executed_update_count_known_before_training':False}
 if 'film_source_contract' in c:
  arm=c.get('film_arm');original=arm=='original_GNNM';members=4 if arm in ('shared4','original_GNNM') else 1
  if (arm not in ('native_M1','shared4','original_GNNM') or run.get('kind')!='model_fit' or run['budget']!={'unit':'optimizer_updates','policy':'native_selector_stop','maximum':1000,'minimum':301,'patience':300}
      or run['selection']!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':300}
      or c.get('members')!=members or c.get('physical_slow_bodies')!=1 or c.get('source_optimizer_count')!=1
      or c.get('learning_rate')!=.001 or c.get('dropout')!=.2 or c.get('weight_decay')!=0 or c.get('width')!=128 or c.get('depth')!=2 or c.get('rank')!=16 or c.get('classes')!=10
      or c.get('backbone')!=('SAGE' if original else 'native_PyG_FiLM') or c.get('is_original_gnnm')!=original or c.get('historical_state_inherited') is not False
      or c.get('initialization_seed')!=run['seed']+1000003*run['body'] or run['seed'] not in (7301,7403,7507) or run['body'] not in (range(4) if arm=='native_M1' else (0,))
      or c.get('fusion')!=('mean_logits' if original else 'mean_probabilities') or c.get('source_TRAIN_forwards_per_update')!=members or c.get('source_TRAIN_backwards_per_update')!=(4 if original else 1)
      or c.get('source_complete_body_constructors')!=1):raise ValueError('Exact source FiLM cap1000/patience300/own CE/new original GNNM contract required')
  return {'optimizer_objects':1,'physical_slow_bodies':1,'known_optimizer_step_calls':None,'budget_unit':'optimizer_updates','registered_budget_units':1000,'native_stopping_updates_minimum':301,'native_stopping_updates_maximum':1000,'source_complete_body_constructors':1,'source_optimizer_constructors':1,'source_discarded_body_constructors':0,'source_TRAIN_forward_calls_upper_bound':1000*members,'source_TRAIN_backward_calls_upper_bound':1000*(4 if original else 1),'law':'literal_FiLM_strict_first_accuracy_stop300_or_cap1000','executed_update_count_known_before_training':False}
 c=run['config'];arm=c.get('meansage_arm');family=c.get('meansage_family');original=arm=='original_gnnm'
 members=4 if arm in {'S4','SA4','SM4','SX4','UX4','PV1','V1','SW4','C4','R4','original_gnnm'} else 1
 bodies=4 if arm=='UX4' else 1
 if (family not in ARMS or arm not in ARMS[family] or run.get('kind')!='model_fit'
       or run.get('budget')!=NATIVE_BUDGET or run.get('selection')!=NATIVE_SELECTION
       or c.get('architecture')!=('original_gnnm' if original else 'native_meansage_'+family)
       or c.get('source_optimizer_count')!=1 or c.get('physical_slow_bodies')!=bodies
       or c.get('members')!=members or c.get('learning_rate')!=.001 or c.get('dropout')!=.4
       or c.get('weight_decay')!=0 or c.get('width')!=(128 if original else 32)
       or c.get('depth')!=2 or c.get('classes')!=8 or c.get('backbone')!='SAGE'
       or c.get('is_original_gnnm')!=original or c.get('historical_state_inherited') is not False
       or c.get('initialization_seed')!=run['seed']+1000003*run['body']
       or c.get('source_TRAIN_forwards_per_update')!=(4 if original else 1)
       or c.get('source_TRAIN_backwards_per_update')!=(4 if original else 1)
       or c.get('source_complete_body_constructors')!=(4 if arm in {'I4','IA4','WI4','UX4'} else 1)
       or c.get('source_constructor_selection')!=('full_four_factory_choose_run_body' if arm in {'I4','IA4','WI4'} else 'whole_single_source_factory')):
  raise ValueError('Exact native meanSAGE peer/tour own source recipe and truthful factory/stopping contract required')
 if run['body'] not in (range(4) if arm in {'P1','PA1','I4','IA4','WI4'} else (0,)):
  raise ValueError('Exact independent source body index required')
 if c.get('fusion')!=('mean_logits' if original else 'mean_probabilities'):
  raise ValueError('Complete original/native serving law required')
 return {'optimizer_objects':1,'physical_slow_bodies':bodies,'known_optimizer_step_calls':None,
  'budget_unit':'optimizer_updates','registered_budget_units':100000,
  'native_stopping_updates_minimum':51,'native_stopping_updates_maximum':100000,
  'source_complete_body_constructors':c['source_complete_body_constructors'],
  'source_optimizer_constructors':1,'source_discarded_body_constructors':c['source_complete_body_constructors']-bodies,
  'source_TRAIN_forward_calls_upper_bound':100000*c['source_TRAIN_forwards_per_update'],
  'source_TRAIN_backward_calls_upper_bound':100000*c['source_TRAIN_backwards_per_update'],
  'law':'native_meanSAGE_'+family+'_post_update_latest_min_total_stop50_or_cap100000',
  'executed_update_count_known_before_training':False}

def require_completion(run):
 if run['budget']['policy']=='native_selector_stop':validate_native_run(run)
 elif run['budget']['policy']!='full_budget':raise ValueError('Unsupported completion policy')

def native_count_quantities(semantics):
 return {'native_stopping_fit_groups':1,'native_optimizer_update_cap_sum':semantics['native_stopping_updates_maximum'],
  'native_optimizer_updates_minimum_sum':semantics['native_stopping_updates_minimum'],
  'registered_optimizer_update_caps':semantics['native_stopping_updates_maximum'],
  'source_complete_body_constructors':semantics['source_complete_body_constructors'],
  'source_discarded_body_constructors':semantics['source_discarded_body_constructors'],
  'source_optimizer_constructors':semantics['source_optimizer_constructors'],
  'source_TRAIN_forward_calls_upper_bound':semantics['source_TRAIN_forward_calls_upper_bound'],
  'source_TRAIN_backward_calls_upper_bound':semantics['source_TRAIN_backward_calls_upper_bound']}
