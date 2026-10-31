"""Narrow native HGT cap300/p30/scheduler/mixed block cost typing; stdlib only."""
BUDGET={'unit':'optimizer_updates','policy':'hgt_native_selector_stop','maximum':300,'minimum':31,'patience':30}
SELECTION={'metric':'pooled_nll','direction':'min','ties':'last','evaluation_interval':1,'checkpoint0':False,'patience_steps':30}
ARMS={'native_M1','factor_M1','global_BE','shared_relation','CP','unrestricted','untied_HGT','wider_BE','original_gnnm','own/own','pool/pool','pool/own','own/pool'}

def validate_hgt_run(run):
 c=run['config'];arm=c.get('hgt_arm');members=1 if arm in ('native_M1','factor_M1') else 4;physical=4 if arm=='untied_HGT' else 1
 constructors=1 if arm=='original_gnnm' else 5 if arm=='untied_HGT' else 3 if arm=='wider_BE' else 2
 backs=2 if arm in ('pool/own','own/pool') else 1
 if (arm not in ARMS or run.get('kind')!='model_fit' or run['budget']!=BUDGET or run['selection']!=SELECTION
      or c.get('architecture')!='native_HGT_typed' or c.get('backbone')!='native_HGB_HGT' or c.get('classes')!=4
      or c.get('width')!=(72 if arm=='wider_BE' else 64) or c.get('depth')!=3 or c.get('heads')!=8
      or c.get('dropout')!=.2 or c.get('weight_decay')!=.0001 or c.get('learning_rate')!=.001 or c.get('fusion')!='mean_logits'
      or c.get('members')!=members or c.get('source_optimizer_count')!=1 or c.get('physical_slow_bodies')!=physical
      or c.get('source_complete_body_constructors')!=constructors or c.get('source_scheduler_rule')!='OneCycle300 maxlr.001 pct_start.05 cosine step(update)'
      or c.get('source_TRAIN_forwards_per_update')!=members or c.get('source_TRAIN_backwards_per_update')!=backs
      or c.get('historical_state_inherited') is not False or c.get('is_original_gnnm')!=(arm=='original_gnnm')
      or c.get('initialization_seed')!=run['seed']+1000003*run['body']):raise ValueError('Exact native HGT typed/source300/p30/OneCycle/mixed ownership contract required')
 if run['body'] not in (range(4) if arm in ('native_M1','factor_M1') else (0,)):raise ValueError('Exact own HGT learner index')
 return {'optimizer_objects':1,'physical_slow_bodies':physical,'known_optimizer_step_calls':None,'budget_unit':'optimizer_updates','registered_budget_units':300,
  'native_stopping_updates_minimum':31,'native_stopping_updates_maximum':300,'source_complete_body_constructors':constructors,
  'source_optimizer_constructors':1,'source_discarded_body_constructors':constructors-physical,
  'source_TRAIN_forward_calls_upper_bound':300*members,'source_TRAIN_backward_calls_upper_bound':300*backs,
  'law':'native_HGT_OneCycle300_latest_pooled_SELECT_CE_p30_'+arm,'executed_update_count_known_before_training':False}
