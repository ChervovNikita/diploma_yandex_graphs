"""Counts and stopping policy for the complete source private-hop panel."""
POLICY='hop_complete_native_selector_stop'
CONDITIONS=('shared4_own','private_missinghop','full_aux','common_nonfull','allblock_missinghop','factor1_allview','native_M1','original_GNNM')
def validate_hop_run(r):
 c,b,s=r['config'],r['budget'],r['selection'];a=c.get('complete_hop_arm');members=1 if a in ('native_M1','factor1_allview') else 4
 if (a not in CONDITIONS or r['seed'] not in (9101,9203,9307) or r['body'] not in ((0,1,2,3) if members==1 else (0,))
     or r['split']!=0 or c.get('initialization_seed')!=r['seed']+1009*r['body'] or c.get('stage')!='comparative'
     or c.get('backbone')!='PolyFormer_mono_K2' or c.get('members')!=members or c.get('physical_slow_bodies')!=1 or c.get('source_optimizer_count')!=1
     or c.get('learning_rate')!=.005 or c.get('width')!=256 or c.get('classes')!=3 or c.get('depth')!=2
     or c.get('is_original_gnnm')!=(a=='original_GNNM')
     or b!={'unit':'optimizer_updates','policy':POLICY,'maximum':2000,'minimum':251,'patience':250}
     or s!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':250}
     or not r['adapter'].endswith('/adapters/private_hop_complete.py:make_adapter')):
  raise ValueError('Exact complete sourcehop2000/p250 conditions/native groupedAdam/seeds required')
 auxiliary=3 if a in ('private_missinghop','full_aux','common_nonfull','allblock_missinghop','factor1_allview') else 0
 return {'optimizer_objects':1,'physical_slow_bodies':1,'known_optimizer_step_calls':None,'budget_unit':'optimizer_updates','registered_budget_units':2000,
  'law':'complete_source_private_hop_one_native_Adam_factual_and_fixed_auxiliary_credit',
  'native_optimizer_update_cap':2000,'native_optimizer_updates_minimum':251,
  'source_complete_body_constructors':1,'source_optimizer_constructors':1,'source_discarded_body_constructors':0,
  'source_TRAIN_forward_calls_upper_bound':2000*(members+auxiliary),'source_TRAIN_backward_calls_upper_bound':2000*(members+auxiliary)}
