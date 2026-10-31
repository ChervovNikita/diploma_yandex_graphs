"""Exact source counts for the four copied-native relation policies."""
POLICIES=('alphaF','allJ','phiJ','relationJ')
def validate_relation_run(r):
 c,b,s=r['config'],r['budget'],r['selection']
 expected={'stage':'comparative','backbone':'literal_publicV2_WikiCS_Session_Polynormer','architecture':'internal_be','members':4,
  'physical_slow_bodies':1,'source_optimizer_count':1,'width':512,'depth':7,'global_layers':2,'classes':10,
  'learning_rate':.001,'weight_decay':0.,'dropout':.5,'local_transition_updates':100,'initialization_seed':r['seed'],
  'is_original_gnnm':False,'historical_state_inherited':False,'fusion':'mean_probabilities'}
 if (c.get('configured_relation_policy') not in POLICIES or r['seed'] not in (6101,6203,6307) or r['body']!=0 or r['split']!=0
     or any(c.get(k)!=v for k,v in expected.items()) or b!={'unit':'optimizer_updates','policy':'full_budget','maximum':1100}
     or s!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False}
     or not r['adapter'].endswith('/adapters/configured_relation.py:make_adapter')):
  raise ValueError('Literal copiednative relation1100/local100 fixed four-policy source required')
 return {'optimizer_objects':1,'physical_slow_bodies':1,'known_optimizer_step_calls':1100,'budget_unit':'optimizer_updates','registered_budget_units':1100,
  'law':'copied_native_relation_FJ_disjoint_oldstate_VJPs_one_Adam','known_source_TRAIN_forward_calls':17600,
  'known_source_TRAIN_backward_calls':17600,'known_relation_output_cotangent_collections':2200,'known_relation_member_reverse_collections':17600}
