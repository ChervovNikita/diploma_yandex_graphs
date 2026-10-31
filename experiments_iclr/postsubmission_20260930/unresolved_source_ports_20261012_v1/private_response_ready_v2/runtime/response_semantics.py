"""Typed full180 private-response source cycles with an exact same-fit source60 prefix."""
from adapters.private_response import CONTRACT
PREFIXES=[{'prefix_id':'source60','cap_step':60,'close_on_selector_stop':False}]
SOURCE_WORK={'shared_f4':(4,1,2,1,9,9,9),'capable_single':(1,1,2,1,3,3,3),'untied4':(4,1,2,1,9,12,9),'ordinary_native4':(4,3,3,0,9,12,3),'row0_single':(1,1,2,1,9,9,9),'native_M1':(1,3,3,0,3,3,3),'original_GNNM':(4,3,3,0,12,12,12)}
FACTORS=(283136447778829174,949102575051525973,521021139669682083)
def validate_response_run(run):
 c,b,s=run['config'],run['budget'],run['selection'];arm=c.get('response_arm');w=SOURCE_WORK.get(arm);seed=run['seed'];body=run['body']
 if w is None:raise ValueError('Literal source response arm required')
 members,adam,maps,functional,forward,encoder,back=w
 expected={'stage':'comparative','private_response_source_contract':CONTRACT,'backbone':'literal_source_NCN_puregcn256_L1','members':members,'physical_slow_bodies':4 if arm in ('untied4','ordinary_native4') else 1,'source_optimizer_count':1,'source_complete_body_constructors':4 if arm in ('untied4','ordinary_native4') else 1,'source_complete_episodes_per_cycle':61,'source_TorchAdam_calls_per_episode':adam,'source_committed_model_maps_per_episode':maps,'source_functional_private_commits_per_episode':functional,'source_virtual_private_maps_per_episode':functional,'source_TRAIN_forwards_per_episode':forward,'source_native_encoder_forwards_per_episode':encoder,'source_TRAIN_backwards_per_episode':back,'learning_rate':.001,'weight_decay':0.,'inner_size':256,'outer_size':64,'initialization_seed':seed+5*body,'factor_seed':FACTORS[seed] if seed in (0,1,2) else None,'is_original_gnnm':arm=='original_GNNM','historical_state_inherited':False,'response_horizon_extension':'source60_to_full180_same_fit'}
 if (seed not in (0,1,2) or body not in ((0,1,2,3) if arm=='native_M1' else (0,)) or any(c.get(k)!=v for k,v in expected.items())
     or b!={'unit':'epochs','policy':'full_budget','maximum':180,'prefixes':PREFIXES}
     or s!={'metric':'MRR','direction':'max','ties':'first','evaluation_interval':5,'checkpoint0':False,'patience_steps':11}
     or not run['adapter'].endswith('/adapters/private_response.py:make_adapter') or run['parents']):
  raise ValueError('Exact source60/full180 endpoint response procedure,cycle61,sourceAdam/objective/sampler/eval5 andowncoldbody required')
 episodes=180*61
 return {'optimizer_objects':1,'physical_slow_bodies':expected['physical_slow_bodies'],'known_optimizer_step_calls':episodes*adam,'budget_unit':'epochs','registered_budget_units':180,'law':'literal_private_response_source_full180_no_stop_source60_selected_prefix',
  'known_source_TRAIN_forward_calls':episodes*forward,'known_source_TRAIN_backward_calls':episodes*back,'known_response_complete_episodes':episodes,'known_response_source_committed_model_maps':episodes*maps,'known_response_functional_private_commits':episodes*functional,'known_response_discarded_virtual_private_maps':episodes*functional,'known_response_native_encoder_forwards':episodes*encoder,'known_response_source60_selected_prefixes':1,'known_response_fresh_prefix_fit_groups':0}
