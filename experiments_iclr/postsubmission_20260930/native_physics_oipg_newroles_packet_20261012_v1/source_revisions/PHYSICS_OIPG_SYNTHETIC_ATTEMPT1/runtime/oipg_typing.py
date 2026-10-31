"""Fixed native WikiPolynormer1100/two-view OIPG contract; metadata only."""
POLICIES={'O','I','P','G','native_M1'}
def validate_oipg_run(run):
    c=run['config'];policy=c.get('oipg_policy');native=policy=='native_M1';members=1 if native else 4
    if (policy not in POLICIES or run['kind']!='model_fit'
            or run['budget']!={'unit':'optimizer_updates','policy':'full_budget','maximum':1100}
            or run['selection']!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':1101}
            or c.get('architecture')!='source_internal_BE_OIPG' or c.get('backbone')!='native_WikiPolynormer'
            or c.get('width')!=512 or c.get('depth')!=7 or c.get('global_depth')!=2 or c.get('classes')!=10
            or c.get('learning_rate')!=.001 or c.get('weight_decay')!=0 or c.get('members')!=members
            or c.get('physical_slow_bodies')!=1 or c.get('source_optimizer_count')!=1 or c.get('base_arm')!=('single' if native else 'be_unit')
            or c.get('allocation_lambda') not in (0.,.05,.5) or c.get('source_local_epochs')!=100
            or c.get('source_TRAIN_forwards_per_update')!=2*members
            or c.get('historical_state_inherited') is not False or c.get('fusion')!='mean_probabilities'
            or c.get('initialization_seed')!=run['seed']+1009*run['body']):
        raise ValueError('Exact public source OIPG/native1100epoch/two-view/permission/selector recipe required')
    return {'optimizer_objects':1,'physical_slow_bodies':1,'known_optimizer_step_calls':1100,
            'budget_unit':'optimizer_updates','registered_budget_units':1100,'source_complete_body_constructors':1,
            'known_source_TRAIN_forward_calls':1100*2*members,'known_source_TRAIN_backward_calls':1100*(3 if policy=='P' else 2 if not native else 1),
            'known_source_old_state_gradient_collections':1100*(3 if policy=='P' else 2 if not native else 1),
            'law':'native_WikiPolynormer1100_localbest_transition_two_views_'+policy}
