"""Fixed native WikiPolynormer1100/two-view OIPG contract; metadata only."""
POLICIES={'O','I','P','G','native_M1','fourbody_coupled'}
def validate_oipg_run(run):
    c=run['config'];policy=c.get('oipg_policy');native=policy=='native_M1';coupled=policy=='fourbody_coupled';members=1 if native else 4;owners=4 if coupled else 1
    if (policy not in POLICIES or run['kind']!='model_fit'
            or run['budget']!={'unit':'epochs' if coupled else 'optimizer_updates','policy':'full_budget','maximum':1100}
            or run['selection']!={'metric':'accuracy','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':1101}
            or c.get('architecture')!='source_internal_BE_OIPG' or c.get('backbone')!='native_WikiPolynormer'
            or c.get('width')!=512 or c.get('depth')!=7 or c.get('global_depth')!=2 or c.get('classes')!=10
            or c.get('learning_rate')!=.001 or c.get('weight_decay')!=0 or c.get('members')!=members
            or c.get('physical_slow_bodies')!=owners or c.get('source_optimizer_count')!=owners or c.get('base_arm')!=('single' if native else 'independent4_contrastive' if coupled else 'be_unit')
            or c.get('allocation_lambda') not in (0.,.05,.5) or c.get('source_local_epochs')!=100
            or c.get('source_TRAIN_forwards_per_update')!=2*members
            or c.get('historical_state_inherited') is not False or c.get('fusion')!='mean_probabilities'
            or c.get('initialization_seed')!=run['seed']+1009*run['body']):
        raise ValueError('Exact public source OIPG/native1100epoch/two-view/permission/selector recipe required')
    reverse=3 if policy=='P' else 1 if native or coupled else 2
    return {'optimizer_objects':owners,'physical_slow_bodies':owners,'known_optimizer_step_calls':1100*owners,
            'budget_unit':run['budget']['unit'],'registered_budget_units':1100,'source_complete_body_constructors':owners,'source_discarded_body_constructors':0,'source_optimizer_constructors':owners,
            'known_source_TRAIN_forward_calls':1100*2*members,'known_source_TRAIN_backward_calls':1100*reverse,
            'known_source_old_state_gradient_collections':1100*reverse,
            'law':'native_WikiPolynormer1100_localbest_transition_two_views_'+policy}
