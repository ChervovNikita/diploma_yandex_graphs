"""Literal TABM saved-best/per-ten/p300/cap5000 typing; no science imports."""
ARMS={'O4_stem_IID','O4_stem_SIGN','O4_graph_IID','O4_graph_SIGN','O4_graph_BALANCED','factor_M1_stem','factor_M1_graph','MS4_IID','MS4_SIGN','coupled_I4','original_gnnm','source_meanSAGE'}
def budget(arm):
    return {'unit':'epochs' if arm=='coupled_I4' else 'optimizer_updates','policy':'tabm_native_selector_stop',
            'maximum':100000 if arm=='source_meanSAGE' else 5000,'minimum':51 if arm=='source_meanSAGE' else 310,
            'patience':50 if arm=='source_meanSAGE' else 300}
def selection(arm):
    return {'metric':'objective' if arm=='source_meanSAGE' else 'accuracy','direction':'min' if arm=='source_meanSAGE' else 'max',
            'ties':'last' if arm=='source_meanSAGE' else 'first','evaluation_interval':1,'checkpoint0':False,
            'patience_steps':50 if arm=='source_meanSAGE' else 300}
def validate_tabm_run(run):
    c=run['config'];arm=c.get('physics_arm');source=arm=='source_meanSAGE';coupled=arm=='coupled_I4'
    routes=1 if arm.startswith('factor_M1') or arm.startswith('MS4') or source else 4
    owners=4 if coupled else 1;forwards=4 if arm.startswith('O4') or arm.startswith('MS4') or coupled or arm=='original_gnnm' else 1
    if (arm not in ARMS or run['kind']!='model_fit' or run['budget']!=budget(arm) or run['selection']!=selection(arm)
            or c.get('architecture')!='source_TABM_joint_dropout' or c.get('backbone')!='SAGE' or c.get('depth')!=2
            or c.get('width')!=(32 if source else 128) or c.get('dropout')!=(.4 if source else .2)
            or c.get('classes') not in (5,10) or c.get('learning_rate')!=.001 or c.get('weight_decay')!=0
            or c.get('members')!=routes or c.get('physical_slow_bodies')!=owners or c.get('source_optimizer_count')!=owners
            or c.get('source_TRAIN_forwards_per_round')!=forwards or c.get('historical_state_inherited') is not False
            or c.get('fusion')!='mean_logits' or c.get('initialization_seed')!=run['seed']+1000003*run['body']):
        raise ValueError('Literal TABM or source meanSAGE completion/ownership recipe required')
    if c['classes']==10 and arm!='original_gnnm':raise ValueError('Only literal original GNNM also controls OIPG WikiCS')
    cap=run['budget']['maximum'];minimum=run['budget']['minimum']
    return {'optimizer_objects':owners,'physical_slow_bodies':owners,'known_optimizer_step_calls':None,
            'budget_unit':run['budget']['unit'],'registered_budget_units':cap,
            'native_stopping_updates_minimum':minimum*owners,'native_stopping_updates_maximum':cap*owners,
            'source_complete_body_constructors':owners,'source_discarded_body_constructors':0,'source_optimizer_constructors':owners,
            'source_TRAIN_forward_calls_upper_bound':cap*forwards,'source_TRAIN_backward_calls_upper_bound':cap*forwards,
            'source_detached_prefix_forward_calls_upper_bound':cap*4 if arm.startswith('O4') or arm.startswith('MS4') or coupled else 0,
            'law':'source_TABM_per_ten_saved_best_p300_or_meanSAGE_latest_total_p50',
            'executed_update_count_known_before_training':False}
