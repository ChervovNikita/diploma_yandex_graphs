"""Exact Bayesian source cap500/patience200 and honest bounded costs."""
BUDGET={'unit':'optimizer_updates','policy':'bsnn_native_selector_stop','maximum':500,'minimum':201,'patience':200}
SELECTION={'metric':'auroc','direction':'max','ties':'first','evaluation_interval':1,'checkpoint0':False,'patience_steps':200,'secondary':'nll','tertiary':'earlier_epoch'}
VARIANTS={'bsnn_cayley_d2':(2,False,.3,0.),'bsnn_richer_cayley_d2_l4':(4,True,.2,.4)}


def validate_bsnn_run(run):
    c=run['config'];variant=c.get('bayesian_bsnn_variant');arm=c.get('bayesian_bsnn_arm');members=4 if arm=='original_GNNM' else 1
    layers,second,dropout,input_dropout=VARIANTS.get(variant,(None,)*4)
    if (variant not in VARIANTS or arm not in ('native_M1','original_GNNM') or run.get('kind')!='model_fit'
            or run.get('budget')!=BUDGET or run.get('selection')!=SELECTION
            or c.get('architecture')!='native_Bayesian_BSNN_typed' or c.get('backbone')!='original_BayesBundleSheafDiffusion_d2'
            or c.get('classes')!=2 or c.get('width')!=32 or c.get('depth')!=layers
            or c.get('second_linear')!=second or c.get('dropout')!=dropout or c.get('input_dropout')!=input_dropout
            or c.get('learning_rate')!=.01 or c.get('weight_decay')!=.0005
            or c.get('members')!=members or c.get('physical_slow_bodies')!=1 or c.get('source_optimizer_count')!=1
            or c.get('source_complete_body_constructors')!=1 or c.get('source_TRAIN_forwards_per_update')!=members
            or c.get('source_TRAIN_backwards_per_update')!=members or c.get('source_SELECT_draws')!=4*members
            or c.get('source_KL_rule')!='native_incidence_mean_sigmoid40_no_layer_multiplier'
            or c.get('source_optimizer_rule')!='source_grouped_Adam_lr.01_both_decay.0005'
            or c.get('historical_state_inherited') is not False or c.get('is_original_gnnm')!=(arm=='original_GNNM')
            or c.get('initialization_seed')!=run['seed']+1009*run['body']
            or not run.get('adapter','').endswith('/adapters/bsnn_bayesian.py:make_adapter')):
        raise ValueError('Exact native Bayesian BSNN tuple/KL/SciPy/groupedAdam/source500/p200 law required')
    if run['body'] not in (range(4) if arm=='native_M1' else (0,)):
        raise ValueError('Own independent source body or one shared boundary bank required')
    if c.get('fusion')!=('mean_probabilities' if arm=='native_M1' else 'mean_logits'):
        raise ValueError('Exact native/adapted Bayesian serving law required')
    return {'optimizer_objects':1,'physical_slow_bodies':1,'known_optimizer_step_calls':None,
            'budget_unit':'optimizer_updates','registered_budget_units':500,
            'native_stopping_updates_minimum':201,'native_stopping_updates_maximum':500,
            'source_complete_body_constructors':1,'source_optimizer_constructors':1,'source_discarded_body_constructors':0,
            'source_TRAIN_forward_calls_upper_bound':500*members,'source_TRAIN_backward_calls_upper_bound':500*members,
            'source_SELECT_forward_calls_upper_bound':500*4*members,
            'source_TRAIN_native_posterior_layer_samples_upper_bound':500*members*layers,
            'source_SELECT_native_posterior_layer_samples_upper_bound':500*4*members*layers,
            'law':'native_Bayesian_BSNN_one_draw_NLL_sigmoid40KL_4MC_SELECT_p200_cap500_'+variant+'_'+arm,
            'executed_update_count_known_before_training':False}
