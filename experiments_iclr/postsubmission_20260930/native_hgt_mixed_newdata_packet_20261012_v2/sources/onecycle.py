"""Portable data-only state for the fixed native Torch2.1 cosine OneCycleLR.

Torch2.1 state_dict includes bound anneal_func. Do not pickle that method or
weaken weights_only loading. Keep the receiver's fixed constructor cosine method.
"""
import math


def data_only(value):
    if value is None or type(value) in (bool,int,str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list,tuple):
        return type(value)(data_only(item) for item in value)
    if type(value) is dict and all(type(key) in (str,int) for key in value):
        return {key:data_only(item) for key,item in value.items()}
    raise ValueError('OneCycle state must contain only finite native data values')


def cosine_guard(scheduler):
    if scheduler.total_steps!=300 or getattr(scheduler.anneal_func,'__name__',None)!='_annealing_cos':
        raise ValueError('Fixed native300-step cosine OneCycle constructor required')


def portable_state(scheduler):
    cosine_guard(scheduler)
    state = scheduler.state_dict()
    if 'anneal_func' not in state:
        raise ValueError('Pinned Torch2.1 OneCycle state contract differs')
    return dict(schema='fixed_native_cosine_OneCycle_data_v1',anneal_strategy='cos',
                scheduler_state=data_only({key:value for key,value in state.items() if key!='anneal_func'}))


def restore_state(scheduler, portable):
    cosine_guard(scheduler)
    if set(portable)!={'schema','anneal_strategy','scheduler_state'} or portable['schema']!='fixed_native_cosine_OneCycle_data_v1' or portable['anneal_strategy']!='cos':
        raise ValueError('Portable native cosine OneCycle schema differs')
    state = data_only(portable['scheduler_state'])
    fresh = scheduler.state_dict()
    if set(state)!=set(fresh)-{'anneal_func'}:
        raise ValueError('Pinned OneCycle state fields differ')
    for name in ('total_steps','_schedule_phases','cycle_momentum','use_beta1','base_lrs'):
        if state[name]!=fresh[name]:
            raise ValueError('Native OneCycle constructor configuration differs: '+name)
    # load_state_dict updates fields supplied; omitted anneal_func remains the
    # receiver's own cosine method, bound to its correct new scheduler instance.
    scheduler.load_state_dict(state)
