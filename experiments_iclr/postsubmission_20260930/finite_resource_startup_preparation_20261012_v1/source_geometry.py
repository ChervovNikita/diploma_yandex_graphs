"""Read actual declared geometry, never invent width/depth1 for missing fields.

Pure metadata; source/runtime equivalence and dominance remain separate proofs.
"""
from __future__ import annotations
import copy

SCALARS=('width','depth','heads','members','classes','stalk_dim','hidden_channels','hidden_dim','num_layers',
 'ffn_multiplier','physical_slow_bodies','source_complete_native_paths','source_optimizer_count',
 'local_transition_updates','attention_dim','rank','routes','basis_rank')
NESTED=('native','encoder','predictor','decoder','source_args','native_args','provider_graph','training',
 'nsd_geometry','source_geometry','architecture_args','posterior_samples','curvature_config')


def extract(config):
    declared={k:copy.deepcopy(config[k]) for k in SCALARS+NESTED if k in config}
    dimensions={k:config.get(k) for k in ('width','depth','heads','members','stalk_dim')}
    native=config.get('native')
    if isinstance(native,dict):
        # PolyFormer source allocations use these fields; top-level defaults
        # in the older generic catalogue did not describe the actual model.
        for destination,source in (('width','hidden'),('depth','nlayer'),('heads','n_head')):
            if source in native:dimensions[destination]=native[source]
        dimensions['ffn_width']=native.get('d_ffn')
        dimensions['native_K']=native.get('K');dimensions['native_multi']=native.get('multi')
    for key in ('encoder','predictor','decoder'):
        value=config.get(key)
        if isinstance(value,dict):
            for name in ('hidden_channels','out_channels','num_layers','heads','hidden_dim','depth','width','input_channels'):
                if name in value:dimensions[key+'.'+name]=value[name]
    return {'actual_declared_geometry':declared,'dimensions':dimensions,
            'missing_common_dimensions':[k for k,v in dimensions.items() if v is None],
            'source_specific_static_geometry_not_a_resource_envelope':True,
            'dominance_or_runtime_admission_proved':False}


def source_branch(config):
    names=('backbone','architecture','fusion','normalization','objective_variant','objective_coefficients',
           'nsd_arm','nsd_stage','growth_arm','growth_stage','label_arm','label_stage','mechanism_arm',
           'view_arm','pt_arm','ported_family','ported_arm','curvature_arm','curvature_stage',
           'link_prototype_arm','bayesian_bsnn_arm','bayesian_bsnn_variant','precision_policy','stage')
    return {k:copy.deepcopy(config[k]) for k in names if k in config}
