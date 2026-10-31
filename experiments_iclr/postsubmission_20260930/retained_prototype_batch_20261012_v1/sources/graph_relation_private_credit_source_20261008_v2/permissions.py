"""Exact pinned WikiCS roles plus native private local scorer and tied-QK banks."""
import json
import re

POLICIES = ('alphaF', 'allJ', 'phiJ', 'relationJ')
QK_MAPS = tuple('models.0.body.global_attn.k_lins.' + str(i) for i in range(2))
SCORERS = tuple('models.0.body.local_convs.' + str(i) +
    '.parametrizations.' + leaf + '.original' for i in range(7)
    for leaf in ('att_src', 'att_dst'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def partition(session, profile_path, policy):
    require(policy in POLICIES, 'One fixed graph-relation policy required')
    profile = json.loads(profile_path.read_text()); spec = profile['tasks']['wikics']
    require(session.task == 'wikics' and session.core_provenance == profile['core_sha256']
            and session.config['model'] == spec['model_config']
            and session.native_provenance == spec['native_provenance'], 'Pinned full WikiCS sources/recipe')
    model = session.model; torch = session.torch
    require(type(model) is session.core['models'].Ensemble and not model.independent
            and model.members == 4 and len(model.models) == 1
            and type(model.models[0]) is session.core['models'].WikiBackbone, 'One native shared M4 body')
    body = model.models[0].body; global_attention = body.global_attn
    require(global_attention.qk_shared is True and not hasattr(global_attention, 'q_lins')
            and global_attention.num_layers == 2 and len(global_attention.k_lins) == 2
            and len(body.local_convs) == 7, 'Exact native q=k tied projection; no Q/K fallback')
    factor_type = session.core['factors'].FactorLinear
    maps = {name: module for name, module in model.named_modules(remove_duplicate=False)
            if isinstance(module, factor_type)}
    require(set(maps) == set(spec['factor_maps']) and set(QK_MAPS) <= set(maps)
            and not spec['extra_private'], 'Exact dense factor map inventory')
    roles = {}
    for name, entry in spec['factor_maps'].items():
        module = maps[name]; shape = entry['weight_shape']
        require(list(module.weight.shape) == shape and list(module.r.shape) == [4, shape[1]]
                and list(module.s.shape) == [4, shape[0]]
                and not (set(module._parameters) - {'weight', 'bias', 'r', 's'}), 'Native dense map shape/parameters')
        for leaf in ('r', 's'):
            roles[name + '.' + leaf] = entry['role']
        roles[name + '.weight'] = 'shared_own'
        if module.bias is not None:
            roles[name + '.bias'] = 'shared_own'
    named = list(model.named_parameters(remove_duplicate=False))
    require(len({id(p) for _, p in named}) == len(named), 'Parameter aliases are unsupported')
    names = {name for name, _ in named}
    require(set(SCORERS) <= names, 'Every original local scorer needs its copied private bank')
    for name, parameter in named:
        require(parameter.requires_grad and parameter.dtype == torch.float32, 'Live original float32 parameters')
        if name in SCORERS:
            require(list(parameter.shape) == [4, 1, 1, 512], 'Exact copied local scorer bank shape')
            roles[name] = 'private_local_scorer'
        elif name not in roles:
            require(not re.fullmatch(r'models\.0\.body\.local_convs\.\d+\.(att_src|att_dst)', name),
                    'An unused common local scorer must not remain')
            require(any(re.fullmatch(pattern, name) for pattern in spec['shared_parameter_patterns']),
                    'Unknown parameter has no fallback role: ' + name)
            roles[name] = 'shared_own'
    require(set(roles) == names, 'Exhaustive unique parameter roles required')
    dense = tuple((name, p) for name, p in named if roles[name] == 'internal_private')
    boundary = tuple((name, p) for name, p in named if roles[name] == 'boundary_private_own')
    require(len(dense) == 56 and len(boundary) == 6, 'Original internal/boundary factor counts')
    relation_names = set(SCORERS) | {name + '.' + leaf for name in QK_MAPS for leaf in ('r', 's')}
    relation = tuple((name, p) for name, p in named if name in relation_names)
    require(len(relation) == 18, 'Four tied-QK factor tensors plus fourteen local scorer banks')
    target = dense if policy == 'phiJ' else relation
    target_ids = {id(p) for _, p in target}
    other = tuple((name, p) for name, p in named if id(p) not in target_ids)
    require(target and other and len(target) + len(other) == len(named)
            and not (target_ids & {id(p) for _, p in other}), 'Two disjoint exhaustive VJP groups')
    require(len(session.optimizers) == 1 and type(session.optimizers[0]) is torch.optim.Adam,
            'One original native Adam')
    optimized = [p for group in session.optimizers[0].param_groups for p in group['params']]
    require(len(optimized) == len(named) and {id(p) for p in optimized} == {id(p) for _, p in named},
            'Every bank must be installed before fresh Adam')
    for group in session.optimizers[0].param_groups:
        require(group['lr'] == .001 and group['eps'] == 1e-8 and group['weight_decay'] == 0.
                and group['betas'] == (.9, .999) and group['amsgrad'] is False, 'Original native Adam settings')
    selectors = ('F', 'F') if policy == 'alphaF' else ('J', 'J') if policy == 'allJ' else ('J', 'F')
    return dict(all=tuple(named), groups=(target, other), selectors=selectors, roles=roles,
                internal_dense=dense, relation=relation, policy=policy)


def unchanged(session):
    current = tuple(session.model.named_parameters(remove_duplicate=False))
    old = session.relation_partition['all']
    require(len(current) == len(old) and all(a == b and p is q for (a, p), (b, q) in zip(current, old)),
            'Parameter names/objects changed after the exact partition')


def metadata(session):
    part = session.relation_partition
    return dict(policy=part['policy'], parameter_roles=dict(part['roles']),
        VJP_groups=[dict(names=[name for name, _ in group], loss=loss,
            tensors=len(group), scalars=sum(p.numel() for _, p in group))
            for group, loss in zip(part['groups'], part['selectors'])],
        local_scorer_bank_names=list(SCORERS), tied_QK_map_names=list(QK_MAPS),
        tied_QK_semantics='Native q=k=sigmoid(k_lins[i](x)); one bank per physical map, no double counting',
        private_relation_tensors=18, dense_internal_tensors=56, private_boundary_tensors=6,
        unknown_or_missing_parameter_fallback=False)
