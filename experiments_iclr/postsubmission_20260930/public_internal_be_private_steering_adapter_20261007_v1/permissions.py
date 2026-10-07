"""Explicit audited factor roles, never an all-private-is-internal fallback."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def partition_parameters(session):
    profile = json.loads((HERE / 'BACKBONE_AUDIT.json').read_text())
    if session.core_provenance != profile['core_sha256']:
        raise ValueError('Exact publicV2 core required')
    filename = Path(session.forward.__func__.__code__.co_filename)
    if hashlib.sha256(filename.read_bytes()).hexdigest() != profile['portable_py_sha256']:
        raise ValueError('Original publicV2 member RNG/forward wrapper required')
    task = session.task
    spec = profile['tasks'][task]
    if session.config['model'] != spec['model_config']:
        raise ValueError('Audited fixed backbone configuration required')
    if session.native_provenance != spec['native_provenance']:
        raise ValueError('Audited pinned native source required')
    model = session.model
    if type(model) is not session.core['models'].Ensemble or model.independent or model.members != 4 or len(model.models) != 1:
        raise ValueError('This adapter covers the existing shared four-member BE bank only')
    body = model.models[0]
    if type(body) is not getattr(session.core['models'], spec['backbone_class']):
        raise ValueError('Audited backbone class required')
    if task == 'collab' and type(body.decoder).__name__ != 'CNLinkPredictor':
        raise ValueError('Other NCN predictors/additive score heads need a separate audit')
    factor_type = session.core['factors'].FactorLinear
    maps = {name: module for name, module in model.named_modules(remove_duplicate=False)
            if isinstance(module, factor_type)}
    if set(maps) != set(spec['factor_maps']):
        raise ValueError('Unknown/missing/aliased factor map; do not assume it is internal')
    roles = {}; private_ids = set()
    for path, entry in spec['factor_maps'].items():
        module = maps[path]
        if list(module.weight.shape) != entry['weight_shape']:
            raise ValueError('Audited dense-map shape changed: ' + path)
        if list(module.r.shape) != [4, entry['weight_shape'][1]] or list(module.s.shape) != [4, entry['weight_shape'][0]]:
            raise ValueError('Original private rank-one factor shape required')
        if set(module._parameters) - {'weight', 'bias', 'r', 's'}:
            raise ValueError('Unclassified parameter inside factor map')
        for leaf in ('r', 's'):
            name = path + '.' + leaf; parameter = getattr(module, leaf)
            roles[name] = entry['role']; private_ids.add(id(parameter))
        roles[path + '.weight'] = 'shared_own'
        if module.bias is not None:
            roles[path + '.bias'] = 'shared_own'
    for name, entry in spec['extra_private'].items():
        parameter = dict(model.named_parameters())[name]
        if list(parameter.shape) != entry['shape']:
            raise ValueError('Audited existing message factor shape changed')
        roles[name] = entry['role']; private_ids.add(id(parameter))
    named = list(model.named_parameters(remove_duplicate=False))
    if len({id(parameter) for _, parameter in named}) != len(named):
        raise ValueError('Aliased parameter names need explicit permission resolution')
    for name, parameter in named:
        if not parameter.requires_grad or parameter.dtype != session.torch.float32:
            raise ValueError('Original trainable float32 parameters required')
        if name not in roles:
            if any(re.fullmatch(pattern, name) for pattern in spec['shared_parameter_patterns']):
                roles[name] = 'shared_own'
            else:
                raise ValueError('Unknown parameter role: ' + name)
    if set(roles) != {name for name, _ in named}:
        raise ValueError('Exhaustive unique parameter coverage required')
    phi = [(name, parameter) for name, parameter in named if roles[name] == 'internal_private']
    psi = [(name, parameter) for name, parameter in named if roles[name] == 'boundary_private_own']
    theta = [(name, parameter) for name, parameter in named if roles[name] == 'shared_own']
    if len(phi) != spec['internal_private_tensor_count'] or len(psi) != spec['boundary_private_tensor_count']:
        raise ValueError('Audited private block counts differ')
    if {id(parameter) for _, parameter in phi + psi} != private_ids:
        raise ValueError('Every existing private parameter must have a declared role')
    if len(session.optimizers) != 1 or type(session.optimizers[0]) is not session.torch.optim.Adam:
        raise ValueError('One unchanged native shared-bank Adam required')
    optimized = [parameter for group in session.optimizers[0].param_groups for parameter in group['params']]
    if len(optimized) != len(named) or {id(parameter) for parameter in optimized} != {id(parameter) for _, parameter in named}:
        raise ValueError('Original exhaustive optimizer parameter groups required')
    return {'all': named, 'shared_own': theta, 'boundary_own': psi, 'internal_private': phi, 'roles': roles}
