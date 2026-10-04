"""Round17 SOURCE ONLY: author did not import or execute this module.

Runtime-only integration of the unchanged modern native bodies and the unchanged
round15 initializer. Named/aliased Adam custody replaces positional state mapping.
The private-bias coordinate transport is an explicit prospective amendment whose
dropout-off, full-graph one-step equivalence must pass before scientific use.
"""
from __future__ import annotations
import copy
import math
import random

DISPLAY = {'polyformer_mono': 'PolyFormer-Mono', 'polynormer_r': 'Polynormer-r'}
BOUNDARIES = {'polyformer_mono': {'lin1': 'stem', 'lin3': 'head'},
              'polynormer_r': {'lin_in': 'stem', 'pred_local': 'local_head',
                              'pred_global': 'global_head'}}
TRANSPORT = 'named_alias_adam_private_bias_coordinate_transport_v1'
TOLERANCES = dict(parameter_atol=1e-6, parameter_rtol=1e-5,
                  gradient_atol=1e-6, gradient_rtol=1e-4,
                  logits_atol=1e-6, logits_rtol=1e-5)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cpu_copy(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def rng_snapshot():
    import numpy as np
    import torch
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(),
                state[2], state[3], state[4]), torch_cpu=torch.get_rng_state().cpu(),
                torch_cuda=[item.cpu() for item in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [])


def rng_restore(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    name, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        require(torch.cuda.is_available() and len(state['torch_cuda']) == torch.cuda.device_count(),
                'CUDA RNG inventory changed')
        torch.cuda.set_rng_state_all([item.cpu() for item in state['torch_cuda']])


def alias_inventory(model):
    """Canonical first name plus every alias; no assumed parameter order."""
    groups, by_id = [], {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity not in by_id:
            by_id[identity] = len(groups)
            groups.append(dict(name=name, aliases=[], shape=list(parameter.shape),
                               requires_grad=parameter.requires_grad))
        groups[by_id[identity]]['aliases'].append(name)
    return groups


def named_optimizer_snapshot(model, optimizer):
    import torch
    inventory = alias_inventory(model)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    seen, groups, states = set(), [], {}
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group['params']:
            require(id(parameter) in names and id(parameter) not in seen,
                    'Optimizer must cover each registered parameter once')
            seen.add(id(parameter))
            name = names[id(parameter)]
            group_names.append(name)
            states[name] = cpu_copy(optimizer.state.get(parameter, {}))
            require(set(states[name]) <= {'step', 'exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'},
                    'Only ordinary Adam state is admitted')
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in states[name]:
                    require(states[name][key].shape == parameter.shape,
                            'Moment/parameter shape mismatch')
        groups.append(dict(names=group_names, options=cpu_copy(
            {key: value for key, value in group.items() if key not in ('params', 'param_names')})))
    require(seen == {id(parameter) for parameter in model.parameters()},
            'Optimizer omitted a native or inactive parameter')
    require(isinstance(optimizer, torch.optim.Adam), 'Only pinned coupled Adam is admitted')
    return dict(schema='named-aliased-adam-v1', aliases=inventory, groups=groups, state=states)


def _state_to_parameter(state, parameter, options):
    import torch
    result = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            # Noncapturable/native Adam retains CPU scalar step even on CUDA.
            if key == 'step':
                device = parameter.device if options.get('capturable') or options.get('fused') else 'cpu'
                result[key] = value.detach().clone().to(device)
            else:
                result[key] = value.detach().clone().to(device=parameter.device, dtype=parameter.dtype)
        else:
            result[key] = copy.deepcopy(value)
    return result


def restore_named_optimizer(model, frozen):
    import torch
    require(frozen['schema'] == 'named-aliased-adam-v1' and
            alias_inventory(model) == frozen['aliases'], 'Native alias custody differs')
    parameters = dict(model.named_parameters())
    groups = [dict(params=[parameters[name] for name in group['names']],
                   **copy.deepcopy(group['options'])) for group in frozen['groups']]
    optimizer = torch.optim.Adam(groups)
    for group, saved in zip(optimizer.param_groups, frozen['groups']):
        for name, parameter in zip(saved['names'], group['params']):
            optimizer.state[parameter] = _state_to_parameter(frozen['state'][name], parameter, group)
    return optimizer


def native_to_raw_name(name, backbone):
    require(name.startswith('models.0.'), 'Expected a native single TeacherFamily alias')
    native_name = name[len('models.0.'):]
    for native_boundary, raw_boundary in BOUNDARIES[backbone].items():
        if native_name == native_boundary + '.weight':
            return raw_boundary + '.weight'
        if native_name == native_boundary + '.bias':
            return raw_boundary + '.B'
    return 'core.' + native_name


def raw_arguments(graph, backbone):
    return (graph.teacher_input,) if backbone == 'polyformer_mono' else (
        graph.teacher_input, graph.teacher_edge_index)


def clone_boundary(native_teacher, boundary_api, members):
    """Never reset, consume or mutate the donor; construction RNG is restored later."""
    backbone = native_teacher.specification['backbone']
    require(native_teacher.members == 1 and len(native_teacher.models) == 1,
            'Warm donor must be the native single predictor')
    donor = native_teacher.models[0]
    for boundary in BOUNDARIES[backbone]:
        require(getattr(donor, boundary).bias is not None, 'Admitted native boundary must have bias')
    if members == 4:
        raw = boundary_api.clone_warm_native_boundary(donor, backbone, members=4)
    else:
        require(members == 1, 'Only K1 AD and K4 continuation clones are admitted')
        constructor = (boundary_api.PolyFormerBoundaryFamily if backbone == 'polyformer_mono'
                       else boundary_api.PolynormerBoundaryFamily)
        raw = boundary_api.set_boundary_identity_(constructor(copy.deepcopy(donor), members=1))
        raw.train(donor.training)
    if backbone == 'polynormer_r':
        raw.set_global_stage(native_teacher.global_stage)
    return raw


def transport_optimizer(native_teacher, native_frozen, raw):
    """Mean K CE: B moments m/K,v/K^2; epsilon/K and coupled decay/K.

    Native W/body states, options and steps stay exact. New R/S have empty state
    and inherit their boundary W group options. Local-head state stays retained
    even when the final global branch does not use it. All alias groups are checked.
    """
    import torch
    backbone, k = native_teacher.specification['backbone'], raw.members
    require(k == 4 and alias_inventory(native_teacher) == native_frozen['aliases'],
            'Native optimizer donor/alias inventory changed')
    old_parameters = dict(native_teacher.named_parameters(remove_duplicate=False))
    new_parameters = dict(raw.named_parameters(remove_duplicate=False))
    old_to_new, mapped_ids, mappings = {}, set(), []
    for row in native_frozen['aliases']:
        new_aliases = [native_to_raw_name(name, backbone) for name in row['aliases']]
        require(all(name in new_parameters for name in new_aliases), 'Missing mapped warm parameter')
        values = [new_parameters[name] for name in new_aliases]
        require(all(value is values[0] for value in values), 'Shared native alias was split')
        require(id(values[0]) not in mapped_ids, 'Distinct native parameters unexpectedly merged')
        mapped_ids.add(id(values[0]))
        actual_aliases = [name for name, parameter in new_parameters.items() if parameter is values[0]]
        require(set(actual_aliases) == set(new_aliases), 'Mapped alias inventory differs')
        old_to_new[row['name']] = values[0]
        native_parameter = old_parameters[row['name']]
        is_bias = new_aliases[0].endswith('.B')
        expected = native_parameter.detach().expand(k, -1) if is_bias else native_parameter.detach()
        require(torch.equal(values[0].detach(), expected), 'Warm parameter/bias copy differs')
        mappings.append(dict(native=row['name'], native_aliases=row['aliases'], raw_aliases=new_aliases,
                             bias_replication=is_bias, step_copied=True,
                             first_moment_scale=1/k if is_bias else 1,
                             second_moment_scale=1/(k*k) if is_bias else 1))
    factors = {name: parameter for name, parameter in raw.named_parameters()
               if name.endswith('.R') or name.endswith('.S')}
    require(mapped_ids | {id(parameter) for parameter in factors.values()} ==
            {id(parameter) for parameter in raw.parameters()}, 'Unaccounted raw parameter')
    groups, state_by_parameter, options_by_parameter = [], {}, {}
    factor_assignments = []
    group_custody = []
    for group_index, group in enumerate(native_frozen['groups']):
        options = copy.deepcopy(group['options'])
        require(not options.get('decoupled_weight_decay', False), 'This transport admits coupled Adam only')
        ordinary_parameters, bias_parameters = [], []
        for name in group['names']:
            parameter = old_to_new[name]
            mapped = native_to_raw_name(name, backbone)
            bias = mapped.endswith('.B')
            item_options = copy.deepcopy(options)
            state = cpu_copy(native_frozen['state'][name])
            if bias:
                item_options['eps'] = options['eps']/k
                item_options['weight_decay'] = options['weight_decay']/k
                for key, scale in (('exp_avg', k), ('exp_avg_sq', k*k), ('max_exp_avg_sq', k*k)):
                    if key in state:
                        state[key] = state[key].expand_as(parameter).clone()/scale
            (bias_parameters if bias else ordinary_parameters).append(parameter)
            state_by_parameter[id(parameter)] = state
            options_by_parameter[id(parameter)] = item_options
            if mapped.endswith('.weight') and mapped.split('.')[0] in BOUNDARIES[backbone].values():
                prefix = mapped.rsplit('.', 1)[0]
                for suffix in ('R', 'S'):
                    factor = factors[prefix + '.' + suffix]
                    ordinary_parameters.append(factor)
                    state_by_parameter[id(factor)] = {}
                    options_by_parameter[id(factor)] = copy.deepcopy(options)
                    factor_assignments.append(dict(name=prefix+'.'+suffix, native_weight=name,
                        empty_state=True, options=copy.deepcopy(options)))
        # Preserve each native shared/body group together; split only the B
        # coordinates whose epsilon/coupled-decay options require transport.
        if ordinary_parameters:
            groups.append(dict(params=ordinary_parameters, **copy.deepcopy(options)))
        if bias_parameters:
            bias_options = copy.deepcopy(options)
            bias_options['eps'], bias_options['weight_decay'] = options['eps']/k, options['weight_decay']/k
            groups.append(dict(params=bias_parameters, **bias_options))
        group_custody.append(dict(native_group=group_index, native_names=group['names'],
            shared_and_new_factor_parameters=len(ordinary_parameters), transported_B_parameters=len(bias_parameters),
            native_shared_group_retained=True, only_B_options_split=True))
    require(len(state_by_parameter) == len(list(raw.parameters())), 'Incomplete transported optimizer')
    optimizer = torch.optim.Adam(groups)
    for parameter in raw.parameters():
        optimizer.state[parameter] = _state_to_parameter(state_by_parameter[id(parameter)], parameter,
                                                       options_by_parameter[id(parameter)])
    return optimizer, dict(operation=TRANSPORT, members=k, mappings=mappings,
        factor_assignments=factor_assignments, native_aliases=native_frozen['aliases'],
        native_group_custody=group_custody,
        raw_aliases=alias_inventory(raw), private_bias_eps_scale=1/k,
        private_bias_coupled_decay_scale=1/k, native_steps_preserved=True,
        all_native_inactive_state_retained=True)


def difference(left, right, atol, rtol):
    import torch
    require(left.shape == right.shape, 'Comparison shape changed')
    left, right = left.detach(), right.detach()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return dict(passed=False, finite=False, max_absolute=None, max_relative=None)
    delta = (left.double()-right.double()).abs()
    denominator = torch.maximum(left.double().abs(), right.double().abs()).clamp_min(atol)
    return dict(passed=bool(torch.allclose(left, right, atol=atol, rtol=rtol)), finite=True,
                max_absolute=float(delta.max()) if delta.numel() else 0,
                max_relative=float((delta/denominator).max()) if delta.numel() else 0,
                atol=atol, rtol=rtol)


def identity_logits_audit(native_teacher, k1, k4, graph):
    import torch
    backbone = native_teacher.specification['backbone']
    native_teacher.eval(); k1.eval(); k4.eval()
    args = raw_arguments(graph, backbone)
    with torch.no_grad():
        native, single, ensemble = native_teacher(graph)[0], k1(*args)[0], k4(*args)
    rows = [difference(native, single, TOLERANCES['logits_atol'], TOLERANCES['logits_rtol'])]
    rows.extend(difference(native, member, TOLERANCES['logits_atol'],
                           TOLERANCES['logits_rtol']) for member in ensemble)
    receipt = dict(passed=all(row['passed'] for row in rows), comparisons=rows,
                   native_to_K1_then_each_K4=True, dropout_off=True)
    require(receipt['passed'], 'Warm identity native/K1/K4 logit equivalence failed')
    return receipt


def optimizer_equivalence_audit(native_teacher, native_frozen, boundary_api, graph, train, sink):
    """Disposable copies only; parent runs in future. No useful checkpoint retained.

    Native and K4 start from identical nonzero warm Adam history. Freeze all new
    R/S, evaluate identical dropout-off mean CE, compare scaled private-bias and
    shared gradients, then compare parameters, moments/steps and predictive logits
    after one actual Adam step. Inactive parameters must stay unchanged too.
    """
    import torch
    import torch.nn.functional as F
    native = copy.deepcopy(native_teacher)
    native.eval()
    raw = clone_boundary(native, boundary_api, 4)
    raw.eval()
    native_optimizer = restore_named_optimizer(native, native_frozen)
    raw_optimizer, transport = transport_optimizer(native, native_frozen, raw)
    for name, parameter in raw.named_parameters():
        if name.endswith('.R') or name.endswith('.S'):
            parameter.requires_grad_(False)
    require(any(state.get('exp_avg') is not None and bool(state['exp_avg'].abs().sum() > 0)
                for state in native_frozen['state'].values()), 'Nonzero native Adam history is required')
    backbone = native.specification['backbone']
    native_optimizer.zero_grad(set_to_none=True); raw_optimizer.zero_grad(set_to_none=True)
    zn = native(graph)[0]
    zr = raw(*raw_arguments(graph, backbone))
    F.cross_entropy(zn[train.nodes], train.labels).backward()
    F.cross_entropy(zr[:, train.nodes].reshape(-1, graph.classes), train.labels.repeat(4)).backward()
    native_parameters, raw_parameters = dict(native.named_parameters()), dict(raw.named_parameters())
    gradient_checks = []
    for name, parameter in native_parameters.items():
        mapped = native_to_raw_name(name, backbone)
        other = raw_parameters[mapped]
        require((parameter.grad is None) == (other.grad is None), 'Native/raw active-gradient custody differs')
        if parameter.grad is None:
            check = dict(passed=True, both_inactive=True)
        elif mapped.endswith('.B'):
            check = difference(parameter.grad.expand_as(other), 4*other.grad,
                               TOLERANCES['gradient_atol'], TOLERANCES['gradient_rtol'])
        else:
            check = difference(parameter.grad, other.grad,
                               TOLERANCES['gradient_atol'], TOLERANCES['gradient_rtol'])
        gradient_checks.append(dict(native=name, raw=mapped, **check))
    sink('optimizer_gradient_checks', gradient_checks)
    require(all(row['passed'] for row in gradient_checks), 'Dropout-off native/K4 gradient equivalence failed')
    native_optimizer.step(); raw_optimizer.step()
    parameter_checks, state_checks = [], []
    for name, parameter in native_parameters.items():
        mapped = native_to_raw_name(name, backbone)
        other = raw_parameters[mapped]
        is_bias = mapped.endswith('.B')
        expected = parameter.expand_as(other) if is_bias else parameter
        parameter_checks.append(dict(native=name, raw=mapped, **difference(expected, other,
            TOLERANCES['parameter_atol'], TOLERANCES['parameter_rtol'])))
        ns, rs = native_optimizer.state[parameter], raw_optimizer.state[other]
        require(set(ns) == set(rs), 'Transported moment keys changed after step')
        for key in ns:
            scale = 4 if is_bias and key == 'exp_avg' else 16 if is_bias and key in (
                'exp_avg_sq', 'max_exp_avg_sq') else 1
            expected_state = ns[key].expand_as(rs[key]) if is_bias and key != 'step' else ns[key]
            state_checks.append(dict(native=name, raw=mapped, state=key,
                **difference(expected_state, rs[key]*scale, TOLERANCES['parameter_atol'],
                             TOLERANCES['parameter_rtol'])))
    with torch.no_grad():
        zn = native(graph)[0]
        zr = raw(*raw_arguments(graph, backbone))
    logit_checks = [difference(zn, row, TOLERANCES['logits_atol'], TOLERANCES['logits_rtol']) for row in zr]
    receipt = dict(passed=all(row['passed'] for row in parameter_checks+state_checks+logit_checks),
        transport=transport, gradient_checks=gradient_checks, parameter_checks=parameter_checks,
        state_checks=state_checks, logit_checks=logit_checks, dropout_off=True, factor_updates_frozen=True,
        full_graph=True, training_labels_only=True, scientific_result=False,
        tolerances=TOLERANCES, floating_point_equivalence_not_bitwise_claim=True)
    sink('optimizer_one_step_equivalence', receipt)
    require(receipt['passed'], 'Dropout-off one-step native/K4 Adam equivalence failed')
    return receipt


def train_update(model, optimizer, graph, train, raw=False):
    import torch
    import torch.nn.functional as F
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(*raw_arguments(graph, graph.teacher_backbone)) if raw else model(graph)
    members = model.members
    require(tuple(logits.shape) == (members, graph.teacher_input.shape[0], graph.classes),
            'Predictive full-trajectory shape mismatch')
    loss = F.cross_entropy(logits[:, train.nodes].reshape(-1, graph.classes), train.labels.repeat(members))
    require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
    loss.backward()
    require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()), 'Nonfinite active parameter gradients')
    nonzero = sum(int(parameter.grad is not None and bool((parameter.grad != 0).any()))
                  for parameter in model.parameters())
    require(nonzero > 0, 'No nonzero native/member training gradients')
    optimizer.step()
    require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()),
            'Nonfinite parameters after Adam update')
    return float(loss.detach()), nonzero


def evaluate(model, graph, validation, raw=False):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        logits = model(*raw_arguments(graph, graph.teacher_backbone)) if raw else model(graph)
        require(bool(torch.isfinite(logits).all()), 'Nonfinite evaluation logits')
        nll = float(F.cross_entropy(logits.mean(0)[validation.nodes], validation.labels))
    require(math.isfinite(nll), 'Nonfinite predictor-validation NLL')
    return logits.detach(), nll


def native_checkpoint(model, optimizer, metadata):
    return dict(schema='graph-init-native-warm-checkpoint-v1', model=cpu_copy(model.state_dict()),
        optimizer=named_optimizer_snapshot(model, optimizer), rng=rng_snapshot(),
        specification=copy.deepcopy(model.specification), global_stage=model.global_stage,
        model_training=model.training, metadata=copy.deepcopy(metadata))


def restore_native(adapter, checkpoint, device):
    require(checkpoint['schema'] == 'graph-init-native-warm-checkpoint-v1', 'Wrong warm checkpoint schema')
    model = adapter.TeacherFamily(checkpoint['specification']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.set_global_stage(checkpoint['global_stage'])
    model.train(checkpoint['model_training'])
    optimizer = restore_named_optimizer(model, checkpoint['optimizer'])
    rng_restore(checkpoint['rng'])
    return model, optimizer


def warm_native(adapter, spec, graph, train, validation, trace, save_transition):
    """Exactly 50 native or 200 local + 50 global; fixed last global warm state."""
    model = adapter.TeacherFamily(spec).to(graph.teacher_input.device)
    optimizer = adapter.optimizer_for(model)
    stage_history, local_best, total_updates = [], None, 0
    stages = [('native', 50)] if spec['backbone'] == 'polyformer_mono' else [('local', 200), ('global', 50)]
    for stage, cap in stages:
        if stage == 'global':
            require(local_best is not None, 'No finite best-local native handoff')
            model.load_state_dict(local_best['model'])
            optimizer = restore_named_optimizer(model, local_best['optimizer'])
            # Native loop restores model/Adam only, not RNG: keep after all200 local updates.
            model.set_global_stage(True)
        best_value, best_epoch = float('inf'), None
        for epoch in range(1, cap+1):
            total_updates += 1
            loss, nonzero = train_update(model, optimizer, graph, train)
            _, value = evaluate(model, graph, validation)
            trace(dict(stage=stage, stage_epoch=epoch, actual_update=total_updates,
                       train_ce=loss, validation_nll=value, nonzero_gradient_tensors=nonzero))
            if value < best_value:
                best_value, best_epoch = value, epoch
                if stage == 'local':
                    local_best = native_checkpoint(model, optimizer, dict(stage=stage, stage_epoch=epoch,
                        actual_update=total_updates, primary_validation_nll=value))
        stage_history.append(dict(stage=stage, updates=cap, diagnostic_best_epoch=best_epoch,
                                  diagnostic_best_nll=best_value, warm_selector='fixed last stage update'))
        if stage == 'local':
            save_transition(local_best)
    metadata = dict(actual_updates=total_updates, warm_stage=stage, warm_stage_epoch=cap,
        fixed_last_warm=True, stages=stage_history,
        local_selected_metadata=local_best['metadata'] if local_best is not None else None,
        native_local_rng_restored=False, local_model_and_adam_restored=local_best is not None,
        warm_R_S_fixed_at_identity_by_native_absence=True, label_scope=['train', 'validation'])
    return model, optimizer, native_checkpoint(model, optimizer, metadata)


def continuation(raw, optimizer, graph, train, validation, trace, save_logits, *, terminal_observer=None, terminal_context=None):
    """Fixed1950 native/patience250 or950 global updates; same warm RNG all arms."""
    backbone = graph.teacher_backbone
    cap, patience = (1950, 250) if backbone == 'polyformer_mono' else (950, None)
    midpoint_epoch = 950 if backbone == 'polyformer_mono' else 450
    logits, value = evaluate(raw, graph, validation, raw=True)
    best = dict(state=cpu_copy(raw.state_dict()), optimizer=named_optimizer_snapshot(raw, optimizer),
                validation_nll=value, continuation_epoch=0, actual_update=50 if patience else 250)
    trace(dict(continuation_epoch=0, validation_nll=value, initializer_checkpoint_eligible=True))
    completed, midpoint_saved = 0, False
    for epoch in range(1, cap+1):
        completed = epoch
        loss, nonzero = train_update(raw, optimizer, graph, train, raw=True)
        logits, value = evaluate(raw, graph, validation, raw=True)
        trace(dict(continuation_epoch=epoch, actual_update=epoch+(50 if patience else 250),
            stage='native' if patience else 'global', train_ce=loss, validation_nll=value,
            nonzero_gradient_tensors=nonzero))
        if epoch == midpoint_epoch:
            save_logits('native_midpoint', logits)
            midpoint_saved = True
            trace(dict(event='native_midpoint_saved', continuation_epoch=epoch,
                native_stage_epoch=1000 if patience else 500, actual_update=1000 if patience else 700,
                stage='native' if patience else 'global'))
        if value < best['validation_nll']:
            best = dict(state=cpu_copy(raw.state_dict()), optimizer=named_optimizer_snapshot(raw, optimizer),
                        validation_nll=value, continuation_epoch=epoch,
                        actual_update=epoch+(50 if patience else 250))
        if patience is not None and epoch-best['continuation_epoch'] >= patience:
            break
    if terminal_observer is not None:
        # Reporting sees copies of the existing last evaluation; no added forward.
        # Scalar context reaches the caller even when capture fails before dispatch.
        # No model/Adam object is supplied; the observer return is ignored.
        import time
        capture_started, capture_cpu_started = time.perf_counter(), time.process_time()
        terminal_context = {} if terminal_context is None else terminal_context
        observation_rng, observation_error = None, None
        try:
            terminal_context['hook_stage'] = 'terminal_endpoint_metadata'
            terminal_context['endpoint'] = dict(
                teacher_backbone=backbone, members=raw.members,
                continuation_updates_completed=completed,
                terminal_actual_update=completed+(50 if patience else 250),
                terminal_native_stage_epoch=completed+50,
                stage='native' if patience else 'global',
                update_cap=cap, patience=patience,
                full_native_cap_reached=completed == cap,
                stop_reason='native_update_cap' if completed == cap else 'native_patience',
                source_native_validation_nll=value,
                terminal_before_selected_checkpoint_restore=True,
                selected_continuation_epoch=best['continuation_epoch'],
                selected_actual_update=best['actual_update'])
            terminal_context['hook_stage'] = 'rng_snapshot'
            observation_rng = rng_snapshot()
            terminal_context['hook_stage'] = 'member_logits_copy'
            terminal_member_logits = cpu_copy(logits)
            terminal_context['hook_stage'] = 'served_logits_mean_and_copy'
            terminal_served_logits = cpu_copy(logits.mean(0))
            terminal_context['endpoint'].update(
                capture_wall_seconds_before_observer=time.perf_counter()-capture_started,
                capture_process_cpu_seconds_before_observer=time.process_time()-capture_cpu_started)
            terminal_context['capture_completed'] = True
            terminal_context['hook_stage'] = 'terminal_observer'
            terminal_observer(dict(
                member_logits=terminal_member_logits,
                served_logits=terminal_served_logits,
                metadata=terminal_context['endpoint']))
            terminal_context['observer_completed'] = True
        except BaseException as error:
            observation_error = error
            raise
        finally:
            if not terminal_context.get('capture_completed', False):
                terminal_context['capture_completed'] = False
                terminal_context.setdefault('endpoint', {}).update(
                    capture_wall_seconds_before_observer=time.perf_counter()-capture_started,
                    capture_process_cpu_seconds_before_observer=time.process_time()-capture_cpu_started)
            if observation_rng is not None:
                restore_started, restore_cpu_started = time.perf_counter(), time.process_time()
                try:
                    rng_restore(observation_rng)
                except BaseException as restore_error:
                    terminal_context['rng_restoration_status'] = 'FAILED'
                    terminal_context['rng_restore_error_type'] = type(restore_error).__name__
                    if observation_error is None:
                        raise
                else:
                    terminal_context['rng_restoration_status'] = 'RESTORED'
                finally:
                    terminal_context['rng_restoration_wall_seconds'] = time.perf_counter()-restore_started
                    terminal_context['rng_restoration_process_cpu_seconds'] = time.process_time()-restore_cpu_started
            else:
                terminal_context['rng_restoration_status'] = 'UNAVAILABLE_NO_COMPLETE_SNAPSHOT'
    raw.load_state_dict(best['state'])
    raw.eval()
    selected_logits, replay_nll = evaluate(raw, graph, validation, raw=True)
    require(abs(replay_nll-best['validation_nll']) <= 1e-6, 'Selected checkpoint replay NLL differs')
    save_logits('selected', selected_logits)
    return best, dict(continuation_updates_completed=completed, update_cap=cap, patience=patience,
        selected_continuation_epoch=best['continuation_epoch'], selected_actual_update=best['actual_update'],
        primary_validation_nll=best['validation_nll'], native_midpoint_saved=midpoint_saved,
        native_midpoint_continuation_epoch=midpoint_epoch, native_midpoint_stage_epoch=1000 if patience else 500,
        native_midpoint_actual_update=1000 if patience else 700,
        native_midpoint_absent_reason=None if midpoint_saved else 'native early stopping before frozen absolute native-stage midpoint',
        selection='pooled mean raw-logit validation NLL; earliest strict tie; epoch0 eligible',
        configuration=0, members=4, global_only=backbone == 'polynormer_r')


def storage_receipt(model):
    return dict(unique_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        unique_parameter_bytes=sum(parameter.numel()*parameter.element_size() for parameter in model.parameters()),
        state_dict_tensor_bytes_with_aliases=sum(item.numel()*item.element_size()
                                               for item in model.state_dict().values()),
        aliases=alias_inventory(model), member_full_trajectory_passes=model.members,
        sharing_does_not_reduce_graph_attention_or_message_passing_pass_count=True)
