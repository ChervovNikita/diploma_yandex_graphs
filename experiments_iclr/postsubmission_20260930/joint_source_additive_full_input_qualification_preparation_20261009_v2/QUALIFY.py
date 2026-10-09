"""Root-injected complete-input engineering witness; no numerical imports.

Never run this file directly. RUN.py supplies the reviewed source, unchanged
native runtime, exact roles/inputs, existing Costs, and explicit root release.
"""
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def binding(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def config_for(contracts, release, old, plan, full_binding, condition):
    kind = contracts.SPECS[condition][0]
    seeds = plan['prospective_seeds']
    return contracts.Config(enabled=True, root_source_review_approved=True,
        intent='qualification', root_qualification_execution_approved=True,
        root_training_approved=False,
        source_seal_sha256=release['reviewed_joint_source']['seal']['sha256'],
        native_qualification_binding=old['native_qualification_receipt']['sha256'],
        study_binding=release['plan_sha256'], role_binding=old['role1']['sha256'],
        full_view_binding=full_binding, condition=condition, base_seed=1,
        body_seeds=tuple(seeds['body_seeds'] if kind == 'independent' else seeds['body_seeds'][:1]),
        adapter_seeds=tuple(seeds['adapter_seeds']),
        member_rng_seeds=tuple(seeds['member_rng_seeds'] if kind != 'single' else seeds['member_rng_seeds'][:1]))


def complete_context(rt, modules, joint, contracts, inputs, role_path, costs, release, old, plan):
    """Same native seams and binding object as joint.prepare_context, once."""
    engine, loader, producer = (modules[key] for key in ('engine', 'role_loader', 'source_views'))
    role = read(role_path)
    loader.source_gate()
    with costs.measure('qualifier_exact_complete_role_loader_native_setup_and_raw_views', gpu=True):
        data = loader.load(inputs, role_path, read(Path(loader.__file__).parent / 'SOURCE_EXPECTATIONS.json'))
        require(data.input_bindings == old['expected_input_files'] == role['input_files'], 'Exact complete public input bindings')
        static = engine.prepare_static(rt, data, costs)
        ctx = engine.prepare_seed(rt, static, role, role['seed'], costs)
        full = modules['metadata'].digest(dict(study=release['plan_sha256'], role_binding=old['role1']['sha256'],
            seed=ctx.seed, inputs=data.input_bindings, feature_shapes=static.feature_shapes,
            label_shapes={key: list(value.shape) for key, value in ctx.label_feats.items()}))
        config = config_for(contracts, release, old, plan, full, 'ADD_PS')
        config.require_enabled()
        sources = joint.source_gate()
        producer.source_gate()
        views = producer.build_family_views(rt, static, ctx, costs, producer.ViewConfig(enabled=True,
            root_source_review_approved=True, native_qualification_binding=config.native_qualification_binding,
            full_view_binding=full, role_binding=config.role_binding,
            source_seal_sha256=sources['source_view_seal_sha256']))
    require(len(ctx.feats) == 25 and len(ctx.label_feats) == 12 and set(views) == set(contracts.FAMILIES), 'Every native and raw-family channel')
    return ctx, views, full


def constructor_report(session):
    torch = session.rt['torch']
    session.verify_buffers()
    session.verify_optimizers()
    private_ids = {id(value) for block in session.ownership.private for value in block}
    names, seen = [], set()
    for member, model in enumerate(session.models):
        for name, value in model.named_parameters():
            require(torch.isfinite(value).all().item(), 'Finite actual initialized parameters')
            if id(value) in private_ids:
                require(name.endswith(('.v', '.u')), 'Only additive incoming/outgoing private factors')
                if name.endswith('.u'):
                    require(torch.count_nonzero(value).item() == 0, 'Additive outgoing factors start exactly zero')
                else:
                    norms = value.float().norm(dim=-2)
                    require(torch.isfinite(norms).all().item() and (norms > 0).all().item(), 'Finite nonzero seeded incoming dictionary columns')
                names.append([member, name, list(value.shape)])
            seen.add(id(value))
    require(seen == {id(value) for value in session.ownership.all_parameters}, 'No omitted actual parameter owner')
    expected = {'ADD_PS': 84049676, 'independent_ADD_PS': 335028272,
        'rank4_single_P': 84049676, 'native_single_P': 83659532}[session.config.condition]
    require(session.counts['actual_owned_total_scalars'] == expected, 'Exact complete native scalar interface')
    return dict(counts=session.counts, finite_initial_parameters=True, additive_u_zero=True,
        dictionary_columns_finite_nonzero=True, private_factor_shapes=names,
        native_BN_private_disjoint=True, actual_Adam_owners=len(session.optimizers))


def bn_state(session, member, site):
    torch = session.rt['torch']
    module = dict(session.models[member].named_modules())[site]
    return session.engine.cpu_tree(torch, {name: getattr(module, name)
        for name in ('running_mean', 'running_var', 'num_batches_tracked')})


@contextmanager
def epoch_observer(session, joint, row):
    """Delegates each original operation once, with read-only native hooks.

    This is sequential instrumentation of one existing train_epoch invocation.
    No alternate objective, gradients, references, optimizer or step is created.
    """
    torch, engine = session.rt['torch'], session.engine
    original_forward, original_grad = session.forward, torch.autograd.grad
    original_steps = [optimizer.step for optimizer in session.optimizers]
    hooks, references, active_call, last_replay = [], {}, {}, {}
    private_member = {id(value): member for member, block in enumerate(session.ownership.private) for value in block}
    outgoing_ids = {id(value) for model in session.models for name, value in model.named_parameters() if name.endswith('.u')}
    owner_of = {id(value): owner for owner, optimizer in enumerate(session.optimizers)
        for group in optimizer.param_groups for value in group['params']}
    stage = {'kind': 'P', 'recipient': None}
    row.update(observed_references=0, observed_replays=0, observed_S_private_VJPs=0,
        observed_Adam_owner_steps=0, attempted_S_private_VJPs=0, attempted_Adam_owner_steps=0,
        stopped_eval_references=0, finite_P_gradients=True,
        finite_S_gradients=True, S_native_or_peer_hook_calls=0, BN_working_passes=0,
        BN_reference_persistence=True, BN_replay_prestate=True, BN_replay_no_persistent_update=True,
        caller_RNG_preserved_by_every_forward=True, S_VJP_rows=[], Adam_step_rows=[])

    def pre_hook(member, site):
        def hook(module, inputs):
            require(active_call.get('member') == member, 'BN hook belongs to current body')
            require(module.training == (active_call['mode'] == 'train'), 'Native BN mode follows active reference/replay or terminal eval')
            before = bn_state(session, member, site)
            expected = {name: active_call['token'].buffers_before[site + '.' + name] for name in before}
            require(engine.exact(torch, before, expected), 'Actual native BN enters exact token pre-buffer values')
            active_call['BN_pre'][site] = before
        return hook

    def post_hook(member, site):
        def hook(module, inputs, output):
            before, after = active_call['BN_pre'][site], bn_state(session, member, site)
            require(torch.isfinite(after['running_mean']).all().item() and torch.isfinite(after['running_var']).all().item(), 'Finite actual native BN working state')
            if active_call['mode'] == 'train':
                require(after['num_batches_tracked'].item() == before['num_batches_tracked'].item() + 1, 'Exactly one native working BN advance')
            else:
                require(engine.exact(torch, before, after), 'Terminal peer eval keeps native BN exact')
            active_call['BN_post'][site] = after
            row['BN_working_passes'] += 1
        return hook

    def forward(token, ids=None, **kwargs):
        require(not active_call, 'Only one complete native replay graph forward at a time')
        replay = torch.is_grad_enabled()
        persist = kwargs.get('persist_buffers', False)
        require(token.role == 'TRAIN' and token.mode == ('train' if session.active[token.member] else 'eval'), 'Only complete actual TRAIN reference/replay paths')
        require(persist == (session.active[token.member] and not replay), 'Only active no-grad references persist BN')
        if replay:
            require(session.active[token.member], 'Stopped member has no gradient replay')
        key = (token.member, token.source)
        before = session.buffer_state(token.member)
        caller = engine.capture_rng(session.rt['numpy'], torch)
        canonical_objects = tuple((name, id(value), value.untyped_storage().data_ptr()) for name, value in session.models[token.member].named_buffers())
        if replay:
            saved = references[key]
            require((token.member, token.source, token.mode, token.role) == saved['identity']
                and engine.exact(torch, token.rng, saved['rng'])
                and engine.exact(torch, token.buffers_before, saved['before'])
                and engine.exact(torch, token.buffers_after, saved['after']), 'Replay uses exact reference pre-BN/RNG token and owned post metadata')
        active_call.update(member=token.member, source=token.source, mode=token.mode,
            token=token, BN_pre={}, BN_post={})
        try:
            result = original_forward(token, ids, **kwargs)
            # Paid native calls remain counted if any later witness fails.
            row['observed_replays' if replay else 'observed_references'] += 1
            if not replay:
                row['stopped_eval_references'] += int(not session.active[token.member])
            require(set(active_call['BN_pre']) == set(joint.BN_SITES) == set(active_call['BN_post']), 'Every actual native BN site observed')
            after = session.buffer_state(token.member)
            working_after = {site + '.' + name: value for site, values in active_call['BN_post'].items() for name, value in values.items()}
            require(engine.exact(torch, after, working_after if persist else before), 'Actual canonical buffers persist exactly once or remain exact')
            if not replay:
                require(key not in references, 'Exactly one reference per member/view')
                references[key] = dict(identity=(token.member, token.source, token.mode, token.role),
                    rng=engine.cpu_tree(torch, token.rng), before=engine.cpu_tree(torch, token.buffers_before),
                    after=engine.cpu_tree(torch, after))
            else:
                last_replay.update(member=token.member, source=token.source)
            require(engine.exact(torch, caller, engine.capture_rng(session.rt['numpy'], torch)), 'Every native forward restores caller RNG')
            require(canonical_objects == tuple((name, id(value), value.untyped_storage().data_ptr()) for name, value in session.models[token.member].named_buffers()), 'Scratch BN preserves original canonical buffer objects/storage')
            return result
        finally:
            active_call.clear()

    def parameter_hook(parameter):
        def hook(gradient):
            require(torch.isfinite(gradient).all().item(), 'Finite actual backward contribution')
            if stage['kind'] == 'S':
                correct = private_member.get(id(parameter)) == stage['recipient']
                row['S_native_or_peer_hook_calls'] += int(not correct)
                require(correct, 'S reaches only assigned recipient private inputs')
            return None
        return hook

    def autograd_grad(*args, **kwargs):
        requested = tuple(args[1] if len(args) > 1 else kwargs['inputs'])
        match = [member for member, block in enumerate(session.ownership.private)
            if tuple(id(value) for value in requested) == tuple(id(value) for value in block)]
        require(len(match) == 1 and match[0] < 3 and session.active[match[0]], 'S VJP exact active assigned recipient block')
        require(kwargs.get('retain_graph') is True and kwargs.get('allow_unused') is True, 'Original source private VJP policy')
        recipient = match[0]
        require(last_replay == dict(member=recipient, source=None), 'S VJP is attached to actual factual recipient replay')
        stamp = session.ownership.grad_stamp()
        row['attempted_S_private_VJPs'] += 1
        observed = dict(recipient=recipient, requested_private_tensors=len(requested),
            operation_completed=False, post_return_witness_passed=False)
        row['S_VJP_rows'].append(observed)
        stage.update(kind='S', recipient=recipient)
        try:
            gradients = original_grad(*args, **kwargs)
            row['observed_S_private_VJPs'] += 1
            observed['operation_completed'] = True
        finally:
            stage.update(kind='P', recipient=None)
        require(session.ownership.grad_stamp() == stamp, 'S VJP itself mutates no owner accumulated gradients')
        require(len(gradients) == len(requested) and all(value is None or torch.isfinite(value).all().item() for value in gradients), 'Finite complete returned recipient S gradients')
        nonzero_outgoing = sum(int(torch.count_nonzero(value).item() > 0) for parameter, value in zip(requested, gradients) if value is not None and id(parameter) in outgoing_ids)
        require(nonzero_outgoing > 0, 'Actual assigned S has an eligible nonzero outgoing gradient; structural null sites permitted')
        observed.update(
            nonzero_outgoing_tensors=nonzero_outgoing, unused_tensors=sum(value is None for value in gradients),
            finite=True, native_and_peer_excluded=True, post_return_witness_passed=True)
        return gradients

    def step(owner):
        def wrapper(*args, **kwargs):
            require(session.kind != 'independent' or session.active[owner], 'Stopped owner has no Adam invocation')
            require(row['observed_references'] == 16
                and row['observed_replays'] == 4 * sum(session.active)
                and row['observed_S_private_VJPs'] == sum(session.active[:3]),
                'All complete old-state references and P/S replays precede every Adam step')
            parameters = tuple(value for group in session.optimizers[owner].param_groups for value in group['params'])
            require(all(owner_of[id(value)] == owner for value in parameters), 'One actual owner of every parameter')
            require(all(value.grad is None or torch.isfinite(value.grad).all().item() for value in parameters), 'All actual pre-step gradients finite')
            native = [value for value in parameters if id(value) not in private_member and value.grad is not None and torch.count_nonzero(value.grad).item() > 0]
            outgoing = [value for value in parameters if id(value) in outgoing_ids and value.grad is not None and torch.count_nonzero(value.grad).item() > 0]
            require(native and outgoing, 'Every actual active Adam owner receives meaningful native P and outgoing gradients')
            # One witness tensor of each kind, owned CPU copies before the real step.
            witnesses = (max(native, key=lambda value: float(value.grad.detach().abs().max().item())),
                max(outgoing, key=lambda value: float(value.grad.detach().abs().max().item())))
            before = [engine.cpu_tree(torch, value) for value in witnesses]
            row['attempted_Adam_owner_steps'] += 1
            observed = dict(owner=owner, operation_completed=False, post_return_witness_passed=False)
            row['Adam_step_rows'].append(observed)
            result = original_steps[owner](*args, **kwargs)
            row['observed_Adam_owner_steps'] += 1
            observed['operation_completed'] = True
            require(all(not torch.equal(old, value.detach().cpu()) for old, value in zip(before, witnesses)), 'Actual native and outgoing Adam tensor changes')
            observed.update(finite_actual_gradients=True,
                nonzero_native_gradient_tensors=len(native), nonzero_outgoing_gradient_tensors=len(outgoing),
                native_and_outgoing_actual_parameter_changes=True, post_return_witness_passed=True)
            return result
        return wrapper

    try:
        for member, model in enumerate(session.models):
            named = dict(model.named_modules())
            for site in joint.BN_SITES:
                hooks.append(named[site].register_forward_pre_hook(pre_hook(member, site)))
                hooks.append(named[site].register_forward_hook(post_hook(member, site)))
        for parameter in session.ownership.all_parameters:
            hooks.append(parameter.register_hook(parameter_hook(parameter)))
        session.forward = forward
        torch.autograd.grad = autograd_grad
        for owner, optimizer in enumerate(session.optimizers):
            optimizer.step = step(owner)
        yield
    finally:
        session.forward = original_forward
        torch.autograd.grad = original_grad
        for optimizer, original in zip(session.optimizers, original_steps):
            optimizer.step = original
        for handle in hooks:
            handle.remove()


def reject_invalid_roles(session, costs):
    torch, engine = session.rt['torch'], session.engine
    wrong = session.rows['TRAIN'].clone()
    wrong[0] = session.ctx.valid_index[0].cpu()
    duplicate = session.rows['TRAIN'].clone()
    require(duplicate.numel() > 1, 'Meaningful complete role duplicate guard')
    duplicate[0] = duplicate[1]
    token = session.token(0, None, 'train')
    reports = []
    calls = []
    handle = session.models[0].register_forward_pre_hook(lambda model, inputs: calls.append(True))
    try:
        with costs.measure('actual_wrong_role_and_duplicate_ID_guards', gpu=True):
            for kind, ids in (('same_length_TRAIN_with_VALID_ID', wrong), ('duplicate_TRAIN_ID', duplicate)):
                before = session.buffer_state(0)
                caller = engine.capture_rng(session.rt['numpy'], torch)
                stamp = session.ownership.stamp(), session.ownership.grad_stamp()
                rejected = False
                try:
                    with torch.no_grad():
                        session.forward(token, ids, persist_buffers=True)
                except (RuntimeError, ValueError):
                    rejected = True
                require(rejected and not calls and engine.exact(torch, before, session.buffer_state(0))
                    and engine.exact(torch, caller, engine.capture_rng(session.rt['numpy'], torch))
                    and stamp == (session.ownership.stamp(), session.ownership.grad_stamp()), 'Bad IDs rejected before native forward without state mutation')
                reports.append(dict(case=kind, rejected_before_native_forward=True, state_unchanged=True))
    finally:
        handle.remove()
    return reports


def one_epoch(session, joint, costs, row, expected_steps, expected_replays, expected_S):
    before = dict(session.counters)
    row['complete'] = False
    try:
        with costs.measure('qualifier_actual_complete_TRAIN_epoch_with_read_only_observers', gpu=True):
            with epoch_observer(session, joint, row):
                observation = session.train_epoch()
            # Source training BCE values are discarded.
            row['replay_max_abs_logit_drift'] = observation['replay_max_abs_logit_drift']
            row['replay_drift_is_acceptance_gate'] = False
            del observation
    finally:
        row['source_counter_delta'] = {key: session.counters[key] - before[key] for key in session.counters}
    delta = row['source_counter_delta']
    require(delta['epochs_completed'] == 1 and delta['reference_forwards'] == 16
        and delta['gradient_replays'] == delta['P_backwards'] == expected_replays
        and delta['S_private_VJPs'] == expected_S and delta['actual_Adam_steps'] == expected_steps,
        'Exact complete source counters for actual bounded epoch')
    require(row['observed_references'] == 16 and row['observed_replays'] == expected_replays
        and row['observed_S_private_VJPs'] == expected_S and row['observed_Adam_owner_steps'] == expected_steps
        and row['BN_working_passes'] == 3 * (16 + expected_replays), 'Independent observations agree with source complete counters')
    row.update(source_counter_delta=delta, complete=True, finite_actual_gradients=True,
        active_members=list(session.active), complete_TRAIN_rows=session.ctx.train_count)


def factual_outputs(session, costs):
    """Factual full KNOWN logits only; no targets, metrics or evaluation API."""
    torch, engine = session.rt['torch'], session.engine
    outputs = []
    with costs.measure('label_free_complete_factual_KNOWN_serving', gpu=True), torch.no_grad():
        for member in range(session.count):
            token = session.token(member, None, 'eval', 'KNOWN')
            before = session.buffer_state(member)
            output, after_rng = session.forward(token)
            require(output.shape == (session.ctx.train_count + session.ctx.valid_count, 5)
                and torch.isfinite(output).all().item(), 'Finite complete factual logits')
            require(engine.exact(torch, token.rng, after_rng)
                and engine.exact(torch, before, session.buffer_state(member)), 'Serving keeps owned BN/RNG state exact')
            outputs.append(output.detach().cpu().clone())
            del output
        pooled = torch.stack([value.sigmoid() for value in outputs]).mean(dim=0)
        require(torch.isfinite(pooled).all().item() and ((pooled >= 0) & (pooled <= 1)).all().item(), 'Finite factual Bernoulli probability pool')
    return dict(member_logits=outputs, probabilities=pooled)


def checkpoint_fresh_restore(session, joint, contracts, rt, modules, ctx, views, costs,
        folder, release, old, plan, full_binding, epoch, row):
    torch, engine = rt['torch'], modules['engine']
    condition = session.config.condition
    identity = dict(purpose='fixed_discarded_engineering_qualification', condition=condition,
        fixed_epoch=epoch, role_binding=old['role1']['sha256'], study_binding=release['plan_sha256'])
    fresh = None
    with costs.measure('qualifier_fixed_snapshot_and_server_only_write', gpu=True):
        state = session.snapshot(epoch, identity)
        path = folder / (condition + '_FIXED_ENGINEERING_STATE.pt')
        torch.save(state, path)
        original = factual_outputs(session, costs)
        original_path = folder / (condition + '_FIXED_FACTUAL_LOGITS.pt')
        torch.save(modules['metadata'].checkpoint_tree(torch, original), original_path)
        row['fixed_checkpoint'] = binding(path)
        row['server_only_original_logits'] = binding(original_path)
    with costs.measure('qualifier_release_live_training_objects_before_fresh_constructor', gpu=True):
        session.release_models()
    try:
        with costs.measure('qualifier_weights_only_fresh_reconstruction_restore', gpu=True):
            del state
            saved = torch.load(path, map_location='cpu', weights_only=True)
            fresh = getattr(joint, condition)(rt, modules, ctx, views, costs,
                config_for(contracts, release, old, plan, full_binding, condition))
            fresh.restore(saved, identity)
            require(engine.exact(torch, engine.cpu_tree(torch, fresh.group.state_dict()), saved['models_state'])
                and engine.exact(torch, engine.cpu_tree(torch, [value.state_dict() for value in fresh.optimizers]), saved['optimizer_states'])
                and engine.exact(torch, fresh.member_rng, saved['rng']), 'Fresh constructor restores actual models/native BN/Adam/owned RNG exactly')
            del saved
            restored = factual_outputs(fresh, costs)
            restored_path = folder / (condition + '_FRESH_FACTUAL_LOGITS.pt')
            torch.save(modules['metadata'].checkpoint_tree(torch, restored), restored_path)
            row.update(server_only_restored_logits=binding(restored_path), weights_only_load=True,
                fresh_model_native_BN_Adam_RNG_restore_exact=True,
                label_free_complete_factual_serving_finite=True,
                serving_max_abs_logit_drift=max(float((left - right).abs().max().item()) for left, right in zip(original['member_logits'], restored['member_logits'])),
                serving_max_abs_probability_drift=float((original['probabilities'] - restored['probabilities']).abs().max().item()),
                serving_drift_is_acceptance_gate=False, training_resumption_qualified=False)
    finally:
        if fresh is not None:
            fresh.release_models()
    del original


def save_fixed_member(session, member, epoch, folder, costs, row):
    """Production member snapshot interface, with a fixed engineering epoch."""
    identity = dict(purpose='fixed_discarded_engineering_own_member_qualification',
        condition=session.config.condition, fixed_epoch=epoch, member=member,
        role_binding=session.config.role_binding, study_binding=session.config.study_binding)
    item = dict(member=member, fixed_epoch=epoch, identity=identity, complete=False)
    row.setdefault('fixed_member_checkpoints', []).append(item)
    with costs.measure('qualifier_fixed_independent_member_snapshot_and_server_only_write', gpu=True):
        state = session.snapshot(epoch, identity, member=member)
        path = folder / ('independent_ADD_PS_MEMBER' + str(member) + '_FIXED_ENGINEERING_STATE.pt')
        session.rt['torch'].save(state, path)
        item.update(checkpoint=binding(path), complete=True)
        del state


def member_checkpoint_fresh_restore(session, joint, contracts, rt, modules, ctx, views, costs,
        folder, release, old, plan, full_binding, row):
    """Restore the four fixed own-member slots into one fresh independent4."""
    torch, engine = rt['torch'], modules['engine']
    slots = row['fixed_member_checkpoints']
    require([slot['member'] for slot in slots] == [0, 1, 2, 3]
        and [slot['fixed_epoch'] for slot in slots] == [0, 1, 1, 1]
        and all(slot['complete'] for slot in slots), 'Fixed member0 terminal and later active-peer checkpoint slots')
    original = factual_outputs(session, costs)
    original_path = folder / 'independent_ADD_PS_FIXED_FACTUAL_LOGITS.pt'
    with costs.measure('qualifier_independent_fixed_factual_logits_server_only_write', gpu=True):
        torch.save(modules['metadata'].checkpoint_tree(torch, original), original_path)
        row['server_only_original_logits'] = binding(original_path)
    with costs.measure('qualifier_release_independent_live_models_before_fresh_constructor', gpu=True):
        session.release_models()
    fresh = None
    row['member_restore_rows'] = []
    try:
        fresh = joint.independent_ADD_PS(rt, modules, ctx, views, costs,
            config_for(contracts, release, old, plan, full_binding, 'independent_ADD_PS'))
        for slot in slots:
            member = slot['member']
            observed = dict(member=member, fixed_epoch=slot['fixed_epoch'], complete=False)
            row['member_restore_rows'].append(observed)
            with costs.measure('qualifier_weights_only_production_independent_own_member_restore', gpu=True):
                saved = torch.load(Path(slot['checkpoint']['path']), map_location='cpu', weights_only=True)
                fresh.restore(saved, slot['identity'], member=member)
                require(engine.exact(torch, engine.cpu_tree(torch, fresh.models[member].state_dict()), saved['model_state'])
                    and engine.exact(torch, engine.cpu_tree(torch, fresh.optimizers[member].state_dict()), saved['optimizer_state'])
                    and engine.exact(torch, fresh.member_rng[member], saved['rng']), 'Actual own-member model/native BN/Adam/RNG restoration exact')
                observed.update(complete=True, weights_only_load=True, own_member_model_BN_Adam_RNG_exact=True)
                del saved
        # Recheck all slots after all four restores; later restores must leave
        # each previously restored disjoint body's state intact.
        with costs.measure('qualifier_all_restored_independent_slots_final_exact_custody', gpu=True):
            for slot in slots:
                member = slot['member']
                saved = torch.load(Path(slot['checkpoint']['path']), map_location='cpu', weights_only=True)
                require(engine.exact(torch, engine.cpu_tree(torch, fresh.models[member].state_dict()), saved['model_state'])
                    and engine.exact(torch, engine.cpu_tree(torch, fresh.optimizers[member].state_dict()), saved['optimizer_state'])
                    and engine.exact(torch, fresh.member_rng[member], saved['rng']), 'Every own-slot remains exact after all four restore operations')
                del saved
        restored = factual_outputs(fresh, costs)
        restored_path = folder / 'independent_ADD_PS_FRESH_FACTUAL_LOGITS.pt'
        with costs.measure('qualifier_independent_restored_factual_logits_server_only_write', gpu=True):
            torch.save(modules['metadata'].checkpoint_tree(torch, restored), restored_path)
            row.update(server_only_restored_logits=binding(restored_path), weights_only_load=True,
                production_member_snapshot_restore_interface_qualified=True,
                fresh_model_native_BN_Adam_RNG_restore_exact=True,
                label_free_complete_factual_serving_finite=True,
                serving_max_abs_logit_drift=max(float((left - right).abs().max().item()) for left, right in zip(original['member_logits'], restored['member_logits'])),
                serving_max_abs_probability_drift=float((original['probabilities'] - restored['probabilities']).abs().max().item()),
                serving_drift_is_acceptance_gate=False, training_resumption_qualified=False,
                scientific_own_selector_executed=False)
    finally:
        if fresh is not None:
            fresh.release_models()
    del original


def retained_totals(report):
    """Counts derive from retained rows even if post-return witnesses failed."""
    epochs = [epoch for row in report['conditions'] for epoch in row['epochs']]
    for key, observed in (('actual_Adam_owner_steps', 'observed_Adam_owner_steps'),
            ('observed_references', 'observed_references'), ('observed_gradient_replays', 'observed_replays'),
            ('actual_S_private_VJPs', 'observed_S_private_VJPs'),
            ('attempted_Adam_owner_steps', 'attempted_Adam_owner_steps'),
            ('attempted_S_private_VJPs', 'attempted_S_private_VJPs')):
        report[key] = sum(row.get(observed, 0) for row in epochs)
    report['actual_P_backwards'] = sum(row.get('source_counter_delta', {}).get('P_backwards', 0) for row in epochs)
    report['completed_operations_counted_at_delegated_return_before_post_witnesses'] = True


def final_observation(report, key, function, errors):
    """Unknown resource values and secondary failures never erase receipts."""
    try:
        report[key] = function()
    except BaseException as error:
        report[key] = None
        errors.append(dict(observation=key, type=type(error).__name__, message=str(error)))


def qualify(rt, modules, joint, contracts, inputs, role_path, folder, costs, release, old, adoption):
    """One invocation: shared one epoch; independent two; scalar constructors."""
    require(release['enabled'] is True and release['root_qualification_execution_approved'] is True
        and adoption['root_source_review_approved'] is True and adoption['qualification_only'] is True
        and not adoption['scientific_training_adopted'], 'Root gated qualification only')
    require(not folder.exists(), 'Fresh root-owned qualification output')
    folder.mkdir(parents=True, exist_ok=False)
    torch, engine = rt['torch'], modules['engine']
    plan = read(HERE / 'PLAN.json')
    require(sha(HERE / 'PLAN.json') == release['plan_sha256'] and plan['enabled'] is False, 'Exact immutable disabled bounded plan')
    started, usage = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    caller = engine.capture_rng(rt['numpy'], torch)
    session, ctx, views = None, None, None
    report = dict(schema='joint-additive-full-native-engineering-qualification-v2', status='started',
        complete=False, qualification_passed=False, source_seal_sha256=release['reviewed_joint_source']['seal']['sha256'],
        root_adoption=release['reviewed_joint_source']['root_adoption'], plan_sha256=release['plan_sha256'],
        native_qualification_binding=old['native_qualification_receipt'], role_binding=old['role1'],
        input_files=old['expected_input_files'], quality_scoring=False, VALID_checkpoint_selection=False,
        scientific_training=False, full_fits=False, automatic_retry=False, TEST_file_access=False,
        raw_states_and_logits_server_only=True, actual_Adam_owner_steps=0, conditions=[], constructor_only=[],
        numerical_drift_is_acceptance_gate=False, no_all_site_nonzero_gradient_requirement=True,
        startup_and_runtime_costs_in_root_receipt=True, original_sources_unchanged=True)
    write(folder / 'QUALIFICATION_REPORT.json', report)
    try:
        ctx, views, full = complete_context(rt, modules, joint, contracts, inputs, role_path, costs, release, old, plan)
        report.update(full_view_binding=full, complete_TRAIN_rows=ctx.train_count,
            factual_KNOWN_rows=ctx.train_count + ctx.valid_count, feature_channels=25, label_channels=12,
            raw_family_views=list(contracts.FAMILIES))
        for condition in ('ADD_PS', 'independent_ADD_PS'):
            row = dict(condition=condition, complete=False, epochs=[])
            report['conditions'].append(row)
            session = getattr(joint, condition)(rt, modules, ctx, views, costs,
                config_for(contracts, release, old, plan, full, condition))
            with costs.measure('qualifier_actual_constructor_interface_and_role_guards', gpu=True):
                row['constructor'] = constructor_report(session)
                row['invalid_role_guards'] = reject_invalid_roles(session, costs)
            first = dict(fixed_epoch=0)
            row['epochs'].append(first)
            one_epoch(session, joint, costs, first, 1 if condition == 'ADD_PS' else 4, 16, 3)
            retained_totals(report)
            if condition == 'independent_ADD_PS':
                save_fixed_member(session, 0, 0, folder, costs, row)
                # Exact existing fitter stop transition: terminal body then
                # zero_grad(set_to_none=True), before its peer-only epoch.
                session.active[0] = False
                session.optimizers[0].zero_grad(set_to_none=True)
                with costs.measure('qualifier_terminal_peer_owned_state_copy', gpu=True):
                    terminal = dict(model=engine.cpu_tree(torch, session.models[0].state_dict()),
                        optimizer=engine.cpu_tree(torch, session.optimizers[0].state_dict()),
                        rng=engine.cpu_tree(torch, session.member_rng[0]))
                stopped = dict(fixed_epoch=1, stopped_peer=0)
                row['epochs'].append(stopped)
                one_epoch(session, joint, costs, stopped, 3, 12, 2)
                retained_totals(report)
                with costs.measure('qualifier_terminal_peer_exact_invariance', gpu=True):
                    require(stopped['stopped_eval_references'] == 4
                        and engine.exact(torch, terminal['model'], engine.cpu_tree(torch, session.models[0].state_dict()))
                        and engine.exact(torch, terminal['optimizer'], engine.cpu_tree(torch, session.optimizers[0].state_dict()))
                        and engine.exact(torch, terminal['rng'], session.member_rng[0])
                        and all(value.grad is None for value in session.models[0].parameters()), 'Stopped terminal peer weights/native BN/Adam/RNG exact and no gradients')
                    stopped.update(terminal_weights_BN_Adam_RNG_exact=True, no_stopped_gradients=True,
                        no_stopped_optimizer_step=True, active_assigned_recipients=[1, 2])
                    del terminal
                for member in (1, 2, 3):
                    save_fixed_member(session, member, 1, folder, costs, row)
                member_checkpoint_fresh_restore(session, joint, contracts, rt, modules, ctx, views, costs,
                    folder, release, old, plan, full, row)
            else:
                checkpoint_fresh_restore(session, joint, contracts, rt, modules, ctx, views, costs,
                    folder, release, old, plan, full, len(row['epochs']) - 1, row)
            session = None
            row['complete'] = True
            write(folder / 'QUALIFICATION_REPORT.json', report)
        for condition in plan['constructor_only']:
            row = dict(condition=condition, actual_Adam_owner_steps=0, complete=False,
                constructor_only=True, numerical_training_qualified=False, factual_serving_qualified=False)
            report['constructor_only'].append(row)
            session = getattr(joint, condition)(rt, modules, ctx, views, costs,
                config_for(contracts, release, old, plan, full, condition))
            with costs.measure('qualifier_complete_rank4_or_native_scalar_constructor_only', gpu=True):
                row['constructor'] = constructor_report(session)
                session.release_models()
            session = None
            row['complete'] = True
        epochs = [epoch for row in report['conditions'] for epoch in row['epochs']]
        require(report['actual_Adam_owner_steps'] == sum(value['observed_Adam_owner_steps'] for value in epochs) == 8
            and sum(value['observed_references'] for value in epochs) == 48
            and sum(value['observed_replays'] for value in epochs) == 44
            and sum(value['observed_S_private_VJPs'] for value in epochs) == 8,
            'Complete fixed plan observed without extra training')
        report.update(status='complete', complete=True, qualification_passed=True,
            observed_references=48, observed_gradient_replays=44, actual_P_backwards=44, actual_S_private_VJPs=8)
    except BaseException as error:
        report.update(status='failed', complete=False, qualification_passed=False,
            failure=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        retained_totals(report)
        cleanup_errors = []
        try:
            with costs.measure('qualifier_final_release_and_caller_RNG_restore', gpu=True):
                if session is not None:
                    session.release_models()
                engine.restore_rng(rt['numpy'], torch, caller)
                require(engine.exact(torch, caller, engine.capture_rng(rt['numpy'], torch)), 'Invocation restores original caller RNG')
                ctx, views = None, None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(rt['device'])
        except BaseException as error:
            cleanup_errors.append(dict(type=type(error).__name__, message=str(error)))
        observation_errors = []
        final_observation(report, 'seconds', lambda: time.perf_counter() - started, observation_errors)
        final_observation(report, 'CPU_user_seconds', lambda: resource.getrusage(resource.RUSAGE_SELF).ru_utime - usage.ru_utime, observation_errors)
        final_observation(report, 'CPU_system_seconds', lambda: resource.getrusage(resource.RUSAGE_SELF).ru_stime - usage.ru_stime, observation_errors)
        final_observation(report, 'cumulative_RSS_peak_bytes', lambda: int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)), observation_errors)
        final_observation(report, 'peak_cuda_allocated_bytes', lambda: int(torch.cuda.max_memory_allocated(rt['device'])), observation_errors)
        final_observation(report, 'peak_cuda_reserved_bytes', lambda: int(torch.cuda.max_memory_reserved(rt['device'])), observation_errors)
        report.update(RSS_and_CUDA_peaks_are_process_lifetime_highwater=True, memory_peaks_reset=False,
            inclusive_setup_constructors_state_IO_restore_serving_cleanup=True,
            instrumentation_and_owned_state_copy_costs_included=True,
            costs=dict(path=str(Path(costs.folder) / 'COST_EVENTS.jsonl'), events=len(costs.rows)))
        if cleanup_errors:
            report.update(status='failed', complete=False, qualification_passed=False, cleanup_failures=cleanup_errors)
        if observation_errors:
            report.update(status='failed', complete=False, qualification_passed=False, resource_observation_failures=observation_errors)
        if report['peak_cuda_reserved_bytes'] is not None and report['peak_cuda_reserved_bytes'] > release['resource_budget']['device_bytes']:
            report.update(status='failed', complete=False, qualification_passed=False,
                resource_failure='Observed process CUDA reserved highwater exceeds root cap')
        write(folder / 'QUALIFICATION_REPORT.json', report)
        write(folder / 'COMPLETE.json', {key: report[key] for key in ('status', 'complete', 'qualification_passed',
            'quality_scoring', 'VALID_checkpoint_selection', 'scientific_training', 'TEST_file_access',
            'actual_Adam_owner_steps', 'actual_S_private_VJPs', 'observed_references', 'observed_gradient_replays',
            'actual_P_backwards', 'attempted_Adam_owner_steps', 'attempted_S_private_VJPs')})
    require(report['complete'], 'Qualification incomplete; preserve partial records and costs, no retry')
    return report
